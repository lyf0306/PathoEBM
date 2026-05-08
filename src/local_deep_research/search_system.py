"""
PathoEBM Search System - Orchestrator Module

Refactored from a monolithic class into a thin coordinator that delegates to
specialized agent and pipeline modules.

Architecture:
  - models/reference_pool.py    → ReferencePool (citation management)
  - agents/                     → Independent agent classes
  - pipeline/                   → Search planning, knowledge processing, prognosis retrieval

This module retains AdvancedSearchSystem as the public API for backward compatibility.
"""

import asyncio
import json
import logging
import os
import re
import textwrap
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ── Search query deduplication helper ──


def _deduplicate_questions(plan: 'SearchPlan') -> 'SearchPlan':
    """Remove duplicate questions from a SearchPlan, remapping indices in place."""
    if not plan or not plan.questions:
        return plan

    seen = set()
    new_questions = []
    old_to_new = {}

    for i, q in enumerate(plan.questions):
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            old_to_new[i] = len(new_questions)
            new_questions.append(q)

    removed = len(plan.questions) - len(new_questions)
    if removed == 0:
        return plan

    # Remap trial_mapping indices
    if plan.trial_mapping:
        new_mapping = []
        for trial_name, indices in plan.trial_mapping:
            new_indices = [old_to_new[i] for i in indices if i in old_to_new]
            if new_indices:
                new_mapping.append((trial_name, new_indices))
        plan.trial_mapping = new_mapping

    # Remap pico / comorbidity indices
    if plan.pico_indices:
        plan.pico_indices = [old_to_new[i] for i in plan.pico_indices if i in old_to_new]
    if plan.comorb_indices:
        plan.comorb_indices = [old_to_new[i] for i in plan.comorb_indices if i in old_to_new]

    plan.questions = new_questions
    logger.info(f"[Dedup] 检索词去重: 移除 {removed} 个重复检索词，保留 {len(new_questions)} 个")
    return plan

from .config import (
    settings,
    get_claude_openai,
    get_deepseek_r1,
    get_deepseek_v3,
    get_gpt4_1,
    get_gpt4_1_mini,
    get_local_model,
)
from .connect_mcp import OrigeneMCPToolClient, mcp_servers
from .search_system_support import (
    compress_all_llm,
    extract_and_convert_list,
    parse_single,
    safe_json_from_text,
)
from .tool_executor import ToolExecutor
from .tool_selector import ToolSelector
from .utilties.search_utilities import (
    ensure_chinese_output,
    invoke_with_timeout_and_retry,
    remove_think_tags,
    write_log_process_safe,
)

# --- New modular imports ---
from .models.reference_pool import ReferencePool
from .agents.followup_agent import FollowupAgent
from .agents.prognosis_agent import PrognosisAgent
from .agents.mdt_report_agent import MDTReportAgent
from .agents.context_bus import AgentContextBus
from .agents.reviewer_agent import ReviewerAgent
from .evaluation.report_evaluator import ReportEvaluator
from .agents.react_search_agent import ReActSearchAgent
from .pipeline.search_planner import SearchPlanner, SearchPlan
from .pipeline.knowledge_processor import KnowledgeProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(ROOT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(
    log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class _Timer:
    """Simple lap timer for pipeline instrumentation."""
    def __init__(self):
        self._t0 = time.monotonic()
        self._prev = self._t0
        self._laps = []

    def lap(self, name):
        t = time.monotonic()
        self._laps.append((name, t - self._prev))
        self._prev = t

    def log(self, extra=None):
        logger.info(f"\n{'='*50}")
        logger.info("⏱ 性能时间线")
        logger.info(f"{'='*50}")
        for name, dur in self._laps:
            logger.info(f"  ├─ {name}: {dur:.1f}s")
        total = time.monotonic() - self._t0
        logger.info(f"  └─ 总耗时: {total:.1f}s")
        if extra:
            for name, dur in extra:
                logger.info(f"  ├─ {name}: {dur:.1f}s")


GLOBAL_LLM_SEMAPHORE = None
GLOBAL_API_SEMAPHORE = None


def get_global_semaphores():
    global GLOBAL_LLM_SEMAPHORE, GLOBAL_API_SEMAPHORE
    if GLOBAL_LLM_SEMAPHORE is None:
        GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(8)
    if GLOBAL_API_SEMAPHORE is None:
        GLOBAL_API_SEMAPHORE = asyncio.Semaphore(10)
    return GLOBAL_LLM_SEMAPHORE, GLOBAL_API_SEMAPHORE


# ── Core clinical trial keywords for differentiated ReAct rounds ──
_LIGHTHOUSE_TRIAL_PATTERNS = re.compile(
    r'\b(PORTEC-(1|2|3|4a?|4)|GOG-(99|0258|209)|NRG-GY018|RUBY|ATTEND|DUO-E|KEYNOTE-775)\b',
    re.IGNORECASE,
)


def _is_core_trial_query(query: str) -> bool:
    """Check if a search query targets a core clinical trial (needs deeper ReAct search)."""
    return bool(_LIGHTHOUSE_TRIAL_PATTERNS.search(query))


class AdvancedSearchSystem:
    """
    Orchestrator for the evidence update pipeline.

    Delegates to specialized modules:
      - SearchPlanner for generating follow-up search questions
      - KnowledgeProcessor for evidence synthesis
      - PrognosisSkill for authoritative survival rate references
      - ClinicalTrialAgent / FollowupAgent / PrognosisAgent / MDTReportAgent
        for final report generation.
    """

    def __init__(
        self,
        max_iterations=2,
        questions_per_iteration=5,
        is_report=True,
        chosen_tools: list[str] = None,
        error_log_path: str = "",
        using_model="deepseek",
        treatment_context: str = "",
        structured_task: dict = None,
    ):
        self.structured_task = structured_task or {}

        # Initialize reference pool with baseline offset
        baseline_refs = self.structured_task.get("baseline_references", {})
        max_idx = baseline_refs.get("max_index", 0)
        self.ref_pool = ReferencePool(baseline_max_index=max_idx)

        self.chosen_tools = chosen_tools
        self.is_report = is_report
        self.max_iterations = max_iterations
        self.questions_per_iteration = questions_per_iteration
        self.treatment_context = treatment_context
        self.knowledge_chunks = []
        self.all_links_of_system = []
        self.questions_by_iteration = {}

        if error_log_path == "":
            error_log_path = os.path.join(
                log_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
        self.error_log_path = error_log_path

        # === Model initialization ===
        self.using_model = using_model

        if self.using_model == "local":
            logger.info("Using Local vLLM Model (DeepSeek-R1-32B / qwen-test)")
            try:
                local_llm = get_local_model(temperature=0.1)
                local_fast_llm = get_local_model(temperature=0.1)
                self.model = local_llm
                self.reasoning_model = local_llm
                self.tool_planning_model = local_llm
                self.report_model = local_llm
                self.fast_model = local_fast_llm
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise e

        elif self.using_model == "deepseek":
            self.model = get_deepseek_r1()
            self.reasoning_model = get_deepseek_r1()
            self.tool_planning_model = get_deepseek_v3()
            self.fast_model = get_deepseek_v3()
            self.report_model = get_deepseek_r1()

        else:
            self.model = get_gpt4_1()
            self.reasoning_model = get_gpt4_1()
            self.tool_planning_model = get_gpt4_1()
            self.fast_model = get_gpt4_1_mini()
            self.report_model = get_gpt4_1()

    async def initialize(self):
        """Initialize MCP tools, tool selector, and tool executor."""
        try:
            self.mcp_tool_client = OrigeneMCPToolClient(mcp_servers, self.chosen_tools)
            await self.mcp_tool_client.initialize()
            self.mcp_tool_dict = self.mcp_tool_client.tool2source

            self.tool_selector = ToolSelector(
                self.tool_planning_model,
                self.reasoning_model,
                self.mcp_tool_client,
                tool_info_data=None,
                embedding_api_key=None,
                embedding_cache=None,
                available_tools=self.chosen_tools,
            )

            self.tool_executor = ToolExecutor(
                self.mcp_tool_client, self.error_log_path, self.fast_model
            )

            logger.info("System initialized in PURE API MODE (Official Databases Only).")

        except Exception as e:
            logger.error(f"Failed to initialize search system: {e}")
            raise e

    # -------------------------------------------------------------------------
    # Pipeline: Search Planning
    # -------------------------------------------------------------------------
    async def _get_follow_up_questions(self, current_knowledge: str, query: str) -> SearchPlan:
        """Delegate to SearchPlanner for generating a structured search plan."""
        planner = SearchPlanner(
            self.tool_planning_model,
            self.structured_task,
            self.questions_per_iteration,
            treatment_context=self.treatment_context,
        )
        return await planner.generate_questions(current_knowledge, query)

    async def _check_evidence_coverage(self, current_knowledge: str) -> dict:
        """
        Assess whether current evidence sufficiently covers all decision points.
        If not, returns targeted gap queries for the missing areas.

        Returns:
            dict with keys:
              - sufficient (bool)
              - reason (str)
              - gap_queries (List[str]) — empty if sufficient
        """
        prompt = textwrap.dedent(f"""
        你是一名临床证据质量评估专家。
        任务：根据患者的治疗决策需求，判断当前收集的证据是否充分。

        【患者的治疗方案与背景】：
        {self.treatment_context[:2000]}

        【当前已收集的证据】：
        {current_knowledge[:4000]}

        【🚨 评估规则】：
        你需要评估以下三个维度的证据覆盖度：
        1. **灯塔临床试验（按患者分期匹配）**：根据患者的 FIGO 分期和风险分层，严格按以下导航库比对：
           - 早期（I-II期）中低危及中高危：GOG-99, PORTEC-1, PORTEC-2
             （若分子分型已知，追加 PORTEC-4a——分子分型指导放疗降/升阶梯）
           - 早期高危（I-II期伴高危因素）及局部晚期（III、IVA期）：PORTEC-3, GOG-0258
           - 晚期（IVB期）及复发一线：GOG-209, NRG-GY018, RUBY, ATTEND, DUO-E
           - 晚期复发（二线及以上）：KEYNOTE-775
           对应试验是否有生存数据？
        2. **PICO 精准查证**：治疗方案相关的疗效数据（尤其是分子分型亚组）是否充分？
        3. **合并症安全性**：患者合并症（心血管疾病、糖尿病等）的相关管理文献是否覆盖？

        【判断标准】：
        - 每个维度只要有 1-2 篇核心文献支撑即可认为"充分"
        - **偏向充分判断**：只要有基本覆盖就算充分，避免过度检索
        - 只有当某个维度完全没有任何相关文献时才判定为"缺口"

        请严格输出以下 JSON 格式（不含其他内容）：
        {{
            "sufficient": true,
            "reason": "简要说明为什么充分或不充分",
            "gap_queries": ["检索词1", "检索词2"]
        }}
        gap_queries 只在 sufficient=false 时填写，最多 3 个，每个须是可执行的 PubMed 检索词。
        """)

        try:
            resp = await invoke_with_timeout_and_retry(
                self.tool_planning_model, prompt, timeout=300.0, max_retries=3
            )
            content = remove_think_tags(resp.content).strip()
            import json
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return {
                    "sufficient": result.get("sufficient", True),
                    "reason": result.get("reason", ""),
                    "gap_queries": result.get("gap_queries", [])[:3],
                }
        except Exception as e:
            logger.warning(f"证据覆盖度评估异常: {e}")

        return {"sufficient": True, "reason": "评估异常，默认通过", "gap_queries": []}

    # -------------------------------------------------------------------------
    # Pipeline: Knowledge Processing
    # -------------------------------------------------------------------------
    async def _answer_query(
        self,
        current_knowledge: str,
        query: str,
        current_iteration: int,
        max_iterations: int,
    ) -> str:
        """Synthesize API findings into evidence summary."""
        processor = KnowledgeProcessor(self.model, self.fast_model)
        existing_refs = [
            f"[{idx}] {self.ref_pool.display_label(idx)} — {ref.title}"
            for idx, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1)
        ]
        return await processor.answer_query(
            current_knowledge, query, current_iteration, max_iterations, existing_refs
        )

    async def _extract_knowledge(self, facts_md: str, refs_in_round: List[Dict]):
        """Extract key info from tool outputs."""
        processor = KnowledgeProcessor(self.model, self.fast_model)
        return await processor.extract_knowledge(facts_md, refs_in_round)

    async def process_multiple_knowledge_chunks(self, query: str, current_key_info: str) -> str:
        """Consolidate knowledge chunks."""
        processor = KnowledgeProcessor(self.model, self.fast_model)
        return await processor.process_multiple_chunks(query, current_key_info)

    async def _extract_structured_data(self, raw_text: str, source_type: str, query: str) -> str:
        """Extract structured data from ClinicalTrials and FDA JSON."""
        processor = KnowledgeProcessor(self.model, self.fast_model)
        return await processor.extract_structured_data(raw_text, source_type, query)

    # -------------------------------------------------------------------------
    # Pipeline: Prognosis Retrieval Track
    # -------------------------------------------------------------------------
    async def _run_prognosis_retrieval_track(self) -> dict:
        """Load authoritative prognosis reference data (static SEER/NCDB tables only)."""
        skill_data = ""
        try:
            from .skills.prognosis.prognosis_skill import PrognosisSkill
            skill = PrognosisSkill()
            skill_data = skill.get_data()
            if skill_data:
                logger.info(f"已加载 PrognosisSkill 参考数据 ({len(skill_data)} 字符)")
        except Exception as e:
            logger.warning(f"加载 PrognosisSkill 失败: {e}")

        return {
            "skill_data": skill_data,
            "population": "",
            "molecular": "",
            "raw_combined": "",
        }

    # -------------------------------------------------------------------------
    # Report Generation (delegates to agent classes)
    # -------------------------------------------------------------------------
    async def _generate_detailed_report(
        self, current_knowledge: str, findings: List[Dict],
        query: str, iteration: int, prognosis_results: dict = None,
        context_bus: AgentContextBus = None,
    ):
        # Allow enough room for R1 thinking + full MDT report output
        if hasattr(self.report_model, 'max_tokens'):
            self.report_model.max_tokens = 16384
        if hasattr(self.report_model, 'max_completion_tokens'):
            self.report_model.max_completion_tokens = 16384

        # ── Save full evidence before any truncation ──
        full_raw_evidence = current_knowledge

        if len(current_knowledge) > 25000:
            logger.warning(f"current_knowledge 过长 ({len(current_knowledge)} 字符)，正在执行安全截断...")
            current_knowledge = current_knowledge[:25000] + "\n\n...[前沿证据数据过长，已执行物理截断]..."

        # Inherit guideline text from draft
        original_guideline_text = "## 二、 核心指南与共识详尽解析\n（未能在草稿中匹配到指南解析部分）"
        guideline_match = re.search(
            r'(## 二、 核心指南与共识详尽解析.*?)(?=\n## 三、|\n## 四、|\Z)',
            self.treatment_context, re.DOTALL
        )
        if guideline_match:
            original_guideline_text = guideline_match.group(1).strip()

        # --- Context Bus: inter-agent communication channel ---
        context_bus = context_bus or AgentContextBus()

        # Extract three prognosis sources
        prognosis_results = prognosis_results or {}
        skill_data = prognosis_results.get("skill_data", "")
        population_data = prognosis_results.get("population", "")
        molecular_data = prognosis_results.get("molecular", "")

        logger.info("多智能体并发: 直接组装ReAct分析 & 定制随访 & 预后提取...")

        # ── Extract trial/PICO analysis directly from ReAct outputs (no ClinicalTrialAgent) ──
        trial_parts = []
        for m in re.finditer(
            r'#### 🎯 (.+?)(?=\n#### 🎯|\n#### 🧬|\n#### 🏥|\Z)',
            full_raw_evidence, re.DOTALL
        ):
            trial_parts.append(m.group(0).strip())

        pico_match = re.search(
            r'#### 🧬 PICO 问题查证(.+?)(?=\n#### 🏥|\Z)',
            full_raw_evidence, re.DOTALL
        )
        if pico_match:
            pico_text = pico_match.group(0).strip()
            # Only include PICO section if it has actual evidence data (not just search direction + LLM instructions)
            instruction_keywords = ['选择标准', '请直接输出', '不要加任何说明', '请输出', '输出你选中']
            pico_evidence_lines = [
                line.strip() for line in pico_text.split('\n')
                if line.strip()
                and '检索方向' not in line
                and not line.strip().startswith('####')
                and not any(kw in line for kw in instruction_keywords)
            ]
            if pico_evidence_lines:
                trial_parts.append(pico_text)
            else:
                logger.info("[PICO] PICO 问题查证无实质内容（仅检索方向或指令残留），已跳过")

        trial_analysis = "\n\n".join(trial_parts) if trial_parts else "（ReAct 检索未产生试验分析数据）"

        # ── Strip confidence tags leaked from ReAct synthesis ──
        trial_analysis = re.sub(
            r'\[[✅⚠️❓🚫]\s*[^\]]*?(?:高置信度|中置信度|低置信度|不可验证)[^\]]*\]',
            '', trial_analysis,
        )

        # ── Extract comorbidity/safety context for MDT agent ──
        safety_match = re.search(
            r'#### 🏥 合并症安全评估(.+?)(?=\Z)',
            full_raw_evidence, re.DOTALL
        )
        comorbidity_context = safety_match.group(0).strip() if safety_match else ""

        # ── Extract HPV / infection / risk factor data from raw evidence ──
        risk_factor_lines = []
        for line in full_raw_evidence.split('\n'):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ['hpv', '人乳头瘤', 'tct', '宫颈病', '宫颈癌筛查',
                                                 '阴道断端细胞学', '宫颈上皮内']):
                risk_factor_lines.append(line.strip())
        risk_factor_context = '\n'.join(risk_factor_lines[:30]).strip() if risk_factor_lines else ""
        if risk_factor_context:
            await context_bus.post("ReActAnalysis", "risk_factor_context", risk_factor_context)
            logger.info("[HPV] 从检索证据中提取了 %d 行 HPV/宫颈相关数据", len(risk_factor_lines))

        await context_bus.post("ReActAnalysis", "trial_analysis", trial_analysis)
        if comorbidity_context:
            await context_bus.post("ReActAnalysis", "comorbidity_context", comorbidity_context)

        # --- Run Agent 1.5 (Follow-up), Agent 3 (Prognosis) — no more ClinicalTrialAgent ---
        surgery_type = self.structured_task.get("surgery_type", "")
        followup_agent = FollowupAgent(
            self.report_model, self.treatment_context,
            surgery_type=surgery_type
        )
        oncology_core = self.structured_task.get("oncology_profile", {})
        prognosis_agent = PrognosisAgent(
            self.report_model, self.treatment_context,
            oncology_core=oncology_core,
            context_bus=context_bus,
        )

        try:
            followup_plan, prognosis_data, trial_analysis = await asyncio.wait_for(
                asyncio.gather(
                    followup_agent.run(),
                    prognosis_agent.run(
                        skill_data=skill_data,
                        population_data=population_data,
                        molecular_data=molecular_data,
                    ),
                    self._consolidate_trial_analysis(trial_analysis),
                ),
                timeout=600.0
            )
            if timer := getattr(self, '_timer', None):
                timer.lap("多智能体并发")
            await context_bus.post("PrognosisAgent", "prognosis_data",
                                   prognosis_data)
        except Exception as e:
            logger.error(f"随访/预后/试验合并并发执行超时或崩溃: {e}")
            followup_plan = "随访方案生成超时失败。"
            prognosis_data = "预后数据提取超时失败。"
            await context_bus.post("System", "agent_failure", str(e))

        logger.info(f"预后专员提取结果: {prognosis_data}")

        # Strip all [^^n] citation markers from prognosis data to prevent
        # LLM from citing references that don't match the content.
        # The prognosis agent sees article titles with [^^n] numbers and often
        # attaches them to claims those articles don't actually support.
        # Instead, the MDT agent will borrow correct citations from the trial
        # analysis when writing treatment-related claims in the prognosis section.
        prognosis_data = re.sub(r'\[\^\^(\d+)\]', '', prognosis_data)
        prognosis_data = re.sub(r' +', ' ', prognosis_data).strip()

        # --- Patient-trial relevance filter (drop trials that don't match stage/histology) ---
        trial_analysis = self._filter_irrelevant_trials(trial_analysis)

        # --- Title-based deduplication before MDT assembly ---
        trial_analysis = self._deduplicate_trial_analysis(trial_analysis)

        # --- Intra-trial deduplication: detect near-duplicate sub-entries within same trial ---
        trial_analysis = self._deduplicate_intra_trial(trial_analysis)

        # --- Demote paper sub-headings (#### → #####) for visual hierarchy ---
        trial_analysis = self._demote_paper_subheadings(trial_analysis)

        # --- 兜底翻译：确保试验分析在进入下游agent之前已经是中文 ---
        trial_analysis = await ensure_chinese_output(
            trial_analysis, self.fast_model, label="TrialAnalysis", logger=logger
        )

        # ── 去薄：移除仅有标题+一句话、无实质数据点的空壳条目 ──
        trial_analysis = ReActSearchAgent._remove_thin_entries(trial_analysis)

        # --- Agent 2: MDT Report Chief Writer (合并症管理 + Assembly) ---
        mdt_agent = MDTReportAgent(
            self.report_model, self.treatment_context, self.structured_task,
            context_bus=context_bus,
        )

        # --- Agent 2a: Treatment Decision Specialist (主要方案) ---
        from .agents.treatment_decision_agent import TreatmentDecisionAgent
        treatment_agent = TreatmentDecisionAgent(
            self.report_model, self.structured_task, context_bus=context_bus,
        )

        # Read reference map from ContextBus
        ref_map_str = ""
        if context_bus:
            ref_msgs = await context_bus.get_by_type("reference_map")
            if ref_msgs:
                ref_map_str = ref_msgs[-1]["content"][:3000]

        # Read risk factor context from ContextBus
        risk_factor_context = ""
        if context_bus:
            rf_msgs = await context_bus.get_by_type("risk_factor_context")
            if rf_msgs:
                risk_factor_context = rf_msgs[-1]["content"][:2000]

        guideline_section_demoted = mdt_agent._extract_guideline_section()
        patient_summary = mdt_agent._build_patient_summary()

        # Generate 主要方案 (TreatmentDecisionAgent)
        main_plan = await treatment_agent.generate_main_plan(
            trial_analysis=trial_analysis,
            ref_map_str=ref_map_str,
            patient_summary=patient_summary,
            guideline_section=guideline_section_demoted,
            safety_context=comorbidity_context,
            risk_factor_context=risk_factor_context,
        )
        logger.info(f"[TreatmentDecisionAgent] 主要方案已生成（{len(main_plan)} 字符）")

        content = await mdt_agent.run(
            trial_analysis, followup_plan, prognosis_data,
            main_treatment_plan=main_plan,
            safety_context=comorbidity_context,
        )
        await context_bus.post("MDTReportAgent", "report_generated",
                               f"报告初稿长度: {len(content)} 字符")
        if timer := getattr(self, '_timer', None):
            timer.lap("MDT撰写")

        # --- Reviewer → Agent Re-generation Loop ---
        # ReviewerAgent is a PURE QUALITY CHECKER: it detects issues but does
        # NOT repair them.  Instead, the pipeline feeds issues back to the
        # original generation agents (TreatmentDecisionAgent, MDTReportAgent,
        # FollowupAgent) for targeted re-generation with their full specialized
        # prompts and surgical rules.  Up to 3 review cycles.
        MAX_REVIEW_CYCLES = 3
        RE_REVIEW_TEMP_BUMP = 0.2

        # Extract sub-sections from 术后处理 for potential individual re-generation
        def _extract_main_plan(report: str) -> str:
            m = re.search(r'### 主要方案\n(.*?)(?=\n### 合并症管理|\Z)', report, re.DOTALL)
            return m.group(1).strip() if m else ""

        def _extract_comorbidity_mgmt(report: str) -> str:
            m = re.search(r'### 合并症管理\n(.*?)(?=\n## 三[、.．] 预后分析|\Z)', report, re.DOTALL)
            return m.group(1).strip() if m else ""

        main_plan_content = _extract_main_plan(content)
        comorbidity_content = _extract_comorbidity_mgmt(content)

        original_temp = getattr(self.report_model, 'temperature', None)
        was_repaired = False
        reviewer = ReviewerAgent(self.report_model)

        # Accumulate all issues across review cycles for stateful re-review
        accumulated_issues: Dict[str, List[str]] = {}

        for cycle in range(MAX_REVIEW_CYCLES):
            cycle_label = "初审" if cycle == 0 else f"第{cycle}轮重审"
            logger.info(f"[Reviewer] {cycle_label}开始...")

            # On re-review: pass accumulated previous issues so reviewer can
            # verify each one was actually fixed (stateful "错题本核销" review)
            _prev_for_review = accumulated_issues if accumulated_issues else None
            issues_by_section = await reviewer.review(
                report=content,
                trial_analysis=trial_analysis,
                followup_plan=followup_plan,
                prognosis_data=prognosis_data,
                treatment_context=self.treatment_context,
                context_bus=context_bus,
                surgery_type=surgery_type,
                previous_issues=_prev_for_review,
            )

            if not issues_by_section:
                if cycle == 0:
                    logger.info("[Reviewer] 初审通过，报告质量合格。")
                else:
                    logger.info(f"[Reviewer] {cycle_label}通过——agent 重生成有效，无新问题。")
                break

            # Merge current issues into accumulated set for next cycle's review
            for section, items in issues_by_section.items():
                accumulated_issues.setdefault(section, []).extend(items)
            logger.info(
                f"[Reviewer] 累计问题池: "
                f"{ {s: len(v) for s, v in accumulated_issues.items()} }"
            )

            total_issues = sum(len(v) for v in issues_by_section.values())
            logger.warning(
                f"[Reviewer] {cycle_label}发现 {total_issues} 个问题"
                f"（{list(issues_by_section.keys())}），触发 agent 重生成..."
            )

            # Bump temperature on re-generation for re-review cycles only
            # Progressive: cycle 1 → +0.2, cycle 2 → +0.4, cycle 3 → +0.6
            if cycle > 0 and original_temp is not None:
                self.report_model.temperature = original_temp + RE_REVIEW_TEMP_BUMP * cycle
                logger.info(
                    f"[Reviewer] {cycle_label} 重生成 temperature: "
                    f"{original_temp + RE_REVIEW_TEMP_BUMP * (cycle - 1):.1f} → {self.report_model.temperature:.1f}"
                )

            # ── Detect cross-section alignment issues ──
            _ALIGNMENT_KW = ["不对齐", "缺失", "不一致", "未提及", "凭空新增"]
            _has_alignment_issues = any(
                any(kw in iss for kw in _ALIGNMENT_KW)
                for iss in (
                    issues_by_section.get("随访方案", [])
                    + issues_by_section.get("合并症管理", [])
                )
            )

            # ── Pre-compute flags & issue lists for all three sections ──
            main_plan_sections = {"主要方案", "病情分析"}
            _needs_main_plan_regen = bool(
                issues_by_section.keys() & main_plan_sections
            )
            _needs_comorbidity_regen = "合并症管理" in issues_by_section
            _needs_followup_regen = "随访方案" in issues_by_section

            main_plan_issues = []
            if _needs_main_plan_regen:
                for sec in main_plan_sections:
                    main_plan_issues.extend(issues_by_section.get(sec, []))
                logger.info(
                    f"[Reviewer] 触发 TreatmentDecisionAgent 重生成 "
                    f"（{len(main_plan_issues)} 个问题）"
                )
                for i, iss in enumerate(main_plan_issues, 1):
                    logger.warning(f"[Reviewer]   → 主要方案 问题#{i}: {iss[:300]}")

            comorbidity_issues = []
            if _needs_comorbidity_regen:
                comorbidity_issues = list(issues_by_section.get("合并症管理", []))
                if _has_alignment_issues:
                    comorbidity_issues.extend(
                        issues_by_section.get("随访方案", [])
                    )
                logger.info(
                    f"[Reviewer] 触发 MDTReportAgent 合并症管理重生成 "
                    f"（{len(comorbidity_issues)} 个问题"
                    f"{'，含跨章节对齐' if _has_alignment_issues else ''}）"
                )
                for i, iss in enumerate(comorbidity_issues, 1):
                    logger.warning(f"[Reviewer]   → 合并症管理 问题#{i}: {iss[:300]}")

            followup_issues = []
            if _needs_followup_regen:
                followup_issues = list(issues_by_section.get("随访方案", []))
                logger.info(
                    f"[Reviewer] 触发 FollowupAgent 重生成 "
                    f"（{len(followup_issues)} 个问题"
                    f"{'，含跨章节对齐' if _has_alignment_issues else ''}）"
                )
                for i, iss in enumerate(followup_issues, 1):
                    logger.warning(f"[Reviewer]   → Followup Agent 问题#{i}: {iss[:300]}")

            # ── Execute re-generations (parallelize when safe) ──
            # Dependency: comorbidity needs main_plan (main_treatment_plan=).
            # When _has_alignment_issues: followup needs latest main_plan +
            #   comorbidity for treatment_context splicing → full serial.
            # When no alignment issues: followup is independent → can run
            #   concurrently with main_plan and/or comorbidity.

            main_plan_task = None
            comorbidity_task = None
            followup_task = None

            # Launch main_plan early if needed (no upstream deps)
            if _needs_main_plan_regen:
                main_plan_task = asyncio.ensure_future(
                    treatment_agent.generate_main_plan(
                        trial_analysis=trial_analysis,
                        ref_map_str=ref_map_str,
                        patient_summary=patient_summary,
                        guideline_section=guideline_section_demoted,
                        safety_context=comorbidity_context,
                        risk_factor_context=risk_factor_context,
                        reviewer_issues=main_plan_issues,
                        previous_output=main_plan_content,
                    )
                )

            # Launch comorbidity early if main_plan NOT being re-gen'd (no dep)
            if _needs_comorbidity_regen and not _needs_main_plan_regen:
                comorbidity_task = asyncio.ensure_future(
                    mdt_agent._generate_comorbidity_management(
                        main_treatment_plan=main_plan_content,
                        safety_context=comorbidity_context,
                        risk_factor_context=risk_factor_context,
                        reviewer_issues=comorbidity_issues,
                        previous_output=comorbidity_content,
                    )
                )

            # Launch followup early when no alignment issues (independent)
            if _needs_followup_regen and not _has_alignment_issues:
                followup_task = asyncio.ensure_future(
                    followup_agent.run(
                        reviewer_issues=followup_issues,
                        previous_output=followup_plan,
                    )
                )

            # ── Resolve main_plan first (comorbidity may depend on it) ──
            if main_plan_task:
                main_plan_content = await main_plan_task

            # Launch comorbidity now if it depends on new main_plan
            if _needs_comorbidity_regen and _needs_main_plan_regen:
                comorbidity_task = asyncio.ensure_future(
                    mdt_agent._generate_comorbidity_management(
                        main_treatment_plan=main_plan_content,
                        safety_context=comorbidity_context,
                        risk_factor_context=risk_factor_context,
                        reviewer_issues=comorbidity_issues,
                        previous_output=comorbidity_content,
                    )
                )

            # ── Resolve comorbidity ──
            if comorbidity_task:
                comorbidity_content = await comorbidity_task

            # Followup with alignment issues: must wait for main_plan + comorbidity
            if _needs_followup_regen and _has_alignment_issues:
                _current_postop = (
                    "## 二、 术后处理\n\n### 主要方案\n"
                    + main_plan_content
                    + "\n\n### 合并症管理\n"
                    + comorbidity_content
                )
                _updated_ctx = re.sub(
                    r'## 二[、.．] 术后处理.*?(?=## 四[、.．] 随访方案|\Z)',
                    _current_postop + "\n\n",
                    followup_agent.treatment_context,
                    flags=re.DOTALL,
                )
                followup_agent.treatment_context = _updated_ctx
                followup_task = asyncio.ensure_future(
                    followup_agent.run(
                        reviewer_issues=followup_issues,
                        previous_output=followup_plan,
                    )
                )

            # ── Resolve followup ──
            if followup_task:
                followup_plan = await followup_task

            # Restore original temperature for next cycle's review
            if original_temp is not None:
                self.report_model.temperature = original_temp

            # ── Re-assemble report with regenerated sections ──
            content = mdt_agent._assemble_final_report(
                mdt_agent._build_patient_summary(),
                mdt_agent._extract_guideline_section(),
                trial_analysis,
                main_plan_content,
                comorbidity_content,
                prognosis_data,
                followup_plan,
            )
            was_repaired = True

        if timer := getattr(self, '_timer', None):
            timer.lap("审查修复")
        if was_repaired:
            await context_bus.post("ReviewerAgent", "repair_applied",
                                   "报告质量问题已通过 agent 重生成修复")
            logger.info("[Reviewer] 报告已通过 agent 重生成修复。")
        else:
            logger.info("[Reviewer] 报告无需修复。")

        # Final safety: strip any lingering think tags from all LLM outputs
        content = remove_think_tags(content)

        # Post-process references
        try:
            new_content, refs_section = self.ref_pool.reindex_references(content)

            # Final safety: strip any remaining citation markers from 术后处理 section
            for punct in ['、', '．', '.', '：', ':']:
                pattern = rf"(##\s*二{re.escape(punct)}\s*术后处理[\s\S]*?)(?=\n##\s*[三四]|\Z)"
                m = re.search(pattern, new_content, re.DOTALL)
                if m:
                    section_text = m.group(1)
                    cleaned = re.sub(r'\[\^?\^?\^?\d+(?:\s*[,、，]\s*\^?\^?\^?\d+)*\s*\]', '', section_text)
                    cleaned = re.sub(r' +', ' ', cleaned)
                    cleaned = cleaned.replace(' 。', '。').replace(' ，', '，')
                    new_content = new_content[:m.start()] + cleaned + new_content[m.end():]
                    break

            # Fix numbering disorder in 术后处理 section
            new_content = self._fix_numbering_in_postop(new_content)

            full_report = new_content + "\n" + refs_section

            # Statistical sanity check: intercept X% vs X% with P<0.05
            full_report = self._statistical_sanity_check(full_report)

            return full_report, full_report
        except Exception as e:
            logger.error(f"Failed to post-process references: {e}")
            fallback_refs = "\n==================================================\n"
            for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                fallback_refs += f"[{i}] {self.ref_pool.display_label(i)}\n    Title: {ref.title or ref.link}\n----------\n"
            return current_knowledge + fallback_refs, current_knowledge + fallback_refs

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    async def analyze_topic(self, query: str) -> Dict:
        """Main execution loop."""
        logger.info("Starting Pure API Validation (High-Performance Concurrent Mode)")

        current_knowledge = ""
        cumulative_raw_evidence = ""
        iteration = 0
        findings = []

        llm_semaphore, api_semaphore = get_global_semaphores()
        self._timer = _Timer()

        await self.initialize()
        self._timer.lap("系统初始化")

        # Start background prognosis retrieval
        prognosis_task = asyncio.create_task(self._run_prognosis_retrieval_track())

        # Create inter-agent communication bus (lifetime = full pipeline)
        context_bus = AgentContextBus()

        # Create ReAct search agent (shared across all iterations for efficiency)
        react_agent = ReActSearchAgent(
            self.fast_model, self.tool_planning_model, self.reasoning_model,
            self.mcp_tool_client, self.chosen_tools, self.error_log_path,
            llm_semaphore, api_semaphore,
            ref_pool=self.ref_pool,
        )

        while iteration < self.max_iterations:
            # --- First iteration: planner generates structured search plan ---
            # --- Later iterations: only if coverage check found gaps ---
            search_plan: Optional[SearchPlan] = None

            if iteration == 0:
                search_plan = await self._get_follow_up_questions(current_knowledge, query)
                questions = search_plan.questions
                if not questions:
                    questions = [query]
                    search_plan = None
                else:
                    # Remove duplicate search queries before dispatching
                    search_plan = _deduplicate_questions(search_plan)
                    questions = search_plan.questions
            else:
                coverage = await self._check_evidence_coverage(current_knowledge)
                if coverage.get("sufficient", True):
                    logger.info("证据覆盖度已达标，无需第二轮检索。")
                    break
                gap_queries = coverage.get("gap_queries", [])[:3]
                if not gap_queries:
                    logger.info("无明确证据缺口，结束检索。")
                    break
                # Dedup gap queries
                seen = set()
                deduped = []
                for q in gap_queries:
                    key = q.strip().lower()
                    if key not in seen:
                        seen.add(key)
                        deduped.append(q)
                if len(deduped) < len(gap_queries):
                    logger.info(f"[Dedup] 补充检索词去重: 移除 {len(gap_queries) - len(deduped)} 个重复")
                questions = deduped
                logger.info(f"证据存在缺口，补充检索 {len(questions)} 个方向")

            self.questions_by_iteration[iteration] = questions
            logger.info(f"Iteration {iteration+1}: Concurrently processing {len(questions)} sub-questions...")

            logger.info(f"启动并发检索 ({len(questions)} 个独立分支)...")

            all_questions_results = []
            task_meta = []  # list of (type, label) for logging

            if (
                iteration == 0
                and search_plan is not None
                and search_plan.has_trial_grouping
            ):
                # ── Structured: per-trial ReAct loops + individual PICO/comorbidity ──
                tasks = []

                # 1. Trial groups: each trial = one execute_trial call covering all sub-queries
                for trial_name, sq_indices in search_plan.trial_mapping:
                    sub_queries = [questions[i] for i in sq_indices]
                    tasks.append(react_agent.execute_trial(trial_name, sub_queries, max_rounds=5))
                    task_meta.append(("trial", trial_name))

                # 2. PICO queries (individual)
                for i in (search_plan.pico_indices or []):
                    tasks.append(react_agent.execute(questions[i], max_rounds=2))
                    task_meta.append(("pico", i))

                # 3. Comorbidity queries (individual)
                for i in (search_plan.comorb_indices or []):
                    tasks.append(react_agent.execute(questions[i], max_rounds=2))
                    task_meta.append(("comorb", i))

                all_questions_results = await asyncio.gather(*tasks)
            else:
                # ── Flat fallback (iteration > 0 or no trial grouping) ──
                tasks = [react_agent.execute(q, max_rounds=2) for q in questions]
                task_meta = [("flat", q) for q in questions]
                all_questions_results = await asyncio.gather(*tasks)

            # ── Logging ──
            for i, syn in enumerate(all_questions_results):
                meta = task_meta[i] if i < len(task_meta) else ("unknown", "")
                meta_type, meta_label = meta
                if meta_type == "trial":
                    desc = f"[Trial] {meta_label}"
                elif meta_type == "pico":
                    desc = f"[PICO] {questions[meta_label][:70]}"
                elif meta_type == "comorb":
                    desc = f"[Comorb] {questions[meta_label][:70]}"
                else:
                    desc = str(meta_label)[:80]
                msg = f"\n{'='*60}\n🔍 独立分支 ReAct 分析完毕 [{i+1}/{len(all_questions_results)}]\n📍 {desc}\n{'-'*60}\n{syn.strip()}\n{'='*60}\n"
                print(msg)
                logger.info(msg)

            # ── Build evidence chunk ──
            chunk_knowledge = f"\n\n### 第 {iteration + 1} 轮检索分析：\n"

            if (
                iteration == 0
                and search_plan is not None
                and search_plan.has_trial_grouping
            ):
                result_idx = 0

                # 1. Trial groups (each already a unified synthesis from execute_trial)
                for trial_name, _ in search_plan.trial_mapping:
                    if result_idx < len(all_questions_results):
                        syn = all_questions_results[result_idx]
                        if syn.strip() and "无相关临床证据" not in syn:
                            chunk_knowledge += f"\n#### 🎯 {trial_name}\n{syn}\n"
                        result_idx += 1

                # 2. PICO section
                if search_plan.pico_indices:
                    pico_parts = []
                    for i in search_plan.pico_indices:
                        if result_idx < len(all_questions_results):
                            syn = all_questions_results[result_idx]
                            if syn.strip() and "无相关临床证据" not in syn:
                                pico_parts.append(f"\n**检索方向**: {questions[i]}\n{syn}")
                            result_idx += 1
                    if pico_parts:
                        chunk_knowledge += "\n#### 🧬 PICO 问题查证\n" + "".join(pico_parts)

                # 3. Comorbidity section
                if search_plan.comorb_indices:
                    comorb_parts = []
                    for i in search_plan.comorb_indices:
                        if result_idx < len(all_questions_results):
                            syn = all_questions_results[result_idx]
                            if syn.strip() and "无相关临床证据" not in syn:
                                comorb_parts.append(f"\n**检索方向**: {questions[i]}\n{syn}")
                            result_idx += 1
                    if comorb_parts:
                        chunk_knowledge += "\n#### 🏥 合并症安全评估\n" + "".join(comorb_parts)
            else:
                # Flat assembly (iteration > 0 or no trial grouping available)
                for q, syn in zip(questions, all_questions_results):
                    if syn.strip() and "无相关临床证据" not in syn:
                        chunk_knowledge += f"\n**检索问题**: {q}\n{syn}\n"

            cumulative_raw_evidence += chunk_knowledge
            current_knowledge = cumulative_raw_evidence
            iteration += 1
            self._timer.lap(f"检索迭代_{iteration}")

        # Await prognosis retrieval
        prognosis_results = await prognosis_task
        self._timer.lap("预后检索(后台)")

        # Build reference map from ONLY the references actually cited in
        # current_knowledge — prune unused search results before downstream agents see them
        cited_ids = set()
        for m in re.finditer(r"\[\^\^(\d+)\]", cumulative_raw_evidence):
            cited_ids.add(int(m.group(1)))

        ref_entries = []
        for idx in sorted(cited_ids):
            ref = self.ref_pool.get_ref_by_idx(idx)
            if ref:
                title = (ref.title or ref.link or "Unknown")[:100]
                ref_entries.append(f"[^^{idx}] {title}")
        ref_map_str = "\n".join(ref_entries)
        await context_bus.post("System", "reference_map", ref_map_str)
        total_raw = len(self.ref_pool.pool)
        logger.info(f"已将 {len(ref_entries)} 条被引用的参考文献映射发布到 ContextBus（原始注册 {total_raw} 条，丢弃 {total_raw - len(ref_entries)} 条未引用记录）")

        final_report = ""
        if self.is_report:
            try:
                final_report_tuple = await self._generate_detailed_report(
                    cumulative_raw_evidence, findings, query, iteration, prognosis_results,
                    context_bus=context_bus,
                )
                if isinstance(final_report_tuple, tuple):
                    final_report = final_report_tuple[1]
                else:
                    final_report = str(final_report_tuple)
            except Exception as e:
                logger.warning(f"Failed to generate detailed report: {e}")
                fallback_refs = "\n==================================================\n"
                for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                    title = ref.title if ref.title else "Source"
                    fallback_refs += f"[{i}] {self.ref_pool.display_label(i)}\n    Title: {title}\n----------\n"
                final_report = current_knowledge + fallback_refs

        self._timer.lap("报告生成")

        # Citation credibility check
        if final_report:
            credibility = self._check_citation_credibility(final_report)
            logger.info(f"\n{'='*50}")
            logger.info("📊 引用可信度报告")
            logger.info(f"{'='*50}")
            logger.info(f"  总引用数: {credibility['total']}")
            logger.info(f"  有效引用: {credibility['valid']}")
            logger.info(f"  断裂引用: {credibility['broken']}")
            logger.info(f"  引用完整率: {credibility['rate']:.1%}")

        # ── MDT Report Quality Evaluation (Tier 1 + Tier 2) ──
        if final_report:
            try:
                evaluator = ReportEvaluator(
                    ref_pool=self.ref_pool,
                    structured_task=self.structured_task,
                )
                eval_results = evaluator.evaluate(final_report)
                logger.info(f"\n{eval_results.summary()}")
            except Exception as e:
                logger.warning(f"[ReportEvaluator] 评估异常: {e}")

        # Log timing
        self._timer.log()

        return {
            "findings": findings,
            "iterations": iteration,
            "questions": self.questions_by_iteration,
            "current_knowledge": current_knowledge,
            "final_report": final_report,
        }

    # -------------------------------------------------------------------------
    # Internal helpers (extracted from the monolithic loop)
    # -------------------------------------------------------------------------
    def _process_tool_result(self, res, unique_articles_dict: dict, global_seen_urls: set):
        """Parse a single tool result and add to the deduplicated dict."""
        res_str = ""
        if isinstance(res, dict) and "content" in res:
            raw_content = res["content"]
            try:
                import ast
                parsed_list = ast.literal_eval(raw_content)
                if isinstance(parsed_list, list):
                    res_str = "".join([item.get("text", "") for item in parsed_list])
            except Exception:
                res_str = str(raw_content)
        else:
            res_str = str(res)

        res_str = res_str.replace('\\n', '\n')
        res_str = re.sub(
            r"^\[?\s*\{\s*['\"]type['\"]\s*:\s*['\"]text['\"]\s*,\s*['\"]text['\"]\s*:\s*['\"]",
            "", res_str
        )
        res_str = re.sub(r"['\"]\s*\}\s*\]?$", "", res_str)

        blocks = res_str.split("\n---\n") if "\n---\n" in res_str else [res_str]
        for block in blocks:
            if not block.strip():
                continue

            url = ""
            pmid_match = (
                re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', block, re.IGNORECASE)
                or re.search(
                    r'["\']?(?:PMID|uid|id)["\']?\s*[:=]\s*["\']?(\d{7,9})["\']?',
                    block, re.IGNORECASE
                )
            )
            nct_match = re.search(r'(NCT\d{8})', block, re.IGNORECASE)

            if pmid_match:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"
            elif nct_match:
                url = f"https://clinicaltrials.gov/study/{nct_match.group(1)}"
            elif "openfda" in block.lower() or "brand_name" in block.lower() or "generic_name" in block.lower():
                url = "https://nctr-crs.fda.gov/fdalabel/ui/search"

            if not url:
                continue

            title = "Unknown Title"
            title_match = (
                re.search(r'^(?:Article )?Title:\s*([^\n]+)', block, re.IGNORECASE | re.MULTILINE)
                or re.search(r'\bTitle:\s*([^\n]+)', block, re.IGNORECASE)
                or re.search(r'"title"\s*:\s*"([^"]+)"', block, re.IGNORECASE)
                or re.search(r'"BriefTitle"\s*:\s*"([^"]+)"', block, re.IGNORECASE)
            )
            if title_match:
                title = title_match.group(1).strip()

            if len(title) < 15 and "FDA" not in title and "Unknown" not in title:
                continue

            if url not in global_seen_urls:
                global_seen_urls.add(url)
                raw_text = block.strip()
                if len(raw_text) > 6000:
                    raw_text = raw_text[:6000] + "\n\n...[文本过长，已执行物理截断]..."
                unique_articles_dict[url] = {
                    "url": url,
                    "title": title,
                    "content": raw_text
                }

    async def _screen_articles(self, articles_list: list, llm_semaphore, query: str) -> list:
        """Screen articles using LLM to select the most relevant ones."""
        if len(articles_list) <= 5:
            return articles_list

        logger.info(f"启动大模型初筛机制，评估 {len(articles_list)} 篇文献/数据...")

        titles_catalog = ""
        for idx, art in enumerate(articles_list):
            if "clinicaltrials.gov" in art["url"]:
                prefix = "🏥 [专属结构化提纯 - 临床试验 NCT]"
            elif "fda.gov" in art["url"] or "nctr-crs.fda.gov" in art["url"]:
                prefix = "💊 [专属结构化提纯 - FDA 药物数据]"
            else:
                prefix = "📄 [PubMed 前沿文献]"
            titles_catalog += f"[{idx}] {prefix} {art['title']}\n"

        screening_prompt = textwrap.dedent(f"""
        你是一名顶尖的妇科肿瘤循证医学文献筛选专家。
        我们为患者检索并初步结构化了以下 {len(articles_list)} 篇候选文献/试验数据。
        为了防止信息过载，请你挑选出最核心、最具有指导意义的 1 到 5 篇。

        【患者病情与检索背景】：
        {self.treatment_context}

        【候选数据菜单】：
        {titles_catalog}

        【🚨 筛选红线（极度重要！）】：
        1. **禁止"以貌取人"**：带有 [专属结构化提纯] 标签的 NCT 或 FDA 数据必须认真评估，若高度相关，请赋予最高优先级！
        2. **严格防噪（宁缺毋滥）**：入组人群与患者明显不符的文献必须果断剔除！
        3. **【灯塔试验绝对豁免权】（最核心红线）**：如果在标题中看到了与患者分期、分子分型和治疗阶段匹配的重磅临床试验代号，按以下导航库比对：
           - 早期（I-II期）中低危及中高危：GOG-99, PORTEC-1, PORTEC-2
             （若分子分型已知，追加 PORTEC-4a——分子分型指导放疗降/升阶梯）
           - 早期高危（I-II期伴高危因素）及局部晚期（III、IVA期）：PORTEC-3, GOG-0258
           - 晚期（IVB期）及复发一线：GOG-209, NRG-GY018, RUBY, ATTEND, DUO-E
           - 晚期复发（二线及以上）：KEYNOTE-775
           匹配的试验你**必须无条件赋予最高优先级选入**！绝对不能把它们当作冗余筛掉！但若该试验的入组人群与患者的实际风险分层明显不符，则不适用豁免权，应按常规判断是否纳入。
        4. 坚决剔除垃圾匹配（如普拉提、骨科等无关领域）和基础动物实验。

        【强制输出格式】：
        请严格输出一个 JSON 数组，包含你选中的文献编号（最多选5个！）。例如：[0, 2, 4]
        """)

        selected_articles = []
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with llm_semaphore:
                    resp = await invoke_with_timeout_and_retry(
                        self.model, screening_prompt, timeout=800.0
                    )
                cleaned_resp = remove_think_tags(resp.content)
                json_match = re.search(r'\[[\d\s,]+\]', cleaned_resp)
                if json_match:
                    selected_indices = json.loads(json_match.group(0))
                    valid_indices = [
                        i for i in set(selected_indices)
                        if isinstance(i, int) and 0 <= i < len(articles_list)
                    ]
                    if valid_indices:
                        selected_articles = [articles_list[i] for i in valid_indices]
                        logger.info(f"成功筛选到 {len(selected_articles)} 篇文献。")
                        break
            except Exception as e:
                logger.warning(f"文献初筛执行报错 (尝试 {attempt+1}/{max_retries}): {e}")

        if not selected_articles:
            logger.error("达到最大重试次数，退回默认选取前 5 篇。")
            selected_articles = articles_list[:5]

        logger.info(f"最终挑选了 {len(selected_articles)} 篇最高价值文献进入直通车。")
        return selected_articles

    async def _consolidate_trial_analysis(self, trial_analysis: str) -> str:
        """
        Consolidate + select top items from ReAct outputs.

        核心试验：不参与筛选，直接从原始分析中保留（ReAct 已产出最终分析）。
        非核心条目（PICO查证等）：LLM 择优保留 1 项。

        Preserves the #### 🎯 [trial name] group headers so that papers
        remain organized under their parent trial in the final report.
        """
        if not trial_analysis or "超时失败" in trial_analysis or "未发现" in trial_analysis:
            return trial_analysis

        # Strip ### narrative headers that break downstream regex extraction.
        # When the ReAct synthesis has no valid papers to extract, the LLM may
        # produce narrative summaries starting with "### TrialName 试验相关文献分析"
        # instead of the expected "#### [Paper Title] [^^n]" format. These H3 lines
        # prematurely terminate the #### 🎯 block regex lookahead.
        trial_analysis = re.sub(r'\n### [^\n]+\n', '\n', trial_analysis)

        lighthouse_trials = [
            "GOG-99", "PORTEC-1", "PORTEC-2", "PORTEC-3", "GOG-0258",
            "GOG-209", "NRG-GY018", "RUBY", "ATTEND", "DUO-E",
            "KEYNOTE-775", "PORTEC-4a"
        ]

        # ── Parse into trial GROUPS (#### 🎯 blocks), not flat sections ──
        # Each group: header line + all child papers until the next 🎯/🧬/🏥
        trial_blocks = re.findall(
            r'#### 🎯 .+?(?=\n#### 🎯|\n#### 🧬|\n#### 🏥|\Z)',
            trial_analysis, re.DOTALL
        )

        result_parts = []

        for block in trial_blocks:
            lines = block.split('\n')
            header_line = lines[0]  # #### 🎯 PORTEC-3

            # Check if this block's header matches a lighthouse trial
            is_lighthouse = any(
                t.lower() in header_line.lower() for t in lighthouse_trials
            )

            # Split block into individual paper sections (each starts with ####)
            paper_sections = re.findall(
                r'#### (?!🎯|🧬|🏥).+?(?=\n#### (?!🎯|🧬|🏥)|\Z)',
                block, re.DOTALL
            )

            if is_lighthouse:
                # Lighthouse trial block → keep ALL papers unconditionally
                result_parts.append(header_line)
                clean_sections = [self._strip_empty_markers(ps) for ps in paper_sections]
                clean_sections = [s for s in clean_sections if s.strip()]
                result_parts.extend(clean_sections)
                logger.info(
                    "[合并] %s: 灯塔试验 → 全部 %d 项保留",
                    header_line, len(paper_sections),
                )
            else:
                core_papers = []
                non_core_papers = []
                for ps in paper_sections:
                    if self._is_empty_paper(ps):
                        continue
                    ps_first_line = ps.split('\n')[0]
                    if any(t.lower() in ps_first_line.lower() for t in lighthouse_trials):
                        core_papers.append(ps.strip())
                    else:
                        non_core_papers.append(ps.strip())

                # Drop empty non-lighthouse blocks entirely
                if not core_papers and not non_core_papers:
                    logger.info(
                        "[合并] %s: 非灯塔试验无有效内容 → 丢弃",
                        header_line,
                    )
                    continue

                # Keep group if it has any core papers
                if core_papers:
                    result_parts.append(header_line)
                    result_parts.extend(core_papers)
                    if non_core_papers:
                        logger.info(
                            "[合并] %s: 核心 %d 项 + 非核心 %d 项 → 待筛选",
                            header_line, len(core_papers), len(non_core_papers),
                        )

        # ── Non-core (🧬 PICO / 🏥 safety) sections ──
        non_core_blocks = re.findall(
            r'#### [🧬🏥] .+?(?=\n#### 🎯|\n#### [🧬🏥]|\n### |\Z)',
            trial_analysis, re.DOTALL
        )
        all_non_core = []
        for block in non_core_blocks:
            paper_sections = re.findall(
                r'#### (?!🎯|🧬|🏥).+?(?=\n#### (?!🎯|🧬|🏥)|\Z)',
                block, re.DOTALL
            )
            for ps in paper_sections:
                if not self._is_empty_paper(ps):
                    cleaned = self._strip_empty_markers(ps).strip()
                    if cleaned:
                        all_non_core.append(cleaned)

        # ── Relevance pre-filter: drop non-core items clearly about wrong disease ──
        all_non_core = self._prefilter_non_core_relevance(all_non_core)

        if len(all_non_core) <= 1:
            selected_non_core = all_non_core
        else:
            selected_non_core = await self._select_non_core_item(all_non_core)

        if selected_non_core:
            result_parts.append("#### 其他试验")
            result_parts.extend(selected_non_core)

        logger.info(
            f"[合并] 完成: {len(trial_analysis)} → "
            f"{sum(len(p) for p in result_parts)} 字符, "
            f"{len(trial_blocks)} 个试验组保留"
        )

        if not result_parts:
            return trial_analysis
        return "\n\n".join(result_parts)

    @staticmethod
    def _is_empty_paper(paper_text: str) -> bool:
        """Check if a paper section contains no clinically useful content."""
        if not paper_text or not paper_text.strip():
            return True
        stripped = paper_text.strip()
        # Empty-result markers from synthesis prompt
        empty_markers = [
            "该检索方向无有效结果",
            "检索结果为空",
            "无法提取相关临床证据",
        ]
        for marker in empty_markers:
            if marker in stripped:
                # If the marker is the dominant content (not buried in real data)
                data_lines = [l for l in stripped.split('\n')
                              if l.strip() and not any(m in l for m in empty_markers)]
                if len(data_lines) < 2:
                    return True
        # Near-empty: only a title line with no data fields
        field_markers = ["研究类型", "样本量", "纳排标准", "入组人群", "干预与对照", "关键结论"]
        has_fields = any(fm in stripped for fm in field_markers)
        if not has_fields and len(stripped) < 100:
            return True

        # Near-empty: most substantive fields say "未明确报告" (no real data)
        placeholder_patterns = ["未明确报告", "未明确", "未报告"]
        substantive = 0
        empty = 0
        for fm in field_markers:
            m = re.search(rf'\*\*{fm}\*\*[：:]\s*(.+)', stripped)
            if m:
                substantive += 1
                val = m.group(1).strip()
                if any(p in val for p in placeholder_patterns):
                    empty += 1
        if substantive >= 3 and empty >= substantive * 2 / 3:
            return True

        return False

    @staticmethod
    def _strip_empty_markers(text: str) -> str:
        """Remove empty-result placeholder lines from a paper section."""
        if not text:
            return text
        lines = text.split('\n')
        # Remove lines that are solely empty-result markers
        empty_markers = [
            "该检索方向无有效结果",
            "检索结果为空",
        ]
        cleaned = [l for l in lines if not any(
            m in l and not any(fm in l for fm in ["研究类型", "样本量", "关键结论"])
            for m in empty_markers
        )]
        return '\n'.join(cleaned)

    # =================================================================
    # Non-core relevance pre-filter — code-level disease gate
    # =================================================================
    def _prefilter_non_core_relevance(self, items: list) -> list:
        """
        Quick code-level gate: drop non-core items that are clearly about
        a different disease (wrong cancer type, non-oncologic, etc.).

        This runs BEFORE the LLM _select_non_core_item to save a round-trip
        and to enforce a hard floor on relevance. Only extreme mismatches are
        filtered here; borderline cases are left for the LLM to judge.
        """
        if not items:
            return items

        # Must mention endometrial/uterine cancer or at minimum gynecologic oncology
        disease_pattern = re.compile(
            r'(?i)'
            r'(endometri|uterine|uterus|womb|'
            r'gynecolog|gyn[aec]|cervical|ovarian|vulvar|vaginal|'
            r'PORTEC|GOG|NRG|RUBY|ATTEND|DUO|KEYNOTE|'
            r'子宫|内膜|宫颈|卵巢|妇科|外阴|阴道|输卵管|盆腔|附件)'  # Chinese terms — ReAct outputs are Chinese per rule 10
        )

        # Clearly wrong disease — if the item name-drops another cancer prominently
        # without any mention of endometrial/uterine, drop it
        wrong_disease_pattern = re.compile(
            r'(?i)\b('
            r'lung cancer|non.small.cell.lung|NSCLC|SCLC|'
            r'prostate|breast cancer|colorectal|colon cancer|'
            r'pancreatic|gastric|hepatocellular|HCC|'
            r'glioblastoma|melanoma|leukemia|lymphoma|myeloma|'
            r'head and neck|thyroid|bladder|renal cell|RCC'
            r')\b'
        )

        # Staging methodology papers — no treatment decision value
        staging_methodology_pattern = re.compile(
            r'(?i)('
            r'FIGO\s*(2009|2023|staging).*compar|'
            r'compar.*FIGO\s*(2009|2023|staging)|'
            r'analysing the clinical outcomes between FIGO|'
            r'FIGO.*stage migration|'
            r'staging system.*comparison|'
            r'reclassification.*FIGO|'
            r'FIGO.*reclassif'
            r')'
        )

        # Papers exclusively about a different endometrial histological subtype
        # (these substrings indicate the WRONG histology for serous carcinoma patients)
        wrong_histology_pattern = re.compile(
            r'(?i)\b('
            r'carcinosarcoma|'        # 癌肉瘤 ≠ 浆液性癌
            r'malignant mixed müllerian|'  # MMMT = carcinosarcoma
            r'clear cell carcinoma'    # 透明细胞癌 ≠ 浆液性癌
            r')\b'
        )

        kept = []
        for item in items:
            # Extract title/first 300 chars for quick check
            head = item[:300]

            if not disease_pattern.search(head):
                logger.info(
                    "[非核心预过滤] 丢弃: 未提及妇科肿瘤相关疾病 → %s",
                    head.split('\n')[0][:100]
                )
                continue

            # If it prominently mentions a wrong disease AND doesn't
            # mention endometrial/uterine specifically, drop it
            endo_mention = bool(re.search(
                r'(?i)(endometri|uterine|uterus|womb)', head
            ))
            wrong_mention = wrong_disease_pattern.search(head)
            if wrong_mention and not endo_mention:
                logger.info(
                    "[非核心预过滤] 丢弃: 研究 %s 与子宫内膜癌无关 → %s",
                    wrong_mention.group(1),
                    head.split('\n')[0][:100]
                )
                continue

            # Drop staging methodology papers — no treatment decision value
            if staging_methodology_pattern.search(head):
                logger.info(
                    "[非核心预过滤] 丢弃: FIGO 分期方法学研究，无治疗决策价值 → %s",
                    head.split('\n')[0][:100]
                )
                continue

            # Drop papers exclusively about a different endometrial histological subtype
            # for serous carcinoma patients (癌肉瘤/透明细胞癌 ≠ 浆液性癌)
            patient_diagnosis = (
                self.structured_task.get("oncology_profile", {})
                .get("diagnosis_and_stage", "")
            )
            if "浆液性" in patient_diagnosis or "serous" in patient_diagnosis.lower():
                wrong_histo = wrong_histology_pattern.search(head)
                if wrong_histo:
                    endo_specific = bool(re.search(
                        r'(?i)(serous|浆液)', head
                    ))
                    if not endo_specific:
                        logger.info(
                            "[非核心预过滤] 丢弃: 非浆液性癌亚型 (%s)，患者为浆液性癌 → %s",
                            wrong_histo.group(1),
                            head.split('\n')[0][:100]
                        )
                        continue

            kept.append(item)

        if len(kept) < len(items):
            logger.info(
                "[非核心预过滤] %d 项 → %d 项 (丢弃 %d 项明显无关的研究)",
                len(items), len(kept), len(items) - len(kept)
            )

        return kept

    # =================================================================
    # LLM-based non-core item selection
    # =================================================================
    async def _select_non_core_item(self, non_core_sections: list) -> list:
        """从非核心条目中用 LLM 选最相关的 1 项，或全部拒绝。"""
        profile = self.structured_task.get("oncology_profile", {}) or {}
        basic_info = profile.get("basic_info", "").strip()
        diagnosis = profile.get("diagnosis_and_stage", "").strip()
        pathology = profile.get("pathology_and_molecular", "").strip()
        patient_context = f"基本信息：{basic_info}\n诊断与分期：{diagnosis}\n病理与分子分型：{pathology}"

        items_text = "\n---\n".join(
            f"【条目 {i+1}】\n{sec}" for i, sec in enumerate(non_core_sections)
        )

        prompt = textwrap.dedent(f"""
        请从以下非核心检索条目中，选出与患者最相关、对治疗决策最有价值的 **1 项**。

        【患者信息】：
        {patient_context}

        选择标准（按优先级排序，必须全部满足）：
        ① **疾病匹配——组织学亚型必须一致**：
           该研究是否针对患者的**确切组织学亚型**？
           - 若患者为浆液性癌 → 条目仅讨论癌肉瘤/透明细胞癌/内膜样癌且未包含浆液性癌 → 淘汰。
           - 若患者为内膜样癌 → 条目仅讨论浆液性癌/癌肉瘤 → 淘汰。
           - 若患者为癌肉瘤 → 条目仅讨论内膜样癌/浆液性癌 → 淘汰。
           不同组织学亚型是不同的疾病实体，治疗方案和预后均不同。
           🔴 **"子宫内膜癌肉瘤" ≠ "子宫内膜浆液性癌"——禁止因都有"子宫内膜"四字而混淆。**
        ② **分期匹配**：内容是否涉及患者当前分期（术后辅助治疗阶段）的治疗决策？
        ③ **决策价值**：是否包含直接影响方案选择的数据（OS HR、毒性数据、亚组分析）？
           🔴 **以下类型直接淘汰**：
           - FIGO 2009 vs 2023 分期系统对比研究（纯方法学，无治疗数据）
           - 分期系统验证/迁移研究（仅讨论分期变化，不涉及治疗方案比较）
           - 单一机构的回顾性分期描述（无治疗干预比较）
           - 纯预后描述性研究（仅描述生存率，无治疗方案对比）
        ④ **证据级别**：是否为 RCT、系统综述/Meta分析，或大样本回顾性研究？

        如果**所有条目都无治疗决策价值**（组织学不匹配、纯方法学、无治疗数据、非辅助治疗阶段），
        请输出 REJECT_ALL。
        如果有一条相关，请直接输出你选中的那条原文（保留 #### 标题和所有内容），不要加任何说明。
        """)

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, prompt, timeout=300.0, max_retries=3
            )
            selected = remove_think_tags(resp.content).strip()
            if "REJECT_ALL" in selected.upper():
                logger.info(
                    f"非核心条目筛选: {len(non_core_sections)} 项全部与患者无关 → 全部丢弃"
                )
                return []
            logger.info(f"非核心条目筛选完成: {len(non_core_sections)} 项 → 选中 1 项 ({len(selected)} 字符)")
            return [selected]
        except Exception as e:
            logger.warning(f"非核心条目筛选异常，保留第一条: {e}")
            return [non_core_sections[0]]

    # =================================================================
    # Title-based deduplication — 合并跨节重复论文
    # =================================================================
    @staticmethod
    def _tokenize_title(title: str) -> set:
        """Extract meaningful tokens (>=3 chars) from a title for fuzzy matching."""
        tokens = re.findall(r'[a-z0-9一-鿿]{3,}', title.lower())
        return set(tokens)

    def _deduplicate_trial_analysis(self, trial_analysis: str) -> str:
        """
        Deduplicate paper entries across different trial sections.

        Uses two-stage matching:
          1. Exact match on first 60 chars of normalized title
          2. Token-overlap fuzzy match (>85%) for LLM-reworded duplicates
             (e.g. "for women with" vs "in women with")

        The trial_analysis has the structure:
          #### 🎯 PORTEC-3
            #### Molecular Classification of PORTEC-3 [^^8]
            ...
          #### 🎯 GOG-0258
            #### Molecular Classification of PORTEC-3 [^^8]   ← duplicate
            ...

        When the same paper appears under multiple trial sections,
        keep only the first occurrence.
        """
        lines = trial_analysis.split('\n')
        seen_titles: dict = {}  # {clean_title: token_set}
        output = []
        skip = False

        for line in lines:
            stripped = line.strip()

            # Detect any #### header
            if stripped.startswith('#### '):
                # Section headers contain emoji — always reset skip
                if '🎯' in stripped or '🧬' in stripped or '🏥' in stripped:
                    skip = False
                else:
                    # Paper entry header: #### Title [^^n]
                    title_text = re.sub(r'\[\^\^\d+\]', '', stripped).strip()
                    title_text = re.sub(r'^####\s+', '', title_text)
                    clean = re.sub(r'\s+', ' ', title_text.lower())
                    tokens_new = self._tokenize_title(clean)

                    is_dup = False
                    for seen_key, seen_tokens in seen_titles.items():
                        # Stage 1: exact prefix match (first 60 chars)
                        if clean[:60] == seen_key[:60]:
                            is_dup = True
                            break
                        # Stage 2: token-overlap fuzzy match for LLM-reworded titles
                        if len(tokens_new) >= 5 and len(seen_tokens) >= 5:
                            overlap = len(tokens_new & seen_tokens) / min(len(tokens_new), len(seen_tokens))
                            if overlap > 0.85:
                                is_dup = True
                                break

                    if is_dup:
                        skip = True
                        logger.info("[去重] 跳过重复论文: %s...", clean[:60])
                        continue
                    else:
                        seen_titles[clean] = tokens_new
                        skip = False

            if not skip:
                output.append(line)

        return '\n'.join(output)

    # =================================================================
    # Intra-trial deduplication — detect near-duplicate sub-entries within same trial
    # =================================================================
    def _deduplicate_intra_trial(self, trial_analysis: str) -> str:
        """
        Deduplicate sub-entries WITHIN the same trial section.

        The existing _deduplicate_trial_analysis deduplicates papers ACROSS
        different trials (same paper title under PORTEC-3 *and* GOG-0258).
        This catches same-trial duplicates like PORTEC-3 final results [9]
        and PORTEC-3 patterns of recurrence [10] which have nearly identical
        survival data but different titles.

        Strategy: extract numerical fingerprints (%, N=, HR, P values) from each
        sub-entry; if two sub-entries under the same trial share >70% fingerprint
        overlap, keep the content-richer one.
        """
        if not trial_analysis:
            return trial_analysis

        # Split into top-level trial sections
        section_pattern = r'(#### [🎯🧬🏥].+?)(?=\n#### [🎯🧬🏥]|\Z)'
        trial_sections = re.findall(section_pattern, trial_analysis, re.DOTALL)
        if not trial_sections:
            return trial_analysis

        output_sections = []

        for section in trial_sections:
            # Sub-entries are #### lines that DON'T start with the emoji headers
            sub_pattern = r'(#### (?![🎯🧬🏥]).+?)(?=\n#### (?![\n🎯🧬🏥])|\Z)'
            subs = re.findall(sub_pattern, section, re.DOTALL)

            if len(subs) <= 1:
                output_sections.append(section)
                continue

            # Build fingerprints: set of numerical tokens
            fingerprinted = []
            for sub in subs:
                nums = set(re.findall(r'\d+%|N[=:]?\s*\d+|HR\s*[\d.]+|P\s*[=<>]\s*[\d.]+|[\d.]+\s*年', sub))
                fingerprinted.append((sub, nums))

            # Pairwise dedup
            keep = [True] * len(fingerprinted)
            for i in range(len(fingerprinted)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(fingerprinted)):
                    if not keep[j]:
                        continue
                    if not fingerprinted[i][1] or not fingerprinted[j][1]:
                        continue
                    intersection = fingerprinted[i][1] & fingerprinted[j][1]
                    smaller = min(len(fingerprinted[i][1]), len(fingerprinted[j][1]))
                    similarity = len(intersection) / smaller if smaller > 0 else 0
                    if similarity > 0.7:
                        # Keep the longer entry
                        if len(fingerprinted[i][0]) >= len(fingerprinted[j][0]):
                            keep[j] = False
                            logger.info(
                                "[intra-trial 去重] 相似度 %.0f%% — 丢弃 %s",
                                similarity * 100,
                                fingerprinted[j][0][:60].replace('\n', ' ')
                            )
                        else:
                            keep[i] = False
                            logger.info(
                                "[intra-trial 去重] 相似度 %.0f%% — 丢弃 %s",
                                similarity * 100,
                                fingerprinted[i][0][:60].replace('\n', ' ')
                            )
                            break

            # Rebuild section with deduped sub-entries
            kept_subs = [fingerprinted[i][0].strip() for i in range(len(fingerprinted)) if keep[i]]
            # Preserve the section header
            first_line = section.split('\n')[0]
            rebuilt = first_line + '\n\n' + '\n\n'.join(kept_subs)
            output_sections.append(rebuilt)

        result = '\n\n'.join(output_sections)
        if result != trial_analysis:
            logger.info("[intra-trial 去重] 完成: %d 字符 → %d 字符", len(trial_analysis), len(result))
        return result

    @staticmethod
    def _demote_paper_subheadings(trial_analysis: str) -> str:
        """
        Demote #### paper headings to ##### within trial groups.

        After consolidation, the structure is:
          #### 🎯 PORTEC-3        ← trial group header (keep)
          #### Paper Title [^^n]   ← paper entry (demote to #####)
          ...

        This creates a clear visual hierarchy in the final MDT report
        without changing the semantics that downstream agents rely on.
        Done as pure string manipulation — no LLM involved.
        """
        lines = trial_analysis.split('\n')
        result = []
        for line in lines:
            if line.startswith('#### ') and not any(
                emoji in line for emoji in ['🎯', '🧬', '🏥']
            ):
                result.append('#' + line)  # #### → #####
            else:
                result.append(line)
        return '\n'.join(result)

    def _filter_irrelevant_trials(self, trial_analysis: str) -> str:
        """
        Remove core trial sections whose inclusion criteria clearly don't
        match the patient's stage and histology.

        E.g. PORTEC-1 (stage I endometrioid only) is irrelevant for a
        stage III serous patient.
        """
        if not trial_analysis or "超时失败" in trial_analysis:
            return trial_analysis

        # Extract patient characteristics from structured_task
        profile = self.structured_task.get("oncology_profile", {}) or {}
        diagnosis = (profile.get("diagnosis_and_stage", "") or "").lower()
        pathology = (profile.get("pathology_and_molecular", "") or "").lower()

        # Determine patient's stage category
        is_early_stage = bool(re.search(
            r'(?i)[iⅠⅰ][a-c]?\d*\s*期|stage\s*i[abc]?\b|早期',
            diagnosis
        ))
        is_advanced_stage = bool(re.search(
            r'(?i)[iⅠⅰ]{2,}[a-c]?\d*\s*期|stage\s*iii|stage\s*iv|局部晚期|晚期|iiic',
            diagnosis
        ))
        has_serous = bool(re.search(r'(?i)浆液|serous', pathology))
        is_endometrioid = bool(re.search(r'(?i)内膜样|endometrioid', pathology))
        has_high_risk = bool(re.search(
            r'(高危|high.risk|g3\b|grade\s*3\b|浆液性|serous|'
            r'lvsi|深肌层浸润|deep.myometrial|non.endometrioid|非子宫内膜样)',
            diagnosis + " " + pathology
        ))
        is_recurrent = bool(re.search(r'(?i)复发|recurr|relapse', diagnosis))
        is_stage4a = bool(re.search(r'(?i)iva\s*期|stage\s*iva\b', diagnosis))
        is_stage4b = bool(re.search(r'(?i)ivb?\s*期|stage\s*iv', diagnosis))
        is_stage4b = is_stage4b and not is_stage4a  # 排除 IVA（归入局部晚期）

        # Trial inclusion criteria (returns True if trial IS relevant to patient)
        trial_rules = {
            # ── Early-stage trials ──
            "PORTEC-1": (    # Stage I endometrioid only
                lambda: is_early_stage and is_endometrioid
            ),
            "PORTEC-2": (    # Stage I-II endometrioid HIR
                lambda: not is_advanced_stage and is_endometrioid
            ),
            "GOG-99": (      # Stage I-II
                lambda: is_early_stage
            ),
            "PORTEC-4a": (   # Stage I HIR — molecular-guided RT (de-escalation/escalation)
                lambda: not is_advanced_stage
            ),
            # ── III / IVA / early-high-risk trials ──
            "PORTEC-3": (    # Stage I high-risk / Stage II / Stage III / IVA
                lambda: is_advanced_stage or has_high_risk
            ),
            "GOG-0258": (    # Stage I high-risk / Stage II / Stage III / IVA
                lambda: is_advanced_stage or has_high_risk or is_recurrent
            ),
            # ── IVB / recurrent-first-line trials ──
            "NRG-GY018": (   # Stage IVB / recurrent (immunotherapy)
                lambda: is_stage4b or is_recurrent
            ),
            "RUBY": (        # Stage IVB / recurrent (immunotherapy)
                lambda: is_stage4b or is_recurrent
            ),
            "GOG-209": (     # Stage IVB / recurrent (chemotherapy)
                lambda: is_stage4b or is_recurrent
            ),
            "ATTEND": (      # Stage IVB / recurrent (immunotherapy)
                lambda: is_stage4b or is_recurrent
            ),
            "DUO-E": (       # Stage IVB / recurrent (immunotherapy)
                lambda: is_stage4b or is_recurrent
            ),
            # ── Second-line ──
            "KEYNOTE-775": ( # Second-line / recurrent
                lambda: is_recurrent
            ),
        }

        # Split by emoji headers ONLY — never split on individual paper titles
        sections = re.findall(
            r'#### [🎯🧬🏥].+?(?=\n#### [🎯🧬🏥]|\Z)',
            trial_analysis, re.DOTALL
        )
        if not sections:
            return trial_analysis

        kept = []
        removed = []

        for sec in sections:
            first_line = sec.split('\n')[0]
            section_label = re.sub(r'^####\s*[🎯🧬🏥]\s*', '', first_line).strip()

            # Extract trial acronym (e.g. PORTEC-3, GOG-99) for exact matching.
            # Substring match (e.g. "PORTEC-1" in "PORTEC-3") would cause
            # false positives — use token-level equality instead.
            trial_acronym_match = re.search(
                r'\b([A-Z]+-\d+[A-Za-z]*)\b', section_label
            )
            section_trial = trial_acronym_match.group(1) if trial_acronym_match else ''

            should_remove = False
            for trial_name, rule_fn in trial_rules.items():
                if section_trial.lower() == trial_name.lower():
                    if not rule_fn():
                        should_remove = True
                        break

            if should_remove:
                removed.append(section_label)
                logger.info(
                    "[相关性过滤] 移除: %s — 该试验入组人群与患者分期/分型不匹配 "
                    "(诊断: %s, 病理: %s)",
                    section_label[:60], diagnosis[:40], pathology[:40]
                )
            else:
                kept.append(sec)

        if removed:
            logger.info("[相关性过滤] 共移除 %d 项: %s", len(removed), "、".join(r[:30] for r in removed))

        return "\n\n".join(kept) if kept else trial_analysis

    # =================================================================
    # Statistical sanity check — 拦截数学矛盾
    # =================================================================
    def _statistical_sanity_check(self, report: str) -> str:
        """
        Find and flag impossible statistical claims.

        Pattern: "X% vs X%" followed by "...P=0.xxx" or "...P<0.xxx"
        where the two percentages are identical but a significant P-value is claimed.
        """
        pattern = r'(\d+)\s*%\s*vs\s*\1\s*%[\s\S]{0,200}?[Pp]\s*[<＝=]\s*0\.\d+'
        matches = list(re.finditer(pattern, report))

        if not matches:
            return report

        # Process in reverse order to preserve offsets
        for m in reversed(matches):
            equal_val = m.group(1)
            p_context = report[m.start():m.end()].split('P')[-1].split('=')[-1].split('<')[-1].strip()[:10]
            logger.warning(
                "[统计校验] ⚠️ 矛盾数据: %s%% vs %s%%，P 值却声称显著 (%s) → 已标记",
                equal_val, equal_val, p_context
            )
            report = (
                report[:m.start()]
                + f"【⚠️ 数据矛盾：{equal_val}% vs {equal_val}% 相等但有显著 P 值】"
                + report[m.end():]
            )

        return report

    # =================================================================
    # Fix numbering disorder in 术后处理 section
    # =================================================================
    @staticmethod
    def _fix_numbering_in_postop(report: str) -> str:
        """
        Renumber all numbered items sequentially in the 术后处理 section.

        Fixes issues like "1、2、4、5、6、3" by reassigning numbers
        in order of appearance.  Respects ### sub-section boundaries
        (主要方案 / 合并症管理) by resetting the counter at each ### header.
        """
        for punct in ['、', '．', '.', '：', ':']:
            pattern = rf"(##\s*二{re.escape(punct)}\s*术后处理[\s\S]*?)(?=\n##\s*[三四]|\Z)"
            m = re.search(pattern, report, re.DOTALL)
            if not m:
                continue
            section = m.group(1)
            lines = section.split('\n')
            new_lines = []
            counter = 0
            for line in lines:
                # Reset counter at ### sub-section boundaries
                if re.match(r'^###\s', line):
                    counter = 0
                    new_lines.append(line)
                    continue
                numbered_match = re.match(r'^(\s*)(\d+)([、．.])\s', line)
                if numbered_match:
                    counter += 1
                    prefix = numbered_match.group(1)
                    sep = numbered_match.group(3)
                    rest = line[numbered_match.end():]
                    new_lines.append(f"{prefix}{counter}{sep}{rest}")
                else:
                    new_lines.append(line)
            if counter > 0:
                new_section = '\n'.join(new_lines)
                report = report[:m.start()] + new_section + report[m.end():]
            break
        return report

    def _check_citation_credibility(self, report: str) -> dict:
        """Check citation integrity: every [^^n] should map to a real ref in pool."""
        citations = re.findall(r'\[\^\^(\d+)\]', report)
        if not citations:
            return {"total": 0, "valid": 0, "broken": 0, "rate": 0.0}

        total = len(citations)
        valid = 0
        broken = 0
        for c in citations:
            try:
                idx = int(c)
                ref = self.ref_pool.get_ref_by_idx(idx)
                if ref:
                    valid += 1
                else:
                    broken += 1
            except ValueError:
                broken += 1

        return {
            "total": total,
            "valid": valid,
            "broken": broken,
            "rate": valid / total if total > 0 else 0.0
        }

    # Note: _reindex_references is now delegated to ref_pool.reindex_references()
