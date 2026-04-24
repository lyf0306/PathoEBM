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
    invoke_with_timeout_and_retry,
    remove_think_tags,
    write_log_process_safe,
)

# --- New modular imports ---
from .models.reference_pool import ReferencePool
from .agents.clinical_trial_agent import ClinicalTrialAgent
from .agents.followup_agent import FollowupAgent
from .agents.prognosis_agent import PrognosisAgent
from .agents.mdt_report_agent import MDTReportAgent
from .agents.context_bus import AgentContextBus
from .agents.reviewer_agent import ReviewerAgent
from .agents.react_search_agent import ReActSearchAgent
from .pipeline.search_planner import SearchPlanner
from .pipeline.knowledge_processor import KnowledgeProcessor
from .pipeline.prognosis_retrieval import PrognosisRetrieval

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


class AdvancedSearchSystem:
    """
    Orchestrator for the evidence update pipeline.

    Delegates to specialized modules:
      - SearchPlanner for generating follow-up search questions
      - KnowledgeProcessor for evidence synthesis
      - PrognosisRetrieval for independent prognosis data track
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
    async def _get_follow_up_questions(self, current_knowledge: str, query: str) -> List[str]:
        """Delegate to SearchPlanner for generating follow-up search questions."""
        planner = SearchPlanner(
            self.tool_planning_model,
            self.structured_task,
            self.questions_per_iteration
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
        1. **灯塔临床试验**：核心试验（PORTEC-3, GOG-0258, NRG-GY018等）是否有对应生存数据？
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
                self.tool_planning_model, prompt, timeout=180.0, max_retries=2
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
        """Independent prognosis retrieval pipeline with skill data."""
        # Load authoritative prognosis reference data
        skill_data = ""
        try:
            from .skills.prognosis.prognosis_skill import PrognosisSkill
            skill = PrognosisSkill()
            skill_data = skill.get_data()
            if skill_data:
                logger.info(f"已加载 PrognosisSkill 参考数据 ({len(skill_data)} 字符)")
        except Exception as e:
            logger.warning(f"加载 PrognosisSkill 失败: {e}")

        # Run dual-tier PubMed retrieval
        retrieval = PrognosisRetrieval(
            self.tool_planning_model,
            self.fast_model,
            self.mcp_tool_client,
            self.error_log_path,
            self.structured_task,
            self.ref_pool,
            self.all_links_of_system,
        )
        pubmed_results = await retrieval.run()

        return {
            "skill_data": skill_data,
            "population": pubmed_results.get("population", ""),
            "molecular": pubmed_results.get("molecular", ""),
            "raw_combined": pubmed_results.get("raw_combined", ""),
        }

    # -------------------------------------------------------------------------
    # Report Generation (delegates to agent classes)
    # -------------------------------------------------------------------------
    async def _generate_detailed_report(
        self, current_knowledge: str, findings: List[Dict],
        query: str, iteration: int, prognosis_results: dict = None,
        context_bus: AgentContextBus = None,
    ):
        # Anti-explosion shield for vLLM
        if hasattr(self.report_model, 'max_tokens'):
            self.report_model.max_tokens = 4096
        if hasattr(self.report_model, 'max_completion_tokens'):
            self.report_model.max_completion_tokens = 4096

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

        logger.info("多智能体并发: 试验沙盒聚类隔离解析 & 定制随访 & 预后提取...")

        # --- Run Agent 1 (Clinical Trial), Agent 1.5 (Follow-up), Agent 3 (Prognosis) ---
        trial_agent = ClinicalTrialAgent(
            self.report_model, self.fast_model, self.treatment_context,
            context_bus=context_bus,
        )
        followup_agent = FollowupAgent(self.report_model, self.treatment_context)
        prognosis_agent = PrognosisAgent(
            self.report_model, self.treatment_context,
            context_bus=context_bus,
        )

        try:
            trial_analysis, followup_plan, prognosis_data = await asyncio.wait_for(
                asyncio.gather(
                    trial_agent.run(current_knowledge),
                    followup_agent.run(),
                    prognosis_agent.run(
                        skill_data=skill_data,
                        population_data=population_data,
                        molecular_data=molecular_data,
                    ),
                ),
                timeout=600.0
            )
            if timer := getattr(self, '_timer', None):
                timer.lap("多智能体并发")
            # Post structured data to the bus for downstream agents
            await context_bus.post("ClinicalTrialAgent", "trial_analysis",
                                   trial_analysis)
            await context_bus.post("PrognosisAgent", "prognosis_data",
                                   prognosis_data)
        except Exception as e:
            logger.error(f"多智能体并发执行彻底超时或崩溃: {e}")
            trial_analysis = "试验解析模块超时失败。"
            followup_plan = "随访方案生成超时失败。"
            prognosis_data = "预后数据提取超时失败。"
            await context_bus.post("System", "agent_failure", str(e))

        logger.info(f"预后专员提取结果: {prognosis_data}")

        # --- Consolidation: merge cross-query duplicate trials before MDT ---
        trial_analysis = await self._consolidate_trial_analysis(trial_analysis)

        # --- Agent 2: MDT Report Chief Writer ---
        mdt_agent = MDTReportAgent(
            self.report_model, self.treatment_context, self.structured_task,
            context_bus=context_bus,
        )
        content = await mdt_agent.run(trial_analysis, followup_plan, prognosis_data)
        await context_bus.post("MDTReportAgent", "report_generated",
                               f"报告初稿长度: {len(content)} 字符")
        if timer := getattr(self, '_timer', None):
            timer.lap("MDT撰写")

        # --- Optional: Reviewer Agent (post-hoc quality check & repair) ---
        reviewer = ReviewerAgent(self.report_model)
        content, was_repaired = await reviewer.review_and_repair(
            report=content,
            trial_analysis=trial_analysis,
            followup_plan=followup_plan,
            prognosis_data=prognosis_data,
            treatment_context=self.treatment_context,
            context_bus=context_bus,
        )
        if timer := getattr(self, '_timer', None):
            timer.lap("审查修复")
        if was_repaired:
            await context_bus.post("ReviewerAgent", "repair_applied",
                                   "报告质量问题已修复")
            logger.info("[Reviewer] 报告已修复并重新处理引用。")
        else:
            logger.info("[Reviewer] 报告无需修复。")

        # Final safety: strip any lingering think tags from all LLM outputs
        content = remove_think_tags(content)

        # Post-process references
        try:
            new_content, refs_section = self.ref_pool.reindex_references(content)
            full_report = new_content + "\n" + refs_section
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
            # --- First iteration: planner generates 5 targeted questions ---
            # --- Later iterations: only if coverage check found gaps ---
            if iteration == 0:
                questions = await self._get_follow_up_questions(current_knowledge, query)
                if not questions:
                    questions = [query]
            else:
                coverage = await self._check_evidence_coverage(current_knowledge)
                if coverage.get("sufficient", True):
                    logger.info("证据覆盖度已达标，无需第二轮检索。")
                    break
                gap_queries = coverage.get("gap_queries", [])[:3]
                if not gap_queries:
                    logger.info("无明确证据缺口，结束检索。")
                    break
                questions = gap_queries
                logger.info(f"证据存在缺口，补充检索 {len(questions)} 个方向")

            self.questions_by_iteration[iteration] = questions
            logger.info(f"Iteration {iteration+1}: Concurrently processing {len(questions)} sub-questions...")

            logger.info(f"启动并发检索 ({len(questions)} 个独立分支)...")
            all_questions_results = await asyncio.gather(
                *(react_agent.execute(q) for q in questions)
            )

            # ReActAgent.execute() now returns synthesized analysis per query
            # with [^^n] citations. No dedup/truncation/answer_query needed.
            all_synthesized = all_questions_results

            chunk_knowledge = f"\n\n### 第 {iteration + 1} 轮检索分析：\n"
            for q, syn in zip(questions, all_synthesized):
                if syn.strip():
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
        3. **【灯塔试验绝对豁免权】（最核心红线）**：如果在标题中看到了重磅临床试验的代号（如 PORTEC-3, GOG-0258, RUBY, NRG 等），只要符合该患者领域，你**必须无条件赋予最高优先级选入**！绝对不能把它们当作冗余筛掉！
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
        Consolidate trial analysis output: merge duplicate entries about the same trial
        (same trial found by multiple queries) into one entry per trial.

        This runs after ClinicalTrialAgent and before MDT placeholder replacement.
        """
        if not trial_analysis or "超时失败" in trial_analysis or "未发现" in trial_analysis:
            return trial_analysis

        consolidate_prompt = textwrap.dedent(f"""
        你是一名妇科肿瘤 MDT 报告编辑专家。
        以下是从临床证据中提取的核心试验分析，可能因多个检索词命中同一试验而导致重复条目。
        请合并重复的试验条目，确保最终输出中每个试验只出现一次。

        【待整理的试验分析】：
        {trial_analysis}

        【合并规则】：
        1. 如果同一试验（如 PORTEC-3、GOG-0258、NRG-GY018 等）出现在多个条目中，合并为一条
        2. 合并时保留所有不重复的数据点（纳入人群、干预方案、生存获益、亚组分析、毒性），去掉冗余信息
        3. 所有文献角标 [^^n] 原样保留，不得删改
        4. 保持原有的 #### 标题格式和 markdown 结构
        5. 最终输出最多 3 项，按临床重要性从高到低排序

        请直接输出合并整理后的内容，不要加额外说明。
        """)

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, consolidate_prompt, timeout=120.0, max_retries=1
            )
            consolidated = remove_think_tags(resp.content).strip()
            logger.info(f"试验分析合并完成: {len(trial_analysis)} → {len(consolidated)} 字符")
            return consolidated
        except Exception as e:
            logger.warning(f"试验分析合并异常，退回原始输出: {e}")
            return trial_analysis

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
