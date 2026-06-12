"""
PathoEBM Search System - Orchestrator Module

Refactored from a monolithic class into a thin coordinator that delegates to
specialized agent and pipeline modules.

Architecture:
  - models/reference_pool.py       → ReferencePool (citation management)
  - agents/                        → Independent agent classes
  - pipeline/                      → Search planning, knowledge processing, mixins

The AdvancedSearchSystem class is composed via mixin inheritance from:
  - pipeline/report_orchestrator.py   → ReportGenerationMixin
  - pipeline/article_processor.py     → ArticleProcessingMixin
  - pipeline/deduplication.py         → DeduplicationMixin
  - pipeline/post_processing.py       → PostProcessingMixin

This module retains AdvancedSearchSystem as the public API for backward compatibility.
"""

import asyncio
import json
import logging
import os
import re
import time
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
from .utilities.search_utilities import (
    ensure_chinese_output,
    invoke_with_timeout_and_retry,
    remove_think_tags,
    strip_llm_preamble,
    write_log_process_safe,
)

# --- New modular imports ---
from .models.reference_pool import ReferencePool
from .concurrency.backpressure import bounded_gather
from .agents.followup_agent import FollowupAgent
from .agents.prognosis_agent import PrognosisAgent
from .agents.mdt_report_agent import MDTReportAgent
from .agents.context_bus import AgentContextBus
from .agents.reviewer_agent import ReviewerAgent
from .evaluation.report_evaluator import ReportEvaluator
from .agents.react_search_agent import ReActSearchAgent
from .pipeline.search_planner import SearchPlanner, SearchPlan
from .pipeline.knowledge_processor import KnowledgeProcessor
from .prompts import prompt_manager

# --- Mixin imports ---
from .pipeline.report_orchestrator import ReportGenerationMixin
from .pipeline.article_processor import ArticleProcessingMixin
from .pipeline.deduplication import DeduplicationMixin
from .pipeline.post_processing import PostProcessingMixin

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
        llm_limit = settings.pipeline.llm_concurrency
        GLOBAL_LLM_SEMAPHORE = asyncio.Semaphore(llm_limit)
        logger.info(f"GLOBAL_LLM_SEMAPHORE initialized with limit={llm_limit}")
    if GLOBAL_API_SEMAPHORE is None:
        api_limit = settings.pipeline.api_concurrency
        GLOBAL_API_SEMAPHORE = asyncio.Semaphore(api_limit)
        logger.info(f"GLOBAL_API_SEMAPHORE initialized with limit={api_limit}")
    return GLOBAL_LLM_SEMAPHORE, GLOBAL_API_SEMAPHORE


# ── Core clinical trial keywords for differentiated ReAct rounds ──
_LIGHTHOUSE_TRIAL_PATTERNS = re.compile(
    r'\b(PORTEC-(1|2|3|4a?|4)|GOG-(99|0258|209)|NRG-GY018|RUBY|KEYNOTE-775)\b',
    re.IGNORECASE,
)


def _is_core_trial_query(query: str) -> bool:
    """Check if a search query targets a core clinical trial (needs deeper ReAct search)."""
    return bool(_LIGHTHOUSE_TRIAL_PATTERNS.search(query))


class AdvancedSearchSystem(
    ReportGenerationMixin,
    ArticleProcessingMixin,
    DeduplicationMixin,
    PostProcessingMixin,
):
    """
    Orchestrator for the evidence update pipeline.

    Delegates to specialized modules:
      - SearchPlanner for generating follow-up search questions
      - KnowledgeProcessor for evidence synthesis
      - PrognosisSkill for authoritative survival rate references
      - ClinicalTrialAgent / FollowupAgent / PrognosisAgent / MDTReportAgent
        for final report generation.

    Composed with mixins:
      - ReportGenerationMixin: _generate_detailed_report + review loop
      - ArticleProcessingMixin: _process_tool_result, _screen_articles, etc.
      - DeduplicationMixin: trial dedup, non-core filtering
      - PostProcessingMixin: statistical sanity, numbering fix, citation check
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
        progress_reporter=None,
        cancel_event: "asyncio.Event | None" = None,
        llm_semaphore: "asyncio.Semaphore | None" = None,
        api_semaphore: "asyncio.Semaphore | None" = None,
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
        self.progress_reporter = progress_reporter
        self.cancel_event = cancel_event
        self._llm_semaphore = llm_semaphore
        self._api_semaphore = api_semaphore

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

    # -------------------------------------------------------------------------
    # Pipeline: Evidence Coverage
    # -------------------------------------------------------------------------
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
        prompt = prompt_manager.get("evidence_coverage").format(
            treatment_context=self.treatment_context[:2000],
            current_knowledge=current_knowledge[:4000],
        )

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
        except asyncio.CancelledError:
            raise
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
        # Prefer instance-level semaphores when provided (per-request isolation)
        if self._llm_semaphore is not None:
            llm_semaphore = self._llm_semaphore
        if self._api_semaphore is not None:
            api_semaphore = self._api_semaphore
        self._timer = _Timer()

        await self.initialize()
        self._timer.lap("系统初始化")
        if self.progress_reporter:
            self.progress_reporter.report("parsing", "结构化提取完成，深搜系统初始化完毕")

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

        # ── Planner generates structured search plan ──
        search_plan = await self._get_follow_up_questions("", query)
        questions = search_plan.questions if search_plan.questions else [query]
        if not search_plan.questions:
            search_plan = None
        else:
            search_plan = _deduplicate_questions(search_plan)
            questions = search_plan.questions

        self.questions_by_iteration[0] = questions
        logger.info("流水线模式: %d 个初始检索问题入队", len(questions))
        if self.progress_reporter:
            self.progress_reporter.report("searching", f"流水线启动: {len(questions)}个方向并行搜索")

        # ── Set up async pipeline ──
        from .concurrency.task_queue import AsyncioTaskQueue
        from .pipeline.knowledge_accumulator import KnowledgeAccumulator
        from .pipeline.coverage_monitor import CoverageMonitor

        queue = AsyncioTaskQueue()
        accumulator = KnowledgeAccumulator()
        done_event = asyncio.Event()
        worker_concurrency = settings.pipeline.worker_concurrency
        max_rounds_per_agent = settings.pipeline.max_rounds_per_agent

        # Enqueue initial questions with meta-type tags
        if search_plan is not None and search_plan.has_trial_grouping:
            for trial_name, sq_indices in search_plan.trial_mapping:
                sub_queries = [questions[i] for i in sq_indices]
                await queue.put(("trial", (trial_name, sub_queries)))
            for i in (search_plan.pico_indices or []):
                await queue.put(("pico", questions[i]))
            for i in (search_plan.comorb_indices or []):
                await queue.put(("comorb", questions[i]))
        else:
            for q in questions:
                await queue.put(("flat", q))

        logger.info("已入队 %d 个初始任务", queue.qsize())

        # ── Pipeline worker ──
        seen_queries: set[str] = set()

        def _normalize_for_dedup(q: str) -> str:
            """Normalize query string for dedup comparison."""
            import re as _re
            q = q.lower().strip()
            q = _re.sub(r'\s+', ' ', q)
            return q[:120]

        async def _pipeline_worker():
            while not done_event.is_set():
                if self.cancel_event and self.cancel_event.is_set():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if done_event.is_set() and queue.empty():
                        break
                    continue
                except asyncio.CancelledError:
                    break

                meta_type, payload = item
                if meta_type is None:  # sentinel
                    queue.task_done()
                    break

                try:
                    if meta_type == "trial":
                        trial_name, sub_queries = payload
                        result = await react_agent.execute_trial(
                            trial_name, sub_queries, max_rounds=max_rounds_per_agent,
                        )
                        meta_label = trial_name
                        display_query = trial_name
                    else:
                        result = await react_agent.execute(
                            payload, max_rounds=max_rounds_per_agent,
                        )
                        meta_label = str(payload)[:80]
                        display_query = str(payload)

                    synthesis = result.get("synthesis", "") if isinstance(result, dict) else str(result)
                    # Strip LLM conversational preamble from each branch
                    synthesis = strip_llm_preamble(synthesis)

                    # Log completion
                    syn_text = synthesis if isinstance(result, dict) else str(result)
                    desc = f"[{meta_type.upper()}] {meta_label}"
                    msg = (
                        f"\n{'='*60}\n"
                        f"🔍 独立分支 ReAct 分析完毕\n📍 {desc}\n{'-'*60}\n"
                        f"{syn_text.strip()}\n{'='*60}\n"
                    )
                    print(msg)
                    logger.info(msg)

                    if synthesis.strip() and "无相关临床证据" not in synthesis:
                        await accumulator.add(
                            query=display_query,
                            synthesis=synthesis,
                            meta_type=meta_type,
                            meta_label=str(meta_label),
                            sufficient=result.get("sufficient", True) if isinstance(result, dict) else True,
                        )

                    # Inject follow-up queries into the pipeline (deduped)
                    if isinstance(result, dict):
                        for fq in result.get("follow_up_queries", []):
                            if fq and str(fq).strip():
                                fq_str = str(fq).strip()
                                norm = _normalize_for_dedup(fq_str)
                                if norm not in seen_queries:
                                    seen_queries.add(norm)
                                    await queue.put(("flat", fq_str))
                                    logger.info("  ↪ 跟进查询入队: %s", fq_str[:80])
                                else:
                                    logger.info("  ⊘ 重复跟进查询已跳过: %s", fq_str[:80])
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("Pipeline worker error [%s]: %s", str(payload)[:60], e)
                finally:
                    queue.task_done()

        # Start worker pool
        workers = [
            asyncio.create_task(_pipeline_worker())
            for _ in range(worker_concurrency)
        ]

        # Start CoverageMonitor (background, non-blocking)
        monitor = CoverageMonitor(
            accumulator=accumulator,
            task_queue=queue,
            coverage_checker=self._check_evidence_coverage,
            check_interval=settings.pipeline.coverage_check_interval,
            max_checks=settings.pipeline.max_coverage_checks,
        )
        monitor_task = asyncio.create_task(monitor.run(done_event))

        # Wait for all tasks to drain
        await queue.join()
        self._timer.lap("流水线检索")

        # Signal shutdown — give monitor time to see the event
        done_event.set()
        await asyncio.sleep(2.0)

        # Final drain: monitor may have injected tasks before seeing done_event
        await queue.join()

        # Send sentinels to unblock workers
        for _ in range(worker_concurrency):
            await queue.put((None, None))

        # Wait for all workers and monitor to finish
        await asyncio.gather(*workers, monitor_task, return_exceptions=True)

        # ── Assemble accumulated knowledge (same structured format as old loop) ──
        snapshot = await accumulator.get_snapshot()
        cumulative_raw_evidence = ""
        iteration = 1

        trial_entries = [e for e in snapshot if e["meta_type"] == "trial"]
        pico_entries = [e for e in snapshot if e["meta_type"] == "pico"]
        comorb_entries = [e for e in snapshot if e["meta_type"] == "comorb"]
        flat_entries = [e for e in snapshot if e["meta_type"] == "flat"]

        chunk_knowledge = "\n\n### 流水线检索分析：\n"

        for e in trial_entries:
            chunk_knowledge += f"\n#### 🎯 {e['meta_label']}\n{e['synthesis']}\n"
        if pico_entries:
            chunk_knowledge += "\n#### 🧬 PICO 问题查证\n"
            for e in pico_entries:
                chunk_knowledge += f"\n**检索方向**: {e['query']}\n{e['synthesis']}"
        if comorb_entries:
            chunk_knowledge += "\n#### 🏥 合并症安全评估\n"
            for e in comorb_entries:
                chunk_knowledge += f"\n**检索方向**: {e['query']}\n{e['synthesis']}"
        if flat_entries:
            for e in flat_entries:
                chunk_knowledge += f"\n**检索问题**: {e['query']}\n{e['synthesis']}\n"

        cumulative_raw_evidence = chunk_knowledge
        current_knowledge = cumulative_raw_evidence

        # ── DEBUG: verify accumulator → assembly pipeline ──
        logger.info(
            "[DEBUG] 快照条目数=%d | trial=%d pico=%d comorb=%d flat=%d | cumulative_raw_evidence 长度=%d",
            len(snapshot), len(trial_entries), len(pico_entries),
            len(comorb_entries), len(flat_entries), len(cumulative_raw_evidence),
        )
        trial_preview = cumulative_raw_evidence.find("#### 🎯")
        pico_preview = cumulative_raw_evidence.find("#### 🧬")
        logger.info(
            "[DEBUG] cumulative_raw_evidence 中 '#### 🎯' 首次出现位置=%d, '#### 🧬' 首次出现位置=%d",
            trial_preview, pico_preview,
        )
        if trial_preview >= 0:
            logger.info(
                "[DEBUG] 🎯 段附近 200 字符:\n%s",
                cumulative_raw_evidence[trial_preview:trial_preview+200],
            )

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
            if self.progress_reporter:
                self.progress_reporter.report("generating", "正在生成MDT报告(含多Agent并行+审查循环)...")
            if self.cancel_event and self.cancel_event.is_set():
                logger.info("收到任务取消信号，跳过报告生成。")
                return {"findings": findings, "iterations": iteration,
                        "questions": self.questions_by_iteration,
                        "current_knowledge": current_knowledge,
                        "final_report": ""}
            try:
                final_report_tuple = await self._generate_detailed_report(
                    cumulative_raw_evidence, findings, query, iteration, prognosis_results,
                    context_bus=context_bus,
                )
                if isinstance(final_report_tuple, tuple):
                    final_report = final_report_tuple[1]
                else:
                    final_report = str(final_report_tuple)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Failed to generate detailed report: {e}")
                # Try ref_pool first, fall back to raw-evidence PMID scan
                fallback_refs = "\n==================================================\n"
                if self.ref_pool.pool:
                    for i, ref in enumerate(self.ref_pool.pool, self.ref_pool.base_idx + 1):
                        title = ref.title if ref.title else "Source"
                        fallback_refs += f"[{i}] {self.ref_pool.display_label(i)}\n    Title: {title}\n----------\n"
                else:
                    fallback_refs = self._build_fallback_reference_list(
                        current_knowledge + "\n" + getattr(self, '_full_raw_evidence', '')
                    )
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
            except asyncio.CancelledError:
                raise
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

    async def cleanup(self):
        """释放异步资源：取消未完成任务、清理连接池。"""
        logger.info("开始清理 AdvancedSearchSystem 资源...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            from .concurrency.connection_pool import close_shared_http_client
            await close_shared_http_client()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        logger.info("AdvancedSearchSystem 资源清理完成。")
