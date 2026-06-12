import ast
import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from ..prompts import prompt_manager

from ..concurrency.task_manager import ErrorBoundary
from ..search_system_support import safe_json_from_text
from ..tool_executor import ToolExecutor
from ..tool_selector import ToolSelector
from ..utilities.search_utilities import (
    ensure_chinese_output,
    english_alpha_ratio,
    invoke_with_timeout_and_retry,
    remove_think_tags,
)

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3  # default rounds; caller can override per-query via execute(max_rounds=N)


class ReActSearchAgent:
    """
    Per-query ReAct search agent with built-in evidence synthesis.

    For each research question:
      1. Tool selection & execution (initial PubMed/CT.gov/FDA search)
      2. LLM review of results — decide if refinement is needed
      3. Optional second pass with refined query
      4. Register discovered sources in ReferencePool ([^^n] IDs)
      5. Synthesize raw results into a compact evidence summary with [^^n] citations

    The caller receives a ready-to-use analysis string per query, eliminating the
    need for downstream deduplication, truncation, or secondary synthesis steps.

    ToolSelector and ToolExecutor are cached per-instance for efficiency.
    """

    def __init__(
        self,
        fast_model,
        tool_planning_model,
        reasoning_model,
        mcp_tool_client,
        chosen_tools: list,
        error_log_path: str,
        llm_semaphore: asyncio.Semaphore,
        api_semaphore: asyncio.Semaphore,
        ref_pool=None,  # ReferencePool instance for [^^n] registration
    ):
        self.fast_model = fast_model
        self.tool_planning_model = tool_planning_model
        self.reasoning_model = reasoning_model
        self.mcp_tool_client = mcp_tool_client
        self.chosen_tools = chosen_tools
        self.error_log_path = error_log_path
        self.llm_semaphore = llm_semaphore
        self.api_semaphore = api_semaphore
        self.ref_pool = ref_pool
        self._selector: Optional[ToolSelector] = None
        self._executor: Optional[ToolExecutor] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def execute(self, query: str, max_rounds: int = MAX_ROUNDS) -> dict:
        """
        Full ReAct cycle with synthesis-gated refinement.

        Round 1: search → synthesize → check sufficiency against original query.
        Additional rounds (up to max_rounds-1): refine query → search → re-check.

        On round 1 with zero results: automatically generates a relaxed query
        and retries before giving up.

        Args:
            query: PubMed search query string.
            max_rounds: Maximum ReAct iterations (default 3; pass 8 for core trials).

        Returns:
            dict with:
              - synthesis (str): analysis text with [^^n] citations
              - sufficient (bool): whether evidence fully answers the query
              - follow_up_queries (list[str]): refined queries to inject into
                the async pipeline when max_rounds is exhausted
        """
        logger.info(f"[Task Started] {query}")
        all_results = []
        current_query = query
        last_refined_query = ""

        self._current_query = query
        self._current_query_type = "flat"

        for round_idx in range(max_rounds):
            async with ErrorBoundary(f"round_{round_idx+1}") as round_boundary:
                # Step 1: Search
                round_results = await self._execute_single_round(
                    current_query, current_query
                )
                if round_results:
                    all_results.extend(round_results)
                    self._log_raw_output(current_query, round_results)

                # --- Query Relaxation: if round 1 returned nothing, try a broader query ---
                if not all_results and round_idx == 0:
                    relaxed = await self._generate_relaxed_query(query)
                    if relaxed and relaxed != query:
                        logger.info(f"  -> 原始检索无结果，尝试宽松检索: {relaxed}")
                        self._was_relaxed = True
                        self._relaxed_query = relaxed
                        current_query = relaxed
                        round_boundary.error = None  # not a failure, just relaxation
                        continue
                    return {"synthesis": "", "sufficient": False, "follow_up_queries": []}

                # Step 2: Register refs from all accumulated results
                ref_map = self._register_refs(all_results)

                # Step 3: Synthesize all evidence accumulated so far
                synthesis = await self._synthesize(query, all_results, ref_map)

                # Step 3b: Strip any [^^n] citations that don't exist in ref_map
                synthesis = self._validate_citations(synthesis, ref_map)

                # Step 3c: Adversarial self-check — DISABLED (pending format alignment)
                # synthesis = await self._adversarial_self_check(synthesis, all_results, ref_map)

                # Step 4: Check sufficiency (always, even on last round —
                #         so the pipeline gets follow_up_queries when needed)
                verdict = await self._check_sufficiency(query, synthesis)
                if verdict.get("sufficient", False):
                    logger.info(f"  -> 检索结果已充分，无需补充")
                    return {"synthesis": synthesis or "", "sufficient": True, "follow_up_queries": []}

                reason = verdict.get("reason", "").strip()
                refined = verdict.get("refined_query", "").strip()

                # Step 5: Not sufficient — either continue or return follow-ups
                if round_idx >= max_rounds - 1:
                    # Dead-end detection: if no quantitative data after all rounds,
                    # there is genuinely no evidence — don't keep generating follow-ups
                    scan = self._scan_quantitative(synthesis)
                    if scan["pct"] == 0 and scan["hr"] == 0 and scan["pval"] == 0 \
                            and scan["mol"] == 0 and scan["tx_effect"] == 0:
                        logger.info(
                            "  -> 所有轮次均无定量数据，判定为无可用证据，停止跟进"
                        )
                        return {
                            "synthesis": synthesis or "",
                            "sufficient": True,
                            "follow_up_queries": [],
                        }
                    follow_ups = [refined] if refined else []
                    logger.info(
                        f"  -> 最后一轮仍未充分，返回 %d 个跟进查询",
                        len(follow_ups),
                    )
                    return {"synthesis": synthesis or "", "sufficient": False, "follow_up_queries": follow_ups}

                logger.info(f"  -> 证据不充分。原因: {reason[:120]}")
                if not refined:
                    logger.info(f"  -> 无补充检索词，停止。")
                    return {"synthesis": synthesis, "sufficient": False, "follow_up_queries": []}

                logger.info(f"  -> 补充检索: {refined[:80]}...")
                current_query = refined
                last_refined_query = refined

            # If ErrorBoundary caught an error, return whatever we have from previous rounds
            if round_boundary.error:
                logger.warning(
                    f"[ErrorBoundary:round_{round_idx+1}] {query[:60]} — "
                    f"轮次失败: {round_boundary.error}"
                )
                if all_results:
                    ref_map = self._register_refs(all_results)
                    try:
                        synthesis = await self._synthesize(query, all_results, ref_map)
                        # Adversarial check in recovery path — DISABLED
                        # try:
                        #     synthesis = await self._adversarial_self_check(
                        #         synthesis, all_results, ref_map
                        #     )
                        # except Exception:
                        #     pass
                        return {"synthesis": synthesis, "sufficient": False, "follow_up_queries": []}
                    except Exception:
                        return {"synthesis": "", "sufficient": False, "follow_up_queries": []}
                return {"synthesis": "", "sufficient": False, "follow_up_queries": []}

        return {"synthesis": "", "sufficient": False, "follow_up_queries": []}

    # -----------------------------------------------------------------
    # Trial-level ReAct: unified loop across multiple sub-queries
    # -----------------------------------------------------------------

    async def execute_trial(self, trial_name: str, sub_queries: list[str], max_rounds: int = MAX_ROUNDS) -> dict:
        """
        Trial-level ReAct loop. Unlike execute() which handles one isolated query,
        this method takes multiple sub_queries targeting different dimensions of
        the same clinical trial (e.g. survival + molecular + toxicity).

        Sub-queries are explored sequentially within a SINGLE ReAct loop, so:
          - Evidence accumulates across rounds (not fragmented per sub-query)
          - Sufficiency check evaluates ALL dimensions globally, not per sub-query
          - If dimension A (survival) is covered but B (molecular) is not, the
            loop continues with sub_queries[1] to fill the gap
          - The final synthesis is a unified trial-level evidence summary

        Returns:
            dict with synthesis, sufficient, and follow_up_queries (see execute()).
        """
        if not sub_queries:
            return {"synthesis": "", "sufficient": True, "follow_up_queries": []}

        logger.info(f"[Trial: {trial_name}] 启动多维度 ReAct 循环（{len(sub_queries)} 个子检索词）")
        all_results = []
        sq_index = 0
        current_query = sub_queries[0]

        self._current_query = sub_queries[0]
        self._current_query_type = "trial"

        for round_idx in range(max_rounds):
            # Step 1: Search
            round_results = await self._execute_single_round(
                current_query, f"{trial_name} round {round_idx+1}"
            )
            if round_results:
                all_results.extend(round_results)
                self._log_raw_output(current_query, round_results)

            # Step 2: Relaxation on round 1 if empty
            if not all_results and round_idx == 0:
                relaxed = await self._generate_relaxed_query(current_query)
                if relaxed and relaxed != current_query:
                    logger.info(f"  [{trial_name}] 原始检索无结果，尝试宽松检索: {relaxed}")
                    self._was_relaxed = True
                    self._relaxed_query = relaxed
                    current_query = relaxed
                    continue
                return {"synthesis": "", "sufficient": False, "follow_up_queries": []}

            # Step 3: Register refs
            ref_map = self._register_refs(all_results)

            # Step 4: Synthesize all evidence accumulated across all sub-queries so far
            combined_ctx = f"{trial_name} {' '.join(sub_queries)}"
            synthesis = await self._synthesize(combined_ctx, all_results, ref_map)
            synthesis = self._validate_citations(synthesis, ref_map)
            # Adversarial self-check DISABLED (pending format alignment)
            # synthesis = await self._adversarial_self_check(synthesis, all_results, ref_map)

            # Step 5: Trial-level sufficiency check (ALWAYS — even on last round)
            verdict = await self._check_trial_sufficiency(trial_name, sub_queries, synthesis)
            if verdict.get("sufficient", False):
                logger.info(f"  [{trial_name}] 所有维度均已覆盖，无需补充")
                return {"synthesis": synthesis or "", "sufficient": True, "follow_up_queries": []}

            reason = verdict.get("reason", "").strip()
            logger.info(f"  [{trial_name}] 维度不完整: {reason[:120]}")

            # Step 6: On last round, return with follow-ups for pipeline
            if round_idx >= max_rounds - 1:
                # Dead-end detection: no quantitative data → genuinely no evidence
                scan = self._scan_quantitative(synthesis)
                if scan["pct"] == 0 and scan["hr"] == 0 and scan["pval"] == 0 \
                        and scan["mol"] == 0 and scan["tx_effect"] == 0:
                    logger.info(
                        "  [%s] 所有轮次均无定量数据，判定为无可用证据，停止跟进",
                        trial_name,
                    )
                    return {
                        "synthesis": synthesis or "",
                        "sufficient": True,
                        "follow_up_queries": [],
                    }
                follow_ups = []
                # Unused sub-queries become pipeline follow-ups
                for i in range(sq_index + 1, len(sub_queries)):
                    follow_ups.append(sub_queries[i])
                # Plus any LLM refined_query
                refined = verdict.get("refined_query", "").strip()
                if refined:
                    if trial_name.lower() not in refined.lower():
                        refined = f"{trial_name} AND {refined}"
                    follow_ups.append(refined)
                logger.info(
                    f"  [{trial_name}] 最后一轮仍未充分，返回 %d 个跟进查询",
                    len(follow_ups),
                )
                return {"synthesis": synthesis or "", "sufficient": False, "follow_up_queries": follow_ups}

            # Step 7: Determine next query for this loop
            # Priority: unused sub_query → LLM refined_query → stop
            next_sq_index = sq_index + 1
            if next_sq_index < len(sub_queries):
                sq_index = next_sq_index
                current_query = sub_queries[sq_index]
                self._current_query = sub_queries[sq_index]
                self._current_query_type = "trial"
                logger.info(
                    f"  [{trial_name}] 切换到下一子检索词 [{sq_index+1}/{len(sub_queries)}]: "
                    f"{current_query[:80]}..."
                )
            else:
                refined = verdict.get("refined_query", "").strip()
                if refined:
                    # Safety net: force-prepend trial name if LLM dropped it
                    if trial_name.lower() not in refined.lower():
                        refined = f"{trial_name} AND {refined}"
                        logger.info(
                            f"  [{trial_name}] LLM 补充检索词丢失试验名，已自动补回: "
                            f"{refined[:80]}..."
                        )
                    else:
                        logger.info(
                            f"  [{trial_name}] 子检索词用尽，使用 LLM 补充检索: "
                            f"{refined[:80]}..."
                        )
                    current_query = refined
                else:
                    logger.info(f"  [{trial_name}] 无补充检索方向，停止。")
                    return {"synthesis": synthesis, "sufficient": False, "follow_up_queries": []}

        return {"synthesis": "", "sufficient": False, "follow_up_queries": []}

    # -----------------------------------------------------------------
    # Synthesis-gated sufficiency check (replaces old raw-summary-based refine)
    # -----------------------------------------------------------------

    async def _check_sufficiency(self, query: str, synthesis: str) -> dict:
        """
        Check if the accumulated synthesis sufficiently answers the original query.

        Uses the same regex fact-check armor as _check_trial_sufficiency:
        pre-scan for quantitative patterns, report them to the LLM as settled fact,
        then let the LLM evaluate the full synthesis with context.
        """
        scan = self._scan_quantitative(synthesis)

        fact_parts = []
        if scan["pct"] > 0:
            fact_parts.append(f"{scan['pct']} survival percentages")
        if scan["hr"] > 0:
            fact_parts.append(f"{scan['hr']} HR values")
        if scan["pval"] > 0:
            fact_parts.append(f"{scan['pval']} P values")
        if scan["mol"] > 0:
            fact_parts.append(f"{scan['mol']} molecular mentions")

        if fact_parts:
            fact_line = (
                "Regex pre-scan CONFIRMED the synthesis contains: "
                + ", ".join(fact_parts)
                + ". These data points objectively exist — do NOT claim they are missing."
            )
        else:
            fact_line = "Regex pre-scan found NO quantitative data (no percentages, HR, P-values, molecular terms)."

        prompt = prompt_manager.get("react_sufficiency").format(
            query=query,
            fact_line=fact_line,
            synthesis=synthesis[:8000],
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, prompt, timeout=180.0, max_retries=2
            )
            content = remove_think_tags(resp.content).strip()
            parsed = safe_json_from_text(content)
            if isinstance(parsed, dict):
                return parsed

            logger.warning(f"充分性检查 JSON 解析失败，尝试正则回退，原始输出片段: {content[:200]}")
            suff_match = re.search(r'"sufficient"\s*:\s*(true|false)', content, re.IGNORECASE)
            if suff_match:
                sufficient = suff_match.group(1).lower() == "true"
                refined = ""
                rq_match = re.search(r'"refined_query"\s*:\s*["\'](.+)', content)
                if rq_match:
                    raw = rq_match.group(1).rstrip('"\'')
                    for delim in [', "', ',\n"', '\n}']:
                        if delim in raw:
                            raw = raw.split(delim)[0]
                    refined = raw.strip()
                logger.info(f"正则回退提取: sufficient={sufficient}, refined_query={refined[:100]}")
                return {"sufficient": sufficient, "reason": "", "refined_query": refined}

            logger.warning(f"连正则也无法提取，原始输出: {content[:200]}")
            return {"sufficient": True, "reason": "JSON解析失败，默认充分", "refined_query": ""}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"充分性检查失败: {e}")
            return {"sufficient": True, "reason": "检查失败，默认充分", "refined_query": ""}

    @staticmethod
    def _scan_quantitative(synthesis: str) -> dict:
        """Count quantitative data patterns in synthesis (for fact-check armor, not judgment)."""
        if not synthesis:
            return {"pct": 0, "hr": 0, "pval": 0, "mol": 0, "tx_effect": 0}

        # Survival percentages: "74.4%", "5-year OS 67.3%" etc.
        pct = len(re.findall(r'\d{1,3}\.\d{1,2}%', synthesis))
        # HR values: "HR 0.73", "HR=0.66", "hazard ratio 0.55 (95% CI..."
        hr = len(re.findall(
            r'(?:HR|hazard\s*ratio)\s*[=:]*\s*\d+\.?\d*',
            synthesis, re.IGNORECASE,
        ))
        # P values: "p=0.032", "p<0.001", "P = 0.044"
        pval = len(re.findall(r'[pP]\s*[=<>]\s*\d+\.?\d*', synthesis))
        # Molecular mentions
        mol = len(re.findall(
            r'\b(p53|TP53|POLE|MMR|MSI|molecular|mismatch|polymerase|abnormal|mutat)',
            synthesis, re.IGNORECASE,
        ))
        # Treatment effect comparisons: "CRT vs CT", "CRT vs RT", "RT vs 观察", etc.
        tx_effect = len(re.findall(
            r'(?:CRT|chemoradi\w+|放疗|化疗)\s*(?:vs|versus|对比|比|优于)\s*(?:CT|RT|化疗|放疗|观察)',
            synthesis, re.IGNORECASE,
        ))

        return {"pct": pct, "hr": hr, "pval": pval, "mol": mol, "tx_effect": tx_effect}

    async def _check_trial_sufficiency(self, trial_name: str, sub_queries: list[str], synthesis: str) -> dict:
        """
        Trial-aware sufficiency check.

        A regex pre-scan counts quantitative patterns and reports them to the LLM
        as a fact-check preamble. The LLM still sees the FULL synthesis text (no
        semantic loss) and makes the final judgment — but it cannot claim "no data
        exists" when the regex has already confirmed numerical patterns are present.
        """
        scan = self._scan_quantitative(synthesis)
        logger.info(
            f"  🔢 [PreScan] {trial_name}: {scan['pct']} 百分比, "
            f"{scan['hr']} HR, {scan['pval']} P值, {scan['mol']} 分子提及, "
            f"{scan['tx_effect']} 治疗效应对比"
        )

        # Build a one-line fact-check summary
        fact_parts = []
        if scan["pct"] > 0:
            fact_parts.append(f"{scan['pct']} 处生存率数值（如 74.4%、67.3%）")
        if scan["hr"] > 0:
            fact_parts.append(f"{scan['hr']} 处 HR 值（如 HR 0.73）")
        if scan["pval"] > 0:
            fact_parts.append(f"{scan['pval']} 处 P 值（如 p=0.032）")
        if scan["mol"] > 0:
            fact_parts.append(f"{scan['mol']} 处分子分型提及（如 p53/TP53/POLE/MMR/MSI）")
        if scan["tx_effect"] > 0:
            fact_parts.append(f"{scan['tx_effect']} 处治疗效应对比（如 CRT vs CT）")
        # Warn if treatment effect data is missing
        tx_warning = ""
        if scan["tx_effect"] == 0 and (scan["hr"] > 0 or scan["pct"] > 0):
            tx_warning = (
                "\n⚠️ **关键警告**：合成文本中存在 HR/百分比数据但未检测到治疗效应对比短语（如 CRT vs CT）。"
                "这些数据极可能仅为预后分层比较（分子亚型 vs 分子亚型），而非治疗方案间对比。"
                "若为 RCT 的辅助分析，治疗效应数据是强制必需的。"
            )

        if fact_parts:
            fact_line = "正则扫描已确认：合成文本中客观存在 " + "、".join(fact_parts) + "。这些数据点不容否认。" + tx_warning
        else:
            fact_line = "正则扫描未检测到任何定量数据（无百分比、HR、P值、分子分型词）。" + tx_warning

        disease_context = ""
        for sq in sub_queries:
            m = re.search(
                r'(endometrial\s+cancer|uterine\s+cancer|ovarian\s+cancer|cervical\s+cancer|breast\s+cancer|'
                r'vulvar\s+cancer|vaginal\s+cancer|endometrial\s+carcinoma)',
                sq, re.IGNORECASE,
            )
            if m:
                disease_context = m.group(1)
                break

        prompt = prompt_manager.get("react_trial_sufficiency").format(
            trial_name=trial_name,
            fact_line=fact_line,
            disease_context=disease_context if disease_context else 'endometrial cancer',
            synthesis=synthesis[:10000],
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, prompt, timeout=180.0, max_retries=2
            )
            content = remove_think_tags(resp.content).strip()
            parsed = safe_json_from_text(content)
            if isinstance(parsed, dict):
                return parsed

            logger.warning(
                f"[{trial_name}] 充分性检查 JSON 解析失败，尝试正则回退: {content[:200]}"
            )
            suff_match = re.search(r'"sufficient"\s*:\s*(true|false)', content, re.IGNORECASE)
            if suff_match:
                sufficient = suff_match.group(1).lower() == "true"
                refined = ""
                rq_match = re.search(r'"refined_query"\s*:\s*["\'](.+)', content)
                if rq_match:
                    raw = rq_match.group(1).rstrip('"\'')
                    for delim in [', "', ',\n"', '\n}']:
                        if delim in raw:
                            raw = raw.split(delim)[0]
                    refined = raw.strip()
                return {"sufficient": sufficient, "reason": "", "refined_query": refined}

            return {"sufficient": True, "reason": "JSON解析失败，默认充分", "refined_query": ""}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[{trial_name}] 充分性检查失败: {e}")
            return {"sufficient": True, "reason": "检查失败，默认充分", "refined_query": ""}

    # -----------------------------------------------------------------
    # Single round of tool selection + execution
    # -----------------------------------------------------------------

    async def _execute_single_round(self, query: str, log_desc: str) -> list:
        """One round of tool selection + execution for a given query.

        Uses ToolSelector for multi-tool routing (PubMed / ClinicalTrials / FDA),
        then enforces the original query string for search_recent_pubmed to
        prevent LLM reformulation.
        """
        executor = await self._get_executor()

        t_calls = await self._build_tool_calls(query, log_desc)
        if not t_calls:
            return []

        try:
            async with self.api_semaphore:
                t_results = await executor.run(t_calls) or []
            return t_results
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Tool execution failed for '{log_desc}': {e}")
            return []

    # -----------------------------------------------------------------
    # Tool call construction with PubMed query protection
    # -----------------------------------------------------------------

    _PUBMED_QUERY_RE = re.compile(
        r'\b(AND|OR|NOT)\b|\[dp\]|\[Title|\[MeSH|\[Publication',
        re.IGNORECASE,
    )

    async def _build_tool_calls(self, query: str, log_desc: str) -> list:
        """
        Build tool calls:
          1. Run ToolSelector for multi-tool routing (ClinicalTrials, FDA, etc.)
          2. Remove any search_recent_pubmed from the selector output (its LLM
             tends to reformulate PubMed queries, causing zero results)
          3. ALWAYS inject a fresh search_recent_pubmed with the exact query

        This guarantees PubMed is always queried with the correct query while
        preserving the ToolSelector's ability to add other tools (CT.gov, FDA).
        """
        selector = await self._get_selector()
        try:
            async with self.llm_semaphore:
                t_calls = await selector.run(query)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Tool selection failed for {log_desc}: {e}")
            t_calls = []

        # Discard any search_recent_pubmed from selector output (LLM reformulates)
        filtered = []
        for call in (t_calls or []):
            if call.get("tool_name") == "search_recent_pubmed":
                orig_q = str(call.get("tool_input", {}).get("query", ""))
                if orig_q != query:
                    logger.info(
                        f"  -> 丢弃 ToolSelector 篡改的 PubMed 查询: "
                        f"{orig_q[:60]}..."
                    )
                continue
            filtered.append(call)

        # Always inject a fresh PubMed call with the original query
        filtered.append({
            "tool_name": "search_recent_pubmed",
            "tool_input": {
                "query": query,
                "max_results": 3,
                "retmax": 3,
                "top_k": 3,
            },
        })

        # Fix single quotes in query arguments of remaining calls
        for call in filtered:
            ti = call.get("tool_input")
            if ti and "query" in ti:
                ti["query"] = str(ti["query"]).replace("'", '"')

        return filtered

    # -----------------------------------------------------------------
    # Reference registration (extracts URLs/titles from tool results)
    # -----------------------------------------------------------------

    def _register_refs(self, results: list) -> str:
        """
        Extract source URLs/titles from tool results, register in ref_pool,
        and return a formatted source map string for the synthesis prompt.

        The source map looks like:
            [^^5] Title of first article
            [^^6] Title of second article
        """
        if not self.ref_pool:
            return ""

        seen_urls: set = set()
        entries: list[tuple[int, str]] = []

        for res in results:
            for url, title in self._iter_refs(res):
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    idx = self.ref_pool.add(title=title, citation="", link=url)
                    if idx > 0:
                        entries.append((idx, title[:100]))

        # Log PMIDs being registered
        all_ref_pmids: list[str] = []
        for res in results:
            all_ref_pmids.extend(self._extract_pmids(str(res)))
        all_ref_pmids = list(dict.fromkeys(all_ref_pmids))
        logger.info(f"  📎 [RefPool] 注册 {len(entries)} 篇文献, PMID: {', '.join(all_ref_pmids) if all_ref_pmids else '(无)'}")

        if not entries:
            return ""

        lines = [f"[^^{idx}] {title}" for idx, title in entries]
        return "\n".join(lines)

    def _iter_refs(self, res):
        """
        Yield (url, title) pairs from a single tool result.

        Handles the same result formats as AdvancedSearchSystem._process_tool_result
        (dict with 'content' containing serialized list, plain string, etc.)
        without duplicating the full complexity.
        """
        # Flatten to string
        res_str = ""
        if isinstance(res, dict) and "content" in res:
            raw = res["content"]
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    res_str = "".join(item.get("text", "") for item in parsed)
            except Exception:
                res_str = str(raw)
        else:
            res_str = str(res)
        res_str = res_str.replace("\\n", "\n")

        blocks = res_str.split("\n---\n") if "\n---\n" in res_str else [res_str]
        seen_in_result: set = set()

        for block in blocks:
            if not block.strip():
                continue

            # Extract URL — use same robust patterns as _extract_pmids
            url = ""
            pmid_match = (
                re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", block, re.IGNORECASE)
                or re.search(r'PMID[:\s]*(\d{7,9})', block, re.IGNORECASE)
                or re.search(r"""["']?(?:uid|id)["']?\s*[:=]\s*["']?(\d{7,9})["']?""", block, re.IGNORECASE)
            )
            nct_match = re.search(r"(NCT\d{8})", block, re.IGNORECASE)

            if pmid_match:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"
            elif nct_match:
                url = f"https://clinicaltrials.gov/study/{nct_match.group(1)}"
            elif "openfda" in block.lower() or "brand_name" in block.lower() or "generic_name" in block.lower():
                url = "https://nctr-crs.fda.gov/fdalabel/ui/search"

            if not url or url in seen_in_result:
                continue
            seen_in_result.add(url)

            # Extract title — multiple fallback patterns
            title = ""
            for pat in [
                r"^(?:Article\s*)?Title:\s*([^\n]+)",
                r'\bTitle:\s*([^\n]+)',
                r'"title"\s*:\s*"([^"]+)"',
                r'"BriefTitle"\s*:\s*"([^"]+)"',
                r'"OfficialTitle"\s*:\s*"([^"]+)"',
                r'"AcronymTitle"\s*:\s*"([^"]+)"',
            ]:
                m = re.search(pat, block, re.IGNORECASE | re.MULTILINE)
                if m:
                    title = m.group(1).strip()
                    break

            # Fallback title: use PMID/NCT if we couldn't extract a real title
            if not title or len(title) < 5:
                if pmid_match:
                    title = f"PubMed PMID:{pmid_match.group(1)}"
                elif nct_match:
                    title = nct_match.group(1)
                elif "openfda" in block.lower():
                    title = "FDA Label"

            # Sanity check: yield if we have both url and title
            if title and len(title) >= 5:
                yield url, title

    # -----------------------------------------------------------------
    # CEBM evidence level extraction
    # -----------------------------------------------------------------

    # Regex: matches "CEBM 1b" embedded in the study type line or legacy format.
    # New format: "**研究类型：**... | CEBM 1b"
    # Legacy format: "- CEBM 证据等级：1b ..."
    _CEBM_LINE_RE = re.compile(
        r'CEBM\s+(1a|1b|2a|2b|3a|3b|4|5|NR)\b', re.IGNORECASE
    )
    # Regex: matches #### Title [^^n] headers (only within synthesized output)
    _CITE_HEADER_RE = re.compile(r'^####\s+.+?\s*\[\^\^(\d+)\]', re.MULTILINE)

    def _extract_and_store_cebm(self, synthesis: str) -> None:
        """
        Extract CEBM evidence levels from synthesis output and persist them
        into ReferencePool entries keyed by their [^^n] citation IDs.

        Parses blocks like:
            #### PORTEC-3 Adjuvant CRT vs RT [^^5]
            ...
            - CEBM 证据等级：1b III 期 RCT
            ...
        """
        if not synthesis or not self.ref_pool:
            return

        # Split synthesis into per-paper blocks based on #### headers
        blocks = re.split(r'\n(?=####\s)', synthesis)
        updated = 0

        for block in blocks:
            # Extract citation ID from header
            cite_match = self._CITE_HEADER_RE.search(block)
            if not cite_match:
                continue
            cite_id = int(cite_match.group(1))

            # Extract CEBM level from the block
            cebm_match = self._CEBM_LINE_RE.search(block)
            if not cebm_match:
                continue
            level = cebm_match.group(1).upper()  # normalize: "1b" stays, "nr" → "NR"

            if level == "NR":
                continue

            if self.ref_pool.update_cebm_level(cite_id, level):
                updated += 1
                logger.debug(
                    f"  📊 [CEBM] ref [^^{cite_id}] → {level}"
                )

        if updated:
            logger.info(f"  📊 [CEBM] 从合成结果中提取了 {updated} 条证据等级")

    # -----------------------------------------------------------------
    # Evidence synthesis
    # -----------------------------------------------------------------

    async def _synthesize(self, query: str, results: list, ref_map: str) -> str:
        """Distill raw tool results into structured evidence entries."""
        if not results:
            return ""
        # Build compact raw text from results
        raw_lines = []
        for r in results[:10]:
            try:
                # Try to parse MCP's list-of-articles format efficiently.
                # MCP returns [{'type': 'text', 'text': 'PMID: ...\nTitle: ...\nAbstract: ...'}, ...]
                # str() + truncation wastes 30-50% chars on structural syntax, so we
                # extract and concatenate the text fields directly.
                raw_content = ""
                if isinstance(r, dict):
                    c = r.get("content", "")
                    if c:
                        parsed = ast.literal_eval(c)
                        if isinstance(parsed, list):
                            texts = []
                            for item in parsed:
                                if isinstance(item, dict) and "text" in item:
                                    txt = item["text"].strip()
                                    # Keep first 8000 chars per article (PMID + Title + full Abstract + partial body)
                                    if len(txt) > 8000:
                                        txt = txt[:8000] + "...[截断]"
                                    texts.append(txt)
                            if texts:
                                raw_content = "\n\n---\n\n".join(texts)
                if not raw_content:
                    raw_content = str(r)
                    if len(raw_content) > 8000:
                        raw_content = raw_content[:8000] + "...[截断]"
            except Exception:
                raw_content = str(r)
                if len(raw_content) > 8000:
                    raw_content = raw_content[:8000] + "...[截断]"
            raw_lines.append(raw_content)
        raw_text = "\n\n".join(raw_lines)

        # Log PMIDs entering synthesis
        synth_pmids = self._extract_pmids(raw_text)
        logger.info(
            f"  🧬 [Synthesis] {query[:80]} → "
            f"{len(synth_pmids)} PMID(s): {', '.join(synth_pmids) if synth_pmids else '(无)'}"
        )

        prompt = prompt_manager.get("react_synthesize").format(
            query=query,
            ref_map=ref_map or "(无)",
            raw_text=raw_text[:25000],
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, prompt, timeout=180.0, max_retries=1
            )
            raw = remove_think_tags(resp.content).strip()
            raw = self._dedup_synthesis(raw)
            raw = self._strip_confidence_tags(raw)
            raw = self._normalize_markdown_format(raw)
            raw = await self._ensure_chinese(raw, query)
            raw = self._trim_visible_refs(raw, max_refs=3)
            self._extract_and_store_cebm(raw)
            return raw
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"证据合成失败 ({query}): {e}")
            return ""

    @staticmethod
    def _dedup_synthesis(text: str) -> str:
        """Collapse repeated sentences/lines from LLM output (repetition loop safety net)."""
        if not text:
            return text

        lines = text.split('\n')
        seen_lines: set = set()
        deduped_lines: list = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                deduped_lines.append(line)
                continue
            # Normalize to catch near-duplicates (case + whitespace collapsed)
            norm = re.sub(r'\s+', '', stripped)
            if norm in seen_lines:
                continue
            seen_lines.add(norm)
            deduped_lines.append(line)

        result = '\n'.join(deduped_lines)

        # Second pass: within-line sentence repetition (e.g. same sentence repeated 9 times in one line)
        # Pattern: detect if a sentence of 10+ chars appears 3+ times consecutively
        sentence_repeat = re.compile(r'(.{15,}?)\1{2,}')
        if sentence_repeat.search(result):
            # Find the longest repeating unit and collapse
            for match in sentence_repeat.finditer(result):
                unit = match.group(1)
                full = match.group(0)
                count = len(full) // len(unit)
                logger.warning(
                    f"[合成去重] 检测到句内重复 {count} 次, "
                    f"片段: {unit[:60]}..."
                )
            result = sentence_repeat.sub(r'\1', result)

        return result

    @staticmethod
    def _strip_confidence_tags(text: str) -> str:
        """Strip internal confidence tags from synthesis output before
        it enters the downstream report pipeline.

        Removes patterns like:
          [✅ 高置信度-原文直接提取]
          [⚠️ 中置信度-原文合理推断]
          [❓ 低置信度-跨文献综合]
          [🚫 不可验证-停止提取]
        """
        if not text:
            return text

        # Match: [emoji_symbol  label-text] — emoji + space + Chinese label
        cleaned = re.sub(
            r'\[[✅⚠️❓🚫]\s*[^\]]*?(?:高置信度|中置信度|低置信度|不可验证)[^\]]*\]',
            '', text,
        )
        # Collapse whitespace artifacts: multiple spaces, space-before-punctuation
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    @staticmethod
    def _limit_references(text: str, max_refs: int = 3) -> str:
        """Keep at most *max_refs* unique [^^n] references, dropping excess ones.

        References are kept in order of first appearance.  If the text cites
        [^^1], [^^3], [^^5], [^^2], [^^7] → only [^^1], [^^3], [^^5] survive;
        [^^2] and [^^7] are stripped.
        """
        if not text:
            return text

        import re as _re
        refs = _re.findall(r'\[\^\^(\d+)\]', text)
        if not refs:
            return text

        seen: list[str] = []
        for n in refs:
            if n not in seen:
                seen.append(n)
        if len(seen) <= max_refs:
            return text

        keep = set(seen[:max_refs])
        drop = set(seen[max_refs:])

        def _replace_dropped(m: _re.Match) -> str:
            return m.group(0) if m.group(1) in keep else ""

        text = _re.sub(r'\[\^\^(\d+)\]', _replace_dropped, text)

        # Clean up artifacts: empty bullet points, orphaned commas, double spaces
        text = _re.sub(r'^\s*[-*]\s*$\n?', '', text, flags=_re.MULTILINE)
        text = _re.sub(r',\s*,', ',', text)
        text = _re.sub(r'  +', ' ', text)
        text = _re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _trim_visible_refs(text: str, max_refs: int = 3) -> str:
        """Limit visible [^^n] citations to *max_refs* in body text.

        Strategy:
          1. Collect all unique [^^n] IDs in order of first appearance.
          2. If ≤ max_refs, return text unchanged.
          3. If > max_refs, keep only the first max_refs IDs in the visible body.
          4. Append ALL unique IDs as an HTML comment so reindex_references()
             can still find them and include them in the final reference list.

        The HTML comment is stripped by report_orchestrator after reindexing.
        """
        if not text:
            return text

        refs = re.findall(r'\[\^\^(\d+)\]', text)
        unique_refs: list[str] = list(dict.fromkeys(refs))
        if len(unique_refs) <= max_refs:
            return text

        keep = set(unique_refs[:max_refs])

        def _replace_dropped(m):
            return m.group(0) if m.group(1) in keep else ""

        text = re.sub(r'\[\^\^(\d+)\]', _replace_dropped, text)

        # Clean up artifacts from stripped citations
        text = re.sub(r'^\s*[-*]\s*$\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Append ALL unique ref IDs as hidden anchor for reindex_references()
        ref_anchor = ' '.join(f'[^^{rid}]' for rid in unique_refs)
        text = text.strip() + f'\n\n<!-- ref_anchor: {ref_anchor} -->'

        return text

    @staticmethod
    def _normalize_markdown_format(text: str) -> str:
        """Post-process LLM synthesis output to enforce consistent Markdown formatting.

        The LLM produces variable-quality formatting (mixed dash characters,
        inconsistent list indentation, etc.).  This pure-code pass normalises
        everything downstream consumers rely on.
        """
        if not text:
            return text

        # 1. Normalise dash characters: em-dash, en-dash, horizontal bar → hyphen-minus
        text = re.sub(r'[–—─]', '-', text)

        # 2. Normalise list indentation: strip only excessive leading whitespace (>4 spaces or tabs)
        #    Preserve "  - " sub-list indentation so tiered lists survive.
        text = re.sub(r'^[ \t]{5,}- ', '  - ', text, flags=re.MULTILINE)
        text = re.sub(r'^\t+- ', '  - ', text, flags=re.MULTILINE)

        # 3. Collapse repeated blank lines (max 2 consecutive newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 4. Strip trailing whitespace on each line
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

        return text.strip()

    @staticmethod
    def _remove_thin_entries(text: str) -> str:
        """Remove ##### entries that lack extracted data points.

        An entry is considered "thin" (data-less) if it has ONLY a title and
        descriptive prose, with no extractable numerical data. We check for:
          - Standard • bullet data points
          - Alternative bullet markers (-, *, etc.)
          - Key clinical-trial data patterns (HR, 95% CI, P values, N=, OS/RFS rates)

        If any of these are found, the entry carries actionable evidence and is kept.
        """
        if not text:
            return text

        # Patterns that indicate substantive extracted data (beyond plain prose)
        _BULLET_RE = re.compile(
            r'^[•◦▪‣·\-*]\s', re.MULTILINE
        )
        _DATA_INDICATOR_RE = re.compile(
            r'(?:HR|OR|RR)\s*[=:≈]\s*[\d.]+|'            # effect size (HR=0.54)
            r'95%\s*CI\s*[：:]\s*[\d.]+|'                # confidence interval
            r'[Pp]\s*[=<>≤≥]\s*0?\.\d+|'                # p-value
            r'N\s*[=:＝]\s*\d[\d,]*|'                    # sample size
            r'(?:OS|RFS|PFS|DFS|CSS|EFS)\s*(?:率|rate)?\s*(?:为|:)?\s*\d+|'  # survival %
            r'\d+% vs \d+%|'                             # comparison like "52% vs 60%"
            r'试验组|对照组|治疗组|观察组|安慰剂组'          # arm description (Chinese)
        )

        def _has_actionable_data(entry: str) -> bool:
            """Check if entry contains extractable clinical evidence."""
            if _BULLET_RE.search(entry):
                return True
            if _DATA_INDICATOR_RE.search(entry):
                return True
            return False

        # ── Pass 1: split at every ##### and #### boundary ──
        parts = re.split(r'(?=^(?:#####|####) )', text, flags=re.MULTILINE)

        kept: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue

            # #### trial-group / section headers — always keep
            if stripped.startswith('#### '):
                kept.append(part)
                continue

            # ##### literature entries — keep only if they have extractable data
            if stripped.startswith('##### '):
                if _has_actionable_data(stripped):
                    kept.append(part)
                else:
                    title_line = stripped.split('\n')[0][6:].strip()
                    logger.info(
                        "  🗑️ [去薄] 移除无数据条目: %s...",
                        title_line[:80]
                    )
                continue

            # Non-header preamble text — keep
            kept.append(part)

        result = ''.join(kept)

        # ── Pass 2: remove #### sections emptied by Pass 1 ──
        sections = re.split(r'(?=^#### )', result, flags=re.MULTILINE)
        final: list[str] = []
        for section in sections:
            stripped = section.strip()
            if not stripped:
                continue
            if stripped.startswith('#### ') and '##### ' not in stripped:
                label = stripped.split('\n')[0][5:].strip()
                logger.info(
                    "  🗑️ [去薄] 移除空试验分组: %s...",
                    label[:60]
                )
                continue
            final.append(section)

        result = ''.join(final)

        # Collapse runs of blank lines produced by removals
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result.strip()

    async def _ensure_chinese(self, text: str, query: str) -> str:
        """Delegate to shared ensure_chinese_output utility."""
        return await ensure_chinese_output(
            text, self.fast_model, label="Synthesis", logger=logger
        )

    async def _generate_relaxed_query(self, original_query: str) -> str:
        """
        Generate a relaxed/broader version of a query that returned zero results.

        Strategy: reduce AND conditions, expand synonyms, remove field restrictions.
        """
        prompt = prompt_manager.get("react_relaxed_query").format(
            original_query=original_query,
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, prompt, timeout=60.0, max_retries=1
            )
            relaxed = remove_think_tags(resp.content).strip()
            # Remove any markdown code block fences or stray backticks
            relaxed = re.sub(r'^```(?:text)?\n?', '', relaxed)
            relaxed = re.sub(r'\n?```$', '', relaxed)
            relaxed = relaxed.strip().strip('"\'')
            if relaxed and relaxed != original_query and len(relaxed) < 500:
                return relaxed
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"生成宽松检索词失败: {e}")
        return ""

    def _validate_citations(self, text: str, ref_map: str) -> str:
        """
        Remove any [^^n] citations that reference IDs not present in ref_map.
        Prevents LLM-hallucinated citations from reaching downstream agents.
        """
        if not text or not ref_map:
            return text

        valid_ids = set()
        for m in re.finditer(r"\[\^\^(\d+)\]", ref_map):
            valid_ids.add(int(m.group(1)))

        if not valid_ids:
            return re.sub(r"\[\^\^(\d+)\]", "", text)

        def _replace(m):
            return m.group(0) if int(m.group(1)) in valid_ids else ""

        return re.sub(r"\[\^\^(\d+)\]", _replace, text)

    # =================================================================
    # Adversarial self-check — per-claim refutation inside ReAct loop
    # =================================================================

    @staticmethod
    def _extract_raw_articles(results: list) -> list[dict]:
        """Extract per-article raw texts from MCP tool results.

        Uses the same parsing logic as _synthesize, but returns individual
        articles keyed by PMID for matching against synthesis blocks.

        Returns:
            list of {"pmid": str, "text": str} — one per article found
        """
        articles: list[dict] = []
        for r in (results or [])[:10]:
            try:
                if isinstance(r, dict):
                    c = r.get("content", "")
                    if c:
                        parsed = ast.literal_eval(c)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "text" in item:
                                    txt = item["text"].strip()
                                    if len(txt) > 8000:
                                        txt = txt[:8000] + "...[截断]"
                                    pmids = ReActSearchAgent._extract_pmids(txt)
                                    pmid = pmids[0] if pmids else ""
                                    articles.append({"pmid": pmid, "text": txt})
            except Exception:
                pass
        return articles

    @staticmethod
    def _parse_synthesis_blocks(synthesis: str) -> list[tuple[str, int]]:
        """Parse synthesis into per-paper blocks: (block_text, cite_id).

        Synthesis uses #### Title [^^n] format (pre-demotion) for paper entries.
        """
        if not synthesis:
            return []
        block_pattern = re.compile(
            r'(#### .+? \[(\d+)\].*?)(?=\n#### |\n#### [🎯🧬🏥]|\Z)',
            re.DOTALL,
        )
        return [(m.group(1), int(m.group(2))) for m in block_pattern.finditer(synthesis)]

    async def _adversarial_self_check(
        self, synthesis: str, results: list, ref_map: str
    ) -> str:
        """Run adversarial verification against each per-paper claim block.

        For each #### Paper Title [^^n] block in the synthesis:
          1. Resolve [^^n] → PMID via ReferencePool
          2. Find the matching raw article in the tool results
          3. Run a fast-model adversarial check: try to refute the claim
             using the original article as ground truth
          4. If refuted, patch the claim with the corrected text in-place

        Returns the corrected synthesis (or original if no refutations).

        This runs INSIDE the ReAct loop — BEFORE sufficiency check — so that
        refuted claims are corrected before the loop decides whether to
        continue searching or return.  All blocks are checked in parallel.
        """
        if not synthesis or not results or not self.ref_pool:
            return synthesis

        blocks = self._parse_synthesis_blocks(synthesis)
        if not blocks:
            return synthesis

        raw_articles = self._extract_raw_articles(results)
        if not raw_articles:
            logger.info("  [对抗性自检] 无原始文献可匹配，跳过")
            return synthesis

        # Build PMID → article text index for O(1) lookup
        pmid_to_article: dict[str, str] = {}
        for art in raw_articles:
            if art["pmid"]:
                pmid_to_article[art["pmid"]] = art["text"]

        logger.info(
            f"  🔍 [对抗性自检] 开始对 {len(blocks)} 个文献块执行对抗性验证..."
        )

        corrections: list[tuple[str, str]] = []  # [(block_text, corrected_block)]

        async def _verify_block(block_text: str, cite_id: int) -> tuple[str, str] | None:
            """Verify one block. Returns (old_block, corrected_block) or None."""
            ref = self.ref_pool.get_ref_by_idx(cite_id)
            if not ref or not ref.link:
                return None

            # Resolve citation → PMID
            ref_pmid = ""
            pmid_m = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', ref.link, re.IGNORECASE)
            if pmid_m:
                ref_pmid = pmid_m.group(1)

            if not ref_pmid or ref_pmid not in pmid_to_article:
                return None

            raw_article = pmid_to_article[ref_pmid]

            prompt = (
                "你是证据审查员。你的任务是：基于【原始文献】的内容，尝试推翻【合成块】中的每一条结论声明。\n\n"
                f"## 合成块（ReAct agent 对该文献的总结）\n{block_text[:3000]}\n\n"
                f"## 原始文献（MCP 返回的完整摘要）\n{raw_article[:6000]}\n\n"
                "## 审查维度\n"
                "逐一检查，只要有一条成立就输出 REFUTED：\n\n"
                "1. **统计误读**：合成块说\"显著有效/有统计学差异\"，但原始文献中对应 HR/OR/RR 的 95% CI 跨 1.0，或 P>0.05\n"
                "2. **数值虚构**：合成块中的具体数值（N、HR、P、百分比）在原始文献中找不到对应数据\n"
                "3. **选择性报告**：合成块只报告了有利的次要终点，但原始文献明确报告了主要终点无统计学差异\n"
                "4. **过度推广**：原始文献是 II 期/单中心/小样本(N<100)，合成块将其当作确定性证据陈述\n"
                "5. **亚组伪装主分析**：合成块作为主要结论报告的，原始文献中明确标注为 post-hoc/亚组/探索性分析\n\n"
                "## 输出格式（严格遵循）\n"
                "对每条有问题的声明，输出一行：\n"
                f"  REFUTED | [^^{cite_id}] | 被推翻的声明（原文摘录） | 应修正为（基于原始文献的正确表述）\n\n"
                "如果合成块中所有声明均站得住，只输出：\n"
                f"  STANDS | [^^{cite_id}]\n\n"
                "不要输出任何其他内容。"
            )

            try:
                resp = await invoke_with_timeout_and_retry(
                    self.fast_model, prompt, timeout=60.0, max_retries=1
                )
                raw = remove_think_tags(resp.content).strip()
            except Exception:
                return None

            if not raw or raw.startswith("STANDS"):
                return None

            # Parse refuted claims and apply corrections within this block
            corrected = block_text
            refuted_count = 0
            for line in raw.split("\n"):
                line = line.strip()
                if not line.startswith("REFUTED"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    original_claim = parts[2]
                    fix = parts[3]
                    if original_claim and fix:
                        if original_claim in corrected:
                            corrected = corrected.replace(original_claim, fix)
                            refuted_count += 1
                            logger.info(
                                f"  ⚠️ [对抗性自检] [^^{cite_id}] 修正: "
                                f"{original_claim[:60]}... → {fix[:60]}..."
                            )
                        else:
                            # Claim text not found verbatim — append caveat
                            caveat = (
                                f"\n\n⚠️ [对抗性自检] 上述分析可能存在问题，"
                                f"请核实：{fix}"
                            )
                            corrected = corrected.rstrip() + caveat
                            refuted_count += 1
                            logger.info(
                                f"  ⚠️ [对抗性自检] [^^{cite_id}] 追加警示: {fix[:80]}..."
                            )

            if refuted_count > 0:
                return (block_text, corrected)
            return None

        # Run all block checks in parallel (each block independently verified)
        check_results = await asyncio.gather(
            *[_verify_block(block, cid) for block, cid in blocks],
            return_exceptions=True,
        )

        for result in check_results:
            if isinstance(result, tuple) and len(result) == 2:
                corrections.append(result)

        if corrections:
            logger.info(
                f"  ⚠️ [对抗性自检] 发现 {len(corrections)} 个文献块存在问题，"
                f"共 {len(blocks)} 个块"
            )
            for old_block, new_block in corrections:
                if old_block in synthesis:
                    synthesis = synthesis.replace(old_block, new_block)
        else:
            logger.info(
                f"  ✅ [对抗性自检] 所有 {len(blocks)} 个文献块均通过验证"
            )

        return synthesis

    # -----------------------------------------------------------------
    # Cached accessors
    # -----------------------------------------------------------------

    async def _get_selector(self) -> ToolSelector:
        if self._selector is None:
            self._selector = ToolSelector(
                self.tool_planning_model,
                self.reasoning_model,
                self.mcp_tool_client,
                tool_info_data=None,
                embedding_api_key=None,
                embedding_cache=None,
                available_tools=self.chosen_tools,
            )
        return self._selector

    async def _get_executor(self) -> ToolExecutor:
        if self._executor is None:
            self._executor = ToolExecutor(
                self.mcp_tool_client, self.error_log_path, self.fast_model
            )
        return self._executor

    # -----------------------------------------------------------------
    # Debug output logging
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_pmids(text: str) -> list[str]:
        """Extract PMIDs from a text blob using multiple regex patterns."""
        pmids: list[str] = []
        for pat in [
            r'PMID[:\s]*(\d{7,9})',
            r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)',
            r'"uid"\s*:\s*"(\d{7,9})"',
        ]:
            pmids.extend(re.findall(pat, text, re.IGNORECASE))
        return list(dict.fromkeys(pmids))  # dedup preserving order

    def _log_raw_output(self, label: str, results: list):
        # Extract and log PMIDs
        all_pmids: list[str] = []
        for res in results:
            all_pmids.extend(self._extract_pmids(str(res)))
        all_pmids = list(dict.fromkeys(all_pmids))
        if all_pmids:
            logger.info(f"  📚 [PMID] {label} → {len(all_pmids)} 篇: {', '.join(all_pmids)}")
        else:
            logger.info(f"  ⚠️ [PMID] {label} → 0 篇（无 PMID）")

        try:
            with open("API_RAW_OUTPUT_CONCURRENT.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"\U0001f50d 检索: {label}\n")
                f.write(f"PMID: {', '.join(all_pmids) if all_pmids else '(无)'}\n")
                f.write(f"{'='*60}\n")
                for idx, res in enumerate(results):
                    f.write(f"--- 片段 {idx+1} ---\n{str(res)}\n\n")
        except Exception:
            pass

        # Structured telemetry
        try:
            from ...telemetry import QueryPerformance, record_query_performance
            record_query_performance(QueryPerformance(
                query=getattr(self, "_current_query", ""),
                query_type=getattr(self, "_current_query_type", "flat"),
                pmids_found=len(all_pmids),
                was_relaxed=getattr(self, "_was_relaxed", False),
                relaxed_query=getattr(self, "_relaxed_query", None),
                timestamp=datetime.now().isoformat(),
            ))
        except Exception:
            pass
