import ast
import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional

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

    async def execute(self, query: str, max_rounds: int = MAX_ROUNDS) -> str:
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
            Synthesized analysis text with [^^n] citations, or empty string.
        """
        logger.info(f"[Task Started] {query}")
        all_results = []
        current_query = query

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
                    return ""

                # Step 2: Register refs from all accumulated results
                ref_map = self._register_refs(all_results)

                # Step 3: Synthesize all evidence accumulated so far
                synthesis = await self._synthesize(query, all_results, ref_map)

                # Step 3b: Strip any [^^n] citations that don't exist in ref_map
                synthesis = self._validate_citations(synthesis, ref_map)

                # Step 4: On last round, return (regardless of sufficiency)
                if round_idx >= max_rounds - 1:
                    return synthesis or ""

                # Step 5: Check if synthesis sufficiently answers the original query
                verdict = await self._check_sufficiency(query, synthesis)
                if verdict.get("sufficient", False):
                    logger.info(f"  -> 检索结果已充分，无需补充")
                    return synthesis

                reason = verdict.get("reason", "").strip()
                refined = verdict.get("refined_query", "").strip()
                logger.info(f"  -> 证据不充分。原因: {reason[:120]}")
                if not refined:
                    logger.info(f"  -> 无补充检索词，停止。")
                    return synthesis

                logger.info(f"  -> 补充检索: {refined[:80]}...")
                current_query = refined

            # If ErrorBoundary caught an error, return whatever we have from previous rounds
            if round_boundary.error:
                logger.warning(
                    f"[ErrorBoundary:round_{round_idx+1}] {query[:60]} — "
                    f"轮次失败: {round_boundary.error}"
                )
                if all_results:
                    ref_map = self._register_refs(all_results)
                    try:
                        return await self._synthesize(query, all_results, ref_map)
                    except Exception:
                        return ""
                return ""

        return ""  # unreachable

    # -----------------------------------------------------------------
    # Trial-level ReAct: unified loop across multiple sub-queries
    # -----------------------------------------------------------------

    async def execute_trial(self, trial_name: str, sub_queries: list[str], max_rounds: int = MAX_ROUNDS) -> str:
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
            Unified synthesis covering all trial dimensions, or empty string.
        """
        if not sub_queries:
            return ""

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
                return ""

            # Step 3: Register refs
            ref_map = self._register_refs(all_results)

            # Step 4: Synthesize all evidence accumulated across all sub-queries so far
            combined_ctx = f"{trial_name} {' '.join(sub_queries)}"
            synthesis = await self._synthesize(combined_ctx, all_results, ref_map)
            synthesis = self._validate_citations(synthesis, ref_map)

            # Step 5: On last round, return
            if round_idx >= max_rounds - 1:
                return synthesis or ""

            # Step 6: Trial-level sufficiency check (across ALL dimensions)
            verdict = await self._check_trial_sufficiency(trial_name, sub_queries, synthesis)
            if verdict.get("sufficient", False):
                logger.info(f"  [{trial_name}] 所有维度均已覆盖，无需补充")
                return synthesis

            reason = verdict.get("reason", "").strip()
            logger.info(f"  [{trial_name}] 维度不完整: {reason[:120]}")

            # Step 7: Determine next query
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
                    return synthesis

        return ""

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

            # Extract URL
            url = ""
            pmid_match = (
                re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", block, re.IGNORECASE)
                or re.search(
                    r"""["']?(?:PMID|uid|id)["']?\s*[:=]\s*["']?(\d{7,9})["']?""",
                    block, re.IGNORECASE,
                )
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

            # Extract title
            title = "Unknown Title"
            title_match = (
                re.search(r"^(?:Article )?Title:\s*([^\n]+)", block, re.IGNORECASE | re.MULTILINE)
                or re.search(r'\bTitle:\s*([^\n]+)', block, re.IGNORECASE)
                or re.search(r'"title"\s*:\s*"([^"]+)"', block, re.IGNORECASE)
                or re.search(r'"BriefTitle"\s*:\s*"([^"]+)"', block, re.IGNORECASE)
            )
            if title_match:
                title = title_match.group(1).strip()

            if len(title) >= 15 or "FDA" in title:
                yield url, title

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
            return await self._ensure_chinese(raw, query)
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
        """Remove ##### entries that lack extracted data points (• bullets).

        An entry consisting of only a header and 1-2 descriptive sentences,
        with no • data-point bullets, is noise — it carries zero actionable
        evidence for downstream agents.
        """
        if not text:
            return text

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

            # ##### literature entries — keep only if they have data bullets
            if stripped.startswith('##### '):
                has_bullet = bool(re.search(r'^• ', stripped, re.MULTILINE))
                if has_bullet:
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
