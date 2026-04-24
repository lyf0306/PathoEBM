import ast
import asyncio
import logging
import re
import textwrap
from typing import List, Optional

from ..search_system_support import safe_json_from_text
from ..tool_executor import ToolExecutor
from ..tool_selector import ToolSelector
from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags

logger = logging.getLogger(__name__)

MAX_ROUNDS = 2  # at most one refinement per query


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

    async def execute(self, query: str) -> str:
        """
        Full ReAct cycle with synthesis-gated refinement.

        Round 1: search → synthesize → check sufficiency against original query.
        Round 2 (if needed): refine query → search → finalize.

        Returns:
            Synthesized analysis text with [^^n] citations, or empty string.
        """
        logger.info(f"[Task Started] {query}")
        all_results = []
        current_query = query

        for round_idx in range(MAX_ROUNDS):
            # Step 1: Search
            round_results = await self._execute_single_round(
                current_query, current_query
            )
            if round_results:
                all_results.extend(round_results)
                self._log_raw_output(current_query, round_results)

            if not all_results:
                return ""

            # Step 2: Register refs from all accumulated results
            ref_map = self._register_refs(all_results)

            # Step 3: Synthesize all evidence accumulated so far
            synthesis = await self._synthesize(query, all_results, ref_map)

            # Step 3b: Strip any [^^n] citations that don't exist in ref_map
            synthesis = self._validate_citations(synthesis, ref_map)

            # Step 4: On last round, return (regardless of sufficiency)
            if round_idx >= MAX_ROUNDS - 1:
                return synthesis or ""

            # Step 5: Check if synthesis sufficiently answers the original query
            verdict = await self._check_sufficiency(query, synthesis)
            if verdict.get("sufficient", False):
                logger.info(f"  -> 检索结果已充分，无需补充")
                return synthesis

            refined = verdict.get("refined_query", "").strip()
            if not refined:
                return synthesis

            logger.info(f"  -> 证据不充分，补充检索: {refined[:80]}...")
            current_query = refined

        return ""  # unreachable

    # -----------------------------------------------------------------
    # Synthesis-gated sufficiency check (replaces old raw-summary-based refine)
    # -----------------------------------------------------------------

    async def _check_sufficiency(self, query: str, synthesis: str) -> dict:
        """
        Check if the accumulated synthesis sufficiently answers the original query.

        Returns:
            {"sufficient": bool, "reason": str, "refined_query": str}
        """
        prompt = textwrap.dedent(f"""
        你是一名循证医学检索策略专家。请评估当前已获取的证据是否足以回答检索问题。

        【原始检索问题】：{query}

        【当前已合成的证据】：
        {synthesis[:2000]}

        请判断：
        1. 是否已找到直接相关的临床试验数据？
        2. 是否包含具体的生存获益数据（HR/OS/PFS/95%CI）？
        3. 是否需要更多证据来填补缺口？

        【输出 JSON 格式】（🚨 JSON 转义铁律：refined_query 中的双引号必须写成 \\" 以保持 JSON 合法）：
        ```json
        {{
            "sufficient": false,
            "reason": "简要理由",
            "refined_query": "(\\"endometrial cancer\\" OR \\"endometrial carcinoma\\") AND \\"p53abn\\" AND 2018:2026[dp]"
        }}
        ```
        """)

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, prompt, timeout=120.0, max_retries=1
            )
            content = remove_think_tags(resp.content).strip()
            parsed = safe_json_from_text(content)
            if isinstance(parsed, dict):
                return parsed

            # Regex fallback for malformed JSON (truncated / unescaped quotes)
            logger.warning(f"充分性检查 JSON 解析失败，尝试正则回退，原始输出片段: {content[:200]}")
            suff_match = re.search(r'"sufficient"\s*:\s*(true|false)', content, re.IGNORECASE)
            if suff_match:
                sufficient = suff_match.group(1).lower() == "true"
                refined = ""
                rq_match = re.search(r'"refined_query"\s*:\s*["\'](.+)', content)
                if rq_match:
                    raw = rq_match.group(1).rstrip('"\'')
                    # Cut at natural delimiters if present (next key or end of JSON)
                    for delim in [', "', ',\n"', '\n}']:
                        if delim in raw:
                            raw = raw.split(delim)[0]
                    refined = raw.strip()
                logger.info(f"正则回退提取: sufficient={sufficient}, refined_query={refined[:100]}")
                return {"sufficient": sufficient, "reason": "", "refined_query": refined}

            logger.warning(f"连正则也无法提取，原始输出: {content[:200]}")
            return {"sufficient": True, "reason": "JSON解析失败，默认充分", "refined_query": ""}
        except Exception as e:
            logger.warning(f"充分性检查失败: {e}")
            return {"sufficient": True, "reason": "检查失败，默认充分", "refined_query": ""}

    # -----------------------------------------------------------------
    # Single round of tool selection + execution
    # -----------------------------------------------------------------

    async def _execute_single_round(self, query: str, log_desc: str) -> list:
        """One round of tool selection + execution for a given query."""
        selector = await self._get_selector()
        executor = await self._get_executor()

        try:
            async with self.llm_semaphore:
                t_calls = await selector.run(query)
        except Exception as e:
            logger.warning(f"Tool selection failed for {log_desc}: {e}")
            return []

        if not t_calls:
            return []

        for call in t_calls:
            if "tool_input" in call and "query" in call["tool_input"]:
                raw_q = str(call["tool_input"]["query"])
                call["tool_input"]["query"] = raw_q.replace("'", '"')

        try:
            async with self.api_semaphore:
                t_results = await executor.run(t_calls) or []
            return t_results
        except Exception as e:
            logger.error(f"Tool execution failed for '{log_desc}': {e}")
            return []

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
        """Distill raw tool results into structured per-trial evidence entries."""
        # Build compact raw text from results
        raw_lines = []
        for r in results[:10]:
            s = str(r)
            if len(s) > 1500:
                s = s[:1500] + "...[截断]"
            raw_lines.append(s)
        raw_text = "\n\n".join(raw_lines)

        prompt = textwrap.dedent(f"""
        你是一名循证医学分析专家。请根据以下检索结果，提取最核心的临床试验证据。

        【检索问题】：{query}

        【可用来源引用】（请使用这些 [^^n] 编号引用对应来源）：
        {ref_map or "(无)"}

        【原始检索结果】：
        {raw_text[:6000]}

        【🚨 输出要求】：
        1. **按临床试验逐条输出**，每项试验一条，最多 3 项。
        2. **保留具体统计数据**：HR、OS、PFS、95%CI、P值、试验名称/编号等。
        3. **引用格式铁律**：使用 [^^n] 引用【可用来源引用】中的编号。**绝对禁止**使用不存在的编号。
        4. 如果没有任何有价值的临床证据，只输出：无相关临床证据。

        【严格输出格式】（每项试验一条）：
        #### [试验名称/编号] 是一项 [试验类型] [^^n]
        - **核心数据**：[OS/DFS/PFS 具体百分比与HR、95%CI、P值]
        - **亚组分析**：[分子分型等差异化数据，如有]
        """)

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, prompt, timeout=120.0, max_retries=1
            )
            return remove_think_tags(resp.content).strip()
        except Exception as e:
            logger.warning(f"证据合成失败 ({query}): {e}")
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

    def _log_raw_output(self, label: str, results: list):
        try:
            with open("API_RAW_OUTPUT_CONCURRENT.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"\U0001f50d 检索: {label}\n")
                f.write(f"{'='*60}\n")
                for idx, res in enumerate(results):
                    f.write(f"--- 片段 {idx+1} ---\n{str(res)}\n\n")
        except Exception:
            pass
