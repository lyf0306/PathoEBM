import logging
import re
import textwrap
from typing import List

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..tool_selector import ToolSelector
from ..tool_executor import ToolExecutor
from ..models.reference_pool import ReferencePool

logger = logging.getLogger(__name__)


class PrognosisRetrieval:
    """
    Plan-and-Execute prognosis retrieval.
    Iteratively: LLM plans next search → execute → observe → decide if more needed.
    """
    def __init__(self, tool_planning_model, fast_model, mcp_tool_client,
                 error_log_path: str, structured_task: dict,
                 ref_pool: ReferencePool, all_links_of_system: List[str]):
        self.tool_planning_model = tool_planning_model
        self.fast_model = fast_model
        self.mcp_tool_client = mcp_tool_client
        self.error_log_path = error_log_path
        self.structured_task = structured_task
        self.ref_pool = ref_pool
        self.all_links_of_system = all_links_of_system

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    async def run(self) -> dict:
        """
        Plan-and-Execute loop for prognosis data retrieval.

        The LLM iteratively decides what to search next:
          Step 1 (bias: NCDB/SEER broad survival) → observe
          Step 2+ (drill: molecular, subgroups)   → observe
          → decide if more needed or COMPLETE

        Returns:
            dict with:
              - "population":   str (NCDB/SEER tagged results)
              - "molecular":    str (molecular/subgroup results)
              - "raw_combined": str (all results concatenated)
        """
        logger.info("启动 Plan-and-Execute 预后数据检索...")

        base_info = self._get_patient_info()
        accumulated = []     # [{"goal": str, "query": str, "evidence": str}]
        all_queries = []     # dedup
        max_steps = 4

        for step in range(max_steps):
            next_query, search_goal = await self._plan_next(
                base_info, accumulated, all_queries, step, max_steps
            )

            if not next_query or next_query.strip().upper() == "COMPLETE":
                logger.info(f"Plan-and-Execute 判定完成，共检索 {step} 步")
                break

            # Execute the planned search
            logger.info(f"  步骤 {step+1}/{max_steps}: {search_goal}")
            logger.info(f"  检索词: {next_query}")

            _, t_results = await self._execute_search(next_query, tag=f"plan_step_{step}")
            parsed = self._parse_tagged_results(t_results, f"步骤{step+1}")

            accumulated.append({
                "goal": search_goal,
                "query": next_query,
                "evidence": parsed,
            })
            all_queries.append(next_query)

        # Split for backward compat (population = NCDB/SEER, molecular = rest)
        pop_evidence, mol_evidence = self._split_by_type(accumulated)
        combined = self._format_combined(accumulated)

        logger.info(
            f"Plan-and-Execute 预后检索完成: "
            f"人口学 {len(pop_evidence)} 字符, 分子 {len(mol_evidence)} 字符"
        )
        return {
            "population":  pop_evidence,
            "molecular":   mol_evidence,
            "raw_combined": combined,
        }

    # -----------------------------------------------------------------
    # Plan: LLM decides what to search next
    # -----------------------------------------------------------------
    async def _plan_next(self, base_info: str, accumulated: list,
                         all_queries: list, step: int, max_steps: int):
        """LLM decides the next search goal and query, or returns COMPLETE."""

        # Build accumulated evidence summary
        if accumulated:
            summary_lines = []
            for i, a in enumerate(accumulated):
                goal = a.get("goal", "")
                evidence_preview = a.get("evidence", "")[:200].replace("\n", " ")
                summary_lines.append(f"步骤{i+1} [{goal}]: {evidence_preview}...")
            acc_summary = "\n".join(summary_lines)
        else:
            acc_summary = "（尚无已收集的证据）"

        queries_str = "\n".join(f"- {q}" for q in all_queries) or "（无）"

        plan_prompt = textwrap.dedent(f"""
        你是一位肿瘤预后数据检索规划专家。
        任务：根据患者信息和当前已收集的证据，决定下一步搜索策略。

        【患者信息】：
        {base_info}

        【当前已收集证据】：
        {acc_summary}

        【已执行过的查询】（请勿重复）：
        {queries_str}

        【进度】：第 {step+1} 步 / 共 {max_steps} 步

        【🚨 策略规则】：
        1. **第1步必须检索大样本生存率**：使用 NCDB/SEER 数据库，必须包含 (NCDB OR SEER OR "National Cancer Database")，分期必须降维为大分期，末尾加 AND survival AND 2018:2026[dp]。
        2. **后续步骤深入分子/亚组**：围绕分子分型（p53abn等）与预后的关联，不加 NCDB/SEER 限定，末尾加 AND (prognos* OR survival OR outcome*) AND 2018:2026[dp]。
        3. **避免重复**：不要生成和已执行过的查询相似的检索词。
        4. **判定完成**：如果已收集到足够的生存率基线 + 分子分型预后数据，只输出：COMPLETE

        请严格按照以下格式输出（不含多余内容）：
        目标：[一句话说明本次搜索目的]
        查询：[具体的 PubMed 检索词]
        """)

        try:
            resp = await invoke_with_timeout_and_retry(
                self.fast_model, plan_prompt, timeout=180.0, max_retries=2
            )
            content = remove_think_tags(resp.content).strip()

            if content.upper() == "COMPLETE":
                return "COMPLETE", ""

            # Parse "目标：" and "查询：" lines
            goal_match = re.search(r"目标[：:]\s*(.+)", content)
            query_match = re.search(r"查询[：:]\s*(.+)", content)

            goal = goal_match.group(1).strip() if goal_match else f"预后检索步骤{step+1}"
            query = query_match.group(1).strip() if query_match else content

            # Clean wrapping quotes
            if query.startswith('"') and query.endswith('"') and query.count('"') == 2:
                query = query.strip('"')

            return query, goal

        except Exception as e:
            logger.warning(f"Plan-and-Execute 规划步骤异常: {e}")
            return "COMPLETE", ""

    # -----------------------------------------------------------------
    # Execute: run a PubMed search via MCP tool
    # -----------------------------------------------------------------
    async def _execute_search(self, prog_query: str, tag: str = ""):
        local_selector = ToolSelector(
            self.tool_planning_model, self.tool_planning_model,
            self.mcp_tool_client, available_tools=["search_recent_pubmed"]
        )
        local_executor = ToolExecutor(
            self.mcp_tool_client, self.error_log_path, self.fast_model
        )

        try:
            force_command = f"Search for survival rates using this EXACT query: {prog_query}"
            t_calls = await local_selector.run(force_command)

            if t_calls:
                for call in t_calls:
                    if 'tool_input' in call:
                        call['tool_input']['max_results'] = 5
                        call['tool_input']['retmax'] = 5
                        call['tool_input']['top_k'] = 5

            t_results = await local_executor.run(t_calls) or []
            return tag, t_results
        except Exception as e:
            logger.error(f"预后检索失败: {e}")
            return tag, []

    # -----------------------------------------------------------------
    # Observe: parse and filter raw results
    # -----------------------------------------------------------------
    def _parse_tagged_results(self, t_results: list, tag: str) -> str:
        RELEVANCE_KEYWORDS = re.compile(
            r'(endometri|uterine|endometrial cancer|endometrial carcinoma|uterine cancer|'
            r'子宫|内膜癌|子宫内膜癌| corpus |uteri|FIGO|PORTEC|GOG|NRG|RUBY|KEYNOTE)',
            re.IGNORECASE
        )

        prog_evidence = ""
        count = 0

        for res in t_results:
            res_str = self._extract_content_string(res)
            blocks = res_str.split("\n---\n") if "\n---\n" in res_str else [res_str]

            for block in blocks:
                if not block.strip() or "Unknown Title" in block:
                    continue
                if not RELEVANCE_KEYWORDS.search(block):
                    continue
                url = self._extract_url(block)
                if not url:
                    continue
                title = self._extract_title(block)
                if url not in self.all_links_of_system:
                    idx = self.ref_pool.add(title=title, citation="", link=url)
                    count += 1
                    prog_evidence += f"\n#### [^^{idx}] {title}\n{block[:3000]}\n"
                    self.all_links_of_system.append(url)
                    if count >= 3:
                        break
            if count >= 3:
                break

        if not prog_evidence.strip():
            return f"未检索到相关的{tag}专属预后文献。"

        logger.info(f"成功抓取 {count} 篇{tag}专属预后文献，已与主库物理隔离。")
        return prog_evidence

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _get_patient_info(self) -> str:
        oncology_core = self.structured_task.get("oncology_profile", {})
        if isinstance(oncology_core, dict):
            diag = oncology_core.get("diagnosis_and_stage", "")
            patho = oncology_core.get("pathology_and_molecular", "")
            return f"{diag} {patho}".strip()
        return str(oncology_core)

    def _split_by_type(self, accumulated: list):
        """Split evidence into population (NCDB/SEER) vs molecular."""
        pop_parts = []
        mol_parts = []
        for a in accumulated:
            query = a.get("query", "")
            evidence = a.get("evidence", "")
            # If query mentions NCDB/SEER → population bucket
            if re.search(r'\b(NCDB|SEER|National Cancer Database)\b', query, re.IGNORECASE):
                pop_parts.append(evidence)
            else:
                mol_parts.append(evidence)
        return "\n".join(pop_parts).strip(), "\n".join(mol_parts).strip()

    def _format_combined(self, accumulated: list) -> str:
        parts = []
        for i, a in enumerate(accumulated):
            goal = a.get("goal", "")
            evidence = a.get("evidence", "")
            if evidence:
                parts.append(f"### 检索步骤 {i+1}: {goal}\n{evidence}")
        return "\n\n".join(parts).strip()

    def _extract_content_string(self, res) -> str:
        if isinstance(res, dict) and "content" in res:
            raw_content = res["content"]
            try:
                import ast
                parsed_list = ast.literal_eval(raw_content)
                if isinstance(parsed_list, list):
                    return "".join([item.get("text", "") for item in parsed_list])
            except Exception:
                return str(raw_content)
        return str(res)

    def _extract_url(self, block: str) -> str:
        pmid_match = (
            re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', block, re.IGNORECASE)
            or re.search(r'["\']?(?:PMID|uid|id)["\']?\s*[:=]\s*["\']?(\d{7,9})["\']?', block, re.IGNORECASE)
        )
        if pmid_match:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"
        return ""

    def _extract_title(self, block: str) -> str:
        title_match = re.search(r'^(?:Article )?Title:\s*([^\n]+)', block, re.IGNORECASE | re.MULTILINE)
        if not title_match:
            title_match = re.search(r'\bTitle:\s*([^\n]+)', block, re.IGNORECASE)
        return title_match.group(1).strip() if title_match else "Prognosis Study"
