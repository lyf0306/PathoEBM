"""
Article processing utilities — parsing, screening, and consolidation.

Extracted from search_system.py as a mixin for AdvancedSearchSystem.
"""

import asyncio
import json
import logging
import re

from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..prompts import prompt_manager

logger = logging.getLogger(__name__)


class ArticleProcessingMixin:
    """
    Mixin providing article-level processing methods.

    Expects the host class to provide:
      - self.model, self.fast_model, self.treatment_context
      - self._prefilter_non_core_relevance, self._select_non_core_item
    """

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
            except asyncio.CancelledError:
                raise
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

        screening_prompt = prompt_manager.get("article_screening").format(
            candidate_count=len(articles_list),
            treatment_context=self.treatment_context,
            titles_catalog=titles_catalog,
        )

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
            except asyncio.CancelledError:
                raise
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
        trial_analysis = re.sub(r'\n### [^\n]+\n', '\n', trial_analysis)

        lighthouse_trials = [
            "GOG-99", "PORTEC-1", "PORTEC-2", "PORTEC-3", "GOG-0258",
            "GOG-209", "NRG-GY018", "RUBY", "ATTEND", "DUO-E",
            "KEYNOTE-775", "PORTEC-4a"
        ]

        # ── Parse into trial GROUPS (#### 🎯 blocks), not flat sections ──
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
        empty_markers = [
            "该检索方向无有效结果",
            "检索结果为空",
        ]
        cleaned = [l for l in lines if not any(
            m in l and not any(fm in l for fm in ["研究类型", "样本量", "关键结论"])
            for m in empty_markers
        )]
        return '\n'.join(cleaned)
