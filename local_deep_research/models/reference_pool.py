import logging
import re
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field
from ..search_system_support import SourcesReference, CEBM_LEVEL_DESCRIPTIONS

logger = logging.getLogger(__name__)

# Match 64-char hex hashes (content hashes, not real PMIDs)
_HASH_PATTERN = re.compile(r'^[0-9a-fA-F]{64,}$')
# Match real PMID-like numeric IDs
_PMID_PATTERN = re.compile(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)')


def _is_hash(s: str) -> bool:
    """Check if a string is a content hash (64+ hex chars, not a real ID)."""
    return bool(_HASH_PATTERN.match(s.strip()))


def _clean_source_label(ref) -> str:
    """Return a human-readable source label for a reference entry."""
    link = ref.link or ""
    # Real PubMed PMID
    m = _PMID_PATTERN.search(link)
    if m:
        return f"PMID: {m.group(1)}"
    # Content hash → show as internal reference
    if _is_hash(link):
        return "来源: 内部文献库"
    # Truncated URL fallback
    if link:
        cleaned = link.replace("https://", "").replace("http://", "")
        if len(cleaned) > 50:
            cleaned = cleaned[:50] + "..."
        return f"来源: {cleaned}"
    return "来源: 未知"


class ReferencePool:
    """Reference pool for citations, supporting baseline offset."""
    def __init__(self, baseline_max_index: int = 0) -> None:
        self.pool: List[SourcesReference] = []
        self.link2idx: dict[str, int] = {}
        self.base_idx = baseline_max_index

    def add(self, title: str, citation: str, link: str, cebm_level: str = "") -> int:
        if not link:
            return -1
        if link in self.link2idx:
            # Update CEBM level if existing entry lacks it and we now have one
            existing_idx = self.link2idx[link]
            if cebm_level:
                actual_idx = existing_idx - self.base_idx - 1
                if 0 <= actual_idx < len(self.pool) and not self.pool[actual_idx].cebm_level:
                    self.pool[actual_idx].cebm_level = cebm_level
            return existing_idx
        idx = self.base_idx + len(self.pool) + 1
        self.link2idx[link] = idx
        self.pool.append(
            SourcesReference(title=title or link, subtitle=citation or "", link=link,
                             cebm_level=cebm_level)
        )
        return idx

    def update_cebm_level(self, idx: int, level: str) -> bool:
        """Update CEBM evidence level for a reference by its pool index."""
        actual_idx = idx - self.base_idx - 1
        if 0 <= actual_idx < len(self.pool):
            self.pool[actual_idx].cebm_level = level
            return True
        return False

    def get_ref_by_idx(self, idx: int):
        actual_idx = idx - self.base_idx - 1
        if 0 <= actual_idx < len(self.pool):
            return self.pool[actual_idx]
        return None

    def display_label(self, idx: int) -> str:
        """Return a human-readable source label for a reference by pool index."""
        ref = self.get_ref_by_idx(idx)
        if ref is None:
            return f"[{idx}] (缺失)"
        return _clean_source_label(ref)

    @staticmethod
    def _normalize_citations(content: str) -> str:
        """
        Fix common LLM citation format issues before reindexing.

        Handles:
          - [^^n,^^m] or [^^n, ^^m] → [^^n] [^^m]  (comma-separated)
          - [^^42, ^49] → [^^42] [^^49]  (inconsistent ^ count)
          - [^^^n] → [^^n]  (triple ^ hallucination)
        """
        # General: any comma-separated [^^...] group with any number of ^ per item
        # e.g. [^^42, ^49], [^^1, ^^2, ^3], [^^1,^^2,^^3]
        def _split_comma_citations(match):
            inner = match.group(1)
            parts = re.split(r'\s*,\s*', inner)
            normalized = []
            for p in parts:
                p = p.strip()
                num = re.sub(r'^\^+', '', p)
                normalized.append(f'[^^{num}]')
            return ' '.join(normalized)

        content = re.sub(r'\[([\^]+\d+(?:\s*,\s*\^+\d+)+)\]', _split_comma_citations, content)
        # Triple ^ hallucination: [^^^59] → [^^59]
        content = re.sub(r'\[\^\^\^+(\d+)\]', r'[^^\1]', content)
        return content

    def reindex_references(self, content: str) -> Tuple[str, str]:
        """
        Re-index [^^n] citation markers sequentially and generate formatted reference text.
        Only affects [^^n] markers — plain [n] references (e.g. guideline footnotes) are left untouched.
        """
        # Pre-process: normalize malformed LLM citation formats before reindexing
        content = self._normalize_citations(content)

        # Find all [^^n] citation markers only (not plain [n] guideline refs)
        citation_pattern = r"\[\^\^(\d+)\]"
        all_cited_ids = [int(m.group(1)) for m in re.finditer(citation_pattern, content)]

        unique_cited_ids = list(dict.fromkeys(all_cited_ids))
        total_pool = len(self.pool)
        logger.info(
            "[RefPool] reindex_references: 在报告中找到 %d 个 [^^n] 引用（去重后 %d 个），"
            "ReferencePool 中共有 %d 条文献",
            len(all_cited_ids), len(unique_cited_ids), total_pool,
        )

        old_id_to_new_id = {}
        new_references_list = []
        current_new_id = self.base_idx + 1

        matched = 0
        for old_id in unique_cited_ids:
            ref_obj = self.get_ref_by_idx(old_id)
            if ref_obj:
                old_id_to_new_id[old_id] = current_new_id
                new_references_list.append((current_new_id, ref_obj))
                current_new_id += 1
                matched += 1

        if unique_cited_ids and not matched:
            logger.warning(
                "[RefPool] 所有 %d 个引用 ID 均未在 ReferencePool 中找到匹配！"
                "引用ID范围: %d-%d, 池中ID范围: 1-%d (base_idx=%d)",
                len(unique_cited_ids),
                min(unique_cited_ids), max(unique_cited_ids),
                total_pool, self.base_idx,
            )
        elif unique_cited_ids:
            missing = len(unique_cited_ids) - matched
            if missing:
                logger.warning(
                    "[RefPool] %d/%d 个引用未在池中找到匹配", missing, len(unique_cited_ids)
                )
            logger.info("[RefPool] 成功匹配 %d 条引用 → 生成 %d 条参考文献", matched, len(new_references_list))
        else:
            logger.warning("[RefPool] 报告中未找到任何 [^^n] 引用标记！参考文献列表将为空。")

        def replace_match(match):
            old_id = int(match.group(1))
            new_id = old_id_to_new_id.get(old_id, old_id)
            return f"[{new_id}]"

        new_content = re.sub(citation_pattern, replace_match, content)

        refs_text = "\n==================== 参考文献 (References) ====================\n"
        if new_references_list:
            for new_idx, ref in new_references_list:
                title = ref.title.replace("\n", " ").strip() if ref.title else ref.link
                if len(title) > 300:
                    title = title[:300] + "..."
                source_label = _clean_source_label(ref)
                # Study type badge (derived from CEBM level → human-readable description)
                study_type_badge = ""
                if ref.cebm_level and ref.cebm_level in CEBM_LEVEL_DESCRIPTIONS:
                    study_type_badge = f" | {CEBM_LEVEL_DESCRIPTIONS[ref.cebm_level]}"
                elif ref.cebm_level:
                    study_type_badge = f" | 证据等级 {ref.cebm_level}"
                refs_text += f"[{new_idx}] {source_label}{study_type_badge}\n"
                refs_text += f"    Title: {title}\n"
                refs_text += f"    Guidelines: 前沿证据合成 (Deep Research)\n"
                refs_text += "-" * 10 + "\n"

        return new_content, refs_text
