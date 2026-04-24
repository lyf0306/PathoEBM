import re
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field
from ..search_system_support import SourcesReference

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

    def add(self, title: str, citation: str, link: str) -> int:
        if not link:
            return -1
        if link in self.link2idx:
            return self.link2idx[link]
        idx = self.base_idx + len(self.pool) + 1
        self.link2idx[link] = idx
        self.pool.append(
            SourcesReference(title=title or link, subtitle=citation or "", link=link)
        )
        return idx

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
          - [^^^n] → [^^n]  (triple ^ hallucination)
        """
        # Comma-separated: [^^15,^^10] or [^^15, ^^10] → [^^15] [^^10]
        content = re.sub(
            r'\[\^\^(\d+)\s*,\s*\^\^(\d+)\]',
            r'[^^\1] [^^\2]',
            content,
        )
        # Three-way comma: [^^1,^^2,^^3] or [^^1, ^^2, ^^3]
        content = re.sub(
            r'\[\^\^(\d+)\s*,\s*\^\^(\d+)\s*,\s*\^\^(\d+)\]',
            r'[^^\1] [^^\2] [^^\3]',
            content,
        )
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
        old_id_to_new_id = {}
        new_references_list = []
        current_new_id = self.base_idx + 1

        for old_id in unique_cited_ids:
            ref_obj = self.get_ref_by_idx(old_id)
            if ref_obj:
                old_id_to_new_id[old_id] = current_new_id
                new_references_list.append((current_new_id, ref_obj))
                current_new_id += 1

        def replace_match(match):
            old_id = int(match.group(1))
            new_id = old_id_to_new_id.get(old_id, old_id)
            return f"[^^{new_id}]"

        new_content = re.sub(citation_pattern, replace_match, content)

        refs_text = "\n==================================================\n"
        if new_references_list:
            for new_idx, ref in new_references_list:
                title = ref.title.replace("\n", " ").strip() if ref.title else ref.link
                if len(title) > 300:
                    title = title[:300] + "..."
                source_label = _clean_source_label(ref)
                refs_text += f"[{new_idx}] {source_label}\n"
                refs_text += f"    Title: {title}\n"
                refs_text += f"    Guidelines: 前沿证据合成 (Deep Research)\n"
                refs_text += "-" * 10 + "\n"

        return new_content, refs_text
