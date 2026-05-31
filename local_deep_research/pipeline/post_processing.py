"""
Post-processing checks — statistical sanity, numbering fix, citation credibility.

Extracted from search_system.py as a mixin for AdvancedSearchSystem.
"""

import logging
import re

logger = logging.getLogger(__name__)


class PostProcessingMixin:
    """
    Mixin providing post-processing quality checks.

    Expects the host class to provide:
      - self.ref_pool
    """

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
