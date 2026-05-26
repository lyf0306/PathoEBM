"""
ReportEvaluator — MDT 报告质量量化评估

Tier 1: 客观规则合规检查（自动化，零 LLM 成本）
Tier 2: 引用健康度（自动化）

使用方式:
    evaluator = ReportEvaluator(ref_pool=ref_pool, structured_task=structured_task)
    results = evaluator.evaluate(report)
    logger.info(results.summary())

也可独立运行脚本对历史报告进行评估:
    python report_evaluator.py --report report.md --task task.json
"""

import argparse
import json
import logging
import re
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# EvaluationResults — 结构化评估结果 + 格式化输出
# =====================================================================
class EvaluationResults:
    """Container for evaluation results with a formatted summary."""

    def __init__(self):
        # Tier 1
        self.compliance_checks: Dict[str, bool] = {}
        # Tier 2
        self.citation_total: int = 0
        self.citation_valid: int = 0
        self.citation_broken: int = 0
        self.unique_citations: int = 0
        self.sections_without_citation: List[str] = []

    @property
    def compliance_rate(self) -> float:
        if not self.compliance_checks:
            return 0.0
        passed = sum(1 for v in self.compliance_checks.values() if v)
        return passed / len(self.compliance_checks)

    @property
    def citation_valid_rate(self) -> float:
        if self.citation_total == 0:
            return 0.0
        return self.citation_valid / self.citation_total

    def summary(self) -> str:
        """返回格式化的评估摘要（日志专用，不写入报告）。"""
        lines = []

        # ── Tier 1 ──
        passed = sum(1 for v in self.compliance_checks.values() if v)
        total = len(self.compliance_checks)
        lines.append(f"{'=' * 50}")
        lines.append("【Rule Compliance】 {}/{} = {:.1%}".format(passed, total, self.compliance_rate))
        lines.append(f"{'=' * 50}")
        for check, ok in self.compliance_checks.items():
            icon = "✅" if ok else "❌"
            lines.append(f"  {icon} {check}")

        # ── Tier 2 ──
        lines.append("")
        lines.append(f"{'=' * 50}")
        lines.append("【Citation Health】")
        lines.append(f"{'=' * 50}")
        lines.append(f"  引用总数: {self.citation_total}")
        lines.append(f"  有效引用: {self.citation_valid}")
        lines.append(f"  断裂引用: {self.citation_broken}")
        lines.append(f"  引用有效占比: {self.citation_valid_rate:.1%}")
        lines.append(f"  唯一引用: {self.unique_citations}")
        if self.sections_without_citation:
            secs = "、".join(self.sections_without_citation[:5])
            lines.append(f"  无引用的章节: {secs}")
        else:
            lines.append(f"  无引用的章节: 无")

        # ── Composite ──
        lines.append("")
        t1_score = self.compliance_rate * 100
        t2_score = self.citation_valid_rate * 100 if self.citation_total > 0 else 0
        composite = t1_score * 0.6 + t2_score * 0.4
        lines.append(f"{'=' * 50}")
        lines.append(f"【综合评分】合规 {t1_score:.1f}/100 + 引用 {t2_score:.1f}/100 = {composite:.1f}/100")
        lines.append(f"{'=' * 50}")

        return "\n".join(lines)


# =====================================================================
# ReportEvaluator — 主评估类
# =====================================================================
class ReportEvaluator:
    """
    MDT report quality evaluator.

    Tier 1: Rule compliance checks (zero LLM cost, regex-based).
    Tier 2: Citation health metrics.

    Usage:
        evaluator = ReportEvaluator(ref_pool, structured_task)
        results = evaluator.evaluate(report)
        logger.info(results.summary())
    """

    def __init__(self, ref_pool=None, structured_task: dict = None):
        self.ref_pool = ref_pool
        self.structured_task = structured_task or {}

    def evaluate(self, report: str) -> EvaluationResults:
        """Run Tier 1 + Tier 2 evaluation on the report."""
        results = EvaluationResults()

        # ── Tier 1: Rule compliance ──
        self._check_section_completeness(report, results)
        self._check_no_citations_in_postop(report, results)
        self._check_merged_incidental_declaration(report, results)
        self._check_no_empty_sections(report, results)
        self._check_no_template_leakage(report, results)
        self._check_no_trial_contamination(report, results)
        self._check_no_orphaned_placeholders(report, results)
        self._check_hpv_tct_compliance(report, results)
        self._check_no_tracking_confirmed_markers(report, results)

        # ── Tier 2: Citation health ──
        self._check_citation_health(report, results)

        return results

    # =================================================================
    # Tier 1 helper: section completeness
    # =================================================================
    def _check_section_completeness(self, report: str, results: EvaluationResults):
        """Check that all 4 main sections (病情分析/术后处理/预后分析/随访方案) are present."""
        required = [
            (r"## 一[、.]\s*病情分析", "## 一"),
            (r"## 二[、.]\s*术后处理", "## 二"),
            (r"## 三[、.]\s*预后分析", "## 三"),
            (r"## 四[、.]\s*随访方案", "## 四"),
        ]
        missing = []
        for pattern, name in required:
            if not re.search(pattern, report):
                missing.append(name)

        results.compliance_checks["四个主章节齐全"] = len(missing) == 0
        if missing:
            results.compliance_checks["四个主章节齐全"] = False

    # =================================================================
    # Tier 1 helper: no citations in 术后处理
    # =================================================================
    def _check_no_citations_in_postop(self, report: str, results: EvaluationResults):
        """术后处理 must NOT contain [^^n] or [n] citation markers (rule 5 in prompt)."""
        for punct in ['、', '．', '.', '：', ':']:
            pattern = rf"(##\s*二{re.escape(punct)}\s*术后处理[\s\S]*?)(?=\n##\s*[三四]|\Z)"
            m = re.search(pattern, report, re.DOTALL)
            if m:
                section_text = m.group(1)
                # Check both [^^n] and [n] citation formats
                citations = re.findall(
                    r'\[\^?\^?\^?\d+(?:\s*[,、，]\s*\^?\^?\^?\d+)*\s*\]',
                    section_text,
                )
                results.compliance_checks["术后处理无角标"] = len(citations) == 0
                return

        # Section heading not found at all
        results.compliance_checks["术后处理无角标"] = False

    # =================================================================
    # Tier 1 helper: merged incidental declaration
    # =================================================================
    def _check_merged_incidental_declaration(self, report: str, results: EvaluationResults):
        """Must contain merged incidental findings declaration (规则 4)."""
        patterns = [
            r"临床意义有限[，,]\s*定期随访即可",
            r"其余偶发发现.*临床意义有限",
            r"偶发发现合并声明",
            r"定期随访即可",
        ]
        for p in patterns:
            if re.search(p, report):
                results.compliance_checks["偶发发现合并声明"] = True
                return
        results.compliance_checks["偶发发现合并声明"] = False

    # =================================================================
    # Tier 1 helper: no empty / failed sections
    # =================================================================
    def _check_no_empty_sections(self, report: str, results: EvaluationResults):
        """No empty sections with None/待补充/生成失败."""
        empty_patterns = [
            r"## .*?\n\n(?:\[此处|【请|None|待补充|未填写|生成失败)",
        ]
        found = False
        for p in empty_patterns:
            if re.search(p, report, re.IGNORECASE):
                found = True
                break
        if "（生成失败）" in report:
            found = True

        results.compliance_checks["无空段/生成失败"] = not found

    # =================================================================
    # Tier 1 helper: no template leakage
    # =================================================================
    def _check_no_template_leakage(self, report: str, results: EvaluationResults):
        """Reuse the same logic as ReviewerAgent._check_template_leakage."""
        leakage_patterns = [
            (r'【\s*[🚨]*\s*内部指令', "内部指令标记"),
            (r'禁止将本指令输出为正文', "指令文字泄漏"),
            (r'本条替换为.*?禁止将本指令', "模板替换指令泄漏"),
            (r'如果已明确患者分子分型，这一条可以省略', "条件判断指令泄漏"),
        ]
        for pattern, desc in leakage_patterns:
            if re.search(pattern, report):
                results.compliance_checks["无模板指令泄漏"] = False
                return
        results.compliance_checks["无模板指令泄漏"] = True

    # =================================================================
    # Tier 1 helper: no cross-trial data contamination
    # =================================================================
    def _check_no_trial_contamination(self, report: str, results: EvaluationResults):
        """Reuse the same logic as ReviewerAgent._check_trial_data_contamination."""
        parts = re.split(r'(?=^####\s)', report, flags=re.MULTILINE)
        trial_sections: Dict[str, List[str]] = {}
        for part in parts:
            if not part.startswith('####'):
                continue
            header_match = re.match(r'####\s+(.+?)\s*是一项', part)
            if not header_match:
                continue
            trial_name = header_match.group(1).strip()
            nums = re.findall(r'\d+%|[\d.]+\s*年(?!\w)|P\s*[=<>]\s*[\d.]+|[\d.]+\s*个月', part)
            if len(nums) >= 3:
                trial_sections[trial_name] = sorted(nums)

        names = list(trial_sections.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if trial_sections[names[i]] == trial_sections[names[j]]:
                    results.compliance_checks["无跨试验数据污染"] = False
                    return
        results.compliance_checks["无跨试验数据污染"] = True

    # =================================================================
    # Tier 1 helper: no orphaned placeholders
    # =================================================================
    def _check_no_orphaned_placeholders(self, report: str, results: EvaluationResults):
        """No orphaned {{...}} placeholders in final output."""
        placeholders = re.findall(r"\{\{.*?\}\}", report)
        results.compliance_checks["无孤立占位符"] = len(placeholders) == 0

    # =================================================================
    # Tier 1 helper: HPV/TCT rule compliance
    # =================================================================
    def _check_hpv_tct_compliance(self, report: str, results: EvaluationResults):
        """
        Per NCCN: routine vaginal cytology is NOT recommended for post-hysterectomy
        endometrial cancer surveillance. Report must NOT recommend TCT/HPV as a
        surveillance item (规则 4 updated per clinical feedback).
        """
        report_lower = report.lower()
        # Detect surveillance-context TCT/HPV recommendations
        tct_patterns = [
            r"常规行.*tct",
            r"tct.*检测",
            r"tct.*随访",
            r"建议.*tct",
            r"行.*tct.*检查",
            r"常规行.*hpv",
            r"hpv.*检测",
        ]
        has_tct_recommendation = any(
            re.search(p, report_lower) for p in tct_patterns
        )
        # If TCT is recommended in surveillance context, that violates NCCN
        results.compliance_checks["HPV/TCT规则合规"] = not has_tct_recommendation

    # =================================================================
    # Tier 1 helper: no "追踪" for confirmed molecular markers
    # =================================================================
    def _check_no_tracking_confirmed_markers(self, report: str, results: EvaluationResults):
        """
        If structured_task has confirmed molecular markers (e.g., p53abn, pMMR),
        check that the report does NOT write "追踪"/"需进一步明确" for those markers (规则 7).
        """
        profile = self.structured_task.get("oncology_profile", {}) or {}
        pathology = profile.get("pathology_and_molecular", "") or ""
        pathology_lower = pathology.lower()

        # Determine which markers are confirmed in structured_task
        confirmed_markers = []
        if "p53" in pathology_lower and any(
            kw in pathology_lower for kw in ["突变", "异常", "abn", "野生", "wt"]
        ):
            confirmed_markers.append("p53")
        if "mmr" in pathology_lower or "msi" in pathology_lower:
            confirmed_markers.append("MMR")
        if "pole" in pathology_lower:
            confirmed_markers.append("POLE")
        if "nsmp" in pathology_lower:
            confirmed_markers.append("NSMP")

        # Check for tracking phrases associated with molecular markers
        tracking_phrases = ["追踪", "需进一步明确", "尚待明确", "待明确"]
        report_lower = report.lower()

        has_tracking = False
        # Search sentences for marker + tracking phrase co-occurrence
        sentences = re.split(r'[。！？\n]', report_lower)
        for sent in sentences:
            if any(phrase in sent for phrase in tracking_phrases):
                for marker in confirmed_markers:
                    if marker.lower() in sent:
                        has_tracking = True
                        break
                if has_tracking:
                    break

        # Even without confirmed markers from structured_task, check if
        # the report itself says "追踪" for molecular markers (possible bug)
        if not confirmed_markers:
            tracking_patterns = [
                r"p53.*(?:追踪|待明确|需进一步|尚待明确)",
                r"分子分型.*(?:追踪|待明确|需进一步|尚待明确)",
                r"MMR.*(?:追踪|待明确|需进一步|尚待明确)",
                r"MSI.*(?:追踪|待明确|需进一步|尚待明确)",
            ]
            for pattern in tracking_patterns:
                if re.search(pattern, report, re.IGNORECASE):
                    has_tracking = True
                    break

        results.compliance_checks["已确认分子分型无'追踪'"] = not has_tracking

    # =================================================================
    # Tier 2: Citation health
    # =================================================================
    def _check_citation_health(self, report: str, results: EvaluationResults):
        """Check citation validity, uniqueness, and section coverage."""
        if not self.ref_pool:
            # Can't validate without ref_pool — record raw counts only
            all_cites = re.findall(r'(?<![\^])\[(\d+)\](?!\])', report)
            pre_cites = re.findall(r'\[\^\^(\d+)\]', report)
            results.citation_total = len(all_cites) + len(pre_cites)
            results.citation_valid = 0
            results.citation_broken = 0
            results.unique_citations = 0
            results.sections_without_citation = []
            logger.info("[ReportEvaluator] ref_pool 为空，引用健康度检查跳过")
            return

        # Find all citations — handle both [^^n] (pre-reindex) and [n] (post-reindex)
        citations_pre = re.findall(r'\[\^\^(\d+)\]', report)
        citations_post = re.findall(r'(?<!\[)\[(\d+)\](?!\]|\[)', report)

        # Filter out false positives: year numbers, dose numbers, etc.
        def _is_likely_citation(num_str: str) -> bool:
            n = int(num_str)
            return 1 <= n <= 500  # reasonable citation range

        all_citations = []
        for c in citations_pre:
            n = int(c)
            if _is_likely_citation(str(n)):
                all_citations.append(("^^", n))
        for c in citations_post:
            n = int(c)
            # Skip common false positives: 2 (years), 3 (cycles), 5 (AUC), etc.
            if n == 2 or n == 3 or n == 5:
                continue
            if _is_likely_citation(str(n)):
                all_citations.append(("plain", n))

        total = len(all_citations)
        valid = 0
        broken = 0
        seen = set()

        for fmt, idx in all_citations:
            if idx <= 0:
                broken += 1
                continue
            ref = self.ref_pool.get_ref_by_idx(idx)
            if ref:
                valid += 1
                seen.add(idx)
            else:
                broken += 1

        results.citation_total = total
        results.citation_valid = valid
        results.citation_broken = broken
        results.unique_citations = len(seen)

        # Sections without any citation
        sections = re.split(r"\n##\s+", report)
        no_cite_sections = []
        for sec in sections:
            if not sec.strip():
                continue
            sec_name = sec.split("\n")[0].strip()[:40]
            # Skip reference section and 随访 section (allowed per rules)
            if "参考" in sec_name or "reference" in sec_name.lower():
                continue
            if "随访" in sec_name:
                continue
            has_citation = bool(re.search(r'\[\d+\]', sec)) or bool(
                re.search(r'\[\^\^\d+\]', sec)
            )
            if not has_citation and len(sec) > 200:
                no_cite_sections.append(sec_name)

        results.sections_without_citation = no_cite_sections


# =====================================================================
# CLI entry point (方案 A: standalone script)
# =====================================================================
def main():
    """Standalone CLI for evaluating historical reports."""
    parser = argparse.ArgumentParser(description="MDT Report Evaluator")
    parser.add_argument("--report", required=True, help="Path to report markdown file")
    parser.add_argument("--task", required=True, help="Path to structured_task JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load report
    with open(args.report, "r", encoding="utf-8") as f:
        report = f.read()

    # Load structured_task
    with open(args.task, "r", encoding="utf-8") as f:
        structured_task = json.load(f)

    evaluator = ReportEvaluator(ref_pool=None, structured_task=structured_task)
    results = evaluator.evaluate(report)

    print(results.summary())


if __name__ == "__main__":
    main()
