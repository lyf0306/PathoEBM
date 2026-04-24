import logging
import re
import textwrap
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags

if TYPE_CHECKING:
    from .context_bus import AgentContextBus

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """
    Post-hoc review agent that checks report quality and triggers targeted fixes.

    Review dimensions:
      1. Placeholder leakage     — orphaned {{...}} tokens in final output
      2. Citation coverage       — substantive claims without [^^n] references
      3. Logical consistency     — contradictions (e.g. "取消放疗" vs "建议放疗")
      4. Prognosis data fidelity — prognosis section actually cites numbers

    This is an optional enhancement layer that does NOT change the default
    execution path — it only intervenes when quality issues are found.
    """

    MAX_REPAIR_ATTEMPTS = 1  # prevent infinite loops

    def __init__(self, report_model):
        self.report_model = report_model

    async def review_and_repair(
        self,
        report: str,
        trial_analysis: str,
        followup_plan: str,
        prognosis_data: str,
        treatment_context: str,
        context_bus: Optional['AgentContextBus'] = None,
    ) -> Tuple[str, bool]:
        """
        Review the report and repair if needed.

        Returns:
          (final_report, was_repaired) — always returns a valid report.
        """
        # Phase 1: Quick structural checks (no LLM call needed)
        issues = self._quick_checks(report)

        # Phase 1.5: Gather inter-agent data from ContextBus (if available)
        bus_trial_data = ""
        if context_bus:
            trial_msgs = await context_bus.get_by_type("trial_analysis")
            if trial_msgs:
                bus_trial_data = trial_msgs[-1]["content"][:2500]

        # Phase 2: LLM-based quality review (only if structural checks pass)
        if not issues:
            issues = await self._deep_review(
                report, trial_analysis, followup_plan, prognosis_data,
                treatment_context, bus_trial_data=bus_trial_data,
            )

        if not issues:
            logger.info("[Reviewer] 审查通过，报告质量合格。")
            return report, False

        logger.warning(f"[Reviewer] 发现 {len(issues)} 个问题，尝试修复...")
        repaired = await self._repair(report, issues)
        logger.info(f"[Reviewer] 修复完成。")
        return repaired, True

    # -----------------------------------------------------------------
    # Phase 1: Rule-based structural checks (zero LLM cost)
    # -----------------------------------------------------------------
    def _quick_checks(self, report: str) -> List[str]:
        issues = []

        # Check 1: Placeholder leakage
        placeholders = re.findall(r"\{\{.*?\}\}", report)
        if placeholders:
            leaked = ", ".join(set(placeholders))
            issues.append(f"报告包含未替换的占位符: {leaked}。请直接替换为对应内容或删除。")

        # Check 2: Empty sections
        empty_patterns = [
            (r"## .*?\n\n(?:\[此处|【请|None|待补充|未填写)", "疑似空段"),
        ]
        for pattern, desc in empty_patterns:
            if re.search(pattern, report, re.IGNORECASE):
                issues.append(f"报告包含{desc}，请补充完整内容。")

        # Check 3: Citation anomaly — consecutive ^^ references are normal,
        # but report sections without any citation suggest missing evidence
        sections = re.split(r"\n##\s+", report)
        for sec in sections:
            if not sec.strip():
                continue
            sec_name = sec.split("\n")[0].strip()[:40]
            has_citation = bool(re.search(r"\[\^\^?\d+]", sec))
            has_md_header = "随访" in sec_name or "预后" in sec_name
            if not has_citation and not has_md_header and len(sec) > 200:
                issues.append(f"章节「{sec_name}」缺少文献角标引用，请补充 [^^n] 标记。")

        return issues

    # -----------------------------------------------------------------
    # Phase 2: LLM-based deep quality review
    # -----------------------------------------------------------------
    async def _deep_review(
        self, report: str, trial_analysis: str,
        followup_plan: str, prognosis_data: str,
        treatment_context: str, bus_trial_data: str = "",
    ) -> List[str]:
        # If context bus has richer trial data than the explicit param, use it
        source_trials = bus_trial_data if len(bus_trial_data) > len(trial_analysis) else trial_analysis

        prompt = textwrap.dedent(f"""
        你是一名顶级的妇科肿瘤 MDT 报告质量控制专家。
        请审查以下 MDT 报告，找出所有质量问题。

        【审查维度】：
        1. **逻辑一致性**：报告是否存在自相矛盾之处？（如先说"取消放疗"后说"建议放疗"）
        2. **数据保真**：预后分析部分是否引用了【预后专员数据】中的具体数字？
        3. **证据支撑**：核心临床结论是否有对应的临床试验数据支撑？
        4. **完整性**：四个章节（病情分析、术后处理、预后分析、随访方案）是否都完整？
        5. **引用准确性**：报告中的 [^^n] 引用是否与实际试验分析内容匹配？（交叉验证：报告中声称的证据是否在原始分析中真实存在，而非幻觉）

        【预后专员提供的原始数据】：
        {prognosis_data[:2000]}

        【临床试验分析原文（用于交叉验证引用准确性）】：
        {source_trials[:2500]}

        【待审查的 MDT 报告】：
        {report[:5000]}

        【输出格式】：
        如果没问题，请只输出：PASS
        如果有问题，请列出具体问题，每行一个，以 "- " 开头：
        - 问题1：...
        - 问题2：...
        """)

        try:
            resp = await invoke_with_timeout_and_retry(
                self.report_model, prompt, timeout=180.0, max_retries=1
            )
            content = remove_think_tags(resp.content).strip()
            if content == "PASS":
                return []
            issues = [line.strip("- ").strip() for line in content.split("\n") if line.startswith("- ")]
            return issues
        except Exception as e:
            logger.warning(f"[Reviewer] 深度审查异常: {e}")
            return []

    # -----------------------------------------------------------------
    # Repair: targeted fix based on issues
    # -----------------------------------------------------------------
    async def _repair(self, report: str, issues: List[str]) -> str:
        issues_text = "\n".join(f"- {iss}" for iss in issues)

        prompt = textwrap.dedent(f"""
        你是一名妇科肿瘤 MDT 报告编辑专家。
        以下 MDT 报告存在质量问题，请直接输出**修复后**的完整报告。

        【需要修复的问题】：
        {issues_text}

        【原始报告】：
        {report[:8000]}

        【🚨 修复规则】：
        1. 只修复上述列出的具体问题，不要修改其他内容。
        2. 占位符问题：直接删除孤立的 {{...}} 标记，或用合理内容替换。
        3. 缺失引用：在合适位置添加 [^^n] 标记。
        4. 逻辑矛盾：保留正确的结论，删除矛盾的表述。
        5. 保持报告的 Markdown 结构完整。
        6. 保留所有文献角标和参考文献格式。

        💡 请直接输出修复后的完整报告，不要加额外说明。
        """)

        for attempt in range(self.MAX_REPAIR_ATTEMPTS + 1):
            try:
                resp = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=240.0, max_retries=1
                )
                repaired = remove_think_tags(resp.content).strip()
                if repaired:
                    # Verify repair didn't make it worse
                    remaining = self._quick_checks(repaired)
                    if len(remaining) < len(issues):
                        return repaired
                    logger.warning(f"[Reviewer] 修复尝试 {attempt+1} 未完全解决问题，继续...")
            except Exception as e:
                logger.warning(f"[Reviewer] 修复异常: {e}")

        return report  # fallback: return original if repair fails
