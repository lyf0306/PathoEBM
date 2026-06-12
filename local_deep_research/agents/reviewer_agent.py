import asyncio
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..utilities.patient_state import classify_surgery
from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..prompts import prompt_manager

if TYPE_CHECKING:
    from .context_bus import AgentContextBus

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """
    Post-hoc review agent — PURE QUALITY CHECKER, no repair.

    Detects quality issues and returns them grouped by section so the
    pipeline orchestrator can feed them back to the original generation
    agents (MDTReportAgent, FollowupAgent) for targeted re-generation
    with their full specialized prompts and surgical rules.

    Quick checks (regex, unambiguous template artifacts only):
      1. Placeholder leakage        — orphaned {{...}} tokens in final output
      2. Empty sections             — [此处]/None/待补充 in section body
      3. Citation coverage          — sections >200 chars without [^^n]
      4. Template instruction leak  — 内部指令/内部标签(NGS状态等)/禁止将本指令输出 leaked to output
      5. Cross-trial contamination  — identical survival numbers across trials
      6. Intra-trial contradictions — p53abn with suspiciously high survival numbers
      7. HPV/HP confusion           — HPV (human papillomavirus) ≠ HP (H. pylori)
      8. Numbering continuity       — numbered lists must start from 1, no gaps
      9. Ghost organ detection      — benign findings on removed organs (uterus/cervix/adnexa)
     10. Orphan [^^n] literals      — LLM output "[^^n]" without replacing n with a number
     11. Comorbidity deduplication  — same finding split across items
     12. PET-CT omission            — high-risk patients must have PET-CT in 疗效评估规划
     13. RT-OS evidence contradiction — recommending RT when cited trials show no OS benefit

    Deep review — Phase 2 (LLM):
     14. Logical consistency        — contradictions
     15. Prognosis data fidelity    — raw numbers cited
     16. Evidence support           — conclusions backed by trial data
     17. Completeness               — all 4 sections present
     18. Citation accuracy          — cross-validate [^^n] against source
     19. Trial context match        — trial population vs patient treatment stage
     20. Referral clinical judgment — trivial findings not over-referred
     21. HPV/TCT coverage           — NOT recommended for post-hysterectomy surveillance per NCCN
     22. Drug toxicity accuracy     — only discuss drugs actually used in referenced trials
     23. Molecular subtyping rigor  — conditional language when NGS pending

    Focused content review — Phase 2.5 (LLM):
     24. Comorbidity completeness    — every comorbidity from history addressed in post-op
     25. Molecular subtyping caution — no definitive typing when NGS pending (post-op section)
     26. Cross-section alignment     — post-op managed comorbidities also appear in follow-up
     27. Prognosis molecular caution — no subtype-specific survival data when NGS pending
     28. Incidental referral judgment — trivial findings not over-referred as individual consults
    """

    # Section definitions — used by _parse_section_tag for validation
    # and _group_issues_by_section for keyword fallback.
    # 术后处理 is now split into 主要方案 (TreatmentDecisionAgent) and
    # 合并症管理 (MDTReportAgent), each with its own ### sub-heading.
    _SECTION_PATTERNS: Dict[str, str] = {
        "病情分析": r"##\s*一[、．.]\s*病情分析",
        "主要方案": r"###\s*主要方案",
        "合并症管理": r"###\s*合并症管理",
        "预后分析": r"##\s*三[、．.]\s*预后分析",
        "随访方案": r"##\s*四[、．.]\s*随访方案",
    }

    _SECTION_KEYWORDS: Dict[str, List[str]] = {
        "病情分析": ["病情分析", "指南", "临床试验", "PICO", "证据"],
        "主要方案": ["主要方案", "术后处理", "肿瘤专科最终方案", "核心方案", "方案依据",
                    "放疗评估", "合并症适配", "疗效评估", "分子分型与复发风险",
                    "分子分型解读"],
        "合并症管理": ["合并症管理", "合并症与治疗期管理", "偶发发现合并声明",
                      "转诊", "随诊", "HPV", "TCT", "宫颈"],
        "预后分析": ["预后分析", "OS", "RFS", "生存率", "预后"],
        "随访方案": ["随访"],
    }

    # ═══════════════════════════════════════════════════════════════
    # Disease context helpers
    # ═══════════════════════════════════════════════════════════════
    _ENDOMETRIAL_RE = re.compile(
        r'endometrial\s+(?:cancer|carcinoma)|uterine\s+(?:cancer|carcinoma)|'
        r'子宫内膜癌|子宫内膜样癌|子宫体癌|宫体癌|endometrioid',
        re.IGNORECASE,
    )

    @classmethod
    def _is_endometrial_cancer(cls, *texts: str) -> bool:
        """Check if any of the given texts indicate an endometrial cancer case."""
        for t in texts:
            if t and cls._ENDOMETRIAL_RE.search(t):
                return True
        return False

    def __init__(self, report_model):
        self.report_model = report_model

    async def review(
        self,
        report: str,
        trial_analysis: str,
        followup_plan: str,
        prognosis_data: str,
        treatment_context: str,
        context_bus: Optional['AgentContextBus'] = None,
        surgery_type: str = "",
        previous_issues: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Review the report and return issues grouped by section.

        Returns:
          Dict[str, List[str]] — e.g. {"术后处理": ["问题1"], "随访方案": ["问题2"]}
          Empty dict means the report passed all checks.
        """
        # Gather inter-agent data from ContextBus (if available)
        bus_trial_data = ""
        if context_bus:
            trial_msgs = await context_bus.get_by_type("trial_analysis")
            if trial_msgs:
                bus_trial_data = trial_msgs[-1]["content"][:2500]

        # Phase 2.5 (LLM) launched concurrently — its inputs (report,
        # treatment_context, surgery_type) are read-only and independent
        # of Phase 1/2 results.  Phase 2.5 always runs regardless.
        focused_task = asyncio.ensure_future(
            self._focused_content_review(report, treatment_context, surgery_type)
        )

        # Phase 1: Quick structural checks (no LLM call needed)
        issues = self._quick_checks(report, surgery_type)
        if issues:
            logger.warning(f"[Reviewer] 快速检查发现 {len(issues)} 个问题")
            for i, iss in enumerate(issues, 1):
                logger.warning(f"[Reviewer]   Quick#{i}: {iss[:300]}")

        # Phase 2: LLM-based quality review (only if structural checks pass)
        if not issues:
            deep_issues = await self._deep_review(
                report, trial_analysis, prognosis_data,
                bus_trial_data=bus_trial_data,
                surgery_type=surgery_type,
                previous_issues=previous_issues,
            )
            if deep_issues:
                logger.warning(f"[Reviewer] 深度审查发现 {len(deep_issues)} 个问题")
                for i, iss in enumerate(deep_issues, 1):
                    logger.warning(f"[Reviewer]   Deep#{i}: {iss[:300]}")
                issues = deep_issues

        # Await Phase 2.5 (was running concurrently with Phase 1 + Phase 2)
        focused_issues = await focused_task
        if focused_issues:
            logger.warning(
                f"[Reviewer] 重点内容审查发现 {len(focused_issues)} 个问题"
            )
            for i, iss in enumerate(focused_issues, 1):
                logger.warning(f"[Reviewer]   Focused#{i}: {iss[:300]}")
            issues.extend(focused_issues)

        if not issues:
            logger.info("[Reviewer] 审查通过，报告质量合格。")
            return {}

        logger.warning(
            f"[Reviewer] 发现 {len(issues)} 个问题，返回 pipeline 触发 agent 重生成"
        )
        for i, iss in enumerate(issues, 1):
            logger.warning(f"[Reviewer]   Issue#{i}: {iss[:300]}")
        grouped = self._group_issues_by_section(issues)
        for section, items in grouped.items():
            logger.warning(f"[Reviewer]   → [{section}] {len(items)} 个问题")
        return grouped

    # -----------------------------------------------------------------
    # Phase 1: Rule-based structural checks (zero LLM cost)
    # -----------------------------------------------------------------
    def _quick_checks(self, report: str, surgery_type: str = "") -> List[str]:
        issues = []

        # Check 1: Placeholder leakage ({{...}} style)
        placeholders = re.findall(r"\{\{.*?\}\}", report)
        if placeholders:
            leaked = ", ".join(set(placeholders))
            issues.append(f"报告包含未替换的占位符: {leaked}。请直接替换为对应内容或删除。")
        # Check 1b: Bracket placeholder leakage ([xxx-如：...] style from mdt_report_agent)
        bracket_placeholders = re.findall(r"\[[^\]]*?(?:条目|占位)[^\]]*?\]", report)
        if bracket_placeholders:
            leaked = ", ".join(set(bracket_placeholders))
            issues.append(f"报告包含未替换的方括号占位符: {leaked}。这些是模板残留，必须替换为患者实际合并症信息或删除。")

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
            has_md_header = any(kw in sec_name for kw in ["随访", "预后", "术后处理", "病情"])
            if not has_citation and not has_md_header and len(sec) > 200:
                issues.append(f"章节「{sec_name}」缺少文献角标引用，请补充 [^^n] 标记。")

        # Check 4: Template instruction leakage
        issues.extend(self._check_template_leakage(report))

        # Check 5: Cross-trial data contamination
        issues.extend(self._check_trial_data_contamination(report))

        # Check 6: Intra-trial molecular subgroup data contradiction
        issues.extend(self._check_intra_trial_contradictions(report))

        # Check 7: HPV / HP terminological confusion
        issues.extend(self._check_hpv_hp_confusion(report))

        # Check 8: Numbering continuity — lists must start at 1 and be consecutive
        issues.extend(self._check_numbering_continuity(report))

        # Check 9: Post-hysterectomy cervix mention — anatomical impossibility
        issues.extend(self._check_post_hysterectomy_cervix_mention(report, surgery_type))

        # Check 10: Orphan [^^n] literal — LLM copied the template's example
        # citation placeholder as-is, without replacing n with an actual number.
        issues.extend(self._check_orphan_citation_placeholders(report))

        # Check 11: Comorbidity deduplication — same finding split across items
        issues.extend(self._check_comorbidity_deduplication(report))

        # Check 12 & 13 removed: PET-CT omission and RT-OS contradiction are
        # complex clinical judgments that belong in the LLM-based Phase 2.5
        # (_focused_content_review), not in regex-based quick checks.
        # Hardcoded trial names (PORTEC-3, GOG-0258) and molecular triggers
        # (p53abn→PET-CT) produced false positives that contradict the
        # "no hardcoded rules" design principle.

        # Check 14: Internal reasoning jargon leakage ("法则一/二/三" etc.)
        issues.extend(self._check_internal_jargon_leakage(report))

        return issues

    # =================================================================
    # Template instruction leakage check
    # =================================================================
    @staticmethod
    def _check_template_leakage(report: str) -> List[str]:
        """
        Check for LLM internal instructions that leaked into the final report.

        Covers three categories of leakage:
        1. Explicit meta-instruction markers (【内部指令】, etc.)
        2. Self-reminder sentences the LLM echoed from the prompt
           ("确保随访方案中没有提到...", "避免使用模糊的...")
        3. Conditional / operational instructions written as if they were
           clinical guidance ("如已行全子宫切除术，患者无宫颈，绝对禁止...")
        """
        leakage_patterns = [
            # ---- explicit meta markers ----
            (r'【\s*[🚨⚠️]*\s*内部指令', "内部指令标记（【内部指令】）泄漏"),
            (r'禁止将本指令输出为正文', "指令文字（'禁止将本指令输出为正文'）泄漏"),
            (r'本条替换为.*?禁止将本指令', "模板替换指令泄漏"),
            (r'如果已明确患者分子分型，这一条可以省略', "条件判断指令泄漏"),
            # ---- self-reminder / operational instructions ----
            (r'确保随访方案中', "自我提醒句式（'确保随访方案中...'）泄漏——这是给模型的指令，不是给患者的医嘱"),
            (r'避免使用模糊的随访频率', "自我提醒句式（'避免使用模糊的随访频率...'）泄漏"),
            (r'通读随访方案全文', "操作指导句式（'通读随访方案全文...'）泄漏"),
            (r'仔细阅读.*手术方式', "操作指导句式（'仔细阅读...手术方式...'）泄漏"),
            (r'输出前(强制)?自检', "自检指令（'输出前强制自检'）泄漏"),
            # ---- conditional instructions phrased like rules ----
            (r'如已行全子宫切除术[,，]\s*患者无宫颈[,，]\s*绝对禁止',
             "条件指令句式（'如已行全子宫切除术...绝对禁止...'）泄漏——直接写针对该患者的具体方案即可"),
            (r'如果手术方式是', "条件指令句式（'如果手术方式是...'）泄漏——直接写针对该患者的具体方案即可"),
            # ---- TreatmentDecisionAgent Phase 2 internal structural labels ----
            # These are meta-instruction labels from the Phase 1→Phase 2 decision summary
            # that must be translated to natural medical language, never printed verbatim.
            (r'NGS状态\s*[：:]', "内部标签（'NGS状态：'）泄漏——应写为自然语言如'NGS分子分型检测结果待回报'"),
            (r'IHC结果\s*[：:]', "内部标签（'IHC结果：'）泄漏——应写为自然语言如'IHC检测提示p53突变型模式'"),
            (r'措辞约束\s*[：:]', "内部标签（'措辞约束：'）泄漏——此为内部元指令，约束内容应融入自然段落，禁止以标签形式输出"),
            (r'复发风险判断\s*[：:]', "内部标签（'复发风险判断：'）泄漏——应直接写成自然段落而非带标签输出"),
            (r'条件性措辞——', "内部指令短语（'条件性措辞——'）泄漏——此为元指令前缀，应直接遵守约束输出条件性语句即可"),
            # ---- template numbering / format instructions ----
            (r'输出序号必须从\s*1\s*开始', "格式指令（'输出序号必须从1开始...'）泄漏——这是给模型的编号规则"),
            (r'逐条列出[,，]\s*每条从', "格式指令（'逐条列出，每条从...'）泄漏"),
            (r'序号从\s*4\s*开始', "格式指令（'序号从4开始'）泄漏——独立列表应从1开始"),
            # ---- meta-label echo (LLM repeats anti-leakage checklist as text) ----
            (r'【\s*核心\s*(临床)?\s*规则\s*】', "元指令标签（'【核心规则】'）泄漏"),
            (r'【\s*临床规则\s*[——-]', "元指令标签（'【临床规则】'）泄漏"),
            (r'输出前最终自检', "自检指令（'输出前最终自检'）泄漏"),
            (r'以下内容绝对禁止出现在你的输出中', "元指令正文泄漏——'绝对禁止出现在输出中'是给模型的指令"),
        ]
        issues = []
        for pattern, desc in leakage_patterns:
            if re.search(pattern, report):
                issues.append(f"模板泄漏：{desc}。请删除泄漏的指令文字，仅保留临床方案内容。")
        return issues

    # =================================================================
    # Internal reasoning jargon leakage check
    # =================================================================
    @staticmethod
    def _check_internal_jargon_leakage(report: str) -> List[str]:
        """
        Detect LLM internal reasoning jargon that leaked into the final report.
        Terms like "法则一/二/三", "组分隔离检查", "反合理化" are prompt-internal
        decision rules — they have no meaning to a clinician reading the report.
        """
        jargon_patterns = [
            (r'法则\s*[一二三]', "内部推理术语「法则一/二/三」泄漏——应改用临床语言描述（如'OS无显著获益 + ≥2项合并症 → 免除该追加治疗'）"),
            (r'组分隔离检查', "内部推理术语「组分隔离检查」泄漏——应改用自然语言描述治疗组分比较逻辑"),
            (r'反合理化', "内部推理术语「反合理化」泄漏——此为prompt内部约束，不应对临床读者输出"),
        ]
        issues = []
        for pattern, desc in jargon_patterns:
            if re.search(pattern, report):
                issues.append(f"内部术语泄漏：{desc}。")
        return issues

    # =================================================================
    # Cross-trial data contamination check
    # =================================================================
    @staticmethod
    def _check_trial_data_contamination(report: str) -> List[str]:
        """
        Check if different clinical trials have identical survival data values.
        Exact matches across multiple data points (OS%, PFS%, p-values) between
        different trials strongly suggest LLM data copying/hallucination.
        """
        parts = re.split(r'(?=^####\s)', report, flags=re.MULTILINE)
        trial_sections: Dict[str, List[str]] = {}
        for part in parts:
            if not part.startswith('####'):
                continue
            header_match = re.match(r'####\s+(.+?)\s+是一项', part)
            if not header_match:
                continue
            trial_name = header_match.group(1).strip()
            nums = re.findall(r'\d+%|[\d.]+\s*年(?!\w)|P\s*[=<>]\s*[\d.]+|[\d.]+\s*个月', part)
            if len(nums) >= 3:
                trial_sections[trial_name] = sorted(nums)

        names = list(trial_sections.keys())
        issues = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if trial_sections[names[i]] == trial_sections[names[j]]:
                    shared = ', '.join(trial_sections[names[i]][:5])
                    issues.append(
                        f"跨试验数据污染风险：试验「{names[i]}」与「{names[j]}」的生存数据数值完全一致"
                        f"（{shared}），可能发生了跨试验数据复制，请核实。"
                    )
        return issues

    # =================================================================
    # Intra-trial molecular subgroup data contradiction check
    # =================================================================
    @staticmethod
    def _check_intra_trial_contradictions(report: str) -> List[str]:
        """
        Check for numerical contradictions within a single trial's
        molecular subgroup analysis.  E.g., p53abn overall DFS is 29%
        but the same paragraph claims CRT vs CT subgroup rates of
        77% vs 60% -- those values belong to p53wt, not p53abn.
        """
        issues = []
        parts = re.split(r'(?=^####\s)', report, flags=re.MULTILINE)
        for part in parts:
            if not part.strip() or not part.startswith('####'):
                continue

            # Only inspect the "分子分型与亚组分析" bullet point
            sg = re.search(
                r'- \*\*分子分型与亚组分析\*\*[：:]\s*(.*?)(?=\n- \*\*|\Z)',
                part, re.DOTALL,
            )
            if not sg:
                continue

            subgroup_text = sg.group(1)

            # Check if p53abn is mentioned in this bullet
            if not re.search(r'p53(异常|突变|abn)', subgroup_text, re.IGNORECASE):
                continue

            # Extract percentage values
            pcts = [int(v) for v in re.findall(r'(\d+)%', subgroup_text)]
            if not pcts:
                continue

            # p53abn in endometrial cancer has very poor prognosis;
            # values >= 70% in the subgroup analysis strongly suggest
            # p53wt data was misattributed to p53abn.
            suspicious_high = [v for v in pcts if v >= 70]
            if suspicious_high:
                trial_m = re.match(r'####\s+(.+?)\s+(是一项|：|$)', part)
                trial_name = trial_m.group(1) if trial_m else "未知试验"
                high_s = '/'.join(str(v) + '%' for v in sorted(set(suspicious_high)))
                issues.append(
                    f"试验「{trial_name}」的分子亚组分析疑似数据归属错误："
                    f"该段落中 p53abn 分组比较出现过高数值（{high_s}），"
                    f"p53abn 型子宫内膜癌预后极差，通常不会达到如此高的生存率。"
                    f"这些数值很可能实际属于 p53 野生型或其他亚组。请核实数据归属。"
                )

        return issues

    # =================================================================
    # HPV / HP terminological confusion check
    # =================================================================
    @staticmethod
    def _check_hpv_hp_confusion(report: str) -> List[str]:
        """
        Detect when the LLM confuses HPV (human papillomavirus, gynecological)
        with HP (Helicobacter pylori, gastrointestinal).

        The most common signature: "HPV" appearing together with
        digestive-system keywords (消化科, 胃, 幽门, 根除).
        """
        issues = []
        # Find lines where "HPV" co-occurs with GI-related keywords
        for m in re.finditer(
            r'^.*HPV[^。\n]{0,60}(?:消化|胃[^肠]|幽门|根除\s*HP).*$',
            report, re.MULTILINE | re.IGNORECASE
        ):
            line = m.group(0).strip()[:120]
            issues.append(
                f"疑似 HPV/HP 混淆：'{line}' —— "
                f"HPV（人乳头瘤病毒）属于妇科下生殖道，不应与消化科或幽门螺杆菌（HP）关联。"
                f"请检查并修正：涉及 HPV 的事项→妇科/阴道镜；涉及 HP 的事项→消化科。"
            )
        return issues

    # =================================================================
    # Numbering continuity check
    # =================================================================
    @staticmethod
    def _check_numbering_continuity(report: str) -> List[str]:
        """
        Check that numbered lists within each section start from 1
        and are consecutive without gaps.

        The most common bug: LLM starts numbering at 4,5,6 under a
        subsection, echoing template instructions like "序号从4开始"
        or losing track of independent list contexts.
        """
        issues = []
        sections = re.split(r'(?=^##\s+[一二三四五六七八九十])', report, flags=re.MULTILINE)

        for sec in sections:
            if not sec.strip():
                continue
            sec_header = sec.split('\n')[0].strip()[:60]

            # Further split by sub-headers to isolate independent lists
            sub_blocks = re.split(r'(?=^#{3,4}\s)', sec, flags=re.MULTILINE)

            for block in sub_blocks:
                block_header = block.split('\n')[0].strip()[:40] if block.strip() else ""
                lines = block.split('\n')
                sequences = []
                current_seq = []

                for line in lines:
                    m = re.match(r'^\s*(\d+)[、．.]\s', line)
                    if m:
                        current_seq.append(int(m.group(1)))
                    else:
                        if len(current_seq) >= 2:
                            sequences.append(current_seq)
                        current_seq = []

                if len(current_seq) >= 2:
                    sequences.append(current_seq)

                label = f"{sec_header} > {block_header}" if block_header else sec_header

                for nums in sequences:
                    if nums[0] != 1:
                        issues.append(
                            f"编号异常：章节「{label}」中编号列表从 {nums[0]} 开始而非 1，"
                            f"编号序列: {nums[:8]}。独立编号列表应从 1 开始。"
                        )
                    else:
                        for i in range(1, len(nums)):
                            if nums[i] != nums[i - 1] + 1:
                                issues.append(
                                    f"编号不连续：章节「{label}」中期望编号 {nums[i-1]+1} 但出现 {nums[i]}，"
                                    f"编号序列: {nums[:8]}。请检查是否跳号。"
                                )
                                break

        return issues

    # =================================================================
    # Removed-organ benign finding check (uterus / cervix / adnexa)
    # =================================================================
    @staticmethod
    def _check_post_hysterectomy_cervix_mention(report: str, surgery_type: str = "") -> List[str]:
        """
        Detect "ghost organ" recommendations: benign findings on surgically
        removed organs (uterus, cervix, ovaries, tubes) that are listed as
        active management/follow-up items despite the organ being absent.

        Uses classify_surgery() for authoritative anatomy, with regex on the
        report text as belt-and-suspenders.
        """
        flags = classify_surgery(surgery_type)
        has_hysterectomy = flags["is_hysterectomy"] or bool(re.search(
            r'全子宫切除|子宫全切|子宫切除(?!.*保留)', report
        ))
        has_bso = flags["is_bso"] or bool(re.search(
            r'双侧(?:卵巢|附件).*切除|双附件.*切除|卵巢.*输卵管.*切除',
            report
        ))

        if not has_hysterectomy and not has_bso:
            return []

        issues = []

        # ---- Cervix-related screening (only if hysterectomy → no cervix) ----
        if has_hysterectomy:
            cervix_error_patterns = [
                (r'宫颈(?!管|口|癌前|上皮|间质|内膜|肌瘤)(?:病变|细胞学|筛查|刮片|涂片)', '宫颈筛查/病变评估'),
                (r'阴道镜检查.*宫颈', '阴道镜检查宫颈'),
                (r'宫颈癌筛查', '宫颈癌筛查'),
                (r'HPV.*宫颈(?!管|口|癌前|上皮|间质|内膜|肌瘤)(?:病变|细胞学|筛查|刮片|涂片|取样|检查)', 'HPV宫颈取样/筛查'),
            ]
            # Negation phrases that indicate the report is already correctly
            # stating that cervical screening is NOT applicable — these should
            # NOT be flagged as ghost-organ errors.
            _NEGATION_WINDOW = 60  # chars before and after match to check
            _NEGATION_PATTERNS = [
                r'宫颈筛查不适用',
                r'无宫颈[，,]\s*常规宫颈筛查不适用',
                r'已行全子宫切除术[，,]\s*无宫颈',
                r'不推荐.*宫颈',
                r'不适用.*宫颈',
                r'宫颈.*不适用',
                r'不存在宫颈',
                r'无需.*宫颈筛查',
                r'宫颈筛查.*(?:无需|不)',
            ]
            for pattern, desc in cervix_error_patterns:
                for m in re.finditer(pattern, report):
                    match_start = m.start()
                    match_end = m.end()
                    # Extract surrounding context for negation check
                    ctx_start = max(0, match_start - _NEGATION_WINDOW)
                    ctx_end = min(len(report), match_end + _NEGATION_WINDOW)
                    surrounding = report[ctx_start:ctx_end]
                    # Skip if the surrounding context explicitly negates the finding
                    if any(re.search(neg_pat, surrounding) for neg_pat in _NEGATION_PATTERNS):
                        logger.debug(
                            f"[Reviewer] 幽灵宫颈检查：跳过「{m.group(0)[:60]}」——"
                            f"上下文已包含否定声明（如'不适用''无宫颈'等）"
                        )
                        continue
                    context = m.group(0)[:80]
                    issues.append(
                        f"解剖学错误（幽灵宫颈）：患者已行全子宫切除术（无宫颈），"
                        f"但报告中出现了「{context}」（{desc}）。已切除子宫的患者不存在宫颈，"
                        f"此项操作在解剖学上不可能。请删除该条目或改为关注阴道残端黏膜。"
                    )

        # ---- Uterine benign findings (hysterectomy → no uterus at all) ----
        if has_hysterectomy:
            uterus_benign_patterns = [
                (r'子宫(多发)?(小)?肌瘤(?!.*切除|.*术后|.*病史)', '子宫肌瘤'),
                (r'子宫腺肌(?:症|病)', '子宫腺肌症'),
                (r'子宫内膜息肉', '子宫内膜息肉'),
                (r'宫腔(?:内)?(?:异常|占位|回声|分离)', '宫腔内异常'),
                (r'宫颈纳(?:氏|囊|特)(?:囊肿)?', '宫颈纳囊'),
                (r'宫颈(?:多发)?(?:小)?囊肿(?!.*切除)', '宫颈囊肿'),
                (r'宫颈息肉', '宫颈息肉'),
                (r'宫颈肥大', '宫颈肥大'),
                (r'宫颈糜烂', '宫颈糜烂'),
                (r'宫颈潴留(?:囊肿)?', '宫颈潴留囊肿'),
            ]
            for pattern, desc in uterus_benign_patterns:
                for m in re.finditer(pattern, report):
                    context = m.group(0)[:80]
                    # Skip clearly historical mentions
                    surrounding = report[max(0, m.start() - 20):m.end() + 30]
                    if re.search(r'既往|病史|因.*行.*切除|术后病理', surrounding):
                        continue
                    issues.append(
                        f"解剖学错误（幽灵子宫/宫颈）：患者已行全子宫切除术（子宫+宫颈均已切除），"
                        f"但报告中提及了「{context}」（{desc}）。"
                        f"该器官已不存在，请删除此条目。如为历史信息需标明'既往'。"
                    )

        # ---- Adnexal benign findings (BSO → no ovaries/tubes) ----
        if has_bso:
            adnexal_benign_patterns = [
                (r'卵巢(多发)?(小)?囊肿(?!.*切除|.*术后)', '卵巢囊肿'),
                (r'卵巢囊性(?:回声|病变|结构)', '卵巢囊性病变'),
                (r'卵巢周围(?:炎|粘连)', '卵巢周围炎'),
                (r'输卵管积水', '输卵管积水'),
                (r'输卵管积液', '输卵管积液'),
                (r'输卵管囊肿', '输卵管囊肿'),
                (r'输卵管增粗', '输卵管增粗'),
                (r'附件(?:区)?囊肿(?!.*切除)', '附件囊肿'),
                (r'附件区囊性', '附件区囊性病变'),
            ]
            for pattern, desc in adnexal_benign_patterns:
                for m in re.finditer(pattern, report):
                    context = m.group(0)[:80]
                    surrounding = report[max(0, m.start() - 20):m.end() + 30]
                    if re.search(r'既往|病史|因.*行.*切除|术后病理', surrounding):
                        continue
                    issues.append(
                        f"解剖学错误（幽灵附件）：患者已行双侧附件切除术（无卵巢/输卵管），"
                        f"但报告中提及了「{context}」（{desc}）。"
                        f"该器官已不存在，请删除此条目。如为历史信息需标明'既往'。"
                    )

        return issues

    # =================================================================
    # Orphan [^^n] citation placeholder check
    # =================================================================
    @staticmethod
    def _check_orphan_citation_placeholders(report: str) -> List[str]:
        """
        Detect [^^n] with the literal letter 'n' — the LLM copied the
        template's example citation placeholder without replacing n with
        an actual reference number.  Valid citations are [^^1], [^^2], etc.
        """
        issues = []
        orphans = re.findall(r'\[\^\^n\]', report)
        if orphans:
            count = len(orphans)
            issues.append(
                f"未替换的引用占位符：报告中出现 {count} 处字面量 '[^^n]'（模板残留），"
                f"请将 '[^^n]' 替换为实际引用编号 '[^^1]'、'[^^2]' 等，"
                f"或删除无法关联引用的 '[^^n]' 标记。"
            )
        return issues

    # =================================================================
    # Comorbidity item deduplication check
    # =================================================================
    # Known examination/imaging modality keywords — when two numbered items
    # both reference the same modality, they likely describe one finding.
    _EXAM_MODALITY_PATTERNS = [
        r'胃镜(?:检查)?(?:提示|示|所见)',
        r'肠镜(?:检查)?(?:提示|示|所见)',
        r'肺(?:部)?CT(?:提示|示)',
        r'冠脉(?:CTA|CT)(?:提示|示)',
        r'心脏(?:超声|彩超)(?:提示|示)',
        r'腹部(?:超声|彩超|CT)(?:提示|示)',
        r'下肢(?:动脉|静脉)?(?:超声|彩超)(?:提示|示)',
        r'(?:动态)?心电图(?:提示|示)',
        r'MR[IA]?\s*(?:检查)?(?:提示|示)',
        r'上腹部(?:CT|增强CT)(?:提示|示)',
    ]

    _DEPARTMENT_KEYWORDS = [
        r'消化科', r'心内科', r'呼吸科', r'内分泌科', r'肾内科',
        r'神经内科', r'妇科', r'皮肤科', r'眼科', r'口腔科',
        r'泌尿外科', r'普外科', r'骨科', r'血液科', r'风湿免疫科',
        r'肝胆外科',
    ]

    # Same-organism pathogens that should not appear in separate items
    _PATHOGEN_PATTERNS = [
        (r'HP\s*\(\s*\+\s*\)|HP\s*现症感染|幽门螺杆菌', 'HP'),
        (r'HPV\s*(\d+\s*)?\(\s*\+\s*\)|HPV\s*阳性|人乳头瘤病毒', 'HPV'),
    ]

    @classmethod
    def _check_comorbidity_deduplication(cls, report: str) -> List[str]:
        """
        Detect when the same clinical finding is split across multiple
        numbered items in the comorbidity management section.

        Common failure modes:
        - Gastroscopy finding split: "胃炎伴糜烂"(item 2) + "HP(+)"(item 3)
        - Same CT finding split across items by anatomical sub-site
        - Same pathogen mentioned in two adjacent items
        """
        issues = []

        # 1. Find all numbered items between ### 合并症管理 and the next
        #    ## / ### boundary (or end of report).
        sec_start = re.search(r'###\s*合并症管理', report)
        if not sec_start:
            return issues

        remainder = report[sec_start.end():]
        # Stop at next ## <number> or ### <sub-section>
        next_boundary = re.search(
            r'^#{2,3}\s+(?:[一二三四五六七八九十]|主要方案|合并症管理)',
            remainder, re.MULTILINE,
        )
        mgmt_text = remainder[:next_boundary.start()] if next_boundary else remainder

        # 2. Extract numbered items.  Supports:
        #      （1）...（2）...（3）...   (fullwidth parentheses)
        #      1、... 2、... 3、...     (halfwidth number + Chinese comma)
        #    Split on the next number marker; each item is a (number, text) pair.
        item_boundary = re.compile(
            r'(?:（\s*(\d+)\s*）|(?<!\d)(\d+)[、．.])'
        )
        raw_items = []
        for m in item_boundary.finditer(mgmt_text):
            num = int(m.group(1) or m.group(2))
            raw_items.append((num, m.end()))  # m.end() = first char of content

        if len(raw_items) < 2:
            return issues

        parsed_items = []
        for idx, (num, content_start) in enumerate(raw_items):
            end = raw_items[idx + 1][1] if idx + 1 < len(raw_items) else len(mgmt_text)
            text = re.sub(r'\s+', '', mgmt_text[content_start:end])
            # Skip section headers falsely matched as numbered items
            # (e.g. "1、 **合并症与治疗期管理**：")
            if '**' in text or '治疗期' in text or '偶发发现' in text:
                continue
            parsed_items.append((num, text))

        # 3. Check ALL pairs (not just adjacent) for signs of over-splitting.
        #    HP + gastritis may be separated by an unrelated item — adjacency
        #    is too brittle.
        for i in range(len(parsed_items)):
            for j in range(i + 1, len(parsed_items)):
                num_a, text_a = parsed_items[i]
                num_b, text_b = parsed_items[j]

                reasons = []

                # Check A: Same examination modality in both items
                for modality_pat in cls._EXAM_MODALITY_PATTERNS:
                    if re.search(modality_pat, text_a) and re.search(modality_pat, text_b):
                        reasons.append("两者均引用同一检查手段")
                        break

                # Check B: Same department referral
                dept_a = None
                dept_b = None
                for dept_pat in cls._DEPARTMENT_KEYWORDS:
                    m_a = re.search(dept_pat, text_a)
                    m_b = re.search(dept_pat, text_b)
                    if m_a:
                        dept_a = m_a.group(0)
                    if m_b:
                        dept_b = m_b.group(0)
                if dept_a and dept_a == dept_b:
                    reasons.append(f"均建议{dept_a}就诊")

                # Check C: Same pathogen scattered across items
                pathogen_duplicate = False
                for pathogen_pat, pathogen_name in cls._PATHOGEN_PATTERNS:
                    in_a = bool(re.search(pathogen_pat, text_a))
                    in_b = bool(re.search(pathogen_pat, text_b))
                    if in_a and in_b:
                        reasons.append(f"两者均提及{pathogen_name}感染")
                        pathogen_duplicate = True
                    elif (in_a or in_b) and dept_a == dept_b:
                        reasons.append(f"{pathogen_name}感染与关联疾病可能被拆分")

                if len(reasons) >= 2 or pathogen_duplicate or (dept_a is not None and dept_a == dept_b):
                    reason_str = "；".join(reasons)
                    snippet_a = re.sub(r'\s+', '', text_a)[:60]
                    snippet_b = re.sub(r'\s+', '', text_b)[:60]
                    issues.append(
                        f"合并症条目重复：第({num_a})条「{snippet_a}…」"
                        f"与第({num_b})条「{snippet_b}…」描述的是同一临床发现"
                        f"（{reason_str}）。上游数据可能包含同一病症的多种描述，"
                        f"请识别并合并为一条，不要对每条描述分别生成条目。"
                    )

        return issues

    # =================================================================
    # Patient-section extraction helpers
    # =================================================================
    @staticmethod
    def _extract_patient_section(report: str) -> str:
        """Extract the patient's own diagnosis/staging section, excluding trial descriptions.

        The report structure: 「一、 病情分析」→ patient data, then later sections contain
        trial evidence synthesis.  We extract only the 病情分析 section so risk-factor
        regexes don't match trial populations (e.g. "PORTEC-3纳入III期患者").
        """
        m = re.search(
            r'(?:##\s*一[、，]\s*病情分析|##\s*病情分析)\s*\n(.*?)(?=\n##\s+(?:二|核心|主要|合并|随访|预后|参考))',
            report, re.DOTALL,
        )
        if m:
            return m.group(1)
        # Fallback: first 2000 chars (usually contains patient summary before trials)
        return report[:2000]

    @staticmethod
    def _has_patient_trigger(report: str, pattern: str, flags=0) -> bool:
        """Match a regex pattern only against patient sections, not trial descriptions."""
        patient_text = ReviewerAgent._extract_patient_section(report)
        return bool(re.search(pattern, patient_text, flags))

    # =================================================================
    # PET-CT baseline assessment omission check
    # =================================================================
    @staticmethod
    def _check_pet_ct_omission(report: str) -> List[str]:
        """
        Check if PET-CT should have been recommended based on the PATIENT's own
        risk factors (not trial population descriptions).

        PET-CT triggers (any one suffices):
          ① High-risk histology: 浆液性癌, 透明细胞癌, 癌肉瘤
          ② FIGO Stage IV only (NOT III — ordinary endometrioid III does not trigger)
          ③ p53abn confirmed (NGS returned, no "待" modifier)
        """
        issues = []
        patient_text = ReviewerAgent._extract_patient_section(report)

        # --- Detect triggers in patient data ONLY ---
        has_high_risk_histology = bool(re.search(
            r'浆液性癌|透明细胞癌|癌肉瘤', patient_text
        ))

        # Match IV[A-I]?期 only (IV期 is FIGO stage 4)
        has_figo_4 = bool(re.search(
            r'IV[A-Z]?\d*\s*期', patient_text
        ))

        # p53abn confirmed only when NGS is NOT pending
        has_p53abn_confirmed = (
            bool(re.search(r'p53abn|p53\s*(?:突变|异常)', patient_text))
            and not bool(re.search(
                r'待回报|待NGS|检测中|结果未出|待确认|NGS.*待|分子分型.*待',
                patient_text
            ))
        )

        needs_pet_ct = has_high_risk_histology or has_figo_4 or has_p53abn_confirmed

        if not needs_pet_ct:
            return issues

        # --- Extract 疗效评估规划 text ---
        plan_match = re.search(
            r'(?:（4）|\(4\))\s*\**疗效评估规划\**\s*[：:]\s*(.*?)(?=\n\s*(?:2、|###|\n\n\s*\n|$))',
            report, re.DOTALL,
        )
        if not plan_match:
            plan_match = re.search(
                r'疗效评估规划\s*[：:]\s*(.*?)(?=\n\s*(?:2、|###|\n\n\s*\n|$))',
                report, re.DOTALL,
            )

        if plan_match:
            plan_text = plan_match.group(0)
            if 'PET-CT' not in plan_text and 'PET/CT' not in plan_text:
                triggers = []
                if has_high_risk_histology:
                    triggers.append('高危组织学类型（浆液性癌/透明细胞癌/癌肉瘤）')
                if has_figo_4:
                    triggers.append('FIGO IV期')
                if has_p53abn_confirmed:
                    triggers.append('p53abn已确认')
                trigger_str = ' + '.join(triggers)
                issues.append(
                    f"PET-CT遗漏：患者具有{trigger_str}，但疗效评估规划中未包含PET-CT基线评估。"
                    f"PET-CT应作为追加项（不替代基线MRI/CT）在疗效评估规划中明确列出。"
                )

        return issues

    # =================================================================
    # RT recommendation section extraction
    # =================================================================
    @staticmethod
    def _extract_rt_section(report: str) -> str:
        """Extract the 放疗评估 / 主要方案 section where RT recommendations live."""
        m = re.search(
            r'(?:放疗评估|主要方案)[\s\S]*?(?=\n##\s+(?:合并|随访|预后|核心|参考)|\n###\s+)',
            report,
        )
        return m.group(0) if m else report

    # =================================================================
    # RT-OS evidence contradiction check
    # =================================================================
    @staticmethod
    def _check_rt_os_contradiction(report: str) -> List[str]:
        """
        Detect when the report's RT recommendation contradicts the evidence
        it cites to justify that recommendation.

        Only checks the RT recommendation section and the evidence it directly
        references — not trial mentions elsewhere in the report (evidence
        synthesis sections describe all trials, not just those supporting RT).
        """
        issues = []

        # Check if report recommends RT
        rt_section = ReviewerAgent._extract_rt_section(report)
        rt_recommended = bool(re.search(
            r'(?:建议放疗|放疗科会诊|推荐.*?放疗|保留放疗|盆腔EBRT|盆腔外照射)',
            rt_section,
        ))

        if not rt_recommended:
            # Also check full report for cases where RT section is hard to isolate
            rt_recommended = bool(re.search(
                r'(?:放疗评估|建议).*?(?:建议放疗|放疗科会诊|推荐.*?放疗|保留放疗)',
                report, re.DOTALL,
            ))
            if not rt_recommended:
                return issues

        # --- Extract the evidence-citing portions near the RT recommendation ---
        # Get context around RT recommendation: 500 chars before and after
        rt_context = ""
        rt_m = re.search(
            r'.{0,500}(?:建议放疗|放疗科会诊|推荐.*?放疗|保留放疗|盆腔EBRT).{0,1000}',
            report, re.DOTALL,
        )
        if rt_m:
            rt_context = rt_m.group(0)

        # --- Pattern 1: RT recommendation cites a no-OS-benefit trial as evidence ---
        # Only flag if the trial name AND the OS non-significance appear NEAR
        # the RT recommendation (i.e. the report is using that trial to argue)
        rt_trials_no_os_benefit = []
        for trial, pattern in [
            ('PORTEC-1', r'PORTEC[- ]?1.*?OS[^。]{0,80}(?:无显著|无统计学|无差异|p\s*[=>]\s*0\.\d)'),
            ('GOG-99', r'GOG[- ]?99.*?OS[^。]{0,80}(?:无显著|无统计学|无差异|p\s*[=>]\s*0\.\d)'),
            ('PORTEC-2', r'PORTEC[- ]?2.*?OS[^。]{0,80}(?:无显著|无统计学|无差异|p\s*[=>]\s*0\.\d)'),
        ]:
            # Only match within RT context, not the entire report
            search_text = rt_context if rt_context else report
            if re.search(pattern, search_text, re.IGNORECASE):
                rt_trials_no_os_benefit.append(trial)

        if rt_trials_no_os_benefit:
            trial_list = '、'.join(rt_trials_no_os_benefit)
            issues.append(
                f"放疗决策与OS证据矛盾：报告建议放疗，但放疗推荐段提及的{trial_list}"
                f"试验中放疗组vs非放疗组的OS均无显著差异。"
                f"若患者有≥2项慢性合并症，OS无显著获益的追加治疗应免除。"
                f"请核实是否有其他试验支持放疗的OS获益。"
            )
            return issues

        # --- Pattern 2: PORTEC-3 used as sole OS evidence for RT ---
        # PORTEC-3: CRT(CT+RT) vs RT — both arms have RT, difference is chemo.
        # It cannot justify "adding RT".  Only flag if PORTEC-3 appears in or
        # near the RT recommendation (not just mentioned in evidence synthesis).
        search_text = rt_context if rt_context else report
        cites_portec3 = bool(re.search(r'PORTEC[- ]?3', search_text, re.IGNORECASE))
        has_rt_specific_os = bool(re.search(
            r'GOG[- ]?0258.*?(?:CRT|放化疗).*?(?:CT|化疗).*?OS.*?(?:HR|p\s*[=>])',
            search_text, re.IGNORECASE | re.DOTALL,
        ))
        has_other_rt_os_evidence = bool(re.search(
            r'(?:EBRT|盆腔放疗|外照射).*?(?:vs|对比|相比|优于).*?(?:观察|化疗|无放疗).*?OS.*?(?:显著|获益|HR|p\s*[=<])',
            search_text, re.IGNORECASE | re.DOTALL,
        ))

        if cites_portec3 and not has_rt_specific_os and not has_other_rt_os_evidence:
            issues.append(
                "放疗决策缺乏OS证据：报告建议放疗，但引用的PORTEC-3比较的是CRT vs RT"
                "（两组都含放疗，差异仅在化疗），不能用于论证'加放疗'的OS获益。"
                "若GOG-0258（CRT vs CT——含放疗vs不含放疗）未显示CRT的OS显著优于CT，"
                "则'加放疗'无OS获益证据。若患者有≥2项慢性合并症，"
                "OS无显著获益 + ≥2项合并症 → 该追加治疗应免除。"            )

        return issues

    # -----------------------------------------------------------------
    # Phase 2: LLM-based deep quality review
    # -----------------------------------------------------------------
    async def _deep_review(
        self, report: str, trial_analysis: str,
        prognosis_data: str,
        bus_trial_data: str = "",
        surgery_type: str = "",
        previous_issues: Optional[Dict[str, List[str]]] = None,
    ) -> List[str]:
        # If context bus has richer trial data than the explicit param, use it
        source_trials = bus_trial_data if len(bus_trial_data) > len(trial_analysis) else trial_analysis

        # 根据实际手术方式生成审查维度 8（HPV/TCT），避免 reviewer 错误标记保留生育功能报告
        is_hysterectomy = bool(surgery_type) and any(
            kw in surgery_type for kw in ["全子宫", "子宫切除", "子宫全切"]
        )
        if is_hysterectomy:
            hpv_tct_dimension = (
                "8. **HPV/TCT 随访覆盖**：审阅【初步会诊草稿】中是否提及了 HPV 阳性史或宫颈病变史。"
                "如果有，报告**不得**在术后处理或随访方案中包含\"常规行 TCT 及 HPV 检测\"等建议。"
                "患者已行全子宫切除术（无宫颈），NCCN 指南不推荐常规阴道细胞学检查用于术后随访。"
                "正确做法是建议妇科查体时关注阴道残端黏膜，必要时行阴道壁 HPV 检测。"
                "HPV 应按常规人群筛查策略管理，不应混入肿瘤专科随访的必查项目中。"
            )
        else:
            hpv_tct_dimension = (
                "8. **HPV/TCT 随访覆盖**：审阅【初步会诊草稿】中是否提及了 HPV 阳性史或宫颈病变史。"
                "如果有，报告应在术后处理或随访方案中包含相应的管理措施。"
                "注意患者手术方式——若非全子宫切除术（如保留生育功能治疗），"
                "子宫宫颈完整，应纳入宫颈癌筛查（TCT+HPV）。"
            )

        # Build "错题本核销" block for re-reviews (stateful review)
        previous_issues_block = ""
        if previous_issues:
            issues_lines = []
            for section, items in previous_issues.items():
                for item in items:
                    issues_lines.append(f"  - [{section}] {item}")
            if issues_lines:
                issues_text = "\n".join(issues_lines)
                block_lines = [
                    "🔴 **【错题本核销——上一轮标记的问题清单】**",
                    "",
                    issues_text,
                    "",
                    "**【验收任务——核销历史问题，但允许纠偏】**：",
                    "这是作者提交的修改稿。请逐一核实上述问题是否被解决：",
                    "1. 作者是否已修正你指出的具体错误（如HR数值、试验名称、错误引用）？",
                    "2. 如果作者以不同的措辞表达了相同的意思但逻辑正确 → 通过。",
                    "   不要因为措辞变化而打回——关注实质而非形式。",
                    "3. **如果经你再次核实，发现原问题实际上是误报（false positive）**",
                    "   → 在输出中注明'[原问题已核实为误报——通过]'，不再计入新问题。",
                    "   例：原文已正确注明'不适用''无宫颈'但你上次未注意到 → 主动承认并通过。",
                    "4. 只有当所有历史问题都已被修正或确认为误报，你才能输出 PASS。",
                    "5. 如果同一问题已被标记≥2次但作者始终以合理方式处理 → 视为争议问题，",
                    "   输出 PASS 并注明'[争议问题——建议人工复核]'。",
                    "",
                ]
                previous_issues_block = "\n".join(block_lines)

        prompt = prompt_manager.get("reviewer_phase2_deep").format(
            previous_issues_block=previous_issues_block,
            hpv_tct_dimension=hpv_tct_dimension,
            prognosis_data=prognosis_data[:2000],
            source_trials=source_trials[:2500],
            report=report[:8000],
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.report_model, prompt, timeout=180.0, max_retries=1
            )
            content = remove_think_tags(resp.content).strip()
            if content == "PASS":
                return []
            issues = [line.strip("- ").strip() for line in content.split("\n") if line.startswith("- ")]
            return issues
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[Reviewer] 深度审查异常: {e}")
            return []

    # -----------------------------------------------------------------
    # Phase 2.5: Focused content integrity review
    # -----------------------------------------------------------------
    async def _focused_content_review(
        self, report: str, treatment_context: str, surgery_type: str = "",
    ) -> List[str]:
        """
        Focused review of five high-risk areas against patient ground truth:

        1. **合并症齐全性** — every comorbidity in the patient's medical history
           must be addressed in 合并症管理 (post-op management). Follow-up 不再承担合并症随诊职责。
        2. **分子分型是否擅自定性（主要方案）** — if molecular status is pending
           (待回报/待NGS), the post-op molecular interpretation must use conditional
           language, not definitive subtyping.
        3. **跨章节合并症对齐** — 随访方案中不应出现独立的合并症随诊条目（合并症管理已在术后处理章节完整覆盖）。
           随访方案中出现"合并症管理""科室随诊""心内科""内分泌科""消化科""血管外科"等独立条目 → 问题。
        4. **偶发发现过度转诊** — if the report declares certain findings are
           "临床意义有限/定期随访即可" (合并声明), check that those same trivial
           findings are NOT individually listed as specialist referrals.
        5. **合并症条目过度拆分** — same organ/system conditions must be merged,
           not split across multiple numbered entries.

        Uses treatment_context (raw EBM report text containing the patient's full
        medical history) as the ground-truth source.
        """
        # Extract relevant sections from the report for focused review
        main_plan_section = self._extract_section_text(report, "主要方案") or ""
        comorbidity_section = self._extract_section_text(report, "合并症管理") or ""
        followup_section = self._extract_section_text(report, "随访方案") or ""

        # Build post-op section from both sub-sections for review
        post_op_section = (
            ("### 主要方案\n" + main_plan_section + "\n\n### 合并症管理\n" + comorbidity_section)
            if main_plan_section or comorbidity_section else ""
        )

        # If 术后处理 is missing entirely, skip (caught by other checks)
        if len(post_op_section) < 50:
            return []

        # Extract patient medical history from treatment_context
        patient_source = treatment_context[:4000] if treatment_context else ""

        prompt = prompt_manager.get("reviewer_phase25_focused").format(
            patient_source=patient_source,
            main_plan_section=main_plan_section[:3000] if main_plan_section else "(未提取到)",
            comorbidity_section=comorbidity_section[:3000] if comorbidity_section else "(未提取到)",
            followup_section=followup_section[:2000],
            surgery_type=surgery_type or "未提供",
        )

        try:
            resp = await invoke_with_timeout_and_retry(
                self.report_model, prompt, timeout=180.0, max_retries=1
            )
            content = remove_think_tags(resp.content).strip()
            if not content or content.upper() in ("PASS", "ALL_PASS"):
                logger.info("[Reviewer] 重点内容审查通过")
                return []
            # Parse per-dimension format: "- X-LABEL: PASS" or "- X-LABEL: [section] issue"
            # Also handle legacy format: "- [section] issue"
            issues = []
            for line in content.split("\n"):
                stripped = line.strip()
                if not stripped.startswith("- "):
                    continue
                body = stripped[2:]  # remove "- "
                # Check if it's a per-dimension line with a PASS
                if re.match(r'^[A-G]-[^:]+:\s*PASS\s*$', body):
                    continue
                # Per-dimension format with issue: "- A-PET-CT: [主要方案] ..."
                dim_match = re.match(r'^[A-G]-[^:]+:\s*(.+)$', body)
                if dim_match:
                    issue_text = dim_match.group(1).strip()
                    if issue_text:
                        issues.append(issue_text)
                else:
                    # Legacy format: "- [section] issue"
                    issues.append(body)
            if issues:
                logger.info(
                    f"[Reviewer] 重点内容审查发现 {len(issues)} 个问题"
                )
            return issues
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[Reviewer] 重点内容审查异常: {e}")
            return []

    # =================================================================
    # Section helpers
    # =================================================================
    @classmethod
    def _parse_section_tag(cls, issue: str) -> Tuple[str, str]:
        """Extract [section] tag from issue text. Returns (section_name, cleaned_issue).

        Only recognizes tags matching a known section name — prevents false
        matches like [^^n] citation markers from being hijacked as section tags.
        """
        m = re.match(r'\[(.+?)\]\s*(.*)', issue)
        if m and m.group(1) in cls._SECTION_PATTERNS:
            return m.group(1), m.group(2)
        return "通用", issue

    @classmethod
    def _group_issues_by_section(cls, issues: List[str]) -> Dict[str, List[str]]:
        """Group issues by their section tag, with keyword fallback for untagged issues."""
        grouped: Dict[str, List[str]] = {}
        for issue in issues:
            section, cleaned = cls._parse_section_tag(issue)
            # Content-based detection fallback for untagged issues
            if section == "通用":
                for sec_name, keywords in cls._SECTION_KEYWORDS.items():
                    if any(kw in issue for kw in keywords):
                        section = sec_name
                        break
            grouped.setdefault(section, []).append(cleaned or issue)
        return grouped

    @classmethod
    def _extract_section_text(cls, report: str, section_key: str) -> str:
        """Extract a section's full content from the report by section name.

        Used by _focused_content_review to isolate sections for LLM review.
        Differs from the old _get_section_content in that it searches by
        heading keywords rather than pre-defined regex patterns, making it
        more robust against formatting variations.
        """
        patterns = {
            "病情分析": r"##\s*一[、．.]\s*病情",
            "主要方案": r"###\s*主要方案",
            "合并症管理": r"###\s*合并症管理",
            "预后分析": r"##\s*三[、．.]\s*预后",
            "随访方案": r"##\s*四[、．.]\s*随访方案",
        }
        pattern = patterns.get(section_key, "")
        if not pattern:
            return ""
        target_match = re.search(pattern, report)
        if not target_match:
            return ""
        section_start = target_match.start()
        # Find the next section boundary: either ## <number> or ### <sub-section>
        next_section = re.search(
            r'^#{2,3}\s+(?:[一二三四五六七八九十]|主要方案|合并症管理)',
            report[section_start + 1:], re.MULTILINE,
        )
        if next_section:
            section_end = section_start + 1 + next_section.start()
        else:
            section_end = len(report)
        return report[section_start:section_end].strip()
