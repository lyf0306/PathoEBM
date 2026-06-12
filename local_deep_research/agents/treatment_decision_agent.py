import asyncio
import logging
import re
import textwrap

from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags, strip_llm_preamble
from ..prompts import prompt_manager

logger = logging.getLogger(__name__)


class TreatmentDecisionAgent:
    """
    Agent 2a: Treatment decision specialist — Plan-and-Execute architecture.

    Phase 1 (_plan_treatment_decision):  Focused evidence analysis + clinical
        decision logic.  Produces a structured decision summary.
    Phase 2 (_write_treatment_plan):  Pure formatting.  Translates the Phase-1
        decision summary into the final output format without re-litigating.

    This split keeps each prompt short enough that attention decay doesn't
    silently drop late-prompt dimensions (RT derivation, PET-CT, 反合理化).
    """

    def __init__(self, report_model, structured_task: dict, context_bus=None):
        self.report_model = report_model
        self.structured_task = structured_task
        self.context_bus = context_bus

    # =================================================================
    # Main entry point
    # =================================================================
    async def generate_main_plan(
        self,
        trial_analysis: str,
        ref_map_str: str,
        patient_summary: str = "",
        guideline_section: str = "",
        safety_context: str = "",
        risk_factor_context: str = "",
        reviewer_issues: list = None,
        previous_output: str = "",
    ) -> str:
        """
        Plan-and-Execute: Phase 1 decides → Phase 2 formats.
        Phase 1 receives only guideline + trial evidence + concise structured patient data.
        Phase 2 receives the decision summary + oncology_core + proposed_plan for formatting.
        """

        # ── Structured patient data (concise, from JSON — replaces narrative texts) ──
        profile = self.structured_task.get("oncology_profile", {}) or {}
        oncology_core = {
            "basic_info": profile.get("basic_info", ""),
            "diagnosis_and_stage": profile.get("diagnosis_and_stage", ""),
            "pathology_and_molecular": profile.get("pathology_and_molecular", ""),
            "surgery_type": profile.get("surgery_type", ""),
        }

        comorbidities = self.structured_task.get("major_comorbidities_affecting_treatment", []) or []
        comorbidity_count = len(comorbidities)

        prelim_plan = self.structured_task.get("preliminary_plan", {}) or {}
        proposed_plan = {
            "main_oncology_treatment": (prelim_plan.get("main_oncology_treatment") or prelim_plan.get("main", "") or "").strip(),
            "follow_up_schedule": prelim_plan.get("follow_up_schedule", ""),
        }

        # ── Reviewer feedback block (goes into Phase 1 only) ──
        feedback_block = ""
        if reviewer_issues:
            issues_text = "\n".join(f"  - {iss}" for iss in reviewer_issues)
            previous_text = previous_output if previous_output else "（无上一轮输出记录）"
            feedback_block = (
                "🔴🔴🔴 **【强制纠错指令 —— 你之前的草稿被医学质控委员会打回】** 🔴🔴🔴\n\n"
                "你之前的草稿犯了以下严重错误，被质控委员会打回：\n"
                f"{issues_text}\n\n"
                "**【你的强制任务——做不到等于失败】**：\n"
                "1. 必须深刻理解上述每条错误。如果指出了数据张冠李戴（如错误的HR值、错误的试验名），"
                "你必须在本次生成中彻底弃用该错误数据，填入正确的数值和引用来源。\n"
                "2. 必须在原文基础上进行精准修复，确保试验名称与数据完全匹配。\n"
                "3. 严禁使用模棱两可的话术绕过问题（如\"需要进一步评估\"替代具体数值）。"
                "修复后的文本必须能明确体现你已经逐条采纳了上述意见。\n"
                "4. 严禁通过改变主语、替换概念、模糊表述等方式\"钻空子\"。"
                "如果质控意见说HR=0.69是错的、正确值是0.54，你必须把0.69替换为0.54。\n\n"
                "⚠️ 未标记问题的部分保持原样即可，不要修改已经正确的部分。\n\n"
                "【你上一轮的完整输出——请在此基础上有针对性地修正上述问题】\n"
                f"---\n{previous_text}\n---\n\n"
                "🔴🔴🔴\n"
            )

        # ── Extract preliminary plan as Phase 1 baseline anchor ──
        prelim_tx = proposed_plan.get("main_oncology_treatment", "")

        # =================================================================
        # Phase 1: Evidence analysis → structured decision summary
        # (guideline + trial evidence + concise patient data + prelim_tx baseline)
        # =================================================================
        logger.info("[TreatmentDecisionAgent] Phase 1: 循证分析与决策...")
        decision_summary = await self._plan_treatment_decision(
            oncology_core=oncology_core,
            comorbidity_count=comorbidity_count,
            guideline_section=guideline_section,
            trial_analysis=trial_analysis,
            feedback_block=feedback_block,
            prelim_tx=prelim_tx,
        )
        logger.info(
            f"[TreatmentDecisionAgent] Phase 1 完成（{len(decision_summary)} 字符）"
        )

        # =================================================================
        # Phase 2: Format decision summary → final output
        # (decision + structured patient context from JSON)
        # =================================================================
        logger.info("[TreatmentDecisionAgent] Phase 2: 格式化输出...")
        final_output = await self._write_treatment_plan(
            oncology_core=oncology_core,
            proposed_plan=proposed_plan,
            decision_summary=decision_summary,
            ref_map_str=ref_map_str,
        )
        logger.info(
            f"[TreatmentDecisionAgent] Phase 2 完成（{len(final_output)} 字符）"
        )

        if "肿瘤专科最终方案" in final_output:
            return final_output

        # Should not reach here — Phase 2 prompt enforces the format marker
        logger.error("[TreatmentDecisionAgent] Phase 2 输出缺少核心方案标记")
        return "1、 **肿瘤专科最终方案**\n（生成失败）\n\n2、 **分子分型与复发风险解读**\n（生成失败）"

    # =================================================================
    # Phase 0: Pre-extract OS treatment effect data from trial evidence
    # =================================================================
    async def _extract_trial_os_data(self, trial_analysis: str,
                                     retry_hint: str = "") -> str:
        """
        Focused single-task LLM call: extract OS treatment effect HR/CI/p-values
        from the trial evidence text into a structured table.

        This SEPARATE call is the key anti-hallucination defence: a simple
        "copy numbers" task is far less prone to training-data fabrication than
        the full clinical-reasoning prompt.  The extracted table is then
        injected into Phase 1 as SYSTEM-VERIFIED authoritative data.
        """
        if not trial_analysis or len(trial_analysis.strip()) < 100:
            return ""

        retry_block = ""
        if retry_hint:
            retry_block = textwrap.dedent(f"""
            🔴🔴🔴 **【上一次提取被系统验证拒绝——请仔细排查以下常见错误】** 🔴🔴🔴
            你上一次的提取结果未通过系统验证。请逐一排查：
            {retry_hint}
            🔴 **请重新逐字从循证文本中提取 OS 治疗效应数据。不要重复上一次的错误。**
            """).strip()

        prompt = prompt_manager.get("treatment_phase0_os").format(
            retry_block=retry_block,
            trial_analysis=trial_analysis,
        )

        try:
            from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags, strip_llm_preamble
            response = await invoke_with_timeout_and_retry(
                self.report_model, prompt, timeout=120.0, max_retries=1
            )
            result = remove_think_tags(response.content).strip()
            result = strip_llm_preamble(result)
            if "|" in result and any(c.isdigit() for c in result):
                is_valid, issues = self._validate_os_extraction(result)
                if is_valid:
                    logger.info(
                        f"[TreatmentDecisionAgent] Phase 0: OS 数据提取完成 + 验证通过 "
                        f"({len(result)} 字符)"
                    )
                    return result
                else:
                    logger.warning(
                        f"[TreatmentDecisionAgent] Phase 0: 语义验证失败: {'; '.join(issues)}"
                    )
                    return ""  # Caller will retry with retry_hint
            else:
                logger.warning(
                    "[TreatmentDecisionAgent] Phase 0: 提取结果格式异常（无表格或无数值）"
                )
                return ""
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[TreatmentDecisionAgent] Phase 0: 提取失败 ({e})")
            return ""

    # -----------------------------------------------------------------
    # Phase 0 output validation — catch common extraction errors
    # -----------------------------------------------------------------
    @staticmethod
    def _validate_os_extraction(extraction_text: str) -> tuple:
        """
        Domain-general validation of Phase 0 extraction output.
        Checks apply to any cancer type and any set of trials.
        Returns (is_valid, list_of_issues).
        """
        issues = []
        lines = extraction_text.strip().split("\n")
        for line in lines:
            if not line.startswith("|") or "---" in line:
                continue
            if "（示例）" in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]
            if len(cells) < 4:
                continue

            # Skip header row
            if "试验" in cells[0] or cells[0] == "Trial":
                continue

            hr_str = cells[2] if len(cells) > 2 else ""
            ci_str = cells[3] if len(cells) > 3 else ""
            comparison = cells[1] if len(cells) > 1 else ""

            # Check if comparison looks like molecular subtyping (prognostic, not treatment effect)
            mol_patterns = [
                r"p53", r"POLE", r"MMR", r"MSI", r"HER2", r"ER\b", r"PR\b",
                r"mut\w* vs", r"wild.?type vs", r"突变", r"野生",
                r"亚型.*vs", r"molecular", r"分子亚型",
            ]
            is_molecular = False
            for pat in mol_patterns:
                if re.search(pat, comparison, re.IGNORECASE):
                    is_molecular = True
                    break
            if is_molecular:
                issues.append(f"疑似预后分层比较（非治疗效应）: {comparison}")
                continue

            # Validate HR numeric range
            try:
                hr_val = float(hr_str)
                if hr_val > 2.5:
                    issues.append(f"HR={hr_val} > 2.5，极可能为预后分层数据（非治疗效应）: {comparison}")
                elif hr_val < 0.15:
                    issues.append(f"HR={hr_val} < 0.15，数值异常偏低: {comparison}")
            except ValueError:
                if hr_str not in ("未提供", "", "-"):
                    issues.append(f"无法解析 HR 值: {hr_str}")

            # Validate CI format
            ci_match = re.match(r"([\d.]+)\s*[-–]\s*([\d.]+)", ci_str)
            if ci_match:
                lower = float(ci_match.group(1))
                upper = float(ci_match.group(2))
                if lower >= upper:
                    issues.append(f"CI 下界({lower}) >= 上界({upper}): {comparison}")

            # Validate p-value in [0, 1]
            p_cell = cells[4] if len(cells) > 4 else ""
            p_match = re.search(r"([\d.]+)", p_cell)
            if p_match:
                try:
                    p_val = float(p_match.group(1))
                    if p_val < 0 or p_val > 1:
                        issues.append(f"p值 {p_val} 不在 [0,1] 范围内: {comparison}")
                except ValueError:
                    pass

        return (len(issues) == 0, issues)

    # -----------------------------------------------------------------
    # Phase 1 output consistency check — 放疗结论 vs 最终核心方案 must agree
    # -----------------------------------------------------------------
    @staticmethod
    def _has_rt_keyword(text: str) -> bool:
        """Check if text contains RT keywords that are NOT negated.

        E.g. "不进行任何辅助放疗或化疗" contains "放疗" but it's negated
        → returns False.  "推荐盆腔EBRT" → returns True.
        """
        import re as _re
        rt_kw = ["EBRT", "VBT", "放疗", "\\bRT\\b", "IMRT", "VMAT"]
        negation_words = ["不", "无需", "免除", "无", "未", "非", "没有", "避免", "omit", "without", "no "]
        neg_pattern = "(?:" + "|".join(negation_words) + ")"
        for kw in rt_kw:
            for m in _re.finditer(kw, text, _re.IGNORECASE):
                # Look back up to 15 chars before this match for negation
                prefix = text[max(0, m.start() - 15):m.start()]
                if not _re.search(neg_pattern, prefix, _re.IGNORECASE):
                    return True
        return False

    # Regex to split plan text from trailing justification clauses.
    # Model sometimes writes "TC方案...共6周期。因缺乏支持放疗的证据..."
    # — the "放疗" in the justification is NOT a treatment component.
    _JUSTIFICATION_SPLIT = re.compile(
        r'[。；，]\s*(?=因|鉴于|由于|基于|经MDT|综合|结合|考虑到|综上|目前|现阶段|鉴于目前)'
    )

    @staticmethod
    def _check_decision_consistency(decision_text: str) -> str:
        """
        Check that 放疗结论 and 最终核心方案 in Phase 1 output don't contradict.
        Returns empty string if consistent, or error description if contradictory.
        Domain-general: only matches Chinese/English keywords, knows nothing
        about specific trials.
        """
        rt_decision = ""
        final_plan = ""

        for line in decision_text.split("\n"):
            if "放疗结论" in line:
                rt_decision = line.split("：", 1)[-1].split(":", 1)[-1].strip() if "：" in line or ":" in line else ""
            if "最终核心方案" in line:
                final_plan = line.split("：", 1)[-1].split(":", 1)[-1].strip() if "：" in line or ":" in line else ""

        if not rt_decision or not final_plan:
            return ""

        # ── Clean plan text: strip trailing justification clauses ──
        # Model sometimes appends reasoning to the plan line, e.g.:
        #   "TC方案...共6周期。因缺乏支持放疗...证据，经MDT讨论..."
        # The "放疗" in the justification triggers a false positive.  Only
        # the regimen part (before the first justification marker) matters.
        parts = TreatmentDecisionAgent._JUSTIFICATION_SPLIT.split(final_plan, maxsplit=1)
        plan_clean = parts[0].strip()

        rt_omit = any(kw in rt_decision for kw in ["免除放疗", "免除", "不推荐放疗", "omit", "不推荐"])
        plan_has_rt = TreatmentDecisionAgent._has_rt_keyword(plan_clean)

        if rt_omit and plan_has_rt:
            return f"放疗结论=免除, 但最终核心方案含放疗组分: {plan_clean[:80]}"

        rt_keep = any(kw in rt_decision for kw in ["保留EBRT", "保留放疗", "保留", "EBRT", "推荐放疗"])
        if rt_keep and not plan_has_rt:
            return f"放疗结论=保留放疗, 但最终核心方案缺少放疗组分: {plan_clean[:80]}"

        return ""

    # -----------------------------------------------------------------
    # Phase 1 output validation — check required fields are non-empty
    # -----------------------------------------------------------------
    _REQUIRED_DECISION_FIELDS: list[tuple[str, str]] = [
        ("最终核心方案", r'最终核心方案[：:]\s*(.+)'),
        ("放疗结论", r'放疗结论[：:]\s*(.+)'),
        ("PET-CT决策", r'PET-CT决策[：:]\s*(.+)'),
    ]

    @classmethod
    def _check_empty_decision_fields(cls, content: str) -> list[str]:
        """Return names of required decision fields whose values are empty/missing."""
        empty = []
        for field_name, pattern in cls._REQUIRED_DECISION_FIELDS:
            m = re.search(pattern, content, re.MULTILINE)
            if not m:
                empty.append(field_name)
                continue
            value = m.group(1).strip()
            # Common patterns that indicate an unfilled field
            if (not value or
                value in ("待定", "待确认", "无", "无信息", "暂无") or
                len(value) < 3):
                empty.append(field_name)
        return empty

    # =================================================================
    # Retry prompt builder — rebuilds prompt with targeted fix hints
    # =================================================================
    @staticmethod
    def _build_retry_prompt(
        base_prompt: str,
        error_type: str,
        error_detail: str,
    ) -> str:
        """
        Build a retry prompt by appending a targeted fix instruction to the
        clean base prompt.

        IMPORTANT: base_prompt is ALREADY the clean original (saved via
        ``base_prompt = prompt`` before any retry augmentations).  Do NOT
        strip "🔴🔴🔴" — the base prompt may legitimately contain that
        marker in other blocks (feedback_block, etc.), and stripping on
        its first occurrence would truncate 90 % of the instructions.
        """

        if error_type == "consistency":
            hint = textwrap.dedent(f"""
            🔴🔴🔴 **【上一轮输出存在内部矛盾——必须修正】** 🔴🔴🔴
            你的上一轮输出存在以下矛盾：
            {error_detail}

            🛑 **【放疗定义——VBT也是放疗】**：放疗包括EBRT（盆腔外照射）和VBT（阴道近距离放疗/后装治疗）。
               "免除放疗"= 既无EBRT也无VBT。若方案中有VBT，放疗结论**绝对不能**写"免除放疗"。

            🛑 **修正方法（按顺序执行——证据驱动，而非格式对齐）**：
            1. 重新审视上方循证证据中放疗相关试验的 OS 数据（重点关注差异组分为放疗的试验，如 GOG-0258）。
            2. 基于证据确定临床正确的放疗结论（四选一，互斥）：
               - `保留EBRT`：仅保留盆腔外照射
               - `保留EBRT+VBT`：EBRT + 阴道近距离放疗推量
               - `单纯VBT`：免除EBRT但保留阴道近距离放疗（这**不是**"免除放疗"）
               - `免除放疗`：**既无EBRT也无VBT**，患者不接受任何形式的放疗
            3. 放疗结论是决策根，最终核心方案是执行——方案必须逐词服从结论：
               - 放疗结论为"免除放疗" → 方案中逐词检查，确认无EBRT/VBT/放疗/后装/近距离字眼
               - 放疗结论为"单纯VBT" → 方案中必须出现VBT/近距离放疗，但**不得**出现EBRT
               - 放疗结论为"保留EBRT" → 方案中必须出现EBRT/盆腔放疗
               - 放疗结论为"保留EBRT+VBT" → 方案中必须同时出现EBRT和VBT
            🔴🔴🔴
            """).strip()
        elif error_type == "empty_fields":
            missing_list = "\n".join(
                f"  - {name}：必须填写具体内容，不得留空"
                for name in error_detail.split(", ")
            )
            hint = textwrap.dedent(f"""
            🔴🔴🔴 **【上一轮输出字段为空——必须填满以下字段】** 🔴🔴🔴
            你的上一轮输出中，以下必需字段的值为空或不完整：
            {missing_list}

            🛑 **填写指南**：
            - 最终核心方案：写完整的化疗方案（药物+剂量+周期数）。🛑 VBT/后装/近距离也是放疗——若方案含VBT，放疗结论不得写"免除放疗"。
            - 放疗结论：必须写以下四者之一（互斥）：
              `保留EBRT` / `保留EBRT+VBT` / `免除放疗`（=既无EBRT也无VBT） / `单纯VBT`（=免除EBRT但保留VBT，≠免除放疗）
            - PET-CT决策：必须写"追加PET-CT"或"不常规追加PET-CT"
            - 以上字段不得留空、不得写"待定"、不得省略——这是对患者出具的法律文书。
            🔴🔴🔴
            """).strip()
        elif error_type == "missing_markers":
            hint = textwrap.dedent(f"""
            🔴🔴🔴 **【上一轮输出缺少必需的决策标记——必须包含】** 🔴🔴🔴
            你的上一轮输出中缺少"### 方案决策"标题或"放疗结论"字段。
            请严格按以下格式输出（从"### 方案决策"开始）：
            ### 方案决策
            - 最终核心方案：[...]
            - 化疗依据：[...]
            - 放疗结论：[保留EBRT / 保留EBRT+VBT / 免除放疗 / 单纯VBT]
            - 放疗依据：[...]
            - PET-CT决策：[追加PET-CT / 不常规追加PET-CT]
            - PET-CT依据：[...]
            🔴🔴🔴
            """).strip()
        else:
            hint = ""

        return base_prompt + "\n\n" + hint if hint else base_prompt

    # =================================================================
    # Phase 1: Evidence analysis → structured decision summary
    # =================================================================
    async def _plan_treatment_decision(
        self,
        oncology_core: dict,
        comorbidity_count: int,
        guideline_section: str,
        trial_analysis: str,
        feedback_block: str,
        prelim_tx: str = "",
    ) -> str:
        """
        Focused prompt: analyse evidence and make decisions.
        Only receives guideline + trial evidence + concise structured patient data.
        Output is a structured decision summary consumed by Phase 2.

        prelim_tx: upstream preliminary plan from the MDT report — serves as the
        baseline anchor.  Evidence from guidelines and trials ADJUSTS this plan,
        it does not replace it from scratch.
        """

        # ── Build concise patient snapshot from structured JSON ──
        diagnosis = oncology_core.get("diagnosis_and_stage", "")
        pathology = oncology_core.get("pathology_and_molecular", "")
        surgery = oncology_core.get("surgery_type", "")

        comorbidity_note = ""
        if comorbidity_count >= 2:
            comorbidity_note = (
                f"该患者有 {comorbidity_count} 项慢性合并症 → "
                "触发法则一否决条件：未证实 OS 获益的追加治疗组分必须免除。\n"
            )

        # ── Detect pending molecular testing → inject foolproof instruction ──
        pending_note = ""
        pending_keywords = ["待回报", "未出", "检测中", "待确认", "结果未出", "待NGS", "NGS.*待",
                            "分子分型.*待", "已送检.*待", "待.*回报"]
        pathology_lower = pathology.lower() if pathology else ""
        diagnosis_lower = diagnosis.lower() if diagnosis else ""
        combined_text = f"{pathology} {diagnosis}"

        if any(
            (kw.startswith("待") and kw in combined_text) or
            (kw.startswith("NGS") and ("待" in pathology or "未" in pathology)) or
            (kw.startswith("分子") and "待" in combined_text) or
            ("待" in combined_text and "回报" in combined_text) or
            ("未出" in combined_text) or
            ("检测中" in combined_text)
            for kw in pending_keywords
        ):
            pending_note = textwrap.dedent("""
            🛑🛑🛑 **【系统强制指令——分子分型结果未出/待回报】** 🛑🛑🛑
            患者最终分子分型结果**尚未明确**（状态：待回报/未出/检测中）。
            当前的IHC结果仅为初筛，不能被当作最终分子分型使用（IHC p53异常 ≠ p53abn最终分型，POLEmut可覆盖p53abn的IHC表型）。

            你的决策必须遵守以下约束：
            ① **核心方案必须是基石方案**：只能根据已确凿的组织学类型（如浆液性癌）和FIGO分期推荐系统化疗等不依赖分子分型的方案。**绝对禁止**将依赖特定基因型的确诊分型（如p53abn/POLEmut/MMRd）的强化治疗（如因"p53abn"而追加放疗）直接写入核心方案。
            ② **分子分型相关的调整必须放入条件分支**：所有基于特定分子分型的治疗调整（如"若NGS确认为p53abn，则建议追加盆腔EBRT"），只能以条件句形式写入，不得作为确定性推荐。
            ③ **分子分型解读必须使用条件性措辞**：通篇不得出现"患者为p53abn亚型/POLEmut亚型"等确定性分型陈述。必须使用"若NGS确认…""待分子分型回报后…"等条件句式。
            ④ **放疗决策**：在分子分型未明确的情况下，除非有其他不依赖分子分型的OS证据支持放疗获益，否则暂不将依赖分子亚型的放疗方案作为确定性推荐。
            🛑🛑🛑
            """).strip()
            logger.info("[TreatmentDecisionAgent] 检测到分子分型结果待回报，注入防呆指令")

        # ── prelim_tx_block: intentionally disabled ──
        # The baseline anchor created an evidence-vs-anchor conflict in Phase 1:
        # "默认保留所有组分" vs "GOG-0258 shows no OS benefit for RT" →
        # the model oscillated between the two positions across retries.
        # Reverting to backup behaviour: evidence alone drives the decision.
        prelim_tx_block = ""

        # ── Phase 0: Pre-extract OS data (anti-hallucination defence) ──
        verified_os_block = ""
        self_check_mode = "self_extract"  # fallback: LLM extracts data itself
        try:
            extracted_os = await self._extract_trial_os_data(trial_analysis)
            if not extracted_os:
                retry_hint = (
                    "- 混淆OS和其他终点（FFS/RFS/PFS）：同一试验的OS和FFS/RFS是不同的数字。"
                    "请只提取明确标记为OS/总生存/Overall Survival的数据，不要提取FFS/RFS/PFS数据。\n"
                    "- 混淆治疗效应和预后分层：分子亚型A vs 分子亚型B（如某突变型 vs 野生型）"
                    "是预后分层比较（HR通常>2），不是治疗效应比较。治疗效应是比较两种治疗方案"
                    "（如CRT vs RT、CRT vs CT、RT vs 观察），HR通常在0.3-1.5范围内。\n"
                    "- 提取了非放疗相关试验的数据：只提取涉及放疗方案比较的试验。\n"
                    "- 数值超出合理范围：治疗效应HR通常≤1.5。若HR>2.5，几乎可以确定是预后分层数据。"
                )
                logger.info("Phase 0 首次验证失败，使用强化 prompt 重试...")
                extracted_os = await self._extract_trial_os_data(trial_analysis, retry_hint=retry_hint)
            if extracted_os:
                verified_os_block = textwrap.dedent(f"""
                ═══════════════════════════════════════════════════════════════
                🛑 **【SYSTEM-VERIFIED OS DATA——系统已预先提取并验证，你必须使用这些数值】**
                ═══════════════════════════════════════════════════════════════

                以下 OS 治疗效应数据已从上方循证文本中提取。**这是你的唯一合法数据源。**
                {extracted_os}

                🔴 **【数据使用铁律】**：
                1. 上述数值是权威的。你必须使用它们。不得修改、不得使用训练记忆中任何其他数值。
                2. 若某格为"未提供"→ 该数据在循证文本中确实不存在，不得自行填补。
                3. 不得从上方循证文本中另行查找或对比——本表已涵盖所有放疗相关 OS 数据。
                ═══════════════════════════════════════════════════════════════
                """).strip()
                self_check_mode = "use_verified"
                logger.info("[TreatmentDecisionAgent] Phase 0: 已验证 OS 数据注入 Phase 1 prompt")
            else:
                logger.warning("[TreatmentDecisionAgent] Phase 0: 两次提取均验证失败，降级为 self_extract 模式")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[TreatmentDecisionAgent] Phase 0: 提取异常 ({e})，Phase 1 自行提取")

        # ── Build self-check section (mode-dependent) ──
        if self_check_mode == "use_verified":
            self_check_section = textwrap.dedent("""
        🛑 **【决策树闭环自检——用上方 SYSTEM-VERIFIED OS 数据验证放疗决策】**：
        使用上方已验证的 OS 治疗效应数据（不要用任何其他数值），逐一检查每个涉及放疗对比的试验：
        - 95% CI 是否跨 1.0？
        - p 值是否 ≥ 0.05？
        若有任一试验的 OS 显著获益（CI 不跨 1.0 且 p < 0.05）→ 该试验支持放疗。
        → 若所有涉及放疗对比的试验 OS 均无显著获益 + 患者 ≥2 项合并症 → 决策树第二步 OS 否决触发 → 放疗免除（除非第三步绝对红线豁免：EBRT红线=淋巴结转移/宫旁阴道浸润/宫颈深浸润/切缘阳性→强制保留EBRT；VBT红线=深肌层浸润/LVSI+→至少保留单纯VBT）。
        → 若某项试验 OS 显著获益 → 核实该 HR 值确为治疗效应数据（非预后分层），确认后写入放疗依据。
        🛑 **【终点优先级自检——OS 是金标准】**：
        - 若某试验仅 FFS/RFS 改善但 OS 无获益 → **FFS 改善不可推翻 OS 否决**（终点优先级第二层）。
        - 若引用的 HR > 2.0 → 几乎肯定是预后分层 HR（亚型A vs 亚型B），不是治疗效应 HR。禁止写入方案依据。
        - 若某试验的差异组分不是放疗（如 PORTEC-3 差异组分=化疗）→ 其 OS 数据与放疗决策无关（归因红线规则二）。
        放疗决策字段必须与此自检结论严格一致。
        """).strip()
        else:
            self_check_section = textwrap.dedent("""
        ### 关键试验OS数据速查
        🛑 **【防幻觉铁律——逐字提取，禁止凭空生成】**：下方每个数值必须从上方的"临床试验深度解析"中逐字复制粘贴。若上方循证数据中确实未提供该数值 → 填"未提供"。**绝对禁止**凭记忆推理、混淆不同试验、或把RFS/PFS数据填入OS栏位。
        - 列出上方出现的每一个涉及放疗对比的试验，写出其 OS 治疗效应数据（HR, 95% CI, p 值）。若某试验未报告 OS 治疗效应，写"该试验未提供 OS 治疗效应数据"。
        ---

        🛑 **【决策树闭环自检——用上方速查数据验证放疗决策】**：
        逐一检查你提取的每个试验的 OS 数据：
        - 95% CI 是否跨 1.0？
        - p 值是否 ≥ 0.05？
        若有任一试验的 OS 显著获益（CI 不跨 1.0 且 p < 0.05）→ 该试验支持放疗。否则所有试验均不支持。
        → 若所有涉及放疗对比的试验 OS 均无显著获益 + 患者 ≥2 项合并症 → 决策树第二步 OS 否决触发 → 放疗免除（除非第三步绝对红线豁免：EBRT红线=淋巴结转移/宫旁阴道浸润/宫颈深浸润/切缘阳性→强制保留EBRT；VBT红线=深肌层浸润/LVSI+→至少保留单纯VBT）。
        → 若某项试验 OS 显著获益 → 回到该试验上方原文逐字核实该 HR 值是否确为治疗效应数据、是否确实来自该试验。核实确认后再写入放疗依据。
        放疗决策字段必须与此自检结论严格一致。
        🛑 **【终点优先级自检——OS 是金标准】**：
        - 若某试验仅 FFS/RFS 改善但 OS 无获益 → **FFS 改善不可推翻 OS 否决**（终点优先级第二层）。
        - 若引用的 HR > 2.0 → 几乎肯定是预后分层 HR（亚型A vs 亚型B），不是治疗效应 HR。禁止写入方案依据。
        - 若某试验的差异组分不是放疗（如 PORTEC-3 差异组分=化疗）→ 其 OS 数据与放疗决策无关（归因红线规则二）。
        🔴 所有数值从上方循证数据逐字提取。中文描述，数值保留原文。
        """).strip()

        prompt = prompt_manager.get("treatment_phase1_decision").format(
            prelim_tx_block=prelim_tx_block,
            verified_os_block=verified_os_block,
            feedback_block=feedback_block,
            pending_note=pending_note,
            self_check_section=self_check_section,
            diagnosis=diagnosis,
            pathology=pathology,
            comorbidity_count=comorbidity_count,
            comorbidity_note=comorbidity_note,
            guideline_section=guideline_section,
            trial_analysis=trial_analysis,
        )

        # Save base prompt to rebuild clean retry prompts (no accumulation)
        base_prompt = prompt

        for attempt in range(3):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=600.0, max_retries=2
                )
                content = remove_think_tags(response.content).strip()
                content = strip_llm_preamble(content)
                # Validate: must contain the decision markers
                if "方案决策" in content and "放疗结论" in content:
                    consistency_issue = self._check_decision_consistency(content)
                    if consistency_issue:
                        logger.warning(
                            f"[TreatmentDecisionAgent] Phase 1 决策摘要内部矛盾: "
                            f"{consistency_issue}，重试中... ({attempt+1})"
                        )
                        if attempt < 2:
                            prompt = self._build_retry_prompt(
                                base_prompt, "consistency", consistency_issue,
                            )
                        continue
                    # Validate: all required fields must have non-empty values
                    empty_fields = self._check_empty_decision_fields(content)
                    if empty_fields:
                        logger.warning(
                            f"[TreatmentDecisionAgent] Phase 1 决策字段为空: "
                            f"{', '.join(empty_fields)}，重试中... ({attempt+1})"
                        )
                        if attempt < 2:
                            prompt = self._build_retry_prompt(
                                base_prompt, "empty_fields",
                                ", ".join(empty_fields),
                            )
                        continue
                    return content
                logger.warning(
                    f"[TreatmentDecisionAgent] Phase 1 输出缺少决策标记，重试中... ({attempt+1})"
                )
                if attempt < 2:
                    prompt = self._build_retry_prompt(
                        base_prompt, "missing_markers", "",
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[TreatmentDecisionAgent] Phase 1 报错: {e}")
                if attempt == 2:
                    return "### 方案决策\n（Phase 1 生成失败）\n"

        return "### 方案决策\n（Phase 1 生成失败）\n"

    # =================================================================
    # Phase 2: Format decision summary → final output
    # =================================================================
    async def _write_treatment_plan(
        self,
        oncology_core: dict,
        proposed_plan: dict,
        decision_summary: str,
        ref_map_str: str,
    ) -> str:
        """
        Pure formatting prompt.  The clinical decisions have been made by
        Phase 1 — this phase only translates them into the required output
        format.  It must NOT modify, override, or re-litigate decisions.
        Receives structured patient data (oncology_core, proposed_plan) for
        context, not the long narrative patient_summary.
        """

        ref_note = ""
        if ref_map_str:
            ref_note = f"""
        【可用来源引用映射】（最终报告中只允许使用以下 [^^n] 编号）：
        {ref_map_str}
        """

        # ── Build concise patient context from structured JSON ──
        diagnosis = oncology_core.get("diagnosis_and_stage", "")
        pathology = oncology_core.get("pathology_and_molecular", "")
        surgery = oncology_core.get("surgery_type", "")
        prelim_tx = proposed_plan.get("main_oncology_treatment", "")

        prompt = prompt_manager.get("treatment_phase2_plan").format(
            ref_note=ref_note,
            diagnosis=diagnosis,
            pathology=pathology,
            surgery=surgery,
            prelim_tx=prelim_tx,
            decision_summary=decision_summary,
        )

        for attempt in range(2):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=300.0, max_retries=1
                )
                content = remove_think_tags(response.content).strip()
                content = strip_llm_preamble(content)
                # Phase 2 safety: must contain the format marker AND
                # must NOT contain meta-commentary about empty fields
                if "肿瘤专科最终方案" in content:
                    _leaked_meta = re.search(
                        r'(?:字段为空|无信息可供输出|无法进行具体评估|无具体决策依据)',
                        content,
                    )
                    if _leaked_meta:
                        logger.warning(
                            f"[TreatmentDecisionAgent] Phase 2 输出含元评注泄漏"
                            f" '{_leaked_meta.group()}'，重试中... ({attempt+1})"
                        )
                        if attempt < 1:
                            # Inject fallback: tell LLM to use prelim_tx, not self-generate
                            fallback_instruction = textwrap.dedent(f"""
                            🔴🔴🔴 **【上一轮输出含元评注泄漏——强制回退】** 🔴🔴🔴
                            你的上一轮输出中出现了"字段为空""无信息可供输出"等元评注，
                            这违反了核心约束 5b。

                            🛑 **你必须回退到上游初步方案中的对应内容来填补缺失字段**，
                            不得凭空编造。上游初步方案如下：
                            ---
                            {prelim_tx if prelim_tx else "（上游初步方案未提供——此时可基于患者背景上下文合理填补，但仍禁止输出元评注）"}
                            ---
                            🔴🔴🔴
                            """).strip()
                            prompt = prompt + "\n\n" + fallback_instruction
                            continue
                    else:
                        return content
                logger.warning(
                    f"[TreatmentDecisionAgent] Phase 2 输出缺少格式标记，重试中... ({attempt+1})"
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[TreatmentDecisionAgent] Phase 2 报错: {e}")
                if attempt == 1:
                    break

        return "1、 **肿瘤专科最终方案**\n（格式化失败）\n\n2、 **分子分型与复发风险解读**\n（格式化失败）"
