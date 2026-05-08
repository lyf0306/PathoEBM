import logging
import re
import textwrap

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags

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

        # =================================================================
        # Phase 1: Evidence analysis → structured decision summary
        # (only guideline + trial evidence + concise structured patient data)
        # =================================================================
        logger.info("[TreatmentDecisionAgent] Phase 1: 循证分析与决策...")
        decision_summary = await self._plan_treatment_decision(
            oncology_core=oncology_core,
            comorbidity_count=comorbidity_count,
            guideline_section=guideline_section,
            trial_analysis=trial_analysis,
            feedback_block=feedback_block,
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

        prompt = textwrap.dedent(f"""
        你是一名数据提取专员。你唯一的任务是从临床试验证据文本中**逐字复制** OS（总生存期）治疗效应数据。
        禁止做任何推理、判断或临床解读。

        {retry_block}
        【循证文本】：
        ---
        {trial_analysis}
        ---

        【提取任务】：
        对循证文本中每一个涉及放疗方案比较的试验，提取其 OS 治疗效应数据。

        ✅ **需要提取的**：
        - 仅 OS（Overall Survival / 总生存期）的治疗效应数据
        - 仅治疗效应 HR（比较两种治疗方案，如 CRT vs RT、CRT vs CT、RT vs 观察）
        - HR 值、95% CI 下界、95% CI 上界、p 值

        ❌ **绝对不要提取的**：
        - RFS/PFS/FFS 数据（即使有 HR 值也不提取）
        - 预后分层比较（如分子亚型A vs 分子亚型B——这是预后分层，不是治疗效应）
        - 单一治疗组内的亚组分析
        - 非放疗相关试验的数据

        🔴 **【逐字复制铁律】**：
        1. 数字必须与原文字逐字一致。禁止四舍五入、禁止调整、禁止凭记忆填补。
        2. 若原文中某数值确实未提供 → 该格填"未提供"。
        3. 若原文中某试验无放疗比较 → 跳过该试验。
        4. 若某试验有多个 OS 时间点，优先提取数据最完整的那一组。

        🔴 **【OS vs 其他终点判定——每个数字必须回答】**：
        - 原文标注的是"OS / 总生存 / Overall Survival"还是"FFS / RFS / PFS"？
          只有明确标为 OS 的才是 OS。同一试验的 OS 和 FFS/RFS 是不同的数字。
        - 原文是比较"方案A vs 方案B"（治疗效应）还是比较"亚型X vs 亚型Y"（预后分层）？
          治疗效应回答"治疗X是否比治疗Y更好"，预后分层回答"某亚型预后是否更差"。
          只有治疗效应 HR 才能用于治疗决策。

        🔴 **【数值范围常识——辅助判断数据类别】**：
        - 治疗效应 HR 通常在 0.3-1.5 范围内。
        - 若 HR > 2.5，极可能为预后分层比较（某亚型 vs 参照亚型），非治疗效应。
        - 若原文中某 HR 值附近同时出现 RFS/PFS/FFS 字样，该 HR 极可能不是 OS 数据。

        输出格式（Markdown 表格，不要任何前缀/后缀文字）：
        | 试验 | 比较 | OS HR | 95% CI | p值 |
        |------|------|-------|--------|-----|
        | （示例）试验名 | （示例）方案A vs 方案B | （示例）0.80 | （示例）0.60-1.05 | （示例）0.12 |
        """)

        try:
            from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
            response = await invoke_with_timeout_and_retry(
                self.report_model, prompt, timeout=120.0, max_retries=1
            )
            result = remove_think_tags(response.content).strip()
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

        rt_omit = any(kw in rt_decision for kw in ["免除放疗", "免除", "不推荐放疗", "omit", "不推荐"])
        plan_has_rt = any(kw in final_plan.upper() for kw in ["EBRT", "VBT", "放疗", "RT", "IMRT", "VMAT"])

        if rt_omit and plan_has_rt:
            return f"放疗结论=免除, 但最终核心方案含放疗组分: {final_plan[:80]}"

        rt_keep = any(kw in rt_decision for kw in ["保留EBRT", "保留放疗", "保留", "EBRT", "推荐放疗"])
        if rt_keep and not plan_has_rt:
            return f"放疗结论=保留放疗, 但最终核心方案缺少放疗组分: {final_plan[:80]}"

        return ""

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
    ) -> str:
        """
        Focused prompt: analyse evidence and make decisions.
        Only receives guideline + trial evidence + concise structured patient data.
        Output is a structured decision summary consumed by Phase 2.
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
        🔴 所有数值从上方循证数据逐字提取。中文描述，数值保留原文。
        """).strip()

        prompt = textwrap.dedent(f"""
        你是一名妇科肿瘤 MDT 首席专家。你的任务是对输入的循证数据进行严密的推演，并产出一份决策摘要。

        {verified_os_block}
        {feedback_block}
        {pending_note}

        【患者基线数据】：
        - 分期与组织学：{diagnosis}
        - 病理与分子：{pathology}
        - 合并症：{comorbidity_count} 项 （≥2项即为多合并症，耐受度下降）

        【核心循证文本】：
        {guideline_section}
        {trial_analysis}

        ═══════════════════════════════════════════════════════════════
        🛑 **【强制思维链（CoT）引导】（你必须在 <think> 标签内严格按此模板思考）**
        ═══════════════════════════════════════════════════════════════
        在给出最终决策前，请在 <think> 标签内按以下4个步骤进行逐层推演：

        Step 1: 提取病理与基线红线
        - 肿瘤分期：___ (是否为IIIC期/明确淋巴结转移/切缘阳性？这决定了是否触发局部放疗绝对红线)
        - 局部高危因素：___ (是否有深肌层浸润≥1/2 或 LVSI+？这决定了阴道残端复发风险)
        - 全身毒性耐受度：合并症共 ___ 项 (是否≥2项？)

        Step 2: 循证组分剥离 (OS归因——极其重要！)
        对循证数据中出现的**每一项临床试验**，执行以下"差异组分归因"分析：
        a) 列出该试验的 方案A（全部治疗组分）vs 方案B（全部治疗组分）
        b) 标记差异组分（A有B无 / B有A无）和共有组分（A和B均有）
        c) 差异组分的 OS 是否有统计学显著差异？（HR跨1或p≥0.05 → 无显著获益）
        d) 🛑 核心铁律：OS获益只能归因于【差异组分】。若某组分在两组中均存在，该试验的OS数据绝对不能用来论证该共有组分的疗效。

        方法示例（仅用于说明归因逻辑，以实际循证数据为准）：
        - 若某试验比较 (化疗+放疗) vs (放疗)，两组都有【放疗】，差异组分是【化疗】
          → OS获益只能证明【化疗】有效，绝对不能用来证明【放疗】有效。
        - 若某试验比较 (化疗+放疗) vs (单纯化疗)，差异组分是【放疗】
          → 这才是决定是否加用【放疗】的真正证据。
        - 若某试验比较 (放疗) vs (观察)，差异组分是【放疗】
          → OS获益可证明【放疗】有效。

        对所有试验完成归因后，汇总结论：
        - 放疗（EBRT/VBT）究竟是否有专属于它的 OS 获益证据？（即是否有差异组分为放疗的试验显示OS显著获益）

        Step 3: 规则综合推演
        - 检查解剖红线：患者是IIIC期或有淋巴结转移吗？ -> 若是，必须无视合并症，强制保留盆腔EBRT。
        - 若未触发解剖红线：放疗是否有OS获益？合并症是否≥2？ -> 若无专属于放疗的OS获益，且合并症≥2，为防毒性叠加，强制免除放疗。
        - 若最终决定保留EBRT：是否有深肌层浸润或LVSI+？ -> 若是，必须在EBRT基础上加用 VBT推量。

        Step 4: PET-CT指征判定
        - 满足以下任一即追加：1.高危组织学(浆液/透明等)；2.FIGO IV期(注意：普通的III期不追加)；3.确诊p53abn。患者满足哪几条？结论是什么？

        ═══════════════════════════════════════════════════════════════
        【输出格式】（<think>结束后，按以下格式输出结果）
        ═══════════════════════════════════════════════════════════════
        ### 方案决策
        - 最终核心方案：[写明具体方案。如保留了放疗，写"TC方案(...,共6周期)+盆腔EBRT(或再加VBT推量)"。若放疗被免除，则仅写"TC方案(...,共6周期)"，绝对禁止写出放疗字眼。]
        - 化疗依据：[1-2句，引用具体试验的疗效数据]
        - 放疗结论：[保留EBRT / 保留EBRT+VBT / 免除放疗 / 单纯VBT]
        - 放疗依据：[必须与放疗结论完全对应。指出具体试验（如GOG-0258）的证据，并结合患者分期/合并症情况说明理由]
        - PET-CT决策：[追加PET-CT / 不常规追加PET-CT]
        - PET-CT依据：[列出满足的指征，或说明不满足指征]

        ### 分子分型解读要点
        - NGS状态：[待回报 / 已确认]
        - IHC结果：[p53突变型模式 / 等]
        - 措辞约束：[若待回报，写"条件性措辞——禁止确定性断言"]
        - 复发风险判断：[1-2句风险评估]
        ---
        """)

        for attempt in range(3):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=600.0, max_retries=2
                )
                content = remove_think_tags(response.content).strip()
                # Validate: must contain the decision markers
                if "方案决策" in content and "放疗结论" in content:
                    consistency_issue = self._check_decision_consistency(content)
                    if consistency_issue:
                        logger.warning(
                            f"[TreatmentDecisionAgent] Phase 1 决策摘要内部矛盾: "
                            f"{consistency_issue}，重试中... ({attempt+1})"
                        )
                        continue
                    return content
                logger.warning(
                    f"[TreatmentDecisionAgent] Phase 1 输出缺少决策标记，重试中... ({attempt+1})"
                )
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

        prompt = textwrap.dedent(f"""
        你是一名妇科肿瘤 MDT 报告撰写专家。
        你收到了一份已经由首席专家做出的**循证决策摘要**。
        你的唯一任务是将该摘要翻译成规定的输出格式。

        🛑🛑🛑 **【核心约束——违反即不合格】**
        1. 你**不得修改、推翻或质疑**决策摘要中的任何决定。
        2. 决策摘要中写"免除放疗"，你就写"免除放疗"。不得写成"建议放疗"。
        3. 决策摘要中写"追加PET-CT"，你就在疗效评估规划中列入PET-CT。
        4. 决策摘要中的 OS HR 值、95% CI、p 值必须原样保留，不得改动数字。
        5. 决策摘要中写"条件性措辞——禁止确定性断言"，你就通篇不得出现
           "患者为p53abn"等确定性分型陈述。
        6. 🛑 **【禁止泄露结构化标签——零容忍】**：
           决策摘要中的以下标签是给撰写系统的内部元指令，**绝对禁止**出现在
           最终报告中：
           - "NGS状态" / "IHC结果" / "措辞约束" / "复发风险判断"
           - "条件性措辞——"（该短语禁止输出，只需遵守其约束）
           你必须将它们翻译成自然医学语言，例如：
           ❌ "NGS状态：待回报" → ✅ "NGS分子分型检测结果待回报"
           ❌ "IHC结果：p53突变型模式" → ✅ "IHC检测提示p53突变型模式"
           ❌ "措辞约束：若NGS确认为p53abn…" → ✅ "若NGS确认为p53abn…"
             （"措辞约束"四个字消失，约束内容作为自然语句输出）
           ❌ "复发风险判断：基于当前…" → ✅ 直接写成自然段落

        {ref_note}
        ─────────────────────────────────────────────
        【患者背景（结构化数据，供你理解上下文，不要从中做决策）】
        - 分期与组织学：{diagnosis}
        - 病理与分子：{pathology}
        - 手术方式：{surgery}
        - 上游初步方案：{prelim_tx}

        ─────────────────────────────────────────────
        🛑 **【已做出的循证决策——你必须照此撰写，不可偏离】**
        {decision_summary}
        ─────────────────────────────────────────────

        【输出格式——严格按此结构输出】：

        1、 **肿瘤专科最终方案**：
           （1）**核心方案**：取决策摘要中"最终核心方案"字段**原样**写入，一字不改。
               🛑 若决策摘要"最终核心方案"写的是"TC方案（卡铂 AUC5+紫杉醇 175mg/m²，Q3W，共6周期）+ 盆腔EBRT"，
               你就逐字写"TC方案（卡铂 AUC5+紫杉醇 175mg/m²，Q3W，共6周期）+ 盆腔EBRT"。
               禁止简化为"化疗+放疗"、"TC方案+盆腔EBRT"等缩略形式。禁止因字数多而删减药物名、剂量、周期数。
               核心方案 = 该患者推荐的全部根治性治疗，一个条目写完。
           （2）**方案依据**：概述指南推荐，**必须引用决策摘要中的具体 OS HR 值及 95% CI**。
               OS 为主，RFS/PFS 为辅。OS 显著 → 两者并陈；OS 不显著 → 先写明
               "OS无显著差异（HR=X.XX, 95%CI X.XX-X.XX）"，再引 RFS/PFS 作为次要参考。
               🛑 禁止：不提 OS 只提 RFS/PFS。
               🛑 某试验只能论证其"差异组分"的获益。若两组共有某治疗，该试验的OS数据不得用于该共有组分的论证。
           （3）**放疗评估**：严格遵循决策摘要的"放疗结论"。
               🛑 **【黑话消除指令——绝对禁止出现AI推理术语】**
               你必须将决策逻辑翻译为临床医生对患者解释的自然专业表达。绝对禁止输出"触发规则C"、"根据法则一"、"OS否决"等字眼。
               ❌ 错误示例："触发规则C，有深肌层浸润和LVSI+，联合VBT推量。"
               ✅ 正确示例："考虑到患者存在深肌层浸润及脉管癌栓（LVSI+），阴道残端复发风险较高，建议在盆腔外照射基础上联合阴道近距离放疗（VBT）推量。"
               ❌ 错误示例："因为合并症≥2且放疗OS不显著触发免除放疗。"
               ✅ 正确示例："鉴于GOG-0258等试验未证实追加放疗的总生存获益，且患者合并多项内科基础疾病，为避免毒性叠加，暂不常规推荐盆腔放疗。"
           （4）**疗效评估规划**：严格根据决策摘要中的[PET-CT决策]，原样复制以下句式（一字不改，不可乱加前缀）：
               - 决策为"追加PET-CT" → 输出："建议基线完善盆腔增强MRI、上腹部增强CT、两肺平扫CT，并追加全身PET-CT评估。完成治疗后3个月复查盆腔增强MRI及胸腹CT。"
               - 决策为"不常规追加" → 输出："建议基线完善盆腔增强MRI、上腹部增强CT、两肺平扫CT评估。完成治疗后3个月复查盆腔增强MRI及胸腹CT。"

        2、 **分子分型与复发风险解读**：
           NGS 待回报→条件性措辞（"若NGS确认为…""待分子分型回报后…"），不得出现确定性断言。
           NGS 已确认→直接基于分型解读。

        🛑 **【输出边界——禁止越界生成】**
        你的输出仅限于上述 1、2 两项。以下内容**绝对禁止**出现：
        - 慢性合并症管理、治疗毒性管理 → 由 MDTReportAgent 负责
        - 随访计划 → 由 FollowupAgent 负责
        - 健康教育、预后评估、其他注意事项 → 由其他 Agent 负责
        输出完第 2 项后立即停止，不要写第 3、4、5...项。

        🛑 **【方案依据自洽性——输出前逐项检查】**：
        决策摘要的"方案决策"对每个治疗组分做出了取舍。方案依据必须与之一致：
        - 某治疗**保留** → 引用支持该获益的 OS 试验数据
        - 某治疗**免除** → 说明免除理由（OS无获益 / 毒性负担 / 合并症），**禁止**引用支持该治疗的数据
        - 某治疗**降阶梯** → 说明降阶梯理由及替代方案
        - 🛑 禁止"虽然OS不显著，但PFS/RFS获益仍具有临床意义……"等复活被否决组分的措辞
        - 🛑 **【"尽管/虽然...但..."句型扫描】**：输出前逐句检查"尽管""虽然""但""然而""不过""考虑到"——若出现在 OS 不显著的上下文后，即为反合理化，必须删除重写。
        - 🛑 禁止在方案依据中论证某治疗获益但方案决策却写"免除" → 逻辑断裂
        - 🛑 **【VBT自洽检查】**：若决策树保留EBRT且患者存在深肌层浸润或LVSI+，方案依据中必须解释为何联合VBT推量。若决策树第二步已免除放疗（OS无获益+合并症），则方案依据中不得出现VBT——放疗已被整体免除，不存在VBT议题。禁止方案依据只论证EBRT而遗漏VBT（当VBT适用时）。

        🔴 **【中文输出】**：所有描述性文字使用中文，数值和统计量保留原文。
        💡 决策摘要已做好，直接翻译成格式即可，不要重新分析。
        """)

        for attempt in range(2):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=300.0, max_retries=1
                )
                content = remove_think_tags(response.content).strip()
                if "肿瘤专科最终方案" in content:
                    return content
                logger.warning(
                    f"[TreatmentDecisionAgent] Phase 2 输出缺少格式标记，重试中... ({attempt+1})"
                )
            except Exception as e:
                logger.error(f"[TreatmentDecisionAgent] Phase 2 报错: {e}")
                if attempt == 1:
                    break

        return "1、 **肿瘤专科最终方案**\n（格式化失败）\n\n2、 **分子分型与复发风险解读**\n（格式化失败）"
