import logging
import re
import textwrap

from ..utilties.patient_state import classify_surgery, build_hpv_followup_rules
from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code-level post-validation: verify critical items appear in LLM output
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r'[一-鿿]')

# Common pathogen Latin abbreviations used in clinical shorthand.
# When a critical item's CJK bigrams don't match the output, we also
# check whether its Latin alias appears (e.g. LLM writes "HP(+)" for
# "幽门螺杆菌感染").  Order matters: earlier keys take priority.
_LATIN_ALIAS_MAP = [
    ("幽门螺杆菌", "HP"),
    ("人乳头瘤病毒", "HPV"),
    ("乙肝", "HBV"),
    ("乙型肝炎", "HBV"),
    ("结核", "TB"),
    ("丙肝", "HCV"),
    ("丙型肝炎", "HCV"),
]

# ---------------------------------------------------------------------------
# Department keyword mapping for comorbidity pre-grouping.
# Each comorbidity item is matched against these keyword lists; the
# department with the most keyword hits wins.  Order is significant:
# when two departments tie, the earlier one takes priority.
# ---------------------------------------------------------------------------
_DEPT_KEYWORDS = [
    ("心内科", [
        "高血压", "冠心病", "冠状动脉", "心绞痛", "心肌梗死", "心梗",
        "心力衰竭", "心衰", "心律失常", "房颤", "心房颤动", "室性早搏",
        "室早", "支架", "搭桥", "主动脉", "动脉硬化", "动脉粥样硬化",
        "血管", "高脂血症", "高血脂", "肥厚型心肌病", "心脏", "心血管",
        "冠脉", "PCI", "CABG", "心律", "心肌",
    ]),
    ("内分泌科", [
        "糖尿病", "血糖", "糖耐量", "甲减", "甲亢", "甲状腺功能减退",
        "甲状腺功能亢进", "桥本", "代谢综合征", "肥胖", "高尿酸",
        "痛风", "骨质疏松", "内分泌",
    ]),
    ("呼吸科", [
        "慢性阻塞性肺", "慢阻肺", "COPD", "哮喘", "支气管扩张",
        "肺气肿", "肺纤维化", "间质性肺炎", "肺动脉高压", "肺功能",
    ]),
    ("消化科", [
        "慢性胃炎", "胃溃疡", "十二指肠", "反流性食管炎", "食管",
        "脂肪肝", "肝硬化", "肝炎", "胆囊", "胆石", "胰腺",
        "肠息肉", "溃疡性结肠炎", "克罗恩", "消化",
    ]),
    ("肾内科", [
        "慢性肾脏病", "慢性肾病", "CKD", "肾功能不全", "肾衰竭",
        "肌酐", "尿蛋白", "肾小球", "肾病综合征", "透析", "肾脏",
    ]),
    ("神经内科", [
        "脑梗死", "脑梗", "脑出血", "卒中", "中风", "TIA",
        "短暂性脑缺血", "帕金森", "癫痫", "阿尔茨海默", "痴呆",
        "偏头痛", "周围神经", "脑血管",
    ]),
    ("风湿免疫科", [
        "系统性红斑狼疮", "SLE", "类风湿", "干燥综合征", "强直性脊柱炎",
        "硬皮病", "皮肌炎", "血管炎", "白塞", "风湿",
    ]),
    ("血液科", [
        "贫血", "缺铁", "血小板减少", "白细胞减少", "凝血", "血友病",
        "骨髓",
    ]),
    ("骨科", [
        "骨关节炎", "椎间盘", "腰椎", "颈椎", "骨折", "骨刺",
        "半月板", "关节",
    ]),
    ("泌尿科", [
        "前列腺", "膀胱", "尿道", "肾结石", "输尿管", "尿潴留",
        "泌尿", "BPH",
    ]),
]


def _classify_item_department(item: str) -> str:
    """Return the best-matching department for a single comorbidity item."""
    best_dept = None
    best_score = 0
    item_lower = item.lower()
    for dept, keywords in _DEPT_KEYWORDS:
        score = sum(1 for kw in keywords if kw.lower() in item_lower)
        if score > best_score:
            best_score = score
            best_dept = dept
    return best_dept or "其他"


def _group_by_department(items: list) -> dict:
    """Group comorbidity items by inferred department using keyword matching.

    Returns a dict mapping department name → list of items.
    Items that don't match any department are placed under "其他".
    """
    if not items:
        return {}
    groups: dict[str, list] = {}
    for item in items:
        dept = _classify_item_department(item)
        groups.setdefault(dept, []).append(item)
    return groups


def _get_cjk_bigrams(text: str) -> set:
    """Extract all adjacent CJK character pairs from text."""
    chars = _CJK_RE.findall(text)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def _check_critical_coverage(critical_items: list, output: str) -> list:
    """
    Return critical items not adequately covered in the output.

    Uses CJK bigram overlap with a ≥40% threshold.  Falls back to Latin
    alias matching for pathogens commonly abbreviated in clinical text
    (e.g. HP for 幽门螺杆菌, HPV for 人乳头瘤病毒).
    """
    output_bigrams = _get_cjk_bigrams(output)
    output_upper = output.upper()
    missing = []
    for item in critical_items:
        item_bigrams = _get_cjk_bigrams(item)
        if not item_bigrams:
            # Pure ASCII / numeric item — literal substring check
            if item.upper() not in output_upper:
                missing.append(item)
            continue
        overlap = item_bigrams & output_bigrams
        threshold = max(1, len(item_bigrams) * 0.4)
        if len(overlap) >= threshold:
            continue  # sufficient bigram overlap
        # Fallback: Latin alias check
        latin_hit = False
        for cn_part, latin in _LATIN_ALIAS_MAP:
            if cn_part in item and latin.upper() in output_upper:
                latin_hit = True
                break
        if latin_hit:
            continue
        logger.warning(
            f"[MDTReportAgent] 致命红线覆盖不足: '{item}' "
            f"({len(overlap)}/{len(item_bigrams)} bigrams matched, "
            f"need ≥{threshold:.0f})"
        )
        missing.append(item)
    return missing


# ---------------------------------------------------------------------------
# Removed-organ noise filter: when organs have been surgically excised, any
# benign findings on those organs have no management target — strip them
# before they reach the LLM and also post-hoc from the LLM output.
# ---------------------------------------------------------------------------

# Patterns for benign findings on removable gynaecological organs.
# These are only filtered when the patient has had the corresponding surgery.
_REMOVED_ORGAN_BENIGN_PATTERNS = [
    # Uterus/cervix — irrelevant after hysterectomy
    r'子宫(多发)?(小)?肌瘤',
    r'子宫肌(壁间|层)(回声|病变)',
    r'子宫腺肌',
    r'子宫内膜息肉',
    r'宫腔(内)?(异常|占位|回声|分离)',
    r'宫颈纳(氏|囊|特)(囊肿)?',
    r'宫颈(多发)?囊肿',
    r'宫颈息肉',
    r'宫颈肥大',
    r'宫颈糜烂',
    r'宫颈潴留(囊肿)?',
    r'宫颈(多发)?(小)?纳囊',
    # Ovary / tube / adnexa — irrelevant after BSO
    r'卵巢(多发)?(小)?囊肿',
    r'卵巢囊性(回声|病变|结构)',
    r'卵巢周围(炎|粘连)',
    r'输卵管积水',
    r'输卵管积液',
    r'输卵管囊肿',
    r'输卵管增粗',
    r'附件(区)?囊肿',
    r'附件区囊性',
]


def _filter_removed_organ_items(items: list, surgery_type: str) -> list:
    """Remove items referencing benign findings on surgically removed organs."""
    if not items or not surgery_type:
        return items

    flags = classify_surgery(surgery_type)
    if not flags["is_hysterectomy"] and not flags["is_bso"]:
        return items

    filtered = []
    for item in items:
        match = None
        for pat in _REMOVED_ORGAN_BENIGN_PATTERNS:
            m = re.search(pat, item)
            if m:
                match = m.group(0)
                break
        if match:
            logger.info(
                "[MDTReportAgent] 过滤已切除器官的良性发现: '%s' (匹配: '%s')",
                item, match,
            )
        else:
            filtered.append(item)

    dropped = len(items) - len(filtered)
    if dropped:
        logger.info(
            "[MDTReportAgent] 已切除器官过滤器: 移除 %d/%d 条",
            dropped, len(items),
        )
    return filtered


class MDTReportAgent:
    """
    Agent 2: Endometrial cancer MDT chief writer.

    Focused responsibility:
      - 病情分析 → built from structured_task data by code (no LLM)
      - 合并症管理 → LLM generates comorbidity management + incidental findings
      - Final report → assembled by code from all parts

    TreatmentDecisionAgent (Agent 2a) handles 主要方案 (肿瘤专科最终方案 +
    分子分型与复发风险解读) separately, applying the clinical trade-off rules
    (OS veto, barrel effect, tumor biology orientation).

    This split ensures each agent focuses on its core cognitive task.
    """
    def __init__(self, report_model, treatment_context: str, structured_task: dict, context_bus=None):
        self.report_model = report_model
        self.treatment_context = treatment_context
        self.structured_task = structured_task
        self.context_bus = context_bus

    # =================================================================
    # Section 1: 病历及病理摘要 (code-built, no LLM)
    # =================================================================
    def _build_patient_summary(self) -> str:
        """
        Build 病历及病理摘要 entirely from structured_task data.
        No LLM call — deterministic, zero hallucination risk.
        """
        profile = self.structured_task.get("oncology_profile", {}) or {}
        basic_info = profile.get("basic_info", "").strip()
        diagnosis = profile.get("diagnosis_and_stage", "").strip()
        pathology = profile.get("pathology_and_molecular", "").strip()
        surgery_type = profile.get("surgery_type", "").strip()

        comorbidities = self.structured_task.get("major_comorbidities_affecting_treatment", []) or []
        prelim_plan = self.structured_task.get("preliminary_plan", {}) or {}
        treatment_plan = (prelim_plan.get("main_oncology_treatment") or prelim_plan.get("main", "") or "").strip()

        parts = []
        if basic_info:
            parts.append(f"**基本信息**：{basic_info}")
        if diagnosis:
            parts.append(f"**诊断及FIGO分期**：{diagnosis}")
        if pathology:
            parts.append(f"**病理及分子分型**：{pathology}")
        if surgery_type:
            parts.append(f"**已行手术**：{surgery_type}")
        if comorbidities:
            parts.append(f"**重大合并症**：{'；'.join(comorbidities)}")
        if treatment_plan:
            parts.append(f"**初步治疗方案**：{treatment_plan}")

        if parts:
            return "\n\n".join(parts)

        # Fallback: extract first relevant portion from treatment_context
        fallback = self.treatment_context[:800].strip()
        return fallback if fallback else "（患者信息见初步会诊草稿）"

    # =================================================================
    # Section: 合并症管理 (LLM-generated, focused on comorbidity triage)
    # =================================================================
    async def _generate_comorbidity_management(
        self,
        main_treatment_plan: str,
        safety_context: str = "",
        risk_factor_context: str = "",
        reviewer_issues: list = None,
        previous_output: str = "",
    ) -> str:
        """LLM generates 合并症管理 (合并症与治疗期管理 + 偶发发现合并声明).

        TreatmentDecisionAgent handles 主要方案 separately.  This method focuses
        exclusively on the three-tier comorbidity triage and formatting.

        Args:
            main_treatment_plan: Output from TreatmentDecisionAgent — used to
                align comorbidity management with the chosen treatment regimen.
            reviewer_issues: If provided, injected as high-priority feedback.
        """

        # Build patient snapshot for context
        profile = self.structured_task.get("oncology_profile", {}) or {}
        basic_info = profile.get("basic_info", "").strip()
        diagnosis = profile.get("diagnosis_and_stage", "").strip()
        pathology = profile.get("pathology_and_molecular", "").strip()
        surgery_type = profile.get("surgery_type", "").strip()

        comorbidities = self.structured_task.get("major_comorbidities_affecting_treatment", []) or []
        critical_infections = self.structured_task.get("critical_infections", []) or []
        incidental = self.structured_task.get("incidental_findings", []) or []
        # Surgical anatomy facts
        surgery_flags = classify_surgery(surgery_type)

        # ── Code-level cleanup: strip benign findings on removed organs ──
        # These have no management target — the organ is in a pathology jar.
        comorbidities = _filter_removed_organ_items(comorbidities, surgery_type)
        critical_infections = _filter_removed_organ_items(critical_infections, surgery_type)
        incidental = _filter_removed_organ_items(incidental, surgery_type)

        surgery_fact = ""
        if surgery_type:
            surgery_fact = f"- 🛑 **手术方式（解剖事实——最高优先级）**：{surgery_type}"
            if surgery_flags["is_hysterectomy"]:
                surgery_fact += (
                    "\n  ⚠️ 该患者已切除子宫 → 无宫颈 → 无宫颈管。"
                    "术后处理和随访方案中**绝对禁止**出现：宫颈、宫颈细胞学、TCT、宫颈筛查、宫颈癌筛查、阴道镜检查宫颈、宫颈病变状态。"
                    "可用的表述：妇科查体关注阴道残端、阴道B超、阴道壁HPV检测。"
                )
            if surgery_flags.get("is_bso"):
                surgery_fact += (
                    "\n  ⚠️ 该患者已切除双侧附件（卵巢+输卵管） → 无输卵管 → 无卵巢。"
                    "**绝对禁止**对已切除器官写随诊/监测建议。"
                    "输卵管/卵巢的病理发现（如'左侧输卵管癌累及''卵巢周围炎'）是术后标本的诊断信息，不是需要管理的合并症——器官已切除，不存在'监测病情变化'的对象。"
                )

        hpv_followup_rules = build_hpv_followup_rules(surgery_type)

        patient_snapshot_parts = []
        if basic_info:
            patient_snapshot_parts.append(f"- 基本信息：{basic_info}")
        if diagnosis:
            patient_snapshot_parts.append(f"- 诊断与分期：{diagnosis}")
        if pathology:
            patient_snapshot_parts.append(f"- 病理与分子分型：{pathology}")
        if surgery_fact:
            patient_snapshot_parts.append(surgery_fact)
        if critical_infections:
            patient_snapshot_parts.append(f"- 🚨 致命红线（活动性感染/炎症，化疗前必须排雷）：{'；'.join(critical_infections)}")
        if comorbidities:
            patient_snapshot_parts.append(f"- 系统底座（慢性合并症）：{'；'.join(comorbidities)}")
        if incidental:
            patient_snapshot_parts.append(f"- 静默噪音（良性偶发发现，无需转诊）：{'；'.join(incidental)}")
        patient_snapshot = "\n".join(patient_snapshot_parts)

        chemo_clearance_warning = self._build_chemo_clearance_warning(critical_infections)

        feedback_block = ""
        if reviewer_issues:
            issues_text = "\n".join(f"  - {iss}" for iss in reviewer_issues)
            previous_text = previous_output if previous_output else "（无上一轮输出记录）"
            feedback_block = (
                "🔴🔴🔴 **【强制纠错指令 —— 你之前的草稿被医学质控委员会打回】** 🔴🔴🔴\n\n"
                "你之前的草稿犯了以下严重错误，被质控委员会打回：\n"
                f"{issues_text}\n\n"
                "**【你的强制任务——做不到等于失败】**：\n"
                "1. 必须深刻理解上述每条错误。你必须在本次生成中彻底修正，不可遗漏任何一条。\n"
                "2. 必须在原文基础上进行精准修复，不可通过改变表述方式绕开问题。\n"
                "3. 严禁使用模棱两可的话术绕过问题。"
                "修复后的文本必须能明确体现你已经逐条采纳了上述意见。\n"
                "4. 严禁通过改变主语、替换概念、模糊表述等方式\"钻空子\"。\n\n"
                "⚠️ 未标记问题的部分保持原样即可，不要修改已经正确的部分。\n\n"
                "【你上一轮的完整输出——请在此基础上有针对性地修正上述问题】\n"
                f"---\n{previous_text}\n---\n\n"
                "🔴🔴🔴\n"
            )

        # Build explicit multi-line lists so the LLM can see each tier's contents.
        # System-base comorbidities are pre-grouped by department so the LLM
        # doesn't have to infer groupings — this structurally prevents splitting.
        critical_items = "\n".join(f"  - {c}" for c in critical_infections) if critical_infections else "（无）"
        comorbidity_groups = _group_by_department(comorbidities)
        if comorbidity_groups:
            comorbidity_lines = []
            for dept, items in comorbidity_groups.items():
                comorbidity_lines.append(f"  - {dept}：{'、'.join(items)}")
            comorbidity_items = "\n".join(comorbidity_lines)
        else:
            comorbidity_items = "（无）"
        incidental_items = "\n".join(f"  - {c}" for c in incidental) if incidental else "（无）"

        prompt = textwrap.dedent(f"""
        你是一名具备顶尖国际视野的妇科肿瘤 MDT 首席专家。
        请根据患者病情和已确定的主要治疗方案，撰写 **「合并症管理」** 章节。

        {feedback_block}
        ─────────────────────────────────────────────
        【患者病情概要】：
        {patient_snapshot}

        【已确定的主要治疗方案（来自 TreatmentDecisionAgent——合并症管理必须与此对齐）】：
        {main_treatment_plan[:3000]}

        ─────────────────────────────────────────────
        🛑🛑🛑 **【三级分流——本任务最核心的规则，违反即不合格】** 🛑🛑🛑

        上游已将患者合并症分为三级，你**必须严格按列表归属输出**：

        **第一级：致命红线（化疗前必须逐条排雷）** → 这些条目必须在下方（1）（2）（3）... 中逐条列出。🚨 来自同一器官/同一检查的多条致命红线必须合并为一条综合条目（如胃镜发现的多处胃部糜烂→合并为一条消化科条目），禁止拆成多条：
        {critical_items}

        **第二级：系统底座（慢性合并症，需专科共同管理）** → 以下已按科室预分组，**每个科室组在输出中必须合并为一条编号条目**，绝对禁止将同科室的条目拆成多条编号：
        {comorbidity_items}

        **第三级：静默噪音（良性偶发发现）** → 🚨 以下条目**绝对禁止**出现在上方的（1）（2）（3）... 编号列表中！全部统一塞进末尾"偶发发现合并声明"一句话带过：
        {incidental_items}

        ─────────────────────────────────────────────
        【合并症输出格式要求】
        请模仿真实三甲医院专家的简练文风撰写合并症管理条目。不需要解释病理机制和原因。

        🛑 **【红线——禁止越权开处方】**：
        - **绝对禁止**写具体用药方案、药物名称、剂量、疗程调整。
        - **绝对禁止**写手术建议、有创操作指征。
        - 治疗决策由对应专科医生做出，你只提供分诊方向和必要的基线评估建议。

        每条格式：[具体检查发现/诊断] + [必要的基线评估或标准随访动作] + [建议XX科就诊/随诊]。

        核心原则：
        - **写实不写虚**：必须写出具体的检查发现来源，而非泛泛的疾病诊断名。
          ✅ 正确：「肺CT提示左肺上叶下舌段、右肺中叶内侧段及两肺下叶散在慢性炎症，建议呼吸科随诊」
          ❌ 错误：「肺部炎症：建议呼吸科就诊」
        - **合并同一系统的疾病**：例如将高血压、冠心病、主动脉钙化统一归为一条。
          ✅ 正确：「患者高血压、冠心病支架植入术后、主动脉管壁钙化，建议心内科随诊」
        - **感染/炎症必须列出**：肺部、消化道、泌尿系等任何部位的感染或活动性炎症均属于致命红线，
          必须在编号列表中逐条列出，禁止遗漏，禁止合并到其他条目中。
        - **慢性病给出标准监测提示**：糖尿病→"建议监测血糖，内分泌科随诊"；高血压→"建议监测血压"。
          这些不是治疗方案，是标准随访动作。
        - **偶发发现极简处理**：脂肪肝、囊肿、良性结节等一句话带过即可。

        {chemo_clearance_warning}

        【HPV 及宫颈病变相关检索信息】（如有，则合并症管理中需酌情考虑病史风险）：
        {risk_factor_context if risk_factor_context else "（无相关检索信息）"}

        【合并症安全性循证检索】（合并症管理决策参考——直接来源于 PubMed 检索的原始证据）：
        {safety_context if safety_context else "（安全性专用检索未产生独立证据，依赖临床判断）"}

        ─────────────────────────────────────────────
        🛑 **【补充规则】**

        1. **【同系统合并 + 正确科室分配】**：同一器官/系统的多个异常必须合并为一条，且分配给正确的专科。
           示例：
           ✅ 正确：[心脑血管系统合并条目 → 「高血压N级、冠状动脉粥样硬化性心脏病、心脏支架置入术后：建议心内科随诊」]
           ❌ 错误：高血压一条、冠心病一条、支架一条、动脉硬化一条——拆碎即不合格！
           🚨 **科室分配强制规则**：骨骼/肋骨/关节/肌肉发现 → 骨科；肝脏/胆囊/胰腺 → 肝胆外科或消化科；肾脏/膀胱 → 泌尿科或肾内科。禁止将不同器官系统的发现强行塞进同一个科室条目。

        2. **【关键脏器炎症/感染——化疗前强制评估】**：肺、消化道、泌尿系等存在感染灶时，必须写入化疗前专科评估（化疗后骨髓抑制期为感染爆发窗口）。

        3. **【负向约束——禁止跨越治疗阶段】**：术后辅助阶段报告**绝对禁止**罗列后线/抢救性临床试验。

        4. **【禁止越权开处方】**：可以写标准随访动作（如"监测血糖""每年复查TCT"）和转诊科室，**绝对禁止**写具体用药方案、药物剂量、疗程调整、手术建议。

        5. **【🔥 已确认事实不得重复"追踪"】**：病情概要中已明确的事实不得写"建议追踪"。

        6. **【编号规则】**：先列致命红线（来自同一器官/系统的多条致命红线必须合并为一条综合条目），再列系统底座（按科室归组，同科室合并为一条）。静默噪音条目**绝对禁止**出现在编号列表中。

        7. 🔴 **【中文输出——全字段强制要求】**：所有描述性文字必须使用中文，数值和统计量保留原文。**绝对禁止**直接复制粘贴英文段落。

        8. **【输出前强制自检——输出不合格即视为生成失败】**：
           a) 静默噪音列表中的每个条目是否全部没有出现在上方编号列表中？
           b) 致命红线列表中的每个条目是否已全部逐条列出、无遗漏、未合并？
           c) 系统底座列表中的每个条目是否已全部列出、无遗漏？（🛑 此项常被遗漏，必须逐条核对）
           d) 同科室/同系统的条目是否已归组为一条？
        9. **【HPV(人乳头瘤病毒)与HP(幽门螺杆菌)绝对不能混淆】**：HPV→妇科，HP→消化科。二者是完全不同的病原体，绝对禁止交叉转诊或混为一谈。
        10. **【🔥 禁止冗余转诊——最高优先级】**：
           - 患者的主治科室是**妇科肿瘤科**，所有妇科良性病变（子宫肌瘤、卵巢囊肿、输卵管积水等）及妇科手术史（肌瘤剥除、剖宫产等）均由妇科肿瘤科团队在常规随访中一并处理，**绝对禁止**在合并症管理中单独创建"妇科随诊"条目。
           - 已行手术（如"开腹子宫肌瘤剥除术""剖宫产"）属于既往手术史，不是需要管理的合并症，**绝对禁止**出现在编号列表中。
           - ✅ 正确做法：妇科相关良性发现不写入合并症编号列表；如需提及，在偶发发现合并声明中一句话带过即可。
           - ❌ 错误示例："子宫多发小肌瘤 → 建议妇科随诊"——患者已经因为子宫内膜癌在妇科肿瘤科随访，此条为废话转诊。
           - 🚨 **【已切除器官的病理发现不是合并症——零容忍】**：如果患者已行双侧附件/卵巢/输卵管切除术，则输卵管/卵巢的任何病理发现（如"左侧输卵管癌累及""卵巢周围炎"）均属于术后标本诊断信息，不是需要管理的合并症——器官已切除、已成病理切片，不存在"监测""随诊"的对象。**绝对禁止**将已切除器官的任何发现写入编号列表。

        {hpv_followup_rules}

        ─────────────────────────────────────────────
        【输出格式——严格按此结构输出】

        你必须将下方所有标记替换为患者的具体检查发现和科室。输出中**绝对禁止**出现：
        - 方括号 `[` `]`
        - "如："、"例如"等举例性引导词
        - 系统分类标签（"致命红线条目""心脑血管系统合并条目"等元描述）
        - 泛泛的诊断名（如"肺部炎症"）——必须写出具体检查来源

        **正确输出示例**（仅供参考格式，具体内容需依据患者实际数据）：
        ```
        1、 **合并症与治疗期管理**：
        （1）胃镜提示慢性胃炎伴活动性HP感染、胃窦糜烂，建议消化科就诊。
        （2）肺CT提示双肺散在慢性炎性病灶，建议呼吸科随诊。
        （3）患者高血压2级、冠心病支架术后，建议监测血压，心内科随诊
        （4）患者2型糖尿病，建议监测血糖，内分泌科随诊
        （5）患者HPV52型阳性，建议妇科随诊，定期宫颈癌筛查

        2、 **偶发发现合并声明**：脂肪肝、肝血管瘤、右肾囊肿等余偶发发现，建议随诊。
        ```

        你的输出必须以上述示例为模板，将每一条替换为患者真实数据。先致命红线，后系统底座（同科室合并）。静默噪音只出现在末尾合并声明中。
        """)

        logger.info("[MDTReportAgent] 正在生成合并症管理...")

        for attempt in range(3):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=1200.0, max_retries=3
                )
                content = remove_think_tags(response.content).strip()
                if "合并症与治疗期管理" not in content:
                    logger.warning(f"[MDTReportAgent] 输出缺少合并症管理，重试中... ({attempt+1})")
                    continue

                # Code-level post-check: every critical item must be covered
                missing = _check_critical_coverage(critical_infections, content)
                if missing:
                    logger.warning(
                        f"[MDTReportAgent] 致命红线条目缺失 {len(missing)} 条: "
                        f"{missing}，重试中... ({attempt+1})"
                    )
                    # Inject missing-item warning into a stronger retry prompt
                    missing_lines = "\n".join(f"  ⚠️ 缺失: {m}" for m in missing)
                    prompt = (
                        f"🔴🔴🔴 **【上一轮输出遗漏了以下致命红线条目——本次必须补充】** 🔴🔴🔴\n"
                        f"{missing_lines}\n\n"
                        f"⚠️ 上述条目必须出现在编号列表中，不可遗漏。\n\n"
                        f"{prompt}"
                    )
                    continue

                return content
            except Exception as e:
                logger.error(f"[MDTReportAgent] 合并症管理生成报错: {e}")
                if attempt == 2:
                    return "1、 **合并症与治疗期管理**\n（生成失败）\n\n2、 **偶发发现合并声明**\n（生成失败）"

        return "1、 **合并症与治疗期管理**\n（生成失败）\n\n2、 **偶发发现合并声明**\n（生成失败）"

    # =================================================================
    # Guideline extraction (same logic, refactored)
    # =================================================================
    def _extract_guideline_section(self) -> str:
        """Extract guideline sub-sections from treatment_context (no heading).

        Returns just the demoted sub-content (#### sub-headings) for the
        assembly code to wrap with the correct `### 2.` heading.
        """
        default = "（未能在草稿中匹配到指南解析部分）"
        match = re.search(
            r'(## 二、 核心指南与共识详尽解析.*?)(?=\n## 三、|\n## 四、|\Z)',
            self.treatment_context, re.DOTALL
        )
        if not match:
            return default

        text = match.group(1).strip()
        lines = text.split('\n')

        # Skip the ## 二、 heading line
        if lines and lines[0].startswith('## '):
            lines = lines[1:]

        # Demote sub-headings: ### -> ####
        demoted = []
        for line in lines:
            if line.startswith('### '):
                line = '#' + line  # ### -> ####
            demoted.append(line)
        return '\n'.join(demoted).strip()

    # =================================================================
    # Chemo pre-clearance: scan incidental_findings for infections
    # =================================================================
    @staticmethod
    def _build_chemo_clearance_warning(critical_infections: list) -> str:
        """
        Build a prominent chemo-clearance warning from the already-classified
        critical_infections list (致命红线).

        The JSON extractor has already classified these as active infections /
        inflammatory lesions / bleeding risks. This method simply formats them
        as a high-priority checklist for the generation prompt.

        Returns a prominent, itemized warning to inject into the prompt,
        or empty string if no critical items found.
        """
        if not critical_infections:
            return ""

        count = len(critical_infections)
        return (
            f"🚨🚨🚨 **【化疗前排雷——强制逐条处理清单】** 🚨🚨🚨\n"
            f"患者即将接受骨髓抑制性化疗，上述第一级致命红线中的 {count} 项"
            f"（活动性感染/炎症/溃疡）必须在化疗前完成专科评估。\n"
            "  ⚠️ 遗漏任何一项，化疗期可能诱发消化道大出血、重症肺炎、HBV再激活等致死性并发症。\n"
            "  ⚠️ 在【合并症与治疗期管理】中必须逐条列出上述致命红线列表中的每一项并给出科室会诊建议，禁止合并、禁止省略、禁止推后到偶发发现声明中。\n"
            "  ⚠️ 致命红线条目已在【三级分流】第一级中列出，请直接从该列表提取——不要重复列出两次。"
        )

    # =================================================================
    # Direct extraction of 病情摘要 from treatment_context (verbatim)
    # =================================================================
    def _extract_disease_block(self) -> str:
        """Extract entire ## 一、 病情摘要与风险判定 from treatment_context verbatim."""
        match = re.search(
            r'(## 一、 病情摘要与风险判定.*?)(?=\n## 二、|\Z)',
            self.treatment_context, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return ""

    # =================================================================
    # Final report assembly (code-level formatting)
    # =================================================================
    def _assemble_final_report(
        self,
        patient_summary: str,
        guideline_section: str,
        trial_analysis: str,
        main_treatment_plan: str,
        comorbidity_management: str,
        prognosis_section: str,
        followup_plan: str,
    ) -> str:
        """Assemble the final MDT report from all parts using code-level formatting.

        术后处理 is assembled from two independent agents:
          - 主要方案 (TreatmentDecisionAgent)
          - 合并症管理 (MDTReportAgent._generate_comorbidity_management)

        预后分析由 PrognosisAgent 直接输出拼接，不经 LLM 二次概括。
        """
        sections = []

        # ── Title ──
        sections.append("# 妇科肿瘤 MDT 最终版会诊报告")

        # ── 一、 病情分析 ──
        disease_block = self._extract_disease_block()
        if disease_block:
            sections.append(disease_block)
            sections.append(f"### 核心指南与共识详尽解析\n{guideline_section}")
            sections.append(f"### 核心临床试验及 PICO 循证解析\n{trial_analysis}")
        else:
            sections.append("## 一、 病情分析")
            sections.append(f"### 1. 病历及病理摘要\n{patient_summary}")
            sections.append(f"### 2. 核心指南与共识详尽解析\n{guideline_section}")
            sections.append(f"### 3. 核心临床试验及 PICO 循证解析\n{trial_analysis}")

        # ── 二、 术后处理 (from two agents) ──
        treatment_parts = []
        treatment_parts.append("## 二、 术后处理\n")
        treatment_parts.append("### 主要方案")
        treatment_parts.append(main_treatment_plan)
        treatment_parts.append("### 合并症管理")
        treatment_parts.append(comorbidity_management)
        sections.append("\n\n".join(treatment_parts))

        # ── 三、 预后分析 (from PrognosisAgent, directly spliced) ──
        prognosis_section = re.sub(r'^#+\s*三[、.]?\s*预后分析\s*\n*', '', prognosis_section.strip())
        sections.append(f"## 三、 预后分析\n{prognosis_section}")

        # ── 四、 随访方案 ──
        followup_plan = re.sub(r'^#+\s*四[、.]?\s*随访(方案|大纲)\s*\n*', '', followup_plan.strip())
        followup_plan = re.sub(r'^#+\s*随访(方案|大纲)\s*\n*', '', followup_plan.strip())
        sections.append(f"## 四、 随访方案\n{followup_plan}")

        report = "\n\n".join(sections)

        # Safety: strip citation markers from 术后处理 section
        report = self._strip_citations_from_treatment_section(report)

        # Normalize overly alarming medical terms for patient-facing report
        report = self._normalize_medical_terms(report)

        return report

    # =================================================================
    # Medical term normalization (soften alarming jargon for patients)
    # =================================================================
    @staticmethod
    def _normalize_medical_terms(text: str) -> str:
        """Replace overly alarming clinical terms with patient-friendly equivalents.

        Applied as a pure-code post-processing safety net — catches LLM leakage
        even when prompts already prefer the softer form.
        """
        _TERM_MAP = {
            "脑梗死": "脑梗",
            "心肌梗死": "心梗",
        }
        for old, new in _TERM_MAP.items():
            text = text.replace(old, new)
        return text

    # =================================================================
    # Citation stripping safety net (for 术后处理 section)
    # =================================================================
    def _strip_citations_from_treatment_section(self, content: str) -> str:
        """
        Code-level cleanup: remove all citation markers from the post-operative
        management section (## 二、 术后处理).
        Supports: [1], [2,3], [^^8], [^^10, ^^12] etc.
        """
        for punct in ['、', '．', '.', '：', ':']:
            pattern = rf"(##\s*二{re.escape(punct)}\s*术后处理[\s\S]*?)(?=\n##\s*[三四]|\Z)"
            m = re.search(pattern, content, re.DOTALL)
            if m:
                section_text = m.group(1)
                cleaned = re.sub(
                    r'\[\^?\^?\^?\d+(?:\s*[,、]\s*\^?\^?\^?\d+)*\s*\]',
                    '', section_text
                )
                cleaned = re.sub(r' +', ' ', cleaned)
                cleaned = cleaned.replace(' 。', '。').replace(' ，', '，')
                content = content[:m.start()] + cleaned + content[m.end():]
                break
        return content

    # =================================================================
    # Main entry point
    # =================================================================
    async def run(self, trial_analysis: str, followup_plan: str, prognosis_data: str,
                  main_treatment_plan: str = "", safety_context: str = "") -> str:
        """
        Assemble the final MDT report.

        TreatmentDecisionAgent.generate_main_plan() must be called BEFORE this
        method to produce main_treatment_plan.  This method handles:
          - 病情分析 (code-built from structured_task)
          - 合并症管理 (LLM — comorbidity triage)
          - 预后分析 (direct splice from PrognosisAgent)
          - 随访方案 (direct splice from FollowupAgent)
          - Final assembly
        """
        # Step 1: Build 病历及病理摘要 from structured data (code, no LLM)
        patient_summary = self._build_patient_summary()
        logger.info(f"[Agent2] 病历摘要已构建（{len(patient_summary)} 字符，代码生成，无 LLM 参与）")

        # Step 2: Extract and demote guideline section
        guideline_section = self._extract_guideline_section()
        logger.info(f"[Agent2] 指南章节已提取（{len(guideline_section)} 字符）")

        # Step 3: Read risk factor context from ContextBus
        risk_factor_context = ""
        if self.context_bus:
            rf_msgs = await self.context_bus.get_by_type("risk_factor_context")
            if rf_msgs:
                risk_factor_context = rf_msgs[-1]["content"][:2000]
                logger.info(f"[Agent2] 读取了 {len(risk_factor_context)} 字符的 HPV/风险因素数据")

        # Step 4: Generate 合并症管理 (LLM — aligned with main treatment plan)
        comorbidity_management = await self._generate_comorbidity_management(
            main_treatment_plan=main_treatment_plan,
            safety_context=safety_context,
            risk_factor_context=risk_factor_context,
        )
        logger.info(f"[Agent2] 合并症管理已生成（{len(comorbidity_management)} 字符）")

        # Step 5: Assemble final report by code
        final_report = self._assemble_final_report(
            patient_summary, guideline_section, trial_analysis,
            main_treatment_plan, comorbidity_management,
            prognosis_data, followup_plan
        )

        logger.info(f"[Agent2] 最终报告已拼接完成（{len(final_report)} 字符）")
        return final_report
