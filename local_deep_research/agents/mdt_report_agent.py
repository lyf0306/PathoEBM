import asyncio
import logging
import re

from ..utilities.patient_state import classify_surgery, build_hpv_followup_rules
from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags, strip_llm_preamble
from ..prompts import prompt_manager

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

        prompt = prompt_manager.get("mdt_comorbidity").format(
            feedback_block=feedback_block,
            patient_snapshot=patient_snapshot,
            main_treatment_plan=main_treatment_plan[:3000],
            critical_items=critical_items,
            comorbidity_items=comorbidity_items,
            incidental_items=incidental_items,
            chemo_clearance_warning=chemo_clearance_warning,
            risk_factor_context=risk_factor_context if risk_factor_context else "（无相关检索信息）",
            safety_context=safety_context if safety_context else "（安全性专用检索未产生独立证据，依赖临床判断）",
            hpv_followup_rules=hpv_followup_rules,
        )

        logger.info("[MDTReportAgent] 正在生成合并症管理...")

        for attempt in range(3):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=1200.0, max_retries=3
                )
                content = remove_think_tags(response.content).strip()
                content = strip_llm_preamble(content)
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
            except asyncio.CancelledError:
                raise
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
        final_report = strip_llm_preamble(final_report)
        return final_report
