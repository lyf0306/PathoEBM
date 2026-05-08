"""
Shared patient state classification utilities.

Centralises surgery-type detection so the same logic isn't duplicated
across mdt_report_agent, followup_skill, and reviewer_agent.
"""

from typing import Dict, Tuple


_HYSTERECTOMY_KW = ("全子宫", "子宫切除", "子宫全切")
_FERTILITY_SPARING_KW = ("保留生育", "fertility", "保育")
_BSO_KW = ("双侧卵巢", "双侧附件", "双附件", "卵巢和输卵管切除", "卵巢与输卵管切除", "输卵管和卵巢切除", "输卵管与卵巢切除")


def classify_surgery(surgery_type: str) -> Dict[str, bool]:
    """
    Classify a free-text surgery_type string into boolean flags.

    Returns:
      {"is_hysterectomy": bool, "is_fertility_sparing": bool, "is_bso": bool}

    is_hysterectomy    — patient has no uterus / no cervix
    is_fertility_sparing — uterus + cervix are intact (conservative treatment)
    is_bso             — patient has no ovaries / no fallopian tubes

    If neither flag is true the surgery type is ambiguous and downstream
    code should treat anatomy as unconfirmed.
    """
    if not surgery_type:
        return {"is_hysterectomy": False, "is_fertility_sparing": False, "is_bso": False}

    return {
        "is_hysterectomy": any(
            kw in surgery_type for kw in _HYSTERECTOMY_KW
        ),
        "is_fertility_sparing": any(
            kw in surgery_type for kw in _FERTILITY_SPARING_KW
        ),
        "is_bso": any(
            kw in surgery_type for kw in _BSO_KW
        ),
    }


def build_surgery_anatomy_rules(surgery_type: str) -> str:
    """
    Build the surgery-anatomy constraint block injected into LLM prompts.

    Returns a Chinese-language rule string suitable for direct injection
    into a generation prompt's clinical rules section.
    """
    if not surgery_type:
        return ""

    flags = classify_surgery(surgery_type)

    if flags["is_hysterectomy"]:
        return (
            "【临床规则——手术方式与解剖——最高优先级】\n"
            "- 患者已行全子宫切除术 → 解剖事实：无子宫 → 无宫颈 → 无宫颈管。\n"
            "- 🚨 **在你的最终输出中，以下词语绝对禁止出现**：宫颈、宫颈细胞学、TCT宫颈筛查、"
            "宫颈癌筛查、阴道镜检查宫颈、宫颈刮片、宫颈涂片、HPV宫颈取样、宫颈病变状态。\n"
            "- 可以且仅可以使用的表述：妇科查体（视诊+触诊阴道残端）、阴道B超、"
            "阴道壁HPV检测、关注阴道残端黏膜情况。"
        )

    if flags["is_fertility_sparing"]:
        return (
            "【临床规则——手术方式与解剖——最高优先级】\n"
            "- 患者接受的是保留生育功能治疗，子宫及宫颈均完整保留。\n"
            "- 随访中**必须**纳入宫颈癌筛查（TCT+HPV），按常规妇科肿瘤筛查频率执行。\n"
            "- 随访检查应包括妇科查体（视诊+触诊宫颈及阴道）、宫颈细胞学检查（TCT）、HPV检测。"
        )

    return (
        "【临床规则——手术方式与解剖】\n"
        "- 请根据【患者临床信息】中的手术方式确定解剖事实。\n"
        "- 如已行全子宫切除术 → 无宫颈，禁止出现宫颈筛查相关表述。\n"
        "- 如为保留生育功能治疗 → 子宫宫颈完整，需纳入宫颈癌筛查。"
    )


def build_hpv_followup_rules(surgery_type: str) -> str:
    """
    Build HPV follow-up rules that change based on whether the patient
    has had a hysterectomy or fertility-sparing treatment.
    """
    if not surgery_type:
        return ""

    flags = classify_surgery(surgery_type)

    if flags["is_hysterectomy"]:
        return (
            "💡 **HPV 随访**：\n"
            "  - 若患者有 HPV 阳性史/宫颈病变史：**必须在【合并症与治疗期管理】中列出**，"
            "格式如\"患者HPV阳性史（XX型），鉴于存在高危HPV感染史，建议在常规妇科查体时"
            "密切关注阴道壁（尤其是阴道残端）黏膜情况，必要时行阴道壁HPV检测，"
            "以防范下生殖道病变（患者已行全子宫切除术，无宫颈，常规宫颈筛查不适用）\"\n"
            "  - **不得**在肿瘤专科随访部分写 TCT/HPV 检测建议\n"
            "  - 对于子宫内膜癌术后患者，NCCN 指南不推荐常规阴道细胞学检查用于术后随访。"
            "**关键临床常识**：已切除子宫的患者无宫颈，\"常规人群HPV筛查策略\"的核心是宫颈筛查，"
            "对此类患者不适用——应关注阴道残端黏膜而非宫颈"
        )
    else:
        return (
            "💡 **HPV 随访**：\n"
            "  - 若患者有 HPV 阳性史/宫颈病变史：**必须在【合并症与治疗期管理】中列出**，"
            "根据患者手术方式制定管理策略。注意患者手术方式——若非全子宫切除术，"
            "子宫宫颈完整，应纳入常规宫颈癌筛查（TCT+HPV），按妇科肿瘤筛查频率执行。\n"
            "  - 若为保留生育功能治疗：随访中需包含宫颈细胞学检查（TCT）和 HPV 检测"
        )
