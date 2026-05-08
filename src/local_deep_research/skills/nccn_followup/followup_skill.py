"""
NCCN Follow-up Skill — 基于 NCCN 指南生成高颗粒度随访方案。

流程：
  1. 提取患者临床信息
  2. 提取旧随访大纲作为「个性化参考」（合并症管理、特殊叮嘱等）
  3. 以 NCCN 指南为准绳生成详细的随访方案
  4. 如果 LLM 输出了 JSON 格式，自动转为自然文本

目录结构：
  skills/nccn_followup/
    ├── __init__.py
    ├── followup_skill.py     ← 本文件
    └── references/           ← 用户在此放 NCCN 指南 .md 文件
        ├── NCCN_2026_子宫内膜癌随访.md
        └── ...
"""

import json
import logging
import os
import re
import textwrap
from pathlib import Path

from ...utilties.patient_state import classify_surgery, build_surgery_anatomy_rules

logger = logging.getLogger(__name__)


class NCCNFollowupSkill:
    """
    基于 NCCN 指南生成高颗粒度随访方案。
    旧大纲仅作为患者个性化信息的参考来源，不以它为主体。
    """

    def __init__(self, references_dir: str = None):
        if references_dir is None:
            references_dir = os.path.join(os.path.dirname(__file__), "references")
        self.references_dir = Path(references_dir)
        self.nccn_content = self._load_references()

    # -----------------------------------------------------------------
    # 加载 NCCN 参考文件
    # -----------------------------------------------------------------
    def _load_references(self) -> str:
        if not self.references_dir.exists():
            logger.warning(f"NCCN 参考文献目录不存在: {self.references_dir}")
            return ""
        md_files = sorted(self.references_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"NCCN 目录下没有 .md 文件: {self.references_dir}")
            return ""

        contents = []
        for f in md_files:
            try:
                text = f.read_text(encoding="utf-8")
                if len(text) > 80000:
                    logger.info(f"{f.name} 较长，截取前 80000 字符")
                    text = text[:80000] + "\n\n...[已截断]..."
                contents.append(f"===== {f.stem} =====\n{text}")
                logger.info(f"已加载 NCCN 参考: {f.name} ({len(text)} 字符)")
            except Exception as e:
                logger.warning(f"加载 NCCN 文件失败 {f}: {e}")

        return "\n\n".join(contents)

    # -----------------------------------------------------------------
    # 从草稿中提取已有的随访大纲（作为个性化参考）
    # -----------------------------------------------------------------
    def _extract_old_followup_plan(self, treatment_context: str) -> str:
        """提取草稿中已有的随访方案/大纲。"""
        patterns = [
            r"## 四[、.]\s*随访(方案|大纲).*?(?=##\s|\Z)",
            r"## 4[、.]\s*随访(方案|大纲).*?(?=##\s|\Z)",
            r"####?\s*随访(方案|大纲).*?(?=##\s|\Z)",
            r"\*\*1\.\s*随访频率\*\*.*?(?=##\s|\Z)",
        ]
        for pat in patterns:
            m = re.search(pat, treatment_context, re.DOTALL)
            if m:
                return m.group(0).strip()
        return ""

    # -----------------------------------------------------------------
    # 从草稿中提取患者临床信息
    # -----------------------------------------------------------------
    def _extract_clinical_context(self, treatment_context: str) -> str:
        """提取病情分析 + 术后处理 + 合并症。"""
        sections = []

        m = re.search(
            r"## 一、\s*病情分析.*?(?=## 二、|\Z)",
            treatment_context, re.DOTALL
        )
        if m:
            sections.append(m.group(0).strip())

        m = re.search(
            r"## 二、\s*术后处理.*?(?=## 三、|\Z)",
            treatment_context, re.DOTALL
        )
        if m:
            sections.append(m.group(0).strip())

        # 合并症关键词提取
        comorbidity_lines = []
        for line in treatment_context.split("\n"):
            if any(kw in line for kw in ["合并", "冠心病", "高血压", "糖尿病",
                                          "心功能", "肾功能", "心律失常", "慢性肾病",
                                          "贫血", "甲减", "甲亢", "高脂血症", "脂肪肝",
                                          "既往史", "并发症", "HPV", "TCT", "动脉硬化",
                                          "斑块", "钙化", "囊肿", "结石", "骨质疏松"]):
                comorbidity_lines.append(line)
        if comorbidity_lines:
            sections.append("【合并症信息】\n" + "\n".join(comorbidity_lines))

        return "\n\n".join(sections) if sections else treatment_context[:2000]

    # -----------------------------------------------------------------
    # 核心：基于 NCCN 生成高颗粒度随访方案
    # -----------------------------------------------------------------
    async def generate(self, report_model, treatment_context: str, surgery_type: str = "",
                       reviewer_issues: list = None, previous_output: str = "") -> str:
        """
        以 NCCN 指南为准绳生成详细随访方案，旧大纲仅作个性化参考。

        Args:
            report_model: LLM 模型实例
            treatment_context: 患者治疗草稿全文（可能含旧随访大纲）
            surgery_type: 患者已接受的手术方式（如全子宫切除术等）
            reviewer_issues: 上一轮审查发现的问题列表，注入 prompt 要求修正
            previous_output: 上一轮实际输出原文，供 LLM 定位需要修正的具体内容。

        Returns:
            高颗粒度的随访方案 markdown 文本
        """
        clinical_context = self._extract_clinical_context(treatment_context)
        if surgery_type:
            clinical_context = "【手术方式】\n" + surgery_type + "\n\n" + clinical_context
        old_plan = self._extract_old_followup_plan(treatment_context)

        # Use shared surgery classifier so logic stays in sync with
        # mdt_report_agent and reviewer_agent.
        surgery_anatomy_rules = build_surgery_anatomy_rules(surgery_type)

        if not self.nccn_content.strip():
            logger.warning("NCCN 参考为空，降级生成")
            nccn_section = "（未加载到 NCCN 指南文件）"
        else:
            nccn_section = self.nccn_content

        # 旧方案作为「个性化参考」传入 prompt
        if old_plan:
            logger.info("发现旧随访大纲，作为个性化参考传入")
            reference_section = f"""
        【旧随访大纲（仅供参考—仅保留其中的患者个性化信息）】：
        ---
        {old_plan}
        ---
        """
        else:
            reference_section = ""

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

        prompt = textwrap.dedent(f"""
        你是一名妇科肿瘤随访方案制定专家。
        请以【NCCN 指南】为准绳，结合【患者临床信息】，制定一份高颗粒度的子宫内膜癌术后随访方案。

        {feedback_block}
        【NCCN 子宫内膜癌随访指南】：
        ---
        {nccn_section}
        ---

        【患者临床信息】：
        ---
        {clinical_context}
        ---
        {reference_section}

        【核心临床规则——理解后内化，按规则输出方案内容】：
        1. **NCCN 优先**：随访频率、检查项目、影像学建议等必须以 NCCN 指南为准绳。
        2. **个体化保留**：如果旧大纲中有针对该患者合并症的具体建议（如心内科随诊、血糖监测方案等），必须继承到新方案中。
        3. **禁止扩写旧大纲**：旧大纲颗粒度通常不足。不要以它为主体进行扩写，而是以 NCCN 指南为主体生成详细方案。
        4. **覆盖七个方面**：随访频率、手术并发症急诊指征、警示症状、随访检查内容、患者教育、心理社会支持、合并症管理。
        5. **[^^n] 角标**：患者临床信息中如果带有 [^^n] 角标，原样保留。

        【临床规则——TCT/HPV】：
        - 全子宫切除术后患者：不推荐常规阴道细胞学（TCT）检查用于子宫内膜癌术后随访。
        - 有 HPV 阳性史或阴道病变史的患者：HPV 随访不属于肿瘤专科复发监测的必查项目。管理方式为妇科查体时关注阴道残端黏膜，必要时行阴道壁 HPV 检测。
        - 唯一例外：仅当患者接受保留生育功能治疗（子宫未切除）时，才需在随访中纳入宫颈癌筛查。
        - **重要医学常识——HPV ≠ HP**：HPV（人乳头瘤病毒）是妇科下生殖道病毒；HP（幽门螺杆菌）是消化道细菌。二者完全不同，禁止混淆。涉及 HPV 的事项对应妇科，涉及 HP 的事项对应消化科。

        【临床规则——影像学】：
        - 术后无症状患者不推荐常规胸腹盆 CT 或 PET-CT 全身筛查。
        - 影像学检查为症状驱动：出现咳嗽等症状时查胸部 CT，出现盆腹腔疼痛时查盆腹 MRI/CT。
        - 术后常规影像学仅推荐妇科查体 + 盆腔/阴道 B 超。

        {surgery_anatomy_rules}

        【强制输出模板——参照真实临床专家随访方案风格】：
        1、 按照[根据NCCN指南确定的风险分层]随访。随访频率：术后前 2 年每 3 个月随访 1 次，第 3-5 年每 6 个月随访 1 次，5 年后每年随访 1 次。

        2、 手术可能导致盆腹腔粘连，如出现腹痛、呕吐、肛门停止排气排便等症状时立即就诊。

        3、 可能提示复发的症状包括：盆腹腔疼痛、明显占位、淋巴结肿大、阴道流血、短期内体重明显减轻、便血、血尿，或其他新出现的盆腔、腹部或肺部症状。当出现相关症状，应立即就诊而非等到下次预约就诊时间。

        4、 随访时检查内容：病史采集、体格/妇科检查、抽血（肿瘤标志物 CA125、HE4不需空腹；血糖（需空腹）；若同时检查肝功能（需空腹））、盆腔及腹B超，不推荐常规阴道细胞学检查。阴道B超术后三月可以开始做，主要观察盆腔局部情况；腹部B超观察盆腹腔淋巴结区域有无异常。当临床医生怀疑有复发征象或病情需要时，根据临床指征选择其他影像学检查（盆腔增强 MRI、肺 CT、上腹部增强 CT，或全身 PET-CT）。

        5、 患者教育：了解治疗相关远期并发症和肿瘤复发征象、提倡健康的生活方式、适当锻炼、均衡营养、减重、性健康指导（如使用阴道扩张器、润滑剂、保湿剂等）。

        6、 如存在心理社会学相关问题（抑郁焦虑情绪、担心肿瘤复发、社交和亲密关系改变等），建议寻求专业人士帮助，必要时心理干预。

        7、 合并症管理与科室随诊：
        - 患者[合并症]，建议[科室]随诊，[具体监测/评估方案]
        - （结合患者具体合并症逐条列出，仅写监测和随诊建议，不讨论药物禁忌症）
        - **静默噪音禁止列出**：单纯囊肿、良性结节、轻度脂肪肝、血管壁钙化、胆囊壁轻度增厚等良性偶发发现不得占用独立随诊条目——这些已在术后处理的"偶发发现合并声明"中统一覆盖。
        - 🚨 **同系统合并**：同一器官/系统的多个相关异常必须合并为一条综合随诊建议（如心内科：高血压+冠心病→一条；消化科：HP感染+胃炎+糜烂→一条），禁止按解剖部位拆成多条。

        【禁忌症原则——内化后执行】：
        - 仅讨论当前治疗阶段的药物毒性监测和合并症常规管理，不讨论尚未进入的后续治疗阶段的方案或药物。
        - 高血压、糖尿病、冠心病等常见合并症可通过药物控制，不应被描述为抗肿瘤药物的绝对禁忌症。初始草稿中的过度保守禁忌症陈述不应继承到最终随访方案中。

        🔴 **【中文输出——全字段强制要求】**：所有描述性文字必须使用中文，数值和统计量保留原文。**绝对禁止**直接复制粘贴英文段落。

        🚨🚨🚨 **【输出前最终自检——以下内容绝对禁止出现在你的输出中】** 🚨🚨🚨
        你的输出将直接呈现给患者和管床医生。**严禁**在最终输出中出现：元指令标签（"内部指令""自检""模板要求"等）、对模型下达命令的句式（"禁止写作""确保随访方案中没有X"等）、条件判断句式（"如果手术方式是X"——你已经看过患者信息，直接给出具体方案）。输出序号必须从 1 开始连续编号。违反以上任何一条，输出将被视为不合格。

        💡 请先在 <think> 标签内审阅 NCCN 指南和患者信息，确认无误后再按模板输出。
        """)

        for attempt in range(2):
            try:
                from ...utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
                res = await invoke_with_timeout_and_retry(
                    report_model, prompt, timeout=180.0, max_retries=2
                )
                result = remove_think_tags(res.content).strip()
                # 后处理：JSON → 自然文本
                return self._ensure_text_format(result)
            except Exception as e:
                logger.warning(f"NCCN 随访方案生成失败 (尝试 {attempt+1}): {e}")

        # 降级
        if old_plan:
            return old_plan
        return "随访方案生成失败，请参考 NCCN 指南常规随访。"

    # -----------------------------------------------------------------
    # JSON → 自然文本（纯代码拼接，不用 LLM）
    # -----------------------------------------------------------------
    @staticmethod
    def _ensure_text_format(text: str) -> str:
        """检测 LLM 输出是否为 JSON，若是则转成自然文本格式。"""
        cleaned = text.strip()
        # 去掉 markdown 代码围栏 ```json ... ```
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
        cleaned = cleaned.strip()

        try:
            obj = json.loads(cleaned)
            return NCCNFollowupSkill._json_to_markdown(obj)
        except (json.JSONDecodeError, ValueError):
            return text

    @staticmethod
    def _json_to_markdown(obj, indent=0) -> str:
        """递归将 JSON 对象转为 markdown 文本。"""
        pad = "  " * indent
        lines = []

        if isinstance(obj, dict):
            # 处理根包装键：{"随访方案": {...}} → 直接展开
            if indent == 0 and len(obj) == 1:
                single_val = next(iter(obj.values()))
                if isinstance(single_val, (dict, list)):
                    return NCCNFollowupSkill._json_to_markdown(single_val, indent)

            for key, value in obj.items():
                if isinstance(value, dict):
                    inner = NCCNFollowupSkill._json_to_markdown(value, indent + 1)
                    if re.match(r'\d+[、.]\s*', key):
                        lines.append(f"{pad}**{key}**")
                    else:
                        lines.append(f"{pad}- **{key}**")
                    if inner.strip():
                        lines.append(inner)
                elif isinstance(value, list):
                    if re.match(r'\d+[、.]\s*', key):
                        lines.append(f"{pad}**{key}**")
                    else:
                        lines.append(f"{pad}- **{key}**")
                    for item in value:
                        lines.append(f"{pad}  - {item}")
                else:
                    lines.append(f"{pad}- {key}：{value}")

        elif isinstance(obj, list):
            for item in obj:
                lines.append(f"{pad}- {item}")

        else:
            lines.append(f"{pad}{obj}")

        return "\n".join(lines)
