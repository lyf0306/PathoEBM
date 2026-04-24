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
                                          "心功能", "肾功能", "既往史", "并发症"]):
                comorbidity_lines.append(line)
        if comorbidity_lines:
            sections.append("【合并症信息】\n" + "\n".join(comorbidity_lines))

        return "\n\n".join(sections) if sections else treatment_context[:2000]

    # -----------------------------------------------------------------
    # 核心：基于 NCCN 生成高颗粒度随访方案
    # -----------------------------------------------------------------
    async def generate(self, report_model, treatment_context: str) -> str:
        """
        以 NCCN 指南为准绳生成详细随访方案，旧大纲仅作个性化参考。

        Args:
            report_model: LLM 模型实例
            treatment_context: 患者治疗草稿全文（可能含旧随访大纲）

        Returns:
            高颗粒度的随访方案 markdown 文本
        """
        clinical_context = self._extract_clinical_context(treatment_context)
        old_plan = self._extract_old_followup_plan(treatment_context)

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

        prompt = textwrap.dedent(f"""
        你是一名妇科肿瘤随访方案制定专家。
        请以【NCCN 指南】为准绳，结合【患者临床信息】，制定一份高颗粒度的子宫内膜癌术后随访方案。

        【NCCN 子宫内膜癌随访指南】：
        ---
        {nccn_section}
        ---

        【患者临床信息】：
        ---
        {clinical_context}
        ---
        {reference_section}

        【🚨 核心规则（极度重要）】：
        1. **NCCN 优先**：随访频率、检查项目、影像学建议等必须以 NCCN 指南为准绳，确保方案具有事实级颗粒度。
        2. **个体化保留**：如果旧大纲中有针对该患者合并症的具体建议（如心内科随诊、血糖监测方案等），必须继承到新方案中。
        3. **禁止扩写旧大纲**：旧大纲颗粒度通常不足。不要以它为主体进行扩写，而是以 NCCN 指南为主体生成详细方案。
        4. **禁止遗漏**：必须覆盖随访频率、常规随访内容（专科查体+辅助检查）、警示症状、生活方式与合并症管理四个方面。
        5. **[^^n] 角标**：患者临床信息中如果带有 [^^n] 角标，原样保留。

        【强制输出模板】：
        **1. 随访频率**
        - （按 NCCN 指南分阶段写出具体复查时间间隔，如：前2年每3个月、第3-5年每6个月、5年后每年一次）

        **2. 常规随访内容**
        - **专科查体**：（详细列出每次随访必须评估的全身及专科项目）
        - **辅助检查**：
          - （详细列出影像学检查项目及频率，如盆腔MRI/超声、胸部CT等）
          - （详细列出肿瘤标志物检测项目及频率）

        **3. 警示症状**
        - （详细列出提示局部复发或远处转移的具体临床症状，列出至少5条）

        **4. 生活方式与合并症管理**
        - （结合患者合并症，给出针对性的放化疗毒性长程预警与科室随诊建议）

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
