import asyncio
import logging
import textwrap

from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags

logger = logging.getLogger(__name__)


class PrognosisAgent:
    """
    Agent 3: Prognosis data extraction specialist.

    Passes the raw structured oncology_core field (diagnosis_and_stage)
    directly to the LLM and lets it extract the correct FIGO stage —
    no Python-side regex parsing.

    The LLM is far more robust against format variations in the upstream JSON
    than any hand-written regex can ever be.
    """

    def __init__(self, report_model, treatment_context: str,
                 oncology_core: dict = None, context_bus=None):
        self.report_model = report_model
        self.treatment_context = treatment_context
        self.oncology_core = oncology_core or {}
        self.context_bus = context_bus

    async def run(self, skill_data="", population_data="", molecular_data="") -> str:
        if not skill_data or not skill_data.strip():
            logger.warning("PrognosisSkill 参考数据为空，降级生成")
            return "权威预后生存率参考数据暂未加载，请参考 SEER/NCDB 公开数据。"

        diagnosis_and_stage = self.oncology_core.get("diagnosis_and_stage", "")

        logger.info(
            "[PrognosisAgent] oncology_core keys=%s",
            list(self.oncology_core.keys()) if self.oncology_core else [],
        )

        # Build the structured-data section for the prompt
        structured_section = ""
        if diagnosis_and_stage:
            structured_section += (
                f"【结构化诊断与分期（来自上游JSON的 diagnosis_and_stage 字段）】：\n"
                f"{diagnosis_and_stage}\n\n"
            )

        if not structured_section:
            structured_section = (
                "（上游未提供结构化数据，请从下方病情草稿中自行提取 FIGO 分期。）\n\n"
            )

        prompt = textwrap.dedent(f"""
        你是一位严谨的肿瘤流行病学数据提取专家。
        任务：从 SEER/NCDB 生存率表中提取患者分期对应的生存率数据。

        {structured_section}
        【当前患者真实病情草稿（补充参考）】：
        {self.treatment_context[:3000]}

        【权威预后参考数据（SEER/NCDB 生存率表——唯一数据源）】：
        {skill_data}

        【🚨 第一步：从结构化字段中提取 FIGO 分期】：
        1. **FIGO 分期提取**：
           - 从上方「结构化诊断与分期」字段中读取患者的分期信息。
           - **优先使用 FIGO 2023 分期**（如字段中同时包含 2009 和 2023 版本）。
           - 如仅有 FIGO 2009 则使用 2009，并在输出中注明。
           - 如结构化字段为空，从病情草稿中查找 FIGO 分期。
           - 提取到的分期（含版本）即为锁定分期，后续只能匹配该分期的 SEER 数据。

        【🚨 第二步：SEER 表数据提取规则】：
        1. **【分期锁定】**：只查找与第一步提取的 FIGO 分期匹配的 SEER 表行。**绝对禁止**选择其他分期。
        2. **大类分期包容**：如果表中没有精确匹配的子分期，使用最接近的大类分期数据，并**如实注明非精确匹配**。
        3. **【🔥 禁止编造】**：你只能引用生存率表中实际存在的数字，绝对禁止编造表中不存在的任何数字。
        4. **【禁止跨癌种混用】**：只提取子宫内膜癌/子宫体癌相关的数据。

        🔴 **【中文输出——全字段强制要求】**：所有描述性文字必须使用中文，数值和统计量保留原文。**绝对禁止**直接复制粘贴英文段落。

        【输出格式——单段连贯叙述，禁止分节重复。直接输出以下章节，将被拼接至最终报告】：

        ## 三、 预后分析

        基于 SEER/NCDB 生存率表，患者诊断为 **FIGO [版本] [分期]**，对应的生存率基线为：
        - **3年OS**：X%（95%CI: X-X）
        - **5年OS**：X%（95%CI: X-X）
        （如表中仅报告了其他指标如 DSS/RFS，如实注明）

        综合判断：[1-2 句话总结上述生存率数据，不做数学加法，不编造合成数字]

        ⚠️ **输出前强制自检**：
        - 确认 FIGO 版本和分期与结构化字段一致（优先 2023）
        - 通篇只出现一套百分数（来自 SEER 表的分期匹配行），没有重复出现
        - 没有编造任何表中不存在的数字
        """)

        for attempt in range(2):
            try:
                res = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=180.0, max_retries=2
                )
                return remove_think_tags(res.content).strip()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        return "（预后生存率数据提取失败，请参考 SEER/NCDB 公开数据。）"
