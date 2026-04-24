import logging
import re
import textwrap

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags

logger = logging.getLogger(__name__)


class ClinicalTrialAgent:
    """
    Agent 1: Clinical trial evidence extraction.
    Takes accumulated search evidence and extracts the most important
    clinical trial data in a structured format for the MDT report.
    """
    def __init__(self, report_model, fast_model, treatment_context: str, context_bus=None):
        self.report_model = report_model
        self.fast_model = fast_model
        self.treatment_context = treatment_context
        self.context_bus = context_bus

    async def run(self, current_knowledge: str) -> str:
        # Read reference map from ContextBus if available
        ref_map_str = ""
        if self.context_bus:
            ref_msgs = await self.context_bus.get_by_type("reference_map")
            if ref_msgs:
                ref_map_str = ref_msgs[-1]["content"][:3000]

        ref_section = ""
        ref_restriction = ""
        if ref_map_str:
            ref_section = f"""
        【可用来源引用映射】（仅以下 [^^n] 编号有效）：
        {ref_map_str}
        """
            ref_restriction = """【可用来源引用映射】列出了所有有效的 [^^n] 编号，每个编号对应一篇具体的文献标题。**引用验证规则**：
- 使用 [^^n] 引用某一试验时，必须先判断该引用的标题和内容是否**语义上匹配**你所声称的试验
- ✅ 允许：标题为 "Phase III Randomized Trial of Carboplatin and Paclitaxel..." 的文献引用为 GOG-0258（即使标题不含 "GOG-0258"）
- ❌ 禁止：将年度综述、系统综述（如 "Gynecologic cancers in 2025: a year in review"）当作原始临床试验引用
- ❌ 禁止：引用的文献讨论的是完全不同的疾病或药物
如果找不到语义匹配的文献，说明该试验的原始论文未被检索到，直接跳过该项试验，**绝对禁止编造**！"""

        prompt = textwrap.dedent(f"""
        你是一名顶尖的妇科肿瘤循证医学分析专家。
        以下是为患者检索到的全部临床证据。请在全部证据中提取**最有临床指导价值的核心试验数据**，
        按临床重要性从高到低排序，输出最多 **3 项**。

        【当前患者真实病情草稿】：
        {self.treatment_context}

        【全部检索证据】：
        {current_knowledge}
        {ref_section}
        【🚨 提取红线】：
        1. **【🚨 灯塔临床试验导航库（核心白名单宪法）】 🚨**
           你【绝对禁止】凭借你自己的医学常识来判断什么试验重要！你必须【严格、且仅依据】以下名单进行比对匹配：
           ==================================================
           - **早期（I-II期）中低危及中高危**：GOG-99, PORTEC-1, PORTEC-2
           - **早期高危（I-II期伴高危因素）及局部晚期（III、IVA期）**：PORTEC-3, GOG-0258
           - **晚期（IVB期）及复发一线**：GOG-209, NRG-GY018, RUBY, ATTEND, DUO-E
           - **晚期复发（二线及以上）**：KEYNOTE-775
           - **分子分型降/升阶梯探索**：PORTEC-4a
           ==================================================
           匹配原则：如果检索证据中包含上述任一试验，**必须优先保留并排在最前面**，任何情况下都不得将其挤出 top 3。
        2. **【数据真空一票否决】**：如果文献中既没有具体的干预方案，也没有具体的生存获益数据，则跳过。
        3. **【防误杀豁免】**：若文献入组标准写"纳入 III 期"，则该患者（如 IIIA 期）完全符合！
        4. **【禁止重复枚举】**：纳入人群的标准只列出最核心的 3-5 条即可，**绝对禁止**重复罗列同质化条目。
        5. **【使用真实试验名】**：标题必须使用文献中的具体试验名（如 PORTEC-3、GOG-0258），回顾性分析则用数据库名（如 "NCDB 回顾性队列分析"）。
        6. **【角标格式铁律】**：引用证据时必须保留原文的 [^^n] 格式（双角号+数字），如 [^^8] [^^12]。**绝对禁止**去掉 ^^ 符号写成 [8] [12]。**绝对禁止**写成 [^^8,^^12] 或 [^^8, ^^12] 等逗号合并格式，每个 [^^n] 必须独立用空格分隔。{ref_restriction}

        【严格格式】（每项一条）：
        #### [具体的试验名称/编号] 是一项 [试验类型] [^^8]
        - **纳入人群**：[核心标准3-5条] 💡 **入组匹配校验**：[明确是否满足]
        - **分组与干预方案**：[参数详情]
        - **整体生存获益**：[精确百分比与HR]
        - **分子分型与亚组分析**：[获益差异]
        - **毒性与本患者指导意义**：[结合患者合并症]

        💡 请先在 <think> 标签内思考，确认已覆盖全部核心证据后再输出！
        """)

        logger.info("Agent 1 正在提取核心临床试验数据...")

        for attempt in range(2):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=300.0, max_retries=2
                )
                content = remove_think_tags(response.content).strip()
                if content:
                    return content
            except Exception as e:
                logger.warning(f"Agent 1 提取失败 (尝试 {attempt+1}): {e}")

        return "未发现完全匹配该患者分子分型与分期的核心前瞻性试验数据。"
