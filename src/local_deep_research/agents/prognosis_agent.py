import logging
import textwrap

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags

logger = logging.getLogger(__name__)


class PrognosisAgent:
    """
    Agent 3: Prognosis data extraction specialist.
    Extracts survival rate data from three sources:
      - Skill reference data (SEER/NCDB authoritative tables)
      - Population-level PubMed literature (NCDB/SEER studies)
      - Molecular-specific PubMed literature
    """
    def __init__(self, report_model, treatment_context: str, context_bus=None):
        self.report_model = report_model
        self.treatment_context = treatment_context
        self.context_bus = context_bus

    async def run(self, skill_data="", population_data="", molecular_data="") -> str:
        # Read reference map from ContextBus if available
        ref_map_str = ""
        if self.context_bus:
            ref_msgs = await self.context_bus.get_by_type("reference_map")
            if ref_msgs:
                ref_map_str = ref_msgs[-1]["content"][:3000]

        ref_section = ""
        if ref_map_str:
            ref_section = f"""
        【可用来源引用映射】（仅以下 [^^n] 编号有效）：
        {ref_map_str}
        """

        prompt = textwrap.dedent(f"""
        你是一位严谨的肿瘤流行病学数据提取专家。
        任务：综合以下三个来源的预后数据，提取患者相关的预后生存率数据。

        【当前患者真实病情草稿】：
        {self.treatment_context}

        【来源一：权威参考基线（SEER/NCDB 生存率表）】：
        {skill_data or "（未提供）"}

        【来源二：大样本人口学预后文献（NCDB/SEER PubMed 检索结果）】：
        {population_data or "（未提供）"}

        【来源三：分子分型精准预后文献】：
        {molecular_data or "（未提供）"}
        {ref_section}
        【🚨 提取红线】：
        1. **【权威数据为基，文献数据为证】**：来源一（生存率表）的分期匹配生存率作为**核心基准数据**必须优先采用。在此基础之上，**必须同时提取**来源二和来源三中与患者分期/分子分型相关的支持性数据（如HR、RFS、亚组分析结果），并保留其 [^^n] 文献角标，形成"基线数据 + 文献佐证"的完整预后描述。
        2. **【FIGO 版本匹配】**：检查患者诊断中写明的 FIGO 版本（如 "FIGO 2023" 或 "FIGO 2009"）。如果没有明确写出版本号，则根据分期格式判断——带有 IA1/IA2/IA3、IIIA1/IIIA2、IIIC1-i 等细分亚期的为 2023 版，仅有 I/II/IIIA/IIIB 等大分期且有 IIIC1/IIIC2 子分期的为 2009 版。**【必须使用与患者版本匹配的列】**，禁止混用。
        3. **【分期包容性】**：如果文献提供了大类分期（如 Stage III，高危患者总体）的生存率，你必须提取并注明，严禁因为没有精确匹配 IIIA1 子分期而漏杀。
        4. **【零串台幻觉】**：你只能提取与子宫内膜癌相关的预后数据。如果专属库里没找到有效数据或混入了无关疾病，必须直接输出："当前检索文献未返回有效的特定预后生存数据。"绝对禁止张冠李戴捏造数字！
        5. **【强制文献角标】**：来源二和来源三中每条数据必须保留其带 ^^ 的角标 [^^n]。输出格式中每个文献数据点后紧跟其角标。每个 [^^n] 必须独立用空格分隔，**绝对禁止**写成 [^^8,^^12] 或 [^^n] 写错为 [^^^n]。{'【可用来源引用映射】列出了所有有效的 [^^n] 编号，只允许使用这些编号！' if ref_map_str else ''}
        6. **【禁止跨段重复】**：不同章节之间禁止复用同一文献的同一组数据。每个 [^^n] 在每个章节中最多出现一次，如果该文献没有该章节的新数据则不要罗列。章节间如有重复数据，只放在最相关的章节，后续章节不再重复。
        7. **【丰富输出格式】**：不要只输出一句话。按以下结构组织：
           - **生存率基线**：[SEER数据，注明分期和版本]
           - **文献佐证**：[关键文献数据点，带 [^^n] 角标]
           - **分子分型预后**：[分子相关数据，带 [^^n] 角标]
        8. **【强制包含 3 年和 5 年生存率】**：**每项生存率数据必须同时包含 3 年 OS 和 5 年 OS 两个指标**（如来源中有此数据）。例如："5 年 OS 为 84%（95%CI: 80%-87%），3 年 OS 为 89%"。如果来源中缺少某个指标，注明"未报告"。
        """)

        for attempt in range(2):
            try:
                res = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=180.0, max_retries=2
                )
                return remove_think_tags(res.content).strip()
            except Exception:
                pass
        return "暂无具体预后生存率数据。"
