import logging
import re
import textwrap

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags

logger = logging.getLogger(__name__)


class MDTReportAgent:
    """
    Agent 2: Endometrial cancer MDT chief writer.
    Synthesizes all agent outputs into the final MDT report.
    """
    def __init__(self, report_model, treatment_context: str, structured_task: dict, context_bus=None):
        self.report_model = report_model
        self.treatment_context = treatment_context
        self.structured_task = structured_task
        self.context_bus = context_bus

    async def run(self, trial_analysis: str, followup_plan: str, prognosis_data: str) -> str:
        # Read reference map from ContextBus if available
        ref_map_str = ""
        if self.context_bus:
            ref_msgs = await self.context_bus.get_by_type("reference_map")
            if ref_msgs:
                ref_map_str = ref_msgs[-1]["content"][:3000]
        incidental_findings = self.structured_task.get("incidental_findings", [])
        incidental_str = "、".join(incidental_findings) if incidental_findings else "无"

        # Extract guideline text from treatment context
        original_guideline_text = "## 二、 核心指南与共识详尽解析\n（未能在草稿中匹配到指南解析部分）"
        guideline_match = re.search(
            r'(## 二、 核心指南与共识详尽解析.*?)(?=\n## 三、|\n## 四、|\Z)',
            self.treatment_context, re.DOTALL
        )
        if guideline_match:
            original_guideline_text = guideline_match.group(1).strip()

        ref_section = ""
        ref_rule = ""
        if ref_map_str:
            ref_section = f"""
        【可用来源引用映射】（最终报告中只允许使用以下 [^^n] 编号）：
        {ref_map_str}
        """
            ref_rule = "\n        7. **【引用格式与来源限制】**：最终报告中的 [^^n] 角标必须来自【可用来源引用映射】中的编号，绝对禁止编造不存在的引用编号！每个 [^^n] 必须独立用空格分隔，**绝对禁止**写成 [^^8,^^12] 或 [^^8, ^^12] 等逗号合并格式。"

        main_prompt = textwrap.dedent(f"""
        你是一名具备顶尖国际视野的妇科肿瘤 MDT 首席专家。
        【初步会诊草稿】：
        {self.treatment_context}
        【助手整理的临床试验深度解析】：
        {trial_analysis}
        【预后专员提取的客观生存事实】（请将其作为不可辩驳的铁证直接采用！）：
        {prognosis_data}
        【翻译官拦截的次要异常】（须进行常规专科转诊）：
        {incidental_str}
        {ref_section}
        你的任务是：输出最终版的 MDT 报告主干。

        ## 🛑 首席专家"元临床思维"法则 (核心红线)
        1. **【尊重原定权威方案】**：草稿中的主干治疗方案已经过权威校验，你不需要对其药物毒性或禁忌症进行纠错。请直接继承该主要方案。
        2. **【个体化降级绝对优先原则（Toxicity Override）】**：仔细阅读【初步会诊草稿】，如果草稿中因为患者的严重合并症明确提出了**"取消常规放疗"、"降级化疗强度"、"姑息性治疗"**等妥协策略，你**绝对禁止**在最终结论中强行恢复标准的高强度指南方案！
        3. **【临床试验与PICO解答的无缝融合】**：在最终结论中，结合上方《核心临床试验循证解析》最新数据以及PICO问题，给出循证视角的最终确板意见。
        4. **【预后数据强制量化】**：在【预后分析】中，必须直接引用【预后专员提取的客观事实】中的具体数据和原封不动的角标（如 [^^11]）。
        5. **【合并症的常规转诊】**：将草稿中提到的重大合并症和次要异常，单独列为带有数字序号的项目给出相关科室的转诊建议即可。
        6. **【三占位符机制】**：你绝对不要自己写第二部分、第三部分和第四部分的主体！
           - 在第二部分原封不动输出 {{{{GUIDELINE_PLACEHOLDER}}}}
           - 在第三部分原封不动输出 {{{{TRIAL_PLACEHOLDER}}}}
           - 在第四部分原封不动输出 {{{{FOLLOWUP_PLACEHOLDER}}}}
        {ref_rule}
        ## 📝 必须使用的固定输出模板：

        # 妇科肿瘤 MDT 最终版会诊报告

        ## 一、 病情分析
        ### 1. 病历及病理摘要
        【在此处详尽总结患者病情、高危因素及FIGO分期。🚨 注意保留诊断原文中的FIGO版本号（2009/2023），禁止擅自转换分期版本！】

        {{{{GUIDELINE_PLACEHOLDER}}}}

        ### 3. 核心临床试验及 PICO 循证解析
        {{{{TRIAL_PLACEHOLDER}}}}

        ## 二、 术后处理
        【请直接输出一个连续的数字列表（1、2、3...）】：
        1、 **肿瘤专科最终方案**：【🚨 极其重要：请模仿真实临床医生简洁干练的会诊风格！】
           - 必须包含：**核心方案名称**（如 TC方案 或 单药卡铂）和 **总周期数**（如 6次 或 6周期）。
           - 必须包含：**近期疗效评估规划**（如"建议完成化疗结束3个月复查盆腔增强MRI/上腹部增强CT/两肺平扫CT或PET-CT"）。
           - 💡 **降级表达**：如果是因严重并发症妥协的方案，请自然地在句首带过即可。
           - ❌ **绝对禁止**：不要像药剂师一样罗列"静脉滴注、AUC 5、175mg/m2"等琐碎的处方级给药细节！
           - ✅ **正确示例**："鉴于患者严重心功能不全，建议取消外照射放疗，行单药卡铂化疗6次。建议完成化疗结束3个月复查盆腔增强MRI/上腹部增强CT及胸部平扫CT。"
        2、 **分子标志物追踪**：追踪分子分型结果，结合最新 PICO 循证结论指导后续风险管理（如果已明确患者分子分型，这一条可以省略）。
        3、 **【合并症/次要异常 1 名称，如：冠心病】**：建议转诊至【对应科室】门诊进一步评估与随诊，【给出简明建议，如评估放化疗风险】。
        4、 **【合并症/次要异常 2 名称，如：浅表性胃炎】**：建议转诊至消化内科门诊进一步评估与处理。
        【⚠️ 请继续用 5、6、7... 顺延列出患者所有的重大合并症和次要异常，每一个异常必须单独占一条！若全部罗列完毕则结束该段。】

        ## 三、 预后分析
        【结合患者特有高危因素，将上方预后专员提取的数据极其精简干练地串联进来，标明具体数据和文献角标】

        ## 四、 随访方案
        {{{{FOLLOWUP_PLACEHOLDER}}}}

        💡 请先在 <think> 标签内思考确认无误后再输出正文模板！
        """)

        logger.info("Agent 2 正在统筹生成 MDT 报告主干并确保全局逻辑一致性...")

        content = ""
        for attempt in range(3):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, main_prompt, timeout=1200.0, max_retries=3
                )
                content = remove_think_tags(response.content)
                for cut_word in ["## 五", "# 五", "参考文献", "References"]:
                    if cut_word in content:
                        content = content.split(cut_word)[0].strip()
                break
            except Exception as e:
                logger.error(f"Agent 2 生成报错: {e}")
                if attempt == 2:
                    content = "报告生成失败"

        # Replace placeholders
        content = self._replace_placeholders(content, original_guideline_text, trial_analysis, followup_plan)
        return content

    def _replace_placeholders(self, content: str, guideline_text: str, trial_analysis: str, followup_plan: str) -> str:
        # Demote guideline headings to fit within "## 一、 病情分析" context:
        #   ##  → ###    (so "## 二、" becomes "### 2.")
        #   ### → ####   (sub-headings within guideline section)
        demoted_lines = []
        for line in guideline_text.split('\n'):
            if line.startswith('## '):
                line = '#' + line  # ## → ###
                # Convert Chinese numbering "二、" to Arabic "2." for consistency
                line = re.sub(r'^###\s*[一二三四五六七八九十]+[、．.]?\s*', '### 2. ', line)
            elif line.startswith('### '):
                line = '#' + line  # ### → ####
            demoted_lines.append(line)
        demoted_guideline = '\n'.join(demoted_lines)

        if "{{GUIDELINE_PLACEHOLDER}}" in content:
            content = content.replace("{{GUIDELINE_PLACEHOLDER}}", demoted_guideline)
        else:
            content = content.replace("### 3. 核心临床试验", f"{demoted_guideline}\n\n### 3. 核心临床试验")

        if "{{TRIAL_PLACEHOLDER}}" in content:
            content = content.replace("{{TRIAL_PLACEHOLDER}}", trial_analysis)

        if "{{FOLLOWUP_PLACEHOLDER}}" in content:
            content = content.replace("{{FOLLOWUP_PLACEHOLDER}}", followup_plan)

        return content
