import json
import logging
from datetime import datetime
from typing import List

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..search_system_support import safe_json_from_text, extract_and_convert_list

logger = logging.getLogger(__name__)


class SearchPlanner:
    """
    Generates follow-up search questions for iterative evidence retrieval.
    Uses the "三足鼎立" structured strategy to distribute query quotas.
    """
    def __init__(self, tool_planning_model, structured_task: dict, questions_per_iteration: int):
        self.tool_planning_model = tool_planning_model
        self.structured_task = structured_task
        self.questions_per_iteration = questions_per_iteration

    async def generate_questions(self, current_knowledge: str, query: str) -> List[str]:
        now = datetime.now().strftime("%Y-%m-%d")
        structured_data = json.dumps(self.structured_task, ensure_ascii=False, indent=2)

        prompt = f"""
        你是一名顶级的"循证医学检索转化专家"（Clinical Evidence Coordinator）。
        上游系统已经为你提取好了【患者的核心结构化数据】，包含初步治疗方案、合并症以及PICO问题。

        【🚨 核心检索策略：三足鼎立（极度重要！）】：
        你的并发检索额度上限为 {self.questions_per_iteration} 个检索词。你必须充分利用这些额度，**绝对不能只查 PICO 问题！**你的检索词必须全面覆盖以下三个维度：

        1. **【灯塔试验复核与预后长期随访（占用 2-4 个额度）】**：
           - 针对上游报告中提及的治疗方案，检索支撑该方案的最核心临床试验（如 PORTEC, GOG, NRG 系列等）的最新随访生存数据。
           - ⚠️ 检索词示例：PORTEC-3 AND ("survival"[Title/Abstract] OR "outcomes"[Title/Abstract]) AND 2018:2026[dp]
        2. **【高价值 PICO 问题精准查证（占用 2-4 个额度）】**：
           - 提取结构化数据中遗留的 PICO 问题，转化为具体的 PubMed 检索词。重点关注特定分子分型（如 NSMP, p53abn）对预后的细分影响或前瞻性证据。
           - ⚠️ 检索词示例：Endometrial cancer AND NSMP AND "recurrence risk" AND 2020:2026[dp]
        （注：上游提供的初步方案已通过严格的指南毒理审核，**你绝对不需要**再生成验证药物毒性、寻找卡铂/顺铂替代方案等纠偏性质的检索词！）

        【上游传入的结构化数据】：
        {structured_data}

        【当前已验证的知识】：
        {current_knowledge}

        【强制输出 JSON 格式】：
        你必须输出合法的 JSON 格式。
        ```json
        {{
            "discovery_reasoning": "简述你如何分配这 {self.questions_per_iteration} 个检索额度，以确保全面覆盖灯塔试验、主干方案审阅和合并症质询。",
            "sub_queries": [
                "灯塔试验检索词1",
                "灯塔试验检索词2 (若有)",
                "主干方案具体放化疗参数审阅词1",
                "主干方案具体放化疗参数审阅词2",
                "合并症/毒性替代/PICO检索词1"
            ]
        }}
        ```
        💡 请直接输出 JSON 代码块，确保充分利用 {self.questions_per_iteration} 个并发额度，不可遗漏对主干方案的深度核查！
        """

        try:
            response = await invoke_with_timeout_and_retry(
                self.tool_planning_model, prompt, timeout=1200.0, max_retries=3
            )

            response_text = remove_think_tags(response.content)

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                parsed = safe_json_from_text(response_text[json_start:json_end])
                if parsed:
                    challenge = parsed.get("clinical_challenge", "")
                    if challenge:
                        logger.info(f"翻译官临床质询: {challenge}")

                    questions = parsed.get("sub_queries", [])
                    return questions[:self.questions_per_iteration]

            return extract_and_convert_list(response_text)[:self.questions_per_iteration]

        except Exception as e:
            logger.warning(f"生成检索问题失败: {e}")
            return []
