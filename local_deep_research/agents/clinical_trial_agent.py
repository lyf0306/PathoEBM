import asyncio
import logging
import re

from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags, strip_llm_preamble
from ..prompts import prompt_manager

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
        # ── 日志：输入证据摘要 ──
        logger.info("=" * 60)
        logger.info("[Agent1 输入] current_knowledge 长度=%d 字符", len(current_knowledge))
        # 前 500 字符和后 500 字符，了解证据结构
        logger.info("[Agent1 输入] 开头500字:\n%s", current_knowledge[:500])
        logger.info("[Agent1 输入] 末尾500字:\n%s", current_knowledge[-500:])
        # 统计 [^^n] 引用数量
        ref_count = len(re.findall(r'\[\^\^\d+]', current_knowledge))
        logger.info("[Agent1 输入] [^^n] 引用数量=%d", ref_count)

        # Read reference map from ContextBus if available
        ref_map_str = ""
        if self.context_bus:
            ref_msgs = await self.context_bus.get_by_type("reference_map")
            if ref_msgs:
                ref_map_str = ref_msgs[-1]["content"][:3000]

        ref_section = ""
        ref_restriction = ""
        if ref_map_str:
            ref_section = f"【可用来源引用映射】：\n{ref_map_str}"
            ref_restriction = """
        **【🛑 严格溯源】**：使用 [^^n] 引用时，必须判断其标题和内容是否与试验语义匹配。不匹配则禁止使用。"""

        # 👇 核心修复逻辑：【灯塔试验雷达扫描】
        # 由 Python 直接扫描文本，找出真正被检索到的灯塔试验
        lighthouse_trials = [
            "GOG-99", "PORTEC-1", "PORTEC-2", "PORTEC-3", "GOG-0258",
            "GOG-209", "NRG-GY018", "RUBY",
            "KEYNOTE-775", "PORTEC-4a"
        ]
        
        # 忽略大小写的精准匹配
        found_trials = []
        for trial in lighthouse_trials:
            if trial.lower() in current_knowledge.lower():
                found_trials.append(trial)
                
        # 根据扫描结果，提示已检测到的试验
        dynamic_radar_directive = ""
        if found_trials:
            detected_str = "、".join(found_trials)
            logger.info("[Agent1 雷达] 在证据中检测到核心试验: %s", detected_str)
            for t in found_trials:
                cnt = current_knowledge.lower().count(t.lower())
                logger.info("[Agent1 雷达]   %s 出现 %d 次", t, cnt)
            dynamic_radar_directive = f"""
        ⚠️ **【关键词扫描提醒】**：以下试验名出现在检索文本中：**{detected_str}**。
        这只是文字匹配提示。当某文献标题以该试验名开头或明确包含该试验名作为研究主体时，才认定为源文献。禁止将仅在其他文章引言中被顺带提及的试验名当作有效证据提取。
        """
        else:
            logger.info("[Agent1 雷达] 未在证据中检测到任何核心白名单试验名称。")

        prompt = prompt_manager.get("clinical_trial_agent").format(
            treatment_context=self.treatment_context,
            current_knowledge=current_knowledge,
            ref_section=ref_section,
            ref_restriction=ref_restriction,
            dynamic_radar_directive=dynamic_radar_directive,
        )

        logger.info("Agent 1 正在提取核心临床试验数据...")

        for attempt in range(2):
            try:
                response = await invoke_with_timeout_and_retry(
                    self.report_model, prompt, timeout=300.0, max_retries=3
                )
                raw = response.content
                # 日志：提取 think 内容
                think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
                if think_match:
                    logger.info("[Agent1 LLM 思考] 共 %d 字符:\n%s", len(think_match.group(1)), think_match.group(1)[:1500])
                else:
                    logger.info("[Agent1 LLM 思考] 未输出 <think> 标签")

                # 日志：raw response（前 3000 字符）
                logger.info("[Agent1 LLM 原始响应] 共 %d 字符，开头:\n%s", len(raw), raw[:3000])

                content = remove_think_tags(raw).strip()
                content = strip_llm_preamble(content)
                if content:
                    logger.info("[Agent1 最终输出] 共 %d 字符:\n%s", len(content), content[:2500])
                    return content
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Agent 1 提取失败 (尝试 {attempt+1}): {e}")

        return "未发现完全匹配该患者分子分型与分期的核心前瞻性试验数据。"