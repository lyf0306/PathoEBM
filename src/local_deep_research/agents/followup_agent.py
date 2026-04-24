"""
Follow-up Agent — 基于 NCCN 指南生成个性化随访方案。

使用 NCCNFollowupSkill 加载 references/ 目录下的最新 NCCN 指南，
用户只需更新 skills/nccn_followup/references/ 下的 .md 文件即可。
"""

import logging
import textwrap
import os

from ..utilties.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..skills.nccn_followup import NCCNFollowupSkill

logger = logging.getLogger(__name__)


class FollowupAgent:
    """
    Agent 1.5: Follow-up plan generator.
    Uses NCCNFollowupSkill to generate guidelines-based personalized follow-up plans.
    """

    def __init__(self, report_model, treatment_context: str):
        self.report_model = report_model
        self.treatment_context = treatment_context
        # 初始化 NCCN Skill（自动加载 references/ 下的 .md 文件）
        skill_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "skills", "nccn_followup", "references"
        )
        self.nccn_skill = NCCNFollowupSkill(references_dir=skill_dir)

    async def run(self) -> str:
        """
        Generate follow-up plan using NCCN guidelines + patient context.

        Returns:
            Follow-up plan markdown text.
        """
        return await self.nccn_skill.generate(self.report_model, self.treatment_context)
