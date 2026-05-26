import json
import logging
import textwrap
from typing import Dict, List

from ..utilities.search_utilities import invoke_with_timeout_and_retry, remove_think_tags
from ..search_system_support import safe_json_from_text
from ..prompts import prompt_manager

logger = logging.getLogger(__name__)


class KnowledgeProcessor:
    """
    Handles evidence synthesis, knowledge extraction, and structured data parsing.
    """
    def __init__(self, model, fast_model):
        self.model = model
        self.fast_model = fast_model

    async def answer_query(self, current_knowledge: str, query: str,
                           current_iteration: int, max_iterations: int,
                           existing_refs: list) -> str:
        refs_block = "\n".join(existing_refs) or "*None yet*"

        prompt = prompt_manager.get("knowledge_answer").format(
            current_knowledge=current_knowledge,
            refs_block=refs_block,
        )

        try:
            response = await invoke_with_timeout_and_retry(
                self.model, prompt, timeout=1200.0, max_retries=3
            )
            return remove_think_tags(response.content)
        except Exception as e:
            logger.error(f"证据合成失败: {e}")
            return "Error synthesizing evidence."

    async def extract_knowledge(self, facts_md: str, refs_in_round: List[Dict]):
        prompt = prompt_manager.get("knowledge_extract").format(
            facts_md=facts_md,
            refs_in_round=json.dumps(refs_in_round),
        )
        try:
            resp = await invoke_with_timeout_and_retry(self.model, prompt, timeout=1200.0)
            cleaned_content = remove_think_tags(resp.content)
            data = safe_json_from_text(cleaned_content) or {}
            return data.get("key_information", ""), data.get("cleaned_refs", [])
        except Exception:
            return facts_md, refs_in_round

    async def process_multiple_chunks(self, query: str, current_key_info: str) -> str:
        if not current_key_info:
            return current_key_info

        prompt = prompt_manager.get("knowledge_multichunk").format(
            current_key_info=current_key_info,
        )
        try:
            resp = await invoke_with_timeout_and_retry(self.model, prompt, timeout=1200.0)
            return remove_think_tags(resp.content)
        except Exception:
            return current_key_info

    async def extract_structured_data(self, raw_text: str, source_type: str, query: str) -> str:
        if source_type == "clinicaltrials":
            prompt = prompt_manager.get("knowledge_ctgov").format(
                query=query,
                raw_text=raw_text[:20000],
            )
        else:
            prompt = prompt_manager.get("knowledge_fda").format(
                query=query,
                raw_text=raw_text[:20000],
            )

        try:
            resp = await invoke_with_timeout_and_retry(self.model, prompt, timeout=800.0)
            return remove_think_tags(resp.content).strip()
        except Exception as e:
            logger.error(f"结构化数据提取失败: {e}")
            return "数据提取失败，未发现有效量化信息。"
