"""PromptManager — 从 _settings/prompts/{version}/ 加载提示词模板。

用法:
    from prompts import prompt_manager
    prompt = prompt_manager.get("search_planner")
    formatted = prompt.format(questions_per_iteration=6, trial_targets="PORTEC-3")
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent  # local_deep_research/
PROMPT_ROOT = PROJECT_ROOT / "_settings" / "prompts"


class PromptManager:
    """单例提示词管理器，惰性加载 + 内存缓存 + 版本切换。"""

    def __init__(self):
        self._version: str = ""
        self._cache: dict[str, str] = {}

    @property
    def version(self) -> str:
        if not self._version:
            self._version = self._read_active_version()
        return self._version

    def _read_active_version(self) -> str:
        try:
            from ..config import get_secret
            v = get_secret("prompts", "version", "v1")
            return v if v.startswith("v") else f"v{v}"
        except Exception:
            return "v1"

    def get(self, name: str) -> str:
        """返回未格式化的提示词模板字符串（调用方用 .format() 自行填充变量）。"""
        if name not in self._cache:
            path = PROMPT_ROOT / self.version / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Prompt '{name}' not found at {path}. "
                    f"Active version: {self.version}"
                )
            self._cache[name] = path.read_text(encoding="utf-8")
        return self._cache[name]

    def reload(self):
        """清空缓存并重新读取版本号（用于热更新）。"""
        self._version = ""
        self._cache.clear()
        logger.info(f"PromptManager reloaded (version={self.version})")


# 全局单例
prompt_manager = PromptManager()
