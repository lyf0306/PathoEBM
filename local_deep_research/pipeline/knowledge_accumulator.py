"""
Thread-safe knowledge accumulator for the async search pipeline.

Collects synthesis results from independently-completing subagents and
provides a consistent snapshot for the coverage monitor and final report
generator — no global barrier required.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class KnowledgeAccumulator:
    """Gathers subagent outputs as they finish, with async-safe access.

    Each subagent calls add() when its ReAct loop completes.  The
    coverage monitor calls get_snapshot() to read current accumulated
    knowledge without blocking any in-flight subagent.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._entries: list[dict[str, Any]] = []

    async def add(
        self,
        query: str,
        synthesis: str,
        meta_type: str = "flat",
        meta_label: str = "",
        sufficient: bool = True,
    ) -> None:
        """Append one subagent's finished synthesis.

        All parameters are captured so the final report assembler can
        reconstruct the same structured output (trial / PICO / comorb
        sections) that the old fork-join loop produced.
        """
        async with self._lock:
            self._entries.append({
                "query": query,
                "synthesis": synthesis,
                "meta_type": meta_type,
                "meta_label": meta_label,
                "sufficient": sufficient,
            })
            logger.debug(
                "Accumulator: +1 entry [%s] %s (total=%d)",
                meta_type, str(meta_label)[:60], len(self._entries),
            )

    async def get_snapshot(self) -> list[dict[str, Any]]:
        """Return a shallow copy of all entries collected so far."""
        async with self._lock:
            return list(self._entries)

    async def get_flat_knowledge(self) -> str:
        """Return all syntheses concatenated as a single string.

        Format matches the old ``current_knowledge`` variable so the
        existing coverage-check and report-generation prompts still work.
        """
        async with self._lock:
            parts: list[str] = []
            for i, e in enumerate(self._entries):
                parts.append(f"\n\n### 检索项 {i+1}: {e['query']}\n{e['synthesis']}")
            return "".join(parts)

    @property
    def count(self) -> int:
        return len(self._entries)
