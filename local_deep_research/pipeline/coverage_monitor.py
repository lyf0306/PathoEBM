"""
Background coverage monitor for the async search pipeline.

Runs alongside worker coroutines, periodically checking whether the
accumulated evidence sufficiently covers all clinical decision points.
When gaps are found, new queries are injected into the task queue
without blocking any in-flight subagent.
"""

import asyncio
import logging
from typing import Callable, Awaitable

from .knowledge_accumulator import KnowledgeAccumulator

logger = logging.getLogger(__name__)


class CoverageMonitor:
    """Periodic evidence-coverage checker — async, non-blocking.

    Instead of the old fork-join barrier (wait for ALL subagents → check
    coverage → launch new batch), this runs as a background coroutine
    that watches the accumulator and injects gap queries into the live
    queue whenever it spots a deficiency.

    The monitor stops when:
      - ``done_event`` is set (pipeline is shutting down)
      - ``max_checks`` rounds have been performed
      - Coverage returns sufficient AND the queue is empty
    """

    def __init__(
        self,
        accumulator: KnowledgeAccumulator,
        task_queue,  # TaskQueue protocol
        coverage_checker: Callable[[str], Awaitable[dict]],
        check_interval: int = 5,
        max_checks: int = 3,
        throttle_seconds: float = 30.0,
    ):
        self._acc = accumulator
        self._queue = task_queue
        self._checker = coverage_checker
        self._check_interval = check_interval
        self._max_checks = max_checks
        self._throttle = throttle_seconds

        self._checks_done = 0
        self._last_check_time = 0.0
        self._last_check_count = 0

    async def run(self, done_event: asyncio.Event) -> int:
        """Run the monitor loop until signalled to stop.

        Returns the number of coverage checks performed.
        """
        logger.info(
            "CoverageMonitor 启动: 每 %d 个任务或 %.0fs 检查一次, 最多 %d 轮",
            self._check_interval, self._throttle, self._max_checks,
        )

        while not done_event.is_set() and self._checks_done < self._max_checks:
            # Wait for trigger: N new completions OR T seconds elapsed
            await self._wait_for_trigger(done_event)

            if done_event.is_set():
                break

            # Run the coverage check
            self._checks_done += 1
            logger.info("CoverageMonitor: 第 %d/%d 轮全局覆盖度检查", self._checks_done, self._max_checks)

            knowledge = await self._acc.get_flat_knowledge()
            if not knowledge.strip():
                logger.info("CoverageMonitor: 尚无累积知识，跳过")
                continue

            try:
                result = await self._checker(knowledge)
            except Exception as e:
                logger.warning("CoverageMonitor: 覆盖度检查异常: %s", e)
                continue

            if result.get("sufficient", True):
                logger.info("CoverageMonitor: 证据覆盖度充足")
                if self._queue.empty():
                    logger.info("CoverageMonitor: 队列已空, 流水线完成")
                    break
                # Otherwise: still have queued tasks, let them finish
                continue

            gap_queries = result.get("gap_queries", [])[:3]
            if not gap_queries:
                logger.info("CoverageMonitor: 无明确证据缺口")
                continue

            # Dedup and inject
            injected = 0
            for gq in gap_queries:
                gq = gq.strip()
                if gq:
                    await self._queue.put(("flat", gq))
                    injected += 1

            logger.info(
                "CoverageMonitor: 发现 %d 个证据缺口, 已注入队列",
                injected,
            )

        logger.info("CoverageMonitor: 完成 (%d 轮检查)", self._checks_done)
        return self._checks_done

    async def _wait_for_trigger(self, done_event: asyncio.Event) -> None:
        """Block until either N tasks complete, T seconds elapse, or we're done."""
        while not done_event.is_set():
            current_count = self._acc.count
            new_since_last = current_count - self._last_check_count
            time_since_last = asyncio.get_event_loop().time() - self._last_check_time

            if new_since_last >= self._check_interval:
                break
            if self._last_check_time > 0 and time_since_last >= self._throttle:
                break

            # Sleep briefly and re-check
            try:
                await asyncio.wait_for(done_event.wait(), timeout=2.0)
                break  # done_event was set
            except asyncio.TimeoutError:
                pass  # re-evaluate trigger conditions

        self._last_check_time = asyncio.get_event_loop().time()
        self._last_check_count = self._acc.count
