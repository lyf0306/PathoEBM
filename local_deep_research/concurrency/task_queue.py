"""
Task queue abstraction for the async search pipeline.

Provides a Protocol-based interface so the pipeline can work with either
an in-process asyncio.Queue (default) or a Redis-backed queue for
horizontal scaling across multiple worker processes.

Usage:
    from local_deep_research.concurrency.task_queue import AsyncioTaskQueue

    queue = AsyncioTaskQueue()
    await queue.put(("pico", "cervical cancer AND pembrolizumab"))
    item = await queue.get()  # -> ("pico", "cervical cancer AND pembrolizumab")
    queue.task_done()
"""

import asyncio
import logging
import time
from typing import Any, Protocol, Optional

logger = logging.getLogger(__name__)


class TaskQueue(Protocol):
    """Protocol for pluggable task queues.

    Implement this protocol to swap in Redis / RabbitMQ / etc.
    without changing pipeline business logic.
    """

    async def put(self, item: Any) -> None: ...

    async def get(self) -> Any: ...

    def task_done(self) -> None: ...

    async def join(self) -> None: ...

    def empty(self) -> bool: ...

    def qsize(self) -> int: ...


class AsyncioTaskQueue:
    """In-process task queue backed by asyncio.Queue.

    Suitable for single-process deployments.  Zero operational overhead,
    zero latency, no external dependencies.

    To scale horizontally, replace with a RedisTaskQueue implementing
    the same interface (TaskQueue Protocol).
    """

    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._total_completed: int = 0
        self._total_enqueued: int = 0

    async def put(self, item: Any) -> None:
        self._total_enqueued += 1
        await self._queue.put(item)

    async def get(self) -> Any:
        return await self._queue.get()

    def task_done(self) -> None:
        self._total_completed += 1
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def total_completed(self) -> int:
        return self._total_completed

    @property
    def total_enqueued(self) -> int:
        return self._total_enqueued
