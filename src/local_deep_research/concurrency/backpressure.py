"""背压控制 — 限制并发协程数，防止资源耗尽。"""

import asyncio
import logging
from typing import Any, Coroutine, List, AsyncIterator

logger = logging.getLogger(__name__)


async def bounded_gather(*coros, concurrency: int = 10,
                         return_exceptions: bool = True) -> list:
    """限制并发的 asyncio.gather。

    用 Semaphore 控制同时执行的协程数，超出限制的自动排队等待。

    Args:
        *coros: 协程对象列表
        concurrency: 同时执行上限
        return_exceptions: 异常是否返回而非抛出
    """
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(coro):
        async with sem:
            return await coro

    return await asyncio.gather(
        *(_bounded(c) for c in coros), return_exceptions=return_exceptions
    )


class BoundedTaskQueue:
    """有界异步任务队列，满时生产者阻塞（背压）。"""

    def __init__(self, maxsize: int = 50):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._results: list = []

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def full(self) -> bool:
        return self._queue.full()

    async def put(self, coro_factory):
        """放入任务工厂函数，队列满时等待。"""
        await self._queue.put(coro_factory)

    async def process(self, concurrency: int = 8) -> list:
        """消费队列中所有任务，限制并发数。"""
        sem = asyncio.Semaphore(concurrency)
        results = []

        async def _worker(item):
            async with sem:
                if callable(item):
                    return await item()
                return await item

        tasks = []
        while not self._queue.empty():
            item = await self._queue.get()
            tasks.append(asyncio.create_task(_worker(item)))

        if tasks:
            raw = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(raw)

        return results
