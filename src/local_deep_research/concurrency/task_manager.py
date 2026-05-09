"""任务隔离与安全 gather — 防止单任务失败影响整批。"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """单个协程的执行结果分类。"""
    success: bool
    value: Any = None
    error: Optional[Exception] = None
    timed_out: bool = False


async def gather_safe(*coros, return_exceptions: bool = True):
    """安全版 asyncio.gather，始终捕获异常并返回 (results, exceptions) 对。

    返回: List[TaskResult]，与原协程序号一一对应。
    """
    raw = await asyncio.gather(*coros, return_exceptions=True)
    results = []
    for item in raw:
        if isinstance(item, BaseException):
            if isinstance(item, asyncio.CancelledError):
                results.append(TaskResult(success=False, error=item))
            else:
                results.append(TaskResult(success=False, error=item))
        else:
            results.append(TaskResult(success=True, value=item))
    return results


class ErrorBoundary:
    """异步上下文管理器，捕获块内异常并抑制传播。

    Usage:
        async with ErrorBoundary("agent1") as boundary:
            await risky_operation()
        if boundary.error:
            logger.warning(f"agent1 失败: {boundary.error}")
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.error: Optional[Exception] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and exc_val is not None:
            self.error = exc_val
            logger.error(
                f"[ErrorBoundary:{self.name}] {exc_type.__name__}: {exc_val}"
            )
            return True  # 抑制异常传播
        return False
