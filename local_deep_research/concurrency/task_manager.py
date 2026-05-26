"""任务隔离与安全 gather — 防止单任务失败影响整批。

自动将所有创建的任务注册到 ShutdownManager，确保关闭时能被取消和追踪。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """单个协程的执行结果分类。"""
    success: bool
    value: Any = None
    error: Optional[Exception] = None
    timed_out: bool = False


def _try_register_tasks(tasks: List[asyncio.Task]):
    """将任务列表注册到 ShutdownManager（失败时静默跳过）。"""
    try:
        from .shutdown import get_shutdown_manager
        mgr = get_shutdown_manager()
        for t in tasks:
            if not t.done():
                mgr.register_task(t)
    except Exception:
        pass


async def gather_safe(*coros, return_exceptions: bool = True):
    """安全版 asyncio.gather，自动注册任务并始终捕获异常。

    所有创建的 asyncio.Task 自动注册到 ShutdownManager，
    确保关闭时能被取消和追踪。

    返回: List[TaskResult]，与原协程序号一一对应。
    """
    tasks = [asyncio.create_task(c) for c in coros]
    _try_register_tasks(tasks)
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for item in raw:
        if isinstance(item, asyncio.CancelledError):
            # 取消所有未完成的兄弟任务后重新抛出，确保关闭信号能向上传播
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise item
        if isinstance(item, BaseException):
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
            if issubclass(exc_type, asyncio.CancelledError):
                self.error = exc_val
                logger.warning(
                    f"[ErrorBoundary:{self.name}] CancelledError — 传播取消信号"
                )
                return False  # NEVER suppress CancelledError
            self.error = exc_val
            logger.error(
                f"[ErrorBoundary:{self.name}] {exc_type.__name__}: {exc_val}"
            )
            return True  # 抑制异常传播
        return False
