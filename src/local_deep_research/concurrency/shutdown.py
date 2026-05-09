"""优雅关闭管理器 — 处理信号、取消任务、释放资源。

在 Windows 上 signal 的工作方式与 Unix 不同，但 signal.signal()
在 Windows 上也支持 SIGINT（Ctrl+C）和 SIGTERM（跨平台）。
"""

import asyncio
import logging
import signal
from typing import Optional, Set

logger = logging.getLogger(__name__)


class ShutdownManager:
    """单例管理器，协调多组件的优雅关闭。

    用法：
        mgr = get_shutdown_manager()
        install_signal_handlers()

        try:
            # 主循环中定期检查
            if mgr.shutdown_event.is_set():
                return partial_result
        finally:
            await mgr.cleanup(grace_period=30.0)
    """

    def __init__(self):
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._cleanup_callbacks: list = []
        self._cleaned_up: bool = False

    def register_task(self, task: asyncio.Task):
        """注册需要追踪的任务。"""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def register_cleanup(self, callback):
        """注册关闭时需调用的清理回调（async callable）。"""
        self._cleanup_callbacks.append(callback)

    async def cleanup(self, grace_period: float = 30.0):
        """执行完整清理流程。"""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        logger.info("开始优雅关闭...")

        # 1. 设置关闭信号
        self.shutdown_event.set()

        # 2. 取消所有追踪的任务
        for task in list(self._tasks):
            if not task.done():
                task.cancel()

        # 3. 等待任务完成或超时
        if self._tasks:
            done, pending = await asyncio.wait(
                self._tasks, timeout=grace_period
            )
            if pending:
                logger.warning(f"{len(pending)} 个任务未能在 {grace_period}s 内完成，强制取消")
                for task in pending:
                    task.cancel()

        # 4. 执行清理回调
        for cb in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb()
                else:
                    cb()
            except Exception as e:
                logger.error(f"清理回调失败: {e}")

        # 5. 关闭 HTTP 连接池
        try:
            from .connection_pool import close_shared_http_client
            await close_shared_http_client()
        except Exception as e:
            logger.debug(f"关闭 HTTP 客户端时: {e}")

        logger.info("优雅关闭完成。")


# ---- 模块级单例 ----

_manager: Optional[ShutdownManager] = None


def get_shutdown_manager() -> ShutdownManager:
    global _manager
    if _manager is None:
        _manager = ShutdownManager()
    return _manager


# ---- 信号处理 ----

def install_signal_handlers(loop: Optional[asyncio.AbstractEventLoop] = None):
    """安装 SIGINT / SIGTERM 处理器，触发优雅关闭。"""
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("无运行中的事件循环，跳过信号处理安装")
            return

    mgr = get_shutdown_manager()

    def _handle(sig, frame):
        name = signal.Signals(sig).name
        logger.warning(f"收到信号 {name}，触发优雅关闭...")
        loop.call_soon_threadsafe(mgr.shutdown_event.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
            logger.debug(f"已注册信号处理器: {signal.Signals(sig).name}")
        except Exception as e:
            logger.debug(f"信号 {sig} 注册失败（可能不受平台支持）: {e}")
