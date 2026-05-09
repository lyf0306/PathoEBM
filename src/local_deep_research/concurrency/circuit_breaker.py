"""熔断器 — 三态状态机，防止对故障下游持续重试。

状态转换：
  CLOSED → (连续失败 N 次) → OPEN
  OPEN   → (recovery_timeout 后) → HALF_OPEN
  HALF_OPEN → 成功 → CLOSED
  HALF_OPEN → 失败 → OPEN
"""

import asyncio
import time
import threading
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CONFIGS: Dict[str, dict] = {
    "llm":       {"failure_threshold": 5, "recovery_timeout": 60.0, "window_seconds": 120.0},
    "mcp_pubmed":   {"failure_threshold": 5, "recovery_timeout": 30.0, "window_seconds": 60.0},
    "mcp_clinical": {"failure_threshold": 5, "recovery_timeout": 30.0, "window_seconds": 60.0},
    "mcp_fda":      {"failure_threshold": 5, "recovery_timeout": 30.0, "window_seconds": 60.0},
    "mcp_ncbi":     {"failure_threshold": 5, "recovery_timeout": 30.0, "window_seconds": 60.0},
    "default":      {"failure_threshold": 5, "recovery_timeout": 30.0, "window_seconds": 60.0},
}

State = str
CLOSED: State = "CLOSED"
OPEN: State = "OPEN"
HALF_OPEN: State = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    """熔断器打开时抛出的异常。"""
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"熔断器 [{name}] 已打开，拒绝请求")


class CircuitBreaker:
    """单服务熔断器，线程安全。"""

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0, window_seconds: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.window_seconds = window_seconds

        self._state: State = CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0
        self._lock = threading.RLock()

    # ---- 状态查询 ----

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def is_open(self) -> bool:
        with self._lock:
            return self._current_state() == OPEN

    # ---- 内部状态机 ----

    def _current_state(self) -> State:
        """基于时间和计数器计算当前状态。"""
        now = time.monotonic()

        if self._state == OPEN:
            if now - self._opened_at >= self.recovery_timeout:
                self._state = HALF_OPEN
                logger.info(f"熔断器 [{self.name}] OPEN → HALF_OPEN（恢复期到）")
            return self._state

        if self._state == CLOSED:
            # 过期失败不计
            if self._failure_count > 0 and now - self._last_failure_time > self.window_seconds:
                self._failure_count = 0
                logger.debug(f"熔断器 [{self.name}] 失败计数过期，归零")
            return CLOSED

        if self._state == HALF_OPEN:
            return HALF_OPEN

        return CLOSED

    # ---- 记录结果 ----

    def record_success(self):
        with self._lock:
            prev = self._current_state()
            if prev == HALF_OPEN:
                logger.info(f"熔断器 [{self.name}] HALF_OPEN → CLOSED（探测成功）")
            self._state = CLOSED
            self._failure_count = 0

    def record_failure(self):
        with self._lock:
            now = time.monotonic()
            prev = self._current_state()

            if prev == CLOSED:
                self._failure_count += 1
                self._last_failure_time = now
                logger.warning(
                    f"熔断器 [{self.name}] 失败 {self._failure_count}/{self.failure_threshold}"
                )
                if self._failure_count >= self.failure_threshold:
                    self._state = OPEN
                    self._opened_at = now
                    logger.error(f"熔断器 [{self.name}] CLOSED → OPEN（达到阈值）")

            elif prev == HALF_OPEN:
                self._state = OPEN
                self._opened_at = now
                self._failure_count = 1
                self._last_failure_time = now
                logger.error(f"熔断器 [{self.name}] HALF_OPEN → OPEN（探测失败）")

            # 已是 OPEN：累加计数但不改变状态

    # ---- 便捷调用 ----

    async def call(self, coro_factory, *args, **kwargs):
        """包装异步调用，自动处理熔断逻辑。

        注意：传入 coro_factory 而非协程对象，避免熔断器 OPEN 时创建无用协程。
        """
        with self._lock:
            st = self._current_state()
        if st == OPEN:
            raise CircuitBreakerOpenError(self.name)

        try:
            result = await coro_factory(*args, **kwargs)
            self.record_success()
            return result
        except (CircuitBreakerOpenError, asyncio.CancelledError):
            raise
        except Exception:
            self.record_failure()
            raise


class CircuitBreakerRegistry:
    """全局熔断器注册表，按名称管理实例。"""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                cfg = DEFAULT_CONFIGS.get(name, DEFAULT_CONFIGS["default"])
                self._breakers[name] = CircuitBreaker(name=name, **cfg)
                logger.debug(f"创建熔断器 [{name}]: {cfg}")
            return self._breakers[name]

    def get_all(self) -> Dict[str, CircuitBreaker]:
        with self._lock:
            return dict(self._breakers)

    def reset_all(self):
        with self._lock:
            for b in self._breakers.values():
                b._state = CLOSED
                b._failure_count = 0
            self._breakers.clear()


# 模块级单例
_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker(name: str = "default") -> CircuitBreaker:
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry.get(name)
