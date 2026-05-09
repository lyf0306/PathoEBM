"""令牌桶限流器 — 按下游服务配置速率，防止超出 API 限制。"""

import time
import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 各外部服务的速率限制（requests/s）
DEFAULT_RATES: Dict[str, dict] = {
    "pubmed":     {"rate": 3.0,  "capacity": 3},    # NCBI E-utilities 无 API key
    "clinical":   {"rate": 5.0,  "capacity": 5},    # ClinicalTrials.gov
    "fda":        {"rate": 5.0,  "capacity": 5},    # OpenFDA
    "ncbi":       {"rate": 3.0,  "capacity": 3},    # NCBI 基因等
    "openai":     {"rate": 500.0, "capacity": 100},  # 快速 LLM，高容量
    "deepseek":   {"rate": 500.0, "capacity": 100},
    "default":    {"rate": 10.0,  "capacity": 10},
}


class RateLimitExceeded(Exception):
    """限流超限异常（仅在非阻塞模式下抛出）。"""
    def __init__(self, service: str, wait_seconds: float):
        self.service = service
        self.wait_seconds = wait_seconds
        super().__init__(f"[{service}] 超过速率限制，需等待 {wait_seconds:.2f}s")


class TokenBucketRateLimiter:
    """令牌桶限流器，协程安全。"""

    def __init__(self, name: str, rate: float, capacity: float):
        self.name = name
        self.rate = rate          # tokens/s
        self.capacity = capacity   # 最大令牌数
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def available_tokens(self) -> float:
        """当前可用令牌数（只读近似值）。"""
        return self._tokens

    def _refill(self, now: float):
        """根据时间差补充令牌（需在锁内调用）。"""
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0):
        """获取令牌，如果不足则等待。"""
        async with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            # 计算等待时间
            deficit = tokens - self._tokens
            wait = deficit / self.rate
            self._tokens = 0
            self._last_refill = now + wait
        await asyncio.sleep(wait)

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """非阻塞获取令牌，不足时返回 False。"""
        async with self._lock:
            now = time.monotonic()
            self._refill(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass


class RateLimiterRegistry:
    """全局限流器注册表。"""

    def __init__(self):
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._lock = asyncio.Lock()

    async def get(self, name: str) -> TokenBucketRateLimiter:
        async with self._lock:
            if name not in self._limiters:
                cfg = DEFAULT_RATES.get(name, DEFAULT_RATES["default"])
                self._limiters[name] = TokenBucketRateLimiter(
                    name=name, rate=cfg["rate"], capacity=cfg["capacity"]
                )
                logger.debug(f"创建限流器 [{name}]: rate={cfg['rate']}/s, burst={cfg['capacity']}")
            return self._limiters[name]

    async def get_all(self) -> Dict[str, TokenBucketRateLimiter]:
        async with self._lock:
            return dict(self._limiters)


# 模块级单例
_registry: Optional[RateLimiterRegistry] = None


async def get_rate_limiter(name: str = "default") -> TokenBucketRateLimiter:
    global _registry
    if _registry is None:
        _registry = RateLimiterRegistry()
    return await _registry.get(name)
