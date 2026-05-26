"""PathoEBM 并发基础设施模块。

提供生产级并发控制原语：
- circuit_breaker: 熔断器
- rate_limiter: 令牌桶限流器
- connection_pool: 共享 HTTP 连接池
- backpressure: 有界并发背压控制
- shutdown: 优雅关闭管理器
- task_manager: 任务隔离与安全 gather
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    get_circuit_breaker,
)
from .rate_limiter import (
    TokenBucketRateLimiter,
    RateLimitExceeded,
    RateLimiterRegistry,
    get_rate_limiter,
)
from .connection_pool import (
    get_shared_http_client,
    close_shared_http_client,
)
from .backpressure import (
    bounded_gather,
    BoundedTaskQueue,
)
from .shutdown import (
    ShutdownManager,
    get_shutdown_manager,
    install_signal_handlers,
)
from .task_manager import (
    gather_safe,
    ErrorBoundary,
    TaskResult,
)

__all__ = [
    # circuit_breaker
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitBreakerRegistry",
    "get_circuit_breaker",
    # rate_limiter
    "TokenBucketRateLimiter",
    "RateLimitExceeded",
    "RateLimiterRegistry",
    "get_rate_limiter",
    # connection_pool
    "get_shared_http_client",
    "close_shared_http_client",
    # backpressure
    "bounded_gather",
    "BoundedTaskQueue",
    # shutdown
    "ShutdownManager",
    "get_shutdown_manager",
    "install_signal_handlers",
    # task_manager
    "gather_safe",
    "ErrorBoundary",
    "TaskResult",
]
