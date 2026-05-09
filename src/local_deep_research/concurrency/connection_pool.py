"""共享 HTTP 连接池 — 复用连接减少握手开销。

为所有 ChatOpenAI 实例提供同一个 httpx.AsyncClient，启用 keepalive 和连接池。
httpx 仅在首次调用 get_shared_http_client() 时惰性导入。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

# 连接池配置
POOL_CONFIG = {
    "max_keepalive_connections": 20,
    "max_connections": 50,
    "keepalive_expiry": 30.0,  # 空闲连接 30 秒后回收
}

TIMEOUT_CONFIG = {
    "connect": 10.0,    # 建连超时
    "read": 300.0,      # 读取超时（LLM 响应可能很慢）
    "write": 30.0,      # 写入超时
    "pool": 5.0,        # 从池中取连接超时
}

_client: Optional[httpx.AsyncClient] = None


def get_shared_http_client():
    """获取全局共享的 httpx.AsyncClient（惰性创建）。"""
    import httpx  # 惰性导入
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(**POOL_CONFIG),
            timeout=httpx.Timeout(**TIMEOUT_CONFIG),
        )
        logger.info("共享 HTTP 客户端已创建 (keepalive=%d, max_conn=%d)",
                     POOL_CONFIG["max_keepalive_connections"],
                     POOL_CONFIG["max_connections"])
    return _client


async def close_shared_http_client():
    """关闭共享 HTTP 客户端，释放所有连接。"""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        logger.info("共享 HTTP 客户端已关闭")
    _client = None
