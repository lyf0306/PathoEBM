import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentContextBus:
    """
    Lightweight message bus for inter-agent communication.

    Agents can post findings and read messages from other agents
    without direct coupling. This is an optional enhancement —
    agents work fine without it.
    """
    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def post(self, sender: str, msg_type: str, content: Any, metadata: Optional[Dict] = None):
        """Post a message from an agent to the bus."""
        async with self._lock:
            self._messages.append({
                "sender": sender,
                "type": msg_type,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            })
            logger.debug(f"[ContextBus] {sender} posted: {msg_type}")

    async def get_all(self) -> List[Dict]:
        """Get all messages."""
        async with self._lock:
            return list(self._messages)

    async def get_by_type(self, msg_type: str) -> List[Dict]:
        """Filter messages by type."""
        async with self._lock:
            return [m for m in self._messages if m["type"] == msg_type]

    async def get_by_sender(self, sender: str) -> List[Dict]:
        """Filter messages by sender agent name."""
        async with self._lock:
            return [m for m in self._messages if m["sender"] == sender]

    async def has_type(self, msg_type: str) -> bool:
        """Check if any message of a given type exists."""
        async with self._lock:
            return any(m["type"] == msg_type for m in self._messages)

    async def clear(self):
        """Clear all messages."""
        async with self._lock:
            self._messages.clear()
