"""Chat session and history management for multi-user conversations.

Provides:
- HistoryStore: Abstract base class for persistence backends
- ChatManager: Manages user sessions and routes messages through LangGraph
"""

import json
import time
import uuid
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
import os
from typing import Dict, List, Optional, Any

import httpx
import redis.asyncio as redis

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage

logger = logging.getLogger("wabi.chat")

# Module-level Redis singleton — lazily initialised on first use
_redis_client: Optional[redis.Redis] = None

def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        from langgraph_app.config import config
        _redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
    return _redis_client


# ---------------------------------------------------------------------------
# Abstract History Store
# ---------------------------------------------------------------------------

class HistoryStore(ABC):
    """Abstract base class for chat history persistence."""

    @abstractmethod
    async def save_message(self, user_id: str, role: str, content: str, timestamp: str) -> None:
        ...

    @abstractmethod
    async def load_history(self, user_id: str) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def delete_history(self, user_id: str) -> None:
        ...

    @abstractmethod
    async def list_users(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def create_user(self, user_id: str, name: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        ...

    @abstractmethod
    async def save_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    async def load_profile(self, user_id: str) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Time boundary helpers
# ---------------------------------------------------------------------------

# Fallback timezone: Singapore / Taipei (UTC+8)
_TZ_UTC8 = timezone(timedelta(hours=8))
_DAY_START_HOUR = 3  # 3 AM as new-day boundary


def _get_day_boundary(tz_name: Optional[str] = None) -> str:
    """Return the most recent 3:00 AM in the user's timezone as a UTC ISO timestamp.

    Falls back to UTC+8 if no timezone is provided or the name is invalid.
    """
    user_tz = _TZ_UTC8
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            user_tz = ZoneInfo(tz_name)
        except Exception:
            pass

    now_local = datetime.now(user_tz)

    if now_local.hour < _DAY_START_HOUR:
        boundary_local = now_local.replace(
            hour=_DAY_START_HOUR, minute=0, second=0, microsecond=0
        ) - timedelta(days=1)
    else:
        boundary_local = now_local.replace(
            hour=_DAY_START_HOUR, minute=0, second=0, microsecond=0
        )

    boundary_utc = boundary_local.astimezone(timezone.utc)
    return boundary_utc.isoformat()


# ---------------------------------------------------------------------------
# Chat Manager
# ---------------------------------------------------------------------------

class ChatManager:
    """Manages multi-user chat sessions and interfaces with LangGraph.

    Each user gets:
    - A unique user_id
    - Independent message history (stored via HistoryStore)
    - A separate LangGraph session
    """

    def __init__(self, store: Optional[HistoryStore] = None):
        if store is None:
            from server.db import PostgresHistoryStore
            store = PostgresHistoryStore()
        self.store = store

    async def create_user(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new user/conversation."""
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        display_name = name or f"User {user_id[-4:]}"
        user_info = await self.store.create_user(user_id, display_name)
        return user_info

    async def get_users(self) -> List[Dict[str, Any]]:
        """List all users."""
        return await self.store.list_users()

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user and their history."""
        return await self.store.delete_user(user_id)

    async def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get message history for a user."""
        history = await self.store.load_history(user_id)
        for msg in history:
            try:
                if isinstance(msg["content"], str) and msg["content"].startswith("[") and '"image_url"' in msg["content"]:
                    msg["content"] = json.loads(msg["content"])
            except Exception:
                pass
        return history

    async def save_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        """Save user profile."""
        await self.store.save_profile(user_id, profile)

    async def get_profile(self, user_id: str) -> Dict[str, Any]:
        """Load user profile."""
        return await self.store.load_profile(user_id)

    def _build_langchain_messages(self, history: List[Dict[str, Any]]) -> List[AnyMessage]:
        """Convert stored history dicts to LangChain message objects."""
        messages: List[AnyMessage] = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    async def process_message(
        self,
        user_id: str,
        content: Any,
        user_context: Optional[Dict[str, Any]] = None,
    ):
        """Process a user message through the LangGraph graph, yielding updates."""
        now = datetime.now(timezone.utc).isoformat()

        # Build the user message
        if isinstance(content, str):
            new_msg = HumanMessage(content=content)
            save_content = content
        else:
            # Multimodal content (text + images)
            new_msg = HumanMessage(content=content)
            save_content = json.dumps(content)

        # Save user message to DB
        await self.store.save_message(user_id, "user", save_content, now)

        # Load user profile and name
        user_profile = await self.store.load_profile(user_id)
        user_info = await self.store.get_user(user_id)
        user_name = user_info.get("name") if user_info else None

        # --- Load conversation history ---
        # With the Supervisor architecture, we load recent messages instead of
        # the old two-phase (today's history �� full history for goalplanning).
        # The Supervisor can query meal_logs via tools for historical data.
        load_recent = getattr(self.store, "load_recent_messages", None)
        if load_recent:
            recent_history = await load_recent(user_id, limit=50)
        else:
            # Fallback: load today's history (legacy stores)
            day_boundary = _get_day_boundary((user_context or {}).get("timezone"))
            load_method = getattr(self.store, "load_history_since", None)
            if load_method:
                recent_history = await load_method(user_id, day_boundary)
            else:
                recent_history = await self.store.load_history(user_id)

        for msg in recent_history:
            try:
                if isinstance(msg["content"], str) and msg["content"].startswith("[") and '"image_url"' in msg["content"]:
                    msg["content"] = json.loads(msg["content"])
            except Exception:
                pass

        # --- Payload Construction for Microservice ---
        session_id = f"web_{user_id}_{int(time.time())}"
        invocation_id = f"{user_id}_{int(time.time() * 1000)}"

        payload = {
            "messages": recent_history,
            "session_id": session_id,
            "user_id": user_id,
            "user_name": user_name,
            "user_profile": user_profile if user_profile else None,
            "thread_id": invocation_id,
            "response_channel": f"response_{invocation_id}",
            "user_context": user_context or {},
        }

        # Redis PubSub and Job Queue pattern
        try:
            redis_client = _get_redis()
            response_channel = payload["response_channel"]
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(response_channel)

            logger.info(f"[{user_id}] Pushing request to worker queue (wabi_ai_queue)...")
            await redis_client.rpush("wabi_ai_queue", json.dumps(payload))

            ai_text = "Sorry, I could not process your request."

            try:
                async with asyncio.timeout(120.0):
                    async for message in pubsub.listen():
                        if message.get('type') != 'message':
                            continue
                        data = json.loads(message['data'])

                        if data.get("status") == "partial":
                            yield {
                                "type": "thinking",
                                "node": data.get("node"),
                                "analysis": data.get("analysis", {})
                            }
                            continue

                        if isinstance(data, dict) and data.get("status") == "error":
                            error_msg = data.get("message", "Unknown Backend Crash")
                            logger.error(f"Intercepted backend crash for user {user_id}: {error_msg}")
                            ai_text = f"🔴 fatal error: {error_msg}"
                            break

                        msgs = data.get("messages", [])
                        if msgs and isinstance(msgs, list):
                            ai_text = msgs[-1].get("content", ai_text)
                        else:
                            ai_text = data.get('ai_text', ai_text)
                        break
            except TimeoutError:
                logger.warning(f"[{user_id}] Timed out waiting for response after 120s")

            await pubsub.unsubscribe(response_channel)

        except Exception as e:
            logger.error(f"[{user_id}] Failed to dispatch AI message to Queue: {e}", exc_info=True)
            ai_text = f"Warning: Backend AI Task Queue unavailable. ({str(e)})"

        # Save AI response to DB
        ai_timestamp = datetime.now(timezone.utc).isoformat()
        await self.store.save_message(user_id, "assistant", ai_text, ai_timestamp)

        yield {
            "type": "final",
            "text": ai_text
        }
