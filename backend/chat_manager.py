"""Chat session and history management for multi-user conversations.

Provides:
- HistoryStore: Abstract base class for persistence backends
- ChatManager: Manages user sessions and routes messages through LangGraph
"""

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
        url = os.environ.get("WABI_REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(url, decode_responses=True)
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
            from db import PostgresHistoryStore
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
        import json
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
            import json
            save_content = json.dumps(content)

        # Save user message to DB
        await self.store.save_message(user_id, "user", save_content, now)

        # Load user profile and name
        user_profile = await self.store.load_profile(user_id)
        user_info = await self.store.get_user(user_id)
        user_name = user_info.get("name") if user_info else None

        # --- Phase 1: Quick intent detection with today's history ---
        day_boundary = _get_day_boundary((user_context or {}).get("timezone"))
        load_method = getattr(self.store, "load_history_since", None)
        if load_method:
            today_history = await load_method(user_id, day_boundary)
        else:
            today_history = await self.store.load_history(user_id)

        import json
        for msg in today_history:
            try:
                if isinstance(msg["content"], str) and msg["content"].startswith("[") and '"image_url"' in msg["content"]:
                    msg["content"] = json.loads(msg["content"])
            except Exception:
                pass

        today_messages = self._build_langchain_messages(today_history)

        # --- Payload Construction for Microservice ---
        session_id = f"web_{user_id}_{int(time.time())}"
        invocation_id = f"{user_id}_{int(time.time() * 1000)}"

        # Prepare base payload with simple serializable dicts instead of LangChain objects
        payload = {
            "messages": today_history,
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
            import json

            redis_client = _get_redis()
            response_channel = payload["response_channel"]
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(response_channel)
            
            logger.info(f"[{user_id}] Pushing request to worker queue (wabi_ai_queue)...")
            await redis_client.rpush("wabi_ai_queue", json.dumps(payload))
            
            ai_text = "Sorry, I could not process your request."
            detected_intent = "chitchat"
            
            start_wait = time.time()
            phase_1_done = False
            # Wait up to 120 seconds in queue
            while time.time() - start_wait < 120.0:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get('type') == 'message':
                    data = json.loads(message['data'])
                    
                    if data.get("status") == "partial":
                        yield {
                            "type": "thinking",
                            "node": data.get("node"),
                            "analysis": data.get("analysis", {})
                        }
                        continue
                    
                    # Check if it was a backend Node Crash
                    if isinstance(data, dict) and data.get("status") == "error":
                        error_msg = data.get("message", "Unknown Backend Crash")
                        logger.error(f"Intercepted backend crash for user {user_id}: {error_msg}")
                        ai_text = f"🔴 fatal error: {error_msg}"
                        phase_1_done = True
                        break
                    
                    # Parse the new nested JSON payload from langgraph_server.py
                    messages = data.get("messages", [])
                    if messages and isinstance(messages, list):
                        ai_text = messages[-1].get("content", ai_text)
                    else:
                        ai_text = data.get('ai_text', ai_text)
                        
                    detected_intent = data.get("analysis", {}).get("intent", data.get("detected_intent", detected_intent))
                    phase_1_done = True
                    break
            
            # Goal planning: re-run with full history for complete context
            if phase_1_done and detected_intent == "goalplanning":
                logger.info(f"[{user_id}] Intent is goalplanning. Pushing Phase 2 job with FULL history.")
                full_history = await self.store.load_history(user_id)
                for msg in full_history:
                    try:
                        if isinstance(msg["content"], str) and msg["content"].startswith("[") and '"image_url"' in msg["content"]:
                            msg["content"] = json.loads(msg["content"])
                    except Exception:
                        pass

                full_payload = {
                    **payload,
                    "messages": full_history,
                    "thread_id": f"{user_id}_full_{int(time.time() * 1000)}",
                }
                await redis_client.rpush("wabi_ai_queue", json.dumps(full_payload))

                start_wait = time.time()
                while time.time() - start_wait < 120.0:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message and message.get('type') == 'message':
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
                            logger.error(f"Intercepted backend crash (Phase 2) for user {user_id}: {error_msg}")
                            ai_text = f"🔴 fatal error (Goalplanning): {error_msg}"
                            break

                        p2_messages = data.get("messages", [])
                        if p2_messages and isinstance(p2_messages, list):
                            ai_text = p2_messages[-1].get("content", ai_text)
                        else:
                            ai_text = data.get('ai_text', ai_text)
                        break
            
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
