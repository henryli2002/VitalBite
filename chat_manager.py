"""Chat session and history management for multi-user conversations.

Provides:
- HistoryStore: Abstract base class for persistence backends
- ChatManager: Manages user sessions and routes messages through LangGraph
"""

import time
import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
import os
from typing import Dict, List, Optional, Any

import httpx

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage

logger = logging.getLogger("wabi.chat")


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

# Singapore / Taipei timezone (UTC+8)
_TZ_UTC8 = timezone(timedelta(hours=8))
_DAY_START_HOUR = 3  # 3 AM as new-day boundary


def _get_day_boundary() -> str:
    """Return the most recent 3:00 AM in UTC+8 as an ISO timestamp (in UTC).

    If current UTC+8 time is before 3 AM, the boundary is yesterday 3 AM.
    """
    now_utc8 = datetime.now(_TZ_UTC8)

    if now_utc8.hour < _DAY_START_HOUR:
        # Before 3 AM — boundary is yesterday 3 AM
        boundary_utc8 = now_utc8.replace(
            hour=_DAY_START_HOUR, minute=0, second=0, microsecond=0
        ) - timedelta(days=1)
    else:
        # At or after 3 AM — boundary is today 3 AM
        boundary_utc8 = now_utc8.replace(
            hour=_DAY_START_HOUR, minute=0, second=0, microsecond=0
        )

    # Convert to UTC for DB comparison (timestamps are stored as UTC)
    boundary_utc = boundary_utc8.astimezone(timezone.utc)
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
            from db import SQLiteHistoryStore
            store = SQLiteHistoryStore()
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
        return await self.store.load_history(user_id)

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
    ) -> str:
        """Process a user message through the LangGraph graph.

        History scoping:
        - goalplanning: full user history
        - all other intents: only messages since the 3 AM boundary (today)

        The graph is invoked with the user's profile injected into state.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Build the user message
        if isinstance(content, str):
            new_msg = HumanMessage(content=content)
            save_content = content
        else:
            # Multimodal content (text + images)
            new_msg = HumanMessage(content=content)
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            save_content = " ".join(text_parts).strip() or "[Image]"

        # Save user message to DB
        await self.store.save_message(user_id, "user", save_content, now)

        # Load user profile and name
        user_profile = await self.store.load_profile(user_id)
        user_info = await self.store.get_user(user_id)
        user_name = user_info.get("name") if user_info else None

        # --- Phase 1: Quick intent detection with today's history ---
        day_boundary = _get_day_boundary()
        today_history = await getattr(self.store, "load_history_since", self.store.load_history)(user_id, day_boundary)
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
            "invoke_full_history": False,
            "full_messages": None
        }

        # Determine AI Engine URL
        ai_url = os.getenv("WABI_AI_URL", "http://localhost:8001/invoke")

        logger.info(f"[{user_id}] Sending graph request to AI Engine ({ai_url})")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(ai_url, json=payload)
                response.raise_for_status()
                response_data = response.json()
                
                ai_text = response_data.get("ai_text", "Sorry, I could not process your request.")
                detected_intent = response_data.get("detected_intent", "chitchat")
                
                # If intent was goalplanning, the remote server handles the re-invocation logic IF we tell it to.
                # However, the remote server needs the FULL history for that.
                # We want to minimize payload size. So if the First pass returned 'goalplanning', 
                # we can send a second request with full history if it's the first time we realize it.
                # Wait, the remote server returns intent AFTER first pass. So we do the second pass here.
                if detected_intent == "goalplanning":
                    logger.info(f"[{user_id}] Intent is goalplanning. Re-requesting AI Engine with full history.")
                    full_history = await self.store.load_history(user_id)
                    payload["invoke_full_history"] = True
                    payload["full_messages"] = full_history
                    payload["thread_id"] = f"{user_id}_full_{int(time.time() * 1000)}"
                    
                    response2 = await client.post(ai_url, json=payload)
                    response2.raise_for_status()
                    response_data = response2.json()
                    ai_text = response_data.get("ai_text", ai_text)

        except Exception as e:
            logger.error(f"[{user_id}] Failed to contact AI Engine: {e}", exc_info=True)
            ai_text = f"Warning: Backend AI Service unavailable. ({str(e)})"

        # Save AI response to DB
        ai_timestamp = datetime.now(timezone.utc).isoformat()
        await self.store.save_message(user_id, "assistant", ai_text, ai_timestamp)

        return ai_text
