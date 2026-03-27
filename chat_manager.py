"""Chat session and history management for multi-user conversations.

Provides:
- HistoryStore: Abstract base class for future database backends (PostgreSQL, Redis, etc.)
- InMemoryHistoryStore: Default in-memory implementation
- ChatManager: Manages user sessions and routes messages through LangGraph
"""

import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage


# ---------------------------------------------------------------------------
# Abstract History Store (for future backend extension)
# ---------------------------------------------------------------------------

class HistoryStore(ABC):
    """Abstract base class for chat history persistence.
    
    Implement this interface to plug in PostgreSQL, Redis, MongoDB, etc.
    """

    @abstractmethod
    async def save_message(self, user_id: str, role: str, content: str, timestamp: str) -> None:
        """Save a single message to storage."""
        ...

    @abstractmethod
    async def load_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Load full message history for a user. Returns list of dicts with role, content, timestamp."""
        ...

    @abstractmethod
    async def delete_history(self, user_id: str) -> None:
        """Delete all messages for a user."""
        ...

    @abstractmethod
    async def list_users(self) -> List[Dict[str, Any]]:
        """List all users with metadata."""
        ...

    @abstractmethod
    async def create_user(self, user_id: str, name: str) -> Dict[str, Any]:
        """Create a new user entry."""
        ...

    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user and their history."""
        ...


# ---------------------------------------------------------------------------
# In-Memory Implementation
# ---------------------------------------------------------------------------

class InMemoryHistoryStore(HistoryStore):
    """Default in-memory history store. Data is lost on restart."""

    def __init__(self):
        # user_id -> list of {role, content, timestamp}
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        # user_id -> {user_id, name, created_at, last_active}
        self._users: Dict[str, Dict[str, Any]] = {}

    async def save_message(self, user_id: str, role: str, content: str, timestamp: str) -> None:
        if user_id not in self._messages:
            self._messages[user_id] = []
        self._messages[user_id].append({
            "role": role,
            "content": content,
            "timestamp": timestamp,
        })
        # Update last_active
        if user_id in self._users:
            self._users[user_id]["last_active"] = timestamp

    async def load_history(self, user_id: str) -> List[Dict[str, Any]]:
        return self._messages.get(user_id, [])

    async def delete_history(self, user_id: str) -> None:
        self._messages.pop(user_id, None)

    async def list_users(self) -> List[Dict[str, Any]]:
        result = []
        for uid, info in self._users.items():
            result.append({
                **info,
                "message_count": len(self._messages.get(uid, [])),
            })
        return result

    async def create_user(self, user_id: str, name: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        user_info = {
            "user_id": user_id,
            "name": name,
            "created_at": now,
            "last_active": now,
        }
        self._users[user_id] = user_info
        self._messages[user_id] = []
        return {**user_info, "message_count": 0}

    async def delete_user(self, user_id: str) -> bool:
        if user_id not in self._users:
            return False
        self._users.pop(user_id, None)
        self._messages.pop(user_id, None)
        return True


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
        self.store = store or InMemoryHistoryStore()
        # Cache of LangChain message objects for active sessions
        self._langchain_histories: Dict[str, List[AnyMessage]] = {}

    async def create_user(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new user/conversation."""
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        display_name = name or f"User {user_id[-4:]}"
        user_info = await self.store.create_user(user_id, display_name)
        self._langchain_histories[user_id] = []
        return user_info

    async def get_users(self) -> List[Dict[str, Any]]:
        """List all users."""
        return await self.store.list_users()

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user and their history."""
        self._langchain_histories.pop(user_id, None)
        return await self.store.delete_user(user_id)

    async def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get message history for a user."""
        return await self.store.load_history(user_id)

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
        graph: Any,
    ) -> str:
        """Process a user message through the LangGraph graph.
        
        Args:
            user_id: The user's ID
            content: Text string or multimodal content list
            graph: The compiled LangGraph graph instance
            
        Returns:
            The AI response text
        """
        now = datetime.now(timezone.utc).isoformat()

        # Build the user message
        if isinstance(content, str):
            new_msg = HumanMessage(content=content)
            save_content = content
        else:
            # Multimodal content (text + images)
            new_msg = HumanMessage(content=content)
            # Extract text for storage
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            save_content = " ".join(text_parts).strip() or "[Image]"

        # Save user message
        await self.store.save_message(user_id, "user", save_content, now)

        # Load full history and build LangChain messages
        history = await self.store.load_history(user_id)
        langchain_messages = self._build_langchain_messages(history[:-1])  # exclude current msg
        langchain_messages.append(new_msg)  # add current with full content (may include images)

        # Build the state for graph invocation
        session_id = f"web_{user_id}_{int(time.time())}"
        initial_state = {
            "messages": langchain_messages,
            "session_id": session_id,
            "user_id": user_id,
        }

        # Invoke the graph
        result = graph.invoke(initial_state)

        # Extract AI response
        ai_text = ""
        if "messages" in result and result["messages"]:
            last_msg = result["messages"][-1]
            if isinstance(last_msg, AIMessage):
                ai_text = last_msg.content
            else:
                ai_text = str(last_msg.content)
        else:
            ai_text = "Sorry, I could not process your request."

        # Save AI response
        ai_timestamp = datetime.now(timezone.utc).isoformat()
        await self.store.save_message(user_id, "assistant", ai_text, ai_timestamp)

        return ai_text
