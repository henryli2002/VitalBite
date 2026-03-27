"""SQLite-backed persistence for WABI Chat.

Implements the HistoryStore ABC from chat_manager using aiosqlite.
Stores users, messages, and user profiles with full async I/O.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import aiosqlite

from chat_manager import HistoryStore


DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "wabi_chat.db")


class SQLiteHistoryStore(HistoryStore):
    """Async SQLite implementation of HistoryStore."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path

    async def init_db(self) -> None:
        """Create tables if they don't exist. Call once on startup."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    profile_json TEXT DEFAULT '{}'
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    has_image INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_user_ts
                ON messages(user_id, timestamp)
            """)
            await db.commit()

    # ------------------------------------------------------------------
    # HistoryStore ABC implementation
    # ------------------------------------------------------------------

    async def save_message(
        self, user_id: str, role: str, content: str, timestamp: str
    ) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (user_id, role, content, timestamp),
            )
            await db.execute(
                "UPDATE users SET last_active = ? WHERE user_id = ?",
                (timestamp, user_id),
            )
            await db.commit()

    async def load_history(self, user_id: str) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT role, content, timestamp FROM messages WHERE user_id = ? ORDER BY id ASC",
                (user_id,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def load_history_since(
        self, user_id: str, since_ts: str
    ) -> List[Dict[str, Any]]:
        """Load messages for a user since a given ISO timestamp."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE user_id = ? AND timestamp >= ? ORDER BY id ASC",
                (user_id, since_ts),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete_history(self, user_id: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            await db.commit()

    async def list_users(self) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users ORDER BY last_active DESC")
            users = await cursor.fetchall()
            result = []
            for u in users:
                u_dict = dict(u)
                # Get message count
                cnt_cursor = await db.execute(
                    "SELECT COUNT(*) FROM messages WHERE user_id = ?",
                    (u_dict["user_id"],),
                )
                cnt = await cnt_cursor.fetchone()
                u_dict["message_count"] = cnt[0] if cnt else 0
                # Remove profile_json from listing (not needed in sidebar)
                u_dict.pop("profile_json", None)
                result.append(u_dict)
            return result

    async def get_user(self, user_id: str) -> Dict[str, Any]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = await cursor.fetchone()
            if row:
                u_dict = dict(row)
                u_dict.pop("profile_json", None)
                return u_dict
            return {}

    async def create_user(self, user_id: str, name: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO users (user_id, name, created_at, last_active, profile_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, name, now, now, "{}"),
            )
            await db.commit()
        return {
            "user_id": user_id,
            "name": name,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }

    async def delete_user(self, user_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
            )
            exists = await cursor.fetchone()
            if not exists:
                return False
            await db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            await db.commit()
            return True

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    async def save_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        """Save or update user profile as JSON."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE users SET profile_json = ? WHERE user_id = ?",
                (json.dumps(profile, ensure_ascii=False), user_id),
            )
            await db.commit()

    async def load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load user profile dict. Returns empty dict if not set."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT profile_json FROM users WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return {}
            return {}
