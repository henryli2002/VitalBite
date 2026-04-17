"""PostgreSQL-backed persistence for WABI Chat.

Implements the HistoryStore ABC from chat_manager using asyncpg.
Stores users, messages, and user profiles with a high-concurrency connection pool.
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import asyncpg

from server.chat_manager import HistoryStore

logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "WABI_DB_URL", "postgresql://wabi_user:wabi_password@localhost:5432/wabi_chat"
)


class PostgresHistoryStore(HistoryStore):
    """Async PostgreSQL implementation of HistoryStore using an asyncpg pool."""

    def __init__(self, db_url: str = DB_URL):
        self._db_url = db_url
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            # Create the connection pool lazily
            self._pool = await asyncpg.create_pool(
                self._db_url,
                min_size=1,
                max_size=50,
                max_inactive_connection_lifetime=300,
            )  # type: ignore
        return self._pool

    async def init_db(self) -> None:
        """Create tables if they don't exist. Call once on startup."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    profile_json TEXT DEFAULT '{}'
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    has_image INTEGER DEFAULT 0
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS meal_logs (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    timestamp TEXT NOT NULL,
                    total_calories REAL DEFAULT 0.0,
                    protein REAL DEFAULT 0.0,
                    carbs REAL DEFAULT 0.0,
                    fat REAL DEFAULT 0.0,
                    items_json TEXT DEFAULT '{}',
                    metadata_json TEXT DEFAULT '{}'
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_user_ts
                ON messages(user_id, timestamp)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_meal_logs_user_ts
                ON meal_logs(user_id, timestamp)
            """)

    # ------------------------------------------------------------------
    # HistoryStore ABC implementation
    # ------------------------------------------------------------------

    async def save_message(
        self, user_id: str, role: str, content: str, timestamp: str
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO users (user_id, name, created_at, last_active, profile_json)
                    VALUES ($1, $2, $3, $4, '{}')
                    ON CONFLICT (user_id) DO UPDATE SET last_active = EXCLUDED.last_active
                    """,
                    user_id,
                    f"User {user_id[-4:]}",
                    timestamp,
                    timestamp,
                )
                await conn.execute(
                    "INSERT INTO messages (user_id, role, content, timestamp) VALUES ($1, $2, $3, $4)",
                    user_id,
                    role,
                    content,
                    timestamp,
                )

    async def load_history(self, user_id: str) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content, timestamp FROM messages WHERE user_id = $1 ORDER BY id ASC",
                user_id,
            )
            return [dict(row) for row in rows]

    async def load_history_since(
        self, user_id: str, since_ts: str
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        """Load messages for a user since a given ISO timestamp."""
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content, timestamp FROM messages "
                "WHERE user_id = $1 AND timestamp >= $2 ORDER BY id ASC",
                user_id,
                since_ts,
            )
            return [dict(row) for row in rows]

    async def load_recent_messages(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Load the most recent N messages for a user."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content, timestamp FROM "
                "(SELECT role, content, timestamp, id FROM messages "
                "WHERE user_id = $1 ORDER BY id DESC LIMIT $2) sub "
                "ORDER BY id ASC",
                user_id,
                limit,
            )
            return [dict(row) for row in rows]

    async def update_image_description(
        self, user_id: str, image_uuid: str, description: str
    ) -> int:
        """Rewrite the most recent `[图片: {uuid}]` placeholder to include a description.

        Returns the number of rows affected. Uses a LIKE search constrained by
        user_id; also re-matches existing placeholders with old descriptions so
        repeated calls overwrite rather than duplicate.
        """
        import re as _re
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, content FROM messages "
                "WHERE user_id = $1 AND content LIKE $2 "
                "ORDER BY id DESC LIMIT 5",
                user_id,
                f"%{image_uuid}%",
            )
            pattern = _re.compile(
                rf"\[图片:\s*{_re.escape(image_uuid)}(?:\s*\|\s*[^\]]*)?\]",
                _re.IGNORECASE,
            )
            new_placeholder = f"[图片: {image_uuid} | {description}]"
            updates: List[tuple[int, str]] = []
            for row in rows:
                content = row["content"] or ""
                if pattern.search(content):
                    updates.append((row["id"], pattern.sub(new_placeholder, content)))
            for msg_id, new_content in updates:
                await conn.execute(
                    "UPDATE messages SET content = $1 WHERE id = $2",
                    new_content,
                    msg_id,
                )
            return len(updates)

    async def delete_history(self, user_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)

    async def list_users(self) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT u.user_id, u.name, u.created_at, u.last_active,
                       COUNT(m.id) AS message_count
                FROM users u
                LEFT JOIN messages m ON m.user_id = u.user_id
                WHERE u.user_id NOT LIKE 'loadtest_%' AND u.name != 'Test User'
                GROUP BY u.user_id, u.name, u.created_at, u.last_active
                ORDER BY u.last_active DESC
                """
            )
            return [
                {
                    "user_id": row["user_id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "last_active": row["last_active"],
                    "message_count": row["message_count"] or 0,
                }
                for row in rows
            ]

    async def get_user(self, user_id: str) -> Dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user_id)
            if row:
                u_dict = dict(row)
                u_dict.pop("profile_json", None)
                return u_dict
            return {}

    async def create_user(self, user_id: str, name: str) -> Dict[str, Any]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc).isoformat()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (user_id, name, created_at, last_active, profile_json) "
                "VALUES ($1, $2, $3, $4, $5)",
                user_id,
                name,
                now,
                now,
                "{}",
            )
        return {
            "user_id": user_id,
            "name": name,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }

    async def delete_user(self, user_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM users WHERE user_id = $1", user_id
            )
            if not exists:
                return False
            async with conn.transaction():
                await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)
                await conn.execute("DELETE FROM users WHERE user_id = $1", user_id)
            try:
                from server.image_store import delete_user_images
                delete_user_images(user_id)
            except Exception as e:
                logger.warning("Failed to clean image directory for %s: %s", user_id, e)
            return True

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    async def save_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        """Save or update user profile as JSON."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET profile_json = $1 WHERE user_id = $2",
                json.dumps(profile, ensure_ascii=False),
                user_id,
            )

    async def load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load user profile dict. Returns empty dict if not set."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT profile_json FROM users WHERE user_id = $1", user_id
            )
            if row and row["profile_json"]:
                try:
                    return json.loads(row["profile_json"])
                except Exception as e:
                    logger.warning(
                        "Failed to parse profile JSON for %s: %s", user_id, e
                    )
            return {}

    # ------------------------------------------------------------------
    # Meal Logs management
    # ------------------------------------------------------------------

    async def save_meal_log(self, user_id: str, meal_data: Dict[str, Any]) -> int:
        """Save a confirmed meal log. Returns the inserted ID."""
        pool = await self._get_pool()
        timestamp = meal_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        async with pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO meal_logs 
                (user_id, timestamp, total_calories, protein, carbs, fat, items_json, metadata_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                user_id,
                timestamp,
                float(meal_data.get("total_calories", 0.0)),
                float(meal_data.get("protein", 0.0)),
                float(meal_data.get("carbs", 0.0)),
                float(meal_data.get("fat", 0.0)),
                json.dumps(meal_data.get("items", []), ensure_ascii=False),
                json.dumps(meal_data.get("metadata", {}), ensure_ascii=False),
            )
            return row_id

    async def load_meal_logs(self, user_id: str, since_ts: str) -> List[Dict[str, Any]]:
        """Load structured meal logs for a user since a given ISO timestamp."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, timestamp, total_calories, protein, carbs, fat, items_json, metadata_json
                FROM meal_logs 
                WHERE user_id = $1 AND timestamp >= $2
                ORDER BY timestamp ASC
                """,
                user_id,
                since_ts,
            )

            results = []
            for row in rows:
                log_dict = dict(row)
                try:
                    log_dict["items"] = json.loads(log_dict.pop("items_json"))
                    log_dict["metadata"] = json.loads(log_dict.pop("metadata_json"))
                except Exception as e:
                    logger.warning(
                        "Failed to parse JSON for meal_log %s: %s", row["id"], e
                    )
                    log_dict["items"] = []
                    log_dict["metadata"] = {}
                results.append(log_dict)

            return results
