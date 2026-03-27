import asyncio
import aiosqlite
from datetime import datetime, timedelta, timezone
import json
import uuid

async def setup_test_data():
    db_path = "/Users/henryli/Desktop/Project/WABI/data/wabi_chat.db"
    
    async with aiosqlite.connect(db_path) as db:
        # Create a test user
        user_id = str(uuid.uuid4())
        await db.execute(
            "INSERT INTO users (user_id, name, created_at, last_active) VALUES (?, ?, ?, ?)",
            (user_id, "Test User", datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat())
        )
        
        # Insert a profile
        profile = {"age": 25, "gender": "male", "fitness_goals": "Lose 5kg"}
        await db.execute(
            "UPDATE users SET profile_json = ? WHERE user_id = ?",
            (json.dumps(profile), user_id)
        )
        
        # Insert a message from 2 days ago
        past_time = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await db.execute(
            "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, "user", "I ate a burger 2 days ago", past_time)
        )
        
        # Insert a message from today
        now_time = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, "user", "I am eating a salad right now", now_time)
        )
        
        await db.commit()
        
        print(f"Created user_id: {user_id}")
        return user_id

if __name__ == "__main__":
    asyncio.run(setup_test_data())
