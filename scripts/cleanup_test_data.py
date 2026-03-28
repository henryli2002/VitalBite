import asyncio
import os
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wabi.cleanup")

DB_URL = os.environ.get("WABI_DB_URL", "postgresql://wabi_user:wabi_password@localhost:5432/wabi_chat")

async def cleanup():
    logger.info(f"Connecting to DB: {DB_URL}")
    try:
        conn = await asyncpg.connect(DB_URL)
        
        # 1. Count test users
        count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE user_id LIKE 'loadtest_%' OR name = 'Test User'")
        logger.info(f"Found {count} test users to delete.")
        
        if count > 0:
            # 2. Delete test users (cascades to messages due to Foreign Key)
            await conn.execute("DELETE FROM users WHERE user_id LIKE 'loadtest_%' OR name = 'Test User'")
            logger.info("Successfully deleted test users and their messages.")
        else:
            logger.info("No test users found.")
            
        await conn.close()
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    asyncio.run(cleanup())
