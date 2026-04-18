import asyncio
import json
from server.db import PostgresHistoryStore

async def main():
    store = PostgresHistoryStore()
    pool = await store._get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT role, content FROM messages ORDER BY id DESC LIMIT 5")
        for r in reversed(rows):
            print(f"ROLE: {r['role']}")
            print(f"CONTENT: {r['content']}")
            print("-" * 40)

asyncio.run(main())
