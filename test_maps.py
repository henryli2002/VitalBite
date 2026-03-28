import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

async def test_maps_tool():
    from langgraph_app.tools.map.google_maps import map_tool
    print("Testing map tool...")
    
    # Force a miss, then a hit
    payload = {
        "textQuery": "test restaurant singapore",
        "maxResultCount": 1
    }
    
    # 1. First call (MISS)
    print("Call 1 (MISS expected)...")
    res1 = await map_tool.search_restaurants(cuisine_type="test restaurant", location="singapore", max_results=1)
    print("Result 1:", res1)
    
    # 2. Second call (HIT expected)
    print("Call 2 (HIT expected)...")
    res2 = await map_tool.search_restaurants(cuisine_type="test restaurant", location="singapore", max_results=1)
    print("Result 2:", res2)

if __name__ == "__main__":
    asyncio.run(test_maps_tool())
