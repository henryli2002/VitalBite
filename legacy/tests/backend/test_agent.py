import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

async def test_agent():
    from langgraph_app.orchestrator.graph import graph
    from langchain_core.messages import HumanMessage
    
    initial_state = {
        "messages": [HumanMessage(content="I want to eat seafood in Jurong")],
        "session_id": "test_session",
        "user_id": "user_test",
        "user_name": "Test User",
        "user_profile": {"dietary_preferences": "None"},
    }
    
    config = {"configurable": {"thread_id": "test_thread"}}
    print("Invoking graph for seafood search...")
    result = await graph.ainvoke(initial_state, config=config)
    print("RESULT:", result.get("messages")[-1].content)
    
    print("\n\nInvoking graph for seafood search AGAIN to trigger CACHE HIT...")
    initial_state2 = {
        "messages": [HumanMessage(content="I want to eat seafood in Jurong")],
        "session_id": "test_session",
        "user_id": "user_test",
        "user_name": "Test User",
        "user_profile": {"dietary_preferences": "None"},
    }
    config2 = {"configurable": {"thread_id": "test_thread2"}}
    result2 = await graph.ainvoke(initial_state2, config=config2)
    print("RESULT 2:", result2.get("messages")[-1].content)

if __name__ == "__main__":
    asyncio.run(test_agent())
