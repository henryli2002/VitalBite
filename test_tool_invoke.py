import asyncio
from langgraph_app.tools.food_recognition_tool import analyze_food_image
from langchain_core.runnables import RunnableConfig

async def main():
    config = RunnableConfig(configurable={"user_id": "test_user"})
    # Since it's a langchain tool, we invoke it
    res = await analyze_food_image.ainvoke({"image_uuid": "1234567890abcdef1234567890abcdef"}, config=config)
    print("Result:", res)

asyncio.run(main())
