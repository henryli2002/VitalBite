import asyncio
import os
from dotenv import load_dotenv
load_dotenv(".env")

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List
from langgraph_app.utils.tracked_llm import get_tracked_llm
import base64

class BoundingBox(BaseModel):
    ymin: int = Field(description="Y min (0-1000)")
    xmin: int = Field(description="X min (0-1000)")
    ymax: int = Field(description="Y max (0-1000)")
    xmax: int = Field(description="X max (0-1000)")

class DetectedFood(BaseModel):
    name: str = Field(description="Name of the food item")
    box: BoundingBox = Field(description="Bounding box of the food item")

class FoodDetection(BaseModel):
    items: List[DetectedFood]

async def main():
    llm = get_tracked_llm(provider="gemini", module="food_recognition", node_name="test")
    structured_llm = llm.with_structured_output(FoodDetection)
    
    with open("burger.jpg", "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")
        
    msg = HumanMessage(content=[
        {"type": "text", "text": "Detect all distinct food items in this image and provide their names and bounding boxes (normalized 0-1000)."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
    ])
    
    res = await structured_llm.ainvoke([msg])
    print(res)

asyncio.run(main())
