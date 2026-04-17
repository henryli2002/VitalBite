import asyncio
from langgraph_app.tools.food_recognition_tool import analyze_food_image
import json

print(json.dumps(analyze_food_image.args_schema.schema(), indent=2))
