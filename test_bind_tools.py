import json
from langgraph_app.tools.food_recognition_tool import analyze_food_image
from langchain_core.utils.function_calling import convert_to_openai_tool

print(json.dumps(convert_to_openai_tool(analyze_food_image), indent=2))
