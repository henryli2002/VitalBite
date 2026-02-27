# 按照id可视化testcase里所有的图片（id为该问题的id）

import os
import base64
from typing import Optional
from langgraph_app.utils.gemini_client import GeminiClient
from langgraph_app.orchestrator.state import GraphState
from langchain_core.messages import HumanMessage, AIMessage
import json

def decode_image(image_b64: str, output_path: str) -> None:
    """Decode a base64 image string and save it to the specified output path."""
    image_data = base64.b64decode(image_b64)
    with open(output_path, "wb") as image_file:
        image_file.write(image_data)

def main():
    test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    if not os.path.exists(test_cases_path):
        print(f"Error: {test_cases_path} not found.")
        return
    
    with open(test_cases_path, "r") as f:
        test_cases = json.load(f)
    
    output_dir = os.path.join(os.path.dirname(__file__), "used_images")
    os.makedirs(output_dir, exist_ok=True)
    
    for case in test_cases:
        case_id = case["id"]
        input_data = case["input"]
        image_b64 = input_data.get("image_data")
        
        if image_b64:
            output_path = os.path.join(output_dir, f"{case_id}.png")
            decode_image(image_b64, output_path)
            print(f"Saved image for case ID {case_id} to {output_path}")
        else:
            print(f"No image data for case ID {case_id}")

if __name__ == "__main__":
    main()