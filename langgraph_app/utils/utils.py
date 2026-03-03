

from langchain_core.messages import HumanMessage
from typing import List, Optional, Any

def detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return "Chinese"
    return "English"

def get_images_from_history(messages: List[Any]) -> Optional[List[str]]:
    """
    Scans message history in reverse to find the last HumanMessage with images.
    """
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        
        if isinstance(msg.content, list):
            image_blocks = []
            for block in msg.content:
                if not isinstance(block, dict):
                    continue
                
                if block.get("type") == "image_url":
                    image_url = block.get("image_url", {}).get("url", "")
                    if "base64," in image_url:
                        image_blocks.append(image_url.split("base64,")[1])
                elif block.get("type") == "image" and block.get("source_type") == "base64":
                    if block.get("data"):
                        image_blocks.append(block.get("data"))
            
            if image_blocks:
                return image_blocks
    return None