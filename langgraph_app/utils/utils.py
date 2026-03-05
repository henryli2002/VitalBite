"""Utility functions for the LangGraph application."""

from typing import List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import base64
import io

def detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return "Chinese"
    return "English"

def _get_text_from_content(content: Any) -> str:
    """Extracts plain text from a LangChain message content part."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return " ".join(text_parts).strip()
    return ""

def get_current_user_text(messages: List[Any]) -> str:
    """Gets the text from the most recent user message."""
    if not messages or not isinstance(messages[-1], HumanMessage):
        return ""
    return _get_text_from_content(messages[-1].content)
