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

def get_all_user_text(messages: List[BaseMessage]) -> str:
    """Extracts all text from user messages in the history."""
    user_texts = [
        _get_text_from_content(msg.content)
        for msg in messages
        if isinstance(msg, HumanMessage)
    ]
    return " ".join(user_texts).strip()


def get_dominant_language(messages: List[BaseMessage], default_lang: str = "Chinese") -> str:
    """
    Determines the dominant language from the conversation history,
    defaulting to a specified language if no text is found.
    """
    all_text = get_all_user_text(messages)
    if not all_text:
        return default_lang

    chinese_chars = 0
    english_chars = 0

    for char in all_text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_chars += 1
        elif 'a' <= char.lower() <= 'z':
            english_chars += 1

    # Simple heuristic: if Chinese characters are present, assume Chinese is dominant.
    # This can be refined if more languages are added.
    if chinese_chars > 0:
        return "Chinese"
    if english_chars > 0:
        return "English"

    return default_lang
