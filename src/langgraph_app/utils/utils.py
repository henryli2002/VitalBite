"""Utility functions for the LangGraph application."""

from typing import List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import base64
import io
import re

# Strip the server-injected `<attached_image ... />` marker before language
# detection. The marker may contain a Chinese `description` (e.g. a food
# summary from analyze_food_image), which would otherwise flip an English-only
# user's message to Chinese.
_PLACEHOLDER_RE = re.compile(r"<attached_image\b[^>]*/>", re.IGNORECASE)

# Explicit-language-override phrases. If the LATEST human message contains any
# of these, we honour that choice even if the history is predominantly in
# another language. Keep this list short and high-precision.
_ENGLISH_OVERRIDE_RE = re.compile(
    r"\b(?:in english|reply in english|respond in english|use english|english please|switch to english)\b",
    re.IGNORECASE,
)
_CHINESE_OVERRIDE_RE = re.compile(
    r"(?:用中文|说中文|讲中文|中文回复|中文回答|改用中文|切换到中文|reply in chinese|respond in chinese|in chinese)",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Any Chinese character → Chinese, else English."""
    text = _PLACEHOLDER_RE.sub(" ", text)
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


def _last_user_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _get_text_from_content(msg.content)
    return ""


def get_dominant_language(messages: List[BaseMessage], default_lang: str = "Chinese") -> str:
    """
    Determine the conversation's language. Simple rule: any Chinese character
    across user messages → Chinese, otherwise English. Exception: if the most
    recent user message explicitly requests a language (e.g. "reply in
    English"), that overrides history.
    """
    last = _PLACEHOLDER_RE.sub(" ", _last_user_text(messages))
    if _ENGLISH_OVERRIDE_RE.search(last):
        return "English"
    if _CHINESE_OVERRIDE_RE.search(last):
        return "Chinese"

    all_text = _PLACEHOLDER_RE.sub(" ", get_all_user_text(messages))
    if not all_text:
        return default_lang
    for char in all_text:
        if '\u4e00' <= char <= '\u9fff':
            return "Chinese"
    return "English"
