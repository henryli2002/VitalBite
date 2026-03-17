"""LLM client factory for configurable provider switching."""

from __future__ import annotations

from typing import Optional

from langgraph_app.config import config
from langgraph_app.utils.gemini_client import GeminiClient
from langgraph_app.utils.openai_client import OpenAIClient
from langgraph_app.utils.bedrock_claude_client import BedrockClaudeClient


def get_llm_client(provider: Optional[str] = None, model_name: Optional[str] = None, module: Optional[str] = None):
    """
    Return an LLM client instance based on provider/config.

    Args:
        provider: One of {"gemini", "openai", "bedrock_claude"}. If None, uses config.LLM_PROVIDER.
        model_name: Optional model override.
    """

    selected = (provider or config.get_provider_for_module(module)).lower()

    if selected == "gemini":
        return GeminiClient(model_name=model_name, module=module)
    if selected == "openai":
        return OpenAIClient(model_name=model_name, module=module)
    if selected in {"bedrock", "bedrock_claude", "claude"}:
        return BedrockClaudeClient(model_name=model_name, module=module)

    raise ValueError(f"Unsupported LLM provider: {selected}")
