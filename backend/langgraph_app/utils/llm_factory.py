"""LLM client factory with singleton caching.

Previously, every call to get_tracked_llm() → get_llm_client() created a brand new
ChatGoogleGenerativeAI/ChatOpenAI instance. At 200 concurrency, this means 200
TLS handshake negotiations happening simultaneously.

LangChain Chat Model instances are thread-safe and reentrant for async calls.
This module caches them by (provider, model_name, module) to reuse connections.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, Any
import os
import datetime
import platform

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph_app.config import config

# Singleton cache: (provider, model, module) -> BaseChatModel instance
_LLM_CACHE: Dict[Tuple[str, str, str], BaseChatModel] = {}


def _get_dynamic_env_context() -> str:
    """Returns a formatted string with the current time and host system information."""
    try:
        # Use timezone-aware datetime for UTC+8 (China Standard Time)
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        cst_now = utc_now.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
        formatted_time = cst_now.strftime("%Y-%m-%dT%H:%M:%S UTC+8")
    except Exception:
        # Fallback in case of any issue
        now = datetime.datetime.now(datetime.timezone.utc)
        formatted_time = now.isoformat()

    system_info = f"{platform.system()} {platform.release()}"
    return f"[System Environment: {formatted_time}, OS: {system_info}]"


def inject_dynamic_context(messages: list) -> list:
    """
    Injects dynamic environment context into the last user message without
    modifying the original list (e.g., from state).
    """
    if not messages:
        return []

    # Create a deep copy to avoid modifying the original list in the state
    try:
        messages_copy = [
            msg.copy(deep=True) if isinstance(msg, BaseMessage) else msg.copy()
            for msg in messages
        ]
    except (AttributeError, TypeError):  # pragma: no cover
        # Fallback for older LangChain versions or unexpected types
        import copy

        messages_copy = copy.deepcopy(messages)

    last_message = messages_copy[-1]
    dynamic_context = _get_dynamic_env_context()

    # Check for LangChain's HumanMessage or a dict with role 'user'
    is_human_message = isinstance(last_message, HumanMessage)
    is_user_dict = isinstance(last_message, dict) and last_message.get("role") == "user"

    if is_human_message:
        # Ensure content is a string before concatenation
        original_content = last_message.content
        if isinstance(original_content, str):
            last_message.content = f"{dynamic_context}\\n\\n{original_content}"
        # If content is a list (e.g., for vision), we might need a more complex strategy,
        # but for now, we'll only handle the string case.

    elif is_user_dict:
        original_content = last_message.get("content", "")
        if isinstance(original_content, str):
            last_message["content"] = f"{dynamic_context}\\n\\n{original_content}"

    return messages_copy


def get_llm_client(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    module: Optional[str] = None,
) -> BaseChatModel:
    """
    Return a standard LangChain Chat Model instance based on provider/config.
    Results are cached by (provider, model, module) to reuse TCP connections.
    """
    selected = (provider or config.get_provider_for_module(module)).lower()
    sampling_params = config.get_sampling_params(selected, module)

    # Resolve the actual model name for cache key
    if selected == "gemini":
        resolved_model = model_name or config.GEMINI_MODEL_NAME
    elif selected == "openai":
        resolved_model = model_name or config.OPENAI_MODEL_NAME
    elif selected in {"bedrock", "bedrock_claude", "claude"}:
        resolved_model = model_name or config.BEDROCK_CLAUDE_MODEL_NAME
    elif selected == "llamacpp":
        resolved_model = model_name or config.LLAMACPP_MODEL_NAME
    else:
        resolved_model = model_name or "unknown"

    cache_key = (selected, resolved_model, module or "")

    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    # Common parameters for LangChain models
    common_params: Dict[str, Any] = {
        "temperature": sampling_params.get("temperature", 0.7),
        "streaming": True,
        "model_kwargs": {},
    }
    if "top_p" in sampling_params:
        if selected in {"openai"}:
            common_params["model_kwargs"]["top_p"] = sampling_params["top_p"]
        elif selected == "llamacpp":
            pass  # top_p handled in extra_body below

    if selected == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        client = ChatGoogleGenerativeAI(model=resolved_model, **common_params)

    elif selected == "openai":
        from langchain_openai import ChatOpenAI

        if "presence_penalty" in sampling_params:
            common_params["model_kwargs"]["presence_penalty"] = sampling_params[
                "presence_penalty"
            ]
        client = ChatOpenAI(model=resolved_model, **common_params)

    elif selected in {"bedrock", "bedrock_claude", "claude"}:
        from langchain_aws import ChatBedrockConverse

        region = os.getenv("AWS_REGION") or "us-east-1"
        client = ChatBedrockConverse(
            model=resolved_model,
            region_name=region,
            max_tokens=config.BEDROCK_CLAUDE_MAX_TOKENS,
            **common_params,
        )
    elif selected == "llamacpp":
        from langchain_openai import ChatOpenAI

        extra_body_params = {}
        for key in [
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "reasoning_budget",
            "reasoning_level",
        ]:
            if key in sampling_params:
                extra_body_params[key] = sampling_params.pop(key)
        client = ChatOpenAI(
            model=resolved_model,
            base_url=config.LLAMACPP_API_BASE,
            extra_body=extra_body_params if extra_body_params else None,
            **common_params,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {selected}")

    _LLM_CACHE[cache_key] = client
    return client
