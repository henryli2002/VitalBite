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
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph_app.config import config

# Singleton cache: (provider, model, module) -> BaseChatModel instance
_LLM_CACHE: Dict[Tuple[str, str, str], BaseChatModel] = {}


def get_llm_client(provider: Optional[str] = None, model_name: Optional[str] = None, module: Optional[str] = None) -> BaseChatModel:
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
    else:
        resolved_model = model_name or "unknown"

    cache_key = (selected, resolved_model, module or "")

    if cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    # Common parameters for LangChain models
    common_params: Dict[str, Any] = {
        "temperature": sampling_params.get("temperature", 0.7),
        "streaming": True,
        "model_kwargs": {}
    }
    if "top_p" in sampling_params:
        if selected in {"openai"}:
            common_params["model_kwargs"]["top_p"] = sampling_params["top_p"]
            
    if selected == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        client = ChatGoogleGenerativeAI(model=resolved_model, **common_params)
        
    elif selected == "openai":
        from langchain_openai import ChatOpenAI
        if "presence_penalty" in sampling_params:
            common_params["model_kwargs"]["presence_penalty"] = sampling_params["presence_penalty"]
        client = ChatOpenAI(model=resolved_model, **common_params)
        
    elif selected in {"bedrock", "bedrock_claude", "claude"}:
        from langchain_aws import ChatBedrockConverse
        region = os.getenv("AWS_REGION") or "us-east-1"
        client = ChatBedrockConverse(
            model=resolved_model,
            region_name=region,
            max_tokens=config.BEDROCK_CLAUDE_MAX_TOKENS,
            **common_params
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {selected}")

    _LLM_CACHE[cache_key] = client
    return client
