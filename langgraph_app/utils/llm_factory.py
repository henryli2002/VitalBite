"""LLM client factory for configurable provider switching."""

from __future__ import annotations

from typing import Optional
import os
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph_app.config import config

def get_llm_client(provider: Optional[str] = None, model_name: Optional[str] = None, module: Optional[str] = None) -> BaseChatModel:
    """
    Return a standard LangChain Chat Model instance based on provider/config.

    Args:
        provider: One of {"gemini", "openai", "bedrock_claude"}. If None, uses config.LLM_PROVIDER.
        model_name: Optional model override.
        module: The functional module (e.g., "router", "chitchat") to load specific sampling params.
    """

    selected = (provider or config.get_provider_for_module(module)).lower()
    sampling_params = config.get_sampling_params(selected, module)
    
    # Common parameters for LangChain models
    common_params = {
        "temperature": sampling_params.get("temperature", 0.7),
        "streaming": True,
        "model_kwargs": {}
    }
    if "top_p" in sampling_params:
        if selected in {"openai"}:
            common_params["model_kwargs"]["top_p"] = sampling_params["top_p"]
            
    if selected == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = model_name or config.GEMINI_MODEL_NAME
        return ChatGoogleGenerativeAI(model=model, **common_params)
        
    if selected == "openai":
        from langchain_openai import ChatOpenAI
        model = model_name or config.OPENAI_MODEL_NAME
        # Map presence_penalty if available
        if "presence_penalty" in sampling_params:
            common_params["model_kwargs"]["presence_penalty"] = sampling_params["presence_penalty"]
        return ChatOpenAI(model=model, **common_params)
        
    if selected in {"bedrock", "bedrock_claude", "claude"}:
        from langchain_aws import ChatBedrockConverse
        model = model_name or config.BEDROCK_CLAUDE_MODEL_NAME
        region = os.getenv("AWS_REGION") or "us-east-1"
        return ChatBedrockConverse(
            model=model,
            region_name=region,
            max_tokens=config.BEDROCK_CLAUDE_MAX_TOKENS,
            **common_params
        )

    raise ValueError(f"Unsupported LLM provider: {selected}")
