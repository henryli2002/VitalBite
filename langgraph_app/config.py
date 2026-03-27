"""Configuration settings for the WABI NTU Agent."""

import os
from typing import Optional, Dict


class Config:
    """Application configuration."""
    
    # Model Configuration
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    BEDROCK_CLAUDE_MODEL_NAME: str = os.getenv("BEDROCK_CLAUDE_MODEL_NAME", "anthropic.claude-3-haiku-20240307-v1:0")

    # Inference configuration
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    BEDROCK_CLAUDE_TEMPERATURE: float = float(os.getenv("BEDROCK_CLAUDE_TEMPERATURE", "0.2"))
    BEDROCK_CLAUDE_MAX_TOKENS: int = int(os.getenv("BEDROCK_CLAUDE_MAX_TOKENS", "2048"))

    # Default sampling parameters for all LLM calls
    DEFAULT_SAMPLING_PARAMS = {
        "temperature": 0.2,
        "top_p": 0.9,
        "presence_penalty": 0.0,
    }

    # Module-specific overrides for sampling parameters (example structure)
    LLM_SAMPLING = {
        "gemini": {
            "router": {"temperature": 0.1, "top_p": 0.8},  # Router module using Gemini
            "clarification": {"temperature": 0.4, "top_p": 0.95},  # Clarification module using Gemini
            "food_recognition": {"temperature": 0.0, "top_p": 0.1},  # Food Recognition MUST be deterministic
            "food_recommendation": {"temperature": 0.5, "top_p": 0.95},  # Food Recommendation module using Gemini
        },
        "openai": {
            "router": {"temperature": 0.1, "top_p": 0.8},  # Router module using OpenAI
            "clarification": {"temperature": 0.4, "top_p": 0.95, "presence_penalty": 0.1},  # Clarification module using OpenAI
            "food_recognition": {"temperature": 0.0, "top_p": 0.1},  # Food Recognition MUST be deterministic
            "food_recommendation": {"temperature": 0.5, "top_p": 0.95, "presence_penalty": 0.2},  # Food Recommendation module using OpenAI
        },
        "bedrock_claude": {
            "router": {"temperature": 0.1, "top_p": 0.8},  # Router module using Bedrock Claude
            "clarification": {"temperature": 0.4, "top_p": 0.95},  # Clarification module using Bedrock Claude
            "food_recognition": {"temperature": 0.0, "top_p": 0.1},  # Food Recognition MUST be deterministic
            "food_recommendation": {"temperature": 0.5, "top_p": 0.95},  # Food Recommendation module using Bedrock Claude
        },
    }

    # LLM Provider Selection: gemini | openai | bedrock_claude
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    
    @classmethod
    def get_provider_for_module(cls, module: Optional[str] = None) -> str:
        """Get the specific LLM provider for a module or fallback to default."""
        if not module:
            return cls.LLM_PROVIDER
        
        # e.g., LLM_PROVIDER_ROUTER=openai
        env_key = f"LLM_PROVIDER_{module.upper()}"
        env_val = os.getenv(env_key)
        if env_val:
            return env_val
            
        return cls.LLM_PROVIDER
    
    # Conversation History Configuration
    # Number of recent messages to include in context (includes both user and AI messages)
    # For example, 6 messages = last 3 conversation turns (3 user + 3 AI)
    HISTORY_MESSAGE_COUNT: int = int(os.getenv("HISTORY_MESSAGE_COUNT", "6"))
    
    # Agent-specific configurations
    ROUTER_HISTORY_COUNT: Optional[int] = None  # If None, uses HISTORY_MESSAGE_COUNT
    CLARIFICATION_HISTORY_COUNT: Optional[int] = None
    RECOGNITION_HISTORY_COUNT: Optional[int] = None
    RECOMMENDATION_HISTORY_COUNT: Optional[int] = None
    
    @classmethod
    def get_history_count(cls, agent_name: str) -> int:
        """
        Get history count for a specific agent.
        
        Args:
            agent_name: Name of the agent (router, clarification, recognition, recommendation)
            
        Returns:
            Number of messages to include in history
        """
        agent_specific = getattr(cls, f"{agent_name.upper()}_HISTORY_COUNT", None)
        return agent_specific if agent_specific is not None else cls.HISTORY_MESSAGE_COUNT

    @classmethod
    def get_sampling_params(cls, provider: str, module: Optional[str] = None) -> Dict[str, float]:
        params = cls.DEFAULT_SAMPLING_PARAMS.copy()

        # Apply module-specific overrides
        if module and provider in cls.LLM_SAMPLING:
            module_params = cls.LLM_SAMPLING[provider].get(module, {})
            params.update(module_params)

        # Apply environment variable overrides (e.g., LLM_SAMPLING_OPENAI_ROUTER_TEMPERATURE=0.1)
        for param_name in ["temperature", "top_p", "presence_penalty"]:
            env_var_key = f"LLM_SAMPLING_{provider.upper()}_{module.upper()}_{param_name.upper()}" if module else f"LLM_SAMPLING_{provider.upper()}_{param_name.upper()}"
            env_value = os.getenv(env_var_key)
            if env_value is not None:
                try:
                    params[param_name] = float(env_value)
                except ValueError:
                    pass  # Ignore invalid env values
        return params


# Create a singleton instance
config = Config()
