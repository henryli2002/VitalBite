"""Configuration settings for the WABI NTU Agent."""

import os
from typing import Optional, Dict


class Config:
    """Application configuration."""

    # Model Configuration
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    BEDROCK_CLAUDE_MODEL_NAME: str = os.getenv(
        "BEDROCK_CLAUDE_MODEL_NAME", "anthropic.claude-3-haiku-20240307-v1:0"
    )

    # Inference configuration
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
            "clarification": {
                "temperature": 0.4,
                "top_p": 0.95,
            },  # Clarification module using Gemini
            "food_recognition": {
                "temperature": 0.0,
                "top_p": 0.1,
            },  # Food Recognition MUST be deterministic
            "food_recommendation": {
                "temperature": 0.5,
                "top_p": 0.95,
            },  # Food Recommendation module using Gemini
        },
        "openai": {
            "router": {"temperature": 0.1, "top_p": 0.8},  # Router module using OpenAI
            "clarification": {
                "temperature": 0.4,
                "top_p": 0.95,
                "presence_penalty": 0.1,
            },  # Clarification module using OpenAI
            "food_recognition": {
                "temperature": 0.0,
                "top_p": 0.1,
            },  # Food Recognition MUST be deterministic
            "food_recommendation": {
                "temperature": 0.5,
                "top_p": 0.95,
                "presence_penalty": 0.2,
            },  # Food Recommendation module using OpenAI
        },
        "bedrock_claude": {
            "router": {
                "temperature": 0.1,
                "top_p": 0.8,
            },  # Router module using Bedrock Claude
            "clarification": {
                "temperature": 0.4,
                "top_p": 0.95,
            },  # Clarification module using Bedrock Claude
            "food_recognition": {
                "temperature": 0.0,
                "top_p": 0.1,
            },  # Food Recognition MUST be deterministic
            "food_recommendation": {
                "temperature": 0.5,
                "top_p": 0.95,
            },  # Food Recommendation module using Bedrock Claude
        },
        "llamacpp": {
            "router": {
                "temperature": 0.1,
                "top_p": 0.9,
                "reasoning_budget": -1,
                "repeat_penalty": 1.1,
                "min_p": 0.05,
            },
            "clarification": {
                "temperature": 0.4,
                "top_p": 0.95,
                "reasoning_budget": -1,
                "repeat_penalty": 1.1,
                "min_p": 0.05,
            },
            "food_recognition": {
                "temperature": 0.0,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "min_p": 0.05,
            },
            "food_recommendation": {
                "temperature": 0.5,
                "top_p": 0.95,
                "reasoning_budget": -1,
                "repeat_penalty": 1.1,
                "min_p": 0.05,
            },
        },
    }

    # LLM Provider Selection: gemini | openai | bedrock_claude | llamacpp
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

    # Llama.cpp Configuration
    LLAMACPP_API_BASE: str = os.getenv(
        "LLAMACPP_API_BASE", "http://msi.tailb813aa.ts.net:8080"
    )
    LLAMACPP_MODEL_NAME: str = os.getenv("LLAMACPP_MODEL_NAME", "qwen")

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

    @classmethod
    def get_sampling_params(
        cls, provider: str, module: Optional[str] = None
    ) -> Dict[str, float]:
        params = cls.DEFAULT_SAMPLING_PARAMS.copy()

        # Apply module-specific overrides
        if module and provider in cls.LLM_SAMPLING:
            module_params = cls.LLM_SAMPLING[provider].get(module, {})
            params.update(module_params)

        # Apply environment variable overrides (e.g., LLM_SAMPLING_OPENAI_ROUTER_TEMPERATURE=0.1)
        float_params = [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "presence_penalty",
            "reasoning_budget",
        ]
        int_params = ["top_k", "reasoning_budget"]
        for param_name in float_params:
            env_var_key = (
                f"LLM_SAMPLING_{provider.upper()}_{module.upper()}_{param_name.upper()}"
                if module
                else f"LLM_SAMPLING_{provider.upper()}_{param_name.upper()}"
            )
            env_value = os.getenv(env_var_key)
            if env_value is not None:
                try:
                    params[param_name] = (
                        int(env_value) if param_name in int_params else float(env_value)
                    )
                except ValueError:
                    pass  # Ignore invalid env values
        return params


# Create a singleton instance
config = Config()
