"""Configuration settings for the WABI NTU Agent."""

import os
from typing import Optional


class Config:
    """Application configuration."""
    
    # Model Configuration
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    BEDROCK_CLAUDE_MODEL_NAME: str = os.getenv("BEDROCK_CLAUDE_MODEL_NAME", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    # LLM Provider Selection: gemini | openai | bedrock_claude
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    
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


# Create a singleton instance
config = Config()
