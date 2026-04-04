"""State definitions for LangGraph workflow."""

import operator
from typing import TypedDict, Optional, List, Any, Literal, Dict
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def _add_logs(left: List[dict], right: List[dict]) -> List[dict]:
    """Reducer for debug logs to ensure they are appended, not overwritten."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right


class AnalysisData(TypedDict, total=False):
    """Analysis results from guardrail and routing."""
    intent: Literal["recognition", "recommendation", "chitchat", "tutorial", "guardrails", "goalplanning"]
    confidence: Optional[float]
    reasoning: Optional[str]
    safety_safe: bool
    safety_reason: Optional[str]
    safety_category: Optional[str]


def _add_timestamps(left: List[str], right: List[str]) -> List[str]:
    """Reducer for message timestamps to ensure they are appended."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class GraphState(TypedDict, total=False):
    """Main state structure for the LangGraph workflow."""
    # Context layer
    user_id: Optional[str]
    user_name: Optional[str]
    session_id: str
    
    # User profile (age, height, weight, health conditions, etc.)
    user_profile: Optional[Dict[str, Any]]
    
    # User context (location lat/lng, IP address, etc. sent from frontend)
    user_context: Optional[Dict[str, Any]]
    response_channel: Optional[str]
    
    # Analysis layer
    analysis: AnalysisData
    
    # Business data layer
    recognition_result: Optional[dict]  # JSON: {foods: [...], nutrition: {...}}
    recommendation_result: Optional[dict]  # JSON: {restaurants: [...]}
    
    # Output layer
    
    # Debug logging
    debug_logs: Annotated[List[dict], _add_logs]
    
    # Message history (LangGraph standard)
    messages: Annotated[List[AnyMessage], add_messages]
    message_timestamps: Annotated[List[str], _add_timestamps]

# Standardized return type for all nodes
class NodeOutput(TypedDict, total=False):
    user_id: Optional[str]
    user_name: Optional[str]
    session_id: str
    user_profile: Optional[Dict[str, Any]]
    analysis: AnalysisData
    recognition_result: Optional[dict]
    recommendation_result: Optional[dict]
    debug_logs: List[dict]
    messages: List[AnyMessage]
    message_timestamps: List[str]

