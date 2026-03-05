"""State definitions for LangGraph workflow."""

from typing import TypedDict, Optional, List, Any, Literal
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AnalysisData(TypedDict):
    """Analysis results from guardrail and routing."""
    intent: Literal["recognition", "recommendation", "exit", "chitchat", "tutorial", "guardrails", "goalplanning"]
    safety_safe: bool
    safety_reason: Optional[str]


class GraphState(TypedDict, total=False):
    """Main state structure for the LangGraph workflow."""
    # Context layer
    patient_id: Optional[str]
    session_id: str
    
    # Analysis layer
    analysis: AnalysisData
    
    # Business data layer
    recognition_result: Optional[dict]  # JSON: {foods: [...], nutrition: {...}}
    recommendation_result: Optional[dict]  # JSON: {restaurants: [...]}
    
    # Output layer
    final_response: str
    
    # Debug logging
    debug_logs: List[dict]
    
    # Message history (LangGraph standard)
    messages: Annotated[List[AnyMessage], add_messages]
