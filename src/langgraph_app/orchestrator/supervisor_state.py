"""Simplified state for the Supervisor-based graph.

Compared to the old GraphState, this removes fields that are now internal
to tools (recognition_result, recommendation_result, meal_time).
"""

from typing import TypedDict, Optional, List, Any, Dict
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Reuse the same reducer from state.py to avoid channel type conflicts
# when guardrail nodes (which reference the old state) run in this graph.
from langgraph_app.orchestrator.state import _add_logs


class SupervisorState(TypedDict, total=False):
    """State for the Supervisor-based graph.

    Transport concerns (``response_channel`` / publish callbacks) intentionally
    live on ``RunnableConfig.configurable`` rather than on state — the graph
    shouldn't know about Redis. See ``_create_supervisor_graph`` in graph.py.
    """
    # Context
    user_id: Optional[str]
    user_name: Optional[str]
    session_id: str
    user_profile: Optional[Dict[str, Any]]
    user_context: Optional[Dict[str, Any]]

    # Guardrails compatibility
    analysis: Optional[Dict[str, Any]]

    # Debug
    debug_logs: Annotated[List[dict], _add_logs]

    # Messages (LangGraph standard)
    messages: Annotated[List[AnyMessage], add_messages]
