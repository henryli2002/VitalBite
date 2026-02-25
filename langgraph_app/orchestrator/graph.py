"""Main LangGraph orchestration workflow."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.orchestrator.nodes.normalization import normalize_input_node
from langgraph_app.orchestrator.nodes.guardrail import global_guardrail_node
from langgraph_app.orchestrator.nodes.router import intent_router_node
from langgraph_app.agents.food_recognition.agent import food_recognition_node
from langgraph_app.agents.food_recommendation.agent import food_recommendation_node
from langgraph_app.agents.clarification.agent import clarification_node



def should_continue_after_guardrail(state: GraphState) -> Literal["unsafe", "safe"]:
    """
    Condition function: Check if input passed safety check.
    
    Args:
        state: Current graph state
        
    Returns:
        "unsafe" if content is unsafe, "safe" otherwise
    """
    analysis = state.get("analysis", {})
    if not analysis.get("safety_safe", True):
        return "unsafe"
    return "safe"


def route_by_intent(state: GraphState) -> Literal["recognition", "recommendation", "clarification", "exit"]:
    """
    Condition function: Route to appropriate agent based on intent.
    
    Args:
        state: Current graph state
        
    Returns:
        Intent string to route to
    """
    analysis = state.get("analysis", {})
    intent = analysis.get("intent", "")
    return intent


def create_graph() -> StateGraph:
    """
    Create and configure the LangGraph workflow.
    
    Returns:
        Configured StateGraph instance
    """
    # Create graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("normalize", normalize_input_node)
    workflow.add_node("guardrail", global_guardrail_node)
    workflow.add_node("router", intent_router_node)
    workflow.add_node("recognition", food_recognition_node)
    workflow.add_node("recommendation", food_recommendation_node)
    workflow.add_node("clarification", clarification_node)
    
    # Define workflow edges
    # START -> normalize
    workflow.set_entry_point("normalize")
    
    # normalize -> guardrail
    workflow.add_edge("normalize", "guardrail")
    
    # guardrail -> condition: safe or unsafe?
    workflow.add_conditional_edges(
        "guardrail",
        should_continue_after_guardrail,
        {
            "unsafe": END,  # End with safety warning
            "safe": "router"  # Continue to routing
        }
    )
    
    # router -> condition: route by intent
    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "recognition": "recognition",
            "recommendation": "recommendation",
            "clarification": "clarification",
            "exit": END
        }
    )

    # clarification -> END (wait for next input)
    workflow.add_edge("clarification", END)
    
    # All agent nodes -> END
    workflow.add_edge("recognition", END)
    workflow.add_edge("recommendation", END)
    
    return workflow.compile()


# Create the graph instance
graph = create_graph()
