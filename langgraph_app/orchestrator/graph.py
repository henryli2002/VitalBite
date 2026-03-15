"""Main LangGraph orchestration workflow."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.orchestrator.nodes.guardrail import (
    input_guardrail_node,
    output_guardrail_node,
)
from langgraph_app.orchestrator.nodes.router import intent_router_node
from langgraph_app.agents.food_recognition.agent import food_recognition_node
from langgraph_app.agents.food_recommendation.agent import food_recommendation_node
from langgraph_app.agents.chitchat.agent import chitchat_node
from langgraph_app.agents.tutorial.agent import tutorial_node
from langgraph_app.agents.guardrails.agent import guardrails_node
from langgraph_app.agents.goalplanning.agent import goalplanning_node

def should_continue(state: GraphState) -> Literal["unsafe", "safe"]:
    """
    Condition function: Check if content passed safety check.
    
    Args:
        state: Current graph state
        
    Returns:
        "unsafe" if content is unsafe, "safe" otherwise
    """
    analysis = state.get("analysis", {})
    if not analysis.get("safety_safe", True):
        return "unsafe"
    return "safe"

def route_by_intent(state: GraphState) -> Literal[
    "recognition", "recommendation", "chitchat", "tutorial", "guardrails", "goalplanning"
]:
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
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("router", intent_router_node)
    workflow.add_node("recognition", food_recognition_node)
    workflow.add_node("recommendation", food_recommendation_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("tutorial", tutorial_node)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("goalplanning", goalplanning_node)
    workflow.add_node("output_guardrail", output_guardrail_node)

    # Define workflow edges
    workflow.set_entry_point("input_guardrail")

    workflow.add_conditional_edges(
        "input_guardrail",
        should_continue,
        {
            "unsafe": "guardrails",
            "safe": "router",
        },
    )

    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "recognition": "recognition",
            "recommendation": "recommendation",
            "chitchat": "chitchat",
            "tutorial": "tutorial",
            "guardrails": "guardrails",
            "goalplanning": "goalplanning",
        },
    )

    workflow.add_edge("recognition", "output_guardrail")
    workflow.add_edge("recommendation", "output_guardrail")
    workflow.add_edge("chitchat", "output_guardrail")
    workflow.add_edge("tutorial", "output_guardrail")
    workflow.add_edge("goalplanning", "output_guardrail")

    workflow.add_conditional_edges(
        "output_guardrail",
        should_continue,
        {
            "unsafe": "guardrails",
            "safe": END,
        },
    )

    workflow.add_edge("guardrails", END)
    
    return workflow.compile()

# Create the graph instance
graph = create_graph()
