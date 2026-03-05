"""Main LangGraph orchestration workflow."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.orchestrator.nodes.guardrail import global_guardrail_node
from langgraph_app.orchestrator.nodes.router import intent_router_node
from langgraph_app.agents.food_recognition.agent import food_recognition_node
from langgraph_app.agents.food_recommendation.agent import food_recommendation_node
from langgraph_app.agents.chitchat.agent import chitchat_node
from langgraph_app.agents.tutorial.agent import tutorial_node
from langgraph_app.agents.guardrails.agent import guardrails_node
from langgraph_app.agents.goalplanning.agent import goalplanning_node



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


def route_by_intent(state: GraphState) -> Literal["recognition", "recommendation", "exit", "chitchat", "tutorial", "guardrails", "goalplanning"]:
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
    workflow.add_node("guardrail", global_guardrail_node)
    workflow.add_node("router", intent_router_node)
    workflow.add_node("recognition", food_recognition_node)
    workflow.add_node("recommendation", food_recommendation_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("tutorial", tutorial_node)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("goalplanning", goalplanning_node)
    
    # Define workflow edges
    # START -> guardrail
    workflow.set_entry_point("guardrail")
    
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
            "chitchat": "chitchat",
            "tutorial": "tutorial",
            "guardrails": "guardrails",
            "goalplanning": "goalplanning",
            "exit": END
        }
    )

    # All agent nodes -> END
    workflow.add_edge("recognition", END)
    workflow.add_edge("recommendation", END)
    workflow.add_edge("chitchat", END)
    workflow.add_edge("tutorial", END)
    workflow.add_edge("guardrails", END)
    workflow.add_edge("goalplanning", END)
    
    return workflow.compile()


# Create the graph instance
graph = create_graph()
