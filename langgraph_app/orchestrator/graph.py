"""Main LangGraph orchestration workflow."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.orchestrator.nodes.guardrails import (
    input_guardrail_node,
    output_guardrail_node,
)
from langgraph_app.orchestrator.nodes.router import intent_router_node
from langgraph_app.agents.food_recognition.agent import recognition_node
from langgraph_app.agents.food_recommendation.agent import food_recommendation_node
from langgraph_app.agents.chitchat.agent import chitchat_node
from langgraph_app.agents.tutorial.agent import tutorial_node
from langgraph_app.agents.goalplanning.agent import goalplanning_node


def should_continue(state: GraphState) -> Literal["unsafe", "safe"]:
    """Condition function to check for safety."""
    analysis = state.get("analysis", {})
    if not analysis.get("safety_safe", True):
        return "unsafe"
    return "safe"


def route_by_intent(
    state: GraphState,
) -> Literal["recognition", "recommendation", "chitchat", "tutorial", "goalplanning"]:
    """Condition function to route based on intent."""
    analysis = state.get("analysis", {})
    intent = analysis.get("intent", "chitchat")

    # Fallback in case "guardrails" or other invalid intent somehow leaks through
    if intent not in [
        "recognition",
        "recommendation",
        "chitchat",
        "tutorial",
        "goalplanning",
    ]:
        return "chitchat"

    return intent  # type: ignore


def create_graph():  # type: ignore
    """Create and configure the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("router", intent_router_node)
    workflow.add_node("recognition", recognition_node)
    workflow.add_node("recommendation", food_recommendation_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("tutorial", tutorial_node)
    workflow.add_node("goalplanning", goalplanning_node)
    workflow.add_node("output_guardrail", output_guardrail_node)

    # Define workflow edges
    workflow.set_entry_point("input_guardrail")

    # If input is unsafe, short-circuit to END
    workflow.add_conditional_edges(
        "input_guardrail",
        should_continue,
        {"unsafe": END, "safe": "router"},
    )

    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "recognition": "recognition",
            "recommendation": "recommendation",
            "chitchat": "chitchat",
            "tutorial": "tutorial",
            "goalplanning": "goalplanning",
        },
    )

    workflow.add_edge("recognition", "output_guardrail")
    workflow.add_edge("recommendation", "output_guardrail")
    workflow.add_edge("chitchat", "output_guardrail")
    workflow.add_edge("tutorial", "output_guardrail")
    workflow.add_edge("goalplanning", "output_guardrail")

    # If output is unsafe, also short-circuit to END.
    # Note: Output guardrail currently does not append an AIMessage in its node directly
    # like input guardrail does, but we bypass the old 'guardrails' agent.
    workflow.add_conditional_edges(
        "output_guardrail",
        should_continue,
        {"unsafe": END, "safe": END},
    )

    return workflow.compile()


# Create the graph instance
graph = create_graph()
