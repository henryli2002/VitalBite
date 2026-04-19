"""Main LangGraph orchestration workflow.

Supports two modes controlled by the USE_SUPERVISOR environment variable:
- USE_SUPERVISOR=1 (default): Supervisor Agent + Tools architecture
- USE_SUPERVISOR=0: Legacy static Router + Workflow DAG (for rollback)
"""

import json
import os
from typing import Literal

import redis.asyncio as redis
from langgraph.graph import StateGraph, END

from langgraph_app.config import config as _app_config
from langgraph_app.orchestrator.thinking import build_thinking_partials
from langgraph_app.utils.utils import get_dominant_language


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------
USE_SUPERVISOR = os.getenv("USE_SUPERVISOR", "1") == "1"
_redis_client = redis.from_url(_app_config.REDIS_URL, decode_responses=True)


# ---------------------------------------------------------------------------
# Supervisor graph (new architecture)
# ---------------------------------------------------------------------------

def _create_supervisor_graph():
    """Create the Supervisor-based graph: input_guardrail → supervisor → output_guardrail."""
    from langgraph_app.orchestrator.supervisor_state import SupervisorState
    from langgraph_app.orchestrator.nodes.guardrails import (
        input_guardrail_node,
        output_guardrail_node,
    )
    from langgraph_app.orchestrator.supervisor import create_supervisor_agent
    from langgraph_app.tools import supervisor_tools

    supervisor_agent = create_supervisor_agent(supervisor_tools)

    # Wrap the supervisor as a node function that invokes the react agent
    async def supervisor_node(state: SupervisorState) -> dict:
        """Run the Supervisor react agent and return its final state."""
        # Build the input for the react agent (it expects a messages-based state)
        agent_input = {
            "messages": state.get("messages", []),
        }
        # Pass context through configurable so tools can access user_id, etc.
        config = {
            "configurable": {
                "user_id": state.get("user_id"),
                "user_context": state.get("user_context", {}),
            }
        }

        # The react agent needs the full state for the prompt builder
        # We pass it via the input since create_react_agent prompt callback receives state
        agent_input["user_profile"] = state.get("user_profile")
        agent_input["user_context"] = state.get("user_context", {})
        response_channel = state.get("response_channel")
        messages = state.get("messages", []) or []
        language = get_dominant_language(messages, default_lang="Chinese")

        streamed_messages = []
        async for chunk in supervisor_agent.astream(
            agent_input,
            config=config,
            stream_mode="updates",
        ):
            if not isinstance(chunk, dict):
                continue
            for inner_node_name, inner_output in chunk.items():
                if not isinstance(inner_output, dict):
                    continue
                messages = inner_output.get("messages", []) or []
                if isinstance(messages, list) and messages:
                    streamed_messages.extend(messages)
                if response_channel:
                    partial_payloads = build_thinking_partials(
                        inner_node_name,
                        inner_output,
                        language=language,
                        context_messages=messages,
                    )
                    for partial_payload in partial_payloads:
                        await _redis_client.publish(
                            response_channel,
                            json.dumps(partial_payload, ensure_ascii=False),
                        )

        return {"messages": streamed_messages}

    def _should_continue(state: SupervisorState) -> Literal["unsafe", "safe"]:
        analysis = state.get("analysis", {})
        if analysis and not analysis.get("safety_safe", True):
            return "unsafe"
        return "safe"

    workflow = StateGraph(SupervisorState)
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("output_guardrail", output_guardrail_node)

    workflow.set_entry_point("input_guardrail")
    workflow.add_conditional_edges(
        "input_guardrail",
        _should_continue,
        {"unsafe": END, "safe": "supervisor"},
    )
    workflow.add_edge("supervisor", "output_guardrail")
    workflow.add_conditional_edges(
        "output_guardrail",
        _should_continue,
        {"unsafe": END, "safe": END},
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Legacy graph (old architecture — kept for rollback)
# ---------------------------------------------------------------------------

def _create_legacy_graph():
    """Create the legacy Router + Workflow DAG."""
    from langgraph_app.orchestrator.state import GraphState
    from langgraph_app.orchestrator.nodes.guardrails import (
        input_guardrail_node,
        output_guardrail_node,
    )
    from langgraph_app.orchestrator.nodes.router import intent_router_node
    from langgraph_app.agents.food_recognition.agent import recognition_node
    from langgraph_app.agents.food_recommendation.agent import food_recommendation_node
    from langgraph_app.agents.chitchat.agent import chitchat_node
    from langgraph_app.agents.goalplanning.agent import goalplanning_node

    def should_continue(state: GraphState) -> Literal["unsafe", "safe"]:
        analysis = state.get("analysis", {})
        if not analysis.get("safety_safe", True):
            return "unsafe"
        return "safe"

    def route_by_intent(
        state: GraphState,
    ) -> Literal["recognition", "recommendation", "chitchat", "goalplanning"]:
        analysis = state.get("analysis", {})
        intent = analysis.get("intent", "chitchat")
        if intent not in ["recognition", "recommendation", "chitchat", "goalplanning"]:
            return "chitchat"
        return intent  # type: ignore

    workflow = StateGraph(GraphState)
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("router", intent_router_node)
    workflow.add_node("recognition", recognition_node)
    workflow.add_node("recommendation", food_recommendation_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("goalplanning", goalplanning_node)
    workflow.add_node("output_guardrail", output_guardrail_node)

    workflow.set_entry_point("input_guardrail")
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
            "goalplanning": "goalplanning",
        },
    )
    workflow.add_edge("recognition", "output_guardrail")
    workflow.add_edge("recommendation", "output_guardrail")
    workflow.add_edge("chitchat", "output_guardrail")
    workflow.add_edge("goalplanning", "output_guardrail")
    workflow.add_conditional_edges(
        "output_guardrail",
        should_continue,
        {"unsafe": END, "safe": END},
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Select and create the active graph
# ---------------------------------------------------------------------------

def create_graph():
    if USE_SUPERVISOR:
        return _create_supervisor_graph()
    return _create_legacy_graph()


graph = create_graph()
