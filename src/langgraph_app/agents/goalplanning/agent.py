"""Goal planning agent for helping users with their diet and nutrition."""

from datetime import datetime, timezone

from langchain_core.messages import SystemMessage
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import inject_dynamic_context
from langgraph_app.utils.logger import get_logger
from langgraph_app.utils.utils import get_dominant_language
from langgraph_app.utils.cascade import invoke_with_cascade
from langgraph_app.utils.semaphores import with_semaphore

logger = get_logger(__name__)


@with_semaphore("goalplanning")
async def goalplanning_node(state: GraphState) -> NodeOutput:
    """Generate a helpful response to user questions about goal planning."""
    messages = state.get("messages", [])
    lang = get_dominant_language(messages)

    user_profile = state.get("user_profile")
    profile_context = ""
    if user_profile:
        profile_context = "\n\nUser Profile & Health Information:\n" + "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_profile.items() if v
        )

    goalplanning_prompt = f"""[ROLE]
You are WABI, an expert nutritional planner and long-term health coach.

[OBJECTIVE]
Help the user define and achieve actionable dietary goals based on their history and profile.

[CONTEXT]{profile_context}

[CONSTRAINTS]
1. ACTIONABILITY: Provide concrete plans or next steps, not just general facts. Recommend specific foods or habit changes.
2. MEDICAL BOUNDARY: Never give medical advice; refer to a doctor for clinical issues.
3. PERSONALIZATION: You MUST strictly adhere to the 'User Profile' constraints (health conditions, allergies). Acknowledge these constraints in your recommendations.
4. CONCISENESS: Keep responses impactful and focused (2-4 sentences).
5. LANGUAGE: Strict adherence to the user's language ('{lang}'), unless explicitly overridden by their latest message."""

    messages_to_send = inject_dynamic_context(
        [SystemMessage(content=goalplanning_prompt)] + messages
    )

    ai_message = await invoke_with_cascade(
        module="goalplanning",
        messages_to_send=messages_to_send,
        lang=lang,
    )
    ai_message.additional_kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
    return {"messages": [ai_message]}
