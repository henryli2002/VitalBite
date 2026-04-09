"""Goal planning agent for helping users with their diet and nutrition."""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage, SystemMessage
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.llm_callback import create_callback_handler
from langgraph_app.utils.llm_factory import inject_dynamic_context
from langgraph_app.utils.logger import get_logger
from langgraph_app.utils.utils import (
    get_dominant_language,
)
from datetime import datetime, timezone
import asyncio

logger = get_logger(__name__)


from langgraph_app.utils.semaphores import with_semaphore

@with_semaphore("goalplanning")
async def goalplanning_node(state: GraphState) -> NodeOutput:
    """
    Generate a helpful response to user questions about goal planning.
    """
    messages = state.get("messages", [])
    client = get_llm_client(module="goalplanning")

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

    last_error: Exception | None = None
    ai_message = None
    sleep_times = [0.2, 0.5]

    for attempt in range(3):
        try:
            messages_to_send = [SystemMessage(content=goalplanning_prompt)] + messages
            messages_to_send = inject_dynamic_context(messages_to_send)
            ai_message = await client.ainvoke(
                messages_to_send,
                config={"callbacks": [create_callback_handler("goalplanning")], "tags": ["final_node_output"]},
            )
            break
        except Exception as e:
            last_error = e
            logger.warning(f"[goalplanning] Generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                await asyncio.sleep(sleep_times[attempt])

    ts = datetime.now(timezone.utc).isoformat()
    if not ai_message:
        fallback = (
            "您好！我可以帮您规划饮食目标。请告诉我您的需求。"
            if lang == "Chinese"
            else "Hello! I can help you plan your dietary goals. Please tell me what you need."
        )
        if last_error:
            logger.error(
                f"[goalplanning] Generation ultimately failed after retries: {last_error}"
            )
        ai_message = AIMessage(content=fallback, additional_kwargs={"timestamp": ts})
    else:
        ai_message.additional_kwargs["timestamp"] = ts

    return {"messages": [ai_message]}
