"""Intent routing node."""

from typing import Dict, Any, Literal
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.tracked_llm import get_tracked_llm
from langgraph_app.config import config
from langgraph_app.utils.logger import get_logger
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph_app.utils.utils import (
    get_all_user_text,
)

import time
import re
from typing import Tuple
import asyncio

logger = get_logger(__name__)


class IntentAnalysis(BaseModel):
    """Structured output for intent routing."""

    intent: Literal[
        "recognition", "recommendation", "chitchat", "tutorial", "goalplanning"
    ]
    confidence: float
    reasoning: str


async def intent_router_node(state: GraphState) -> NodeOutput:
    """
    Route user input to appropriate agent based on intent.
    This router focuses only on the high-level user goal.
    """
    client = get_tracked_llm(module="router", node_name="intent_router")
    messages = state.get("messages", [])

    current_hour = time.localtime().tm_hour
    current_minute = time.localtime().tm_min
    current_time = current_hour + current_minute / 60.0
    if 7 <= current_time < 9.5:
        meal_time = "breakfast time"
    elif 11.5 <= current_time < 13.5:
        meal_time = "lunch time"
    elif 17.5 <= current_time < 19.5:
        meal_time = "dinner time"
    else:
        meal_time = "not meal time"

    # Build profile context for routing awareness
    user_profile = state.get("user_profile")
    profile_context = ""
    if user_profile:
        profile_context = "\n\nUser Profile & Health Information:\n" + "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_profile.items() if v
        )

    system_prompt = f"""[ROLE]
You are the intent router for WABI, an AI food assistant.

[OBJECTIVE]
Identify the user's primary goal based on the conversation history.

[CONTEXT]
Current Time: {current_hour}:{current_minute:02d} ({meal_time}){profile_context}

[INTENT RULES]
1. "recognition": Goal is to identify food/nutrition. STRICT REQ: There MUST be a valid food image.
2. "recommendation": Finding places to eat. Triggers on explicit requests or implicit signs of hunger during meal times.
3. "goalplanning": Diet planning, habit building, and long-term nutrition goals.
4. "tutorial": User needs help, asks for image recognition without an image, provides vague inputs ("I want this" without context), or inputs meaningless noise.
5. "chitchat": Default. Greetings, unrelated topics, or non-food/blurry images.

[CONSTRAINTS]
Output exactly matching the requested JSON schema (intent, confidence, reasoning)."""

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            structured_llm = client.with_structured_output(IntentAnalysis)
            messages_to_send = [SystemMessage(content=system_prompt)] + messages

            if last_error:
                error_feedback = f"Your previous response failed validation with this error: {str(last_error)}. Please correct your JSON output and ensure it strictly follows the schema."
                messages_to_send.append(SystemMessage(content=error_feedback))

            result = await structured_llm.ainvoke(messages_to_send, config={"callbacks": []})

            logger.info(
                f"[router] Intent detected: {result.intent} (confidence: {result.confidence})"
            )

            return {
                "analysis": {
                    "intent": result.intent,
                    "safety_safe": True,
                    "safety_reason": None,
                },
                "debug_logs": [
                    {
                        "node": "router",
                        "status": "success",
                        "llm_response": result.model_dump()
                        if hasattr(result, "model_dump")
                        else {},
                    }
                ],
            }
        except Exception as e:
            last_error = e
            logger.warning(
                f"[router] Intent routing failed on attempt {attempt + 1}: {e}"
            )
            if attempt < 2:
                await asyncio.sleep(1)

    logger.error(
        f"[router] Intent routing ultimately failed after retries: {last_error}",
        exc_info=True,
    )
    return {
        "analysis": {
            "intent": "chitchat",
            "safety_safe": True,
            "safety_reason": None,
        },
        "debug_logs": [{"node": "router", "status": "error", "error": str(last_error)}],
    }
