"""Supervisor Agent — the central brain of WABI.

Replaces the static Router + Workflow with a reactive agent that autonomously
decides which tools to call based on user intent.  Uses LangGraph's
``create_react_agent`` for the LLM ↔ Tool loop.
"""

import logging
from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import SystemMessage

from langgraph_app.utils.agent_utils import (
    build_profile_context,
    build_daily_cal_ref,
    detect_meal_time,
)
from langgraph_app.utils.utils import get_dominant_language
from langgraph_app.utils.llm_factory import get_llm_client

logger = logging.getLogger("wabi.supervisor")


def _build_system_prompt(state: Dict[str, Any]) -> list:
    """Build the Supervisor's system prompt from the current graph state.

    Returns a list of messages (just the SystemMessage) to prepend.
    """
    user_profile = state.get("user_profile") or {}
    user_context = state.get("user_context") or {}
    messages = state.get("messages", [])

    profile_context = build_profile_context(user_profile)
    behavioral_notes = user_profile.get("behavioral_notes") or "N/A"
    daily_cal_ref = build_daily_cal_ref(user_profile)
    meal_time = detect_meal_time(user_context.get("timezone"))
    lang = get_dominant_language(messages)

    meal_context = (
        f"Current meal period: {meal_time}."
        if meal_time != "not meal time"
        else "Currently outside main meal hours (likely a snack)."
    )

    system_text = f"""[ROLE]
You are WABI, a friendly and expert AI health & nutrition assistant.

[USER CONTEXT]{profile_context}
- Long-term behavioral traits: {behavioral_notes}
- Daily calorie reference: {daily_cal_ref}
- {meal_context}

[TOOL USAGE RULES]
1. Image handling: user messages may contain placeholders of the form `[图片: <32-hex-id>]` or `[图片: <32-hex-id> | description]`, where `<32-hex-id>` is a 32-character hexadecimal string.
   - If the placeholder already has a description AND the user is only asking about it conversationally, you may answer directly from the description without re-running the tool.
   - Otherwise, you MUST call the `analyze_food_image` tool. Its only argument is named `image_uuid` and takes the 32-hex id exactly as written in the placeholder (no brackets, no prefix). Example: if the message contains `[图片: 7b0ed022bf0d4a96815cc1c5a440e9c4]`, call `analyze_food_image(image_uuid="7b0ed022bf0d4a96815cc1c5a440e9c4")`.
   - After the tool returns, generate a clear summary using a Markdown table. Do NOT dump raw tool JSON to the user.
2. When the user asks for restaurant recommendations, call `search_restaurants` with appropriate parameters. Then present the results in a friendly, formatted way.
3. For general conversation, goal planning, or diet advice, respond directly WITHOUT calling any tools. Use the user profile and behavioral traits to personalize your response.
4. For compound requests (e.g., "analyze this food AND recommend similar restaurants"), chain multiple tool calls sequentially.
5. After analyzing food, evaluate whether the meal fits the user's goals and daily calorie budget. Reference the meal period context.

[RESPONSE FORMAT]
- When presenting nutrition data, ALWAYS use a Markdown table:
  | Item | Weight | Calories | Fat | Carbs | Protein |
  | :--- | :--- | :--- | :--- | :--- | :--- |
- Keep responses concise but informative (2-5 sentences for chitchat, structured tables for nutrition).
- Be warm and encouraging about healthy choices; gently flag concerns without being preachy.

[LANGUAGE]
Respond in the same language the user is using (detected: '{lang}'). Follow any explicit language preference from the user.

[SAFETY]
- Never provide medical advice; refer to a doctor for clinical issues.
- If health conditions or allergies are in the user profile, actively flag conflicts."""

    return [SystemMessage(content=system_text)] + list(messages)


def create_supervisor_agent(tools: Sequence):
    """Create and return a compiled Supervisor agent graph.

    Uses LangGraph's ``create_react_agent`` which handles the LLM ↔ Tool
    react loop internally.  The caller is responsible for wrapping this
    with input/output guardrails.

    The model is resolved lazily via a factory callable so the graph can be
    compiled at import time without requiring API keys to be present.
    """
    from langgraph.prebuilt import create_react_agent  # noqa: LangGraphDeprecatedSinceV10

    def _model_factory(state, runtime):
        return get_llm_client(module="supervisor")

    agent = create_react_agent(
        model=_model_factory,
        tools=list(tools),
        prompt=_build_system_prompt,
        name="supervisor",
    )

    return agent
