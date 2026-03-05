"""Goal planning agent for helping users with their diet and nutrition."""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.utils import (
    detect_language,
    get_current_user_text,
)
from time import sleep


def goalplanning_node(state: GraphState) -> GraphState:
    """
    Generate a helpful response to user questions about goal planning.
    """
    state = state.copy()
    messages = state.setdefault("messages", [])
    client = get_llm_client(module="goalplanning")

    current_text = get_current_user_text(messages)
    lang = detect_language(current_text)

    goalplanning_prompt = f"""You are a nutritional assistant helping users plan their diet and eating goals.

Generate a response based on the following rules:
1. Focus on long-term planning. Help the user define their goals (e.g., weight loss, muscle gain, balanced diet).
2. Incorporate the user's history provided in the conversation context. This includes previous meals, stated preferences, and personal data.
3. Provide actionable suggestions. Instead of just giving information, suggest concrete plans, meal ideas, or next steps.
4. Do not give medical advice. If the user asks for medical advice, gently decline and suggest they consult a doctor.
5. LANGUAGE: Your entire response must be in the same language as the user input ('{lang}').
6. TONE: Stay encouraging, supportive, and informative.

Keep the response concise (2-4 sentences)."""
    
    final_response = ""
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            final_response = client.generate(
                messages=messages,
                system_prompt=goalplanning_prompt
            )
            break
        except Exception as e:
            last_error = e
            print(f"Goal planning generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                sleep(1)

    if not final_response:
        fallback = "您好！我可以帮您规划饮食目标。请告诉我您的需求。" if lang == "Chinese" else "Hello! I can help you plan your dietary goals. Please tell me what you need."
        if last_error:
            print(f"Goal planning generation ultimately failed after retries: {last_error}")
        final_response = fallback

    messages.append(AIMessage(content=final_response))
    state["final_response"] = final_response
    
    return state
