"""Goal planning agent for helping users with their diet and nutrition."""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage, SystemMessage, AnyMessage
from langgraph.types import interrupt
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.tracked_llm import get_tracked_llm
from langgraph_app.utils.logger import get_logger
from langgraph_app.utils.utils import (
    get_dominant_language,
)
from datetime import datetime
import asyncio

logger = get_logger(__name__)


async def goalplanning_node(state: GraphState) -> NodeOutput:
    """
    Generate a helpful response to user questions about goal planning.
    """
    messages = state.get("messages", [])
    client = get_tracked_llm(module="goalplanning", node_name="goalplanning")

    lang = get_dominant_language(messages)

    goalplanning_prompt = f"""You are a nutritional assistant helping users plan their diet and eating goals.

Generate a response based on the following rules:
1. Focus on long-term planning. Help the user define their goals (e.g., weight loss, muscle gain, balanced diet).
2. Incorporate the user's history provided in the conversation context. This includes previous meals, stated preferences, and personal data.
3. Provide actionable suggestions. Instead of just giving information, suggest concrete plans, meal ideas, or next steps.
4. Do not give medical advice. If the user asks for medical advice, gently decline and suggest they consult a doctor.
5. LANGUAGE: Your entire response MUST be in the specific language requested by the user. (Note: The user's overall conversational language is '{lang}', but if they explicitly asked for a different language for the response, you MUST follow their explicit request!)
6. TONE: Stay encouraging, supportive, and informative.

Keep the response concise (2-4 sentences)."""

    final_response = ""
    last_error: Exception | None = None
    ai_message = None

    for attempt in range(3):
        try:
            messages_to_send = [SystemMessage(content=goalplanning_prompt)] + messages
            ai_message = await client.ainvoke(
                messages_to_send, config={"tags": ["final_node_output"]}
            )
            break
        except Exception as e:
            last_error = e
            logger.warning(f"[goalplanning] Generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                await asyncio.sleep(1)

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
        ai_message = AIMessage(content=fallback)

    msg: AnyMessage = ai_message  # type: ignore

    return {"messages": [msg], "message_timestamps": [datetime.utcnow().isoformat()]}
