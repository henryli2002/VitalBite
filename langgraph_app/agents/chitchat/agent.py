"""Chit-chat agent for handling general conversation."""

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


from langgraph_app.utils.semaphores import with_semaphore

@with_semaphore("chitchat")
async def chitchat_node(state: GraphState) -> NodeOutput:
    """
    Generate a friendly response for general conversation.
    """
    messages = state.get("messages", [])
    client = get_tracked_llm(module="chitchat", node_name="chitchat")

    lang = get_dominant_language(messages)

    user_profile = state.get("user_profile")
    profile_context = ""
    if user_profile:
        profile_context = "\n\nUser Profile & Health Information:\n" + "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_profile.items() if v
        )

    system_instruction = f"""[ROLE]
You are WABI, a friendly and empathetic AI food assistant.

[OBJECTIVE]
Engage in general conversation, building rapport while naturally incorporating the user's personal context.

[CONTEXT]{profile_context}

[CONSTRAINTS]
1. CONCISENESS: Reply in 1-3 conversational sentences.
2. PERSONALIZATION: Actively leverage the 'User Profile' above. If health/diet constraints are present, seamlessly reflect them in your advice or chat. 
3. AMBIGUITY: If input is unclear or an unrelated image is sent, politely ask for clarification.
4. LANGUAGE: Strict adherence to the user's language ('{lang}'), unless explicitly overridden by their latest message."""

    last_error: Exception | None = None
    ai_message = None

    for attempt in range(3):
        try:
            messages_to_send = [SystemMessage(content=system_instruction)] + messages
            ai_message = await client.ainvoke(
                messages_to_send, config={"tags": ["final_node_output"]}
            )
            break
        except Exception as e:
            last_error = e
            logger.warning(f"[chitchat] Generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                await asyncio.sleep(1)

    if not ai_message:
        fallback = (
            "您好！我可以帮您识别食物图片或推荐餐厅。请告诉我您需要什么帮助，或者上传一张食物图片。"
            if lang == "Chinese"
            else "Hello! I can help you recognize food images or recommend restaurants. Please tell me what you need help with, or upload a food image."
        )
        if last_error:
            logger.error(f"[chitchat] Generation ultimately failed after retries: {last_error}")
        ai_message = AIMessage(content=fallback)

    # Cast to AnyMessage to satisfy typing if needed
    msg: AnyMessage = ai_message  # type: ignore

    return {"messages": [msg], "message_timestamps": [datetime.utcnow().isoformat()]}
