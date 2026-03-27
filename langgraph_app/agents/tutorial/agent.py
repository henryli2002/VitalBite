"""Tutorial agent for explaining how to use the app."""

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


async def tutorial_node(state: GraphState) -> NodeOutput:
    """
    Generate a helpful response to user questions about the app.
    """
    messages = state.get("messages", [])
    client = get_tracked_llm(module="tutorial", node_name="tutorial")

    lang = get_dominant_language(messages)

    tutorial_prompt = f"""You are a helpful assistant explaining how to use the food analysis and recommendation app. The user needs help with the app. This might be an explicit request for instructions, or it might be inferred because they tried to do something but failed (e.g., asking for image recognition without providing an image).

Generate a response based on the following rules:
1. If the user seems to be stuck or missing information for a task, provide a specific, helpful tip. For example, if they ask for image analysis without an image, tell them they need to upload an image first.
2. If the user asks a general question about how to use the app, explain the core features (food recognition, restaurant recommendation, goal planning).
3. If the user asks about a specific feature, provide a clear and concise explanation of how it works.
4. LANGUAGE: Your entire response MUST be in the specific language requested by the user. (Note: The user's overall conversational language is '{lang}', but if they explicitly asked for a different language for the response, you MUST follow their explicit request!)
5. TONE: Stay helpful, professional, and patient.

Keep the response concise (2-4 sentences)."""

    final_response = ""
    last_error: Exception | None = None
    ai_message = None

    for attempt in range(3):
        try:
            messages_to_send = [SystemMessage(content=tutorial_prompt)] + messages
            ai_message = await client.ainvoke(
                messages_to_send, config={"tags": ["final_node_output"]}
            )
            break
        except Exception as e:
            last_error = e
            logger.warning(f"[tutorial] Generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                await asyncio.sleep(1)

    if not ai_message:
        fallback = (
            "您好！我可以帮您识别食物图片或推荐餐厅。请告诉我您需要什么帮助，或者上传一张食物图片。"
            if lang == "Chinese"
            else "Hello! I can help you recognize food images or recommend restaurants. Please tell me what you need help with, or upload a food image."
        )
        if last_error:
            logger.error(f"[tutorial] Generation ultimately failed after retries: {last_error}")
        ai_message = AIMessage(content=fallback)

    msg: AnyMessage = ai_message  # type: ignore

    return {"messages": [msg], "message_timestamps": [datetime.utcnow().isoformat()]}
