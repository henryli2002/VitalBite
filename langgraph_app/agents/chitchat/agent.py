"""Chit-chat agent for handling general conversation."""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage, SystemMessage, AnyMessage
from langgraph.types import interrupt
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.utils import (
    get_dominant_language,
)
from datetime import datetime
from time import sleep


def chitchat_node(state: GraphState) -> NodeOutput:
    """
    Generate a friendly response for general conversation.
    """
    messages = state.get("messages", [])
    client = get_llm_client(module="chitchat")

    lang = get_dominant_language(messages)

    system_instruction = f"""You are a friendly and helpful food assistant. You are having a general conversation with the user.

Generate a response based on the following rules:
1. If the user's input is unclear, ambiguous, or provides a blurry/non-food image, politely ask for clarification.
2. For general greetings, questions, or off-topic conversation, provide a friendly and helpful response.
3. Keep the response concise and conversational (1-3 sentences).
4. LANGUAGE: Your entire response MUST be in the specific language requested by the user. (Note: The user's overall conversational language is '{lang}', but if they explicitly asked for a different language for the response, you MUST follow their explicit request!)
5. TONE: Stay helpful, professional, and engaging."""
    
    last_error: Exception | None = None
    ai_message = None

    for attempt in range(3):
        try:
            messages_to_send = [SystemMessage(content=system_instruction)] + messages
            ai_message = client.invoke(messages_to_send, config={"tags": ["final_node_output"]})
            break
        except Exception as e:
            last_error = e
            print(f"Chitchat generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                sleep(1)

    if not ai_message:
        fallback = "您好！我可以帮您识别食物图片或推荐餐厅。请告诉我您需要什么帮助，或者上传一张食物图片。" if lang == "Chinese" else "Hello! I can help you recognize food images or recommend restaurants. Please tell me what you need help with, or upload a food image."
        if last_error:
            print(f"Chitchat generation ultimately failed after retries: {last_error}")
        ai_message = AIMessage(content=fallback)

    # Cast to AnyMessage to satisfy typing if needed
    msg: AnyMessage = ai_message # type: ignore

    return {
        "messages": [msg],
        "message_timestamps": [datetime.utcnow().isoformat()]
    }
