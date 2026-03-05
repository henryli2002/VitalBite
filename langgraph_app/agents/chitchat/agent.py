"""Chit-chat agent for handling general conversation."""

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


def chitchat_node(state: GraphState) -> GraphState:
    """
    Generate a friendly response for general conversation.
    """
    state = state.copy()
    messages = state.setdefault("messages", [])
    client = get_llm_client(module="chitchat")

    current_text = get_current_user_text(messages)
    lang = detect_language(current_text)

    system_instruction = f"""You are a friendly and helpful food assistant. You are having a general conversation with the user.

Generate a response based on the following rules:
1. If the user's input is unclear, ambiguous, or provides a blurry/non-food image, politely ask for clarification.
2. For general greetings, questions, or off-topic conversation, provide a friendly and helpful response.
3. Keep the response concise and conversational (1-3 sentences).
4. LANGUAGE: Your entire response must be in the same language as the user input ('{lang}').
5. TONE: Stay helpful, professional, and engaging."""
    
    final_response = ""
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            final_response = client.generate(
                messages=messages,
                system_prompt=system_instruction
            )
            break
        except Exception as e:
            last_error = e
            print(f"Chitchat generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                sleep(1)

    if not final_response:
        fallback = "您好！我可以帮您识别食物图片或推荐餐厅。请告诉我您需要什么帮助，或者上传一张食物图片。" if lang == "Chinese" else "Hello! I can help you recognize food images or recommend restaurants. Please tell me what you need help with, or upload a food image."
        if last_error:
            print(f"Chitchat generation ultimately failed after retries: {last_error}")
        final_response = fallback

    messages.append(AIMessage(content=final_response))
    state["final_response"] = final_response
    
    return state
