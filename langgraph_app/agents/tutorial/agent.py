"""Tutorial agent for explaining how to use the app."""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.utils import (
    get_dominant_language,
)
from datetime import datetime
from time import sleep


def tutorial_node(state: GraphState) -> GraphState:
    """
    Generate a helpful response to user questions about the app.
    """
    state = state.copy()
    messages = state.setdefault("messages", [])
    client = get_llm_client(module="tutorial")

    lang = get_dominant_language(messages)

    tutorial_prompt = f"""You are a helpful assistant explaining how to use the food analysis and recommendation app. The user needs help with the app. This might be an explicit request for instructions, or it might be inferred because they tried to do something but failed (e.g., asking for image recognition without providing an image).

Generate a response based on the following rules:
1. If the user seems to be stuck or missing information for a task, provide a specific, helpful tip. For example, if they ask for image analysis without an image, tell them they need to upload an image first.
2. If the user asks a general question about how to use the app, explain the core features (food recognition, restaurant recommendation, goal planning).
3. If the user asks about a specific feature, provide a clear and concise explanation of how it works.
4. LANGUAGE: Your entire response should be in the same language as the user's dominant language in the conversation ('{lang}'). However, if the user specifically asks for another language, please switch to that language.
5. TONE: Stay helpful, professional, and patient.

Keep the response concise (2-4 sentences)."""
    
    final_response = ""
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            final_response = client.generate(
                messages=messages,
                system_prompt=tutorial_prompt
            )
            break
        except Exception as e:
            last_error = e
            print(f"Tutorial generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                sleep(1)

    if not final_response:
        fallback = "您好！我可以帮您识别食物图片或推荐餐厅。请告诉我您需要什么帮助，或者上传一张食物图片。" if lang == "Chinese" else "Hello! I can help you recognize food images or recommend restaurants. Please tell me what you need help with, or upload a food image."
        if last_error:
            print(f"Tutorial generation ultimately failed after retries: {last_error}")
        final_response = fallback

    messages.append(AIMessage(content=final_response))
    state.setdefault("message_timestamps", []).append(datetime.utcnow().isoformat())
    state["final_response"] = final_response
    
    return state
