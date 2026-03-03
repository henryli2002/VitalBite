"""Tutorial agent for explaining how to use the app."""

from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.utils import detect_language
from langgraph_app.config import config
from time import sleep


from langgraph_app.utils.utils import detect_language, get_images_from_history

...

def tutorial_node(state: GraphState) -> GraphState:
    """
    Generate a helpful response to user questions about the app.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with tutorial response
    """
    state = state.copy()
    # ensure messages exists
    messages_list: List[Any] = state.setdefault("messages", [])  # type: ignore[assignment]
    client = get_llm_client(module="tutorial")
    input_data = state.get("input", {})
    text = input_data.get("text", "")
    lang = detect_language(text)
    messages = messages_list
    history_text = ""
    if messages:
        history_count = config.get_history_count("tutorial")
        relevant_msgs = messages[-history_count:-1] if len(messages) > 1 else []
        for msg in relevant_msgs:
            role = "User"
            content = str(msg.content)
            if hasattr(msg, "type"):
                if msg.type == "ai":
                    role = "AI"
                elif msg.type == "human":
                    role = "User"
            history_text += f"{role}: {content}\n"

    images_to_process = get_images_from_history(messages)

    image_prompt_part = ""
    if images_to_process:
        image_prompt_part = f"The user has provided {len(images_to_process)} images in a previous message. They may be referring to these images in their current message. The images are provided in order."

    tutorial_prompt = f"""The user needs help with the app. This might be an explicit request for instructions, or it might be inferred because they tried to do something but failed (e.g., asking for image recognition without providing an image).

Conversation History:
{history_text}

Current User input: {text}
{image_prompt_part}

Generate a response based on the following rules:
1.  If the user seems to be stuck or missing information for a task, provide a specific, helpful tip. For example, if they ask for image analysis without an image, tell them they need to upload an image first.
2.  If the user asks a general question about how to use the app, explain the core features (food recognition, restaurant recommendation, goal planning).
3.  If the user asks about a specific feature, provide a clear and concise explanation of how it works.
4.  LANGUAGE: Use the same language as the user input (e.g., English or Chinese).
5.  TONE: Stay helpful, professional, and patient.

Keep the response concise (2-4 sentences)."""
    final_response = ""
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            if not images_to_process:
                final_response = client.generate_text(
                    prompt=tutorial_prompt,
                    system_instruction="You are a helpful assistant explaining how to use the food analysis and recommendation app."
                )
            else:
                final_response = client.generate_vision(
                    images_b64=images_to_process,
                    prompt=tutorial_prompt,
                    system_instruction="You are a helpful assistant explaining how to use the food analysis and recommendation app."
                )
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(f"Tutorial generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                sleep(1)

    if not final_response:
        fallback = "您好！我可以帮您识别食物图片或推荐餐厅。请告诉我您需要什么帮助，或者上传一张食物图片。" if lang == "Chinese" else "Hello! I can help you recognize food images or recommend restaurants. Please tell me what you need help with, or upload a food image."
        if last_error:
            print(f"Tutorial generation ultimately failed after retries: {last_error}")
        final_response = fallback

    state["final_response"] = final_response
    messages_list.append(AIMessage(content=final_response))
    state["input"] = { "text": "", "image_data": None, "source": ""}

    return state
