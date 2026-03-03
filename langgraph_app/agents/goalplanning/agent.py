"""Goal planning agent for helping users with their diet and nutrition."""

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

def goalplanning_node(state: GraphState) -> GraphState:
    """
    Generate a helpful response to user questions about goal planning.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with goal planning response
    """
    state = state.copy()
    # ensure messages exists
    messages_list: List[Any] = state.setdefault("messages", [])  # type: ignore[assignment]
    client = get_llm_client(module="goalplanning")
    input_data = state.get("input", {})
    text = input_data.get("text", "")
    lang = detect_language(text)
    messages = messages_list
    history_text = ""
    if messages:
        history_count = config.get_history_count("goalplanning")
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
        image_prompt_part = f"The user has provided {len(images_to_process)} images in a previous message. These images may represent past meals that should be considered as part of their eating history."

    goalplanning_prompt = f"""The user wants to create a diet plan or set nutritional goals. Your role is to be a supportive nutritional assistant.

Conversation History:
{history_text}

Current User input: {text}
{image_prompt_part}

Generate a response based on the following rules:
1.  Focus on long-term planning. Help the user define their goals (e.g., weight loss, muscle gain, balanced diet).
2.  Incorporate the user's history. This includes previous meals (which may be in the images), stated preferences, and personal data like weight or activity level if provided.
3.  Provide actionable suggestions. Instead of just giving information, suggest concrete plans, meal ideas, or next steps.
4.  Do not give medical advice. If the user asks for medical advice, gently decline and suggest they consult a doctor.
5.  LANGUAGE: Use the same language as the user input (e.g., English or Chinese).
6.  TONE: Stay encouraging, supportive, and informative.

Keep the response concise (2-4 sentences)."""
    final_response = ""
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            if not images_to_process:
                final_response = client.generate_text(
                    prompt=goalplanning_prompt,
                    system_instruction="You are a nutritional assistant helping users plan their diet and eating goals. Do not provide medical advice."
                )
            else:
                final_response = client.generate_vision(
                    images_b64=images_to_process,
                    prompt=goalplanning_prompt,
                    system_instruction="You are a nutritional assistant helping users plan their diet and eating goals. Do not provide medical advice."
                )
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(f"Goal planning generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                sleep(1)

    if not final_response:
        fallback = "您好！我可以帮您规划饮食目标。请告诉我您的需求。" if lang == "Chinese" else "Hello! I can help you plan your dietary goals. Please tell me what you need."
        if last_error:
            print(f"Goal planning generation ultimately failed after retries: {last_error}")
        final_response = fallback

    state["final_response"] = final_response
    messages_list.append(AIMessage(content=final_response))
    state["input"] = { "text": "", "image_data": None, "source": ""}

    return state
