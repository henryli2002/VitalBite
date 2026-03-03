"""Guardrails agent for handling malicious or unsafe inputs."""

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

def guardrails_node(state: GraphState) -> GraphState:
    """
    Generate a safe response to malicious or inappropriate user input.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with a safe response
    """
    state = state.copy()
    # ensure messages exists
    messages_list: List[Any] = state.setdefault("messages", [])  # type: ignore[assignment]
    client = get_llm_client(module="guardrails")
    input_data = state.get("input", {})
    text = input_data.get("text", "")
    lang = detect_language(text)
    messages = messages_list
    history_text = ""
    if messages:
        history_count = config.get_history_count("guardrails")
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

    guardrails_prompt = f"""The user's input has been flagged as potentially malicious, unsafe, or inappropriate.

Conversation History:
{history_text}

Current User input: {text}
{image_prompt_part}

Generate a response based on the following rules:
1.  Do not repeat or engage with the malicious content.
2.  Politely refuse to fulfill the request, stating that it violates safety policies.
3.  Redirect the conversation back to the app's core features (food recognition, etc.).
4.  Do not be preachy or judgmental.
5.  LANGUAGE: Use the same language as the user input (e.g., English or Chinese).
6.  TONE: Stay firm, professional, and safe.

Keep the response concise (1-2 sentences)."""
    final_response = ""
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            if not images_to_process:
                final_response = client.generate_text(
                    prompt=guardrails_prompt,
                    system_instruction="You are a safety-conscious assistant. Your role is to handle inappropriate or malicious user requests safely and professionally."
                )
            else:
                final_response = client.generate_vision(
                    images_b64=images_to_process,
                    prompt=guardrails_prompt,
                    system_instruction="You are a safety-conscious assistant. Your role is to handle inappropriate or malicious user requests safely and professionally."
                )
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(f"Guardrails generation failed on attempt {attempt + 1}: {e}")
            if attempt < 2:
                sleep(1)

    if not final_response:
        fallback = "抱歉，我无法处理此请求。我可以帮您识别食物或推荐餐厅。" if lang == "Chinese" else "I'm sorry, I cannot process this request. I can help you identify food or recommend restaurants."
        if last_error:
            print(f"Guardrails generation ultimately failed after retries: {last_error}")
        final_response = fallback

    state["final_response"] = final_response
    messages_list.append(AIMessage(content=final_response))
    state["input"] = { "text": "", "image_data": None, "source": ""}

    return state
