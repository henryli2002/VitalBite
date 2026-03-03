from langchain_core.messages import HumanMessage
from langgraph_app.orchestrator.state import GraphState

def normalize_input_node(state: GraphState) -> GraphState:
    state = state.copy()
    messages = state.get("messages", [])

    text = ""
    image_data = None

    if not messages:
        return state

    last_msg = messages[-1]

    if not isinstance(last_msg, HumanMessage):
        return state

    content = last_msg.content

    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_blocks = []
        image_blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "text":
                text_blocks.append(block.get("text", ""))
            elif block.get("type") == "image_url":
                # Assumes base64 format like "data:image/jpeg;base64,{b64_string}"
                image_url = block.get("image_url", {}).get("url", "")
                if "base64," in image_url:
                    image_blocks.append(image_url.split("base64,")[1])

        text = " ".join(text_blocks).strip()
        if image_blocks:
            image_data = image_blocks

    state["input"] = {
        "text": text,
        "image_data": image_data,
        "source": "current_user_turn"
    }

    return state