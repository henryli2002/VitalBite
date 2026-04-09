"""Chit-chat agent for handling general conversation."""

from datetime import datetime, timezone

from langchain_core.messages import SystemMessage
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import inject_dynamic_context
from langgraph_app.utils.logger import get_logger
from langgraph_app.utils.utils import get_dominant_language
from langgraph_app.utils.cascade import invoke_with_cascade
from langgraph_app.utils.semaphores import with_semaphore

logger = get_logger(__name__)


@with_semaphore("chitchat")
async def chitchat_node(state: GraphState) -> NodeOutput:
    """Generate a friendly response for general conversation."""
    messages = state.get("messages", [])
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

    messages_to_send = inject_dynamic_context(
        [SystemMessage(content=system_instruction)] + messages
    )

    ai_message = await invoke_with_cascade(
        module="chitchat",
        messages_to_send=messages_to_send,
        lang=lang,
    )
    ai_message.additional_kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
    return {"messages": [ai_message]}
