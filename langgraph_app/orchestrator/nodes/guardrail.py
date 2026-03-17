"""Guardrail nodes for input and output safety checks."""

from typing import Dict, Any, Literal
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.logger import setup_logger
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

logger = setup_logger(__name__)

class SafetyCheck(BaseModel):
    safe: bool
    reason: str | None = None
    category: str | None = None

IntentType = Literal["recognition", "recommendation", "chitchat", "tutorial", "guardrails", "goalplanning"]

def _check_safety(text_to_check: str, intent: Any, node_name: str) -> NodeOutput:
    """
    Generic function to check text for harmful content.
    
    Args:
        text_to_check: The text to be analyzed.
        intent: The current intent to be preserved in the state.
        node_name: The name of the node for logging.
        
    Returns:
        NodeOutput with the safety analysis.
    """
    if not text_to_check:
        # Empty input is considered safe
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "safety_category": None,
                "intent": intent
            }
        }

    client = get_llm_client(module="guardrail")
    
    # Safety check prompt
    system_prompt = """Analyze the following text for harmful, inappropriate, or unsafe content.

Respond with a JSON object containing:
- "safe": boolean indicating if the content is safe
- "reason": string explaining why it's unsafe (null if safe)
- "category": string indicating the category of unsafe content ("self_harm", "illegal", "violence", "hate", "sexual"), null if safe

Only mark as unsafe if the content contains:
- Hate speech, discrimination, or harassment
- Violence or threats
- Illegal activities
- Explicit sexual content
- Self-harm or suicide references

Be lenient with food-related queries, even if they mention dietary restrictions or health concerns."""

    try:
        result = client.generate_structured(
            messages=[HumanMessage(content=text_to_check)],
            schema=SafetyCheck,
            system_prompt=system_prompt
        )
        
        if not result.safe:
            logger.warning(f"[{node_name}] Safety check failed. Reason: {result.reason}, Category: {result.category}")
            return {
                "analysis": {
                    "safety_safe": False,
                    "safety_reason": result.reason,
                    "safety_category": result.category,
                    "intent": intent
                },
                "debug_logs": [{
                    "node": node_name,
                    "status": "warning",
                    "reason": result.reason
                }]
            }
        
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "safety_category": None,
                "intent": intent
            }
        }
    except Exception as e:
        # On error, default to safe but log the issue
        logger.error(f"[{node_name}] Guardrail check encountered an error: {e}", exc_info=True)
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": f"Safety check error: {str(e)}",
                "safety_category": None,
                "intent": intent
            },
            "debug_logs": [{
                "node": node_name,
                "status": "error",
                "error": str(e)
            }]
        }

def _extract_text(obj: Any) -> str:
    if not obj:
        return ""
    if isinstance(obj, str):
        return obj
    
    content = ""
    if isinstance(obj, dict):
        content = obj.get("content", "")
    elif hasattr(obj, "content"):
        content = obj.content
    else:
        return str(obj)
        
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return " ".join(texts)
    elif isinstance(content, str):
        return content
    return str(content)

def input_guardrail_node(state: GraphState) -> NodeOutput:
    """
    Check the user's input for harmful content.
    """
    messages = state.get("messages", [])
    latest_message = messages[-1] if messages else None
    text_to_check = _extract_text(latest_message)
    intent = state.get("analysis", {}).get("intent", "chitchat")
    return _check_safety(text_to_check, intent, "input_guardrail")

def output_guardrail_node(state: GraphState) -> NodeOutput:
    """
    Check the agent's final response for harmful content.
    """
    text_to_check = _extract_text(state.get("final_response", ""))
    intent = state.get("analysis", {}).get("intent", "chitchat")
    return _check_safety(text_to_check, intent, "output_guardrail")
