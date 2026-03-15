"""Guardrail nodes for input and output safety checks."""

from typing import Dict, Any
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from pydantic import BaseModel

class SafetyCheck(BaseModel):
    safe: bool
    reason: str | None = None

def _check_safety(text_to_check: str, intent: str) -> Dict[str, Any]:
    """
    Generic function to check text for harmful content.
    
    Args:
        text_to_check: The text to be analyzed.
        intent: The current intent to be preserved in the state.
        
    Returns:
        A dictionary with the safety analysis.
    """
    if not text_to_check:
        # Empty input is considered safe
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "intent": intent
            }
        }

    client = get_llm_client()
    
    # Safety check prompt
    safety_prompt = f"""Analyze the following text for harmful, inappropriate, or unsafe content.

Text: {text_to_check}

Respond with a JSON object containing:
- "safe": boolean indicating if the content is safe
- "reason": string explaining why it's unsafe (null if safe)

Only mark as unsafe if the content contains:
- Hate speech, discrimination, or harassment
- Violence or threats
- Illegal activities
- Explicit sexual content
- Self-harm or suicide references

Be lenient with food-related queries, even if they mention dietary restrictions or health concerns."""

    try:
        result = client.generate_structured(safety_prompt, SafetyCheck)
        
        if not result.safe:
            # Set safety warning response
            safety_message = "抱歉，您的输入包含不当内容，我无法处理此请求。"
            if result.reason:
                safety_message += f" 原因：{result.reason}"
            
            return {
                "analysis": {
                    "safety_safe": False,
                    "safety_reason": result.reason,
                    "intent": intent
                },
                "final_response": safety_message
            }
        
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "intent": intent
            }
        }
    except Exception as e:
        # On error, default to safe but log the issue
        print(f"Guardrail check failed: {e}")
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": f"Safety check error: {str(e)}",
                "intent": intent
            }
        }

def input_guardrail_node(state: GraphState) -> GraphState:
    """
    Check the user's input for harmful content.
    """
    latest_message = state.get("messages", [])[-1]
    text_to_check = latest_message.content if latest_message else ""
    intent = state.get("analysis", {}).get("intent", "")
    return _check_safety(text_to_check, intent)

def output_guardrail_node(state: GraphState) -> GraphState:
    """
    Check the agent's final response for harmful content.
    """
    text_to_check = state.get("final_response", "")
    intent = state.get("analysis", {}).get("intent", "")
    return _check_safety(text_to_check, intent)
