"""Global guardrail node for safety checks."""

from typing import Dict, Any
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client


def global_guardrail_node(state: GraphState) -> GraphState:
    """
    Check user input for harmful content using Gemini.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with safety analysis
    """
    client = get_llm_client()
    input_text = state.get("input", {}).get("text", "")
    
    if not input_text:
        # Empty input is considered safe
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "intent": state.get("analysis", {}).get("intent", "")
            }
        }
    
    # Safety check prompt
    safety_prompt = f"""Analyze the following user input for harmful, inappropriate, or unsafe content.

User input: {input_text}

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
        from pydantic import BaseModel
        
        class SafetyCheck(BaseModel):
            safe: bool
            reason: str | None = None
        
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
                    "intent": state.get("analysis", {}).get("intent", "")
                },
                "final_response": safety_message
            }
        
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "intent": state.get("analysis", {}).get("intent", "")
            }
        }
    except Exception as e:
        # On error, default to safe but log the issue
        print(f"Guardrail check failed: {e}")
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": f"Safety check error: {str(e)}",
                "intent": state.get("analysis", {}).get("intent", "")
            }
        }
