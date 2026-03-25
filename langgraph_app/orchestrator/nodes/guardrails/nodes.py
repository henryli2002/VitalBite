"""Public guardrail node functions for LangGraph integration.

This module provides the main entry points for the guardrail system.
"""

from typing import Any

from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.orchestrator.nodes.guardrails.config import (
    SafetyCategory,
    ThreatLevel,
)
from langgraph_app.orchestrator.nodes.guardrails.scorer import get_scorer
from langgraph_app.orchestrator.nodes.guardrails.responses import get_standard_response
from langgraph_app.orchestrator.nodes.guardrails.normalizer import TextNormalizer
from langgraph_app.utils.tracked_llm import get_tracked_llm
from langgraph_app.utils.logger import setup_logger
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


logger = setup_logger(__name__)


class SafetyCheck(BaseModel):
    """Structured output for LLM-based safety check."""

    safe: bool
    reason: str | None = None
    category: str | None = None


def _extract_text(obj: Any) -> str:
    """Extract text content from various message formats."""
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


def _check_safety(
    text_to_check: str,
    intent: Any,
    node_name: str,
    messages: list | None = None,
    use_llm_fallback: bool = True,
) -> NodeOutput:
    """
    Generic function to check text for harmful content and prompt injection.

    Args:
        text_to_check: The text to be analyzed.
        intent: The current intent to be preserved in the state.
        node_name: The name of the node for logging.
        messages: The message history for context.
        use_llm_fallback: Whether to use LLM-based safety check as secondary defense.

    Returns:
        NodeOutput with the safety analysis and potential short-circuit message.
    """
    if not text_to_check:
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "safety_category": None,
                "intent": intent,
            }
        }

    # Stage 1: Fast regex-based detection (defense in depth - layer 1)
    scorer = get_scorer()
    security_score = scorer.score(text_to_check)

    logger.info(
        f"[{node_name}] Regex security check: "
        f"threat_level={security_score.overall_threat_level.name}, "
        f"risk_score={security_score.total_risk_score:.2f}, "
        f"triggered={security_score.triggered_categories}"
    )

    # If high/critical threat detected from regex, short-circuit immediately
    if security_score.overall_threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
        reasoning = f"Regex detected high-risk patterns: {', '.join(security_score.triggered_categories)}"
        logger.warning(f"[{node_name}] {reasoning}")

        matched_details = []
        for result in security_score.detection_results:
            if result.is_triggered:
                matched_details.extend(result.matched_patterns[:3])

        return {
            "analysis": {
                "safety_safe": False,
                "safety_reason": reasoning,
                "safety_category": SafetyCategory.PROMPT_INJECTION.value,
                "intent": "guardrails",
            },
            "messages": [
                AIMessage(
                    content=get_standard_response("prompt_injection", text_to_check)
                )
            ],
            "debug_logs": [
                {
                    "node": node_name,
                    "status": "warning",
                    "reason": "prompt_injection_regex",
                    "threat_level": security_score.overall_threat_level.name,
                    "risk_score": security_score.total_risk_score,
                    "triggered_categories": security_score.triggered_categories,
                    "matched_patterns": matched_details,
                }
            ],
        }

    # Stage 2: LLM-based safety check (defense in depth - layer 2)
    if not use_llm_fallback:
        return {
            "analysis": {
                "safety_safe": True,
                "safety_reason": None,
                "safety_category": None,
                "intent": intent,
            },
            "debug_logs": [
                {
                    "node": node_name,
                    "status": "info",
                    "reason": "regex_passed",
                    "threat_level": security_score.overall_threat_level.name,
                    "risk_score": security_score.total_risk_score,
                }
            ],
        }

    return _llm_safety_check(text_to_check, intent, node_name, security_score)


def _llm_safety_check(
    text_to_check: str,
    intent: Any,
    node_name: str,
    regex_score,
) -> NodeOutput:
    """Secondary LLM-based safety check."""
    client = get_tracked_llm(module="guardrail", node_name="guardrail")

    system_prompt = """Analyze the following text for harmful, inappropriate, or unsafe content.

Respond with a JSON object containing:
- "safe": boolean indicating if the content is safe
- "reason": string explaining why it's unsafe (null if safe)
- "category": string indicating the category of unsafe content ("self_harm", "illegal", "violence", "hate", "sexual", "food_safety_risk"), null if safe

Only mark as unsafe if the content contains:
- Hate speech, discrimination, or harassment ("hate")
- Violence or threats ("violence")
- Illegal activities ("illegal")
- Explicit sexual content ("sexual")
- Self-harm or suicide references ("self_harm")
- Dangerous, toxic, or poisoned food, or any severe food safety risk (e.g. eating poisonous mushrooms, intentionally spoiling food) ("food_safety_risk")

Be lenient with normal food-related queries, even if they mention dietary restrictions or health concerns. Only flag severe food safety risks."""

    last_error = None
    structured_llm = client.with_structured_output(SafetyCheck)

    for attempt in range(3):
        try:
            messages_to_send = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text_to_check),
            ]
            if last_error:
                error_feedback = f"Your previous response failed validation with this error: {str(last_error)}. Please correct your JSON output and ensure it strictly follows the schema."
                messages_to_send.append(SystemMessage(content=error_feedback))

            result = structured_llm.invoke(messages_to_send, config={"callbacks": []})

            if not result.safe:
                logger.warning(
                    f"[{node_name}] LLM safety check failed. "
                    f"Reason: {result.reason}, Category: {result.category}"
                )
                return {
                    "analysis": {
                        "safety_safe": False,
                        "safety_reason": result.reason,
                        "safety_category": result.category,
                        "intent": "guardrails",
                    },
                    "messages": [
                        AIMessage(
                            content=get_standard_response(
                                result.category, text_to_check
                            )
                        )
                    ],
                    "debug_logs": [
                        {
                            "node": node_name,
                            "status": "warning",
                            "reason": result.reason,
                            "category": result.category,
                            "regex_score": regex_score.total_risk_score,
                            "regex_triggered": regex_score.triggered_categories,
                        }
                    ],
                }

            # Safe according to LLM
            return {
                "analysis": {
                    "safety_safe": True,
                    "safety_reason": None,
                    "safety_category": None,
                    "intent": intent,
                },
                "debug_logs": [
                    {
                        "node": node_name,
                        "status": "info",
                        "reason": "llm_passed",
                        "regex_score": regex_score.total_risk_score,
                        "regex_triggered": regex_score.triggered_categories,
                    }
                ],
            }

        except Exception as e:
            last_error = e
            logger.warning(
                f"[{node_name}] Guardrail LLM check failed on attempt {attempt + 1}: {e}"
            )
            if attempt < 2:
                import time

                time.sleep(1)

    # On error, default to safe but log the issue
    logger.error(
        f"[{node_name}] Guardrail check encountered an error after retries: {last_error}",
        exc_info=True,
    )
    return {
        "analysis": {
            "safety_safe": True,
            "safety_reason": f"Safety check error: {str(last_error)}",
            "safety_category": None,
            "intent": intent,
        },
        "debug_logs": [
            {
                "node": node_name,
                "status": "error",
                "error": str(last_error),
                "regex_score": regex_score.total_risk_score,
            }
        ],
    }


def input_guardrail_node(state: GraphState) -> NodeOutput:
    """
    Check the user's input for harmful content and prompt injection.
    Only checks the latest message to avoid false positives from history.
    """
    messages = state.get("messages", [])

    # Get only the text of the LAST message from the user
    latest_message = messages[-1] if messages else None
    text_to_check = _extract_text(latest_message)

    intent = state.get("analysis", {}).get("intent", "chitchat")
    return _check_safety(text_to_check, intent, "input_guardrail", messages)


def output_guardrail_node(state: GraphState) -> NodeOutput:
    """
    Check the agent's final response for harmful content.
    """
    messages = state.get("messages", [])
    latest_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            latest_ai_message = msg
            break

    text_to_check = _extract_text(latest_ai_message) if latest_ai_message else ""
    intent = state.get("analysis", {}).get("intent", "chitchat")
    return _check_safety(text_to_check, intent, "output_guardrail")
