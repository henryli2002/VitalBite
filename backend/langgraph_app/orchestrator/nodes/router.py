"""Intent routing node."""

from typing import Dict, Any, Literal, Optional
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.llm_callback import create_callback_handler
from langgraph_app.config import config
from langgraph_app.utils.logger import get_logger
from pydantic import BaseModel
from langchain_core.messages import SystemMessage

import re
import asyncio
import json
import os
import redis.asyncio as redis

# Module-level Redis singleton — reused across all requests
_redis_client: redis.Redis = None  # type: ignore[assignment]

def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        url = os.environ.get("WABI_REDIS_URL", "redis://redis:6379/0")
        _redis_client = redis.from_url(url, decode_responses=True)
    return _redis_client

logger = get_logger(__name__)


class IntentAnalysis(BaseModel):
    """Structured output for intent routing."""

    intent: Literal[
        "recognition", "recommendation", "chitchat", "goalplanning"
    ]
    confidence: float
    reasoning: str


from langgraph_app.utils.semaphores import with_semaphore


def _extract_text_from_chunk_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    out.append(text)
        return "".join(out)
    return ""


def _extract_field(text: str, field_name: str) -> str:
    pattern = rf"{field_name}\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|$)"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_intent_output(raw_text: str) -> Optional[IntentAnalysis]:
    intent = _extract_field(raw_text, "INTENT").lower()
    confidence_str = _extract_field(raw_text, "CONFIDENCE")
    reasoning = _extract_field(raw_text, "REASONING")
    allowed = {"recognition", "recommendation", "chitchat", "goalplanning"}
    if intent not in allowed:
        return None
    try:
        confidence = float(confidence_str)
    except Exception:
        return None
    confidence = max(0.0, min(1.0, confidence))
    if not reasoning:
        return None
    return IntentAnalysis(intent=intent, confidence=confidence, reasoning=reasoning)


@with_semaphore("intent")
async def intent_router_node(state: GraphState) -> NodeOutput:
    """
    Route user input to appropriate agent based on intent.
    This router focuses only on the high-level user goal.
    """
    client = get_llm_client(module="router")
    messages = state.get("messages", [])
    response_channel = state.get("response_channel")

    from datetime import datetime, timezone, timedelta
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None

    # Resolve user timezone: frontend-provided first, then IP fallback, then UTC+8
    user_context = state.get("user_context") or {}
    tz_name = user_context.get("timezone")
    if not tz_name:
        user_ip = user_context.get("user_ip")
        if user_ip:
            from langgraph_app.tools.map.ip_location import get_timezone_from_ip_async
            try:
                tz_name = await get_timezone_from_ip_async(user_ip)
            except Exception:
                pass

    try:
        user_tz = ZoneInfo(tz_name) if (ZoneInfo and tz_name) else timezone(timedelta(hours=8))
    except Exception:
        user_tz = timezone(timedelta(hours=8))

    _now_local = datetime.now(user_tz)
    current_hour = _now_local.hour
    current_minute = _now_local.minute
    current_time = current_hour + current_minute / 60.0
    if 7 <= current_time < 9.5:
        meal_time = "breakfast time"
    elif 11.5 <= current_time < 13.5:
        meal_time = "lunch time"
    elif 17.5 <= current_time < 19.5:
        meal_time = "dinner time"
    else:
        meal_time = "not meal time"

    # Build profile context for routing awareness
    user_profile = state.get("user_profile")
    profile_context = ""
    if user_profile:
        profile_context = "\n\nUser Profile & Health Information:\n" + "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in user_profile.items()
            if v
        )

    system_prompt = f"""[ROLE]
You are the intent router for WABI, an AI food assistant.

[OBJECTIVE]
Identify the user's primary goal based on the conversation history. Pay close attention to whether a food image is present.

[CONTEXT]
Current Time: {current_hour}:{current_minute:02d} ({meal_time}){profile_context}

[INTENT RULES]
1. "recognition": Goal is to identify food/nutrition from an image. If a food image is present, confidence for this intent should be VERY HIGH (>0.9).
2. "recommendation": Finding places to eat. Triggers on explicit requests or implicit signs of hunger during meal times.
3. "goalplanning": Diet planning, habit building, long-term nutrition goals, or questions about eating history and patterns.
4. "chitchat": Default for everything else — greetings, unrelated topics, non-food/blurry images, vague inputs without context, requests for image recognition without an attached image, or meaningless noise.

[CONSTRAINTS]
Output with EXACTLY this plain-text format (no markdown, no code block, no extra labels):
INTENT: <recognition|recommendation|chitchat|goalplanning>
CONFIDENCE: <0.00-1.00>
REASONING: <brief but specific why this intent fits the user message>"""

    last_error: Exception | None = None
    sleep_times = [0.2, 0.5]
    for attempt in range(3):
        try:
            messages_to_send = [SystemMessage(content=system_prompt)] + messages

            if last_error:
                error_feedback = f"Your previous response failed validation: {str(last_error)}. Return strictly in INTENT/CONFIDENCE/REASONING format."
                messages_to_send.append(SystemMessage(content=error_feedback))

            streamed_text = ""
            last_published_reasoning = ""
            pub_client = _get_redis() if response_channel else None

            async for chunk in client.astream(
                messages_to_send,
                config={"callbacks": [create_callback_handler("intent_router")]},
            ):
                piece = _extract_text_from_chunk_content(getattr(chunk, "content", ""))
                if not piece:
                    continue
                streamed_text += piece

                if pub_client and response_channel:
                    intent_so_far = _extract_field(streamed_text, "INTENT").lower()
                    confidence_so_far = _extract_field(streamed_text, "CONFIDENCE")
                    reasoning_so_far = _extract_field(streamed_text, "REASONING")
                    if (
                        reasoning_so_far
                        and reasoning_so_far != last_published_reasoning
                    ):
                        if len(reasoning_so_far) - len(
                            last_published_reasoning
                        ) < 12 and not reasoning_so_far.endswith(
                            (".", "!", "?", "。", "！", "？", ";", "；")
                        ):
                            continue
                        try:
                            confidence_value = (
                                float(confidence_so_far) if confidence_so_far else None
                            )
                        except Exception:
                            confidence_value = None
                        partial_payload = {
                            "status": "partial",
                            "node": "intent_router",
                            "analysis": {
                                "intent": intent_so_far or "chitchat",
                                "confidence": confidence_value,
                                "reasoning": reasoning_so_far,
                                "_stream": True,
                            },
                        }
                        try:
                            await pub_client.publish(
                                response_channel, json.dumps(partial_payload)
                            )
                            last_published_reasoning = reasoning_so_far
                        except Exception as publish_err:
                            logger.warning(
                                f"[router] Streaming publish failed, continue without partial stream: {publish_err}"
                            )
                            pub_client = None

            result = _parse_intent_output(streamed_text)
            if result is None:
                raise ValueError(f"Invalid router output format: {streamed_text[:300]}")

            logger.info(
                f"[router] Intent detected: {result.intent} (confidence: {result.confidence})"
            )

            return {
                "analysis": {
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "safety_safe": True,
                    "safety_reason": None,
                },
                "meal_time": meal_time,
                "debug_logs": [
                    {
                        "node": "router",
                        "status": "success",
                        "llm_response": result.model_dump()
                        if hasattr(result, "model_dump")
                        else {},
                    }
                ],
            }
        except Exception as e:
            last_error = e
            logger.warning(
                f"[router] Intent routing failed on attempt {attempt + 1}: {e}"
            )
            if attempt < 2:
                await asyncio.sleep(sleep_times[attempt])

    logger.error(
        f"[router] Intent routing ultimately failed after retries: {last_error}",
        exc_info=True,
    )
    return {
        "analysis": {
            "intent": "chitchat",
            "confidence": 0.0,
            "reasoning": "Intent router failed after retries, fallback to chitchat.",
            "safety_safe": True,
            "safety_reason": None,
        },
        "meal_time": meal_time,
        "debug_logs": [{"node": "router", "status": "error", "error": str(last_error)}],
    }
