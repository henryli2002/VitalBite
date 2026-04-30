"""Intent routing node — local Qwen3.5-9B zero-shot model via mlx_lm."""

from typing import Any, Literal, Optional
from datetime import datetime, timezone, timedelta
import asyncio
import json
import re
import threading

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment,misc]

from langchain_core.messages import HumanMessage, AIMessage

from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.config import config
from langgraph_app.utils.logger import get_logger
from pydantic import BaseModel

import redis.asyncio as redis

logger = get_logger(__name__)

# ── Redis singleton ───────────────────────────────────────────────────────────

_redis_client: redis.Redis = None  # type: ignore[assignment]


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
    return _redis_client


# ── Local model singleton ─────────────────────────────────────────────────────

MODEL_PATH = "/Users/henryli/.omlx/models/Qwen3.5-9B-OptiQ-4bit"

_model = None
_tokenizer = None
_model_lock = threading.Lock()


def _get_local_model():
    global _model, _tokenizer
    if _model is None:
        with _model_lock:
            if _model is None:
                from mlx_lm import load
                logger.info("[router] Loading local intent model...")
                _model, _tokenizer = load(MODEL_PATH)
                logger.info("[router] Local intent model loaded.")
    return _model, _tokenizer


# ── Data types ────────────────────────────────────────────────────────────────

class IntentAnalysis(BaseModel):
    intent: Literal["recognition", "recommendation", "chitchat", "goalplanning"]
    confidence: float
    reasoning: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_field(text: str, field_name: str) -> str:
    m = re.search(
        rf"{field_name}\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def _parse_intent_output(raw: str) -> Optional[IntentAnalysis]:
    # Strip garbage after first <|im_end|>
    raw = raw.split("<|im_end|>")[0]
    intent = _extract_field(raw, "INTENT").lower()
    conf_str = _extract_field(raw, "CONFIDENCE")
    reasoning = _extract_field(raw, "REASONING")
    allowed = {"recognition", "recommendation", "chitchat", "goalplanning"}
    if intent not in allowed:
        return None
    try:
        confidence = max(0.0, min(1.0, float(conf_str)))
    except Exception:
        return None
    if not reasoning:
        return None
    return IntentAnalysis(intent=intent, confidence=confidence, reasoning=reasoning)


def _lc_messages_to_dicts(messages: list) -> list[dict]:
    """Convert LangChain messages to plain role/content dicts for mlx_lm."""
    result = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            texts, has_image = [], False
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        has_image = True
            text = " ".join(t for t in texts if t)
            if has_image:
                text = f"{text} <attached_image uuid=local/>".strip()
            content = text
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        result.append({"role": role, "content": content})
    return result


def _run_local_inference(system_prompt: str, history: list[dict]) -> str:
    """Synchronous inference — runs in a thread executor."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _get_local_model()

    msgs = [{"role": "system", "content": system_prompt}] + history
    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=128,
        sampler=make_sampler(temp=0.1),
        verbose=False,
    )


# ── Main node ─────────────────────────────────────────────────────────────────

from langgraph_app.utils.semaphores import with_semaphore


@with_semaphore("intent")
async def intent_router_node(state: GraphState) -> NodeOutput:
    """Route user input to the appropriate agent using the local LoRA model."""
    messages = state.get("messages", [])
    response_channel = state.get("response_channel")

    # ── Resolve timezone ──────────────────────────────────────────────────────
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

    _now = datetime.now(user_tz)
    current_hour, current_minute = _now.hour, _now.minute
    t = current_hour + current_minute / 60.0
    if 7 <= t < 9.5:
        meal_time = "breakfast time"
    elif 11.5 <= t < 13.5:
        meal_time = "lunch time"
    elif 17.5 <= t < 19.5:
        meal_time = "dinner time"
    else:
        meal_time = "not meal time"

    # ── Build system prompt ───────────────────────────────────────────────────
    user_profile = state.get("user_profile")
    profile_section = ""
    if user_profile:
        lines = "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in user_profile.items() if v
        )
        profile_section = f"\n\nUser Profile & Health Information:\n{lines}"

    system_prompt = f"""[ROLE]
You are the intent router for WABI, an AI food assistant.

[OBJECTIVE]
Identify the user's primary goal based on the conversation history. Pay close attention to whether a food image is present.

[CONTEXT]
Current Time: {current_hour}:{current_minute:02d} ({meal_time}){profile_section}

[IMAGE FORMAT]
Attached images are represented as server-injected markers at the end of a user message:
  <attached_image uuid=<32-hex-id>/>
If such a marker is present in the current message, treat it as a food image being attached.

[INTENT RULES]
1. "recognition": Goal is to identify food/nutrition from an image. If an <attached_image> marker is present, confidence for this intent should be VERY HIGH (>0.9).
2. "recommendation": Finding places to eat. Triggers on explicit requests or implicit signs of hunger during meal times.
3. "goalplanning": Diet planning, habit building, long-term nutrition goals, or questions about eating history and patterns.
4. "chitchat": Default for everything else — greetings, unrelated topics, vague inputs without context, requests for image recognition WITHOUT an <attached_image> marker present, or meaningless noise.

[CONSTRAINTS]
Output with EXACTLY this plain-text format (no markdown, no code block, no extra labels):
INTENT: <recognition|recommendation|chitchat|goalplanning>
CONFIDENCE: <0.00-1.00>
REASONING: <brief but specific why this intent fits the user message>"""

    history_dicts = _lc_messages_to_dicts(messages)

    # ── Inference with retry ──────────────────────────────────────────────────
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = await asyncio.to_thread(_run_local_inference, system_prompt, history_dicts)
            result = _parse_intent_output(raw)
            if result is None:
                raise ValueError(f"Invalid router output: {raw[:200]}")

            logger.info(f"[router] Intent: {result.intent} (conf={result.confidence:.2f})")

            # Publish final result to Redis if streaming channel present
            if response_channel:
                pub = _get_redis()
                try:
                    await pub.publish(
                        response_channel,
                        json.dumps({
                            "status": "partial",
                            "node": "intent_router",
                            "analysis": {
                                "intent": result.intent,
                                "confidence": result.confidence,
                                "reasoning": result.reasoning,
                                "_stream": False,
                            },
                        }),
                    )
                except Exception as e:
                    logger.warning(f"[router] Redis publish failed: {e}")

            return {
                "analysis": {
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "safety_safe": True,
                    "safety_reason": None,
                },
                "meal_time": meal_time,
                "debug_logs": [{"node": "router", "status": "success", "llm_response": result.model_dump()}],
            }

        except Exception as e:
            last_error = e
            logger.warning(f"[router] Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))

    logger.error(f"[router] Failed after retries: {last_error}", exc_info=True)
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
