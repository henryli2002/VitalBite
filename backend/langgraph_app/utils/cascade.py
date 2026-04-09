"""Two-tier LLM cascade with timeout.

Tier 1 — Primary model (whatever LLM_PROVIDER is configured).
Tier 2 — llamacpp local model with truncated, image-stripped history.
Tier 3 — Static user-facing error message in the correct language.

The cascade fires on:
  - All retry attempts exhausted (API error, rate-limit, etc.)
  - Primary call wall-clock time exceeds timeout_s (slow response)

Usage::

    from langgraph_app.utils.cascade import invoke_with_cascade

    ai_msg = await invoke_with_cascade(
        module="chitchat",
        messages_to_send=messages,
        lang=lang,
    )
"""

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage

from langgraph_app.config import config
from langgraph_app.utils.retry import with_retry
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.llm_callback import create_callback_handler

logger = logging.getLogger("wabi.cascade")

_ERROR_MESSAGES = {
    "Chinese": "当前访问人数较多，请稍候再试。",
    "English": "Service is temporarily busy. Please try again in a moment.",
}


def _error_message(lang: str) -> str:
    return _ERROR_MESSAGES.get(lang, _ERROR_MESSAGES["English"])


def _truncate_for_fallback(messages: list[BaseMessage], max_turns: int) -> list[BaseMessage]:
    """Keep all SystemMessages + last *max_turns* human/AI pairs."""
    system = [m for m in messages if isinstance(m, SystemMessage)]
    others = [m for m in messages if not isinstance(m, SystemMessage)]
    return system + others[-(max_turns * 2):]


def _strip_images(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Replace multimodal content with text-only for models that don't support vision."""
    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg.content, list):
            texts = []
            for part in msg.content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part["text"])
            if not texts:
                continue  # drop purely-image messages
            result.append(msg.__class__(content=" ".join(texts)))
        else:
            result.append(msg)
    return result


async def invoke_with_cascade(
    module: str,
    messages_to_send: list[BaseMessage],
    lang: str,
    *,
    timeout_s: float | None = None,
    invoke_tags: list[str] | None = None,
) -> AIMessage:
    """Invoke the primary LLM; cascade to llamacpp then error on failure.

    Args:
        module:           LLM module name (e.g. "chitchat", "goalplanning").
        messages_to_send: Full message list including system prompt.
        lang:             Detected user language for the final error fallback.
        timeout_s:        Wall-clock timeout for the primary call (incl. retries).
                          Defaults to config.PRIMARY_LLM_TIMEOUT_S.
        invoke_tags:      LangChain callback tags to attach.
    """
    if timeout_s is None:
        timeout_s = config.PRIMARY_LLM_TIMEOUT_S

    invoke_cfg: dict[str, Any] = {
        "callbacks": [create_callback_handler(module)],
        "tags": invoke_tags or ["final_node_output"],
    }

    primary = get_llm_client(module=module)
    primary_provider = config.get_provider_for_module(module)

    # ── Tier 1: primary model ──────────────────────────────────────────────
    result: AIMessage | None = None
    try:
        result = await asyncio.wait_for(
            with_retry(
                lambda: primary.ainvoke(messages_to_send, config=invoke_cfg),
                attempts=3,
                base=0.3,
                cap=5.0,
                fallback=None,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("[%s] Primary LLM timed out after %.1fs, cascading", module, timeout_s)
    except Exception as exc:
        logger.warning("[%s] Primary LLM failed: %s, cascading", module, exc)

    if result is not None:
        return result

    # ── Tier 2: llamacpp fallback ──────────────────────────────────────────
    if primary_provider != "llamacpp":
        fallback_msgs = _strip_images(
            _truncate_for_fallback(messages_to_send, config.FALLBACK_LLM_MAX_HISTORY_TURNS)
        )
        fallback_cfg: dict[str, Any] = {
            "callbacks": [create_callback_handler(f"{module}_fallback")],
            "tags": ["final_node_output", "degraded"],
        }
        try:
            llamacpp = get_llm_client(provider="llamacpp", module=module)
            result = await with_retry(
                lambda: llamacpp.ainvoke(fallback_msgs, config=fallback_cfg),
                attempts=2,
                base=0.5,
                cap=8.0,
                fallback=None,
            )
        except Exception as exc:
            logger.warning("[%s] llamacpp fallback also failed: %s", module, exc)

    if result is not None:
        result.additional_kwargs["degraded"] = True
        return result

    # ── Tier 3: static error ───────────────────────────────────────────────
    logger.error("[%s] All cascade tiers exhausted, returning static error", module)
    return AIMessage(
        content=_error_message(lang),
        additional_kwargs={"degraded": True, "cascade_exhausted": True},
    )
