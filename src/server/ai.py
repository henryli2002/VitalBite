"""FastAPI standalone server encapsulating the LangGraph execution.
Serves as an autonomous Worker Node that pulls tasks from a Redis Queue.
"""

import logging
import os
import asyncio
import json
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

# CRITICAL: load_dotenv() MUST be executed BEFORE any LangChain components are imported.
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import redis.asyncio as redis

# Import graph *after* .env is fully populated in os.environ
from langgraph_app.orchestrator.graph import graph

app = FastAPI(title="WABI AI Worker Node", version="1.0.0")
logger = logging.getLogger("wabi.ai.worker")
logging.basicConfig(level=logging.INFO)

from langgraph_app.config import config as _app_config

redis_client = redis.from_url(_app_config.REDIS_URL, decode_responses=True)

# Track the dispatcher task for health reporting
_active_tasks: set[asyncio.Task] = set()
_dispatcher_task: asyncio.Task | None = None


def build_thinking_partial(
    node_name: str, node_output: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Build frontend-friendly partial thinking payload for each key node."""

    # --- Supervisor architecture nodes ---
    if node_name == "supervisor":
        # The supervisor node wraps the react agent; its output contains messages
        # including tool calls and responses. Extract useful info.
        messages = node_output.get("messages", []) or []
        if not messages:
            return None
        last = messages[-1] if messages else None
        if last is None:
            return None
        # Check for tool call messages (AIMessage with tool_calls)
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            tool_names = [tc.get("name", "unknown") for tc in tool_calls]
            return {
                "status": "partial",
                "node": "tool_call",
                "analysis": {"reasoning": f"Calling tools: {', '.join(tool_names)}"},
            }
        return None

    if node_name == "tools":
        # Tool execution results from the react agent
        messages = node_output.get("messages", []) or []
        if messages:
            last = messages[-1]
            content = getattr(last, "content", "")
            if content:
                snippet = str(content)[:200]
                return {
                    "status": "partial",
                    "node": "tool_result",
                    "analysis": {"reasoning": f"Tool returned: {snippet}"},
                }
        return None

    # --- Legacy architecture nodes ---
    if node_name in ("router", "intent_router"):
        analysis = node_output.get("analysis")
        if analysis:
            return {"status": "partial", "node": "intent_router", "analysis": analysis}
        return None

    if node_name == "chitchat":
        messages = node_output.get("messages", []) or []
        answer = ""
        if messages and isinstance(messages, list):
            first = messages[0]
            answer = getattr(first, "content", "") if first else ""
        if answer:
            return {
                "status": "partial",
                "node": "chitchat",
                "analysis": {"reasoning": f"Answering: {str(answer)[:180]}"},
            }
        return None

    if node_name == "recommendation":
        rec = node_output.get("recommendation_result") or {}
        restaurants = rec.get("restaurants") or []
        if not isinstance(restaurants, list) or not restaurants:
            return None
        names = [str(r.get("name", "")) for r in restaurants[:3] if isinstance(r, dict)]
        names = [n for n in names if n]
        if not names:
            return None
        return {
            "status": "partial",
            "node": "recommendation",
            "analysis": {
                "reasoning": f"Finding restaurants: Found restaurants: {', '.join(names)}"
            },
        }

    return None


def _format_image_annotation(refs: List[Dict[str, Any]]) -> str:
    """Trusted, server-injected annotation describing attached images.

    Synthesised on the fly from the ``image_refs`` column (never stored in the
    text ``content`` column). The Supervisor system prompt teaches the LLM to
    look for these ``<attached_image uuid=.../>`` markers and pass the UUID to
    the ``analyze_food_image`` tool.
    """
    lines = []
    for ref in refs or []:
        if not isinstance(ref, dict):
            continue
        uid = (ref.get("uuid") or "").strip()
        if not uid:
            continue
        desc = (ref.get("description") or "").strip()
        if desc:
            lines.append(f"<attached_image uuid={uid} description=\"{desc}\"/>")
        else:
            lines.append(f"<attached_image uuid={uid}/>")
    return "\n".join(lines)


def build_langchain_messages(history: List[Dict]) -> List[BaseMessage]:
    """Convert JSON messages back into LangChain message objects.

    For user messages with attached images, the trusted ``image_refs`` list is
    rendered as ``<attached_image uuid=.../>`` markers and appended to the
    text. The Supervisor never parses UUIDs out of the raw user text — it only
    trusts these server-injected markers.
    """
    messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content") or ""
        timestamp = msg.get("timestamp")
        refs = msg.get("image_refs") or []
        extra = {"timestamp": timestamp} if timestamp else {}
        if role == "user":
            annotation = _format_image_annotation(refs)
            final_content = f"{content}\n\n{annotation}" if (content and annotation) else (annotation or content)
            messages.append(HumanMessage(content=final_content, additional_kwargs=extra))
        elif role == "assistant":
            messages.append(AIMessage(content=content, additional_kwargs=extra))
    return messages


async def process_task(payload: Dict[str, Any]):
    user_id = payload.get("user_id", "unknown")
    thread_id = payload.get("thread_id", "")
    response_channel = payload.get("response_channel")

    logger.info(f"[{user_id}] Picked up AI task (thread_id: {thread_id})...")

    initial_state = {
        "messages": build_langchain_messages(payload.get("messages", [])),
        "session_id": payload.get("session_id", ""),
        "user_id": user_id,
        "user_name": payload.get("user_name"),
        "user_profile": payload.get("user_profile"),
        "user_context": payload.get("user_context", {}),
        "response_channel": response_channel,
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        accumulated_state = initial_state.copy()

        async for output in graph.astream(initial_state, config=config):
            for node_name, node_output in output.items():
                logger.info(f"[{user_id}] Node completed: {node_name}")
                accumulated_state.update(node_output)
                if response_channel:
                    partial_payload = build_thinking_partial(node_name, node_output)
                    if partial_payload:
                        await redis_client.publish(
                            response_channel, json.dumps(partial_payload)
                        )

        result = accumulated_state

        # Extract final AI response (safe null-check)
        final_msg = "No response generated."
        result_messages = result.get("messages", [])
        if result_messages:
            last_msg = result_messages[-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                raw_content = last_msg.content
                if isinstance(raw_content, str):
                    final_msg = raw_content
                elif isinstance(raw_content, list):
                    # LangChain might return a list of dicts for multimodal/tool responses
                    text_parts = [
                        c.get("text", "")
                        for c in raw_content
                        if isinstance(c, dict) and "text" in c
                    ]
                    final_msg = (
                        " ".join(text_parts).strip() if text_parts else str(raw_content)
                    )
                else:
                    final_msg = str(raw_content)

        response_payload = {
            "status": "success",
            "messages": [{"role": "assistant", "content": final_msg}],
            "analysis": result.get("analysis", {}),
        }

        logger.info(
            f"[{user_id}] Queue execution complete. Intent: {result.get('analysis', {}).get('intent', 'unknown')}"
        )
        logger.info(
            f"[{user_id}] Publishing response to Redis Channel: {response_channel}"
        )

        await redis_client.publish(response_channel, json.dumps(response_payload))

    except Exception as e:
        logger.error(f"[{user_id}] Task execution failed: {e}", exc_info=True)
        # BUBBLE UP CRITICAL ERROR TO THE FRONTEND
        error_payload = {
            "status": "error",
            "message": f"Backend Error: {str(e)}",
            "node_failure": True,
        }
        if response_channel:
            await redis_client.publish(response_channel, json.dumps(error_payload))


async def _run_task(payload: Dict[str, Any]) -> None:
    """Wrap process_task so we can track and clean up the asyncio.Task."""
    task = asyncio.current_task()
    _active_tasks.add(task)
    try:
        await process_task(payload)
    finally:
        _active_tasks.discard(task)


async def dispatcher() -> None:
    """Single Redis connection that spawns an independent task per job.

    Concurrency is governed entirely by the per-agent Semaphores in
    semaphores.py — not by the number of workers here.
    """
    logger.info("Dispatcher started, listening on 'wabi_ai_queue'")
    while True:
        try:
            task_data = await redis_client.blpop("wabi_ai_queue", timeout=0)
            if task_data:
                _, payload_str = task_data
                asyncio.create_task(_run_task(json.loads(payload_str)))
        except Exception as e:
            logger.error(f"Dispatcher error: {e}")
            await asyncio.sleep(2)


@app.on_event("startup")
async def startup_event():
    """Start the dispatcher and pre-warm caches."""
    global _dispatcher_task
    # Pre-warm IP geolocation cache (saves 3-5s on first recommendation)
    try:
        from langgraph_app.tools.map.ip_location import get_location_from_ip_async

        loc = await get_location_from_ip_async()
        logger.info(f"Pre-warmed IP geolocation cache: {loc}")
    except Exception as e:
        logger.warning(f"IP geolocation pre-warm failed (non-fatal): {e}")

    _dispatcher_task = asyncio.create_task(dispatcher())
    logger.info("AI dispatcher started")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dispatcher_alive": _dispatcher_task is not None
        and not _dispatcher_task.done(),
        "active_tasks": len(_active_tasks),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
