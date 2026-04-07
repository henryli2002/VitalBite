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

REDIS_URL = os.environ.get("WABI_REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Strict concurrency limit to prevent API throttling
MAX_CONCURRENT_WORKERS = 200
WORKER_TASKS = []


def build_thinking_partial(
    node_name: str, node_output: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Build frontend-friendly partial thinking payload for each key node."""
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


def build_langchain_messages(history: List[Dict]) -> List[BaseMessage]:
    """Convert JSON messages back into LangChain message objects."""
    messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        timestamp = msg.get("timestamp")
        if role == "user":
            messages.append(
                HumanMessage(
                    content=content,
                    response_metadata={"timestamp": timestamp} if timestamp else {},
                )
            )
        elif role == "assistant":
            messages.append(
                AIMessage(
                    content=content,
                    response_metadata={"timestamp": timestamp} if timestamp else {},
                )
            )
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

        analysis = result.get("analysis", {})
        detected_intent = analysis.get("intent", "chitchat")

        # If intent is goalplanning and we have full history provided, re-invoke
        if (
            detected_intent == "goalplanning"
            and payload.get("invoke_full_history")
            and payload.get("full_messages")
        ):
            logger.info(
                f"[{user_id}] Goalplanning detected in Queue. Reprocessing with FULL history."
            )
            full_msgs = build_langchain_messages(payload.get("full_messages", []))
            full_state = {
                "messages": full_msgs,
                "session_id": payload.get("session_id", ""),
                "user_id": user_id,
                "user_name": payload.get("user_name"),
                "user_profile": payload.get("user_profile"),
                "user_context": payload.get("user_context", {}),
                "response_channel": response_channel,
            }
            full_config = {"configurable": {"thread_id": f"{thread_id}_full"}}

            accumulated_state = full_state.copy()
            async for output in graph.astream(full_state, config=full_config):
                for node_name, node_output in output.items():
                    logger.info(f"[{user_id}] Goalplanning Node completed: {node_name}")
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
                final_msg = last_msg.content

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


async def worker_loop(worker_id: int):
    """Continuously pop tasks from the Redis list."""
    logger.info(f"Worker {worker_id} started, ready to pop 'wabi_ai_queue'")
    while True:
        try:
            # Block aggressively (timeout 0 = unlimited) until list has item
            task_data = await redis_client.blpop("wabi_ai_queue", timeout=0)
            if task_data:
                _, payload_str = task_data
                payload = json.loads(payload_str)
                await process_task(payload)
        except Exception as e:
            logger.error(f"Worker {worker_id} fatal exception: {e}")
            await asyncio.sleep(2)


@app.on_event("startup")
async def startup_event():
    """Launch N concurrent async workers and pre-warm caches."""
    # Pre-warm IP geolocation cache (saves 3-5s on first recommendation)
    try:
        from langgraph_app.tools.map.ip_location import get_location_from_ip_async

        loc = await get_location_from_ip_async()
        logger.info(f"Pre-warmed IP geolocation cache: {loc}")
    except Exception as e:
        logger.warning(f"IP geolocation pre-warm failed (non-fatal): {e}")

    logger.info(f"Booting {MAX_CONCURRENT_WORKERS} AI Message Brokers...")
    for i in range(MAX_CONCURRENT_WORKERS):
        task = asyncio.create_task(worker_loop(i))
        WORKER_TASKS.append(task)


@app.get("/health")
async def health():
    return {"status": "ok", "message_brokers": len(WORKER_TASKS)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
