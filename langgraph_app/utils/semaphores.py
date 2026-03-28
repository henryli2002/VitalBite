"""Semantic Backpressure logic using Asyncio Semaphores.

This module provides macroscopic concurrency limits for Individual Agent Nodes within LangGraph.
It allows us to set `MAX_CONCURRENT_WORKERS` to 200 to instantly process fast intent classification
and chitchat payloads, while strictly capping heavy Vision and RAG workloads to prevent API throttling
(429 Too Many Requests) and memory overflow.

IMPORTANT: Semaphores are created lazily via a registry to avoid binding
to a "ghost" event loop that exists at module-import time but is NOT the
loop used by uvicorn at runtime (Python 3.12+ behavior).
"""

import asyncio
import functools
from typing import Callable, Any, Dict

# ---------- Lazy Semaphore Registry ----------
# Semaphore limits (name -> max_concurrent)
_SEMAPHORE_LIMITS: Dict[str, int] = {
    "intent":          300,   # Lightweight text classification
    "chitchat":        200,   # Flash-lite text, ~1s/req
    "recommendation":   200,   # LLM + Google Maps API + LLM, moderate
    "recognition":      30,   # HEAVY: Vision + FAISS. Each req carries ~2-4MB base64 image.
    "tutorial":         200,   # Medium-weight text LLM
    "goalplanning":     200,   # Needs full history pull, high memory per req
}

# Runtime cache: created per event-loop at first access
_SEMAPHORE_CACHE: Dict[str, asyncio.Semaphore] = {}


def _get_semaphore(name: str) -> asyncio.Semaphore:
    """Get or create a semaphore for the current running event loop."""
    if name not in _SEMAPHORE_CACHE:
        limit = _SEMAPHORE_LIMITS.get(name, 100)
        _SEMAPHORE_CACHE[name] = asyncio.Semaphore(limit)
    return _SEMAPHORE_CACHE[name]


# ---------- Public Accessors ----------
# These are callables, not raw Semaphore objects.
# Usage: `async with get_intent_semaphore(): ...`  or via `@with_semaphore("intent")`

def with_semaphore(name: str):
    """Decorator to enforce a lazily-initialized asyncio.Semaphore limit on async Graph nodes.
    
    Args:
        name: Key into _SEMAPHORE_LIMITS (e.g. "intent", "recognition").
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            sem = _get_semaphore(name)
            async with sem:
                return await func(*args, **kwargs)
        return wrapper
    return decorator
