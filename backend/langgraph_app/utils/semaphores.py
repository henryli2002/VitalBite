"""Semantic Backpressure logic using Asyncio Semaphores.

This module provides macroscopic concurrency limits for Individual Agent Nodes within LangGraph.
It allows fast intent classification and chitchat to run at full concurrency while capping
heavy Vision workloads to prevent API throttling (429 Too Many Requests) and memory overflow.

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
    "intent":          200,   # Lightweight text classification
    "chitchat":        200,   # Flash-lite text, ~1s/req
    "recommendation":   100,   # LLM + Google Maps API + LLM, moderate
    "recognition":      50,    # HEAVY: Vision model, ~2-4MB base64 per req. Capped for RAM safety
    "goalplanning":     100,   # Needs full history pull, high memory per req
}

# Runtime cache: created per event-loop at first access
_SEMAPHORE_CACHE: Dict[str, asyncio.Semaphore] = {}


def get_semaphore(name: str) -> asyncio.Semaphore:
    """Get or create a semaphore for the current running event loop."""
    if name not in _SEMAPHORE_CACHE:
        limit = _SEMAPHORE_LIMITS.get(name, 100)
        _SEMAPHORE_CACHE[name] = asyncio.Semaphore(limit)
    return _SEMAPHORE_CACHE[name]


# ---------- Public Accessors ----------
# These are callables, not raw Semaphore objects.
# Usage: `async with get_semaphore("intent"): ...`  or via `@with_semaphore("intent")`

def with_semaphore(name: str):
    """Decorator to enforce a lazily-initialized asyncio.Semaphore limit on async Graph nodes.
    
    Args:
        name: Key into _SEMAPHORE_LIMITS (e.g. "intent", "recognition").
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            sem = get_semaphore(name)
            async with sem:
                return await func(*args, **kwargs)
        return wrapper
    return decorator
