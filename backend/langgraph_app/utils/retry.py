"""Async retry with exponential backoff + full jitter.

Full jitter desynchronises concurrent retriers, preventing thundering-herd
re-bursts after a shared rate-limit event.

Usage:
    result = await with_retry(
        lambda: client.ainvoke(messages),
        fallback=None,          # return None on exhaustion instead of raising
    )
    if result is None:
        ...  # cascade to next tier
"""

import asyncio
import random
from typing import Any, Callable, Awaitable

# Sentinel: if caller does not pass `fallback`, exhaustion re-raises last error.
_RAISE = object()


async def with_retry(
    coro_fn: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 3,
    base: float = 0.3,
    cap: float = 5.0,
    fallback: Any = _RAISE,
) -> Any:
    """Retry *coro_fn* with exponential backoff and full jitter.

    Args:
        coro_fn:  Zero-argument async callable, called fresh each attempt.
        attempts: Maximum number of attempts (including the first).
        base:     Base sleep seconds. Doubles each attempt.
        cap:      Maximum sleep ceiling in seconds.
        fallback: Value to return when all attempts are exhausted.
                  Pass nothing (or ``_RAISE``) to re-raise the last exception.

    Sleep schedule (illustrative, actual value is random in [0, window]):
        attempt 0 fails → sleep ∈ [0, base]
        attempt 1 fails → sleep ∈ [0, base * 2]
        attempt 2 fails → sleep ∈ [0, min(cap, base * 4)]
        ...
    """
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            return await coro_fn()
        except Exception as exc:
            last_err = exc
            if attempt < attempts - 1:
                window = min(cap, base * (2 ** attempt))
                await asyncio.sleep(random.uniform(0, window))

    if fallback is _RAISE:
        raise last_err  # type: ignore[misc]
    return fallback
