"""IP Geolocation with Redis caching.

The old implementation used synchronous `requests.get()` inside an async pipeline,
which blocks the entire event loop for up to 5 seconds — catastrophic at 200 concurrency.

This version uses httpx.AsyncClient + Redis cache (TTL 1 hour, since server IP rarely changes).
"""

import os
import json
import hashlib
from typing import Optional, Tuple
import httpx
import redis.asyncio as redis
from langgraph_app.utils.logger import get_logger

logger = get_logger(__name__)

REDIS_URL = os.environ.get("WABI_REDIS_URL", "redis://localhost:6379/0")
_redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# In-process cache (since server IP essentially never changes during a container lifecycle)
_ip_location_cache: Optional[Tuple[float, float]] = None


async def get_location_from_ip_async() -> Optional[Tuple[float, float]]:
    """
    Async version: Get lat/lng from server's public IP.
    Results are cached in-memory (process-level) and Redis (cross-restart).
    """
    global _ip_location_cache

    # Level 1: In-process memory cache (0ms)
    if _ip_location_cache is not None:
        return _ip_location_cache

    # Level 2: Redis cache (sub-ms)
    try:
        cached = await _redis_client.get("wabi_cache:ip_location")
        if cached:
            data = json.loads(cached)
            _ip_location_cache = (data["lat"], data["lng"])
            logger.info(f"IP location loaded from Redis cache: {_ip_location_cache}")
            return _ip_location_cache
    except Exception as e:
        logger.warning(f"Redis IP cache read failed: {e}")

    # Level 3: Live HTTP call (async, non-blocking)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://ip-api.com/json/", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                result = (data.get("lat"), data.get("lon"))
                _ip_location_cache = result
                # Cache in Redis for 1 hour
                try:
                    await _redis_client.setex(
                        "wabi_cache:ip_location", 3600,
                        json.dumps({"lat": result[0], "lng": result[1]})
                    )
                except Exception:
                    pass
                logger.info(f"IP location fetched live: {result}")
                return result
    except Exception as e:
        logger.warning(f"Failed to get location from IP (async): {e}")
    return None


def get_location_from_ip() -> Optional[Tuple[float, float]]:
    """Synchronous fallback — returns cached value or None.
    
    IMPORTANT: This should NOT be called in async code paths anymore.
    Use get_location_from_ip_async() instead.
    """
    if _ip_location_cache is not None:
        return _ip_location_cache
    
    # Synchronous fallback for backwards compatibility
    import requests
    try:
        response = requests.get("http://ip-api.com/json/", timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            return data.get("lat"), data.get("lon")
    except Exception as e:
        logger.warning(f"Failed to get location from IP: {e}")
    return None
