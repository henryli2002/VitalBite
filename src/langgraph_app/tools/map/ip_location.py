"""IP Geolocation with Redis caching.

The old implementation used synchronous `requests.get()` inside an async pipeline,
which blocks the entire event loop for up to 5 seconds — catastrophic at 200 concurrency.

This version uses httpx.AsyncClient + Redis cache (TTL 1 hour, since server IP rarely changes).
"""

import os
import json
import hashlib
from typing import Optional, Tuple, Dict
import httpx
import redis.asyncio as redis
from langgraph_app.utils.logger import get_logger

logger = get_logger(__name__)

from langgraph_app.config import config as _app_config
_redis_client = redis.from_url(_app_config.REDIS_URL, decode_responses=True)

# In-process cache (since server IP essentially never changes during a container lifecycle)
_ip_location_cache: Optional[Tuple[float, float]] = None
_ip_timezone_cache: Dict[str, str] = {}  # ip -> IANA timezone string


async def get_location_from_ip_async(ip_address: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """
    Async version: Get lat/lng from a specific IP address, or the server's public IP if None.
    Results are cached in-memory (process-level) and Redis (cross-restart).
    """
    global _ip_location_cache

    # If asking for server IP (no IP provided)
    is_server_ip = not ip_address or ip_address in ["127.0.0.1", "localhost", "0.0.0.0"]
    
    # Use global cache for server IP only
    if is_server_ip and _ip_location_cache is not None:
        return _ip_location_cache

    cache_key = f"wabi_cache:ip_location:{ip_address}" if not is_server_ip else "wabi_cache:ip_location"

    # Level 2: Redis cache (sub-ms)
    try:
        cached = await _redis_client.get(cache_key)
        if cached:
            data = json.loads(cached)
            result = (data["lat"], data["lng"])
            if is_server_ip:
                _ip_location_cache = result
            logger.info(f"IP location loaded from Redis cache ({ip_address or 'server'}): {result}")
            return result
    except Exception as e:
        logger.warning(f"Redis IP cache read failed: {e}")

    # Level 3: Live HTTP call (async, non-blocking)
    try:
        async with httpx.AsyncClient() as client:
            url = f"http://ip-api.com/json/{ip_address}" if not is_server_ip else "http://ip-api.com/json/"
            response = await client.get(url, timeout=5.0)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                result = (data.get("lat"), data.get("lon"))
                
                if is_server_ip:
                    _ip_location_cache = result
                    
                # Cache in Redis for 1 hour (server) or 24 hours (user IP)
                ttl = 3600 if is_server_ip else 86400
                try:
                    await _redis_client.setex(
                        cache_key, ttl,
                        json.dumps({"lat": result[0], "lng": result[1]})
                    )
                except Exception:
                    pass
                logger.info(f"IP location fetched live ({ip_address or 'server'}): {result}")
                return result
    except Exception as e:
        logger.warning(f"Failed to get location from IP {ip_address} (async): {e}")
    return None


async def get_timezone_from_ip_async(ip_address: str) -> Optional[str]:
    """Get IANA timezone string from an IP address.

    Uses in-process cache → Redis cache (24h TTL) → live ip-api.com call.
    Returns None for local/loopback addresses or on failure.
    """
    if not ip_address or ip_address in ("127.0.0.1", "localhost", "0.0.0.0"):
        return None

    if ip_address in _ip_timezone_cache:
        return _ip_timezone_cache[ip_address]

    cache_key = f"wabi_cache:ip_timezone:{ip_address}"
    try:
        cached = await _redis_client.get(cache_key)
        if cached:
            _ip_timezone_cache[ip_address] = cached
            return cached
    except Exception as e:
        logger.warning(f"Redis timezone cache read failed: {e}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://ip-api.com/json/{ip_address}", timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("timezone"):
                tz = data["timezone"]
                _ip_timezone_cache[ip_address] = tz
                try:
                    await _redis_client.setex(cache_key, 86400, tz)
                except Exception:
                    pass
                logger.info(f"IP timezone fetched live ({ip_address}): {tz}")
                return tz
    except Exception as e:
        logger.warning(f"Failed to get timezone from IP {ip_address}: {e}")

    return None


def get_location_from_ip(ip_address: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Synchronous fallback — returns cached value or None.
    
    IMPORTANT: This should NOT be called in async code paths anymore.
    Use get_location_from_ip_async() instead.
    """
    is_server_ip = not ip_address or ip_address in ["127.0.0.1", "localhost", "0.0.0.0"]
    if is_server_ip and _ip_location_cache is not None:
        return _ip_location_cache
    
    # Synchronous fallback for backwards compatibility
    import requests
    try:
        url = f"http://ip-api.com/json/{ip_address}" if not is_server_ip else "http://ip-api.com/json/"
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            return data.get("lat"), data.get("lon")
    except Exception as e:
        logger.warning(f"Failed to get location from IP {ip_address}: {e}")
    return None
