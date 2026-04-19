"""Restaurant search tool for the Supervisor agent.

Wraps Google Maps Places v1 text search with:
- Browser-GPS → IP geolocation fallback (via RunnableConfig)
- Per-user Redis pagination state that uses Google's ``nextPageToken`` so
  "give me more" requests keep fetching fresh results until the API is exhausted.
"""

import json
import hashlib
import logging
from typing import Optional

import redis.asyncio as redis
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from langgraph_app.config import config as _app_config
from langgraph_app.tools.map.google_maps import map_tool
from langgraph_app.tools.map.ip_location import get_location_from_ip_async

logger = logging.getLogger("wabi.tools.recommendation")

_redis_client = redis.from_url(_app_config.REDIS_URL, decode_responses=True)

PER_PAGE = 5
STATE_TTL_SECONDS = 1800  # 30 min — long enough to "keep asking for more"


async def _resolve_location(
    lat: Optional[float],
    lng: Optional[float],
    user_ip: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    """Prefer explicit coords, else IP lookup on the *client's* IP only.

    Deliberately does NOT fall back to the server's public IP — a Singapore
    host resolving a user in China would silently return Singapore coordinates.
    """
    if lat is not None and lng is not None:
        return lat, lng
    if not user_ip or user_ip in ("127.0.0.1", "localhost", "0.0.0.0"):
        return None, None
    ip_loc = await get_location_from_ip_async(user_ip)
    if ip_loc:
        return ip_loc
    return None, None


def _state_key(
    user_id: str,
    query: str,
    cuisine_type: Optional[str],
    lat: Optional[float],
    lng: Optional[float],
    radius_km: float,
) -> str:
    payload = json.dumps(
        {
            "q": query,
            "c": cuisine_type or "",
            "lat": round(lat, 3) if lat is not None else None,
            "lng": round(lng, 3) if lng is not None else None,
            "r": radius_km,
        },
        sort_keys=True,
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"wabi_cache:recs_state:{user_id}:{digest}"


async def _load_state(key: str) -> dict:
    try:
        raw = await _redis_client.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning("recs_state GET failed: %s", e)
    return {"results": [], "next_token": None, "exhausted": False}


async def _save_state(key: str, state: dict) -> None:
    try:
        await _redis_client.setex(key, STATE_TTL_SECONDS, json.dumps(state, ensure_ascii=False))
    except Exception as e:
        logger.warning("recs_state SET failed: %s", e)


class SearchRestaurantsInput(BaseModel):
    query: str = Field(
        default="restaurants",
        description="What the user is looking for. Defaults to 'restaurants' if the user just asks for general recommendations.",
    )
    cuisine_type: Optional[str] = Field(
        default=None,
        description="Optional cuisine filter (e.g., 'Japanese', 'Italian').",
    )
    lat: Optional[float] = Field(
        default=None, description="User's latitude. Leave empty if unknown."
    )
    lng: Optional[float] = Field(
        default=None, description="User's longitude. Leave empty if unknown."
    )
    radius_km: float = Field(default=5.0, description="Search radius in kilometers.")
    page: int = Field(
        default=1,
        description="1-based page number. Each page returns up to 5 restaurants. Increment this (2, 3, 4, ...) when the user asks for 'more' or 'different' recommendations; the server uses Google's pagination token to fetch fresh results under the hood.",
    )


@tool("search_restaurants", args_schema=SearchRestaurantsInput)
async def search_restaurants(
    query: str,
    cuisine_type: Optional[str] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius_km: float = 5.0,
    page: int = 1,
    config: RunnableConfig = None,
) -> str:
    """Search for restaurants using Google Maps based on a natural language query.

    CRITICAL INSTRUCTION FOR LLM:
    1. If the user asks for a recommendation, DO NOT ASK FOR CLARIFICATION. Immediately call this tool using the default query 'restaurants'.
    2. ALWAYS execute this tool to fetch fresh results for the current turn. NEVER rely on or copy past recommendations from the chat history.
    3. If the user asks for "more", "different", or "another batch" of restaurants, YOU MUST increase the `page` parameter to 2, 3, etc., and call this tool again to fetch new results.

    Args:
        query: What the user is looking for (e.g., "healthy salad place nearby").
        cuisine_type: Optional cuisine filter (e.g., "Japanese", "Italian").
        lat: User's latitude (if known).
        lng: User's longitude (if known).
        radius_km: Search radius in kilometers.
        page: 1-based page index. Each page returns up to 5 restaurants.

    Returns:
        A JSON string with ``{"restaurants": [...], "page": N, "exhausted": bool}``.
        When ``exhausted`` is true and the slice is empty, no more restaurants
        are available for this query; suggest the user try a different query
        or wider radius.
    """
    # Pull the per-turn user_context from the runtime config (not exposed to LLM).
    user_ctx = {}
    user_id = "anon"
    if config and isinstance(config, dict):
        configurable = config.get("configurable") or {}
        user_ctx = configurable.get("user_context") or {}
        user_id = configurable.get("user_id") or "anon"

    # Fall back: browser GPS → client IP.
    lat, lng = await _resolve_location(
        lat if lat is not None else user_ctx.get("lat"),
        lng if lng is not None else user_ctx.get("lng"),
        user_ctx.get("user_ip"),
    )

    page = max(1, int(page))
    key = _state_key(user_id, query, cuisine_type, lat, lng, radius_km)
    state = await _load_state(key)

    needed = page * PER_PAGE

    try:
        # Fetch more batches from Google until we have enough OR API is exhausted.
        while len(state["results"]) < needed and not state["exhausted"]:
            lat_lng = (lat, lng) if lat is not None and lng is not None else None
            batch = await map_tool.search_restaurants(
                location=query,
                cuisine_type=cuisine_type,
                radius_km=radius_km,
                lat_lng=lat_lng,
                max_results=20,
                page_token=state["next_token"],
            )
            new_results = batch.get("restaurants", []) if isinstance(batch, dict) else []
            state["results"].extend(new_results)
            state["next_token"] = batch.get("next_page_token") if isinstance(batch, dict) else None
            if not state["next_token"] or not new_results:
                state["exhausted"] = True

        await _save_state(key, state)

        start_idx = (page - 1) * PER_PAGE
        end_idx = start_idx + PER_PAGE
        slice_ = state["results"][start_idx:end_idx]

        if not slice_:
            return json.dumps(
                {
                    "restaurants": [],
                    "page": page,
                    "exhausted": True,
                    "message": (
                        "No more restaurants available for this query. "
                        "Suggest the user try a different cuisine, area, or a wider radius."
                    ),
                },
                ensure_ascii=False,
            )

        display_guidance = (
            "RESTAURANT RENDERING RULES (authoritative):\n"
            "1. Open with ONE short sentence introducing the list (match the user's language).\n"
            "2. Immediately after the intro, emit EXACTLY ONE fenced code block tagged `restaurants` whose body is a JSON array. The array MUST preserve the full `restaurants` list from this tool result in the same order. Do NOT shrink it, summarize it, or drop entries. Output every restaurant returned for the current page exactly once. Each object MUST have these keys:\n"
            "   - name        (string, copy verbatim from the tool result)\n"
            "   - address     (string, copy verbatim)\n"
            "   - rating      (number or null, copy verbatim)\n"
            "   - cuisine     (string, a short, human-friendly category inferred from `types` e.g. 'Japanese', 'Cafe', 'Hotpot'. Keep it in the reply's language.)\n"
            "   - health      (string, ONE of: 'healthy', 'balanced', 'indulgent'. Pick based on cuisine + restaurant name; favour 'healthy' for salad/veg/poke/steamed/lean-protein spots, 'indulgent' for fried/BBQ/dessert/burger/hotpot, else 'balanced'.)\n"
            "   - advice      (string, 10-30 char concrete dining advice in the reply's language — e.g. 'Order grilled over fried, skip sugary drinks'. Personalize when the user's profile has relevant flags.)\n"
            "3. DO NOT render a Markdown table. DO NOT repeat the list in prose. DO NOT add extra keys.\n"
            "4. Do NOT merge entries together, and do NOT replace the full list with only the top 3 or another shorter subset.\n"
            "5. After the fenced block, you may add ONE short closing sentence (optional).\n"
            "6. If the user asks 'more', call this tool again with page+1 — do NOT reuse past results.\n"
            "Example skeleton:\n"
            "附近有一些餐厅可以考虑：\n"
            "```restaurants\n"
            "[{\"name\":\"...\",\"address\":\"...\",\"rating\":4.2,\"cuisine\":\"日料\",\"health\":\"healthy\",\"advice\":\"选清汤拉面，少加味精\"}]\n"
            "```"
        )

        return json.dumps(
            {
                "restaurants": slice_,
                "page": page,
                "exhausted": state["exhausted"] and end_idx >= len(state["results"]),
                "location_used": (
                    {"lat": lat, "lng": lng} if lat is not None and lng is not None else None
                ),
                "display_guidance": display_guidance,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error("search_restaurants tool failed: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})
