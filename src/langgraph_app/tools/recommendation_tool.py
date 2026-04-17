"""Restaurant search tool for the Supervisor agent.

Wraps the Google Maps restaurant search as a LangChain @tool that returns
raw restaurant JSON. Does NOT do LLM-based query extraction or formatting —
the Supervisor handles natural language packaging.
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langgraph_app.tools.tools import search_restaurants_tool as _raw_search
from langgraph_app.tools.map.ip_location import get_location_from_ip_async
from langgraph_app.utils.semaphores import with_semaphore

logger = logging.getLogger("wabi.tools.recommendation")


async def _resolve_location(
    lat: Optional[float],
    lng: Optional[float],
    user_ip: Optional[str] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Resolve user location: explicit coords -> IP fallback."""
    if lat is not None and lng is not None:
        return lat, lng
    # IP-based fallback
    ip_loc = await get_location_from_ip_async(user_ip)
    if ip_loc:
        return ip_loc
    return None, None


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
    max_results: int = Field(
        default=5, description="Maximum number of results to return."
    )


@tool("search_restaurants", args_schema=SearchRestaurantsInput)
async def search_restaurants(
    query: str,
    cuisine_type: Optional[str] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius_km: float = 5.0,
    max_results: int = 5,
) -> str:
    """Search for restaurants using Google Maps based on a natural language query.

    Args:
        query: What the user is looking for (e.g., "healthy salad place nearby").
        cuisine_type: Optional cuisine filter (e.g., "Japanese", "Italian").
        lat: User's latitude (if known).
        lng: User's longitude (if known).
        radius_km: Search radius in kilometers.
        max_results: Maximum number of results to return.

    Returns:
        A JSON string containing a list of restaurants with name, address,
        rating, and other details. Returns an error JSON on failure.
    """
    try:
        raw_result = await _raw_search.ainvoke(
            {
                "location": query,
                "cuisine_type": cuisine_type,
                "radius_km": radius_km,
                "lat": lat,
                "lng": lng,
                "max_results": max_results,
            }
        )
        result_dict = json.loads(raw_result)
        restaurants = result_dict.get("restaurants", [])
        if not restaurants:
            return json.dumps(
                {
                    "restaurants": [],
                    "message": "No restaurants found matching the criteria.",
                },
                ensure_ascii=False,
            )
        return json.dumps({"restaurants": restaurants}, ensure_ascii=False)
    except Exception as e:
        logger.error("search_restaurants tool failed: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})
