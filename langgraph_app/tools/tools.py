"""LangChain tools for Google Maps integration."""
from typing import Optional, List, Dict, Any, Tuple
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json

from .map.google_maps import map_tool
from .map.ip_location import get_location_from_ip


class SearchRestaurantsInput(BaseModel):
    """Input parameters for searching restaurants."""
    location: Optional[str] = Field(None, description="The location to search in (e.g., '台北市大安区').")
    cuisine_type: Optional[str] = Field(None, description="The type of food or cuisine (e.g., '素食', '日式', 'Italian').")
    radius_km: Optional[float] = Field(5.0, description="The search radius in kilometers.")
    lat: Optional[float] = Field(None, description="The user's current latitude, if available.")
    lng: Optional[float] = Field(None, description="The user's current longitude, if available.")
    max_results: Optional[int] = Field(5, description="The maximum number of results to return.")


@tool("get_user_location_by_ip")
def get_user_location_by_ip_tool() -> str:
    """
    Get the user's current approximate location (latitude and longitude) based on their IP address.
    Use this when the user hasn't provided their explicit location but you need coordinates for searching.
    Returns a JSON string containing lat and lng, or an error status.
    """
    loc = get_location_from_ip()
    if loc:
        return json.dumps({"status": "success", "lat": loc[0], "lng": loc[1]})
    else:
        return json.dumps({"status": "error", "message": "Could not determine location from IP."})


@tool("search_restaurants", args_schema=SearchRestaurantsInput)
async def search_restaurants_tool(
    location: Optional[str] = None,
    cuisine_type: Optional[str] = None,
    radius_km: Optional[float] = 5.0,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    max_results: Optional[int] = 5
) -> str:
    """
    Search for restaurants using Google Maps API based on location, cuisine, and optional coordinates.
    Returns a JSON string containing a list of recommended restaurants with details like name, address, rating, and price level.
    """
    
    lat_lng: Optional[Tuple[float, float]] = None
    if lat is not None and lng is not None:
        lat_lng = (lat, lng)

    results = await map_tool.search_restaurants(
        location=location,
        cuisine_type=cuisine_type,
        radius_km=radius_km,
        lat_lng=lat_lng,
        max_results=max_results
    )

    if not results:
        return json.dumps({"status": "error", "message": "No restaurants found matching the criteria or API error occurred."})

    return json.dumps({"status": "success", "restaurants": results}, ensure_ascii=False)

# List of tools to be bound to LLMs
map_tools = [search_restaurants_tool, get_user_location_by_ip_tool]
