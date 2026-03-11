"""Google Maps API Tools for location and place searching."""
from typing import Dict, Any, List, Optional, Tuple
import os
import requests
import logging

logger = logging.getLogger(__name__)

class GoogleMapsTool:
    """Wrapper for Google Maps API, specifically focusing on Places for food recommendation."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_MAPS_API_KEY is not set. Google Maps tools will fail.")
            
        # Using Google Places API (New) endpoints
        self.text_search_url = "https://places.googleapis.com/v1/places:searchText"
        self.nearby_search_url = "https://places.googleapis.com/v1/places:searchNearby"

    def search_restaurants(
        self,
        location: Optional[str] = None,
        cuisine_type: Optional[str] = None,
        radius_km: Optional[float] = None,
        lat_lng: Optional[Tuple[float, float]] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for restaurants based on location/coordinates and cuisine.
        """
        if not self.api_key:
            return []

        # Construct the text query
        query_parts = []
        if cuisine_type:
            query_parts.append(cuisine_type)
        if location:
            query_parts.append(location)
            
        # Default fallback
        if not query_parts and not lat_lng:
            query_parts = ["restaurant"]
        elif not query_parts and lat_lng:
            query_parts = ["restaurant"]

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.id,places.types"
        }

        # If we have lat/lng, we can use Nearby Search (or Text Search with location bias)
        # For simplicity and power, Text Search (New) with location bias is very effective
        payload: Dict[str, Any] = {
            "textQuery": " ".join(query_parts) if query_parts else "restaurant",
            "maxResultCount": max_results,
        }

        if lat_lng:
            radius_meters = int(radius_km * 1000) if radius_km else 5000 # default 5km
            payload["locationBias"] = {
                "circle": {
                    "center": {
                        "latitude": lat_lng[0],
                        "longitude": lat_lng[1]
                    },
                    "radius": radius_meters
                }
            }

        try:
            response = requests.post(self.text_search_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            places = data.get("places", [])
            formatted_results = []
            
            for place in places:
                formatted_results.append({
                    "name": place.get("displayName", {}).get("text", "Unknown"),
                    "address": place.get("formattedAddress", "Unknown"),
                    "rating": place.get("rating", 0.0),
                    "price_level": place.get("priceLevel", "UNKNOWN"),
                    "place_id": place.get("id"),
                    "types": place.get("types", [])
                })
                
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Google Maps API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return []

# Singleton instance
map_tool = GoogleMapsTool()
