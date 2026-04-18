"""Google Maps API Tools for location and place searching with REDIS CACHING."""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import hashlib
import httpx
import redis.asyncio as redis
from langgraph_app.utils.logger import get_logger
from langgraph_app.config import config as _app_config
from langgraph_app.utils.retry import with_retry

logger = get_logger(__name__)


class GoogleMapsTool:
    """Wrapper for Google Maps API with 24-hour Redis caching for identical queries."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            logger.warning(
                "GOOGLE_MAPS_API_KEY is not set. Google Maps tools will fail."
            )

        self.text_search_url = "https://places.googleapis.com/v1/places:searchText"
        self.redis_client = redis.from_url(_app_config.REDIS_URL, decode_responses=True)
        # Persistent HTTP client — reuses TCP connections across requests
        self._http_client = httpx.AsyncClient(timeout=10.0)
        self.CACHE_TTL_SECONDS = 86400  # 24 hours for successful results
        self.CACHE_TTL_EMPTY = 3600     # 1 hour for empty results

    def _generate_cache_key(self, payload: Dict[str, Any]) -> str:
        """Generates an MD5 hash cache key based on the API request payload."""
        payload_str = json.dumps(payload, sort_keys=True)
        return "maps_cache_" + hashlib.md5(payload_str.encode('utf-8')).hexdigest()

    async def search_restaurants(
        self,
        location: Optional[str] = None,
        cuisine_type: Optional[str] = None,
        radius_km: Optional[float] = None,
        lat_lng: Optional[Tuple[float, float]] = None,
        max_results: int = 20,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for restaurants based on location/coordinates and cuisine asynchronously.
        Implements Redis caching to avoid massive Google Maps billing during scale.

        Returns a dict ``{"restaurants": [...], "next_page_token": str | None}``.
        Pass the returned ``next_page_token`` back in subsequent calls to fetch the
        next batch (Google Places Text Search supports up to 60 results across 3 pages).
        """
        if not self.api_key:
            return {"restaurants": [], "next_page_token": None}

        query_parts = []
        if cuisine_type:
            # 严格映射到全英文以方便 Google Maps 识别，并保障缓存 100% 聚合
            c_norm = cuisine_type.lower().strip()
            synonyms = {
                "咖啡": "cafe", "咖啡店": "cafe", "咖啡馆": "cafe", "coffee": "cafe", "coffee shop": "cafe", "cafe": "cafe",
                "汉堡": "hamburger restaurant", "汉堡店": "hamburger restaurant", "burger": "hamburger restaurant", "burgers": "hamburger restaurant",
                "火锅": "hotpot restaurant", "火锅店": "hotpot restaurant", "hot pot": "hotpot restaurant", "hotpot": "hotpot restaurant",
                "寿司": "sushi restaurant", "寿司店": "sushi restaurant", "sushi": "sushi restaurant",
                "面条": "noodle restaurant", "面馆": "noodle restaurant", "拉面": "noodle restaurant", "noodle": "noodle restaurant", "noodles": "noodle restaurant",
                "牛肉面": "beef noodle restaurant", "披萨": "pizza restaurant", "披萨店": "pizza restaurant", "pizza": "pizza restaurant",
                "中餐": "chinese restaurant", "中餐厅": "chinese restaurant", "chinese": "chinese restaurant", "chinese food": "chinese restaurant",
                "日料": "japanese restaurant", "日本料理": "japanese restaurant", "japanese": "japanese restaurant", "japanese food": "japanese restaurant",
                "韩餐": "korean restaurant", "韩国料理": "korean restaurant", "korean": "korean restaurant", "korean food": "korean restaurant",
                "素食": "vegetarian restaurant", "vegetarian": "vegetarian restaurant", "vegan": "vegan restaurant"
            }
            c_norm = synonyms.get(c_norm, c_norm)
            query_parts.append(c_norm)
            
        if location:
            query_parts.append(location)

        if not query_parts:
            query_parts = ["restaurant"]

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.id,places.types,nextPageToken",
        }

        payload: Dict[str, Any] = {
            "textQuery": " ".join(query_parts),
            "pageSize": min(max(max_results, 1), 20),  # Places v1 caps pageSize at 20
        }
        if page_token:
            payload["pageToken"] = page_token

        if lat_lng:
            radius_meters = int(radius_km * 1000) if radius_km else 5000
            payload["locationBias"] = {
                "circle": {
                    "center": {"latitude": lat_lng[0], "longitude": lat_lng[1]},
                    "radius": radius_meters,
                }
            }

        # 1. Check Redis Cache
        cache_key = self._generate_cache_key(payload)
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Redis Cache HIT for key: {cache_key}")
                await self.redis_client.incr("wabi_metrics:cache_hit")
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Redis Cache GET failed: {e}")

        # 2. Cache MISS -> Call Google (with retry)
        logger.info(f"Redis Cache MISS: Fetching live data from Google Maps for {payload['textQuery']}...")
        try:
            await self.redis_client.incr("wabi_metrics:cache_miss")

            async def _do_request():
                resp = await self._http_client.post(
                    self.text_search_url, json=payload, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            data = await with_retry(_do_request, attempts=3, base=1.0, cap=20.0, fallback=None)
            if data is None:
                logger.error("Google Maps API failed after retries")
                return {"restaurants": [], "next_page_token": None}

            places = data.get("places", [])
            formatted_results = []
            for place in places:
                formatted_results.append(
                    {
                        "name": place.get("displayName", {}).get("text", "Unknown"),
                        "address": place.get("formattedAddress", "Unknown"),
                        "rating": place.get("rating", 0.0),
                        "price_level": place.get("priceLevel", "UNKNOWN"),
                        "place_id": place.get("id"),
                        "types": place.get("types", []),
                    }
                )

            batch = {
                "restaurants": formatted_results,
                "next_page_token": data.get("nextPageToken"),
            }

            # 3. Store the successful response in Redis Cache
            cache_ttl = self.CACHE_TTL_SECONDS if formatted_results else self.CACHE_TTL_EMPTY
            try:
                await self.redis_client.setex(
                    name=cache_key,
                    time=cache_ttl,
                    value=json.dumps(batch, ensure_ascii=False),
                )
            except Exception as e:
                logger.warning(f"Redis Cache SET failed: {e}")

            return batch

        except httpx.HTTPStatusError as e:
            logger.error(f"Error calling Google Maps API: {e}")
            logger.error(f"Response: {e.response.text}")
            return {"restaurants": [], "next_page_token": None}
        except Exception as e:
            logger.error(f"Unknown error during Google Maps search: {e}")
            return {"restaurants": [], "next_page_token": None}


# Singleton instance
map_tool = GoogleMapsTool()
