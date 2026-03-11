import requests
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def get_location_from_ip() -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude based on the server's public IP using ip-api.com.
    Returns:
        Tuple of (latitude, longitude) or None if fails.
    """
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            return data.get("lat"), data.get("lon")
    except Exception as e:
        logger.warning(f"Failed to get location from IP: {e}")
    return None
