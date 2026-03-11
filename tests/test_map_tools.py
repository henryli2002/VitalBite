import os
import sys
import json
import requests
from dotenv import load_dotenv

# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_app.tools.tools import search_restaurants_tool, get_user_location_by_ip_tool

def test_map_tools():
    # 1. Load environment variables first!
    load_dotenv()
    
    # Reload the api_key in map_tool since it was initialized before load_dotenv() was called
    from langgraph_app.tools.map.google_maps import map_tool
    map_tool.api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    
    # 2. Test IP Location Tool
    print("--- Testing IP Location Tool ---")
    
    # Let's get raw IP for reference first
    try:
        raw_ip_response = requests.get('https://api.ipify.org?format=json')
        print(f"[Reference] Current Public IP: {raw_ip_response.json().get('ip')}")
    except Exception as e:
        print(f"[Reference] Failed to get public IP directly: {e}")
        
    print("\nCalling `get_user_location_by_ip_tool`...")
    ip_result_json = get_user_location_by_ip_tool.invoke({})
    print(f"Tool Result: {ip_result_json}")
    
    ip_result = json.loads(ip_result_json)
    
    if ip_result.get("status") != "success":
        print("❌ IP Location failed. Aborting further tests.")
        return
        
    lat = ip_result.get("lat")
    lng = ip_result.get("lng")
    print(f"✅ Extracted Location: Lat={lat}, Lng={lng}")
    
    
    # 3. Test Google Maps Search Restaurants Tool
    print("\n\n--- Testing Search Restaurants Tool ---")
    
    if not os.environ.get("GOOGLE_MAPS_API_KEY"):
        print("⚠️ Warning: GOOGLE_MAPS_API_KEY is not set in .env")
        print("The tool will gracefully return an empty array if there is no key.")
        
    search_input = {
        "location": None, # Explicitly test the IP-based coordinate search
        "cuisine_type": "cafe", # Let's search for nearby cafes
        "radius_km": 2.0,
        "lat": lat,
        "lng": lng
    }
    
    print(f"Calling `search_restaurants_tool` with inputs:")
    print(json.dumps(search_input, indent=2))
    
    raw_result = search_restaurants_tool.invoke(search_input)
    
    print("\n⬇️ Raw Result from `search_restaurants_tool`:")
    print(raw_result)
    
    # Parse the result for better readability in output
    parsed_result = json.loads(raw_result)
    if parsed_result.get("status") == "success":
        restaurants = parsed_result.get("restaurants", [])
        print(f"\n✅ Successfully found {len(restaurants)} restaurants/cafes!")
        for i, rest in enumerate(restaurants):
            print(f"{i+1}. {rest.get('name')} (Rating: {rest.get('rating')})")
            print(f"   Address: {rest.get('address')}")
    else:
        print(f"\n❌ Search failed: {parsed_result.get('message')}")

if __name__ == "__main__":
    test_map_tools()