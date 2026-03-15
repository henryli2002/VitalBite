"""Food recommendation agent for restaurant and food suggestions."""

from typing import Dict, Any, List
from datetime import datetime
import json
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_app.utils.utils import (
    get_dominant_language,
)
from langgraph_app.tools.tools import search_restaurants_tool
from langgraph_app.tools.map.ip_location import get_location_from_ip


class RecommendationQuery(BaseModel):
    """Structured query parameters for recommendations."""
    radius_km: float | None = None
    cuisine_type: str | None = Field(description="cuisine type or null. If an image of food is provided, infer the cuisine from it.")
    dietary_restrictions: List[str] = Field(default_factory=list)
    price_range: str | None = Field(description="budget|moderate|expensive or null")
    # TODO: Add logic to receive precise lat/lng from frontend after prompting user for location permission
    # lat: float | None = Field(None, description="latitude if provided in context")
    # lng: float | None = Field(None, description="longitude if provided in context")


def food_recommendation_node(state: GraphState) -> GraphState:
    """
    Provide restaurant and food recommendations based on user query.
    """
    state = state.copy()
    messages = state.setdefault("messages", [])
    client = get_llm_client(module="food_recommendation")

    lang = get_dominant_language(messages)

    extraction_prompt = f"""Extract recommendation parameters from the user's query, considering the conversation history and any images provided. An image might give clues about the desired cuisine.

Your response must be a JSON object with the requested schema. Ensure your response understands the user's dominant language ('{lang}')."""

    try:
        # Step 1: Use local messages to append extraction instruction
        local_messages = messages.copy()
        local_messages.append(HumanMessage(content=extraction_prompt))

        query_params = client.generate_structured(
            messages=local_messages, 
            schema=RecommendationQuery,
            system_prompt="You are an expert food recommendation assistant."
        )
        
        # Determine location fallback
        final_lat = None
        final_lng = None
        
        # Currently default to IP location.
        # TODO: Update this to use explicit user location or frontend provided lat/lng
        ip_loc = get_location_from_ip()
        if ip_loc:
            final_lat, final_lng = ip_loc
        
        # Step 2: Get restaurant recommendations using actual tool
        raw_result = search_restaurants_tool.invoke({
            "location": None, # Force no text location bias for now, rely purely on coordinates/ip
            "cuisine_type": query_params.cuisine_type,
            "radius_km": query_params.radius_km or 5.0,
            "lat": final_lat,
            "lng": final_lng
        })
        
        try:
            result_dict = json.loads(raw_result)
            restaurants = result_dict.get("restaurants", [])
        except Exception:
            restaurants = []
        
        if not restaurants:
            state["recommendation_result"] = {"restaurants": []}
            state["final_response"] = "抱歉，没有找到符合您要求的餐厅。请尝试调整搜索条件。" if lang == "Chinese" else "Sorry, no matching restaurants were found. Please try adjusting your search criteria."
            messages.append(AIMessage(content=state["final_response"]))
            state.setdefault("message_timestamps", []).append(datetime.utcnow().isoformat())
            return state
        
        # Step 3: Format recommendations into natural language
        formatting_prompt = f"""Convert this restaurant list into a friendly, natural language recommendation based on the user's query.

Restaurants:
{json.dumps(restaurants, ensure_ascii=False, indent=2)}

Provide a warm, helpful response that:
1. Acknowledges the user's request.
2. Lists the recommended restaurants with key details.
3. Mentions why each restaurant might be a good fit.
4. Keeps it concise and friendly.
"""

        system_instruction = (
            f"You are a friendly food recommendation assistant. Provide helpful, personalized restaurant suggestions. Your entire response should be in the user's dominant language ('{lang}'). However, if the user specifically asks for another language, please switch to that language."
        )
        
        local_messages_2 = messages.copy()
        local_messages_2.append(HumanMessage(content=formatting_prompt))

        final_response = client.generate(
            messages=local_messages_2,
            system_prompt=system_instruction,
        )
        
        state["recommendation_result"] = {
            "restaurants": restaurants,
            "query_params": query_params.model_dump()
        }
        state["final_response"] = final_response
        messages.append(AIMessage(content=final_response))
        state.setdefault("message_timestamps", []).append(datetime.utcnow().isoformat())
        return state

    except Exception as e:
        state["recommendation_result"] = None
        state["final_response"] = f"抱歉，推荐过程中出现错误：{str(e)}"
        messages.append(AIMessage(content=state["final_response"]))
        state.setdefault("message_timestamps", []).append(datetime.utcnow().isoformat())
        return state
