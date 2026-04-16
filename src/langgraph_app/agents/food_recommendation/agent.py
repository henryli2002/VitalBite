"""Food recommendation agent for restaurant and food suggestions."""

from typing import Dict, Any, List
from datetime import datetime, timezone
import json
import asyncio
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.llm_callback import create_callback_handler
from langgraph_app.utils.llm_factory import inject_dynamic_context
from langgraph_app.utils.logger import get_logger
from langgraph_app.utils.retry import with_retry
from langgraph_app.config import config as _config
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langgraph_app.utils.utils import (
    get_dominant_language,
)
from langgraph_app.tools.tools import search_restaurants_tool
from langgraph_app.tools.map.ip_location import get_location_from_ip_async

logger = get_logger(__name__)


class RecommendationQuery(BaseModel):
    """Structured query parameters for recommendations."""

    radius_km: float | None = None
    cuisine_type: str | None = Field(
        description="cuisine type or null. If an image of food is provided, infer the cuisine from it."
    )
    dietary_restrictions: List[str] = Field(default_factory=list)
    price_range: str | None = Field(description="budget|moderate|expensive or null")
    count: int | None = Field(5, description="The number of restaurant recommendations requested by the user, default is 5")
    # TODO: Add logic to receive precise lat/lng from frontend after prompting user for location permission
    # lat: float | None = Field(None, description="latitude if provided in context")
    # lng: float | None = Field(None, description="longitude if provided in context")


class Restaurant(BaseModel):
    """Represents a single restaurant recommendation."""

    name: str = Field(description="Name of the restaurant.")
    address: str = Field(description="Address of the restaurant.")
    rating: float = Field(description="Rating of the restaurant.")
    user_ratings_total: int = Field(description="Total number of user ratings.")
    summary: str = Field(
        description="A short summary of why this restaurant is a good fit, highlighting aspects relevant to the user's query."
    )


class Recommendation(BaseModel):
    """A complete recommendation response, including a title, list of restaurants, and conclusion."""

    title: str = Field(
        description="A warm, friendly title for the recommendation list, acknowledging the user's request."
    )
    restaurants: List[Restaurant] = Field(
        description="A list of recommended restaurants."
    )
    conclusion: str = Field(
        description="A concluding remark or a friendly question to encourage further interaction."
    )
    label_address: str = Field(
        description="The localized word for 'Address', matching the language requested by the user."
    )
    label_rating: str = Field(
        description="The localized word for 'Rating', matching the language requested by the user."
    )
    label_reviews: str = Field(
        description="The localized word for 'reviews', matching the language requested by the user."
    )
    label_reason: str = Field(
        description="The localized word for 'Recommendation reason', matching the language requested by the user."
    )


from langgraph_app.utils.semaphores import with_semaphore

@with_semaphore("recommendation")
async def food_recommendation_node(state: GraphState) -> NodeOutput:
    """
    Provide restaurant and food recommendations based on user query.
    """
    messages = state.get("messages", [])
    client = get_llm_client(module="food_recommendation")

    lang = get_dominant_language(messages)

    user_profile = state.get("user_profile")
    profile_context = ""
    if user_profile:
        profile_context = "\n\nUser Profile & Health Information:\n" + "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}" for k, v in user_profile.items() if v
        )

    extraction_prompt = f"""[OBJECTIVE]
Extract restaurant search parameters from the user's query and conversation history.

[CONTEXT]
User Image (if any) might implicitly suggest a cuisine.

[CONSTRAINTS]
1. SCHEMA: Output exactly matching the JSON schema.
2. LANGUAGE: Understand the user's language ('{lang}') but output standard parameters."""

    try:
        # Step 1: Use local messages to append extraction instruction
        local_messages = messages.copy()
        local_messages.append(HumanMessage(content=extraction_prompt))

        structured_llm = client.with_structured_output(RecommendationQuery)
        sys_content = f"""[ROLE]
You are WABI, an expert food recommendation assistant.

[CONTEXT]{profile_context}"""

        query_params = await with_retry(
            lambda: structured_llm.ainvoke(
                inject_dynamic_context([SystemMessage(content=sys_content)] + local_messages),
                config={"callbacks": [create_callback_handler("food_recommendation_query")]},
            ),
            attempts=3,
            base=0.3,
            cap=5.0,
            fallback=None,
        )

        if query_params is None:
            raise Exception("Failed to generate RecommendationQuery after retries.")

        # Determine location fallback
        final_lat = None
        final_lng = None
        
        # 1. Try to get precise location from user_context (frontend Geolocation API)
        user_context = state.get("user_context") or {}
        if user_context.get("lat") is not None and user_context.get("lng") is not None:
            final_lat = user_context.get("lat")
            final_lng = user_context.get("lng")
            logger.info(f"[{state.get('user_id')}] Using precise frontend location: lat={final_lat}, lng={final_lng}")
        else:
            # 2. Fallback to IP-based location
            client_ip = user_context.get("user_ip")
            logger.info(f"[{state.get('user_id')}] No frontend location. Falling back to IP-based location for IP: {client_ip}")
            ip_loc = await get_location_from_ip_async(client_ip)
            if ip_loc:
                final_lat, final_lng = ip_loc
                logger.info(f"[{state.get('user_id')}] Obtained IP location: lat={final_lat}, lng={final_lng}")

        # Step 2: Get restaurant recommendations using actual tool
        raw_result = await search_restaurants_tool.ainvoke(
            {
                "location": None,  # Force no text location bias for now, rely purely on coordinates/ip
                "cuisine_type": query_params.cuisine_type, # Restore user preference handling
                "radius_km": query_params.radius_km or 5.0,
                "lat": final_lat,
                "lng": final_lng,
                "max_results": max(20, query_params.count or 20), # Keep 20 to allow LLM ample choice
            }
        )

        try:
            result_dict = json.loads(raw_result)
            restaurants = result_dict.get("restaurants", [])
        except Exception:
            restaurants = []

        if not restaurants:
            error_msg = (
                "抱歉，没有找到符合您要求的餐厅。请尝试调整搜索条件。"
                if lang == "Chinese"
                else "Sorry, no matching restaurants were found. Please try adjusting your search criteria."
            )
            return {
                "recommendation_result": {"restaurants": []},
                "messages": [AIMessage(content=error_msg, additional_kwargs={"timestamp": datetime.now(timezone.utc).isoformat()})],
            }

        # Step 3: Format recommendations into a structured object
        formatting_prompt = f"""[DATA]
Restaurants:
{json.dumps(restaurants, ensure_ascii=False, indent=2)}

[TASK]
Convert this raw restaurant list into a structured JSON `Recommendation` based on the user's query."""

        system_instruction = f"""[ROLE]
You are WABI, an expert food curator.

[OBJECTIVE]
Transform raw restaurant data into helpful, personalized suggestions.

[CONTEXT]{profile_context}

[CONSTRAINTS]
1. HEALTH-CONSCIOUS SELECTION: From the provided list, select the best 5 restaurants for a health-conscious user. If the user asks for a different number, follow that number. Prioritize options that are generally considered light and healthy.
2. EXPLICIT REASONING: In the `summary` for each restaurant, you MUST include a brief explanation of why it's a healthy choice (e.g., "features grilled options," "known for fresh salads," "offers light and fresh dishes").
3. HEALTH-AWARE CONCLUSION: In the `conclusion` field, add a gentle, encouraging healthy eating tip. If the user's request was for something less healthy (e.g., burgers, fried food), the tone should be encouraging and suggest moderation without being preachy.
4. PERSONALIZATION: CRITICAL - Actively reference the 'User Profile'. Highlight why these restaurants fit their goals, and explicitly warn if a restaurant conflicts with any allergies/diets!
5. SCHEMA: Output strictly matching the requested JSON schema.
6. LANGUAGE: The user's language is '{lang}'. Strictly abide by any explicit language requests from the user."""

        local_messages_2 = messages.copy()
        local_messages_2.append(HumanMessage(content=formatting_prompt))

        structured_llm_2 = client.with_structured_output(Recommendation)
        structured_response = await with_retry(
            lambda: structured_llm_2.ainvoke(
                inject_dynamic_context([SystemMessage(content=system_instruction)] + local_messages_2),
                config={"callbacks": [create_callback_handler("food_recommendation")], "tags": ["final_node_output"]},
            ),
            attempts=3,
            base=0.3,
            cap=5.0,
            fallback=None,
        )

        if structured_response is None:
            raise Exception("Failed to generate Recommendation after retries.")

        # Step 4: Convert the structured response to a formatted markdown string
        markdown_response = f"### {structured_response.title}\n\n"
        for r in structured_response.restaurants:
            markdown_response += f"#### {r.name}\n"
            markdown_response += (
                f"- **{structured_response.label_address}**: {r.address}\n"
            )
            markdown_response += f"- **{structured_response.label_rating}**: {r.rating} ({r.user_ratings_total} {structured_response.label_reviews})\n"
            markdown_response += (
                f"- **{structured_response.label_reason}**: {r.summary}\n\n"
            )
        markdown_response += f"{structured_response.conclusion}"

        ts = datetime.now(timezone.utc).isoformat()
        return {
            "recommendation_result": {
                "restaurants": restaurants,
                "query_params": query_params.model_dump()
                if hasattr(query_params, "model_dump")
                else {},
            },
            "messages": [AIMessage(content=markdown_response, additional_kwargs={"timestamp": ts})],
        }

    except Exception as e:
        error_msg = (
            f"抱歉，推荐过程中出现错误：{str(e)}"
            if lang == "Chinese"
            else f"Sorry, an error occurred during recommendation: {str(e)}"
        )
        return {
            "recommendation_result": None,
            "messages": [AIMessage(content=error_msg, additional_kwargs={"timestamp": datetime.now(timezone.utc).isoformat()})],
        }
