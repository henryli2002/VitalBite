"""Food recommendation agent for restaurant and food suggestions."""

from typing import Dict, Any, List
from datetime import datetime
import json
import asyncio
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.tracked_llm import get_tracked_llm
from langgraph_app.utils.logger import get_logger
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
    client = get_tracked_llm(
        module="food_recommendation", node_name="food_recommendation"
    )

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
        query_params = None
        last_error_1 = None

        for attempt in range(3):
            try:
                sys_content = f"""[ROLE]
You are WABI, an expert food recommendation assistant.

[CONTEXT]{profile_context}"""
                if last_error_1:
                    sys_content += f"\n\nNOTE: Your previous attempt failed validation with this error: {str(last_error_1)}. Please correct your JSON output and ensure it strictly follows the schema."

                query_params = await structured_llm.ainvoke(
                    [SystemMessage(content=sys_content)] + local_messages,
                    config={"callbacks": []},
                )
                break
            except Exception as e:
                last_error_1 = e
                logger.warning(
                    f"[food_recommendation] Extraction failed on attempt {attempt + 1}: {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(1)

        if query_params is None:
            raise last_error_1 or Exception(
                "Failed to generate RecommendationQuery after retries."
            )

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
                "max_results": max(15, query_params.count or 15), # Keep 15 to allow LLM ample choice
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
                "messages": [AIMessage(content=error_msg)],
                "message_timestamps": [datetime.utcnow().isoformat()],
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
1. PERSONALIZATION: CRITICAL - Actively reference the 'User Profile'. Highlight why these restaurants fit their goals, and explicitly warn if a restaurant conflicts with any allergies/diets!
2. SCHEMA: Output strictly matching the requested JSON schema.
3. LANGUAGE: The user's language is '{lang}'. Strictly abide by any explicit language requests from the user."""

        local_messages_2 = messages.copy()
        local_messages_2.append(HumanMessage(content=formatting_prompt))

        structured_llm_2 = client.with_structured_output(Recommendation)
        structured_response = None
        last_error_2 = None

        for attempt in range(3):
            try:
                sys_content_2 = system_instruction
                if last_error_2:
                    sys_content_2 += f"\n\nNOTE: Your previous attempt failed validation with this error: {str(last_error_2)}. Please correct your JSON output and ensure it strictly follows the schema."

                structured_response = await structured_llm_2.ainvoke(
                    [SystemMessage(content=sys_content_2)] + local_messages_2,
                    config={"tags": ["final_node_output"]},
                )
                break
            except Exception as e:
                last_error_2 = e
                logger.warning(
                    f"[food_recommendation] Formatting failed on attempt {attempt + 1}: {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(1)

        if structured_response is None:
            raise last_error_2 or Exception(
                "Failed to generate Recommendation after retries."
            )

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

        return {
            "recommendation_result": {
                "restaurants": restaurants,
                "query_params": query_params.model_dump()
                if hasattr(query_params, "model_dump")
                else {},
            },
            "messages": [AIMessage(content=markdown_response)],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }

    except Exception as e:
        error_msg = (
            f"抱歉，推荐过程中出现错误：{str(e)}"
            if lang == "Chinese"
            else f"Sorry, an error occurred during recommendation: {str(e)}"
        )
        return {
            "recommendation_result": None,
            "messages": [AIMessage(content=error_msg)],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }
