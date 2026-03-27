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
from langgraph_app.tools.map.ip_location import get_location_from_ip

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


async def food_recommendation_node(state: GraphState) -> NodeOutput:
    """
    Provide restaurant and food recommendations based on user query.
    """
    messages = state.get("messages", [])
    client = get_tracked_llm(
        module="food_recommendation", node_name="food_recommendation"
    )

    lang = get_dominant_language(messages)

    extraction_prompt = f"""Extract recommendation parameters from the user's query, considering the conversation history and any images provided. An image might give clues about the desired cuisine.

Your response must be a JSON object with the requested schema. Ensure your response understands the user's dominant language ('{lang}')."""

    try:
        # Step 1: Use local messages to append extraction instruction
        local_messages = messages.copy()
        local_messages.append(HumanMessage(content=extraction_prompt))

        structured_llm = client.with_structured_output(RecommendationQuery)
        query_params = None
        last_error_1 = None

        for attempt in range(3):
            try:
                sys_content = "You are an expert food recommendation assistant."
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

        # Currently default to IP location.
        # TODO: Update this to use explicit user location or frontend provided lat/lng
        ip_loc = get_location_from_ip()
        if ip_loc:
            final_lat, final_lng = ip_loc

        # Step 2: Get restaurant recommendations using actual tool
        raw_result = await search_restaurants_tool.ainvoke(
            {
                "location": None,  # Force no text location bias for now, rely purely on coordinates/ip
                "cuisine_type": query_params.cuisine_type,
                "radius_km": query_params.radius_km or 5.0,
                "lat": final_lat,
                "lng": final_lng,
                "max_results": query_params.count or 5,
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
        formatting_prompt = f"""Convert this restaurant list into a structured recommendation based on the user's query.

Restaurants:
{json.dumps(restaurants, ensure_ascii=False, indent=2)}

Your response must be a JSON object that conforms to the `Recommendation` schema. Base your summaries on the user's query and preferences. The entire response (including the localized label fields) must be in the specific language requested by the user. (Note: The user's overall conversational language is '{lang}', but if they explicitly asked for a different language for the response, you MUST follow their explicit request!)
"""

        system_instruction = f"You are a friendly food recommendation assistant. Provide helpful, personalized restaurant suggestions in a structured format. Your entire response should be in the user's dominant language ('{lang}'). However, if the user specifically asks for another language, please switch to that language."

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
