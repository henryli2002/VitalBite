"""Food recommendation agent for restaurant and food suggestions."""

from typing import Dict, Any, List
import json
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_app.utils.utils import (
    detect_language,
    get_current_user_text,
)


class RecommendationQuery(BaseModel):
    """Structured query parameters for recommendations."""
    location: str | None = None
    radius_km: float | None = None
    cuisine_type: str | None = Field(description="cuisine type or null. If an image of food is provided, infer the cuisine from it.")
    dietary_restrictions: List[str] = Field(default_factory=list)
    price_range: str | None = Field(description="budget|moderate|expensive or null")


def _get_restaurants_mock(
    location: str | None,
    cuisine_type: str | None,
    dietary_restrictions: List[str]
) -> List[Dict[str, Any]]:
    """
    Mock function to simulate restaurant search.
    """
    # Mock restaurant data
    mock_restaurants = [
        {
            "name": "健康素食餐厅",
            "cuisine": "素食",
            "address": "台北市大安区",
            "rating": 4.5,
            "price_range": "中等",
            "dietary_options": ["素食", "无麸质"]
        },
        {
            "name": "日式料理店",
            "cuisine": "日式",
            "address": "台北市信义区",
            "rating": 4.8,
            "price_range": "中高",
            "dietary_options": ["海鲜", "低卡"]
        },
        {
            "name": "地中海风味餐厅",
            "cuisine": "地中海",
            "address": "台北市中山区",
            "rating": 4.3,
            "price_range": "中等",
            "dietary_options": ["健康", "低钠"]
        }
    ]
    
    # Filter based on criteria (simplified)
    filtered = []
    for rest in mock_restaurants:
        if cuisine_type and cuisine_type.lower() not in rest["cuisine"].lower():
            continue
        if dietary_restrictions:
            # Check if restaurant supports any of the restrictions
            if not any(dr.lower() in " ".join(rest["dietary_options"]).lower() 
                      for dr in dietary_restrictions):
                continue
        filtered.append(rest)
    
    return filtered[:5]  # Return top 5


def food_recommendation_node(state: GraphState) -> GraphState:
    """
    Provide restaurant and food recommendations based on user query.
    """
    state = state.copy()
    messages = state.setdefault("messages", [])
    client = get_llm_client(module="food_recommendation")

    current_text = get_current_user_text(messages)
    lang = detect_language(current_text)

    extraction_prompt = f"""Extract recommendation parameters from the user's query, considering the conversation history and any images provided. An image might give clues about the desired cuisine.

Your response must be a JSON object with the requested schema. Ensure your response understands the user's language ('{lang}')."""

    try:
        # Step 1: Use local messages to append extraction instruction
        local_messages = messages.copy()
        local_messages.append(HumanMessage(content=extraction_prompt))

        query_params = client.generate_structured(
            messages=local_messages, 
            schema=RecommendationQuery,
            system_prompt="You are an expert food recommendation assistant."
        )
        
        # Step 2: Get restaurant recommendations (mock for now)
        restaurants = _get_restaurants_mock(
            query_params.location,
            query_params.cuisine_type,
            query_params.dietary_restrictions
        )
        
        if not restaurants:
            state["recommendation_result"] = {"restaurants": []}
            state["final_response"] = "抱歉，没有找到符合您要求的餐厅。请尝试调整搜索条件。" if lang == "Chinese" else "Sorry, no matching restaurants were found. Please try adjusting your search criteria."
            messages.append(AIMessage(content=state["final_response"]))
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
            f"You are a friendly food recommendation assistant. Provide helpful, personalized restaurant suggestions. Your entire response must be in {lang}."
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
        return state

    except Exception as e:
        state["recommendation_result"] = None
        state["final_response"] = f"抱歉，推荐过程中出现错误：{str(e)}"
        messages.append(AIMessage(content=state["final_response"]))
        return state
