"""Food recommendation agent for restaurant and food suggestions."""

from typing import Dict, Any, List
import json
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langgraph_app.config import config


class RecommendationQuery(BaseModel):
    """Structured query parameters for recommendations."""
    location: str | None = None
    radius_km: float | None = None
    cuisine_type: str | None = None
    dietary_restrictions: List[str] = []
    price_range: str | None = None  # e.g., "budget", "moderate", "expensive"


def _get_restaurants_mock(
    location: str | None,
    cuisine_type: str | None,
    dietary_restrictions: List[str]
) -> List[Dict[str, Any]]:
    """
    Mock function to simulate restaurant search.
    
    In production, this would call Google Maps API or similar service.
    
    Args:
        location: Location string
        cuisine_type: Type of cuisine
        dietary_restrictions: List of dietary restrictions
        
    Returns:
        List of restaurant dictionaries
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
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with recommendation results and final response
    """
    state = state.copy()
    state.setdefault("messages", [])
    client = get_llm_client()
    input_data = state.get("input", {})
    text = input_data.get("text", "")
    
    # Step 1: Extract query parameters
    messages = state.get("messages", [])
    history_text = ""
    if messages:
        history_count = config.get_history_count("recommendation")
        relevant_msgs = messages[-history_count:-1] if len(messages) > 1 else []
        for msg in relevant_msgs:
            role = "User"
            content = str(msg.content)
            if hasattr(msg, "type"):
                if msg.type == "ai":
                    role = "AI"
                elif msg.type == "human":
                    role = "User"
            history_text += f"{role}: {content}\n"

    extraction_prompt = f"""Extract recommendation parameters from the user's query, considering the conversation history context.

Conversation History:
{history_text}

Current User Query: {text}

Extract the following information and respond in JSON format:
{{
    "location": "location string or null",
    "radius_km": number or null,
    "cuisine_type": "cuisine type or null",
    "dietary_restrictions": ["list", "of", "restrictions"],
    "price_range": "budget|moderate|expensive or null"
}}

Always respond in the same language as the user (Chinese if Chinese detected, otherwise English)."""

    try:
        query_params = client.generate_structured(extraction_prompt, RecommendationQuery)
        
        # Step 2: Get restaurant recommendations (mock for now)
        restaurants = _get_restaurants_mock(
            query_params.location,
            query_params.cuisine_type,
            query_params.dietary_restrictions
        )
        
        if not restaurants:
            state["recommendation_result"] = {"restaurants": []}
            state["final_response"] = "抱歉，没有找到符合您要求的餐厅。请尝试调整搜索条件。"
            return state
        
        # Step 3: Format recommendations into natural language
        formatting_prompt = f"""Convert this restaurant list into a friendly, natural language recommendation.

Restaurants:
{json.dumps(restaurants, ensure_ascii=False, indent=2)}

User's original query: {text}

Provide a warm, helpful response that:
1. Acknowledges the user's request
2. Lists the recommended restaurants with key details
3. Mentions why each restaurant might be a good fit
4. Keeps it concise and friendly

Always respond in the same language as the user (Chinese if Chinese detected, otherwise English)."""

        system_instruction = (
            "You are a friendly food recommendation assistant. Provide helpful, personalized restaurant suggestions. "
            "Always respond in the same language as the user (Chinese if Chinese detected, otherwise English)."
        )
        final_response = client.generate_text(
            formatting_prompt,
            system_instruction=system_instruction,
        )
        state["recommendation_result"] = {
            "restaurants": restaurants,
            "query_params": query_params.model_dump()
        }
        state["final_response"] = final_response
        state["messages"].append(AIMessage(content=final_response))
        return state


    except Exception as e:
        state["recommendation_result"] = None
        state["final_response"] = f"抱歉，推荐过程中出现错误：{str(e)}"
        state["messages"].append(AIMessage(content=state["final_response"]))
        return state
        
