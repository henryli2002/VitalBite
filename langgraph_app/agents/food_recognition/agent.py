"""Food recognition agent for identifying foods from images."""

from typing import Dict, Any
import json
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.gemini_client import GeminiClient
from langchain_core.messages import AIMessage
from langgraph_app.utils.utils import detect_language
from langgraph_app.config import config

def food_recognition_node(state: GraphState) -> GraphState:
    """
    Recognize foods from image and provide nutritional information.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with recognition results and final response
    """
    state = state.copy()
    client = GeminiClient()
    input_data = state.get("input", {})
    image_data = input_data.get("image_data")
    text = input_data.get("text", "")

    lang = detect_language(text)
    
    if not image_data:
        state["recognition_result"] = None
        state["final_response"] = "抱歉，未检测到图像数据，无法进行食物识别。" if lang == "Chinese" else "Sorry, no image data detected for food recognition."
        state["messages"].append(AIMessage(content=state["final_response"]))
        return state
    
    # Construct vision prompt for food recognition
    messages = state.get("messages", [])
    history_text = ""
    if messages:
        history_count = config.get_history_count("recognition")
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

    vision_prompt = f"""Analyze this image and identify all food items present, considering any relevant conversation history.

Conversation History:
{history_text}

User query: {text}

Please provide a detailed analysis in JSON format with the following structure:
{{
    "foods": [
        {{
            "name": "food item name",
            "estimated_weight_grams": number,
            "estimated_calories": number,
            "confidence": float (0.0-1.0)
        }}
    ],
    "nutrition": {{
        "total_calories": number,
        "total_protein_grams": number,
        "total_carbs_grams": number,
        "total_fat_grams": number,
        "health_score": float (0.0-10.0, where 10 is very healthy)
    }},
    "health_assessment": "brief text assessment of the meal's healthiness"
}}
Use the following guidelines:
- Be as accurate as possible with estimates. If you cannot identify certain items, mark confidence as low.
- Use same language as user input for any text in the response."""

    try:
        # Use vision generation
        response_text = client.generate_vision(
            image_data,
            vision_prompt,
            system_instruction="You are a nutrition expert. Analyze food images and provide accurate nutritional information."
        )
        
        # Try to parse JSON from response
        # The response might be wrapped in markdown code blocks
        json_text = response_text.strip()
        if json_text.startswith("```"):
            # Extract JSON from code block
            lines = json_text.split("\n")
            json_text = "\n".join([line for line in lines if not line.strip().startswith("```")])
        
        recognition_data = json.loads(json_text)
        
        # Format natural language response
        foods_list = recognition_data.get("foods", [])
        nutrition = recognition_data.get("nutrition", {})
        health_assessment = recognition_data.get("health_assessment", "")
        
        response_parts = ["我识别到以下食物：\n"] if lang == "Chinese" else ["I have identified the following foods:\n"]
        for food in foods_list:
            name = food.get("name", "未知食物" if lang == "Chinese" else "Unknown food")
            weight = food.get("estimated_weight_grams", 0)
            calories = food.get("estimated_calories", 0)
            if lang == "Chinese":
                response_parts.append(f"- {name}: 约 {weight}g, {calories} 卡路里")
            else:
                response_parts.append(f"- {name}: approximately {weight}g, {calories} calories")
        
        response_parts.append(f"\n营养总览：" if lang == "Chinese" else "\nNutrition Overview:")
        response_parts.append(f"- 总卡路里: {nutrition.get('total_calories', 0)}" if lang == "Chinese" else f"- Total Calories: {nutrition.get('total_calories', 0)}")
        response_parts.append(f"- 蛋白质: {nutrition.get('total_protein_grams', 0)}g" if lang == "Chinese" else f"- Protein: {nutrition.get('total_protein_grams', 0)}g")
        response_parts.append(f"- 碳水化合物: {nutrition.get('total_carbs_grams', 0)}g" if lang == "Chinese" else f"- Carbohydrates: {nutrition.get('total_carbs_grams', 0)}g")
        response_parts.append(f"- 脂肪: {nutrition.get('total_fat_grams', 0)}g" if lang == "Chinese" else f"- Fat: {nutrition.get('total_fat_grams', 0)}g")
        response_parts.append(f"- 健康评分: {nutrition.get('health_score', 0)}/10" if lang == "Chinese" else f"- Health Score: {nutrition.get('health_score', 0)}/10")
        
        if health_assessment:
            response_parts.append(f"\n健康评估: {health_assessment}" if lang == "Chinese" else f"\nHealth Assessment: {health_assessment}")
        
        final_response = "\n".join(response_parts)
        state["recognition_result"] = recognition_data
        state["final_response"] = final_response
        state["messages"].append(AIMessage(content=final_response))
        
        return state
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, use the raw response
        state["recognition_result"] = None
        state["final_response"] = f"抱歉，无法解析识别结果：{str(e)}。系统回复内容：{response_text}" if lang == "Chinese" else f"Sorry, unable to parse recognition result: {str(e)}. System response: {response_text}"
        state["messages"].append(AIMessage(content=state["final_response"]))
        return state
    except Exception as e:
        state["recognition_result"] = None
        state["final_response"] = f"抱歉，食物识别过程中出现错误：{str(e)}" if lang == "Chinese" else f"Sorry, an error occurred during food recognition: {str(e)}"
        state["messages"].append(AIMessage(content=state["final_response"]))
        return state
