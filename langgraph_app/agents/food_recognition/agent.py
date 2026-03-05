"""Food recognition agent for identifying foods from images."""

import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.utils import (
    detect_language,
    get_current_user_text,
)
from langchain_core.messages import AIMessage, HumanMessage

class FoodItem(BaseModel):
    name: str = Field(description="food item name")
    estimated_weight_grams: float = Field(description="estimated weight in grams")
    estimated_calories: float = Field(description="estimated calories")
    confidence: float = Field(description="confidence between 0.0-1.0")

class NutritionData(BaseModel):
    total_calories: float = Field(description="total calories")
    total_protein_grams: float = Field(description="total protein in grams")
    total_carbs_grams: float = Field(description="total carbohydrates in grams")
    total_fat_grams: float = Field(description="total fat in grams")
    health_score: float = Field(description="health score 0.0-10.0 (10 is very healthy)")

class RecognitionResult(BaseModel):
    foods: List[FoodItem]
    nutrition: NutritionData
    health_assessment: str = Field(description="brief text assessment of the meal's healthiness")

def food_recognition_node(state: GraphState) -> GraphState:
    """
    Recognize foods from image and provide nutritional information.
    """
    state = state.copy()
    messages = state.setdefault("messages", [])
    client = get_llm_client(module="food_recognition")

    current_text = get_current_user_text(messages)
    lang = detect_language(current_text)

    # Instead of parse_content_for_llm, we just let the LLM see the `messages`.
    # But we want to ensure it focuses on extracting data from the image.
    system_instruction = (
        "You are a nutrition expert. Analyze the latest food images provided by the user in the conversation and provide accurate nutritional information."
    )

    try:
        # Step 1: Extract structured data from the images in the conversation
        recognition_data_obj = client.generate_structured(
            messages=messages,
            schema=RecognitionResult,
            system_prompt=system_instruction,
        )
        
        recognition_data = recognition_data_obj.model_dump()
        
        # Step 2: Create a natural language response
        final_prompt = f"""You are a helpful nutrition assistant. Your task is to interpret the following nutritional analysis data and answer the user's query in a conversational and easy-to-understand way.

Nutritional Analysis (JSON):
{json.dumps(recognition_data, indent=2)}

Guidelines:
- **Language**: Your entire response must be in the same language as the user's query, which is '{lang}'.
- **Synthesize, Don't Just Repeat**: Do not just list the data. Explain what it means in response to the user's question.
- **Address the Query**: If the user asks a specific question, answer it directly.
- **Summarize if General**: If the user's query is general, provide a summary of the key findings from the analysis, including the health assessment.

Provide only the natural language response."""

        # Use a local copy of messages to append the instructions without saving to state
        local_messages = messages.copy()
        local_messages.append(HumanMessage(content=final_prompt))

        final_response = client.generate(
            messages=local_messages,
            system_prompt="You are a helpful nutrition assistant."
        )
        
        state["recognition_result"] = recognition_data
        state["final_response"] = final_response
        messages.append(AIMessage(content=final_response))

    except Exception as e:
        print(f"Error in food recognition node: {e}")
        state["recognition_result"] = None
        state["final_response"] = f"抱歉，食物识别过程中出现错误：{str(e)}" if lang == "Chinese" else f"Sorry, an error occurred during food recognition: {str(e)}"
        messages.append(AIMessage(content=state["final_response"]))

    return state
