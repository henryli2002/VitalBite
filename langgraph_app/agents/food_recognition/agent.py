"""Food recognition agent for identifying foods from images."""

from datetime import datetime
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langgraph_app.orchestrator.state import GraphState
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.utils import (
    get_dominant_language,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

    lang = get_dominant_language(messages)

    # Instead of parse_content_for_llm, we just let the LLM see the `messages`.
    # But we want to ensure it focuses on extracting data from the image.
    system_instruction = (
        "You are a nutrition expert. Analyze the latest food images provided by the user in the conversation and provide accurate nutritional information."
    )

    try:
        # Step 1: Extract structured data from the images in the conversation
        last_error = None
        recognition_data_obj = None
        
        structured_llm = client.with_structured_output(RecognitionResult)
        
        for attempt in range(3):
            try:
                current_system_instruction = system_instruction
                if last_error:
                    current_system_instruction += f"\n\nNOTE: Your previous attempt failed validation with this error: {str(last_error)}. Please correct your JSON output and ensure it strictly follows the schema."
                    
                messages_to_send = [SystemMessage(content=current_system_instruction)] + messages
                recognition_data_obj = structured_llm.invoke(messages_to_send, config={"callbacks": []})
                break
            except Exception as e:
                last_error = e
                print(f"Food recognition extraction failed on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    import time
                    time.sleep(1)
                    
        if recognition_data_obj is None:
            raise last_error or Exception("Failed to generate structured data after retries.")
        
        recognition_data = recognition_data_obj.model_dump() if hasattr(recognition_data_obj, "model_dump") else recognition_data_obj
        
        # Step 2: Create a natural language response
        final_prompt = f"""You are a helpful nutrition assistant. Your task is to interpret the following nutritional analysis data and answer the user's query in a conversational and easy-to-understand way.
        
        Nutritional Analysis (JSON):
        {json.dumps(recognition_data, indent=2)}
        
        Guidelines:
        - **Language**: Your entire response should be in the same language as the user's dominant language in the conversation, which is '{lang}'. However, if the user specifically asks for another language, please switch to that language.
        - **Synthesize, Don't Just Repeat**: Do not just list the data. Explain what it means in response to the user's question.
        - **Address the Query**: If the user asks a specific question, answer it directly.
        - **Summarize if General**: If the user's query is general, provide a summary of the key findings from the analysis, including the health assessment.
        
        Provide only the natural language response."""
        
        local_messages = messages.copy()
        local_messages.append(HumanMessage(content=final_prompt))
        
        messages_to_send_2 = [SystemMessage(content="You are a helpful nutrition assistant.")] + local_messages
        ai_message = client.invoke(messages_to_send_2)
        final_response = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        
        state["recognition_result"] = recognition_data
        state["final_response"] = final_response
        messages.append(AIMessage(content=final_response))
        state.setdefault("message_timestamps", []).append(datetime.utcnow().isoformat())

    except Exception as e:
        print(f"Error in food recognition node: {e}")
        state["recognition_result"] = None
        state["final_response"] = f"抱歉，食物识别过程中出现错误：{str(e)}" if lang == "Chinese" else f"Sorry, an error occurred during food recognition: {str(e)}"
        messages.append(AIMessage(content=state["final_response"]))
        state.setdefault("message_timestamps", []).append(datetime.utcnow().isoformat())

    return state
