"""
Unified food recognition agent that handles both single-image and spatial measurement
workflows, using a RAG-based approach for high accuracy.
"""
import json
from datetime import datetime
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.tools.nutrition.fndds import fndds_nutrition_search_tool
# Import both sizing tools
from langgraph_app.tools.vision.spatial import estimate_volume_from_two_images_tool
# from langgraph_app.tools.vision.single_image import estimate_size_from_single_image_tool # No longer needed
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from pydantic import BaseModel, Field
from typing import List
from langgraph_app.utils.utils import get_dominant_language

# --- Pydantic Schemas ---
class FoodItem(BaseModel):
    """Describes a single food item with its estimated weight."""
    food_name: str = Field(description="The name of the food item.")
    estimated_weight_g: float = Field(description="The estimated weight of the food item in grams.")

class FoodAnalysis(BaseModel):
    """Describes all food items found in an image."""
    foods: List[FoodItem] = Field(description="A list of all food items identified in the image, each with an estimated weight.")


def recognition_node(state: GraphState) -> NodeOutput:
    """
    A unified node that performs food recognition. It internally decides which
    sizing method to use (LLM direct estimation or two-image spatial measurement)
    based on the intent routed from the previous node.
    """
    messages = state.get("messages", [])
    client = get_llm_client(module="food_recognition_rag")
    lang = get_dominant_language(messages)
    
    # Get the intent from the analysis layer of the state
    intent = state.get("analysis", {}).get("intent")
    
    identified_foods: List[FoodItem] = []

    # --- Step 1: LLM identifies food and estimates weight directly ---
    print("Step 1: MLLM analyzing image to identify food and estimate weight...")
    last_error = None
    structured_llm = client.with_structured_output(FoodAnalysis)
    
    for attempt in range(3):
        try:
            system_prompt = """You are an expert nutritionist and portion size estimator. Analyze the latest user-provided image. 
            For each food item you identify, provide both its name and a realistic, CONSERVATIVE estimation of its weight in grams.
            IMPORTANT: Tend to underestimate rather than overestimate portion sizes. Many food items appear larger in close-up photos than they actually are. Rely on standard, average serving sizes for your baseline.
            CRITICAL: The `food_name` MUST be in English, regardless of the user's language, because it will be used to search an English nutritional database. For example, if you see an apple, output 'apple', not '苹果'."""
            
            if last_error:
                system_prompt += f"\n\nNOTE: Your previous attempt failed validation with this error: {str(last_error)}. Please correct your JSON output and ensure it strictly follows the schema."
            
            messages_to_send = [SystemMessage(content=system_prompt)] + messages
            food_analysis_obj = structured_llm.invoke(messages_to_send, config={"callbacks": []})
            identified_foods = food_analysis_obj.get("foods", []) if isinstance(food_analysis_obj, dict) else food_analysis_obj.foods
            print(f"LLM Food Analysis: {identified_foods}")
            break
        except Exception as e:
            last_error = e
            print(f"Error in RAG node (Step 1 - MLLM Analysis) attempt {attempt + 1}: {e}")
            if attempt < 2:
                import time
                time.sleep(1)
                
    if not identified_foods and last_error:
        print(f"Ultimately failed RAG node (Step 1 - MLLM Analysis): {last_error}")
        error_message = f"抱歉，分析食物图片时出错：{last_error}" if lang == "Chinese" else f"Sorry, an error occurred while analyzing the food image: {last_error}"
        return {
            "messages": [AIMessage(content=error_message)],
            "message_timestamps": [datetime.utcnow().isoformat()]
        }

    if not identified_foods:
        no_food_message = "图片中未识别到食物。" if lang == "Chinese" else "No food items were identified in the image."
        return {
            "messages": [AIMessage(content=no_food_message)],
            "message_timestamps": [datetime.utcnow().isoformat()]
        }
        
    # --- Step 2: Use RAG tool to get nutritional info ---
    print("Step 2: Retrieving nutritional info from FNDDS via RAG tool...")
    all_nutrition_results = []
    for food_item in identified_foods:
        try:
            # We request the top 3 results from the tool to give the final LLM options
            tool_result_json = fndds_nutrition_search_tool.invoke({"food_description": food_item.food_name, "top_k": 3})
            tool_result = json.loads(tool_result_json)
            all_nutrition_results.append({
                "identified_name": food_item.food_name,
                "estimated_weight_g": food_item.estimated_weight_g,
                "potential_matches": tool_result
            })
        except Exception as e:
            print(f"Error calling FNDDS tool for '{food_item.food_name}': {e}")
            all_nutrition_results.append({"error": str(e), "identified_name": food_item.food_name})

    recognition_result = {"final_analysis": all_nutrition_results}
    
    # --- Step 3: LLM summarizes all results ---
    print("Step 3: LLM summarizing all information and selecting the best match...")
    summary_prompt = f"""You are a helpful nutrition assistant. We have analyzed the user's meal.
    Here is the data, including the food identified from the image, its estimated weight, and a list of POTENTIAL matches from our nutritional database (per 100g):
    {json.dumps(all_nutrition_results, indent=2, ensure_ascii=False)}

    Your task is to synthesize this into a single, easy-to-understand summary.
    
    CRITICAL INSTRUCTIONS:
    1.  **Acknowledge the Meal FIRST**: You MUST begin your response by explicitly stating what food items you have identified from the image and their estimated weights. For example: "I see you have an apple (estimated 150g) and a cup of coffee (estimated 250g)."
        - IMPORTANT: You MUST translate the English food names back into the user's language (e.g., '{lang}') for this opening sentence.
    2.  **Select the Best Match**: For each identified food, review the `potential_matches`. Some matches might be wildly incorrect (e.g., matching 'tea' to 'Long Island Iced Tea' which has alcohol). **Refer back to the original image** and use your common sense to **choose the candidate that visually and contextually best matches the food in the picture**.
    3.  **Calculate Totals**: Once you've selected the best match, adjust its nutritional values based on the 'estimated_weight_g'. For example, if an item's estimated weight is 150g, calculate 1.5 times its per-100g nutritional values.
    4.  **Sum and Summarize**: Sum up the total nutrition for the entire meal based on your selected matches.
    5.  Provide a final health assessment and a friendly tip.
    6.  **Language**: Your entire response should be in the same language as the user's dominant language in the conversation, which is '{lang}'. However, if the user specifically asks for another language, please switch to that language.
    7.  **Tone & Perspective**: Present the final nutritional information as YOUR OWN expert analysis. Do NOT say things like "According to the database..." or "Based on my lookup...". Speak as a unified, intelligent nutritionist.

    Provide only the final, conversational response to the user. Do not explain your matching process unless asked.
    """
    
    try:
        messages_to_send_3 = [SystemMessage(content="You are a helpful nutrition assistant.")] + messages + [HumanMessage(content=summary_prompt)]
        ai_message = client.invoke(messages_to_send_3, config={"tags": ["final_node_output"]})
        msg: AnyMessage = ai_message # type: ignore
        return {
            "recognition_result": recognition_result,
            "messages": [msg],
            "message_timestamps": [datetime.utcnow().isoformat()]
        }
    except Exception as e:
        print(f"Error in RAG node (Step 3 - Summary): {e}")
        error_message = f"抱歉，总结营养信息时出错：{e}" if lang == "Chinese" else f"Sorry, an error occurred while summarizing the nutritional information: {e}"
        return {
            "recognition_result": recognition_result,
            "messages": [AIMessage(content=error_message)],
            "message_timestamps": [datetime.utcnow().isoformat()]
        }
