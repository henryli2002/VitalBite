"""
Unified food recognition agent that handles both single-image and spatial measurement
workflows, using a RAG-based approach for high accuracy.
"""

import json
import time
from datetime import datetime
from langgraph_app.utils.logger import get_logger
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.tracked_llm import get_tracked_llm
from langgraph_app.tools.nutrition.fndds import fndds_nutrition_search_tool

# from langgraph_app.tools.vision.spatial import estimate_volume_from_two_images_tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from pydantic import BaseModel, Field
from typing import List
from langgraph_app.utils.utils import get_dominant_language

logger = get_logger(__name__)


# --- Pydantic Schemas ---
class FoodItem(BaseModel):
    """Describes a single food item identified from image."""

    food_name: str = Field(description="The name of the food item in English.")


class FoodAnalysis(BaseModel):
    """Describes all food items found in an image."""

    foods: List[FoodItem] = Field(
        description="A list of all food items identified in the image."
    )


class PortionEstimate(BaseModel):
    """Estimates how many standard portions the user ate."""

    food_name: str = Field(description="The name of the food item.")
    num_portions: float = Field(
        description="Estimated number of standard portions the user ate. For '1 piece' type foods, this is the number of pieces."
    )


class PortionAnalysis(BaseModel):
    """Estimates portions for all identified foods."""

    portions: List[PortionEstimate] = Field(
        description="List of foods with estimated number of portions."
    )


def recognition_node(state: GraphState) -> NodeOutput:
    """
    A unified node that performs food recognition with structured RAG workflow.

    Steps:
    1. LLM identifies food names from image
    2. RAG retrieves nutritional info + standard portion weights
    3. LLM estimates number of portions
    4. Python calculates actual weights
    5. LLM generates final summary
    """
    messages = state.get("messages", [])
    client = get_tracked_llm(
        module="food_recognition_rag", node_name="food_recognition"
    )
    lang = get_dominant_language(messages)

    intent = state.get("analysis", {}).get("intent")
    step_metrics = []

    identified_foods: List[FoodItem] = []
    step_start = time.time()
    last_error = None
    structured_llm = client.with_structured_output(FoodAnalysis)

    # --- Step 1: LLM identifies food names ---
    logger.info("Step 1: Identifying food names from image...")
    for attempt in range(3):
        try:
            system_prompt = """You are an expert nutritionist. Analyze the latest user-provided image and identify all food items.
            For each food item, provide ONLY its name in English.
            CRITICAL: The `food_name` MUST be in English, regardless of the user's language, because it will be used to search an English nutritional database."""

            if last_error:
                system_prompt += f"\n\nNOTE: Your previous attempt failed validation with this error: {str(last_error)}. Please correct your JSON output and ensure it strictly follows the schema."

            messages_to_send = [SystemMessage(content=system_prompt)] + messages
            food_analysis_obj = structured_llm.invoke(
                messages_to_send, config={"callbacks": []}
            )
            identified_foods = (
                food_analysis_obj.get("foods", [])
                if isinstance(food_analysis_obj, dict)
                else food_analysis_obj.foods
            )
            logger.info(f"Step 1 complete: {identified_foods}")
            break
        except Exception as e:
            last_error = e
            logger.error(f"Step 1 attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(1)

    step_time = time.time() - step_start
    step_metrics.append(
        {"step": 1, "name": "identify_foods", "time_seconds": round(step_time, 2)}
    )

    if not identified_foods and last_error:
        logger.error(f"Step 1 ultimately failed: {last_error}")
        error_message = (
            f"抱歉，分析食物图片时出错：{last_error}"
            if lang == "Chinese"
            else f"Sorry, an error occurred while analyzing the food image: {last_error}"
        )
        return {
            "messages": [AIMessage(content=error_message)],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }

    if not identified_foods:
        no_food_message = (
            "图片中未识别到食物。"
            if lang == "Chinese"
            else "No food items were identified in the image."
        )
        return {
            "messages": [AIMessage(content=no_food_message)],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }

    # --- Step 2: RAG retrieves nutritional info ---
    step_start = time.time()
    logger.info("Step 2: Retrieving nutritional info from FNDDS...")
    rag_results = []
    for food_item in identified_foods:
        try:
            tool_result_json = fndds_nutrition_search_tool.invoke(
                {"food_description": food_item.food_name, "top_k": 3}
            )
            tool_result = json.loads(tool_result_json)

            standard_portion_weight = None
            standard_portion_desc = None
            for match in tool_result:
                if match.get("standard_portion_weight_g"):
                    standard_portion_weight = match["standard_portion_weight_g"]
                    standard_portion_desc = match.get(
                        "standard_portion_description", ""
                    )
                    break

            rag_results.append(
                {
                    "food_name": food_item.food_name,
                    "standard_portion_weight_g": standard_portion_weight,
                    "standard_portion_description": standard_portion_desc,
                    "potential_matches": tool_result,
                }
            )
        except Exception as e:
            logger.error(f"Step 2 FNDDS error for '{food_item.food_name}': {e}")
            rag_results.append(
                {
                    "food_name": food_item.food_name,
                    "error": str(e),
                }
            )

    step_time = time.time() - step_start
    step_metrics.append(
        {"step": 2, "name": "rag_retrieval", "time_seconds": round(step_time, 2)}
    )
    logger.info(f"Step 2 complete: {len(rag_results)} foods processed")

    # --- Step 2.5: LLM estimates portions ---
    step_start = time.time()
    logger.info("Step 2.5: Estimating portions...")
    portion_llm = client.with_structured_output(PortionAnalysis)

    portion_prompt = f"""Analyze the user's meal image and estimate the number of pieces/portions.

IMPORTANT:
The database provides the weight for ONE piece/slice/fry, NOT for a full serving.

{rag_results}

For each food above:
1. Check the "standard_portion_description" - it tells you the weight of ONE unit
2. Look at the image and count how many pieces you see
3. The number you count is your answer (num_portions)

Note: A typical serving of "1 piece" foods (fries, chips, etc.) is often 10-30 pieces. 
A typical serving of "1 cup" foods is 0.5-2 cups.

Provide num_portions as a decimal number (e.g., 15.0 for 15 pieces, 1.5 for 1.5 cups)."""

    portion_estimates = []
    for attempt in range(3):
        try:
            messages_for_portion = [SystemMessage(content=portion_prompt)] + messages
            portion_obj = portion_llm.invoke(
                messages_for_portion, config={"callbacks": []}
            )
            portion_estimates = (
                portion_obj.get("portions", [])
                if isinstance(portion_obj, dict)
                else portion_obj.portions
            )
            logger.info(f"Step 2.5 complete: {portion_estimates}")
            break
        except Exception as e:
            logger.error(f"Step 2.5 attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(1)

    step_time = time.time() - step_start
    step_metrics.append(
        {"step": 2.5, "name": "estimate_portions", "time_seconds": round(step_time, 2)}
    )

    # --- Step 3: Calculate actual weights ---
    step_start = time.time()
    logger.info("Step 3: Calculating actual weights...")
    all_nutrition_results = []
    portion_lookup = {p.food_name: p.num_portions for p in portion_estimates}

    for rag_result in rag_results:
        food_name = rag_result.get("food_name")

        if "error" in rag_result:
            all_nutrition_results.append(
                {
                    "identified_name": food_name,
                    "error": rag_result["error"],
                }
            )
            continue

        standard_portion_weight = rag_result.get("standard_portion_weight_g")
        standard_portion_desc = rag_result.get("standard_portion_description", "")

        num_portions = portion_lookup.get(food_name, 1.0)
        num_portions = max(0.1, min(50.0, num_portions))

        calculated_weight = (
            standard_portion_weight * num_portions if standard_portion_weight else None
        )

        all_nutrition_results.append(
            {
                "identified_name": food_name,
                "num_portions": num_portions,
                "standard_portion_description": standard_portion_desc,
                "standard_portion_weight_g": standard_portion_weight,
                "calculated_weight_g": calculated_weight,
                "potential_matches": rag_result.get("potential_matches", []),
            }
        )

    step_time = time.time() - step_start
    step_metrics.append(
        {"step": 3, "name": "calculate_weights", "time_seconds": round(step_time, 2)}
    )
    logger.info("Step 3 complete")

    recognition_result = {
        "final_analysis": all_nutrition_results,
        "step_metrics": step_metrics,
    }

    # --- Step 4: LLM generates summary ---
    step_start = time.time()
    logger.info("Step 4: Generating final summary...")
    summary_prompt = f"""You are a helpful nutrition assistant. We have analyzed the user's meal.
    Here is the data: food name, number of portions, standard portion weight, calculated weight, and potential matches:
    {json.dumps(all_nutrition_results, indent=2, ensure_ascii=False)}

    Your task is to synthesize this into a single, easy-to-understand summary.
    
    CRITICAL INSTRUCTIONS:
    1.  **Acknowledge the Meal FIRST**: State what food items you identified and their calculated weights.
        - Translate English food names into the user's language ('{lang}').
    2.  **Select the Best Match**: Review `potential_matches` and choose the one that best matches the food in the image.
    3.  **Calculate Totals**: Use the 'calculated_weight_g' exactly as provided. Multiply per-100g nutritional values by (calculated_weight_g / 100).
    4.  **Sum and Summarize**: Calculate total nutrition for the entire meal.
    5.  Provide a final health assessment and a friendly tip.
    6.  **Language**: Your response MUST be in '{lang}'.
    7.  **Tone & Perspective**: Present the final nutritional information as YOUR OWN expert analysis.

    Provide only the final, conversational response to the user. Do not explain your matching process unless asked.
    """

    try:
        messages_to_send_3 = (
            [SystemMessage(content="You are a helpful nutrition assistant.")]
            + messages
            + [HumanMessage(content=summary_prompt)]
        )
        ai_message = client.invoke(
            messages_to_send_3, config={"tags": ["final_node_output"]}
        )
        msg: AnyMessage = ai_message

        step_time = time.time() - step_start
        step_metrics.append(
            {"step": 4, "name": "generate_summary", "time_seconds": round(step_time, 2)}
        )
        logger.info(f"Step 4 complete")

        total_time = sum(m["time_seconds"] for m in step_metrics)
        logger.info(
            f"Recognition complete. Total time: {total_time:.2f}s, Steps: {step_metrics}"
        )

        return {
            "recognition_result": recognition_result,
            "messages": [msg],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }
    except Exception as e:
        logger.error(f"Step 4 failed: {e}")
        error_message = (
            f"抱歉，总结营养信息时出错：{e}"
            if lang == "Chinese"
            else f"Sorry, an error occurred while summarizing the nutritional information: {e}"
        )
        return {
            "recognition_result": recognition_result,
            "messages": [AIMessage(content=error_message)],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }
