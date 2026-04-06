"""
Unified food recognition agent using a local fine-tuned model for highly accurate
nutrition estimation and an LLM for final summarization.
"""

import json
import time
import asyncio
from datetime import datetime
from typing import List, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langgraph_app.utils.logger import get_logger
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.tracked_llm import get_tracked_llm
from langgraph_app.utils.llm_factory import inject_dynamic_context
from langgraph_app.utils.utils import get_dominant_language
from langgraph_app.utils.semaphores import with_semaphore

from .predictor import extract_image_bytes, predict_nutrition

logger = get_logger(__name__)


@with_semaphore("recognition")
async def recognition_node(state: GraphState) -> NodeOutput:
    """
    A unified node that performs food recognition.

    Steps:
    1. Extract image from user messages.
    2. Pass image to the fine-tuned local model to get nutrition estimates.
    3. Pass the predictions and the image to the LLM to write a contextual summary.
    """
    messages = state.get("messages", [])
    client = get_tracked_llm(module="food_recognition", node_name="food_recognition")
    lang = get_dominant_language(messages)

    step_metrics = []

    # --- Profile Setup ---
    user_profile = state.get("user_profile")
    profile_context = ""
    if user_profile:
        profile_context = "\n\nUser Profile & Health Information:\n" + "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in user_profile.items()
            if v
        )

    # --- Step 1 & 2: Local Model Prediction ---
    step_start = time.time()
    logger.info("Step 1: Extracting image and running local prediction...")

    image_bytes = extract_image_bytes(messages)

    if not image_bytes:
        error_message = (
            "未在消息中找到有效的图片。"
            if lang == "Chinese"
            else "No valid image found in the messages."
        )
        return {
            "messages": [AIMessage(content=error_message)],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }

    try:
        # Predict using local finetuned MobileNetV3
        nutrition_estimates = predict_nutrition(image_bytes)
        logger.info(f"Local prediction successful: {nutrition_estimates}")
    except Exception as e:
        logger.error(f"Local prediction failed: {e}")
        error_message = (
            f"抱歉，分析食物图片时出错：{e}"
            if lang == "Chinese"
            else f"Sorry, an error occurred while analyzing the food image: {e}"
        )
        return {
            "messages": [AIMessage(content=error_message)],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }

    step_time = time.time() - step_start
    step_metrics.append(
        {
            "step": 1,
            "name": "local_nutrition_prediction",
            "time_seconds": round(step_time, 2),
        }
    )

    recognition_result = {
        "final_analysis": nutrition_estimates,
        "step_metrics": step_metrics,
    }

    # --- Step 3: LLM generates summary ---
    step_start = time.time()
    logger.info("Step 2: Generating final summary via LLM...")
    summary_prompt = f"""[TOTAL MEAL NUTRITION DATA]
{json.dumps(nutrition_estimates, indent=2, ensure_ascii=False)}

[TASK]
Look at the user's image and the predicted nutritional data above. 
Note that this data represents the TOTAL estimated nutritional value for ALL food items combined in the image.
Identify what the foods are, and synthesize this data into a structured, easy-to-understand summary."""

    try:
        system_content = f"""[ROLE]
You are WABI, an expert nutrition assistant.

[OBJECTIVE]
Summarize the user's meal, identify the foods in the image, and provide a structured nutritional assessment based on the provided TOTAL MEAL NUTRITION DATA.

[CONTEXT]{profile_context}

[CONSTRAINTS]
1. IDENTIFY: Briefly tell the user what foods you see in the image.
2. STRUCTURED DATA: Present the nutrition data (mass, calories, fat, carbs, protein) using a clean, well-formatted Markdown table or bulleted list for a beautiful UI presentation.
3. TOTAL AMOUNT: Clearly explain to the user that the reported data represents the TOTAL amount for the entire meal (all items combined) shown in the image.
4. ACCURACY: Report the numbers exactly as provided in the data. Do not recalculate them or use external databases.
5. PERSONALIZATION: CRITICAL - Explicitly evaluate the meal against the 'User Profile'. Call out if it violates allergies or helps their goals.
6. LANGUAGE: The response (including table headers and labels) MUST be entirely in '{lang}'.
7. TONE: Present as YOUR OWN expert analysis. Be professional, supportive, and conversational."""

        messages_to_send = (
            [SystemMessage(content=system_content)]
            + messages
            + [HumanMessage(content=summary_prompt)]
        )
        messages_to_send = inject_dynamic_context(messages_to_send)
        ai_message = await client.ainvoke(
            messages_to_send, config={"tags": ["final_node_output"]}
        )
        msg: AnyMessage = ai_message

        step_time = time.time() - step_start
        step_metrics.append(
            {"step": 2, "name": "generate_summary", "time_seconds": round(step_time, 2)}
        )
        logger.info("Step 2 complete")

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
        logger.error(f"Step 2 (LLM summary) failed: {e}")
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
