"""
Unified food recognition agent using LLM-based object detection,
cropping, and a local fine-tuned model for highly accurate,
item-by-item nutrition estimation, culminating in a final summary.
"""

import io
import json
import time
import asyncio
import base64
from datetime import datetime
from typing import List, Dict

from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langgraph_app.utils.logger import get_logger
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.tracked_llm import get_tracked_llm
from langgraph_app.utils.llm_factory import inject_dynamic_context
from langgraph_app.utils.utils import get_dominant_language
from langgraph_app.utils.semaphores import with_semaphore

from .predictor import extract_image_bytes, predict_nutrition
from .schemas import FoodDetection

logger = get_logger(__name__)


@with_semaphore("recognition")
async def recognition_node(state: GraphState) -> NodeOutput:
    """
    A unified node that performs food recognition.

    Steps:
    1. Extract image from user messages.
    2. Pass image to LLM to get bounding boxes for distinct food items.
    3. Crop image into N pieces based on bounding boxes.
    4. Run fine-tuned local model on each crop to get nutrition estimates.
    5. Aggregate and pass the data to LLM for final tabular summary.
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

    # --- Step 1: Extract image ---
    step_start = time.time()
    logger.info("Step 1: Extracting image...")
    image_bytes = extract_image_bytes(messages)

    if not image_bytes:
        return {
            "messages": [
                AIMessage(
                    content="未在消息中找到有效的图片。"
                    if lang == "Chinese"
                    else "No valid image found."
                )
            ],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }

    # --- Step 2: Object Detection via LLM ---
    step_start = time.time()
    logger.info("Step 2: LLM Object Detection...")
    try:
        structured_llm = client.with_structured_output(FoodDetection)
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        detect_msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Detect all distinct food portions or dishes in this image. Group items by 'plate' or 'serving'. For example, a burger and a side of fries are TWO separate items. A plate of salad (even if ingredients are visibly unmixed) is ONE single item. Do NOT detect individual ingredients within a single dish. For each detected dish/portion, provide its name and its bounding box (ymin, xmin, ymax, xmax normalized between 0 and 1000).",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ]
        )
        detection_res = await structured_llm.ainvoke([detect_msg])
        detected_items = getattr(detection_res, "items", [])
        if not detected_items and isinstance(detection_res, dict):
            detected_items = detection_res.get("items", [])
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        detected_items = []

    step_time = time.time() - step_start
    step_metrics.append(
        {"step": 2, "name": "object_detection", "time_seconds": round(step_time, 2)}
    )

    # --- Step 3: Cropping and Prediction ---
    step_start = time.time()
    logger.info(
        f"Step 3: Cropping and Local Prediction for {len(detected_items)} items..."
    )

    itemized_nutrition = []
    total_nutrition = {
        "calculated_weight_g": 0.0,
        "total_calories": 0.0,
        "total_fat": 0.0,
        "total_carb": 0.0,
        "total_protein": 0.0,
    }

    try:
        if not detected_items:
            # Fallback to full image
            logger.info("No items detected. Falling back to full image.")
            res = predict_nutrition(image_bytes)
            itemized_nutrition.append({"name": "Full Meal", "nutrition": res})
            for k in total_nutrition:
                total_nutrition[k] += res.get(k, 0.0)
        else:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = img.size
            for item in detected_items:
                # Calculate pixel coords
                if isinstance(item, dict):
                    box = item.get("box", {})
                    name = item.get("name", "Unknown Food")
                    ymin = (
                        box.get("ymin", 0)
                        if isinstance(box, dict)
                        else getattr(box, "ymin", 0)
                    )
                    xmin = (
                        box.get("xmin", 0)
                        if isinstance(box, dict)
                        else getattr(box, "xmin", 0)
                    )
                    ymax = (
                        box.get("ymax", 1000)
                        if isinstance(box, dict)
                        else getattr(box, "ymax", 1000)
                    )
                    xmax = (
                        box.get("xmax", 1000)
                        if isinstance(box, dict)
                        else getattr(box, "xmax", 1000)
                    )
                else:
                    box = getattr(item, "box", None)
                    name = getattr(item, "name", "Unknown Food")
                    if box is None:
                        continue
                    ymin = (
                        getattr(box, "ymin", 0)
                        if not isinstance(box, dict)
                        else box.get("ymin", 0)
                    )
                    xmin = (
                        getattr(box, "xmin", 0)
                        if not isinstance(box, dict)
                        else box.get("xmin", 0)
                    )
                    ymax = (
                        getattr(box, "ymax", 1000)
                        if not isinstance(box, dict)
                        else box.get("ymax", 1000)
                    )
                    xmax = (
                        getattr(box, "xmax", 1000)
                        if not isinstance(box, dict)
                        else box.get("xmax", 1000)
                    )

                # Ensure valid crop box
                crop_xmin = max(0, min(w, int(xmin * w / 1000)))
                crop_ymin = max(0, min(h, int(ymin * h / 1000)))
                crop_xmax = max(0, min(w, int(xmax * w / 1000)))
                crop_ymax = max(0, min(h, int(ymax * h / 1000)))

                if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
                    continue  # Invalid box

                crop_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
                buf = io.BytesIO()
                crop_img.save(buf, format="JPEG")

                res = predict_nutrition(buf.getvalue())
                itemized_nutrition.append({"name": name, "nutrition": res})

                for k in total_nutrition:
                    total_nutrition[k] += res.get(k, 0.0)
                    total_nutrition[k] = round(total_nutrition[k], 2)

    except Exception as e:
        logger.error(f"Cropping/Prediction failed: {e}")
        return {
            "messages": [
                AIMessage(
                    content=f"抱歉，分析时出错：{e}"
                    if lang == "Chinese"
                    else f"Sorry, error: {e}"
                )
            ],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }

    step_time = time.time() - step_start
    step_metrics.append(
        {"step": 3, "name": "crop_and_predict", "time_seconds": round(step_time, 2)}
    )

    recognition_result = {
        "itemized_analysis": itemized_nutrition,
        "total_analysis": total_nutrition,
        "step_metrics": step_metrics,
    }

    # --- Step 4: LLM generates final summary ---
    step_start = time.time()
    logger.info("Step 4: Generating final structured summary via LLM...")

    summary_data = {"items": itemized_nutrition, "total": total_nutrition}

    summary_prompt = f"""[NUTRITION BREAKDOWN DATA]
{json.dumps(summary_data, indent=2, ensure_ascii=False)}

[TASK]
We have used object detection and a specialized model to calculate the nutrition for each item separately, and summed them into 'total'.
Please synthesize this data into a structured, easy-to-understand summary."""

    try:
        system_content = f"""[ROLE]
You are WABI, an expert nutrition assistant.

[OBJECTIVE]
Summarize the user's meal with an item-by-item breakdown and total, based strictly on the [NUTRITION BREAKDOWN DATA].

[CONTEXT]{profile_context}

[CONSTRAINTS]
1. IDENTIFY: Greet the user and identify the foods detected (e.g., "I see a burger and fries...").
2. STRICT FORMATTING: You MUST use a Markdown table to display the nutrition data. Do not write the data in paragraphs. Use the exact table format shown below.
3. ITEMIZATION: Provide the data for each individual item. DO NOT include a "Total" or "Summary" row in the table. The frontend will calculate the total automatically.
4. ACCURACY: Report the exact numbers from the data. Do not recalculate or modify them.
5. PERSONALIZATION: Explicitly evaluate the meal against the 'User Profile'. Call out allergies or goals.
6. LANGUAGE: The response MUST be entirely in '{lang}'.

[REQUIRED TABLE FORMAT]
| 项目 (Item) | 重量 (Mass) | 热量 (Calories) | 脂肪 (Fat) | 碳水 (Carbs) | 蛋白质 (Protein) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [Item 1] | [mass] g | [cal] kcal | [fat] g | [carbs] g | [protein] g |
| [Item 2] | [mass] g | [cal] kcal | [fat] g | [carbs] g | [protein] g |

(Note: translate the table headers to '{lang}' if it is not Chinese)"""

        messages_to_send = (
            [SystemMessage(content=system_content)]
            + messages
            + [HumanMessage(content=summary_prompt)]
        )
        messages_to_send = inject_dynamic_context(messages_to_send)
        ai_message = await client.ainvoke(
            messages_to_send, config={"tags": ["final_node_output"]}
        )

        step_time = time.time() - step_start
        step_metrics.append(
            {"step": 4, "name": "generate_summary", "time_seconds": round(step_time, 2)}
        )
        logger.info(
            f"Step 4 complete. Total steps time: {sum(m['time_seconds'] for m in step_metrics)}s"
        )

        return {
            "recognition_result": recognition_result,
            "messages": [ai_message],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }
    except Exception as e:
        logger.error(f"Step 4 (LLM summary) failed: {e}")
        return {
            "recognition_result": recognition_result,
            "messages": [
                AIMessage(
                    content=f"抱歉，总结出错：{e}"
                    if lang == "Chinese"
                    else f"Sorry, error: {e}"
                )
            ],
            "message_timestamps": [datetime.utcnow().isoformat()],
        }
