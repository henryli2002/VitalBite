"""
Unified food recognition agent using LLM-based object detection,
cropping, and a local fine-tuned model for highly accurate,
item-by-item nutrition estimation, culminating in a final summary.
"""

import json
import time
import asyncio
import base64
import os
import io
from datetime import datetime, timezone
from typing import List, Dict, Optional

import redis.asyncio as redis
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage

from langgraph_app.utils.logger import get_logger
from langgraph_app.orchestrator.state import GraphState, NodeOutput
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.llm_callback import create_callback_handler
from langgraph_app.utils.llm_factory import inject_dynamic_context
from langgraph_app.utils.utils import get_dominant_language
from langgraph_app.utils.semaphores import with_semaphore
from langgraph_app.utils.retry import with_retry
from langgraph_app.config import config

from .predictor import extract_image_bytes, predict_nutrition
from .schemas import FoodDetection

logger = get_logger(__name__)


def _calculate_tdee(user_profile: dict) -> int | None:
    """Estimate TDEE via Mifflin-St Jeor + PAL. Returns None if data is insufficient."""
    try:
        weight = float(user_profile.get("weight_kg") or 0)
        height = float(user_profile.get("height_cm") or 0)
        age    = float(user_profile.get("age") or 0)
    except (ValueError, TypeError):
        return None

    if not (weight and height and age):
        return None

    gender = (user_profile.get("gender") or "").lower()
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == "female":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 78  # average

    goals = (user_profile.get("fitness_goals") or "").lower()
    pal = 1.5 if any(w in goals for w in ("high intensity", "active", "athlete")) else 1.2
    return round(bmr * pal)


# Module-level Redis singleton for publishing thinking updates
_redis_client: Optional[redis.Redis] = None

def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        from langgraph_app.config import config
        _redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
    return _redis_client


async def _send_thinking_update(
    redis_client: Optional[redis.Redis],
    response_channel: Optional[str],
    message: str,
):
    """Helper to send a partial "thinking" update."""
    if not redis_client or not response_channel:
        return
    try:
        payload = {
            "status": "partial",
            "node": "recognition",
            "analysis": {"reasoning": message, "_stream": True},
        }
        await redis_client.publish(response_channel, json.dumps(payload))
    except Exception as e:
        logger.warning(f"[recognition] Failed to publish thinking update: {e}")


async def _estimate_nutrition_with_llm(
    client,
    detected_items: list,
    lang: str,
) -> list:
    """Fallback: ask the LLM to estimate nutrition by food name (B-strategy for Step 3).

    Returns a list matching the itemized_nutrition format:
        [{"name": str, "nutrition": {calculated_weight_g, total_calories, ...}}, ...]
    """
    names = [
        (item.get("name") if isinstance(item, dict) else getattr(item, "name", "Unknown Food"))
        for item in detected_items
    ]
    prompt = (
        f"Estimate the nutritional content for these food items: {', '.join(names)}.\n"
        "For each item return a JSON object with keys: "
        "name, calculated_weight_g, total_calories, total_fat, total_carb, total_protein.\n"
        "Return a JSON array only, no extra text."
    )
    try:
        response = await with_retry(
            lambda: client.ainvoke(
                [HumanMessage(content=prompt)],
                config={"callbacks": [create_callback_handler("food_recognition_fallback")]},
            ),
            attempts=3,
            base=0.8,
            cap=15.0,
            fallback=None,
        )
        if response is None:
            raise ValueError("LLM returned None")
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        items_data = json.loads(raw)
        result = []
        for item in items_data:
            nutrition = {
                "calculated_weight_g": float(item.get("calculated_weight_g", 0)),
                "total_calories": float(item.get("total_calories", 0)),
                "total_fat": float(item.get("total_fat", 0)),
                "total_carb": float(item.get("total_carb", 0)),
                "total_protein": float(item.get("total_protein", 0)),
            }
            result.append({"name": item.get("name", "Unknown"), "nutrition": nutrition})
        return result
    except Exception as e:
        logger.warning(f"[recognition] LLM nutrition estimation also failed: {e}")
        # Last resort: zeros with a note — at least Step 4 can still run
        return [
            {
                "name": n,
                "nutrition": {
                    "calculated_weight_g": 0.0,
                    "total_calories": 0.0,
                    "total_fat": 0.0,
                    "total_carb": 0.0,
                    "total_protein": 0.0,
                },
            }
            for n in names
        ]


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
    response_channel = state.get("response_channel")
    client = get_llm_client(module="food_recognition")
    lang = get_dominant_language(messages)

    redis_client = _get_redis() if response_channel else None

    try:
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

        tdee = _calculate_tdee(user_profile) if user_profile else None
        daily_cal_ref = (
            f"~{tdee} kcal (your estimated daily needs)"
            if tdee
            else "~2000 kcal (average adult estimate)"
        )

        # --- Meal Context (set by router) ---
        meal_time = state.get("meal_time") or "not meal time"
        meal_context = (
            f"\n\n[MEAL CONTEXT]\nThis food is being consumed at {meal_time}."
            if meal_time != "not meal time"
            else "\n\n[MEAL CONTEXT]\nThis food is being consumed outside main meal hours (likely a snack)."
        )

        # --- Step 1: Extract image ---
        await _send_thinking_update(
            redis_client, response_channel, "Step 1/4: Preparing image"
        )
        step_start = time.time()
        logger.info("Step 1: Extracting image...")
        image_bytes = extract_image_bytes(messages)

        if not image_bytes:
            content = (
                "抱歉，我没有在您的消息中找到图片，也没有找到有效的食物。请重新上传图片后再试。"
                if lang == "Chinese"
                else "Sorry, I couldn't find an image in your message. Please upload a food photo and try again."
            )
            return {
                "messages": [
                    AIMessage(
                        content=content,
                        additional_kwargs={"timestamp": datetime.now(timezone.utc).isoformat()},
                    )
                ],
            }

        # --- Step 2: Object Detection via LLM ---
        await _send_thinking_update(
            redis_client, response_channel, "Step 2/4: Detecting food items"
        )
        step_start = time.time()
        logger.info("Step 2: LLM Object Detection...")
        detected_items = []
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
            detection_res = await asyncio.wait_for(
                with_retry(
                    lambda: structured_llm.ainvoke([detect_msg]),
                    attempts=3,
                    base=0.8,
                    cap=15.0,
                    fallback=None,
                ),
                timeout=config.PRIMARY_LLM_TIMEOUT_S,
            )
            if detection_res is not None:
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
        num_items = len(detected_items) if detected_items else 1
        await _send_thinking_update(
            redis_client,
            response_channel,
            f"Step 3/4: Analyzing {num_items} item(s)",
        )
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
        nutrition_source = "local_model"

        if not detected_items:
            # No food bounding boxes returned — tell the user directly
            logger.info("No food items detected in image. Returning clean message.")
            no_food_msg = (
                "抱歉，我没有在图片中检测到任何食物。请尝试上传一张清晰的食物照片。"
                if lang == "Chinese"
                else "Sorry, I couldn't detect any food items in the image. Please try uploading a clearer photo of your meal."
            )
            return {
                "messages": [AIMessage(content=no_food_msg)],
                "recognition_result": None,
                "debug_logs": [{"node": "recognition", "status": "no_food_detected"}],
            }
        else:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = img.size
            local_model_ok = True

            for i, item in enumerate(detected_items):
                if isinstance(item, dict):
                    box = item.get("box", {})
                    name = item.get("name", "Unknown Food")
                    ymin = box.get("ymin", 0) if isinstance(box, dict) else getattr(box, "ymin", 0)
                    xmin = box.get("xmin", 0) if isinstance(box, dict) else getattr(box, "xmin", 0)
                    ymax = box.get("ymax", 1000) if isinstance(box, dict) else getattr(box, "ymax", 1000)
                    xmax = box.get("xmax", 1000) if isinstance(box, dict) else getattr(box, "xmax", 1000)
                else:
                    box = getattr(item, "box", None)
                    name = getattr(item, "name", "Unknown Food")
                    if box is None:
                        continue
                    ymin = getattr(box, "ymin", 0) if not isinstance(box, dict) else box.get("ymin", 0)
                    xmin = getattr(box, "xmin", 0) if not isinstance(box, dict) else box.get("xmin", 0)
                    ymax = getattr(box, "ymax", 1000) if not isinstance(box, dict) else box.get("ymax", 1000)
                    xmax = getattr(box, "xmax", 1000) if not isinstance(box, dict) else box.get("xmax", 1000)

                crop_xmin = max(0, min(w, int(xmin * w / 1000)))
                crop_ymin = max(0, min(h, int(ymin * h / 1000)))
                crop_xmax = max(0, min(w, int(xmax * w / 1000)))
                crop_ymax = max(0, min(h, int(ymax * h / 1000)))

                if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
                    continue  # Invalid box

                crop_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
                buf = io.BytesIO()
                crop_img.save(buf, format="JPEG")

                try:
                    res = predict_nutrition(buf.getvalue())
                    itemized_nutrition.append({"name": name, "nutrition": res})
                    for k in total_nutrition:
                        total_nutrition[k] += res.get(k, 0.0)
                        total_nutrition[k] = round(total_nutrition[k], 2)
                    await _send_thinking_update(
                        redis_client, response_channel,
                        f"Step 3/4: Analyzing item {i + 1}/{num_items}",
                    )
                except Exception as e:
                    # B-strategy: one item fails → assume systemic, switch all to LLM estimation
                    logger.warning(
                        f"[recognition] Local model failed on item '{name}' (attempt {i+1}): {e}. "
                        "Switching all items to LLM estimation."
                    )
                    local_model_ok = False
                    break

            if not local_model_ok:
                nutrition_source = "llm_estimate"
                await _send_thinking_update(
                    redis_client, response_channel, "Step 3/4: Using LLM nutrition estimation"
                )
                itemized_nutrition = await _estimate_nutrition_with_llm(client, detected_items, lang)
                total_nutrition = {k: 0.0 for k in total_nutrition}
                for entry in itemized_nutrition:
                    for k in total_nutrition:
                        total_nutrition[k] = round(total_nutrition[k] + entry["nutrition"].get(k, 0.0), 2)

        step_time = time.time() - step_start
        step_metrics.append(
            {"step": 3, "name": "crop_and_predict", "time_seconds": round(step_time, 2)}
        )

        recognition_result = {
            "itemized_analysis": itemized_nutrition,
            "total_analysis": total_nutrition,
            "step_metrics": step_metrics,
            "nutrition_source": nutrition_source,
        }

        # --- Step 4: LLM generates final summary ---
        await _send_thinking_update(
            redis_client, response_channel, "Step 4/4: Generating summary"
        )
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

[CONTEXT]{profile_context}{meal_context}

[CONSTRAINTS]
1. IDENTIFY: Greet the user and identify the foods detected (e.g., "I see a burger and fries...").
2. STRICT FORMATTING: You MUST use a Markdown table to display the nutrition data. Do not write the data in paragraphs. Use the exact table format shown below.
3. ITEMIZATION: Provide the data for each individual item. DO NOT include a "Total" or "Summary" row in the table. The frontend will calculate the total automatically.
4. ACCURACY: Report the exact numbers from the data. Do not recalculate or modify them.
5. PERSONALIZATION: Explicitly evaluate the meal against the 'User Profile'. Call out allergies or goals.
6. LANGUAGE: The response MUST be entirely in '{lang}'.
7. MEAL FIT: Add one sentence after the table assessing whether the total caloric load is appropriate given the [MEAL CONTEXT]. Reference: daily needs = {daily_cal_ref}. Meal targets: breakfast ~25-30%, lunch ~35-40%, dinner ~30-35%, snack ~10-15% (additional) of daily needs.

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
            invoke_cfg = {
                "callbacks": [create_callback_handler("food_recognition")],
                "tags": ["final_node_output"],
            }
            # with_retry handles transient failures; cascade to llamacpp on exhaustion/timeout
            ai_message = None
            try:
                ai_message = await asyncio.wait_for(
                    with_retry(
                        lambda: client.ainvoke(messages_to_send, config=invoke_cfg),
                        attempts=3,
                        base=0.8,
                        cap=15.0,
                        fallback=None,
                    ),
                    timeout=config.PRIMARY_LLM_TIMEOUT_S,
                )
            except Exception as e:
                logger.warning(f"[recognition] Step 4 primary failed ({type(e).__name__}), cascading")

            if ai_message is None:
                # Cascade: llamacpp receives only the numeric summary (no image)
                from langgraph_app.utils.cascade import invoke_with_cascade
                summary_msgs = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=summary_prompt),
                ]
                ai_message = await invoke_with_cascade(
                    module="food_recognition",
                    messages_to_send=summary_msgs,
                    lang=lang,
                    timeout_s=config.PRIMARY_LLM_TIMEOUT_S,
                )

            step_time = time.time() - step_start
            step_metrics.append({"step": 4, "name": "generate_summary", "time_seconds": round(step_time, 2)})
            logger.info(f"Step 4 complete. Total steps time: {sum(m['time_seconds'] for m in step_metrics)}s")

            ai_message.additional_kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
            if nutrition_source == "llm_estimate":
                ai_message.additional_kwargs["nutrition_source"] = "llm_estimate"
            return {
                "recognition_result": recognition_result,
                "messages": [ai_message],
            }
        except Exception as e:
            logger.error(f"Step 4 (LLM summary) failed: {e}")
            return {
                "recognition_result": recognition_result,
                "messages": [
                    AIMessage(
                        content=f"抱歉，总结出错：{e}"
                        if lang == "Chinese"
                        else f"Sorry, error: {e}",
                        additional_kwargs={"timestamp": datetime.now(timezone.utc).isoformat()},
                    )
                ],
            }
    except Exception as e:
        logger.error(f"Recognition node failed: {e}")
        return {
            "messages": [
                AIMessage(
                    content=f"抱歉，识别出错：{e}"
                    if lang == "Chinese"
                    else f"Sorry, error: {e}",
                    additional_kwargs={"timestamp": datetime.now(timezone.utc).isoformat()},
                )
            ],
        }