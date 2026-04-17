"""Food recognition tool for the Supervisor agent.

Wraps the core recognition pipeline (object detection + cropping + local model
prediction) as a LangChain @tool that returns structured JSON. Does NOT generate
natural language summaries — that is the Supervisor's responsibility.
"""

import json
import io
import base64
import asyncio
import logging
from typing import Optional

from PIL import Image
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
import re

from langgraph_app.agents.food_recognition.predictor import predict_nutrition
from langgraph_app.agents.food_recognition.schemas import FoodDetection
from langgraph_app.utils.llm_factory import get_llm_client
from langgraph_app.utils.llm_callback import create_callback_handler
from langgraph_app.utils.retry import with_retry
from langgraph_app.utils.semaphores import with_semaphore
from langgraph_app.config import config

logger = logging.getLogger("wabi.tools.food_recognition")


async def _estimate_nutrition_with_llm(client, detected_items: list) -> list:
    """Fallback: ask the LLM to estimate nutrition by food name.

    Returns a list of dicts: [{"name": str, "nutrition": {...}}, ...]
    """
    names = [
        (
            item.get("name")
            if isinstance(item, dict)
            else getattr(item, "name", "Unknown Food")
        )
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
                config={
                    "callbacks": [create_callback_handler("food_recognition_fallback")]
                },
            ),
            attempts=3,
            base=0.8,
            cap=15.0,
            fallback=None,
        )
        if response is None:
            raise ValueError("LLM returned None")
        raw = response.content.strip()
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
        logger.warning("LLM nutrition estimation failed: %s", e)
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


async def _run_recognition_pipeline(image_bytes: bytes) -> dict:
    """Core recognition pipeline: detection -> cropping -> prediction.

    Returns structured dict with items, total_nutrition, and nutrition_source.
    """
    client = get_llm_client(module="food_recognition")

    # Step 1: Object detection via LLM
    detected_items = []
    try:
        structured_llm = client.with_structured_output(FoodDetection)
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        detect_msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Detect all distinct food portions or dishes in this image. "
                        "Group items by 'plate' or 'serving'. For example, a burger and a side "
                        "of fries are TWO separate items. A plate of salad is ONE single item. "
                        "Do NOT detect individual ingredients within a single dish. "
                        "For each detected dish/portion, provide its name and its bounding box "
                        "(ymin, xmin, ymax, xmax normalized between 0 and 1000)."
                    ),
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
        logger.error("Object detection failed: %s", e)

    # Step 2: Cropping and prediction
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
        # No items detected — run local model on the full image
        try:
            res = predict_nutrition(image_bytes)
            itemized_nutrition.append({"name": "Full Meal", "nutrition": res})
            for k in total_nutrition:
                total_nutrition[k] += res.get(k, 0.0)
        except Exception as e:
            logger.warning(
                "Local model failed on full image: %s. Using LLM estimate.", e
            )
            nutrition_source = "llm_estimate"
            itemized_nutrition = await _estimate_nutrition_with_llm(
                client, [{"name": "Meal"}]
            )
    else:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        local_model_ok = True

        for item in detected_items:
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

            crop_xmin = max(0, min(w, int(xmin * w / 1000)))
            crop_ymin = max(0, min(h, int(ymin * h / 1000)))
            crop_xmax = max(0, min(w, int(xmax * w / 1000)))
            crop_ymax = max(0, min(h, int(ymax * h / 1000)))

            if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
                continue

            crop_img = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
            buf = io.BytesIO()
            crop_img.save(buf, format="JPEG")

            try:
                res = predict_nutrition(buf.getvalue())
                itemized_nutrition.append({"name": name, "nutrition": res})
                for k in total_nutrition:
                    total_nutrition[k] += res.get(k, 0.0)
                    total_nutrition[k] = round(total_nutrition[k], 2)
            except Exception as e:
                logger.warning(
                    "Local model failed on item '%s': %s. Switching to LLM estimation.",
                    name,
                    e,
                )
                local_model_ok = False
                break

        if not local_model_ok:
            nutrition_source = "llm_estimate"
            itemized_nutrition = await _estimate_nutrition_with_llm(
                client, detected_items
            )
            total_nutrition = {k: 0.0 for k in total_nutrition}
            for entry in itemized_nutrition:
                for k in total_nutrition:
                    total_nutrition[k] = round(
                        total_nutrition[k] + entry["nutrition"].get(k, 0.0), 2
                    )

    return {
        "items": itemized_nutrition,
        "total_calories": total_nutrition.get("total_calories", 0.0),
        "total_nutrition": total_nutrition,
        "nutrition_source": nutrition_source,
    }


def _build_description(result: dict) -> str:
    """Short human-readable description for the DB placeholder, e.g. '汉堡+薯条, 850kcal'."""
    items = result.get("items") or []
    names = []
    for it in items:
        if isinstance(it, dict):
            name = it.get("name")
            if name:
                names.append(str(name))
    joined = "+".join(names[:3]) if names else "meal"
    total_cal = result.get("total_calories", 0.0) or 0.0
    try:
        cal_str = f"{int(round(float(total_cal)))}kcal"
    except Exception:
        cal_str = "?kcal"
    return f"{joined}, {cal_str}"


class AnalyzeFoodImageInput(BaseModel):
    image_uuid: str = Field(
        description="The 32-hex ID of the image exactly as written in the placeholder, without brackets or prefix. For example: 7b0ed022bf0d4a96815cc1c5a440e9c4"
    )


@tool("analyze_food_image", args_schema=AnalyzeFoodImageInput)
async def analyze_food_image(
    image_uuid: str,
    config: RunnableConfig = None,
) -> str:
    """Analyze a food image to detect items and estimate nutritional content.

    Args:
        image_uuid: The short UUID handle from a ``[图片: {uuid}]`` placeholder
            in the user's message. The image bytes are loaded from the server's
            image registry — never passed inline.

    Returns:
        A JSON string containing detected food items with per-item and total
        nutritional breakdown (calories, fat, carbs, protein, weight).
    """
    try:
        from server.image_store import load_image
        from server.db import PostgresHistoryStore
    except Exception as e:
        logger.error("image_store / db import failed: %s", e)
        return json.dumps({"error": f"image store unavailable: {e}"})

    user_id = None
    if config and isinstance(config, dict):
        user_id = (config.get("configurable") or {}).get("user_id")
    if not user_id:
        return json.dumps({"error": "missing user_id in runtime config"})

    image_uuid = (image_uuid or "").strip()
    if not image_uuid:
        return json.dumps({"error": "empty image_uuid"})

    # Defense: extract the 32-hex uuid in case the LLM included brackets or prefix
    match = re.search(r"([a-fA-F0-9]{32})", image_uuid)
    if match:
        image_uuid = match.group(1).lower()
    else:
        return json.dumps(
            {"error": "invalid image_uuid format, expected 32-hex string"}
        )

    try:
        image_bytes, _mime = load_image(user_id, image_uuid)
    except FileNotFoundError:
        return json.dumps({"error": f"image {image_uuid} not found"})
    except Exception as e:
        logger.error("load_image failed: %s", e)
        return json.dumps({"error": f"failed to load image: {e}"})

    try:
        result = await _run_recognition_pipeline(image_bytes)
    except Exception as e:
        logger.error("analyze_food_image pipeline failed: %s", e, exc_info=True)
        return json.dumps({"error": str(e)})

    # Best-effort: enrich the DB placeholder with a short description.
    try:
        store = PostgresHistoryStore()
        description = _build_description(result)
        await store.update_image_description(user_id, image_uuid, description)
    except Exception as e:
        logger.warning("Failed to update image description placeholder: %s", e)

    return json.dumps(result, ensure_ascii=False)
