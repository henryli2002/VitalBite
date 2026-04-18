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


# Post-analysis rendering rules, injected into the tool's return JSON so the
# Supervisor only pays this token cost when the tool actually runs.  Captures
# the v3.3 Step-4 summary contract: item-name translation, strict table, meal
# fit math against the user's daily budget, and allergy/goal flags.
_POST_ANALYSIS_GUIDANCE = {
    "apply": "When rendering this tool result to the user, follow these rules EXACTLY.",
    "rules": [
        "1. IDENTIFY: open with a brief, warm sentence naming the foods detected.",
        "2. TRANSLATE ITEM NAMES: the `items[*].name` field is always in English. "
        "Translate each name into the user's language before displaying it. "
        "Keep the original English only when no common translation exists or the "
        "name is a proper/brand name.",
        "3. TABLE: render a Markdown table with exactly six columns in this order — "
        "Item, Weight, Calories, Fat, Carbs, Protein. Headers and units MUST be "
        "in the user's language. "
        "For Chinese responses the headers MUST be: "
        "`| 项目 | 重量 | 热量 | 脂肪 | 碳水 | 蛋白质 |`. "
        "For English responses keep the English headers as written above.",
        "4. NO TOTAL ROW: do NOT append a Total / 总计 row — the frontend sums the "
        "columns automatically.",
        "5. ACCURACY: copy the numeric values from `items[*].nutrition` verbatim. "
        "Do not recompute, round, or re-scale.",
        "6. PERSONALIZATION: cross-check the detected items against the user's "
        "allergies / dietary restrictions / health conditions from [USER CONTEXT] "
        "in the system prompt. If any item conflicts, flag it explicitly in one "
        "short sentence after the table.",
        "7. MEAL FIT: internally consider the current meal period and the daily "
        "calorie reference in [USER CONTEXT] to judge whether the total calories "
        "fit the user's meal budget. Then state a ONE-sentence plain-language "
        "verdict (e.g. 'a reasonable lunch', 'a bit heavy for dinner'). "
        "Do NOT expose any percentage math, target ranges, or internal reasoning "
        "numbers to the user.",
        "8. LANGUAGE: the response supports Chinese and English ONLY. Detect the "
        "user's language from recent messages and write the ENTIRE reply (prose + "
        "table + flags) in that language.",
    ],
}


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


def _downscale_for_detection(image_bytes: bytes, max_side: int = 1024, quality: int = 85) -> bytes:
    """Shrink oversized photos before sending to the vision API.

    Food detection doesn't need full-resolution — large PNGs (multi-MB, 2K+
    long side) bloat request time and push us past the detection timeout.
    Returns JPEG bytes. Falls back to the original bytes on any failure.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        long_side = max(img.size)
        if long_side > max_side:
            scale = max_side / long_side
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning("Image downscale failed, using original: %r", e)
        return image_bytes


async def _run_recognition_pipeline(image_bytes: bytes) -> dict:
    """Core recognition pipeline: detection -> cropping -> prediction.

    Returns structured dict with items, total_nutrition, and nutrition_source.
    """
    client = get_llm_client(module="food_recognition")

    # Step 1: Object detection via LLM
    detected_items = []
    try:
        structured_llm = client.with_structured_output(FoodDetection)
        detect_bytes = _downscale_for_detection(image_bytes)
        img_b64 = base64.b64encode(detect_bytes).decode("utf-8")
        detect_msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Detect all distinct food portions or dishes in this image. "
                        "Group items by 'plate' or 'serving'. For example, a burger and a side "
                        "of fries are TWO separate items. A plate of salad (even if ingredients "
                        "are visibly unmixed) is ONE single item. Do NOT detect individual "
                        "ingredients within a single dish, and do NOT return sauces, condiments, "
                        "spices, or garnishes as separate items — treat them as part of the dish "
                        "they accompany. "
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
            timeout=config.HEAVY_LLM_TIMEOUT_S,
        )
        if detection_res is not None:
            detected_items = getattr(detection_res, "items", [])
            if not detected_items and isinstance(detection_res, dict):
                detected_items = detection_res.get("items", [])
    except Exception as e:
        logger.error("Object detection failed: %r", e, exc_info=True)

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
        "display_guidance": _POST_ANALYSIS_GUIDANCE,
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
        description="The 32-hex UUID taken from the server-injected `<attached_image uuid=.../>` marker on the user's message. Pass the UUID only (no angle brackets, no attributes). Example: 7b0ed022bf0d4a96815cc1c5a440e9c4"
    )


@tool("analyze_food_image", args_schema=AnalyzeFoodImageInput)
async def analyze_food_image(
    image_uuid: str,
    config: RunnableConfig = None,
) -> str:
    """Analyze a food image to detect items and estimate nutritional content.

    CRITICAL INSTRUCTION FOR LLM: ALWAYS perform a fresh tool call when the user uploads a new image or asks for analysis. DO NOT answer from past conversation history.

    Args:
        image_uuid: The 32-hex UUID from a ``<attached_image uuid=.../>``
            marker that the server injected into the user's message. The
            image bytes are loaded from the server's image registry — never
            passed inline.

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
