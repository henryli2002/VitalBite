"""
WABI Food Recognition Evaluation Script

Compares four approaches:
1. Graph Pipeline: recognition_node (food ID → RAG → portion estimation → weight calc)
2. Direct LLM: Zero-shot, LLM directly estimates nutrition from image
3. Few-shot LLM: LLM estimates nutrition with reference images as calibration anchors
4. Fine-tuned: MobileNetV3-Small fine-tuned on training split, local inference

Metric: wMAPE = Σ|y_i - ŷ_i| / Σy_i

Usage:
    python eval/run_eval.py --n 30
    python eval/run_eval.py --n 30 --provider gemini
    python eval/run_eval.py --resume
"""

import sys
import os
import io
import asyncio
import argparse
import json
import time
import base64
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# --- Setup project path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR / "food_dataset"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

METRICS = ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"]
NUM_FEWSHOT_EXAMPLES = 5


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────
class DirectNutritionEstimate(BaseModel):
    """Direct estimation of total nutritional values from a food image."""
    total_mass_g: float = Field(description="Estimated total mass of all food in grams")
    total_calories_kcal: float = Field(description="Estimated total calories in kcal")
    total_fat_g: float = Field(description="Estimated total fat in grams")
    total_carb_g: float = Field(description="Estimated total carbohydrates in grams")
    total_protein_g: float = Field(description="Estimated total protein in grams")


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """Load and merge ground truth labels with images."""
    print("Loading dataset...")
    gt = pd.read_excel(DATASET_DIR / "dishes.xlsx")
    with open(DATASET_DIR / "dish_images.pkl", "rb") as f:
        images_df = pickle.load(f)
    merged = images_df.merge(gt, left_on="dish", right_on="dish_id", how="inner")
    print(f"  Ground truth: {len(gt)} | Images: {len(images_df)} | Matched: {len(merged)}")
    return merged


def image_to_base64_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_image_message(image_bytes: bytes, text: str = "What food is in this image?") -> HumanMessage:
    image_url = image_to_base64_url(image_bytes)
    return HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": text},
    ])


# ──────────────────────────────────────────────
# Few-shot: Reference Pool
# ──────────────────────────────────────────────
def select_fewshot_references(dataset: pd.DataFrame, test_ids: set, seed: int = 42) -> pd.DataFrame:
    """
    Select diverse reference examples from the dataset (excluding test samples).
    Stratified by calorie range to give the LLM a good calibration spread.
    """
    pool = dataset[~dataset["dish"].isin(test_ids)].copy()

    # Stratify by calorie quintiles for diversity
    pool["cal_bin"] = pd.qcut(pool["total_calories"], q=NUM_FEWSHOT_EXAMPLES, labels=False, duplicates="drop")
    refs = pool.groupby("cal_bin", group_keys=False).apply(
        lambda g: g.sample(n=1, random_state=seed)
    )

    # If we got fewer than needed due to binning, fill randomly
    if len(refs) < NUM_FEWSHOT_EXAMPLES:
        extra = pool[~pool["dish"].isin(refs["dish"])].sample(
            n=NUM_FEWSHOT_EXAMPLES - len(refs), random_state=seed
        )
        refs = pd.concat([refs, extra])

    print(f"  Few-shot references: {len(refs)} samples (cal range: "
          f"{refs['total_calories'].min():.0f}-{refs['total_calories'].max():.0f} kcal)")
    return refs.head(NUM_FEWSHOT_EXAMPLES)


# ──────────────────────────────────────────────
# Method 1: Graph Pipeline
# ──────────────────────────────────────────────
async def run_graph_recognition(image_bytes: bytes, provider: str = None) -> Dict:
    from langgraph_app.agents.food_recognition.agent import recognition_node

    message = build_image_message(image_bytes, "Please analyze the food in this image.")
    state = {
        "messages": [message],
        "user_id": "eval_user", "user_name": "Eval", "session_id": "eval_session",
        "user_profile": {}, "user_context": {}, "response_channel": "",
        "analysis": {"intent": "recognition", "confidence": 1.0, "reasoning": "eval"},
        "recognition_result": None, "recommendation_result": None,
        "message_timestamps": [], "debug_logs": [],
    }

    result = await recognition_node(state)
    recognition_result = result.get("recognition_result", {})
    final_analysis = recognition_result.get("final_analysis", [])

    total_mass = total_calories = total_fat = total_carb = total_protein = 0.0
    food_names = []

    for food in final_analysis:
        if "error" in food:
            continue
        food_names.append(food.get("identified_name", "unknown"))
        weight_g = food.get("calculated_weight_g")
        if weight_g is None:
            continue
        total_mass += weight_g
        matches = food.get("potential_matches", [])
        if matches:
            nutrients = matches[0].get("nutrients_per_100g", {})
            factor = weight_g / 100.0
            for key, target in [("Energy", "cal"), ("Protein", "pro"), ("Carbohydrate", "carb"), ("Total Fat", "fat")]:
                val = nutrients.get(key, {})
                if isinstance(val, dict):
                    v = val.get("value", 0) * factor
                    if target == "cal": total_calories += v
                    elif target == "pro": total_protein += v
                    elif target == "carb": total_carb += v
                    elif target == "fat": total_fat += v

    return {
        "total_mass": round(total_mass, 2), "total_calories": round(total_calories, 2),
        "total_fat": round(total_fat, 2), "total_carb": round(total_carb, 2),
        "total_protein": round(total_protein, 2), "food_names": food_names,
    }


# ──────────────────────────────────────────────
# Method 2: Direct LLM (zero-shot)
# ──────────────────────────────────────────────
async def run_direct_llm(image_bytes: bytes, provider: str = None) -> Dict:
    from langgraph_app.utils.tracked_llm import get_tracked_llm

    client = get_tracked_llm(module="food_recognition", node_name="eval_direct")
    structured_llm = client.with_structured_output(DirectNutritionEstimate)

    system_prompt = """[ROLE]
You are an expert nutritionist with deep knowledge of food composition.

[OBJECTIVE]
Look at the food image and estimate the TOTAL nutritional values for ALL food shown.

[CONSTRAINTS]
1. Estimate the total mass in grams of all food visible.
2. Estimate total calories (kcal), total fat (g), total carbohydrates (g), total protein (g).
3. Be as accurate as possible based on visual portion sizes.
4. Output ONLY the structured JSON."""

    message = build_image_message(image_bytes, "Estimate the total nutritional content of this meal.")

    for attempt in range(3):
        try:
            result = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt), message],
                config={"callbacks": []},
            )
            if isinstance(result, dict):
                return {
                    "total_mass": result.get("total_mass_g", 0),
                    "total_calories": result.get("total_calories_kcal", 0),
                    "total_fat": result.get("total_fat_g", 0),
                    "total_carb": result.get("total_carb_g", 0),
                    "total_protein": result.get("total_protein_g", 0),
                }
            else:
                return {
                    "total_mass": result.total_mass_g, "total_calories": result.total_calories_kcal,
                    "total_fat": result.total_fat_g, "total_carb": result.total_carb_g,
                    "total_protein": result.total_protein_g,
                }
        except Exception as e:
            print(f"    Direct attempt {attempt+1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(1)

    return {m: 0 for m in METRICS}


# ──────────────────────────────────────────────
# Method 3: Few-shot LLM (with reference images)
# ──────────────────────────────────────────────
async def run_fewshot_llm(image_bytes: bytes, ref_rows: pd.DataFrame, provider: str = None) -> Dict:
    """
    Give the LLM a few reference food images with their ground-truth nutrition,
    then ask it to estimate the test image. Visual calibration anchors.
    """
    from langgraph_app.utils.tracked_llm import get_tracked_llm

    client = get_tracked_llm(module="food_recognition", node_name="eval_fewshot")
    structured_llm = client.with_structured_output(DirectNutritionEstimate)

    system_prompt = """[ROLE]
You are an expert nutritionist with deep knowledge of food composition.

[OBJECTIVE]
You will be shown several REFERENCE food images with their verified nutritional values,
followed by a TARGET image. Use the references as calibration to estimate the target's nutrition.

[CONSTRAINTS]
1. Study the reference images carefully - they show real portion sizes with lab-measured values.
2. Use them to calibrate your sense of scale: how much food looks like X grams, Y calories, etc.
3. For the TARGET image, estimate total mass (g), calories (kcal), fat (g), carbohydrates (g), protein (g).
4. Output ONLY the structured JSON for the TARGET image."""

    # Build multi-image message: references first, then target
    content_parts = []

    for i, (_, ref_row) in enumerate(ref_rows.iterrows()):
        ref_url = image_to_base64_url(ref_row["rgb_image"])
        content_parts.append({"type": "image_url", "image_url": {"url": ref_url}})
        content_parts.append({"type": "text", "text": (
            f"REFERENCE {i+1}: "
            f"total_mass={ref_row['total_mass']:.1f}g, "
            f"calories={ref_row['total_calories']:.0f}kcal, "
            f"fat={ref_row['total_fat']:.1f}g, "
            f"carb={ref_row['total_carb']:.1f}g, "
            f"protein={ref_row['total_protein']:.1f}g"
        )})

    # Add target image
    target_url = image_to_base64_url(image_bytes)
    content_parts.append({"type": "image_url", "image_url": {"url": target_url}})
    content_parts.append({"type": "text", "text": "TARGET: Estimate the total nutritional content of this meal."})

    message = HumanMessage(content=content_parts)

    for attempt in range(3):
        try:
            result = await structured_llm.ainvoke(
                [SystemMessage(content=system_prompt), message],
                config={"callbacks": []},
            )
            if isinstance(result, dict):
                return {
                    "total_mass": result.get("total_mass_g", 0),
                    "total_calories": result.get("total_calories_kcal", 0),
                    "total_fat": result.get("total_fat_g", 0),
                    "total_carb": result.get("total_carb_g", 0),
                    "total_protein": result.get("total_protein_g", 0),
                }
            else:
                return {
                    "total_mass": result.total_mass_g, "total_calories": result.total_calories_kcal,
                    "total_fat": result.total_fat_g, "total_carb": result.total_carb_g,
                    "total_protein": result.total_protein_g,
                }
        except Exception as e:
            print(f"    Fewshot attempt {attempt+1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(1)

    return {m: 0 for m in METRICS}


# ──────────────────────────────────────────────
# Method 4: Fine-tuned MobileNetV3
# ──────────────────────────────────────────────
_finetuned_model = None
_finetuned_norm = None
_finetuned_transform = None


def _load_finetuned_model():
    """Lazy-load the fine-tuned model (once)."""
    global _finetuned_model, _finetuned_norm, _finetuned_transform
    if _finetuned_model is not None:
        return _finetuned_model, _finetuned_norm, _finetuned_transform

    import torch
    import torch.nn as nn
    from torchvision import models, transforms as T

    model_dir = EVAL_DIR / "model"
    model_path = model_dir / "best_model.pt"
    norm_path = model_dir / "norm_stats.json"

    if not model_path.exists() or not norm_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {model_dir}. Run: python eval/train_model.py"
        )

    # Load norm stats
    with open(norm_path, "r") as f:
        norm = json.load(f)
    _finetuned_norm = norm

    # Build model
    backbone = models.mobilenet_v3_small(weights=None)
    backbone.classifier = nn.Sequential(
        nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 5),
    )

    # Load weights
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    backbone.load_state_dict(state_dict)
    backbone.to(device)
    backbone.eval()
    _finetuned_model = (backbone, device)

    _finetuned_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"  Fine-tuned model loaded on {device}")
    return _finetuned_model, _finetuned_norm, _finetuned_transform


async def run_finetuned(image_bytes: bytes) -> Dict:
    """Run the fine-tuned MobileNetV3 model for nutrition estimation."""
    import torch
    from PIL import Image as PILImage

    (model, device), norm, transform = _load_finetuned_model()
    target_mean = np.array(norm["target_mean"], dtype=np.float32)
    target_std = np.array(norm["target_std"], dtype=np.float32)

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(img_tensor).cpu().numpy()[0]

    # Denormalize
    pred_raw = pred_norm * target_std + target_mean
    pred_raw = np.maximum(pred_raw, 0)  # nutrition values can't be negative

    return {
        "total_mass": round(float(pred_raw[0]), 2),
        "total_calories": round(float(pred_raw[1]), 2),
        "total_fat": round(float(pred_raw[2]), 2),
        "total_carb": round(float(pred_raw[3]), 2),
        "total_protein": round(float(pred_raw[4]), 2),
    }


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def compute_wmape(ground_truths: List[float], predictions: List[float]) -> float:
    gt = np.array(ground_truths, dtype=float)
    pred = np.array(predictions, dtype=float)
    denom = gt.sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(gt - pred).sum() / denom)


def compute_all_wmape(records: List[Dict], method: str) -> Dict[str, float]:
    results = {}
    for metric in METRICS:
        gts = [r["ground_truth"][metric] for r in records]
        preds = [r[method][metric] for r in records]
        results[metric] = compute_wmape(gts, preds)
    return results


def per_sample_ape(gt_val: float, pred_val: float) -> Optional[float]:
    if gt_val > 0:
        return round(abs(gt_val - pred_val) / gt_val, 6)
    return None


# ──────────────────────────────────────────────
# Main Evaluation Loop
# ──────────────────────────────────────────────
METHODS = ["graph", "direct", "fewshot", "finetuned"]


async def evaluate(n: int, provider: str = None, seed: int = 42, resume: bool = False):
    # Resolve model name
    from langgraph_app.config import config as app_config
    resolved_provider = (provider or app_config.LLM_PROVIDER).lower()
    if resolved_provider == "gemini":
        model_name = app_config.GEMINI_MODEL_NAME
    elif resolved_provider == "openai":
        model_name = app_config.OPENAI_MODEL_NAME
    else:
        model_name = app_config.BEDROCK_CLAUDE_MODEL_NAME
    print(f"Model: {model_name} (provider: {resolved_provider})")

    dataset = load_dataset()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = RESULTS_DIR / "checkpoint.json"
    results_path = RESULTS_DIR / f"eval_{timestamp}.json"

    # Load checkpoint
    records = []
    evaluated_ids = set()
    if resume and checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        records = checkpoint.get("records", [])
        evaluated_ids = {r["dish_id"] for r in records}
        print(f"Resuming from checkpoint: {len(records)} already done")

    # Sample test IDs
    random.seed(seed)
    all_ids = dataset["dish"].tolist()
    random.shuffle(all_ids)
    remaining_ids = [did for did in all_ids if did not in evaluated_ids]
    sample_ids = remaining_ids[:max(0, n - len(records))]
    total_to_run = len(sample_ids)
    test_id_set = set(sample_ids) | evaluated_ids

    # Select few-shot reference images (stratified, excluding test set)
    ref_rows = select_fewshot_references(dataset, test_id_set, seed=seed)
    ref_dish_ids = ref_rows["dish"].tolist()
    print(f"  Reference dishes: {ref_dish_ids}")

    if total_to_run == 0 and records:
        print("All samples already evaluated.")
    else:
        print(f"\nRunning {total_to_run} samples × 3 methods...\n")

    for idx, dish_id in enumerate(sample_ids):
        row = dataset[dataset["dish"] == dish_id].iloc[0]
        image_bytes = row["rgb_image"]
        ground_truth = {m: float(row[m]) for m in METRICS}

        current = idx + 1 + len(evaluated_ids)
        total = len(evaluated_ids) + total_to_run
        print(f"[{current}/{total}] {dish_id}")
        print(f"  GT: mass={ground_truth['total_mass']:.1f}g, "
              f"cal={ground_truth['total_calories']:.0f}kcal, "
              f"fat={ground_truth['total_fat']:.1f}g, "
              f"carb={ground_truth['total_carb']:.1f}g, "
              f"prot={ground_truth['total_protein']:.1f}g")

        results_per_method = {}
        times_per_method = {}

        # --- Graph ---
        t0 = time.time()
        try:
            graph_result = await run_graph_recognition(image_bytes, provider)
            times_per_method["graph"] = time.time() - t0
            results_per_method["graph"] = graph_result
            print(f"  Graph  ({times_per_method['graph']:.1f}s): "
                  f"mass={graph_result['total_mass']:.1f}g, cal={graph_result['total_calories']:.0f}, "
                  f"foods={graph_result.get('food_names', [])}")
        except Exception as e:
            times_per_method["graph"] = time.time() - t0
            results_per_method["graph"] = {m: 0 for m in METRICS}
            results_per_method["graph"]["error"] = str(e)
            print(f"  Graph  FAILED ({times_per_method['graph']:.1f}s): {e}")

        await asyncio.sleep(1)

        # --- Direct ---
        t0 = time.time()
        try:
            direct_result = await run_direct_llm(image_bytes, provider)
            times_per_method["direct"] = time.time() - t0
            results_per_method["direct"] = direct_result
            print(f"  Direct ({times_per_method['direct']:.1f}s): "
                  f"mass={direct_result['total_mass']:.1f}g, cal={direct_result['total_calories']:.0f}")
        except Exception as e:
            times_per_method["direct"] = time.time() - t0
            results_per_method["direct"] = {m: 0 for m in METRICS}
            results_per_method["direct"]["error"] = str(e)
            print(f"  Direct FAILED ({times_per_method['direct']:.1f}s): {e}")

        await asyncio.sleep(1)

        # --- Few-shot ---
        t0 = time.time()
        try:
            fewshot_result = await run_fewshot_llm(image_bytes, ref_rows, provider)
            times_per_method["fewshot"] = time.time() - t0
            results_per_method["fewshot"] = fewshot_result
            print(f"  Fewshot({times_per_method['fewshot']:.1f}s): "
                  f"mass={fewshot_result['total_mass']:.1f}g, cal={fewshot_result['total_calories']:.0f}")
        except Exception as e:
            times_per_method["fewshot"] = time.time() - t0
            results_per_method["fewshot"] = {m: 0 for m in METRICS}
            results_per_method["fewshot"]["error"] = str(e)
            print(f"  Fewshot FAILED ({times_per_method['fewshot']:.1f}s): {e}")

        # --- Fine-tuned ---
        t0 = time.time()
        try:
            finetuned_result = await run_finetuned(image_bytes)
            times_per_method["finetuned"] = time.time() - t0
            results_per_method["finetuned"] = finetuned_result
            print(f"  Tuned  ({times_per_method['finetuned']:.1f}s): "
                  f"mass={finetuned_result['total_mass']:.1f}g, cal={finetuned_result['total_calories']:.0f}")
        except Exception as e:
            times_per_method["finetuned"] = time.time() - t0
            results_per_method["finetuned"] = {m: 0 for m in METRICS}
            results_per_method["finetuned"]["error"] = str(e)
            print(f"  Tuned  FAILED ({times_per_method['finetuned']:.1f}s): {e}")

        # Build record
        record = {
            "dish_id": dish_id,
            "ground_truth": ground_truth,
        }
        for method in METHODS:
            res = results_per_method[method]
            record[method] = {m: res.get(m, 0) for m in METRICS}
            record[f"{method}_ape"] = {
                m: per_sample_ape(ground_truth[m], res.get(m, 0)) for m in METRICS
            }
            record[f"{method}_time_s"] = round(times_per_method.get(method, 0), 2)
            if "error" in res:
                record[f"{method}_error"] = res["error"]
            if method == "graph" and "food_names" in res:
                record["graph_foods"] = res["food_names"]

        records.append(record)

        # Save checkpoint
        checkpoint_data = {
            "model": model_name, "provider": resolved_provider,
            "seed": seed, "n": n, "completed": len(records),
            "fewshot_refs": ref_dish_ids, "records": records,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        print()

        # Rate limit between samples
        if idx < total_to_run - 1:
            await asyncio.sleep(1)

    # ── Final metrics ──
    valid_records = records  # keep all, even failures (they score 0)

    print("=" * 80)
    print(f"EVALUATION RESULTS  |  Model: {model_name}  |  {len(records)} samples")
    print("=" * 80)

    all_wmape = {}
    for method in METHODS:
        all_wmape[method] = compute_all_wmape(valid_records, method)

    header = f"{'Metric':<18} {'Graph':>10} {'Direct':>10} {'Fewshot':>10} {'Winner':>10}"
    print(header)
    print("-" * len(header))
    for metric in METRICS:
        vals = {m: all_wmape[m][metric] for m in METHODS}
        best = min(vals, key=vals.get)
        print(f"{metric:<18} {vals['graph']:>9.1%} {vals['direct']:>9.1%} {vals['fewshot']:>9.1%} {best:>10}")

    # Timing
    print()
    for method in METHODS:
        avg_t = np.mean([r.get(f"{method}_time_s", 0) for r in records])
        errs = sum(1 for r in records if f"{method}_error" in r)
        print(f"  {method:<8} avg={avg_t:.1f}s  errors={errs}/{len(records)}")

    # Save
    final_output = {
        "meta": {
            "timestamp": timestamp, "model": model_name, "provider": resolved_provider,
            "seed": seed, "n_evaluated": len(records),
            "fewshot_refs": ref_dish_ids, "num_fewshot_examples": NUM_FEWSHOT_EXAMPLES,
        },
        "wmape": {m: {k: round(v, 6) for k, v in all_wmape[m].items()} for m in METHODS},
        "records": records,
    }
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")
    return final_output


def main():
    parser = argparse.ArgumentParser(description="WABI Food Recognition Evaluation")
    parser.add_argument("--n", type=int, default=20, help="Number of samples (default: 20)")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    asyncio.run(evaluate(n=args.n, provider=args.provider, seed=args.seed, resume=args.resume))


if __name__ == "__main__":
    main()
