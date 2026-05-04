"""Evaluate reasoning quality of intent router outputs using Gemini.

Samples are shuffled to avoid order bias.
Evaluated in batches (default 10/call) to reduce API cost.
Expected intent is NOT passed — only reasoning quality is scored (0-4).

Usage:
    python eval/intent/eval_reasoning.py
    python eval/intent/eval_reasoning.py --model gemini-3.1-pro-preview --batch 10
    python eval/intent/eval_reasoning.py --only 9B-zero

Output:
    eval/intent/results/reasoning_eval.jsonl   — per-sample scores
    prints summary table to stdout
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from google import genai
from google.genai import types

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "test.jsonl"

MODELS = {
    "0.8B-zero":  "no_profile/trace_Qwen3.5-0.8B-OptiQ-4bit.jsonl",
    "0.8B-LoRA-expA": "lora/trace_exp_A_all_8layers.jsonl",
    "9B-zero":    "no_profile/trace_Qwen3.5-9B-OptiQ-4bit.jsonl",
    "gemma-4e4b": "no_profile/trace_gemma-4-e4b-it-4bit.jsonl",
    "A-Gemini":   "no_profile/trace_track_a.jsonl",
}

EVAL_SYSTEM = """\
Score each reasoning (0-4). Evaluate independently; ignore whether the predicted intent is correct.
4=cites specific signals from user message, 3=right but generic, 2=partial/gap, 1=weak, 0=circular/empty.
Reply for each: SAMPLE <n> / SCORE: <0-4> / CRITIQUE: <10 words max>
"""

SAMPLE_TMPL = """\
[{n}] User: {utterance}{image_note}
Predicted intent: {predicted}
Reasoning: {reasoning}"""


def load_dataset(path: Path) -> dict[str, dict]:
    with open(path) as f:
        return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def load_trace(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def parse_batch_output(text: str, n: int) -> list[tuple[int | None, str]]:
    results: list[tuple[int | None, str]] = [(None, "")] * n
    # Normalize separators: "SAMPLE 1 / SCORE" → "SAMPLE 1\nSCORE"
    text = re.sub(r"\s*/\s*", "\n", text)
    blocks = re.split(r"SAMPLE\s+(\d+)", text)
    i = 1
    while i + 1 < len(blocks):
        try:
            idx = int(blocks[i]) - 1
        except ValueError:
            i += 2
            continue
        content = blocks[i + 1]
        score_m = re.search(r"SCORE:\s*(\d)", content)
        critique_m = re.search(r"CRITIQUE:\s*(.+)", content)
        score = int(score_m.group(1)) if score_m else None
        if score is not None:
            score = max(0, min(4, score))
        critique = critique_m.group(1).strip() if critique_m else ""
        if 0 <= idx < n:
            results[idx] = (score, critique)
        i += 2
    return results


def evaluate_batch(
    samples_batch: list[dict],
    dataset: dict[str, dict],
    client: genai.Client,
    model: str,
    batch_idx: int,
) -> list[tuple[int | None, str]]:
    parts = []
    for j, sample in enumerate(samples_batch, 1):
        ds = dataset.get(sample["id"], {})
        utterance = ds.get("utterance", "(unknown)")
        has_image = bool(ds.get("image_marker"))
        image_note = "\n[An image is attached to this message]" if has_image else ""
        parts.append(SAMPLE_TMPL.format(
            n=j,
            utterance=utterance,
            image_note=image_note,
            predicted=sample["predicted_intent"],
            reasoning=sample["reasoning"],
        ))

    user_text = "\n\n".join(parts)

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=EVAL_SYSTEM,
                    temperature=0.0,
                    max_output_tokens=2048,
                ),
            )
            return parse_batch_output(resp.text, len(samples_batch))
        except Exception as e:
            if attempt == 2:
                print(f"  [error] batch {batch_idx}: {e}", file=sys.stderr)
                return [(None, f"API error: {e}")] * len(samples_batch)
            time.sleep(2 ** attempt)
    return [(None, "")] * len(samples_batch)


def evaluate_model(
    label: str,
    traces: list[dict],
    dataset: dict[str, dict],
    client: genai.Client,
    model: str,
    batch_size: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(traces)
    rng.shuffle(shuffled)

    all_results = []
    total = len(shuffled)
    n_batches = (total + batch_size - 1) // batch_size

    for b in range(n_batches):
        batch = shuffled[b * batch_size : (b + 1) * batch_size]
        scores = evaluate_batch(batch, dataset, client, model, b + 1)

        for sample, (score, critique) in zip(batch, scores):
            mark = str(score) if score is not None else "?"
            done = b * batch_size + batch.index(sample) + 1
            print(f"  [{done:3d}/{total}] {sample['predicted_intent']:<16} score={mark}  {sample['id'][:38]}")
            all_results.append({
                "model": label,
                "id": sample["id"],
                "category": sample.get("category", ""),
                "predicted_intent": sample["predicted_intent"],
                "expected_intent": sample["expected_intent"],
                "correct": sample.get("correct", False),
                "confidence": sample.get("confidence", 0.0),
                "reasoning": sample["reasoning"],
                "reasoning_score": score,
                "critique": critique,
            })

        time.sleep(0.5)

    return all_results


def print_summary(all_results: list[dict]):
    from collections import defaultdict

    by_model: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    print("\n" + "═" * 76)
    print(f"{'Model':<14} {'n':>4} {'avg_conf':>9} {'avg_reason':>11}   score dist (0→4)")
    print("─" * 76)

    for label, rows in by_model.items():
        scored = [r for r in rows if r["reasoning_score"] is not None]
        avg_conf = sum(r["confidence"] for r in rows) / len(rows)
        avg_score = sum(r["reasoning_score"] for r in scored) / len(scored) if scored else 0
        dist = [sum(1 for r in scored if r["reasoning_score"] == s) for s in range(5)]
        dist_str = "  ".join(f"{s}:{d:2d}" for s, d in enumerate(dist))
        print(f"{label:<14} {len(rows):>4} {avg_conf:>9.3f} {avg_score:>11.3f}   {dist_str}")

    print("═" * 76)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--batch", type=int, default=10, help="samples per API call")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None, help="run only one model label")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    dataset = load_dataset(DATASET_PATH)

    out_path = RESULTS_DIR / "reasoning_eval.jsonl"
    all_results: list[dict] = []

    models_to_run = {k: v for k, v in MODELS.items() if args.only is None or k == args.only}

    for label, rel_path in models_to_run.items():
        trace_path = RESULTS_DIR / rel_path
        if not trace_path.exists():
            print(f"[skip] {label}: not found", file=sys.stderr)
            continue
        traces = load_trace(trace_path)
        print(f"\n{'═'*60}\nEvaluating: {label}  ({len(traces)} samples, batch={args.batch})\n{'═'*60}")
        results = evaluate_model(label, traces, dataset, client, args.model, args.batch, args.seed)
        all_results.extend(results)

    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved → {out_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
