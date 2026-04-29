"""Evaluate all LoRA fine-tuned adapters on the test set.

For each experiment adapter, starts a temporary mlx_lm server via omlx SDK
OR calls the model directly via mlx_lm.generate. We use the direct generation
path (no server needed) to avoid port conflicts.

Usage:
    python eval/intent/train/run_eval_sweep.py
    python eval/intent/train/run_eval_sweep.py --exp exp_A_all_8layers

Output:
    eval/intent/results/lora/ — one trace_<exp>.jsonl per experiment
    Updates eval/intent/results/result.md LoRA section
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INTENT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(INTENT_DIR))

from runner_lib import parse_output, print_report, load_jsonl, write_trace, build_chat_messages

SITE = "/Applications/oMLX.app/Contents/Python/framework-mlx-framework/lib/python3.11/site-packages"
sys.path.insert(0, SITE)

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

BASE_MODEL = "/Users/henryli/.omlx/models/Qwen3.5-0.8B-OptiQ-4bit"
ADAPTERS_DIR = TRAIN_DIR / "adapters"
DATASET = INTENT_DIR / "dataset" / "test.jsonl"
RESULTS_DIR = INTENT_DIR / "results" / "lora"

EXPS = [
    "exp_A_all_8layers",
    "exp_B_attn_8layers",
    "exp_C_mlp_8layers",
    "exp_D_all_16layers",
]

EXP_LABELS = {
    "exp_A_all_8layers":  "LoRA-A (all, 8L, r=8)",
    "exp_B_attn_8layers": "LoRA-B (attn, 8L, r=8)",
    "exp_C_mlp_8layers":  "LoRA-C (mlp, 8L, r=8)",
    "exp_D_all_16layers": "LoRA-D (all, 16L, r=8)",
}


def eval_experiment(exp_name: str, samples: list[dict]) -> list[dict]:
    adapter_path = str(ADAPTERS_DIR / exp_name)
    print(f"\n{'═'*60}")
    print(f"Loading: {exp_name}")
    print(f"Adapter: {adapter_path}")

    model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)

    traces = []
    for i, sample in enumerate(samples):
        msgs = build_chat_messages(sample, profile=None)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        t0 = time.perf_counter()
        raw = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=128,
            sampler=make_sampler(temp=0.1),
            verbose=False,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        parsed = parse_output(raw)
        if parsed is None:
            predicted = "chitchat"
            confidence = 0.0
            reasoning = f"[parse error] {raw[:80]}"
        else:
            predicted = parsed["intent"]
            confidence = parsed["confidence"]
            reasoning = parsed["reasoning"]

        expected = sample["expected_intent"]
        ok = predicted == expected
        mark = "✓" if ok else "✗"
        print(f"  [{i+1:3d}] {expected:<16} → {predicted:<16} {confidence:.2f}  {elapsed_ms}ms  {mark}")

        traces.append({
            "id": sample["id"],
            "category": sample.get("category", "normal"),
            "expected_intent": expected,
            "predicted_intent": predicted,
            "confidence": confidence,
            "reasoning": reasoning,
            "correct": ok,
            "elapsed_ms": elapsed_ms,
            "raw_output": raw,
        })

    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, help="Run single experiment by name")
    parser.add_argument(
        "--dataset",
        default=str(DATASET),
    )
    args = parser.parse_args()

    samples = load_jsonl(Path(args.dataset))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    exps = [args.exp] if args.exp else EXPS

    for exp_name in exps:
        adapter_path = ADAPTERS_DIR / exp_name
        if not adapter_path.exists():
            print(f"[skip] Adapter not found: {adapter_path}", file=sys.stderr)
            continue

        traces = eval_experiment(exp_name, samples)

        trace_path = RESULTS_DIR / f"trace_{exp_name}.jsonl"
        write_trace(traces, trace_path)

        label = EXP_LABELS.get(exp_name, exp_name)
        print_report(traces, track=label)

    print("\nAll done.")


if __name__ == "__main__":
    main()
