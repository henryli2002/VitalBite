"""Prepare MLX-LM LoRA training data from train.jsonl.

Converts each training sample into a chat-format record that mlx-lm expects:
    {"messages": [system, ...history, user, assistant]}

The assistant message is the ground-truth INTENT/CONFIDENCE/REASONING output,
which is the LoRA training target.

Output files (written to train/):
    mlx_train.jsonl   — 90% of train.jsonl (shuffled, seed=42)
    mlx_valid.jsonl   — remaining 10%

Usage:
    python eval/intent/train/prepare_mlx.py
    python eval/intent/train/prepare_mlx.py --seed 0 --valid-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# allow sibling import of runner_lib
INTENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(INTENT_DIR))

from runner_lib import _build_system_prompt
from gen_profiles import make_profile

TRAIN_JSONL = INTENT_DIR / "dataset" / "train.jsonl"
OUT_DIR = Path(__file__).parent


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_assistant_turn(sample: dict) -> str:
    """Build the ground-truth assistant output string."""
    conf = sample["confidence"]
    reasoning = sample["reasoning"]
    intent = sample["expected_intent"]
    return (
        f"INTENT: {intent}\n"
        f"CONFIDENCE: {conf:.2f}\n"
        f"REASONING: {reasoning}"
    )


def build_mlx_record(sample: dict, profile: dict | None = None) -> dict:
    """Convert a dataset sample to mlx-lm chat format."""
    msgs: list[dict] = [{"role": "system", "content": _build_system_prompt(profile)}]

    for turn in sample.get("history", []):
        msgs.append({"role": turn["role"], "content": turn["content"]})

    utterance = sample.get("utterance", "")
    image_marker = sample.get("image_marker")
    if image_marker:
        uuid = image_marker["uuid"]
        marker = f"<attached_image uuid={uuid}/>"
        current_content = f"{utterance} {marker}".strip() if utterance.strip() else marker
    else:
        current_content = utterance

    msgs.append({"role": "user", "content": current_content})
    msgs.append({"role": "assistant", "content": build_assistant_turn(sample)})

    return {"messages": msgs}


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--no-profile", action="store_true", help="Build prompts without user profile injection")
    args = parser.parse_args()

    samples = load_jsonl(TRAIN_JSONL)
    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_valid = max(1, int(len(samples) * args.valid_ratio))
    valid_samples = samples[:n_valid]
    train_samples = samples[n_valid:]

    def _profile(i):
        return None if args.no_profile else make_profile(seed=args.seed, index=i)

    train_records = [build_mlx_record(s, _profile(i)) for i, s in enumerate(train_samples)]
    valid_records = [build_mlx_record(s, _profile(i)) for i, s in enumerate(valid_samples)]

    train_path = OUT_DIR / "mlx_train.jsonl"
    valid_path = OUT_DIR / "mlx_valid.jsonl"
    write_jsonl(train_records, train_path)
    write_jsonl(valid_records, valid_path)

    print(f"seed         : {args.seed}")
    print(f"profile      : {'disabled' if args.no_profile else 'enabled'}")
    print(f"train records: {len(train_records)}")
    print(f"valid records: {len(valid_records)}")
    print(f"Wrote → {train_path}")
    print(f"Wrote → {valid_path}")

    # ── quick sanity check ────────────────────────────────────────────────────
    sample = train_records[0]
    print("\n--- first record assistant turn ---")
    print(sample["messages"][-1]["content"])


if __name__ == "__main__":
    main()
