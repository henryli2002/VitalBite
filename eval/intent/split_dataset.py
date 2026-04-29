#!/usr/bin/env python3
"""Split dataset into train/test sets.

Normal  : 300 total → 250 train, 50 test
Abnormal: 100 total → 50 train,  50 test

Usage:
    python split_dataset.py [--seed 42]

Output:
    dataset/train.jsonl
    dataset/test.jsonl
"""

import json
import random
import argparse
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split(records, n_train, rng):
    shuffled = records[:]
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    normal   = load_jsonl(DATASET_DIR / "normal.jsonl")
    abnormal = load_jsonl(DATASET_DIR / "abnormal.jsonl")

    assert len(normal)   == 300, f"Expected 300 normal, got {len(normal)}"
    assert len(abnormal) == 100, f"Expected 100 abnormal, got {len(abnormal)}"

    normal_train,   normal_test   = split(normal,   250, rng)
    abnormal_train, abnormal_test = split(abnormal,  50, rng)

    train = normal_train + abnormal_train
    test  = normal_test  + abnormal_test

    rng.shuffle(train)
    rng.shuffle(test)

    write_jsonl(train, DATASET_DIR / "train.jsonl")
    write_jsonl(test,  DATASET_DIR / "test.jsonl")

    # ── report ────────────────────────────────────────────────────────────────
    def intent_dist(records):
        from collections import Counter
        return dict(Counter(r["expected_intent"] for r in records))

    def category_dist(records):
        from collections import Counter
        return dict(Counter(r["category"] for r in records))

    print(f"seed           : {args.seed}")
    print(f"train total    : {len(train)}  ({len(normal_train)} normal + {len(abnormal_train)} abnormal)")
    print(f"test  total    : {len(test)}   ({len(normal_test)} normal + {len(abnormal_test)} abnormal)")
    print(f"\ntrain intent distribution : {intent_dist(train)}")
    print(f"test  intent distribution : {intent_dist(test)}")
    print(f"\ntrain category distribution : {category_dist(train)}")
    print(f"test  category distribution : {category_dist(test)}")
    print(f"\nWrote → {DATASET_DIR / 'train.jsonl'}")
    print(f"Wrote → {DATASET_DIR / 'test.jsonl'}")


if __name__ == "__main__":
    main()
