"""Track E — Heuristic baseline (zero LLM calls).

Rule:
  - image_marker present  → recognition
  - no image_marker       → recommendation

This upper-bounds what a trivial if-else can achieve and establishes a
minimum bar for all other tracks.

Run:
    python eval/intent/run_track_e.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runner_lib import (
    print_report,
    load_jsonl,
    write_trace,
)


def predict(sample: dict) -> tuple[str, float]:
    if sample.get("image_marker"):
        return "recognition", 1.0
    return "recommendation", 1.0


def run(dataset_path: Path) -> None:
    samples = load_jsonl(dataset_path)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    trace_path = results_dir / "trace_track_e.jsonl"

    correct = 0
    traces: list[dict] = []

    print(f"\nTrack E  |  heuristic (image→recognition, else→recommendation)")
    print(f"{'ID':<20} {'Expected':<16} {'Predicted':<16} {'Conf':>6}  OK?")
    print("-" * 68)

    for sample in samples:
        predicted, confidence = predict(sample)
        expected = sample["expected_intent"]
        ok = predicted == expected
        if ok:
            correct += 1

        mark = "✓" if ok else "✗"
        print(f"{sample['id']:<20} {expected:<16} {predicted:<16} {confidence:>6.2f}  {mark}")

        traces.append({
            "id": sample["id"],
            "expected_intent": expected,
            "predicted_intent": predicted,
            "confidence": confidence,
            "reasoning": "heuristic: image→recognition, no image→recommendation",
            "correct": ok,
            "elapsed_ms": 0,
        })

    print_report(traces, track="E")
    write_trace(traces, trace_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track E: heuristic baseline")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "dataset" / "test.jsonl"),
    )
    args = parser.parse_args()
    run(Path(args.dataset))
