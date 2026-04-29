"""Aggregate all track traces and print a comparison table.

Run after all (or any subset of) track runners have written their traces:
    python eval/intent/eval_report.py

Reads:
    results/trace_track_a_seed42.jsonl   (Track A — Gemini API)
    results/trace_track_b.jsonl          (Track B — Qwen3.5-9B zero-shot)
    results/trace_track_c.jsonl          (Track C — Gemma4-4B zero-shot)
    results/trace_track_d.jsonl          (Track D — LoRA fine-tuned)
    results/trace_track_e.jsonl          (Track E — heuristic)
"""

from __future__ import annotations

import json
from pathlib import Path

from runner_lib import compute_ece, ALLOWED_INTENTS

RESULTS_DIR = Path(__file__).parent / "results"

TRACKS = [
    ("A", "Gemini API (live router)", "trace_track_a_seed42.jsonl"),
    ("B", "Qwen3.5-9B zero-shot", "trace_track_b.jsonl"),
    ("C", "Gemma4-4B zero-shot", "trace_track_c.jsonl"),
    ("D", "LoRA fine-tuned", "trace_track_d.jsonl"),
    ("E", "Heuristic (if-else)", "trace_track_e.jsonl"),
]


def load_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def macro_f1(traces: list[dict]) -> float:
    f1_sum = 0.0
    for intent in ALLOWED_INTENTS:
        tp = sum(1 for t in traces if t["expected_intent"] == intent and t["predicted_intent"] == intent)
        fp = sum(1 for t in traces if t["expected_intent"] != intent and t["predicted_intent"] == intent)
        fn = sum(1 for t in traces if t["expected_intent"] == intent and t["predicted_intent"] != intent)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1_sum += f1
    return f1_sum / len(ALLOWED_INTENTS)


def pct(traces: list[dict], n: int) -> int:
    latencies = sorted(t.get("elapsed_ms", 0) for t in traces)
    if not latencies:
        return 0
    return latencies[min(int(len(latencies) * n / 100), len(latencies) - 1)]


def main() -> None:
    rows = []
    for track_id, label, filename in TRACKS:
        traces = load_traces(RESULTS_DIR / filename)
        if not traces:
            rows.append((track_id, label, None))
            continue

        total = len(traces)
        correct = sum(1 for t in traces if t["correct"])
        acc = correct / total
        mf1 = macro_f1(traces)
        ece = compute_ece(traces)
        low_conf = sum(1 for t in traces if t.get("confidence", 1.0) < 0.6) / total
        p50 = pct(traces, 50)
        p95 = pct(traces, 95)

        rows.append((track_id, label, {
            "n": total,
            "acc": acc,
            "macro_f1": mf1,
            "ece": ece,
            "low_conf": low_conf,
            "p50": p50,
            "p95": p95,
        }))

    # ── header ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  Intent Router Evaluation — Comparison Table")
    print(f"{'─'*90}")
    print(f"  {'Track':<4} {'Description':<28} {'N':>5} {'Acc':>7} {'F1':>7} "
          f"{'ECE':>7} {'LowConf':>8} {'p50ms':>7} {'p95ms':>7}")
    print(f"{'─'*90}")

    for track_id, label, m in rows:
        if m is None:
            print(f"  {track_id:<4} {label:<28}  (no trace file found)")
            continue
        print(
            f"  {track_id:<4} {label:<28} {m['n']:>5} {m['acc']:>7.1%} "
            f"{m['macro_f1']:>7.3f} {m['ece']:>7.3f} {m['low_conf']:>8.1%} "
            f"{m['p50']:>7} {m['p95']:>7}"
        )

    print(f"{'═'*90}\n")


if __name__ == "__main__":
    main()
