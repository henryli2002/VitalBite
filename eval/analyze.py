"""
Analyze evaluation results: per-sample and aggregate wMAPE for all methods.

Usage:
    python eval/analyze.py eval/results/eval_XXXX.json
"""

import sys
import json
import numpy as np
from pathlib import Path

METRICS = ["total_mass", "total_calories", "total_fat", "total_carb", "total_protein"]
METRIC_LABELS = {
    "total_mass": "Mass(g)",
    "total_calories": "Cal(kcal)",
    "total_fat": "Fat(g)",
    "total_carb": "Carb(g)",
    "total_protein": "Prot(g)",
}


def per_sample_error(gt: float, pred: float) -> float:
    if gt == 0:
        return float("nan")
    return abs(gt - pred) / gt


def analyze(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    records = data.get("records", [])
    model = meta.get("model", "unknown")

    # Auto-detect which methods are present
    methods = []
    for m in ["graph", "direct", "fewshot", "finetuned"]:
        if any(m in r for r in records):
            methods.append(m)

    print("=" * 80)
    print(f"File: {Path(filepath).name}  |  Model: {model}  |  Samples: {len(records)}")
    if "fewshot_refs" in meta:
        print(f"Few-shot refs: {meta['fewshot_refs']}")
    print("=" * 80)

    # ── Per-sample table ──
    errors = {method: {m: [] for m in METRICS} for method in methods}

    # Header
    col_w = 8
    print(f"\n{'#':<4} {'dish_id':<20}", end="")
    for metric in METRICS:
        label = METRIC_LABELS[metric]
        for method in methods:
            print(f" {method[0].upper()}_{label:>{col_w - 2}}", end="")
        print(" |", end="")
    print()
    print("-" * (25 + len(METRICS) * len(methods) * (col_w + 3) + len(METRICS) * 2))

    for i, rec in enumerate(records):
        gt = rec["ground_truth"]
        print(f"{i + 1:<4} {rec['dish_id']:<20}", end="")
        for metric in METRICS:
            for method in methods:
                pred = rec.get(method, {}).get(metric, 0)
                err = per_sample_error(gt[metric], pred)
                errors[method][metric].append(err)
                s = f"{err:.0%}" if not np.isnan(err) else "N/A"
                print(f" {s:>{col_w + 1}}", end="")
            print(" |", end="")
        print()

    # ── Aggregate: wMAPE ──
    print(f"\n{'=' * 80}")
    print("AGGREGATE wMAPE (lower is better)")
    print("=" * 80)
    header = f"{'Metric':<15}" + "".join(f" {m:>10}" for m in methods) + "     Winner"
    print(header)
    print("-" * len(header))
    for metric in METRICS:
        vals = {}
        for method in methods:
            gts = [r["ground_truth"][metric] for r in records]
            preds = [r.get(method, {}).get(metric, 0) for r in records]
            gt_sum = sum(gts)
            vals[method] = (
                sum(abs(g - p) for g, p in zip(gts, preds)) / gt_sum
                if gt_sum
                else float("nan")
            )
        best = min(vals, key=vals.get)
        line = f"{METRIC_LABELS[metric]:<15}"
        for method in methods:
            line += f" {vals[method]:>9.1%}"
        line += f"     {best}"
        print(line)

    # ── Aggregate: MAPE (per-sample average) ──
    print(f"\n{'Metric':<15}" + "".join(f" {m:>10}" for m in methods) + "     Winner")
    print("-" * (15 + len(methods) * 11 + 10))
    for metric in METRICS:
        vals = {}
        for method in methods:
            vals[method] = np.nanmean(errors[method][metric])
        best = min(vals, key=vals.get)
        line = f"{METRIC_LABELS[metric]:<15}"
        for method in methods:
            line += f" {vals[method]:>9.1%}"
        line += f"     {best}"
        print(line)
    print("(MAPE = per-sample average, more sensitive to outliers)")

    # ── Win rate ──
    print(f"\n{'Metric':<15}", end="")
    for method in methods:
        print(f" {method:>10}", end="")
    print()
    print("-" * (15 + len(methods) * 11))
    for metric in METRICS:
        wins = {m: 0 for m in methods}
        total_valid = 0
        for i in range(len(records)):
            method_errors = {}
            for method in methods:
                e = errors[method][metric][i]
                if not np.isnan(e):
                    method_errors[method] = e
            if len(method_errors) == len(methods):
                total_valid += 1
                best = min(method_errors, key=method_errors.get)
                wins[best] += 1
        line = f"{METRIC_LABELS[metric]:<15}"
        for method in methods:
            pct = wins[method] / total_valid if total_valid else 0
            line += f" {wins[method]:>4}({pct:.0%})"
        print(line)

    # ── Timing ──
    print()
    for method in methods:
        times = [r.get(f"{method}_time_s", 0) for r in records]
        errs = sum(1 for r in records if f"{method}_error" in r)
        print(f"  {method:<8} avg={np.mean(times):.1f}s  errors={errs}/{len(records)}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval/analyze.py <result.json>")
        sys.exit(1)
    for f in sys.argv[1:]:
        analyze(f)


if __name__ == "__main__":
    main()
