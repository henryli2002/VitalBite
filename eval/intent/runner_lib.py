"""Shared utilities for all intent-eval track runners."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

# ── prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
[ROLE]
You are the intent router for WABI, an AI food assistant.

[OBJECTIVE]
Identify the user's primary goal based on the conversation history. Pay close attention to whether a food image is present.

[CONTEXT]
Current Time: 14:00 (not meal time)

[IMAGE FORMAT]
Attached images are represented as server-injected markers at the end of a user message:
  <attached_image uuid=<32-hex-id>/>
If such a marker is present in the current message, treat it as a food image being attached.

[INTENT RULES]
1. "recognition": Goal is to identify food/nutrition from an image. If an <attached_image> marker is present, confidence for this intent should be VERY HIGH (>0.9).
2. "recommendation": Finding places to eat. Triggers on explicit requests or implicit signs of hunger during meal times.
3. "goalplanning": Diet planning, habit building, long-term nutrition goals, or questions about eating history and patterns.
4. "chitchat": Default for everything else — greetings, unrelated topics, vague inputs without context, requests for image recognition WITHOUT an <attached_image> marker present, or meaningless noise.

[CONSTRAINTS]
Output with EXACTLY this plain-text format (no markdown, no code block, no extra labels):
INTENT: <recognition|recommendation|chitchat|goalplanning>
CONFIDENCE: <0.00-1.00>
REASONING: <brief but specific why this intent fits the user message>"""

ALLOWED_INTENTS = {"recognition", "recommendation", "chitchat", "goalplanning"}


def _build_system_prompt(profile: Optional[dict] = None) -> str:
    profile_section = ""
    if profile:
        lines = "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in profile.items() if v
        )
        profile_section = f"\n\nUser Profile & Health Information:\n{lines}"
    return SYSTEM_PROMPT.replace(
        "[IMAGE FORMAT]",
        profile_section + "\n\n[IMAGE FORMAT]" if profile_section else "[IMAGE FORMAT]",
    )


# ── message builder ───────────────────────────────────────────────────────────

def build_chat_messages(sample: dict, profile: Optional[dict] = None) -> list[dict]:
    """Convert a dataset sample to OpenAI-style message list (text-only).

    Image markers are embedded as <attached_image uuid=.../> text so local
    models without vision can still reason about image presence.
    """
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
    return msgs


# ── output parser ─────────────────────────────────────────────────────────────

def _extract_field(text: str, field: str) -> str:
    m = re.search(rf"{field}\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|$)", text,
                  flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_output(raw: str) -> Optional[dict]:
    """Parse INTENT/CONFIDENCE/REASONING text; returns None on failure."""
    intent = _extract_field(raw, "INTENT").lower()
    conf_str = _extract_field(raw, "CONFIDENCE")
    reasoning = _extract_field(raw, "REASONING")

    if intent not in ALLOWED_INTENTS:
        return None
    try:
        confidence = max(0.0, min(1.0, float(conf_str)))
    except Exception:
        return None
    if not reasoning:
        return None
    return {"intent": intent, "confidence": confidence, "reasoning": reasoning}


# ── evaluation metrics ────────────────────────────────────────────────────────

def compute_ece(traces: list[dict], n_bins: int = 10) -> float:
    """Expected Calibration Error — lower is better."""
    bins = [{"conf_sum": 0.0, "correct": 0, "count": 0} for _ in range(n_bins)]
    for t in traces:
        conf = t.get("confidence", 0.0)
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx]["conf_sum"] += conf
        bins[idx]["count"] += 1
        if t.get("correct"):
            bins[idx]["correct"] += 1

    total = len(traces)
    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        avg_conf = b["conf_sum"] / b["count"]
        avg_acc = b["correct"] / b["count"]
        ece += (b["count"] / total) * abs(avg_conf - avg_acc)
    return ece


def _intent_f1_block(traces: list[dict]) -> tuple[float, str]:
    lines = []
    f1_sum = 0.0
    for intent in sorted(ALLOWED_INTENTS):
        tp = sum(1 for t in traces if t["expected_intent"] == intent and t["predicted_intent"] == intent)
        fp = sum(1 for t in traces if t["expected_intent"] != intent and t["predicted_intent"] == intent)
        fn = sum(1 for t in traces if t["expected_intent"] == intent and t["predicted_intent"] != intent)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1_sum += f1
        lines.append(f"  {intent:<16} {tp:>4} {fp:>4} {fn:>4}  {p:>6.2f}  {r:>6.2f}  {f1:>6.2f}")
    return f1_sum / len(ALLOWED_INTENTS), "\n".join(lines)


def print_report(traces: list[dict], track: str) -> None:
    """Print accuracy split by normal/abnormal, per-class F1, ECE, latency."""
    normal   = [t for t in traces if t.get("category") == "normal"]
    abnormal = [t for t in traces if t.get("category") != "normal"]

    total   = len(traces)
    correct = sum(1 for t in traces if t["correct"])

    n_correct = sum(1 for t in normal if t["correct"])
    a_correct = sum(1 for t in abnormal if t["correct"])

    print(f"\n{'═'*72}")
    print(f"Track {track}")
    print(f"  Overall  : {correct}/{total} ({correct/total:.1%})")
    if normal:
        print(f"  Normal   : {n_correct}/{len(normal)} ({n_correct/len(normal):.1%})")
    if abnormal:
        # breakdown by subcategory group
        for cat in ("injection", "food_safety", "edge"):
            sub = [t for t in abnormal if t.get("category") == cat]
            if sub:
                sc = sum(1 for t in sub if t["correct"])
                print(f"  {cat:<10} : {sc}/{len(sub)} ({sc/len(sub):.1%})")

    print(f"{'─'*72}")
    print(f"  {'Intent':<16} {'TP':>4} {'FP':>4} {'FN':>4}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"  {'─'*58}")
    macro_f1, f1_lines = _intent_f1_block(traces)
    print(f1_lines)
    print(f"\n  macro-F1: {macro_f1:.3f}")

    ece = compute_ece(traces)
    low_conf = sum(1 for t in traces if t.get("confidence", 1.0) < 0.6) / total if total else 0
    print(f"  ECE:      {ece:.3f}")
    print(f"  low-conf (<0.6): {low_conf:.1%}")

    latencies = sorted(t["elapsed_ms"] for t in traces)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    print(f"  latency  p50={p50}ms  p95={p95}ms  p99={p99}ms")
    print()


# ── dataset loader ────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_trace(traces: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Trace → {path}", file=sys.stderr)
