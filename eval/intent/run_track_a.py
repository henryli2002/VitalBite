"""Track A — evaluate the live intent router (intent_router_node) on a dataset.

Run:
    python eval/intent/run_track_a.py
    python eval/intent/run_track_a.py --dataset eval/intent/dataset/smoke_en.jsonl
    python eval/intent/run_track_a.py --seed 99          # different user cohort
    python eval/intent/run_track_a.py --no-profile       # run without any user profile

What it does:
    1. Loads a .jsonl dataset (one sample per line).
    2. For each sample, builds a GraphState:
         - messages  = history turns + current utterance (+ image marker if present)
         - user_profile = dynamically generated via gen_profiles.make_profile(seed, index)
         - user_context = {"timezone": "Asia/Taipei"}  (fixed; affects meal_time label)
    3. Calls intent_router_node(state) directly (no Redis, no WebSocket).
    4. Compares predicted intent to expected_intent.
    5. Prints a per-sample result table and a summary to stdout.
    6. Writes full traces to results/trace_track_a_<seed>.jsonl.
"""

import asyncio
import base64
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

# --- path setup so we can import from src/ ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage, AIMessage

from langgraph_app.orchestrator.nodes.router import intent_router_node
from gen_profiles import make_profile  # sibling module


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_message(turn: dict):
    """Convert a history dict {role, content} to a LangChain message."""
    if turn["role"] == "user":
        return HumanMessage(content=turn["content"])
    return AIMessage(content=turn["content"])


def _resolve_image_b64(uuid: str) -> Optional[str]:
    """Resolve a UUID to base64 image bytes.

    Search order:
      1. eval/intent/fixtures/{uuid}.jpg  — eval-local copies (fast, no user-dir scan)
      2. data/images/**/{uuid}.jpg        — real app image store
    Returns None (with a stderr warning) if not found anywhere.
    """
    for candidate in [
        FIXTURES_DIR / f"{uuid}.jpg",
        *sorted((PROJECT_ROOT / "data" / "images").rglob(f"{uuid}.jpg")),
    ]:
        if candidate.exists():
            return base64.b64encode(candidate.read_bytes()).decode()
    print(f"  [warn] image not found for uuid={uuid}", file=sys.stderr)
    return None


def _build_current_message(utterance: str, image_marker: Optional[dict]) -> HumanMessage:
    """Build the current-turn HumanMessage.

    v3.3 style: injects base64 image as multipart content.
    (A future v4.1 runner would instead append <attached_image uuid=.../> text.)
    """
    if not image_marker:
        return HumanMessage(content=utterance)

    b64 = _resolve_image_b64(image_marker["uuid"])
    if not b64:
        return HumanMessage(content=utterance)

    parts = []
    if utterance.strip():
        parts.append({"type": "text", "text": utterance})
    parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return HumanMessage(content=parts)


def _build_state(sample: dict, profile: Optional[dict]) -> dict:
    history_msgs = [_build_message(t) for t in sample.get("history", [])]
    current_msg = _build_current_message(
        sample["utterance"], sample.get("image_marker")
    )
    return {
        "messages": history_msgs + [current_msg],
        "user_profile": profile,
        "user_context": {"timezone": "Asia/Taipei"},
        "response_channel": None,   # disables Redis Pub/Sub in router
    }


# ── main eval loop ────────────────────────────────────────────────────────────

async def run(dataset_path: Path, seed: int, no_profile: bool, output_dir: Path) -> None:
    samples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace_track_a.jsonl"

    correct = 0
    traces = []

    print(f"\n{'ID':<20} {'Expected':<16} {'Predicted':<16} {'Conf':>6}  {'ms':>6}  OK?")
    print("-" * 72)

    for i, sample in enumerate(samples):
        profile = None if no_profile else make_profile(seed=seed, index=i)
        state = _build_state(sample, profile)

        t0 = time.perf_counter()
        result = await intent_router_node(state)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        analysis = result.get("analysis", {})
        predicted = analysis.get("intent", "unknown")
        confidence = analysis.get("confidence", 0.0)
        reasoning = analysis.get("reasoning", "")
        expected = sample["expected_intent"]
        ok = predicted == expected

        if ok:
            correct += 1

        mark = "✓" if ok else "✗"
        print(
            f"{sample['id']:<20} {expected:<16} {predicted:<16} "
            f"{confidence:>6.2f}  {elapsed_ms:>6}  {mark}"
        )

        traces.append({
            "id": sample["id"],
            "category": sample.get("category", "normal"),
            "expected_intent": expected,
            "predicted_intent": predicted,
            "confidence": confidence,
            "reasoning": reasoning,
            "correct": ok,
            "elapsed_ms": elapsed_ms,
            "profile": profile,
        })

    # ── summary + trace ───────────────────────────────────────────────────────
    from runner_lib import print_report, write_trace
    label = "A (Gemini API, no-profile)" if no_profile else "A (Gemini API, with-profile)"
    print_report(traces, track=label)
    write_trace(traces, trace_path)
    print(f"Trace → {trace_path.relative_to(PROJECT_ROOT)}\n")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track A: eval intent_router_node")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "dataset" / "test.jsonl"),
        help="Path to .jsonl dataset file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for profile generation (default: 42).",
    )
    parser.add_argument(
        "--no-profile", action="store_true",
        help="Pass user_profile=None to the router (profile-agnostic baseline).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results" / "no_profile"),
    )
    args = parser.parse_args()

    asyncio.run(run(Path(args.dataset), args.seed, args.no_profile, Path(args.output_dir)))
