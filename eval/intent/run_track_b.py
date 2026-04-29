"""Track B — Qwen3.5-9B-OptiQ-4bit via omlx, zero-shot.

Thinking mode is disabled (/no_think) so the model returns plain text directly.

Run:
    python eval/intent/run_track_b.py
    python eval/intent/run_track_b.py --dataset eval/intent/dataset/smoke_en.jsonl
    python eval/intent/run_track_b.py --base-url http://127.0.0.1:8002 --model Qwen3.5-9B-OptiQ-4bit
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from runner_lib import (
    build_chat_messages,
    parse_output,
    print_report,
    load_jsonl,
    write_trace,
    ALLOWED_INTENTS,
)

OMLX_BASE_URL = "http://127.0.0.1:8002/v1"
OMLX_API_KEY  = "omlx-5nz1zmtjkx0ywcom"
DEFAULT_MODEL  = "Qwen3.5-9B-OptiQ-4bit"


def run(dataset_path: Path, base_url: str, model: str) -> None:
    client = OpenAI(base_url=base_url, api_key=OMLX_API_KEY)
    samples = load_jsonl(dataset_path)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    trace_path = results_dir / "trace_track_b.jsonl"

    correct = 0
    traces: list[dict] = []

    print(f"\nTrack B  |  model={model}")
    print(f"{'ID':<20} {'Expected':<16} {'Predicted':<16} {'Conf':>6}  {'ms':>6}  OK?")
    print("-" * 72)

    for sample in samples:
        msgs = build_chat_messages(sample)

        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.1,
            max_tokens=128,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        raw = resp.choices[0].message.content or ""
        parsed = parse_output(raw)

        if parsed is None:
            predicted = "chitchat"
            confidence = 0.0
            reasoning = f"[parse error] {raw[:120]}"
        else:
            predicted = parsed["intent"]
            confidence = parsed["confidence"]
            reasoning = parsed["reasoning"]

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
            "expected_intent": expected,
            "predicted_intent": predicted,
            "confidence": confidence,
            "reasoning": reasoning,
            "correct": ok,
            "elapsed_ms": elapsed_ms,
            "raw_output": raw,
        })

    print_report(traces, track="B")
    write_trace(traces, trace_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track B: Qwen3 zero-shot via omlx")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "dataset" / "test.jsonl"),
    )
    parser.add_argument("--base-url", default=OMLX_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    run(Path(args.dataset), args.base_url, args.model)
