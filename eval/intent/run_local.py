"""Zero-shot evaluation for any local model via omlx (OpenAI-compatible API).

Trace is saved to results/trace_<model_name>.jsonl automatically.

Run:
    python eval/intent/run_local.py --model Qwen3.5-0.8B-OptiQ-4bit
    python eval/intent/run_local.py --model gemma-4-e4b-it-4bit
"""

import argparse
import re
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
)
from gen_profiles import make_profile

OMLX_BASE_URL = "http://127.0.0.1:8002/v1"
OMLX_API_KEY  = "omlx-5nz1zmtjkx0ywcom"


def is_qwen(model: str) -> bool:
    return "qwen" in model.lower()


def run(dataset_path: Path, model: str, base_url: str, seed: int, no_profile: bool, output_dir: Path) -> None:
    client = OpenAI(base_url=base_url, api_key=OMLX_API_KEY)
    samples = load_jsonl(dataset_path)

    results_dir = output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", model)
    trace_path = results_dir / f"trace_{safe_name}.jsonl"

    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if is_qwen(model) else {}

    correct = 0
    traces: list[dict] = []

    print(f"\nZero-shot  |  model={model}")
    if is_qwen(model):
        print("           |  thinking=disabled")
    print(f"           |  profile={'disabled' if no_profile else f'seed={seed}'}")
    print(f"{'ID':<40} {'Expected':<16} {'Predicted':<16} {'Conf':>6}  {'ms':>6}  OK?")
    print("-" * 92)

    for i, sample in enumerate(samples):
        profile = None if no_profile else make_profile(seed=seed, index=i)
        msgs = build_chat_messages(sample, profile=profile)

        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.1,
            max_tokens=128,
            extra_body=extra_body or None,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        raw = resp.choices[0].message.content or ""
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
        if ok:
            correct += 1

        mark = "✓" if ok else "✗"
        print(
            f"{sample['id']:<40} {expected:<16} {predicted:<16} "
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
            "raw_output": raw,
        })

    print_report(traces, track=f"zero-shot / {model}")
    write_trace(traces, trace_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name as listed in omlx")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "dataset" / "test.jsonl"),
    )
    parser.add_argument("--base-url", default=OMLX_BASE_URL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results" / "with_profile"),
    )
    args = parser.parse_args()

    run(Path(args.dataset), args.model, args.base_url, args.seed, args.no_profile,
        Path(args.output_dir))
