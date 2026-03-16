"""Generate a markdown file listing error cases per LLM provider, with optional inline images.

Parses test result txt files (with ❌ rows) and joins with test case inputs
(without dumping base64). Output is grouped by provider with expected/actual
intent, input text, image presence/length, and clarification if captured.

Usage:
    PYTHONPATH=. python tests/router_intent/generate_error_markdown.py \
        --results tests/router_intent/test_results_gemini.txt \
        --results tests/router_intent/test_results_openai.txt \
        --results tests/router_intent/test_results_bedrock_claude.txt \
        --cases tests/router_intent/test_cases.json \
        --out tests/router_intent/error_cases_per_llm.md \
        --show-images  # 可选，启用则在 markdown 中内联 data URI 图片（文件会变大）

Notes:
- Does not emit base64 image data; only reports has_image and length.
- Clarification lines are captured from result files if present (lines starting
  with "     └─ Clarification Output:").
"""

import argparse
import json
import os
import re
from typing import Dict, List, Any, Optional


RowPattern = re.compile(r"^\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|")


def load_cases(paths: List[str]) -> Dict[int, Dict[str, Any]]:
    cases: Dict[int, Dict[str, Any]] = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARN] case file not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for case in data:
            cid = int(case.get("id", -1))
            if cid not in cases:
                cases[cid] = case
    return cases


def parse_result_file(path: str):
    errors = []
    last_id: Optional[int] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = RowPattern.match(line)
            if m:
                cid = int(m.group(1))
                category = m.group(2).strip()
                expected = m.group(3).strip()
                actual = m.group(4).strip()
                status = "❌" in line
                last_id = cid
                if status:
                    errors.append({
                        "id": cid,
                        "category": category,
                        "expected": expected,
                        "actual": actual,
                    })
    return errors


def build_markdown(provider: str, errors: List[Dict[str, Any]], cases: Dict[int, Dict[str, Any]], show_images: bool) -> List[str]:
    lines = [f"## {provider}"]
    if not errors:
        lines.append("(No errors)")
        lines.append("")
        return lines

    for err in errors:
        case = cases.get(err["id"], {})
        input_data = case.get("input", {}) if case else {}
        text = input_data.get("text", "")
        image_b64 = input_data.get("image_data")
        has_image = bool(image_b64)
        img_len = len(image_b64) if isinstance(image_b64, str) else 0

        lines.append(f"### ID {err['id']} — {err['category']}")
        lines.append(f"- Expected: {err['expected']}")
        lines.append(f"- Actual: {err['actual']}")
        lines.append(f"- Input text: {text}")
        lines.append(f"- Has image: {has_image} (length={img_len})")
        if show_images and has_image and isinstance(image_b64, str):
            lines.append("")
            lines.append(f"![case-{err['id']}](data:image/png;base64,{image_b64})")
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", help="Result txt files to parse (defaults to common providers)")
    parser.add_argument("--cases", nargs="+", help="Test case JSON files (defaults to test_cases.json & test_cases_2.json)")
    parser.add_argument("--out", default="tests/router_intent/error_cases_per_llm.md", help="Output markdown file")
    parser.add_argument("--show-images", action="store_true", help="Inline images as data URI in markdown (file will grow)")
    args = parser.parse_args()

    default_results = [
        "tests/router_intent/test_results_gemini.txt",
        "tests/router_intent/test_results_openai.txt",
        "tests/router_intent/test_results_bedrock_claude.txt",
    ]
    results = args.results if args.results else default_results

    default_cases = [
        "tests/router_intent/test_cases.json",
        "tests/router_intent/test_cases_2.json",
    ]
    case_files = args.cases if args.cases else default_cases

    cases = load_cases(case_files)

    md: List[str] = ["# Error Cases Per LLM", ""]

    for res_path in results:
        provider = os.path.splitext(os.path.basename(res_path))[0].replace("test_results_", "")
        errors = parse_result_file(res_path)
        md.extend(build_markdown(provider, errors, cases, args.show_images))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote markdown to {args.out}")


if __name__ == "__main__":
    main()
