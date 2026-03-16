"""
Generate markdown files listing error cases and success cases per LLM provider.

Parses test result txt files (with ❌ / ✅ rows) and joins with test case inputs
(without dumping base64). Output is grouped by provider with expected/actual
intent, input text, image presence/length.

Usage:
    PYTHONPATH=. python tests/router_intent/generate_case_markdown.py \
        --results tests/router_intent/test_results_gemini.txt \
        --results tests/router_intent/test_results_openai.txt \
        --results tests/router_intent/test_results_bedrock_claude.txt \
        --cases tests/router_intent/test_cases.json \
        --cases tests/router_intent/test_cases_2.json \
        --out-dir tests/router_intent \
        --show-images

Outputs:
    error_cases_per_llm.md
    success_cases_per_llm.md

Notes:
- Does not emit base64 image data unless --show-images is enabled.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Any


RowPattern = re.compile(r"^\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|")


# -----------------------------
# Load test cases
# -----------------------------
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


# -----------------------------
# Parse result txt
# -----------------------------
def parse_result_file(path: str):

    errors = []
    successes = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:

            m = RowPattern.match(line)
            if not m:
                continue

            cid = int(m.group(1))
            category = m.group(2).strip()
            expected = m.group(3).strip()
            actual = m.group(4).strip()

            row = {
                "id": cid,
                "category": category,
                "expected": expected,
                "actual": actual,
            }

            if "❌" in line:
                errors.append(row)

            elif "✅" in line:
                successes.append(row)

    return errors, successes


# -----------------------------
# Extract case input info
# -----------------------------
def extract_input(case):

    input_data = case.get("input", {}) if case else {}

    text = input_data.get("text", "")

    image_b64 = input_data.get("image_data")

    has_image = bool(image_b64)

    img_len = len(image_b64) if isinstance(image_b64, str) else 0

    return text, has_image, img_len, image_b64


# -----------------------------
# Build error markdown
# -----------------------------
def build_error_markdown(provider, errors, cases, show_images):

    lines = [f"## {provider}", ""]

    if not errors:
        lines.append("(No errors)")
        lines.append("")
        return lines

    for err in errors:

        case = cases.get(err["id"], {})

        text, has_image, img_len, image_b64 = extract_input(case)

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


# -----------------------------
# Build success markdown
# -----------------------------
def build_success_markdown(provider, successes, cases, show_images):

    lines = [f"## {provider}", ""]

    if not successes:
        lines.append("(No successful cases captured)")
        lines.append("")
        return lines

    for suc in successes:

        case = cases.get(suc["id"], {})

        text, has_image, img_len, image_b64 = extract_input(case)

        lines.append(f"### ID {suc['id']} — {suc['category']}")
        lines.append(f"- Expected: {suc['expected']}")
        lines.append(f"- Actual: {suc['actual']}")
        lines.append(f"- Input text: {text}")
        lines.append(f"- Has image: {has_image} (length={img_len})")

        if show_images and has_image and isinstance(image_b64, str):
            lines.append("")
            lines.append(f"![case-{suc['id']}](data:image/png;base64,{image_b64})")

        lines.append("")

    return lines


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        nargs="+",
        help="Result txt files to parse"
    )

    parser.add_argument(
        "--cases",
        nargs="+",
        help="Test case JSON files"
    )

    parser.add_argument(
        "--out-dir",
        default="tests/router_intent",
        help="Output directory"
    )

    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Inline images as data URI"
    )

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

    error_md = ["# Error Cases Per LLM", ""]
    success_md = ["# Success Cases Per LLM", ""]

    for res_path in results:

        provider = os.path.splitext(os.path.basename(res_path))[0].replace(
            "test_results_", ""
        )

        errors, successes = parse_result_file(res_path)

        error_md.extend(
            build_error_markdown(provider, errors, cases, args.show_images)
        )

        success_md.extend(
            build_success_markdown(provider, successes, cases, args.show_images)
        )

    os.makedirs(args.out_dir, exist_ok=True)

    error_path = os.path.join(args.out_dir, "error_cases_per_llm.md")
    success_path = os.path.join(args.out_dir, "success_cases_per_llm.md")

    with open(error_path, "w", encoding="utf-8") as f:
        f.write("\n".join(error_md))

    with open(success_path, "w", encoding="utf-8") as f:
        f.write("\n".join(success_md))

    print(f"Wrote error markdown to {error_path}")
    print(f"Wrote success markdown to {success_path}")


if __name__ == "__main__":
    main()