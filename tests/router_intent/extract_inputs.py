"""Extract input fields from test case files and optionally render markdown with images.

Usage (text summary only):
    PYTHONPATH=. python tests/router_intent/extract_inputs.py \
        --files tests/router_intent/test_cases.json \
        --out-text tests/router_intent/inputs_summary.txt

Usage (markdown with inline images):
    PYTHONPATH=. python tests/router_intent/extract_inputs.py \
        --files tests/router_intent/test_cases.json \
        --out-md tests/router_intent/inputs_markdown.md

Notes:
- Text summary captures: id, category, text, has_image, image_b64_length.
- Markdown输出：按用例生成段落，若有图片则嵌入 data URI（文件会变大，请酌情使用）。
"""

import argparse
import json
import os
from typing import Any, Dict, List


def load_cases(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    input_data = case.get("input", {})
    text = input_data.get("text", "")
    image_b64 = input_data.get("image_data")
    has_image = bool(image_b64)
    image_len = len(image_b64) if isinstance(image_b64, str) else 0

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "text": text,
        "has_image": has_image,
        "image_b64_length": image_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="Paths to test case JSON files")
    parser.add_argument("--out-text", default="tests/router_intent/inputs_summary.txt", help="Output text summary file")
    parser.add_argument("--out-md", default=None, help="Optional markdown output with inline images")
    args = parser.parse_args()

    summaries: List[str] = []
    md_lines: List[str] = []

    for path in args.files:
        cases = load_cases(path)
        if not cases:
            continue
        summaries.append(f"=== {path} ===")
        if args.out_md:
            md_lines.append(f"# {path}\n")
        for case in cases:
            s = summarize_case(case)
            line = (
                f"ID {s['id']}: {s['category']} | text={s['text']} | "
                f"has_image={s['has_image']} | image_b64_length={s['image_b64_length']}"
            )
            summaries.append(line)

            if args.out_md:
                md_lines.append(f"## ID {s['id']} — {s['category']}")
                md_lines.append(f"- text: {s['text']}")
                md_lines.append(f"- has_image: {s['has_image']}")
                md_lines.append(f"- image_b64_length: {s['image_b64_length']}")
                if s["has_image"]:
                    md_lines.append("")
                    md_lines.append(f"![case-{s['id']}](data:image/png;base64,{case.get('input', {}).get('image_data','')})")
                md_lines.append("")

    os.makedirs(os.path.dirname(args.out_text), exist_ok=True)
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write("\n".join(summaries))
    print(f"Wrote summary to {args.out_text}")

    if args.out_md and md_lines:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"Wrote markdown to {args.out_md}")


if __name__ == "__main__":
    main()
