import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, cast

from langchain_core.messages import HumanMessage, AIMessage
from langgraph_app.orchestrator.nodes.router import intent_router_node
from langgraph_app.orchestrator.state import GraphState


# PYTHONPATH=. python3 tests/router_intent/intent_router_test.py --turn 1

# Add project root to sys.path to allow imports from langgraph_app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """Loads test cases from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    with open(file_path, "r") as f:
        return json.load(f)


def iter_providers(arg_provider: str | None) -> List[str]:
    """Return providers to test: single if provided, otherwise all three."""
    all_providers = ["gemini", "openai", "bedrock_claude",]
    if arg_provider == "all":
        return all_providers
    if arg_provider:
        return [arg_provider]
    return all_providers


# Approx pricing (USD) per 1K tokens (rough, for reference only)
PRICE_PER_1K = {
    # Gemini Flash 2.5 官方价：$0.15/百万输入，$0.60/百万输出 → 换算1K
    "gemini": {"input": 0.00015, "output": 0.00060},   
    # GPT-4o mini 官方价：$0.15/百万输入，$0.60/百万输出 → 换算1K（你这部分数值是对的）
    "openai": {"input": 0.00015, "output": 0.00060},  
    # Claude 3 Haiku 官方价：$0.25/百万输入，$1.25/百万输出 → 换算1K（你原来的数值偏高）
    "bedrock_claude": {"input": 0.00025, "output": 0.00125}  
}



def estimate_tokens_from_text(text: str) -> int:
    # crude heuristic: 4 chars ~ 1 token
    return max(1, int(len(text) / 4))


def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICE_PER_1K.get(provider, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1000) * price["input"] + (output_tokens / 1000) * price["output"]


def run_tests(test_turn: int, provider: str | None):
    if test_turn == 1:
        test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    else:
        test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases_2.json")
    test_cases = load_test_cases(test_cases_path)
    if not test_cases:
        return

    for llm_provider in iter_providers(provider):
        os.environ["LLM_PROVIDER"] = llm_provider
        print(f"\n=== Running with LLM_PROVIDER={llm_provider} ===")

        results_by_intent: Dict[str, Dict[str, float]] = {}
        total_router_correct = 0
        total_baseline_correct = 0
        total_count = len(test_cases)

        aggregated_metrics = {
            "router": {
                "total_time": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "count": 0,
            },
            "overall": {
                "total_time": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "count": 0,
            },
        }

        output_lines: List[str] = []

        header = f"{'ID':<4} | {'Category':<35} | {'Expected':<15} | {'Actual':<15} | {'Baseline':<10} | {'Status'}"
        print(header)
        output_lines.append(header)

        divider = "-" * 110
        print(divider)
        output_lines.append(divider)

        for case in test_cases:
            case_id = case["id"]
            category = case["category"]
            input_data = case["input"]
            # Convert simple message dicts to LangChain message objects
            messages = []
            for msg in case["messages"]:
                if msg["type"] == "human":
                    content = msg["content"]
                    if input_data.get("image_data") and msg == case["messages"][-1]:
                        # Add image to the last human message
                        messages.append(HumanMessage(content=[{"type": "text", "text": content}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_data['image_data']}"}}]))
                    else:
                        messages.append(HumanMessage(content=content))
                elif msg["type"] == "ai":
                    messages.append(AIMessage(content=msg["content"]))

            # Prepare GraphState
            state: GraphState = {
                "messages": messages,
                "analysis": {
                    "intent": "chitchat",  # Default
                    "safety_safe": True,
                    "safety_reason": None,
                },
            }

            history_text = "\n".join([str(m.content) for m in messages])
            user_text = str(input_data.get("text", ""))

            router_input_tokens = estimate_tokens_from_text(user_text + history_text)
            router_output_tokens = 0
            router_elapsed = 0.0
            router_cost = 0.0

            actual_intent = "error"
            analysis: Dict[str, Any] = {}

            router_start = time.perf_counter()
            try:
                router_result = intent_router_node(state)
                router_elapsed = time.perf_counter() - router_start
                analysis = cast(Dict[str, Any], router_result.get("analysis", {}))
                actual_intent = analysis.get("intent", "error")
                router_output_tokens = estimate_tokens_from_text(json.dumps(analysis, ensure_ascii=False))
                router_cost = estimate_cost(llm_provider, router_input_tokens, router_output_tokens)
            except Exception as e:
                router_elapsed = time.perf_counter() - router_start
                print(f"Error running nodes for case {case_id}: {e}")
                actual_intent = "error"
                router_output_tokens = estimate_tokens_from_text(actual_intent)
                router_cost = estimate_cost(llm_provider, router_input_tokens, router_output_tokens)

            case_input_tokens = router_input_tokens
            case_output_tokens = router_output_tokens
            case_time = router_elapsed
            case_cost = router_cost

            aggregated_metrics["router"]["total_time"] += router_elapsed
            aggregated_metrics["router"]["total_input_tokens"] += router_input_tokens
            aggregated_metrics["router"]["total_output_tokens"] += router_output_tokens
            aggregated_metrics["router"]["total_cost"] += router_cost
            aggregated_metrics["router"]["count"] += 1

            aggregated_metrics["overall"]["total_time"] += case_time
            aggregated_metrics["overall"]["total_input_tokens"] += case_input_tokens
            aggregated_metrics["overall"]["total_output_tokens"] += case_output_tokens
            aggregated_metrics["overall"]["total_cost"] += case_cost
            aggregated_metrics["overall"]["count"] += 1

            expected_intent = case["expected_analysis"]["intent"]

            # Baseline Logic: If image -> recognition, else -> recommendation
            baseline_intent = "recognition" if input_data.get("image_data") else "recommendation"

            router_correct = actual_intent == expected_intent
            baseline_correct = baseline_intent == expected_intent

            if router_correct:
                total_router_correct += 1
            if baseline_correct:
                total_baseline_correct += 1

            # Track by intent
            if expected_intent not in results_by_intent:
                results_by_intent[expected_intent] = {
                    "total": 0,
                    "router_correct": 0,
                    "baseline_correct": 0,
                }

            results_by_intent[expected_intent]["total"] += 1
            if router_correct:
                results_by_intent[expected_intent]["router_correct"] += 1
            if baseline_correct:
                results_by_intent[expected_intent]["baseline_correct"] += 1

            status = "✅" if router_correct else "❌"
            row = f"{case_id:<4} | {category[:35]:<35} | {expected_intent:<15} | {actual_intent:<15} | {baseline_intent:<10} | {status}"
            print(row)
            output_lines.append(row)

            router_metrics_row = (
                "     router_metrics: "
                f"time={router_elapsed:.2f}s, "
                f"tokens_in={router_input_tokens}, tokens_out={router_output_tokens}, "
                f"cost=${router_cost:.4f}"
            )
            print(router_metrics_row)
            output_lines.append(router_metrics_row)

            case_metrics_row = (
                "     case_totals: "
                f"time={case_time:.2f}s, "
                f"tokens_in={case_input_tokens}, tokens_out={case_output_tokens}, "
                f"cost=${case_cost:.4f}"
            )
            print(case_metrics_row)
            output_lines.append(case_metrics_row)

        # Print Summary Table
        summary_header_sep = "\n" + "=" * 80
        summary_header = f"{'Intent':<40} | {'Count':<6} | {'Router Acc':<12} | {'Baseline Acc'}"
        summary_divider = "-" * 80

        print(summary_header_sep)
        print(summary_header)
        print(summary_divider)

        output_lines.extend([summary_header_sep, summary_header, summary_divider])

        for intent, stats in results_by_intent.items():
            router_acc = (stats["router_correct"] / stats["total"]) * 100
            baseline_acc = (stats["baseline_correct"] / stats["total"]) * 100
            summary_row = f"{intent[:40]:<40} | {stats['total']:<6} | {router_acc:>10.1f}% | {baseline_acc:>11.1f}%"
            print(summary_row)
            output_lines.append(summary_row)

        print(summary_divider)
        output_lines.append(summary_divider)

        overall_router_acc = (total_router_correct / total_count) * 100
        overall_baseline_acc = (total_baseline_correct / total_count) * 100
        overall_row = f"{'OVERALL':<40} | {total_count:<6} | {overall_router_acc:>10.1f}% | {overall_baseline_acc:>11.1f}%"
        overall_footer = "=" * 80

        print(overall_row)
        print(overall_footer)

        output_lines.extend([overall_row, overall_footer])

        def format_metric_line(label: str, data: Dict[str, float]) -> str:
            count = data.get("count", 0)
            total_cost = data.get("total_cost", 0.0)
            total_cases = data.get("total_cases", count)

            if count == 0:
                avg_time = 0.0
                avg_in = 0.0
                avg_out = 0.0
            else:
                avg_time = data["total_time"] / count
                avg_in = data["total_input_tokens"] / count
                avg_out = data["total_output_tokens"] / count

            avg_cost = total_cost / total_cases if total_cases else 0.0

            return (
                f"{label}: count={count}, avg_time={avg_time:.2f}s, "
                f"avg_tokens_in={avg_in:.1f}, avg_tokens_out={avg_out:.1f}, "
                f"avg_cost=${avg_cost:.4f}, total_cost=${total_cost:.4f}"
            )

        metrics_header = "\n" + "=" * 80 + "\nMETRIC AVERAGES" + "\n" + "-" * 80
        print(metrics_header)
        output_lines.append(metrics_header)

        router_metrics_line = format_metric_line("Router", aggregated_metrics["router"])
        overall_metrics_line = format_metric_line("Overall", aggregated_metrics["overall"])

        print(router_metrics_line)
        print(overall_metrics_line)

        output_lines.extend([
            router_metrics_line,
            overall_metrics_line,
        ])

        # Save to file per provider
        if test_turn == 1:
            output_file_path = os.path.join(os.path.dirname(__file__), f"test_results_{llm_provider}.txt")
        else:
            output_file_path = os.path.join(os.path.dirname(__file__), f"test_results_{test_turn}_{llm_provider}.txt")
        with open(output_file_path, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nTest results saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["gemini", "openai", "bedrock_claude", "all"], default=None,
                        help="Specify a single LLM provider to test; omit to run all three.")
    parser.add_argument("--turn", type=int, default=1, choices=[1, 2],
                        help="Choose test_cases.json (1) or test_cases_2.json (2). Default: 2")
    args = parser.parse_args()

    run_tests(test_turn=args.turn, provider=args.llm)
