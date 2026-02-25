import json
import os
import sys
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_app.orchestrator.nodes.router import intent_router_node
from langgraph_app.agents.clarification.agent import clarification_node
from langgraph_app.orchestrator.state import GraphState

# PYTHONPATH=. python3 tests/router_intent/intent_router_test.py

# Add project root to sys.path to allow imports from langgraph_app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

TEST_TURN = 2

def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """Loads test cases from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return []
    with open(file_path, "r") as f:
        return json.load(f)

def run_tests():
    if TEST_TURN == 1:
        test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
        test_cases = load_test_cases(test_cases_path)
    elif TEST_TURN == 2:
        test_cases_path = os.path.join(os.path.dirname(__file__), "test_cases_2.json")
        test_cases = load_test_cases(test_cases_path)
        
    if not test_cases:
        return

    results_by_category = {}
    total_router_correct = 0
    total_baseline_correct = 0
    total_count = len(test_cases)

    output_lines = []
    
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
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        # Prepare GraphState
        state: GraphState = {
            "input": input_data,
            "messages": messages,
            "analysis": {
                "intent": "clarification", # Default
                "safety_safe": True,
                "safety_reason": None
            }
        }

        # Run Router Node
        try:
            router_result = intent_router_node(state)
            analysis = router_result.get("analysis", {})
            actual_intent = analysis.get("intent", "error")
            
            # If intent is clarification, also run clarification agent
            clarification_response = ""
            if actual_intent == "clarification":
                # Router only returns analysis, we need to merge with state to pass to clarification_node
                state.update(router_result)
                clarification_result = clarification_node(state)
                clarification_response = clarification_result.get("final_response", "")
        except Exception as e:
            print(f"Error running nodes for case {case_id}: {e}")
            actual_intent = "error"
            clarification_response = ""

        expected_intent = case["expected_analysis"]["intent"]
        
        # Baseline Logic: If image -> recognition, else -> recommendation
        baseline_intent = "recognition" if input_data.get("image_data") else "recommendation"
        
        router_correct = (actual_intent == expected_intent)
        baseline_correct = (baseline_intent == expected_intent)

        if router_correct: total_router_correct += 1
        if baseline_correct: total_baseline_correct += 1

        # Track by category
        if category not in results_by_category:
            results_by_category[category] = {"total": 0, "router_correct": 0, "baseline_correct": 0}
        
        results_by_category[category]["total"] += 1
        if router_correct: results_by_category[category]["router_correct"] += 1
        if baseline_correct: results_by_category[category]["baseline_correct"] += 1

        status = "✅" if router_correct else "❌"
        row = f"{case_id:<4} | {category[:35]:<35} | {expected_intent:<15} | {actual_intent:<15} | {baseline_intent:<10} | {status}"
        print(row)
        output_lines.append(row)

        # If it's clarification, print the agent's output
        if actual_intent == "clarification" and clarification_response:
            clar_row = f"     └─ Clarification Output: {clarification_response}"
            print(clar_row)
            output_lines.append(clar_row)

        # If it's a clarification or a mismatch, provide more detail
        if not router_correct or expected_intent == "clarification":
            # In a real scenario, we might want to see the LLM's reasoning if the node returned it.
            # Currently intent_router_node only returns intent, safety_safe, safety_reason.
            pass

    # Print Summary Table
    summary_header_sep = "\n" + "="*80
    summary_header = f"{'Category':<40} | {'Count':<6} | {'Router Acc':<12} | {'Baseline Acc'}"
    summary_divider = "-" * 80
    
    print(summary_header_sep)
    print(summary_header)
    print(summary_divider)
    
    output_lines.extend([summary_header_sep, summary_header, summary_divider])
    
    for cat, stats in results_by_category.items():
        router_acc = (stats["router_correct"] / stats["total"]) * 100
        baseline_acc = (stats["baseline_correct"] / stats["total"]) * 100
        summary_row = f"{cat[:40]:<40} | {stats['total']:<6} | {router_acc:>10.1f}% | {baseline_acc:>11.1f}%"
        print(summary_row)
        output_lines.append(summary_row)

    print(summary_divider)
    output_lines.append(summary_divider)
    
    overall_router_acc = (total_router_correct / total_count) * 100
    overall_baseline_acc = (total_baseline_correct / total_count) * 100
    overall_row = f"{'OVERALL':<40} | {total_count:<6} | {overall_router_acc:>10.1f}% | {overall_baseline_acc:>11.1f}%"
    overall_footer = "="*80
    
    print(overall_row)
    print(overall_footer)
    
    output_lines.extend([overall_row, overall_footer])

    # Save to file
    if TEST_TURN == 1:
        output_file_path = os.path.join(os.path.dirname(__file__), "test_results.txt")
    else:
        output_file_path = os.path.join(os.path.dirname(__file__), f"test_results_{TEST_TURN}.txt")
    with open(output_file_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nTest results saved to {output_file_path}")

if __name__ == "__main__":
    run_tests()
