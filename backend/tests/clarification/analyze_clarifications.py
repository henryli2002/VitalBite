import json
import os
import re
import base64
import sys

def analyze_clarifications(test_cases_path, test_results_path, output_path="tests/clarification/clarification_analysis.txt"):
    # Load original test cases
    test_cases = {}
    if os.path.exists(test_cases_path):
        with open(test_cases_path, "r", encoding="utf-8") as f:
            original_cases = json.load(f)
            for case in original_cases:
                test_cases[case["id"]] = case
    else:
        print(f"Error: {test_cases_path} not found.")
        return

    # Parse test results to extract clarification outputs
    clarification_outputs = {}
    if os.path.exists(test_results_path):
        with open(test_results_path, "r", encoding="utf-8") as f:
            results_content = f.read()
            # Regex to find ID and the subsequent clarification output
            # Assuming clarification output always follows a line with "└─ Clarification Output: "
            pattern = re.compile(r"^(\d{1,4})\s+\|.*?\n(?:\s+└─ Clarification Output: (.*?)(?:\n|$\n))?", re.MULTILINE)
            
            # Find all matches
            matches = pattern.finditer(results_content)
            
            current_id = None
            for match in matches:
                if match.group(1): # This is the main ID line
                    current_id = int(match.group(1))
                if match.group(2) and current_id is not None:
                    clarification_outputs[current_id] = match.group(2).strip()
                    current_id = None # Reset after capturing clarification

    else:
        print(f"Error: {test_results_path} not found.")
        return

    # Generate analysis report
    report_lines = []
    report_lines.append("Clarification Analysis Report\n")
    report_lines.append("=" * 30 + "\n")

    # Only iterate through cases that actually produced clarification output
    for case_id in sorted(clarification_outputs.keys()):
        case = test_cases.get(case_id) # Use .get() in case ID is in results but not original cases (shouldn't happen)
        if not case: # Should not happen if test_cases is populated correctly
            continue

        original_text = case["input"].get("text", "")
        original_image_present = bool(case["input"].get("image_data"))
        expected_intent = case["expected_analysis"].get("intent", "N/A")
        category = case["category"]

        report_lines.append(f"Case ID: {case_id}")
        report_lines.append(f"  Category: {category}")
        report_lines.append(f"  Original Text: \"{original_text}\"")
        report_lines.append(f"  Original Image Present: {original_image_present}")
        report_lines.append(f"  Expected Intent: {expected_intent}")
        
        clarification_output = clarification_outputs.get(case_id, "N/A (Error in parsing or missing from results)")
        report_lines.append(f"  Clarification AI Response: \"{clarification_output}\"")
        report_lines.append("-" * 20)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Analysis report saved to {output_path}")

if __name__ == "__main__":
    # Set paths relative to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(project_root)
    
    test_cases_path = os.path.join(project_root, "tests/router_intent/test_cases.json")
    test_results_path = os.path.join(project_root, "tests/router_intent/test_results.txt")
    output_analysis_path = os.path.join(project_root, "tests/clarification/clarification_analysis.txt")

    analyze_clarifications(test_cases_path, test_results_path, output_analysis_path)
