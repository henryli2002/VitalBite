import re
import pandas as pd
from pathlib import Path


def parse_results_from_file(filepath):
    """
    Parses the evaluation results file line-by-line to be more robust.
    """
    all_results = {}
    current_model = None

    with open(filepath, "r") as f:
        for line in f:
            # Check for a new model section
            model_match = re.search(r"--- Results for (.+?) ---", line)
            if model_match:
                current_model = model_match.group(1).strip()
                if current_model not in all_results:
                    all_results[current_model] = {}
                continue

            # Check for the evaluation summary table header
            if "Metric" in line and "Graph" in line and "Direct" in line:
                # This is the summary table, we need the 'finetuned' column from it.
                # However, the 'Winner' column tells us which is best, which is what we need.
                continue

            # Look for lines that contain metric data and a 'Winner'
            # Example: total_mass 100.0% 49.5% 77.3% 28.2% finetuned
            parts = line.strip().split()
            if len(parts) > 2 and "%" in parts[1]:
                metric_name = parts[0]
                # The winner is the last part of the line
                winner_method = parts[-1]

                # We are interested in the wMAPE for the finetuned model
                # The structure is: Metric, Graph, Direct, Fewshot, Finetuned, Winner
                if len(parts) == 6:
                    try:
                        finetuned_wmape_str = parts[4].strip("%")
                        finetuned_wmape = float(finetuned_wmape_str) / 100.0
                        if current_model:
                            all_results[current_model][metric_name] = finetuned_wmape
                    except (ValueError, IndexError):
                        continue
    return all_results


def generate_report(results, output_path):
    if not results:
        print("No results could be parsed. Cannot generate report.")
        return

    df = pd.DataFrame(results).T

    # Ensure we only process numeric columns for the mean calculation
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if not numeric_cols.empty:
        df["average_wMAPE"] = df[numeric_cols].mean(axis=1)
        df = df.sort_values("average_wMAPE")

        # Format as percentage
        for col in df.columns:
            if "wMAPE" in col:
                df[col] = df[col].apply(lambda x: f"{x:.2%}")
            else:
                # Apply to all metric columns
                df[col] = df[col].apply(lambda x: f"{x:.2%}")
    else:
        print("No numeric data found to calculate average wMAPE.")
        # Create an empty column if it doesn't exist to prevent KeyErrors
        df["average_wMAPE"] = "N/A"

    report = "# Model Evaluation Report\n\n"
    report += "## Objective\n"
    report += "This report summarizes the performance of various fine-tuned CNN models for food nutrition estimation. The primary metric used is the **Weighted Mean Absolute Percentage Error (wMAPE)**, where lower is better.\n\n"
    report += "## Performance Summary\n\n"

    if not df.empty:
        report += df.to_markdown()
        report += "\n\n"

        best_model = df.index[0]
        best_avg_wmape = df.loc[best_model, "average_wMAPE"]

        report += "## Conclusion\n"
        report += f"The best performing model is **`{best_model}`** with an average wMAPE of **{best_avg_wmape}** across all nutritional metrics.\n"
    else:
        report += "Could not generate a comparative table as no data was parsed.\n"

    with open(output_path, "w") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    EVAL_DIR = Path(__file__).resolve().parent
    results_file = EVAL_DIR / "model_eval.txt"
    report_file = EVAL_DIR / "evaluation_report.md"

    parsed_results = parse_results_from_file(results_file)
    generate_report(parsed_results, report_file)
