import argparse
import itertools
import os
import subprocess
import time
from datetime import datetime

# Define parameter grid
PARAM_GRID = {
    "temperature": [0.0, 0.2, 0.5],
    "top_p": [0.5, 0.8, 0.95],
}

def run_grid_search(provider: str, turn: int):
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("tests", "router_intent", "grid_search_results", f"{provider}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting grid search for {provider} (Turn {turn})")
    print(f"Results will be saved to: {results_dir}")
    
    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    
    summary_file = os.path.join(results_dir, "summary.csv")
    with open(summary_file, "w") as f:
        f.write("run_id,temperature,top_p,router_acc,overall_acc,avg_cost\n")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        temp = params["temperature"]
        top_p = params["top_p"]
        
        run_id = f"run_{i+1}_t{temp}_p{top_p}"
        print(f"\n[{i+1}/{len(combinations)}] Running with temperature={temp}, top_p={top_p}...")
        
        # Set environment variables for this run
        # We override ROUTER parameters specifically as that's what intent_router_test focuses on
        env = os.environ.copy()
        env[f"LLM_SAMPLING_{provider.upper()}_ROUTER_TEMPERATURE"] = str(temp)
        env[f"LLM_SAMPLING_{provider.upper()}_ROUTER_TOP_P"] = str(top_p)
        # Also set for clarification since it's part of the test flow
        env[f"LLM_SAMPLING_{provider.upper()}_CLARIFICATION_TEMPERATURE"] = str(temp) 
        env[f"LLM_SAMPLING_{provider.upper()}_CLARIFICATION_TOP_P"] = str(top_p)
        
        # Run the existing test script
        # Note: intent_router_test.py writes to a file based on provider name. 
        # We need to capture its output or move the result file.
        cmd = [
            "python", 
            "tests/router_intent/intent_router_test.py", 
            "--turn", str(turn), 
            "--llm", provider
        ]
        
        start_time = time.time()
        try:
            # Use Popen to print output in real-time while capturing it
            process = subprocess.Popen(
                cmd, 
                env=env, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                cwd=os.getcwd(),
                bufsize=1, # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            if process.stdout:
                for line in process.stdout:
                    print(line, end="") # Print to console
                    output_lines.append(line)
                
            process.wait()
            output = "".join(output_lines)
            
            if process.returncode != 0:
                print(f"  -> Run failed with return code {process.returncode}")
                with open(os.path.join(results_dir, f"{run_id}_error.log"), "w") as f:
                    f.write(output)
                continue
            
            # Save full output log
            log_path = os.path.join(results_dir, f"{run_id}.log")
            with open(log_path, "w") as f:
                f.write(output)
            
            # Extract key metrics from output for summary
            router_acc = "N/A"
            overall_acc = "N/A"
            avg_cost = "N/A"
            
            for line in output.splitlines():
                if "Router Acc" in line and "%" in line:
                    # Example line: "Category | Count | Router Acc | Baseline Acc" -> followed by data rows
                    # We need the OVERALL row
                    pass
                if line.startswith("OVERALL"):
                    # Example: OVERALL | 20 | 95.0% | 80.0%
                    parts = line.split("|")
                    if len(parts) >= 4:
                        router_acc = parts[2].strip()
                        overall_acc = parts[3].strip()
                if "Overall: count=" in line:
                     # Example: Overall: count=5, avg_time=1.2s, ..., avg_cost=$0.0012, ...
                     if "avg_cost=$" in line:
                         avg_cost = line.split("avg_cost=$")[1].split(",")[0]

            print(f"  -> Metrics: Router Acc={router_acc}, Cost=${avg_cost}")
            
            # Append to summary
            with open(summary_file, "a") as f:
                f.write(f"{run_id},{temp},{top_p},{router_acc},{overall_acc},{avg_cost}\n")
                
            # Rename/move the generated result file to our result dir
            # The test script generates "test_results_{provider}.txt" or "test_results_{turn}_{provider}.txt"
            src_filename = f"test_results_{provider}.txt" if turn == 1 else f"test_results_{turn}_{provider}.txt"
            src_path = os.path.join("tests", "router_intent", src_filename)
            if os.path.exists(src_path):
                dst_path = os.path.join(results_dir, f"{run_id}_details.txt")
                os.rename(src_path, dst_path)
                
        except subprocess.CalledProcessError as e:
            print(f"  -> Run failed: {e}")
            with open(os.path.join(results_dir, f"{run_id}_error.log"), "w") as f:
                f.write(e.stderr)

    print(f"\nGrid search complete. Summary saved to {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["gemini", "openai", "bedrock_claude", "all"], default="all",
                      help="LLM provider to test. Use 'all' to run sequentially for all providers.")
    parser.add_argument("--turn", type=int, default=1)
    args = parser.parse_args()
    
    if args.provider == "all":
        providers = ["gemini", "openai", "bedrock_claude"]
        for p in providers:
            run_grid_search(p, args.turn)
    else:
        run_grid_search(args.provider, args.turn)