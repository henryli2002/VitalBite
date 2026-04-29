#!/usr/bin/env bash
# Run all LoRA experiments sequentially, log each to experiments/logs/
set -uo pipefail

SITE="/Applications/oMLX.app/Contents/Python/framework-mlx-framework/lib/python3.11/site-packages"
PYTHON="/Applications/oMLX.app/Contents/Python/cpython-3.11/bin/python3.11"
TRAIN_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$TRAIN_DIR/experiments/logs"
mkdir -p "$LOG_DIR"

EXPS=(
    "exp_A_all_8layers"
    "exp_B_attn_8layers"
    "exp_C_mlp_8layers"
    "exp_D_all_16layers"
)

for EXP in "${EXPS[@]}"; do
    CONFIG="$TRAIN_DIR/experiments/${EXP}.yaml"
    LOG="$LOG_DIR/${EXP}.log"
    echo "════════════════════════════════════════"
    echo "Starting: $EXP"
    echo "Config  : $CONFIG"
    echo "Log     : $LOG"
    echo "════════════════════════════════════════"

    if PYTHONPATH="$SITE" "$PYTHON" -m mlx_lm lora \
        -c "$CONFIG" --grad-checkpoint 2>&1 | tee "$LOG"; then
        echo "Done: $EXP"
    else
        echo "FAILED: $EXP (exit $?) — continuing to next experiment"
    fi
    echo ""
done

echo "All experiments complete."
