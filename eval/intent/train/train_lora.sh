#!/usr/bin/env bash
# LoRA fine-tuning on gemma-4-e4b-it-4bit via mlx-lm.
#
# Prerequisites:
#   pip install mlx-lm
#   python eval/intent/train/prepare_mlx.py   (generates mlx_train.jsonl + mlx_valid.jsonl)
#
# After training, fuse the adapter into a new model directory:
#   python -m mlx_lm.fuse \
#       --model ~/.omlx/models/gemma-4-e4b-it-4bit \
#       --adapter-path ./adapters \
#       --save-path ~/.omlx/models/gemma-4-intent-lora

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${HOME}/.omlx/models/gemma-4-e4b-it-4bit"
TRAIN_FILE="${SCRIPT_DIR}/mlx_train.jsonl"
VALID_FILE="${SCRIPT_DIR}/mlx_valid.jsonl"
ADAPTER_DIR="${SCRIPT_DIR}/adapters"

# Sanity checks
if [ ! -d "${BASE_MODEL}" ]; then
  echo "ERROR: base model not found at ${BASE_MODEL}"
  exit 1
fi

if [ ! -f "${TRAIN_FILE}" ]; then
  echo "Generating training data..."
  python3 "${SCRIPT_DIR}/prepare_mlx.py"
fi

echo "Base model : ${BASE_MODEL}"
echo "Train      : ${TRAIN_FILE}"
echo "Valid      : ${VALID_FILE}"
echo "Adapters   → ${ADAPTER_DIR}"
echo ""

python3 -m mlx_lm.lora \
  --model "${BASE_MODEL}" \
  --train \
  --data "${SCRIPT_DIR}" \
  --train-file mlx_train.jsonl \
  --valid-file mlx_valid.jsonl \
  --adapter-path "${ADAPTER_DIR}" \
  --batch-size 4 \
  --num-layers 8 \
  --lora-rank 8 \
  --lora-scale 20.0 \
  --iters 600 \
  --val-batches 5 \
  --save-every 200 \
  --learning-rate 1e-4

echo ""
echo "Training complete. Adapter saved to: ${ADAPTER_DIR}"
echo ""
echo "To fuse and register in omlx:"
echo "  python3 -m mlx_lm.fuse \\"
echo "    --model ${BASE_MODEL} \\"
echo "    --adapter-path ${ADAPTER_DIR} \\"
echo "    --save-path \${HOME}/.omlx/models/gemma-4-intent-lora"
echo ""
echo "Then run Track D:"
echo "  python3 eval/intent/run_track_d.py --model gemma-4-intent-lora"
