#!/bin/bash
# Download model weights into $HF_HOME on /ocean. Run on a LOGIN node
# (compute nodes have no internet) with HF_TOKEN set or after `hf auth login`.
#   bash benchmark/download_weights.sh
set -euo pipefail
source benchmark/env.sh
mkdir -p "$HF_HOME"

CHECKPOINTS=(nvidia/Alpamayo-1.5-10B SIMFORGE-AI/SIMFORGE-D1.5)
for model in "${CHECKPOINTS[@]}"; do
  uv run hf download "$model"
done

# Each checkpoint builds its VLM from the processor/config named by
# vlm_name_or_path (e.g. nvidia/Cosmos-Reason2-8B for Alpamayo 1.5); the driver
# also loads the Qwen3-VL-2B processor. Fetch their config/tokenizer files only.
# A bare folder name (SIMFORGE's "vlm") ships inside the checkpoint itself.
VLM_REPOS=(Qwen/Qwen3-VL-2B-Instruct)
for model in "${CHECKPOINTS[@]}"; do
  config=$(uv run hf download "$model" config.json)
  vlm=$(uv run python -c "import json, sys; print(json.load(open(sys.argv[1]))['vlm_name_or_path'])" "$config")
  [[ "$vlm" == */* ]] && VLM_REPOS+=("$vlm")
done
for model in "${VLM_REPOS[@]}"; do
  uv run hf download "$model" --exclude "*.safetensors" "*.bin" "*.pt" "*.png"
done

# Fail here rather than on the compute node: build every config offline.
HF_HUB_OFFLINE=1 uv run python -c "
import sys
from alpasim_driver.models.alpamayo1_5_model import load_checkpoint_config
for checkpoint in sys.argv[1:]:
    print(checkpoint, '-> VLM', load_checkpoint_config(checkpoint, 'sdpa').vlm_name_or_path)
" "${CHECKPOINTS[@]}"
echo "Weights cached under $HF_HOME"
