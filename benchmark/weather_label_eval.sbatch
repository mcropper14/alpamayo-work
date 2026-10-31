#!/bin/bash
# Accuracy and latency of the driver's VQA weather labels. Repo root:
#   sbatch benchmark/weather_label_eval.sbatch
#SBATCH -A cis260312p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH -t 00:45:00
#SBATCH --job-name=weather_label_eval
#SBATCH --output=benchmark/logs/%x_%j.out
#SBATCH --error=benchmark/logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source benchmark/env.sh
export HF_HUB_OFFLINE=1

uv run python benchmark/weather_label_eval.py "$@"
