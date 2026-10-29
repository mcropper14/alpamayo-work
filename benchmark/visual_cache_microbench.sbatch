#!/bin/bash
# Isolated vision-encoder benchmark of the visual feature cache. Repo root:
#   sbatch benchmark/visual_cache_microbench.sbatch
#SBATCH -A cis260312p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH -t 00:45:00
#SBATCH --job-name=vcache_microbench
#SBATCH --output=benchmark/logs/%x_%j.out
#SBATCH --error=benchmark/logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source benchmark/env.sh
export HF_HUB_OFFLINE=1

uv run python benchmark/visual_cache_microbench.py "$@"
