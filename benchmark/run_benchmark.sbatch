#!/bin/bash
# Run one prepared benchmark arm on a single L40S. Submit from the repo root:
#   bash benchmark/prepare.sh ARM        # login node, once
#   sbatch benchmark/run_benchmark.sbatch ARM [RUNS_ROOT]
# RUNS_ROOT is an argument because Bridges-2 sets SBATCH_EXPORT=NONE: the
# submitting shell's environment does not reach the job.
#SBATCH -A cis260312p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH -t 16:00:00
#SBATCH --job-name=alpasim_bench
#SBATCH --output=benchmark/logs/%x_%j.out
#SBATCH --error=benchmark/logs/%x_%j.err

set -eo pipefail  # setup_local_env.sh reads unset variables
cd "$SLURM_SUBMIT_DIR"
source benchmark/env.sh
source setup_local_env.sh
# Compute nodes have no internet; everything must already be in $HF_HOME.
export HF_HUB_OFFLINE=1
# Renderer, physics and driver share one GPU; let freed blocks be reused
# instead of fragmenting the 48 GB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Scenes each service keeps loaded on the GPU. Matches the runtime's
# n_concurrent_rollouts (4): a larger cache grows until the GPU is full.
SCENE_CACHE_SIZE=4

ARM=$1
RUNS_ROOT=${2:-$RUNS_ROOT}
RUN_DIR="$RUNS_ROOT/$ARM"
[[ -f "$RUN_DIR/generated-user-config-0.yaml" ]] || {
  echo "Run 'bash benchmark/prepare.sh $ARM' on a login node first ($RUN_DIR)." >&2
  exit 1
}
SCENESET_DIR=$(sed -n 's/^ *data_dir: *//p' "$RUN_DIR/generated-user-config-0.yaml")
SCENESET_IN_CONTAINER="/mnt/nre-data/${SCENESET_DIR#"$NRE_ARTIFACTS"/}"
SERVICE_LOGS="$RUN_DIR/service-logs"
PTX_CACHE_DIR="$RUN_DIR/ptx_cache"
mkdir -p "$SERVICE_LOGS" "$PTX_CACHE_DIR" "$RUN_DIR/driver"

{
  echo "arm: $ARM"
  echo "slurm_job_id: $SLURM_JOB_ID"
  echo "node: $(hostname)"
  echo "git_commit: $(git rev-parse HEAD)"
  echo "git_dirty: $([[ -n $(git status --porcelain -- src) ]] && echo true || echo false)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
  echo "torch: $(uv run python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
  echo "started: $(date -Iseconds)"
} > "$RUN_DIR/benchmark-env.yaml"

cleanup() {
  echo "Shutting down background services..."
  kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting NRE renderer..."
singularity exec --nv \
  -B "$REPO:$REPO" \
  -B "$NRE_ARTIFACTS:/mnt/nre-data" \
  -B "$NRE_ARTIFACTS/ego-hoods:/mnt/ego-hoods" \
  -B "$PTX_CACHE_DIR:/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main/libs/nrend/ptx_cache" \
  "$NRE_CONTAINER" \
  /app/internal/scripts/pycena/runtime/pycena_nrm_full serve-grpc \
    --port=6006 --host=0.0.0.0 \
    --artifact-glob="$SCENESET_IN_CONTAINER/*.usdz" \
    --egocar-hood-dir=/mnt/ego-hoods \
    --no-enable-nrend \
    --download-cache-dir /tmp/nre-cache-dir \
    --cache-size=$SCENE_CACHE_SIZE --max-workers=4 \
    --enable-editing-actors > "$SERVICE_LOGS/renderer.log" 2>&1 &
RENDERER_PID=$!

echo "Starting physics..."
uv run physics_server --host=0.0.0.0 --port=6007 \
  --artifact-glob="$SCENESET_DIR/*.usdz" \
  --use-ground-mesh=true --cache-size=$SCENE_CACHE_SIZE --log-level=INFO \
  > "$SERVICE_LOGS/physics.log" 2>&1 &
PHYSICS_PID=$!

echo "Starting controller..."
uv run python -m alpasim_controller.server --port=6008 \
  --log_dir="$RUN_DIR" --log-level=INFO \
  --config="$RUN_DIR/controller-config.yaml" \
  > "$SERVICE_LOGS/controller.log" 2>&1 &
CONTROLLER_PID=$!

echo "Starting driver..."
uv run -m alpasim_driver.main \
  --config-path="$RUN_DIR" --config-name=driver-config.yaml \
  host=0.0.0.0 port=6005 \
  > "$SERVICE_LOGS/driver.log" 2>&1 &
DRIVER_PID=$!

wait_for_port() {
  local port=$1 name=$2 timeout=$3 pid=$4
  echo "Waiting for $name on port $port..."
  for ((i = 1; i <= timeout; i++)); do
    if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      echo "$name is up on port $port after ${i}s."
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name exited before binding port $port. Last log lines:" >&2
      tail -20 "$SERVICE_LOGS/$name.log" >&2
      return 1
    fi
    sleep 1
  done
  echo "TIMEOUT: $name did not bind port $port within ${timeout}s; see $SERVICE_LOGS" >&2
  return 1
}
wait_for_port 6007 physics 120 "$PHYSICS_PID"
wait_for_port 6008 controller 120 "$CONTROLLER_PID"
wait_for_port 6006 renderer 300 "$RENDERER_PID"
# Loading a 10B checkpoint from Lustre takes several minutes.
wait_for_port 6005 driver 1200 "$DRIVER_PID"

echo "Running simulation..."
uv run python -m alpasim_runtime.simulate \
  --user-config="$RUN_DIR/generated-user-config-0.yaml" \
  --network-config="$RUN_DIR/generated-network-config.yaml" \
  --eval-config="$RUN_DIR/eval-config.yaml" \
  --log-dir="$RUN_DIR" \
  --log-level=INFO

echo "finished: $(date -Iseconds)" >> "$RUN_DIR/benchmark-env.yaml"
echo "Done. Summarize with: uv run python benchmark/summarize.py"
