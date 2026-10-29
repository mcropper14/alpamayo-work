#!/bin/bash
#SBATCH -A cis260312p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH -t 04:00:00
#SBATCH --job-name=alpasim_vavam
#SBATCH --output=vavam_%j.log
#SBATCH --error=vavam_%j.err

set -e
export PYTHONUNBUFFERED=1
export UV_LINK_MODE=copy

WORKDIR="/ocean/projects/cis260312p/mcropper/Documents/alpasim"
LOGDIR="${WORKDIR}/test_vavam_gpu"
CONTAINER="${WORKDIR}/containers/nre-26.04.sif"

NRE_ARTIFACTS="${WORKDIR}/data/nre-artifacts"
EGO_HOODS="${NRE_ARTIFACTS}/ego-hoods"
SCENESET_DIR="${NRE_ARTIFACTS}/single-scene-stage"

# Driver Checkpoints & Tokenizer
VAVAM_DIR="${WORKDIR}/data/drivers/vavam"
CHECKPOINT="${VAVAM_DIR}/VAM_width_1024_pretrained_139k.pt"
TOKENIZER="${VAVAM_DIR}/VQ_ds16_16384_llamagen_encoder.jit"

# Writable PTX cache overlay for Singularity container
PTX_CACHE_DIR="${LOGDIR}/ptx_cache"
mkdir -p "${PTX_CACHE_DIR}"

cd "${WORKDIR}"
source setup_local_env.sh

cleanup() {
    echo "Shutting down background services..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. Start NRE Renderer (Singularity with GPU) on Port 6006
echo "Starting NRE Renderer..."
singularity exec --nv \
    -B "${WORKDIR}:${WORKDIR}" \
    -B "${NRE_ARTIFACTS}:/mnt/nre-data" \
    -B "${EGO_HOODS}:/mnt/ego-hoods" \
    -B "${PTX_CACHE_DIR}:/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main/libs/nrend/ptx_cache" \
    "${CONTAINER}" \
    /app/internal/scripts/pycena/runtime/pycena_nrm_full serve-grpc \
        --port=6006 \
        --host=0.0.0.0 \
        --artifact-glob="/mnt/nre-data/single-scene-stage/*.usdz" \
        --egocar-hood-dir="/mnt/ego-hoods" \
        --no-enable-nrend \
        --download-cache-dir /tmp/nre-cache-dir \
        --cache-size=16 \
        --max-workers=4 \
        --enable-editing-actors > "${LOGDIR}/nre.log" 2>&1 &
sleep 5

# 2. Start Physics Service on Port 6007
echo "Starting Physics Service..."
uv run physics_server \
    --host=0.0.0.0 \
    --port=6007 \
    --artifact-glob="${SCENESET_DIR}/*.usdz" \
    --use-ground-mesh=true \
    --cache-size=16 \
    --log-level=INFO > "${LOGDIR}/physics.log" 2>&1 &
sleep 2

# 3. Start Controller on Port 6008
echo "Starting Controller Service..."
uv run python -m alpasim_controller.server \
    --port=6008 \
    --log_dir="${LOGDIR}" \
    --log-level=INFO \
    --config="${LOGDIR}/controller-config.yaml" > "${LOGDIR}/controller.log" 2>&1 &
sleep 2

# 4. Start VaVAM Driver on Port 6005
echo "Starting VaVAM Driver..."
uv run -m alpasim_driver.main \
    --config-path="${LOGDIR}" \
    --config-name=driver-config.yaml \
    host=0.0.0.0 \
    port=6005 \
    output_dir="${LOGDIR}" \
    model.checkpoint_path="${CHECKPOINT}" \
    model.tokenizer_path="${TOKENIZER}" > "${LOGDIR}/driver.log" 2>&1 &
sleep 5


wait_for_port() {
    local port=$1
    local name=$2
    local timeout=${3:-120}
    echo "Waiting for ${name} on port ${port}..."
    for ((i=1; i<=timeout; i++)); do
        if (echo > "/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
            echo "${name} is online on port ${port} (${i}s)."
            return 0
        fi
        sleep 1
    done
    echo "TIMEOUT: ${name} failed to bind port ${port} within ${timeout}s"
    return 1
}

# Wait for all services
wait_for_port 6007 "Physics" 60
wait_for_port 6008 "Controller" 60
wait_for_port 6005 "VaVAM Driver" 120
wait_for_port 6006 "NRE Renderer" 120


# 5. Run Runtime Simulation Orchestrator (Foreground)
echo "Executing Runtime Simulation..."
uv run python -m alpasim_runtime.simulate \
    --user-config="${LOGDIR}/generated-user-config-0.yaml" \
    --network-config="${LOGDIR}/generated-network-config.yaml" \
    --log-dir="${LOGDIR}" \
    --log-level=INFO \
    --eval-config="${LOGDIR}/eval-config.yaml"

echo "Simulation completed successfully!"
