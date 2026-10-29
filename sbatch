#!/bin/bash
#SBATCH -A cis260312p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH -t 01:00:00
#SBATCH --job-name=alpasim_vavam
#SBATCH --output=vavam_%j.log
#SBATCH --error=vavam_%j.err

set -e
export PYTHONUNBUFFERED=1

WORKDIR="/ocean/projects/cis260312p/mcropper/Documents/alpasim"
CONTAINER="${WORKDIR}/containers/nre-26.04.sif"
COMPOSE="${WORKDIR}/test_vavam_gpu/docker-compose.yaml"

cd "${WORKDIR}"
source setup_local_env.sh

if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    exit 1
fi

if [ ! -f "${COMPOSE}" ]; then
    echo "ERROR: Compose file not found at ${COMPOSE}"
    exit 1
fi

# Ensure all background jobs are terminated on exit or abort
cleanup() {
    echo "Cleaning up background services..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. Parse service commands directly from docker-compose.yaml
extract_cmd() {
    python3 -c "
import yaml, sys
with open('${COMPOSE}') as f:
    cfg = yaml.safe_load(f)
cmd = cfg.get('services', {}).get('$1', {}).get('command', '')
if isinstance(cmd, list):
    import shlex
    print(' '.join(shlex.quote(str(x)) for x in cmd))
else:
    print(cmd or '')
"
}

CMD_NRE=$(extract_cmd "renderer")
[ -z "${CMD_NRE}" ] && CMD_NRE=$(extract_cmd "sensorsim")
CMD_PHYSICS=$(extract_cmd "physics")
CMD_CONTROLLER=$(extract_cmd "controller")
CMD_DRIVER=$(extract_cmd "driver")
CMD_RUNTIME=$(extract_cmd "runtime-0")
[ -z "${CMD_RUNTIME}" ] && CMD_RUNTIME=$(extract_cmd "runtime")

echo "=== Extracted Commands ==="
echo "NRE: ${CMD_NRE}"
echo "Physics: ${CMD_PHYSICS}"
echo "Controller: ${CMD_CONTROLLER}"
echo "Driver: ${CMD_DRIVER}"
echo "Runtime: ${CMD_RUNTIME}"
echo "=========================="

# 2. Launch NRE renderer inside Singularity container
echo "Starting NRE Renderer..."
singularity exec --nv \
    -B /ocean/projects/cis260312p/mcropper:/ocean/projects/cis260312p/mcropper \
    "${CONTAINER}" ${CMD_NRE} > "${WORKDIR}/test_vavam_gpu/nre.log" 2>&1 &
sleep 5

# 3. Launch native Python microservices in background
echo "Starting Physics Service..."
uv run ${CMD_PHYSICS} > "${WORKDIR}/test_vavam_gpu/physics.log" 2>&1 &
sleep 2

echo "Starting Controller Service..."
uv run ${CMD_CONTROLLER} > "${WORKDIR}/test_vavam_gpu/controller.log" 2>&1 &
sleep 2

echo "Starting Driver (VaVAM)..."
uv run ${CMD_DRIVER} > "${WORKDIR}/test_vavam_gpu/driver.log" 2>&1 &
sleep 5

# 4. Run the runtime orchestrator in foreground
echo "Executing Runtime Simulation..."
uv run ${CMD_RUNTIME}

echo "Simulation complete! Check metrics in ${WORKDIR}/test_vavam_gpu/eval/"
