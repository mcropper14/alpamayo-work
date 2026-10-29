#!/bin/bash
# Generate the run directory for one benchmark arm. Run on a LOGIN node: the
# wizard downloads missing scenes from Hugging Face.
#   bash benchmark/prepare.sh ARM [SCENES_FILE]
# ARM is one of the keys of ARM_OVERRIDES in benchmark/env.sh.  Set RUNS_ROOT
# and N_ROLLOUTS to prepare a smoke run outside benchmark/runs, e.g.
#   RUNS_ROOT=$PWD/benchmark/smoke_runs N_ROLLOUTS=1 bash benchmark/prepare.sh ARM benchmark/scenes_smoke.txt
set -euo pipefail
source benchmark/env.sh

ARM=$1
SCENES_FILE=${2:-benchmark/scenes_dev.txt}
OVERRIDES=$(arm_overrides "$ARM")
RUN_DIR="$RUNS_ROOT/$ARM"

if [[ -e "$RUN_DIR" ]]; then
  echo "$RUN_DIR already exists; move or delete it first." >&2
  exit 1
fi

mkdir -p "$HF_HOME"
SCENE_IDS=$(grep -v '^#' "$SCENES_FILE" | grep -v '^$' | paste -sd, -)

# shellcheck disable=SC2086  # OVERRIDES is a list of Hydra overrides
uv run alpasim_wizard \
  deploy=local topology=1gpu $OVERRIDES \
  wizard.dry_run=true \
  wizard.log_dir="$RUN_DIR" \
  "scenes.scene_ids=[$SCENE_IDS]" \
  runtime.simulation_config.n_rollouts=$N_ROLLOUTS

# The wizard writes container paths and docker-compose hostnames; the sbatch
# runs services directly on the compute node instead.
sed -i \
  -e "s#/mnt/nre-data#$NRE_ARTIFACTS#g" \
  -e "s#\b\(driver\|physics\|renderer\|controller\|prometheus\)-0:#localhost:#g" \
  "$RUN_DIR"/generated-user-config-*.yaml "$RUN_DIR/generated-network-config.yaml"
sed -i \
  -e "s#/mnt/drivers#$DRIVERS_DIR#g" \
  -e "s#/mnt/output/driver#$RUN_DIR/driver#g" \
  "$RUN_DIR/driver-config.yaml"

if grep -n "/mnt/" "$RUN_DIR"/generated-*.yaml "$RUN_DIR/driver-config.yaml"; then
  echo "WARNING: container paths left in the generated configs above." >&2
fi
echo "Prepared $RUN_DIR. Submit with: sbatch benchmark/run_benchmark.sbatch $ARM $RUNS_ROOT"
