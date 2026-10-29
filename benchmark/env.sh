# Shared paths for the benchmark scripts. Source from the repo root.
REPO="$(pwd)"
[[ -f "$REPO/benchmark/env.sh" ]] || { echo "Run from the alpasim repo root" >&2; return 1; }

NRE_ARTIFACTS="$REPO/data/nre-artifacts"
DRIVERS_DIR="$REPO/data/drivers"
NRE_CONTAINER="$REPO/containers/nre-26.04.sif"
# Model weights live on /ocean: the home quota (25 GiB) cannot hold them.
export HF_HOME="$REPO/data/hf_cache"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1

# Wizard overrides per benchmark arm. The Alpamayo-family arms share the
# 4-camera, 6-sample public_2601 comparison preset.
declare -A ARM_OVERRIDES=(
  [vavam]="driver=vavam"
  [alpamayo1_5]="driver=alpamayo1_5 +exp/presets=alpamayo1_5_comparison_4cam"
  [simforge_d1_5]="driver=alpamayo1_5 +exp/presets=alpamayo1_5_comparison_4cam driver.model.checkpoint_path=SIMFORGE-AI/SIMFORGE-D1.5"
  # Visual feature cache arms (src/driver/src/alpasim_driver/visual_cache).
  [alpamayo1_5_vcache_exact]="driver=alpamayo1_5 +exp/presets=alpamayo1_5_comparison_4cam +driver.model.visual_cache.enabled=true"
  [alpamayo1_5_vcache_neardup]="driver=alpamayo1_5 +exp/presets=alpamayo1_5_comparison_4cam +driver.model.visual_cache.enabled=true +driver.model.visual_cache.strategy=near_duplicate"
  # Safety validator arms (src/wizard/configs/exp/sim/safety_validator*.yaml).
  # The safety presets sample 5 candidates; pin the comparison preset's 6 so
  # these arms differ from alpamayo1_5 only by the validator and its sensors.
  [alpamayo1_5_safety_lidar]="driver=alpamayo1_5 +exp/presets=alpamayo1_5_comparison_4cam +exp/sim=safety_validator driver.model.num_trajectory_samples=6"
  [alpamayo1_5_safety_radar]="driver=alpamayo1_5 +exp/presets=alpamayo1_5_comparison_4cam +exp/sim=safety_validator_radar driver.model.num_trajectory_samples=6 runtime.simulation_config.render_lidar_for_driver=false"
  [alpamayo1_5_safety_fused]="driver=alpamayo1_5 +exp/presets=alpamayo1_5_comparison_4cam +exp/sim=safety_validator_radar driver.model.num_trajectory_samples=6"
  # VaVAM gives one candidate, so the validator can only slow it down.
  [vavam_safety_fused]="driver=vavam +exp/sim=safety_validator_radar"
)
N_ROLLOUTS=${N_ROLLOUTS:-3}
# Where prepared arms live; benchmark/summarize.py reads benchmark/runs.
RUNS_ROOT=${RUNS_ROOT:-$REPO/benchmark/runs}

arm_overrides() {
  local arm=$1
  [[ -n "${ARM_OVERRIDES[$arm]:-}" ]] || {
    echo "Unknown arm '$arm'. Known: ${!ARM_OVERRIDES[*]}" >&2
    return 1
  }
  echo "${ARM_OVERRIDES[$arm]}"
}
