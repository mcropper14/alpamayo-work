# physical_ai_av_lidar

Physics-based LiDAR processing pipeline for **[nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)**, providing the sensor validation layer for the Alpamayo autonomous safety selector.

Processes raw Draco-compressed LiDAR spins → obstacle detection → physics constraint evaluation → recommended action, with an interactive BEV + camera visualiser.

---

## Architecture

```
nvidia/PhysicalAI-Autonomous-Vehicles (HuggingFace, gated)
           │
           │  lidar_top_360fov.chunk_XXXX.zip
           │  └── <clip_uuid>.lidar_top360_fov.parquet
           │      (200 spins × 10 Hz, Draco-compressed)
           ▼
┌──────────────────────────────────────────────┐
│  LidarLoader                                  │
│  from_hf()   · from_file()   · synthetic()   │
└──────────────┬───────────────────────────────┘
               │  SpinFrame
               ▼
┌──────────────────────────────────────────────┐
│  LidarPreprocessor                            │
│  range gate · intensity norm · ego shadow     │
│  ground segmentation · motion compensation   │
└──────────────┬───────────────────────────────┘
               │  ProcessedSpin
               ▼
┌──────────────────────────────────────────────┐
│  LidarDetector                                │
│  voxel downsample · DBSCAN cluster            │
│  bbox fit · heuristic classification          │
└──────────────┬───────────────────────────────┘
               │  DetectionFrame
               ▼
┌──────────────────────────────────────────────┐
│  SimpleObjectTracker                          │
│  nearest-neighbour association                │
│  relative velocity · TTC per object           │
└──────────────┬───────────────────────────────┘
               │  DetectionFrame (with velocity / TTC)
               ▼
┌──────────────────────────────────────────────┐
│  PhysicsReasoner                              │
│  TIME_TO_COLLISION   critical < 2.0 s        │
│  OBSTACLE_PROXIMITY  critical < 4.0 m        │
│  LATERAL_ACCEL       critical > 4.5 m/s²     │
│  LONGITUDINAL_DECEL  critical > 9.0 m/s²     │
│  YAW_RATE            critical > 0.8 rad/s    │
│  PATH_CORRIDOR       object in 2m×40m fwd    │
└──────────────┬───────────────────────────────┘
               │  PhysicsReport
               ▼
        safety_validator.py
        (Alpamayo × physics arbitration)
```

---

## Requirements

- Python 3.10+
- NVIDIA AV Dataset license accepted at https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
- HuggingFace account token with dataset access

---

## Installation

```bash
# Clone and set up the conda environment (recommended)
conda env create -f environment.yml
conda activate physai-lidar

# Or install with pip directly
pip install -e .
```

---

## Setup

### HuggingFace authentication

Log in once via the CLI — the token is persisted and picked up automatically:

```bash
huggingface-cli login
```

Or set the environment variable per session:

```bash
# Linux / macOS
export HF_TOKEN=hf_...

# Windows cmd
set HF_TOKEN=hf_...
```

> **Note:** `HF_TOKEN` in a `.env` file is listed in `.gitignore` — never commit your token.

---

## Interactive Visualiser

The main entry point. Generates a self-contained HTML file with BEV LiDAR, camera feed, physics sidebar, and timeline.

```bash
# Real clip from HuggingFace (downloads ~19 GB chunk on first run, cached after)
python visualise.py --clip-id <uuid> --output viewer.html

# Limit to first 20 spins (faster for testing)
python visualise.py --clip-id <uuid> --max-spins 20 --output viewer.html

# Higher camera resolution (default is 960×540)
python visualise.py --clip-id <uuid> --cam-width 1280 --cam-height 720 --cam-quality 85 --output viewer.html

# Synthetic data — no downloads needed
python visualise.py --scenario emergency_stop --output viewer.html
```

**Viewer controls:** `←` / `→` to step spins, `Space` to play/pause.

### Finding a valid clip ID

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset",
    filename="clip_index.parquet",
    cache_dir="./cache",
)
df = pd.read_parquet(path)

# Pick any valid training clip
valid = df[(df["clip_is_valid"] == True) & (df["split"] == "train")]
sample = valid.sample(1)
print(sample.index[0], "chunk:", sample["chunk"].iloc[0])
```

### AlpaSim scenes

AlpaSim scene IDs (prefixed `clipgt-`) are **not** raw dataset clip IDs and will not be found in `clip_index.parquet`. To use an AlpaSim scene, run AlpaSim first and load the output data with `LidarLoader.from_file()`:

```bash
uv run alpasim_wizard +deploy=local_oss \
  defines.filesystem=/home/ubuntu/alpasim/data \
  "scenes.scene_ids=[clipgt-<scene-uuid>]" \
  wizard.log_dir=/tmp/alpasim_out
```

```python
loader = LidarLoader.from_file(
    lidar_path="/tmp/alpasim_out/<scene-uuid>/lidar_top_360fov.parquet",
    ego_path="/tmp/alpasim_out/<scene-uuid>/egomotion.parquet",
)
```

---

## Python API

### Synthetic data (no downloads)

```python
from physical_ai_av_lidar import LidarPipeline

pipe = LidarPipeline()
pipe.load_synthetic(scenario="emergency_stop", n_spins=20)

for report in pipe.run(ego_speed_ms=22.0):
    print(
        f"Spin {report.scan_index:3d} | "
        f"nearest {report.nearest_range_m:6.1f} m | "
        f"TTC {report.nearest_ttc_s:5.2f} s | "
        f"{report.overall_status.value:8s} | "
        f"→ {report.recommended_action.value}"
    )
```

Scenarios: `highway`, `urban`, `emergency_stop`, `lane_change`.

### Local files

```python
pipe = LidarPipeline()
pipe.load_from_file(
    lidar_path="<uuid>.lidar_top360_fov.parquet",
    ego_path="<uuid>.egomotion.parquet",   # optional
)
for report in pipe.run():
    if report.has_critical_violation:
        for c in report.critical_constraints:
            print(f"  CRITICAL {c.name}: {c.measured:.2f} / {c.limit:.2f}")
```

### Individual pipeline stages

```python
from physical_ai_av_lidar import (
    LidarLoader, LidarPreprocessor, LidarDetector,
    SimpleObjectTracker, PhysicsReasoner,
    PreprocessConfig, DetectionConfig, PhysicsConfig,
)

loader   = LidarLoader.synthetic(scenario="urban")
pre      = LidarPreprocessor(PreprocessConfig(apply_motion_compensation=True))
det      = LidarDetector(DetectionConfig(voxel_size=0.2))
tracker  = SimpleObjectTracker()
reasoner = PhysicsReasoner(PhysicsConfig(ttc_critical=2.5))

for spin in loader.iter_spins():
    proc   = pre.process(spin)
    frame  = det.detect(proc)
    frame  = tracker.update(frame)
    report = reasoner.reason(frame, ego_speed_ms=12.0)
    print(report.to_dict())
```

---

## PhysicsReport reference

| Field | Type | Description |
|---|---|---|
| `scan_index` | int | Spin index in clip (0–199) |
| `timestamp_us` | int | Wall-clock timestamp, microseconds |
| `ego_speed_ms` | float | Ego speed m/s (from egomotion vx/vy or supplied) |
| `lateral_accel_ms2` | float | Lateral acceleration m/s² |
| `longitudinal_accel_ms2` | float | Longitudinal acceleration m/s² |
| `yaw_rate_rads` | float | Yaw rate rad/s |
| `nearest_range_m` | float | Range to nearest detected object |
| `nearest_ttc_s` | float | Time-to-collision, seconds (∞ if no closing object) |
| `path_corridor_clear` | bool | No object in 2 m × 40 m forward corridor |
| `occupancy_grid` | OccupancyGrid | 0.2 m/cell BEV occupancy map |
| `constraints` | List[ConstraintResult] | All constraint results |
| `overall_status` | ConstraintStatus | Worst-case: PASS / WARN / CRITICAL |
| `recommended_action` | RecommendedAction | Physics-derived action |
| `has_critical_violation` | bool | Any CRITICAL constraint present |

---

## Dataset format

```
nvidia/PhysicalAI-Autonomous-Vehicles/
├── clip_index.parquet              # chunk mapping for all 306,152 clips
├── lidar/lidar_top_360fov/
│   └── lidar_top_360fov.chunk_XXXX.zip
│       └── <clip_uuid>.lidar_top360_fov.parquet
├── labels/egomotion/
│   └── egomotion.chunk_XXXX.zip
│       └── <clip_uuid>.egomotion.parquet
└── camera/{sensor_name}/
    └── {sensor_name}.chunk_XXXX.zip
        └── <clip_uuid>.{sensor_name}.mp4
```

**LiDAR parquet columns (after Draco decode):**

| Column | Type | Description |
|---|---|---|
| `scan_index` | int64 | Spin number (0–199) |
| `timestamp` | int64 | System clock, microseconds |
| `x` | float32 | Forward, metres (rig frame) |
| `y` | float32 | Left, metres |
| `z` | float32 | Up, metres |
| `intensity` | float32 | Return intensity, normalised 0–1 |
| `ring` | int32 | Beam channel index |
| `return_type` | uint8 | 0=first, 1=second return |

**Rig coordinate frame:** origin at rear axle on ground, X=forward, Y=left, Z=up.

**Cameras:** 7 × fisheye (f-theta) 1080p mp4 at ~30 fps. Frame timestamps are approximate — synchronised to LiDAR by anchoring both streams to their first sample timestamp.

---

## Known limitations

- **TTC uses globally nearest object** — an object behind the vehicle can dominate the TTC and proximity constraints. No forward-only filtering before selecting the reference object.
- **LONGITUDINAL_DECEL rarely fires** — requires `prev_ego_speed` to be tracked across spins; not currently threaded through the visualiser's main loop.
- **Camera sync is approximate** — uses `t0 + dt × fps` frame seeking with no absolute timestamp alignment. The dataset ships per-clip frame timestamps parquets which are not yet consumed.
- **No lane change actions** — `LANE_CHANGE_LEFT` and `LANE_CHANGE_RIGHT` are defined in `RecommendedAction` but the action recommender always falls through to `DECELERATE` instead.
- **SafetyValidator not yet built** — the Alpamayo × physics arbitration layer (`safety_validator.py`) is designed but not implemented. The Alpamayo trajectory inversion bug (reasoning says NUDGE_LEFT, trajectory goes RIGHT) is detectable via PATH_CORRIDOR but requires SafetyValidator to cross-reference stated intent vs observed ego lateral velocity.

---

## Tests

```bash
# Standalone
python tests/test_pipeline.py

# With pytest
pytest tests/ -v
```

45 tests across loader, preprocessing, detection, physics reasoning, and pipeline integration.

---

## Project structure

```
├── visualise.py                 Interactive BEV + camera HTML viewer
├── run_example.py               CLI pipeline runner
├── run_camera.py                Camera loader test utility
├── pyproject.toml               Package metadata and dependencies
├── environment.yml              Conda environment
├── .gitignore                   Excludes dataset cache, extracted files, tokens
├── physical_ai_av_lidar/
│   ├── __init__.py              Public API
│   ├── schema.py                Field definitions, constants, dataclasses
│   ├── loader.py                LiDAR parquet loader (local / HF / synthetic)
│   ├── camera_loader.py         Camera mp4 loader and frame extractor
│   ├── preprocessing.py         Range filter, ground seg, motion compensation
│   ├── detection.py             DBSCAN clustering, bbox, classification, tracking
│   ├── physics_reasoning.py     Constraints, occupancy grid, action recommendation
│   └── pipeline.py              End-to-end orchestrator
└── tests/
    └── test_pipeline.py         45 unit + integration tests
```