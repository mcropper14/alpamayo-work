"""
schema.py
─────────────────────────────────────────────────────────────────────────────
Canonical field definitions for the nvidia/PhysicalAI-Autonomous-Vehicles
LiDAR parquet format.

Dataset facts (from official README + wiki):
  • Sensor  : 1× top-mounted, 360 °-FOV rotating LiDAR
  • Capture : 10 Hz (≈ 200 spins per 20-second clip)
  • Storage : <clip_uuid>.lidar_top360_fov.parquet  (one per clip)
  • Layout  : all spins concatenated; grouped by scan_index

Rig coordinate frame (rear-axle origin, shared with cameras/radars):
  X  →  forward
  Y  →  left (when facing forward)
  Z  →  up

Per-point fields (documented in physical_ai_av devkit wiki §3 Sensor Details):
  scan_index        int64    sequential spin number within the clip (0–199)
  timestamp         int64    system wall-clock in microseconds
  sensor_timestamp  int64    sensor-internal timestamp in microseconds
  x                 float32  point position – forward (m)
  y                 float32  point position – left    (m)
  z                 float32  point position – up      (m)
  intensity         float32  return intensity  (0–255 raw, normalised 0-1 here)
  ring              int32    beam / ring index  (laser channel id)
  range             float32  radial distance from sensor origin (m)
  return_type       uint8    0 = first, 1 = second (dual-return mode)

Ego-motion parquet fields (used for motion-compensation):
  timestamp  int64    microseconds (same clock as lidar timestamps)
  x, y, z    float64  ego position in local frame (m)
  qx,qy,qz,qw float64  quaternion orientation

Sensor extrinsics parquet:
  qx,qy,qz,qw  float64  rotation quaternion (lidar-in-rig)
  x, y, z      float64  translation (lidar origin in rig frame, metres)
"""

from dataclasses import dataclass, field
from typing import Final

# ── Raw parquet column names ──────────────────────────────────────────────────

LIDAR_COLS: Final[list[str]] = [
    "scan_index",
    "timestamp",
    "sensor_timestamp",
    "x",
    "y",
    "z",
    "intensity",
    "ring",
    "range",
    "return_type",
]

EGO_MOTION_COLS: Final[list[str]] = [
    "timestamp",
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "vx", "vy", "vz",          # linear velocity (m/s) – available in some clips
    "roll", "pitch", "yaw",    # Euler angles (radians) derived from quaternion
]

EXTRINSICS_COLS: Final[list[str]] = [
    "qx", "qy", "qz", "qw",
    "x", "y", "z",
]

# ── Dataset constants ─────────────────────────────────────────────────────────

LIDAR_HZ: Final[float] = 10.0          # nominal spin rate
CLIP_DURATION_S: Final[float] = 20.0   # every clip is 20 seconds
SPINS_PER_CLIP: Final[int] = 200       # 10 Hz × 20 s

# Sensor range / quality thresholds (used by filters)
RANGE_MIN_M: Final[float] = 0.5        # ignore points within sensor blind-spot
RANGE_MAX_M: Final[float] = 200.0      # manufacturer-rated max range
INTENSITY_MIN: Final[float] = 0.0
INTENSITY_MAX: Final[float] = 1.0      # after normalisation

# Physics validation limits (shared with the safety selector)
PHYSICS_MAX_LATERAL_ACCEL:  Final[float] = 4.5    # m/s²
PHYSICS_MAX_LONG_ACCEL:     Final[float] = 6.0    # m/s²
PHYSICS_MAX_DECEL:          Final[float] = 9.0    # m/s²
PHYSICS_MIN_TTC_S:          Final[float] = 2.0    # seconds


@dataclass
class LidarExtrinsics:
    """Quaternion + translation of LiDAR in rig frame."""
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    tx: float = 0.0   # forward offset (m)
    ty: float = 0.0   # left offset    (m)
    tz: float = 1.8   # height above rear-axle (m) – approx for roof mount


@dataclass
class ClipMetadata:
    """Lightweight metadata for a single 20-second clip."""
    clip_id: str = ""
    country: str = ""
    month: int = 0
    hour_of_day: int = 0
    platform_class: str = ""       # hyperion_8 | hyperion_8.1
    extrinsics: LidarExtrinsics = field(default_factory=LidarExtrinsics)
