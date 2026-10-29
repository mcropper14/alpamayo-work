"""
physical_ai_av_lidar
────────────────────────────────────────────────────────────────────────────
LiDAR processing pipeline for the nvidia/PhysicalAI-Autonomous-Vehicles
dataset, supporting physics-based reasoning in the Alpamayo safety selector.

Modules
───────
  schema            – Field definitions, constants, data classes
  loader            – Parquet loading (local, HuggingFace, synthetic)
  preprocessing     – Range filter, motion compensation, ground segmentation
  detection         – Voxel clustering, bounding boxes, object classification
  physics_reasoning – TTC, occupancy grid, corridor check, action recommendation
  pipeline          – End-to-end orchestrator

Quick start (no dataset download required)
──────────────────────────────────────────
  from physical_ai_av_lidar.pipeline import LidarPipeline

  pipe = LidarPipeline()
  pipe.load_synthetic(scenario="emergency_stop")

  for report in pipe.run():
      print(f"Spin {report.scan_index:3d} | "
            f"nearest {report.nearest_range_m:6.1f}m | "
            f"TTC {report.nearest_ttc_s:5.1f}s | "
            f"action {report.recommended_action.value}")
"""

from .schema import (
    LidarExtrinsics,
    ClipMetadata,
    LIDAR_HZ,
    SPINS_PER_CLIP,
    PHYSICS_MIN_TTC_S,
    PHYSICS_MAX_LATERAL_ACCEL,
    PHYSICS_MAX_LONG_ACCEL,
    PHYSICS_MAX_DECEL,
)
from .loader import LidarLoader, SpinFrame, find_chunk_for_clip
from .preprocessing import LidarPreprocessor, PreprocessConfig, ProcessedSpin
from .detection import (
    LidarDetector,
    SimpleObjectTracker,
    DetectionConfig,
    DetectionFrame,
    DetectedObject,
    ObjectClass,
    BoundingBox3D,
)
from .physics_reasoning import (
    PhysicsReasoner,
    PhysicsConfig,
    PhysicsReport,
    ConstraintResult,
    ConstraintStatus,
    RecommendedAction,
    OccupancyGrid,
)
from .pipeline import LidarPipeline, PipelineConfig, SpinTiming
from .camera_loader import (
    CameraLoader,
    MultiCameraLoader,
    CAMERA_NAMES,
    CAMERA_LABELS,
)

__all__ = [
    # Schema
    "LidarExtrinsics", "ClipMetadata",
    "LIDAR_HZ", "SPINS_PER_CLIP",
    "PHYSICS_MIN_TTC_S", "PHYSICS_MAX_LATERAL_ACCEL",
    "PHYSICS_MAX_LONG_ACCEL", "PHYSICS_MAX_DECEL",
    # Loader
    "LidarLoader", "SpinFrame", "find_chunk_for_clip",
    # Preprocessing
    "LidarPreprocessor", "PreprocessConfig", "ProcessedSpin",
    # Detection
    "LidarDetector", "SimpleObjectTracker", "DetectionConfig",
    "DetectionFrame", "DetectedObject", "ObjectClass", "BoundingBox3D",
    # Physics
    "PhysicsReasoner", "PhysicsConfig", "PhysicsReport",
    "ConstraintResult", "ConstraintStatus", "RecommendedAction", "OccupancyGrid",
    # Pipeline
    "LidarPipeline", "PipelineConfig", "SpinTiming",
    # Camera
    "CameraLoader", "MultiCameraLoader", "CAMERA_NAMES", "CAMERA_LABELS",
]

__version__ = "0.1.0"