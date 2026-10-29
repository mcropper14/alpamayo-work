"""
pipeline.py
─────────────────────────────────────────────────────────────────────────────
Top-level orchestrator for the LiDAR processing pipeline.

Ties together:
  LidarLoader  →  LidarPreprocessor  →  LidarDetector  →
  SimpleObjectTracker  →  PhysicsReasoner  →  PhysicsReport

Intended as the interface consumed by safety_validator.py in the autonomous
safety selector.

Usage
─────
  from physical_ai_av_lidar.pipeline import LidarPipeline, PipelineConfig

  pipe = LidarPipeline(PipelineConfig())

  # Option A: from real data
  pipe.load_from_file("path/to/<uuid>.lidar_top360_fov.parquet")

  # Option B: from synthetic data (no dataset access needed)
  pipe.load_synthetic(scenario="emergency_stop")

  for report in pipe.run():
      print(report.recommended_action, report.nearest_range_m)

  # Or process a single spin by index:
  report = pipe.process_spin(scan_index=5, ego_speed_ms=25.0)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from .loader import LidarLoader, SpinFrame
from .preprocessing import LidarPreprocessor, PreprocessConfig, ProcessedSpin
from .detection import DetectionConfig, DetectionFrame, LidarDetector, SimpleObjectTracker
from .physics_reasoning import PhysicsConfig, PhysicsReasoner, PhysicsReport
from .schema import ClipMetadata, LidarExtrinsics


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    extrinsics: LidarExtrinsics = field(default_factory=LidarExtrinsics)
    # Tracker
    tracker_max_assoc_dist: float = 3.0
    # Ego state (populated per-spin externally, or estimated from ego_df)
    default_ego_speed: float = 0.0


# ── Timing diagnostics ────────────────────────────────────────────────────────

@dataclass
class SpinTiming:
    scan_index: int
    load_ms: float = 0.0
    preprocess_ms: float = 0.0
    detect_ms: float = 0.0
    track_ms: float = 0.0
    physics_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.load_ms + self.preprocess_ms + self.detect_ms + self.track_ms + self.physics_ms


# ── Pipeline ──────────────────────────────────────────────────────────────────

class LidarPipeline:
    """
    End-to-end LiDAR processing pipeline.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._preprocessor = LidarPreprocessor(
            config=self.cfg.preprocess,
            extrinsics=self.cfg.extrinsics,
        )
        self._detector = LidarDetector(config=self.cfg.detection)
        self._tracker = SimpleObjectTracker(
            max_association_dist=self.cfg.tracker_max_assoc_dist
        )
        self._reasoner = PhysicsReasoner(config=self.cfg.physics)
        self._loader: Optional[LidarLoader] = None
        self._prev_ego_speed: Optional[float] = None
        self._prev_timestamp_us: Optional[int] = None

    # ── Loaders ───────────────────────────────────────────────────────────────

    def load_from_file(
        self,
        lidar_path: str,
        ego_path: Optional[str] = None,
        metadata: Optional[ClipMetadata] = None,
    ) -> "LidarPipeline":
        self._loader = LidarLoader.from_file(lidar_path, ego_path, metadata)
        self._reset_state()
        return self

    def load_from_hf(
        self,
        clip_uuid: str,
        chunk_id: int = 0,
        cache_dir: Optional[str] = None,
    ) -> "LidarPipeline":
        self._loader = LidarLoader.from_hf(clip_uuid, chunk_id, cache_dir)
        self._reset_state()
        return self

    def load_synthetic(
        self,
        scenario: str = "highway",
        n_spins: int = 20,
        seed: int = 42,
    ) -> "LidarPipeline":
        self._loader = LidarLoader.synthetic(
            scenario=scenario, n_spins=n_spins, seed=seed
        )
        self._reset_state()
        return self

    # ── Main entry points ─────────────────────────────────────────────────────

    def run(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        ego_speed_ms: Optional[float] = None,
    ) -> Iterator[PhysicsReport]:
        """
        Process all spins and yield a PhysicsReport per spin.

        Parameters
        ----------
        ego_speed_ms : Optional[float]
            Override ego speed for all spins. If None, extracted from ego_df.
        """
        if self._loader is None:
            raise RuntimeError("No data loaded. Call load_from_file() or load_synthetic() first.")

        for spin in self._loader.iter_spins(start=start, stop=stop):
            speed = ego_speed_ms or self._extract_ego_speed(spin)
            lat_a, lon_a, yaw_r = self._extract_dynamics(spin)
            yield self._process_one_spin(spin, speed, lat_a, lon_a, yaw_r)

    def process_spin(
        self,
        scan_index: int,
        ego_speed_ms: float = 0.0,
        lateral_accel_ms2: float = 0.0,
        longitudinal_accel_ms2: float = 0.0,
        yaw_rate_rads: float = 0.0,
    ) -> PhysicsReport:
        """Process a single spin by index with explicit ego state."""
        if self._loader is None:
            raise RuntimeError("No data loaded.")
        spin = self._loader.get_spin(scan_index)
        return self._process_one_spin(
            spin, ego_speed_ms, lateral_accel_ms2, longitudinal_accel_ms2, yaw_rate_rads
        )

    def run_with_timing(
        self, start: int = 0, stop: Optional[int] = None
    ) -> Iterator[tuple[PhysicsReport, SpinTiming]]:
        """Same as run() but also yields timing diagnostics."""
        if self._loader is None:
            raise RuntimeError("No data loaded.")

        for spin in self._loader.iter_spins(start=start, stop=stop):
            timing = SpinTiming(scan_index=spin.scan_index)
            speed = self._extract_ego_speed(spin)
            lat_a, lon_a, yaw_r = self._extract_dynamics(spin)

            t0 = time.perf_counter()
            processed = self._preprocessor.process(spin)
            timing.preprocess_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            det_frame = self._detector.detect(processed)
            timing.detect_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            det_frame = self._tracker.update(det_frame)
            timing.track_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            dt_s = self._dt_seconds(spin.timestamp_us)
            report = self._reasoner.reason(
                det_frame, speed, lat_a, lon_a, yaw_r,
                prev_ego_speed=self._prev_ego_speed, dt_s=dt_s,
            )
            timing.physics_ms = (time.perf_counter() - t0) * 1000

            self._update_state(speed, spin.timestamp_us)
            yield report, timing

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _process_one_spin(
        self,
        spin: SpinFrame,
        ego_speed_ms: float,
        lateral_accel_ms2: float,
        longitudinal_accel_ms2: float,
        yaw_rate_rads: float,
    ) -> PhysicsReport:
        processed = self._preprocessor.process(spin)
        det_frame = self._detector.detect(processed)
        det_frame = self._tracker.update(det_frame)
        dt_s = self._dt_seconds(spin.timestamp_us)
        report = self._reasoner.reason(
            det_frame,
            ego_speed_ms,
            lateral_accel_ms2,
            longitudinal_accel_ms2,
            yaw_rate_rads,
            prev_ego_speed=self._prev_ego_speed,
            dt_s=dt_s,
        )
        self._update_state(ego_speed_ms, spin.timestamp_us)
        return report

    def _extract_ego_speed(self, spin: SpinFrame) -> float:
        """Extract ego speed from ego_pose or fall back to config default."""
        if spin.ego_pose is not None and len(spin.ego_pose) >= 7:
            # ego_pose = [x, y, z, qx, qy, qz, qw]
            # If vx,vy,vz available (indices 7-9):
            if len(spin.ego_pose) >= 10:
                import numpy as np
                vx = float(spin.ego_pose[7])
                vy = float(spin.ego_pose[8])
                return float((vx**2 + vy**2) ** 0.5)
        return self.cfg.default_ego_speed

    def _extract_dynamics(self, spin: SpinFrame):
        """Return (lateral_a, long_a, yaw_rate) – zeros if not available."""
        return 0.0, 0.0, 0.0

    def _dt_seconds(self, timestamp_us: int) -> float:
        if self._prev_timestamp_us is None:
            return 0.1  # nominal LiDAR period
        return max((timestamp_us - self._prev_timestamp_us) / 1e6, 1e-4)

    def _update_state(self, speed: float, timestamp_us: int):
        self._prev_ego_speed = speed
        self._prev_timestamp_us = timestamp_us

    def _reset_state(self):
        self._prev_ego_speed = None
        self._prev_timestamp_us = None
        self._tracker = SimpleObjectTracker(
            max_association_dist=self.cfg.tracker_max_assoc_dist
        )

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def n_spins(self) -> int:
        return len(self._loader) if self._loader else 0

    @property
    def metadata(self) -> Optional[ClipMetadata]:
        return self._loader.metadata if self._loader else None
