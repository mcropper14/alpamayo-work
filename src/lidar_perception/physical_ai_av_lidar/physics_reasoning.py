"""
physics_reasoning.py
─────────────────────────────────────────────────────────────────────────────
Stage 3 – Physics-based reasoning for the safety selector.

This module takes DetectionFrame + vehicle dynamics and produces:
  • PhysicsReport  – structured evidence for the decision validator
  • SafetyVerdict  – PASS / WARN / CRITICAL per constraint
  • RecommendedAction – what physics alone says the ego should do

These outputs are consumed by the safety_validator.py to cross-check the
Alpamayo VLA model's proposed action.

Physical constraints enforced:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  TTC         time-to-collision with nearest detected object         │
  │  Kinematics  lateral-g, longitudinal-g, yaw rate vs. vehicle limits │
  │  Safe Corridor  any object inside the predicted path corridor?      │
  │  Occupancy Grid  2-D BEV occupancy map (0.2 m resolution)          │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import (
    PHYSICS_MAX_LATERAL_ACCEL,
    PHYSICS_MAX_LONG_ACCEL,
    PHYSICS_MAX_DECEL,
    PHYSICS_MIN_TTC_S,
)
from .detection import DetectedObject, DetectionFrame, ObjectClass


# ── Types ─────────────────────────────────────────────────────────────────────

class ConstraintStatus(str, Enum):
    PASS     = "PASS"
    WARN     = "WARN"
    CRITICAL = "CRITICAL"


class RecommendedAction(str, Enum):
    MAINTAIN_SPEED    = "MAINTAIN_SPEED"
    DECELERATE        = "DECELERATE"
    HARD_BRAKE        = "HARD_BRAKE"
    EMERGENCY_STOP    = "EMERGENCY_STOP"
    LANE_CHANGE_LEFT  = "LANE_CHANGE_LEFT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
    YIELD             = "YIELD"


@dataclass
class ConstraintResult:
    name: str
    status: ConstraintStatus
    measured: float
    limit: float
    margin: float           # positive = safe, negative = violated
    description: str = ""


@dataclass
class OccupancyGrid:
    """
    Bird's-eye-view (BEV) 2-D occupancy grid.

    Origin = ego rear axle; X = forward, Y = left.
    Cell value: 0 = free, 1 = occupied, 0.5 = unknown.
    """
    data: np.ndarray        # (rows, cols) float32
    resolution: float       # metres per cell
    x_min: float            # metres (forward extent min)
    x_max: float
    y_min: float            # metres (lateral extent min)
    y_max: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def is_occupied(self, x_m: float, y_m: float) -> bool:
        col = int((x_m - self.x_min) / self.resolution)
        row = int((y_m - self.y_min) / self.resolution)
        if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
            return self.data[row, col] >= 0.5
        return False


@dataclass
class PhysicsReport:
    """
    Complete physics analysis for one spin / planning cycle.

    This is the primary output consumed by safety_validator.py.
    """
    scan_index: int
    timestamp_us: int

    # --- Kinematics (from ego-motion / IMU) ---
    ego_speed_ms: float = 0.0
    lateral_accel_ms2: float = 0.0
    longitudinal_accel_ms2: float = 0.0
    yaw_rate_rads: float = 0.0

    # --- Nearest obstacle summary ---
    nearest_object: Optional[DetectedObject] = None
    nearest_range_m: float = float("inf")
    nearest_ttc_s: float = float("inf")

    # --- All constraint results ---
    constraints: List[ConstraintResult] = field(default_factory=list)

    # --- Occupancy grid ---
    occupancy_grid: Optional[OccupancyGrid] = None

    # --- Safe corridor check ---
    path_corridor_clear: bool = True
    corridor_intrusion_dist_m: Optional[float] = None  # metres of clearance

    # --- Final physics recommendation ---
    recommended_action: RecommendedAction = RecommendedAction.MAINTAIN_SPEED
    overall_status: ConstraintStatus = ConstraintStatus.PASS

    @property
    def has_critical_violation(self) -> bool:
        return any(c.status == ConstraintStatus.CRITICAL for c in self.constraints)

    @property
    def critical_constraints(self) -> List[ConstraintResult]:
        return [c for c in self.constraints if c.status == ConstraintStatus.CRITICAL]

    def to_dict(self) -> dict:
        return {
            "scan_index": self.scan_index,
            "timestamp_us": self.timestamp_us,
            "ego_speed_ms": round(self.ego_speed_ms, 3),
            "lateral_accel_ms2": round(self.lateral_accel_ms2, 3),
            "longitudinal_accel_ms2": round(self.longitudinal_accel_ms2, 3),
            "yaw_rate_rads": round(self.yaw_rate_rads, 4),
            "nearest_range_m": round(self.nearest_range_m, 2),
            "nearest_ttc_s": round(self.nearest_ttc_s, 2) if self.nearest_ttc_s != float("inf") else None,
            "path_corridor_clear": self.path_corridor_clear,
            "recommended_action": self.recommended_action.value,
            "overall_status": self.overall_status.value,
            "constraints": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "measured": round(c.measured, 4),
                    "limit": round(c.limit, 4),
                    "margin": round(c.margin, 4),
                }
                for c in self.constraints
            ],
        }


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsConfig:
    # TTC thresholds (s)
    ttc_critical: float = PHYSICS_MIN_TTC_S       # 2.0 s
    ttc_warn: float = 4.0

    # Kinematic limits
    max_lateral_accel: float = PHYSICS_MAX_LATERAL_ACCEL   # 4.5 m/s²
    max_long_accel: float = PHYSICS_MAX_LONG_ACCEL          # 6.0 m/s²
    max_decel: float = PHYSICS_MAX_DECEL                    # 9.0 m/s²
    max_yaw_rate: float = 0.8                               # rad/s

    # Occupancy grid
    grid_resolution: float = 0.2        # m/cell
    grid_x_range: Tuple[float, float] = (-10.0, 80.0)
    grid_y_range: Tuple[float, float] = (-20.0, 20.0)

    # Safe corridor (rectangle in front of ego)
    corridor_half_width: float = 2.0    # metres either side of forward axis
    corridor_length: float = 40.0       # metres ahead

    # Warn range (obstacle within this even if TTC is OK)
    proximity_warn_m: float = 8.0
    proximity_critical_m: float = 4.0


# ── Physics Reasoner ─────────────────────────────────────────────────────────

class PhysicsReasoner:
    """
    Derive physics-based safety constraints from LiDAR detections + ego state.

    Parameters
    ----------
    config : PhysicsConfig
    """

    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.cfg = config or PhysicsConfig()

    def reason(
        self,
        detection_frame: DetectionFrame,
        ego_speed_ms: float = 0.0,
        lateral_accel_ms2: float = 0.0,
        longitudinal_accel_ms2: float = 0.0,
        yaw_rate_rads: float = 0.0,
        prev_ego_speed: Optional[float] = None,
        dt_s: float = 0.1,
    ) -> PhysicsReport:
        """
        Run all physics checks and return a PhysicsReport.

        Parameters
        ----------
        detection_frame : DetectionFrame
            Detected objects from LiDAR.
        ego_speed_ms : float
            Current ego speed in m/s (from wheel odometry / IMU).
        lateral_accel_ms2 : float
            Lateral acceleration m/s² (from IMU).
        longitudinal_accel_ms2 : float
            Longitudinal acceleration m/s² (positive = accelerating).
        yaw_rate_rads : float
            Yaw rate rad/s (from IMU gyroscope).
        prev_ego_speed : Optional[float]
            Previous cycle speed (for finite-diff accel estimate if IMU absent).
        dt_s : float
            Time since last cycle (seconds).
        """
        report = PhysicsReport(
            scan_index=detection_frame.scan_index,
            timestamp_us=detection_frame.timestamp_us,
            ego_speed_ms=ego_speed_ms,
            lateral_accel_ms2=lateral_accel_ms2,
            longitudinal_accel_ms2=longitudinal_accel_ms2,
            yaw_rate_rads=yaw_rate_rads,
        )

        # If IMU accel not provided, estimate from speed delta
        if prev_ego_speed is not None and abs(longitudinal_accel_ms2) < 0.01:
            report.longitudinal_accel_ms2 = (ego_speed_ms - prev_ego_speed) / max(dt_s, 1e-3)

        # 1. Nearest object & TTC
        nearest = detection_frame.nearest
        if nearest is not None:
            report.nearest_object = nearest
            report.nearest_range_m = nearest.range_m
            # Use tracker-derived TTC if available, else kinematic estimate
            if nearest.ttc_s is not None and nearest.ttc_s != float("inf"):
                report.nearest_ttc_s = nearest.ttc_s
            elif nearest.relative_velocity is not None and nearest.relative_velocity > 0.1:
                report.nearest_ttc_s = nearest.range_m / nearest.relative_velocity
            else:
                # Conservative: assume closing at current ego speed
                if ego_speed_ms > 0.5:
                    report.nearest_ttc_s = nearest.range_m / ego_speed_ms
                else:
                    report.nearest_ttc_s = float("inf")

        # 2. Run all constraint checks
        report.constraints = self._check_constraints(report)

        # 3. Occupancy grid
        report.occupancy_grid = self._build_occupancy_grid(detection_frame)

        # 4. Safe corridor check
        report.path_corridor_clear, report.corridor_intrusion_dist_m = (
            self._check_corridor(detection_frame, report.occupancy_grid)
        )
        if not report.path_corridor_clear:
            report.constraints.append(ConstraintResult(
                name="PATH_CORRIDOR",
                status=ConstraintStatus.CRITICAL,
                measured=report.corridor_intrusion_dist_m or 0.0,
                limit=self.cfg.corridor_half_width,
                margin=-(report.corridor_intrusion_dist_m or 0.0),
                description="Obstacle detected inside predicted path corridor",
            ))

        # 5. Derive overall status
        if any(c.status == ConstraintStatus.CRITICAL for c in report.constraints):
            report.overall_status = ConstraintStatus.CRITICAL
        elif any(c.status == ConstraintStatus.WARN for c in report.constraints):
            report.overall_status = ConstraintStatus.WARN
        else:
            report.overall_status = ConstraintStatus.PASS

        # 6. Recommend action
        report.recommended_action = self._recommend_action(report)

        return report

    # ── Constraint checks ──────────────────────────────────────────────────

    def _check_constraints(self, report: PhysicsReport) -> List[ConstraintResult]:
        results = []
        cfg = self.cfg

        # TTC
        ttc = report.nearest_ttc_s
        if ttc != float("inf"):
            if ttc < cfg.ttc_critical:
                status = ConstraintStatus.CRITICAL
            elif ttc < cfg.ttc_warn:
                status = ConstraintStatus.WARN
            else:
                status = ConstraintStatus.PASS
            results.append(ConstraintResult(
                name="TIME_TO_COLLISION",
                status=status,
                measured=ttc,
                limit=cfg.ttc_critical,
                margin=ttc - cfg.ttc_critical,
                description=f"TTC {ttc:.2f}s (threshold: {cfg.ttc_critical}s)",
            ))

        # Proximity (even if TTC OK, static obstacles close = warn)
        r = report.nearest_range_m
        if r < cfg.proximity_critical_m:
            prox_status = ConstraintStatus.CRITICAL
        elif r < cfg.proximity_warn_m:
            prox_status = ConstraintStatus.WARN
        else:
            prox_status = ConstraintStatus.PASS
        if prox_status != ConstraintStatus.PASS:
            results.append(ConstraintResult(
                name="OBSTACLE_PROXIMITY",
                status=prox_status,
                measured=r,
                limit=cfg.proximity_critical_m,
                margin=r - cfg.proximity_critical_m,
                description=f"Nearest obstacle at {r:.2f}m",
            ))

        # Lateral acceleration
        lat_a = abs(report.lateral_accel_ms2)
        lat_status = (
            ConstraintStatus.CRITICAL if lat_a > cfg.max_lateral_accel else
            ConstraintStatus.WARN     if lat_a > cfg.max_lateral_accel * 0.8 else
            ConstraintStatus.PASS
        )
        results.append(ConstraintResult(
            name="LATERAL_ACCEL",
            status=lat_status,
            measured=lat_a,
            limit=cfg.max_lateral_accel,
            margin=cfg.max_lateral_accel - lat_a,
        ))

        # Longitudinal deceleration
        lon_a = report.longitudinal_accel_ms2
        if lon_a < 0:  # braking
            braking = abs(lon_a)
            lon_status = (
                ConstraintStatus.CRITICAL if braking > cfg.max_decel else
                ConstraintStatus.WARN     if braking > cfg.max_decel * 0.8 else
                ConstraintStatus.PASS
            )
            results.append(ConstraintResult(
                name="LONGITUDINAL_DECEL",
                status=lon_status,
                measured=braking,
                limit=cfg.max_decel,
                margin=cfg.max_decel - braking,
            ))

        # Yaw rate
        yr = abs(report.yaw_rate_rads)
        yr_status = (
            ConstraintStatus.CRITICAL if yr > cfg.max_yaw_rate else
            ConstraintStatus.WARN     if yr > cfg.max_yaw_rate * 0.8 else
            ConstraintStatus.PASS
        )
        results.append(ConstraintResult(
            name="YAW_RATE",
            status=yr_status,
            measured=yr,
            limit=cfg.max_yaw_rate,
            margin=cfg.max_yaw_rate - yr,
        ))

        return results

    # ── Occupancy grid ─────────────────────────────────────────────────────

    def _build_occupancy_grid(self, frame: DetectionFrame) -> OccupancyGrid:
        cfg = self.cfg
        x_min, x_max = cfg.grid_x_range
        y_min, y_max = cfg.grid_y_range
        res = cfg.grid_resolution

        cols = int((x_max - x_min) / res)
        rows = int((y_max - y_min) / res)
        grid = np.zeros((rows, cols), dtype=np.float32)

        for obj in frame.objects:
            cx, cy = obj.bbox.center[0], obj.bbox.center[1]
            ex, ey = obj.bbox.extent[0], obj.bbox.extent[1]

            # Mark cells covered by the object footprint
            x0 = max(cx - ex, x_min)
            x1 = min(cx + ex, x_max)
            y0 = max(cy - ey, y_min)
            y1 = min(cy + ey, y_max)

            col0 = int((x0 - x_min) / res)
            col1 = int((x1 - x_min) / res) + 1
            row0 = int((y0 - y_min) / res)
            row1 = int((y1 - y_min) / res) + 1

            col0 = np.clip(col0, 0, cols - 1)
            col1 = np.clip(col1, 0, cols)
            row0 = np.clip(row0, 0, rows - 1)
            row1 = np.clip(row1, 0, rows)

            grid[row0:row1, col0:col1] = 1.0

        return OccupancyGrid(
            data=grid,
            resolution=res,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
        )

    # ── Corridor check ─────────────────────────────────────────────────────

    def _check_corridor(
        self,
        frame: DetectionFrame,
        grid: OccupancyGrid,
    ) -> Tuple[bool, Optional[float]]:
        cfg = self.cfg
        clear = True
        min_intrusion = None

        for obj in frame.objects:
            cx, cy = obj.bbox.center[0], obj.bbox.center[1]
            # Is the object within the forward safety corridor?
            if 0 < cx < cfg.corridor_length and abs(cy) < cfg.corridor_half_width + obj.bbox.extent[1]:
                clear = False
                # Lateral intrusion depth
                intrusion = cfg.corridor_half_width - (abs(cy) - obj.bbox.extent[1])
                if min_intrusion is None or intrusion > min_intrusion:
                    min_intrusion = intrusion

        return clear, min_intrusion

    # ── Action recommendation ─────────────────────────────────────────────

    def _recommend_action(self, report: PhysicsReport) -> RecommendedAction:
        ttc = report.nearest_ttc_s
        r = report.nearest_range_m
        speed = report.ego_speed_ms
        cfg = self.cfg

        # Emergency situations
        if ttc < 1.0 or r < cfg.proximity_critical_m:
            return RecommendedAction.EMERGENCY_STOP

        if ttc < cfg.ttc_critical or not report.path_corridor_clear:
            return RecommendedAction.HARD_BRAKE

        if ttc < cfg.ttc_warn or r < cfg.proximity_warn_m:
            # Check if lateral evasion is feasible (object directly ahead)
            nearest = report.nearest_object
            if nearest is not None:
                lateral_offset = abs(nearest.bbox.center[1])
                if lateral_offset < 1.0:
                    # Object is in lane – need to brake or change lanes
                    return RecommendedAction.DECELERATE
                else:
                    return RecommendedAction.DECELERATE

        # Kinematic constraint violations
        if abs(report.lateral_accel_ms2) > cfg.max_lateral_accel:
            return RecommendedAction.DECELERATE
        if report.longitudinal_accel_ms2 < -cfg.max_decel:
            return RecommendedAction.HARD_BRAKE

        return RecommendedAction.MAINTAIN_SPEED
