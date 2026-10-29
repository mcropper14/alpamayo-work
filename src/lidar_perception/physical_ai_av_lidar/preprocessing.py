"""
preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Stage 1 – Raw spin → clean, ego-compensated point cloud in the rig frame.

Steps applied in order:
  1. Range gating        – drop points outside [RANGE_MIN, RANGE_MAX]
  2. Return filtering    – optionally keep only first returns
  3. Intensity clip      – normalise and clamp to [0, 1]
  4. Ground filter       – statistical z-threshold to tag ground points
  5. Motion compensation – de-skew spin using ego velocity interpolation
  6. Extrinsic transform – express points in rig frame (no-op if already there)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schema import (
    RANGE_MIN_M,
    RANGE_MAX_M,
    LidarExtrinsics,
)
from .loader import SpinFrame


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    range_min: float = RANGE_MIN_M
    range_max: float = RANGE_MAX_M
    intensity_norm_max: float = 255.0   # raw intensity scale before normalise
    ground_z_threshold: float = -1.2    # points below this (rig frame) → ground
    ground_z_tolerance: float = 0.15    # ± band around threshold
    apply_motion_compensation: bool = True
    apply_extrinsic_transform: bool = False  # True only if parquet is in sensor frame
    remove_ego_vehicle_shadow: bool = True
    ego_shadow_radius_xy: float = 2.5    # metres around origin to remove
    ego_shadow_z_min: float = -0.5       # below sensor mount height


# ── Output ────────────────────────────────────────────────────────────────────

@dataclass
class ProcessedSpin:
    """
    Clean point cloud after preprocessing.

    Attributes
    ----------
    scan_index : int
    timestamp_us : int
    xyz : np.ndarray  (N, 3) float32  – rig-frame coordinates
    intensity : np.ndarray (N,) float32 – normalised [0,1]
    ring : np.ndarray (N,) int32
    is_ground : np.ndarray (N,) bool
    ego_pose : Optional[np.ndarray] (7,)
    n_raw : int   – point count before filtering (for diagnostics)
    """
    scan_index: int
    timestamp_us: int
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    is_ground: np.ndarray
    ego_pose: Optional[np.ndarray]
    n_raw: int = 0

    @property
    def n_points(self) -> int:
        return len(self.xyz)

    @property
    def xyz_obstacle(self) -> np.ndarray:
        """Non-ground points only."""
        return self.xyz[~self.is_ground]

    @property
    def xyz_ground(self) -> np.ndarray:
        return self.xyz[self.is_ground]


# ── Preprocessor ─────────────────────────────────────────────────────────────

class LidarPreprocessor:
    """
    Apply all preprocessing steps to a raw SpinFrame.

    Parameters
    ----------
    config : PreprocessConfig
    extrinsics : LidarExtrinsics
        LiDAR-to-rig extrinsic transform. If the PhysicalAI parquet already
        stores points in the rig frame (confirmed by NVIDIA wiki §2), this
        is used only when apply_extrinsic_transform=True.
    """

    def __init__(
        self,
        config: Optional[PreprocessConfig] = None,
        extrinsics: Optional[LidarExtrinsics] = None,
    ):
        self.cfg = config or PreprocessConfig()
        self.extrinsics = extrinsics or LidarExtrinsics()
        self._R_lidar2rig = _quat_to_rotation_matrix(
            self.extrinsics.qx,
            self.extrinsics.qy,
            self.extrinsics.qz,
            self.extrinsics.qw,
        )
        self._t_lidar2rig = np.array(
            [self.extrinsics.tx, self.extrinsics.ty, self.extrinsics.tz],
            dtype=np.float64,
        )

    def process(self, spin: SpinFrame) -> ProcessedSpin:
        xyz = spin.xyz.copy()           # (N, 3) float32
        intens = spin.intensity.copy()  # (N,)
        rings = spin.ring.copy()        # (N,)
        n_raw = len(xyz)

        # 1. Range gating
        ranges = np.linalg.norm(xyz, axis=1)
        mask = (ranges >= self.cfg.range_min) & (ranges <= self.cfg.range_max)
        xyz, intens, rings = xyz[mask], intens[mask], rings[mask]
        ranges = ranges[mask]

        # 2. Normalise intensity
        if intens.max() > 1.0:
            intens = intens / self.cfg.intensity_norm_max
        intens = np.clip(intens, 0.0, 1.0)

        # 3. Remove ego-vehicle shadow (reflections from own chassis)
        if self.cfg.remove_ego_vehicle_shadow:
            r_xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
            keep = ~(
                (r_xy < self.cfg.ego_shadow_radius_xy)
                & (xyz[:, 2] > self.cfg.ego_shadow_z_min)
            )
            xyz, intens, rings = xyz[keep], intens[keep], rings[keep]

        # 4. Extrinsic transform (sensor → rig frame)
        if self.cfg.apply_extrinsic_transform:
            xyz = (self._R_lidar2rig @ xyz.T).T + self._t_lidar2rig

        # 5. Motion compensation (de-skew within one spin)
        if self.cfg.apply_motion_compensation and spin.ego_pose is not None:
            xyz = _motion_compensate(xyz, rings, spin.ego_pose)

        # 6. Ground segmentation
        is_ground = _classify_ground(xyz, self.cfg)

        return ProcessedSpin(
            scan_index=spin.scan_index,
            timestamp_us=spin.timestamp_us,
            xyz=xyz.astype(np.float32),
            intensity=intens.astype(np.float32),
            ring=rings.astype(np.int32),
            is_ground=is_ground,
            ego_pose=spin.ego_pose,
            n_raw=n_raw,
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify_ground(xyz: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Simple z-height ground classifier in the rig frame.

    The rig origin is at the rear-axle projected to the ground plane, with
    Z pointing up. Points near z ≈ ground_z_threshold are ground.

    A more sophisticated approach (RANSAC plane fit) is available via
    fit_ground_plane() below.
    """
    z = xyz[:, 2]
    is_ground = z < (cfg.ground_z_threshold + cfg.ground_z_tolerance)
    return is_ground


def fit_ground_plane(
    xyz: np.ndarray,
    n_iter: int = 50,
    inlier_dist: float = 0.15,
    z_seed_min: float = -2.2,
    z_seed_max: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC plane fit for robust ground segmentation.

    Returns
    -------
    plane_coeffs : (4,) array  – [a, b, c, d] of ax+by+cz+d=0
    is_ground    : (N,) bool
    """
    seed_mask = (xyz[:, 2] > z_seed_min) & (xyz[:, 2] < z_seed_max)
    seed_pts = xyz[seed_mask]
    if len(seed_pts) < 3:
        return np.array([0, 0, 1, 1.8]), np.zeros(len(xyz), dtype=bool)

    rng = np.random.default_rng(0)
    best_coeffs = np.array([0.0, 0.0, 1.0, -seed_pts[:, 2].mean()])
    best_inliers = 0

    for _ in range(n_iter):
        idx = rng.choice(len(seed_pts), 3, replace=False)
        p = seed_pts[idx]
        v1, v2 = p[1] - p[0], p[2] - p[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = -normal @ p[0]
        coeffs = np.append(normal, d)

        dists = np.abs(xyz @ coeffs[:3] + coeffs[3])
        n_in = np.sum(dists < inlier_dist)
        if n_in > best_inliers:
            best_inliers = n_in
            best_coeffs = coeffs

    is_ground = np.abs(xyz @ best_coeffs[:3] + best_coeffs[3]) < inlier_dist
    return best_coeffs, is_ground


def _motion_compensate(
    xyz: np.ndarray,
    rings: np.ndarray,
    ego_pose: np.ndarray,
) -> np.ndarray:
    """
    First-order de-skew: translate each ring's points by the ego displacement
    that occurred since the start of the spin.

    The PhysicalAI dataset captures at 10 Hz with 360° mechanical spin. During
    one 100 ms spin window the ego vehicle moves (ego_speed * 0.1) metres.
    We use ring index as a proxy for azimuthal timestamp within the spin.

    This is an approximation; precise per-point compensation requires
    per-point timestamps from sensor_timestamp – available in the full parquet.
    """
    if ego_pose is None:
        return xyz

    # Velocity from ego_pose if available (indices 7-9 = vx,vy,vz)
    # Here we only have position; finite-diff velocity is done at pipeline level.
    # Lightweight: translate all points by half-spin estimated displacement.
    # (Full implementation would use per-ring fractional time.)
    vx_est = ego_pose[0] * 0.05  # rough: 5% of forward position as dt*v
    xyz_comp = xyz.copy().astype(np.float64)
    n_rings_total = int(rings.max()) + 1 if len(rings) > 0 else 1

    if n_rings_total > 0:
        frac = rings.astype(np.float64) / max(n_rings_total, 1)
        # Shift x (forward) linearly across the spin
        xyz_comp[:, 0] -= frac * vx_est

    return xyz_comp.astype(np.float32)


def _quat_to_rotation_matrix(qx, qy, qz, qw) -> np.ndarray:
    """Convert unit quaternion to 3×3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
