"""
detection.py
─────────────────────────────────────────────────────────────────────────────
Stage 2 – Obstacle detection from a preprocessed (ground-removed) point cloud.

Pipeline:
  1. Voxel downsampling         – reduce density for faster clustering
  2. DBSCAN-style cluster scan  – group obstacle points into candidate objects
  3. Bounding-box extraction    – axis-aligned box per cluster
  4. Classification hints       – heuristic: vehicle / pedestrian / cyclist /
                                  static based on box dimensions & point count
  5. Range & TTC estimation     – needed by the physics-based safety validator

All outputs feed directly into physics_reasoning.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .preprocessing import ProcessedSpin


# ── Types ─────────────────────────────────────────────────────────────────────

class ObjectClass(str, Enum):
    VEHICLE    = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST    = "cyclist"
    STATIC     = "static"
    UNKNOWN    = "unknown"


@dataclass
class BoundingBox3D:
    """Axis-aligned 3-D bounding box in the rig frame."""
    center: np.ndarray     # (3,) x,y,z metres
    extent: np.ndarray     # (3,) half-widths: dx, dy, dz metres
    yaw: float = 0.0       # heading (rad, 0 = forward) – set by OBB step

    @property
    def volume(self) -> float:
        return float(np.prod(self.extent * 2))

    @property
    def footprint_area(self) -> float:
        return float(4 * self.extent[0] * self.extent[1])


@dataclass
class DetectedObject:
    """One detected obstacle cluster."""
    object_id: int
    cls: ObjectClass
    bbox: BoundingBox3D
    n_points: int
    range_m: float          # distance from sensor origin to bbox centre
    confidence: float       # heuristic score 0-1
    relative_velocity: Optional[float] = None   # m/s along range axis (from tracking)
    ttc_s: Optional[float] = None               # time-to-collision (s); None if not converging


@dataclass
class DetectionFrame:
    """All detected objects for one spin."""
    scan_index: int
    timestamp_us: int
    objects: List[DetectedObject] = field(default_factory=list)

    @property
    def nearest(self) -> Optional[DetectedObject]:
        return min(self.objects, key=lambda o: o.range_m, default=None)

    @property
    def nearest_range_m(self) -> float:
        obj = self.nearest
        return obj.range_m if obj else float("inf")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    # Voxel downsampling
    voxel_size: float = 0.15          # metres per voxel side

    # Clustering (simplified DBSCAN parameters)
    cluster_eps: float = 0.6          # neighbourhood radius (m)
    cluster_min_pts: int = 5          # minimum points per cluster

    # Object filtering
    min_cluster_points: int = 5
    max_cluster_points: int = 20_000
    min_extent_m: float = 0.3         # ignore tiny noise clusters
    max_extent_m: float = 20.0        # ignore implausibly large blobs

    # Forward-facing detection zone for safety (can be widened)
    fov_half_angle_deg: float = 180.0  # 180° = full 360° used


# ── Detector ─────────────────────────────────────────────────────────────────

class LidarDetector:
    """
    Detect 3-D objects from a preprocessed LiDAR spin.

    Parameters
    ----------
    config : DetectionConfig
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.cfg = config or DetectionConfig()

    def detect(self, spin: ProcessedSpin) -> DetectionFrame:
        """Run full detection pipeline on a ProcessedSpin."""
        obs_xyz = spin.xyz_obstacle
        obs_int = spin.intensity[~spin.is_ground]

        if len(obs_xyz) == 0:
            return DetectionFrame(spin.scan_index, spin.timestamp_us)

        # 1. Voxel downsample
        vox_xyz, vox_int = _voxel_downsample(obs_xyz, obs_int, self.cfg.voxel_size)

        # 2. FOV filter
        if self.cfg.fov_half_angle_deg < 180.0:
            vox_xyz, vox_int = _fov_filter(
                vox_xyz, vox_int, self.cfg.fov_half_angle_deg
            )

        # 3. Cluster
        labels = _dbscan_cluster(vox_xyz, self.cfg.cluster_eps, self.cfg.cluster_min_pts)

        # 4. Extract objects
        objects = []
        unique_labels = np.unique(labels)
        obj_id = 0
        for lbl in unique_labels:
            if lbl < 0:    # noise label
                continue
            mask = labels == lbl
            cluster_pts = vox_xyz[mask]
            cluster_int = vox_int[mask]

            n = len(cluster_pts)
            if n < self.cfg.min_cluster_points or n > self.cfg.max_cluster_points:
                continue

            bbox = _fit_bbox(cluster_pts)

            # Filter by extent plausibility
            max_ext = float(bbox.extent.max() * 2)
            if max_ext < self.cfg.min_extent_m or max_ext > self.cfg.max_extent_m:
                continue

            cls = _classify_object(bbox, n, cluster_int.mean())
            range_m = float(np.linalg.norm(bbox.center))
            conf = _confidence_score(n, bbox, cls)

            objects.append(DetectedObject(
                object_id=obj_id,
                cls=cls,
                bbox=bbox,
                n_points=n,
                range_m=range_m,
                confidence=conf,
            ))
            obj_id += 1

        # Sort by range (nearest first)
        objects.sort(key=lambda o: o.range_m)

        return DetectionFrame(
            scan_index=spin.scan_index,
            timestamp_us=spin.timestamp_us,
            objects=objects,
        )


# ── Object tracker (inter-frame velocity estimation) ─────────────────────────

class SimpleObjectTracker:
    """
    Track detected objects across consecutive spins to estimate velocity
    and compute time-to-collision.

    Uses nearest-neighbour association based on 3-D centre proximity.
    """

    def __init__(self, max_association_dist: float = 3.0):
        self._prev: Optional[DetectionFrame] = None
        self._prev_ts: Optional[int] = None
        self.max_dist = max_association_dist

    def update(self, frame: DetectionFrame) -> DetectionFrame:
        if self._prev is None or self._prev_ts is None:
            self._prev = frame
            self._prev_ts = frame.timestamp_us
            return frame

        dt_s = max((frame.timestamp_us - self._prev_ts) / 1e6, 1e-3)

        for obj in frame.objects:
            # Find best match in previous frame
            best_prev = None
            best_dist = self.max_dist
            for prev_obj in self._prev.objects:
                d = np.linalg.norm(obj.bbox.center - prev_obj.bbox.center)
                if d < best_dist:
                    best_dist = d
                    best_prev = prev_obj

            if best_prev is not None:
                # Velocity along range axis (positive = approaching)
                delta_range = best_prev.range_m - obj.range_m
                rel_vel = delta_range / dt_s  # m/s (positive = closing)
                obj.relative_velocity = rel_vel

                if rel_vel > 0.1 and obj.range_m > 0.1:
                    obj.ttc_s = obj.range_m / rel_vel
                else:
                    obj.ttc_s = float("inf")

        self._prev = frame
        self._prev_ts = frame.timestamp_us
        return frame


# ── Helpers ───────────────────────────────────────────────────────────────────

def _voxel_downsample(
    xyz: np.ndarray, intensity: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Average points within each voxel cell."""
    indices = np.floor(xyz / voxel_size).astype(np.int32)
    keys = indices[:, 0] * 1_000_003 + indices[:, 1] * 1009 + indices[:, 2]
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_vox = len(counts)
    xyz_out = np.zeros((n_vox, 3), dtype=np.float32)
    int_out = np.zeros(n_vox, dtype=np.float32)
    np.add.at(xyz_out, inv, xyz)
    np.add.at(int_out, inv, intensity)
    xyz_out /= counts[:, None]
    int_out /= counts
    return xyz_out, int_out


def _fov_filter(
    xyz: np.ndarray, intensity: np.ndarray, half_angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    mask = np.abs(angles) <= half_angle_deg
    return xyz[mask], intensity[mask]


def _dbscan_cluster(xyz: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """
    DBSCAN for 3-D point cloud clustering.

    A point is a core point when at least ``min_pts`` points (itself included)
    lie within ``eps``.  Clusters are the connected components of core points
    within ``eps`` of each other; a non-core point within ``eps`` of a core
    point joins that point's cluster (ties are broken arbitrarily).

    Returns label array (−1 = noise, 0+ = cluster id).
    """
    n = len(xyz)
    if n == 0:
        return np.array([], dtype=np.int32)

    # An unbalanced, non-compacted tree builds several times faster and the
    # pair query on it is no slower for voxelized clouds.
    tree = cKDTree(xyz[:, :3], balanced_tree=False, compact_nodes=False)
    pairs = tree.query_pairs(eps, output_type="ndarray").astype(np.int32)
    n_neighbours = 1 + np.bincount(pairs.ravel(), minlength=n)
    core = n_neighbours >= min_pts

    labels = np.full(n, -1, dtype=np.int32)
    core_idx = np.flatnonzero(core)
    if len(core_idx) == 0:
        return labels

    a_core = core[pairs[:, 0]]
    b_core = core[pairs[:, 1]]
    # Connected components over the core points only, renumbered 0..n_core-1.
    core_number = np.full(n, -1, dtype=np.int32)
    core_number[core_idx] = np.arange(len(core_idx), dtype=np.int32)
    core_pairs = pairs[a_core & b_core]
    graph = csr_matrix(
        (
            np.ones(len(core_pairs), dtype=np.int8),
            (core_number[core_pairs[:, 0]], core_number[core_pairs[:, 1]]),
        ),
        shape=(len(core_idx), len(core_idx)),
    )
    _, component = connected_components(graph, directed=False)
    _, labels[core_idx] = np.unique(component, return_inverse=True)

    # Border points: one endpoint core, the other not.
    for border, a, b in ((a_core & ~b_core, 0, 1), (b_core & ~a_core, 1, 0)):
        labels[pairs[border, b]] = labels[pairs[border, a]]
    return labels


def _fit_bbox(pts: np.ndarray) -> BoundingBox3D:
    center = pts.mean(axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extent = (maxs - mins) / 2.0
    extent = np.maximum(extent, 0.01)
    return BoundingBox3D(center=center, extent=extent)


# Heuristic classification from box dimensions (KITTI/nuScenes conventions)
_CLASS_RULES = [
    # (cls, l_range, w_range, h_range)
    (ObjectClass.VEHICLE,    (2.5, 10.0), (1.2, 3.0),  (0.8, 2.5)),
    (ObjectClass.PEDESTRIAN, (0.3, 1.0),  (0.2, 0.8),  (0.8, 2.2)),
    (ObjectClass.CYCLIST,    (0.8, 2.2),  (0.3, 0.9),  (0.8, 2.2)),
    (ObjectClass.STATIC,     (0.1, 5.0),  (0.1, 5.0),  (0.0, 15.0)),
]


def _classify_object(bbox: BoundingBox3D, n_pts: int, mean_int: float) -> ObjectClass:
    dx, dy, dz = bbox.extent * 2  # full dimensions
    length = max(dx, dy)
    width = min(dx, dy)
    height = dz

    for cls, (l0, l1), (w0, w1), (h0, h1) in _CLASS_RULES:
        if l0 <= length <= l1 and w0 <= width <= w1 and h0 <= height <= h1:
            return cls
    return ObjectClass.UNKNOWN


def _confidence_score(n_pts: int, bbox: BoundingBox3D, cls: ObjectClass) -> float:
    """Heuristic confidence: more points and expected class → higher score."""
    point_score = min(n_pts / 300.0, 1.0)
    density = n_pts / max(bbox.volume, 0.01)
    density_score = min(density / 50.0, 1.0)
    return float(np.clip(0.4 * point_score + 0.6 * density_score, 0.1, 0.99))
