# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Ground-truth radar model for the driver's safety validator.

The renderer has no radar, so scans are simulated from the simulated traffic.
For each radar, every actor yields at most one detection, at the point of its
footprint closest to the radar:

- Visibility: the point is within ``max_range_m`` and ``half_fov_deg``, and not
  hidden behind a nearer actor (its azimuth inside a nearer actor's extent).
- Detection: kept with probability ``detection_probability``.
- Measurement: Gaussian noise on range, azimuth and range rate; range rate is
  the relative velocity along the line of sight (negative when closing).
- Clutter: a Poisson number of false alarms with a static object's range rate.

The model is 2-D (bird's-eye view); heights and elevation are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from alpasim_runtime.config import RadarConfig
from alpasim_utils.geometry import DynamicTrajectory, Pose, Trajectory
from alpasim_utils.radar_payload import RadarScan
from alpasim_utils.scenario import TrafficObject, TrafficObjects

# Backward difference used for ego and actor velocities.
_VELOCITY_DT_US = 50_000
_MIN_CLUTTER_RANGE_M = 2.0


@dataclass(frozen=True)
class _Kinematics:
    se3: np.ndarray  # (4, 4) pose in the local frame
    velocity_xy: np.ndarray  # (2,) m/s in the local frame


@dataclass(frozen=True)
class _Target:
    range_m: float
    azimuth_rad: float
    range_rate_mps: float
    # Azimuth interval the actor covers, None if it reaches behind the radar.
    azimuth_extent: tuple[float, float] | None


def _kinematics(trajectory: Trajectory, time_us: int) -> _Kinematics:
    se3 = trajectory.interpolate_pose(time_us).as_se3().astype(np.float64)
    prev_us = max(trajectory.time_range_us.start, time_us - _VELOCITY_DT_US)
    if prev_us == time_us:
        return _Kinematics(se3, np.zeros(2))
    prev_xy = trajectory.interpolate_pose(prev_us).as_se3()[:2, 3]
    velocity = (se3[:2, 3] - prev_xy) / ((time_us - prev_us) * 1e-6)
    return _Kinematics(se3, velocity)


def _actor_kinematics(obj: TrafficObject, time_us: int) -> _Kinematics | None:
    if obj.is_static:
        se3 = obj.trajectory.first_pose.as_se3().astype(np.float64)
        return _Kinematics(se3, np.zeros(2))
    if obj.trajectory.is_empty() or time_us not in obj.trajectory.time_range_us:
        return None  # a moving actor outside its track is not in the scene
    return _kinematics(obj.trajectory, time_us)


def _target(
    obj: TrafficObject,
    actor: _Kinematics,
    radar: _Kinematics,
    radar_from_local: np.ndarray,
) -> _Target:
    half_extents = np.array([obj.aabb.x / 2, obj.aabb.y / 2])
    radar_in_actor = np.linalg.inv(actor.se3) @ radar.se3[:, 3]
    closest_in_actor = np.append(
        np.clip(radar_in_actor[:2], -half_extents, half_extents), [0.0, 1.0]
    )
    closest_local = actor.se3 @ closest_in_actor

    corners_in_actor = np.array(
        [
            [sx * half_extents[0], sy * half_extents[1], 0.0, 1.0]
            for sx in (1, -1)
            for sy in (1, -1)
        ]
    )
    corners_in_radar = (radar_from_local @ actor.se3 @ corners_in_actor.T).T
    extent = None
    if np.all(corners_in_radar[:, 0] > 0):
        corner_azimuths = np.arctan2(corners_in_radar[:, 1], corners_in_radar[:, 0])
        extent = (float(corner_azimuths.min()), float(corner_azimuths.max()))

    in_radar = radar_from_local @ closest_local
    offset_local = closest_local[:2] - radar.se3[:2, 3]
    line_of_sight = offset_local / max(np.linalg.norm(offset_local), 1e-6)
    return _Target(
        range_m=float(np.hypot(in_radar[0], in_radar[1])),
        azimuth_rad=float(np.arctan2(in_radar[1], in_radar[0])),
        range_rate_mps=float((actor.velocity_xy - radar.velocity_xy) @ line_of_sight),
        azimuth_extent=extent,
    )


def _is_hidden(target: _Target, nearer: list[_Target]) -> bool:
    return any(
        other.azimuth_extent is not None
        and other.azimuth_extent[0] <= target.azimuth_rad <= other.azimuth_extent[1]
        for other in nearer
    )


def _scan_one(
    cfg: RadarConfig,
    ego_se3: np.ndarray,
    ego_prev_se3: np.ndarray,
    dt_s: float,
    traffic: list[tuple[TrafficObject, _Kinematics]],
    rng: np.random.Generator,
) -> list[tuple[float, ...]]:
    """Rig-frame detection rows of one radar (see ``RadarScan``), noise applied."""
    rig_to_radar = (
        Pose(
            np.asarray(cfg.rig_to_radar.translation_m, dtype=np.float32),
            np.asarray(cfg.rig_to_radar.rotation_xyzw, dtype=np.float32),
        )
        .as_se3()
        .astype(np.float64)
    )
    radar_se3 = ego_se3 @ rig_to_radar
    # The radar's own velocity, including the lever arm of the ego's yaw rate.
    radar_velocity = (
        (radar_se3[:2, 3] - (ego_prev_se3 @ rig_to_radar)[:2, 3]) / dt_s
        if dt_s > 0
        else np.zeros(2)
    )
    radar = _Kinematics(radar_se3, radar_velocity)
    radar_from_local = np.linalg.inv(radar_se3)
    half_fov = np.radians(cfg.half_fov_deg)

    targets = sorted(
        (_target(obj, actor, radar, radar_from_local) for obj, actor in traffic),
        key=lambda t: t.range_m,
    )
    detections = []
    for i, target in enumerate(targets):
        if target.range_m > cfg.max_range_m or abs(target.azimuth_rad) > half_fov:
            continue
        if _is_hidden(target, targets[:i]):
            continue
        if rng.random() >= cfg.detection_probability:
            continue
        detections.append(
            (
                target.range_m + rng.normal(0.0, cfg.range_noise_m),
                target.azimuth_rad + rng.normal(0.0, np.radians(cfg.azimuth_noise_deg)),
                target.range_rate_mps + rng.normal(0.0, cfg.range_rate_noise_mps),
            )
        )

    for _ in range(rng.poisson(cfg.false_alarms_per_scan)):
        azimuth = rng.uniform(-half_fov, half_fov)
        line_of_sight = radar_se3[:2, :2] @ np.array([np.cos(azimuth), np.sin(azimuth)])
        detections.append(
            (
                rng.uniform(_MIN_CLUTTER_RANGE_M, cfg.max_range_m),
                azimuth,
                float(-radar_velocity @ line_of_sight),
            )
        )

    # Radar polar frame -> rig frame.
    rows = []
    for range_m, azimuth, range_rate in detections:
        direction_in_radar = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
        point_in_rig = rig_to_radar[:3, :3] @ (range_m * direction_in_radar)
        point_in_rig += rig_to_radar[:3, 3]
        los_in_rig = (rig_to_radar[:3, :3] @ direction_in_radar)[:2]
        los_in_rig /= np.linalg.norm(los_in_rig)
        rows.append((*point_in_rig[:2], range_rate, *los_in_rig))
    return rows


def simulate_radar_scan(
    radars: list[RadarConfig],
    ego_trajectory: Trajectory | DynamicTrajectory,
    traffic_objs: TrafficObjects,
    time_us: int,
    rng: np.random.Generator,
) -> RadarScan:
    """Detections of all ``radars`` at ``time_us``, in the rig frame.

    Static actors must have a non-empty trajectory (their pose is its first).
    """
    ego_se3 = ego_trajectory.interpolate_pose(time_us).as_se3().astype(np.float64)
    prev_us = max(ego_trajectory.time_range_us.start, time_us - _VELOCITY_DT_US)
    ego_prev_se3 = ego_trajectory.interpolate_pose(prev_us).as_se3().astype(np.float64)
    traffic = [
        (obj, kinematics)
        for obj in traffic_objs.values()
        if (kinematics := _actor_kinematics(obj, time_us)) is not None
    ]

    rows = [
        row
        for cfg in radars
        for row in _scan_one(
            cfg, ego_se3, ego_prev_se3, (time_us - prev_us) * 1e-6, traffic, rng
        )
    ]
    detections = np.array(rows, dtype=np.float32).reshape(-1, 5)
    return RadarScan(
        timestamp_us=time_us,
        xy=detections[:, 0:2],
        range_rate=detections[:, 2],
        line_of_sight=detections[:, 3:5],
    )
