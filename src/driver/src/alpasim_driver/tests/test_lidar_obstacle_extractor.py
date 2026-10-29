# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
from alpasim_driver.lidar_obstacle_extractor import LidarObstacleExtractor
from alpasim_utils.geometry import Pose
from alpasim_utils.lidar_payload import pack_lidar_spin, unpack_lidar_spin

_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
# 1 deg nose-down pitch, about the tilt of the recorded Hyperion roof LiDARs.
_PITCHED = np.array(
    [0.0, np.sin(np.radians(0.5)), 0.0, np.cos(np.radians(0.5))], dtype=np.float32
)


def _mount(lidar_in_rig: list[float], rotation_xyzw: np.ndarray = _IDENTITY) -> Pose:
    return Pose(np.array(lidar_in_rig, dtype=np.float32), rotation_xyzw)


def _extract(points_in_rig: np.ndarray, rig_to_lidar: Pose) -> list[dict]:
    """Run the extractor on rig-frame points seen by a LiDAR at ``rig_to_lidar``."""
    lidar_from_rig = np.linalg.inv(rig_to_lidar.as_se3())
    xyz = (points_in_rig @ lidar_from_rig[:3, :3].T + lidar_from_rig[:3, 3]).astype(
        np.float32
    )
    payload = pack_lidar_spin(xyz.tobytes(), len(xyz), 0, rig_to_lidar)
    extractor = LidarObstacleExtractor(
        min_confidence=0.1, min_range_m=2.0, max_range_m=80.0
    )
    return extractor.extract(unpack_lidar_spin(payload), ego_speed=10.0)


def _road(rig_to_lidar: Pose) -> np.ndarray:
    """Rig-frame road returns of a spinning LiDAR: one ring per downward beam,
    ray-cast onto the z = 0 road plane, with a few cm of surface roughness."""
    elevations = np.radians(np.linspace(-25.0, -1.0, 64))
    azimuths = np.linspace(-np.pi, np.pi, 1800, endpoint=False)
    elevation, azimuth = np.meshgrid(elevations, azimuths, indexing="ij")
    rays_in_lidar = np.stack(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        axis=-1,
    ).reshape(-1, 3)
    rig_from_lidar = rig_to_lidar.as_se3().astype(np.float64)
    rays_in_rig = rays_in_lidar @ rig_from_lidar[:3, :3].T
    origin_in_rig = rig_from_lidar[:3, 3]
    distance = -origin_in_rig[2] / rays_in_rig[:, 2]
    hits = (distance > 0) & (distance < 80.0)
    road = origin_in_rig + rays_in_rig[hits] * distance[hits, None]
    road[:, 2] += np.random.default_rng(0).normal(0.0, 0.02, len(road))
    return road


def _car(rng: np.random.Generator, rear_x: float) -> np.ndarray:
    n = 800
    return np.column_stack(
        [
            np.full(n, rear_x),  # rear face, as seen from behind
            rng.uniform(-0.9, 0.9, n),
            rng.uniform(0.3, 1.5, n),
        ]
    )


def test_road_returns_are_not_obstacles() -> None:
    for mount in (
        _mount([1.17, 0.0, 1.93]),
        _mount([0.5, 0.0, 2.37]),
        _mount([1.17, 0.0, 1.93], _PITCHED),
    ):
        assert _extract(_road(mount), mount) == []


def test_obstacles_are_reported_in_rig_frame() -> None:
    mount = _mount([1.17, 0.0, 1.93], _PITCHED)
    points_in_rig = np.vstack([_road(mount), _car(np.random.default_rng(0), 15.0)])

    (obstacle,) = _extract(points_in_rig, mount)

    assert abs(obstacle["x"] - 15.0) < 0.3
    assert abs(obstacle["y"]) < 0.3
