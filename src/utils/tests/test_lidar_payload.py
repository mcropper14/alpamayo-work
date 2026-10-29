# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_utils.geometry import Pose
from alpasim_utils.lidar_payload import pack_lidar_spin, unpack_lidar_spin

# Roof LiDAR yawed 90 deg left, 1.2 m ahead of the rear axle and 1.9 m up.
_MOUNT = Pose(
    np.array([1.2, 0.0, 1.9], dtype=np.float32),
    np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32),
)


def _pack(xyz: np.ndarray, timestamp_us: int = 0) -> bytes:
    return pack_lidar_spin(xyz.tobytes(), len(xyz), timestamp_us, _MOUNT)


def test_round_trip() -> None:
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)

    spin = unpack_lidar_spin(_pack(xyz, timestamp_us=123))

    assert spin.timestamp_us == 123
    np.testing.assert_array_equal(spin.xyz, xyz)
    np.testing.assert_allclose(spin.rig_to_sensor.vec3, _MOUNT.vec3)
    np.testing.assert_allclose(spin.rig_to_sensor.quat, _MOUNT.quat)


def test_sensor_to_rig_applies_mount() -> None:
    spin = unpack_lidar_spin(_pack(np.zeros((0, 3), dtype=np.float32)))

    # 10 m along the sensor's x axis is 10 m to the left of the yawed sensor;
    # the road directly below the sensor is at rig height 0.
    points_in_sensor = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, -1.9]])
    np.testing.assert_allclose(
        spin.sensor_to_rig(points_in_sensor),
        [[1.2, 10.0, 1.9], [1.2, 0.0, 0.0]],
        atol=1e-5,
    )


def test_rejects_truncated_payload() -> None:
    payload = _pack(np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError):
        unpack_lidar_spin(payload[:-1])


def test_rejects_foreign_payload() -> None:
    with pytest.raises(ValueError):
        unpack_lidar_spin(b"not lidar data")
