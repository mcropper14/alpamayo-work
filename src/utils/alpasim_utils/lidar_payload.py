# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Binary LiDAR spin payload forwarded from the runtime to the driver.

The runtime packs a sensorsim ``LidarRenderReturn`` into
``DriveRequest.renderer_data``; the driver unpacks it.  Points are in the
LiDAR sensor frame (sensorsim returns them in "end-of-spin lidar space"); the
payload carries the sensor's mounting pose so the driver can move results into
the rig frame.  sensorsim returns no point intensities, so none are carried.
Layout::

    8 bytes    magic b"LIDAR002"
    8 bytes    timestamp_us (uint64, little-endian)
    4 bytes    n_points     (uint32, little-endian)
    12 bytes   rig_to_sensor translation (float32 x, y, z; m, rig frame)
    16 bytes   rig_to_sensor rotation    (float32 quaternion x, y, z, w)
    N*12 bytes xyz          (float32 x, y, z per point, sensor frame)

``rig_to_sensor`` is the sensor's pose in the rig frame (the same convention
as a camera's ``rig_to_camera``): ``p_rig = R @ p_sensor + t``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from alpasim_utils.geometry import Pose

MAGIC = b"LIDAR002"
_HEADER = struct.Struct("<QI3f4f")
HEADER_SIZE = len(MAGIC) + _HEADER.size


@dataclass(frozen=True)
class LidarSpin:
    timestamp_us: int
    xyz: np.ndarray  # (N, 3) float32, sensor frame: x forward, y left, z up
    rig_to_sensor: Pose  # sensor pose in the rig frame

    def sensor_to_rig(self, points_in_sensor: np.ndarray) -> np.ndarray:
        """Transform (..., 3) sensor-frame points into the rig frame."""
        se3 = self.rig_to_sensor.as_se3().astype(np.float64)
        rotation, translation = se3[:3, :3], se3[:3, 3]
        # Column products instead of ``@``: a (N, 3) x (3, 3) matmul goes to a
        # multi-threaded BLAS, which on shared many-core nodes took seconds.
        return (
            points_in_sensor[..., 0:1] * rotation[:, 0]
            + points_in_sensor[..., 1:2] * rotation[:, 1]
            + points_in_sensor[..., 2:3] * rotation[:, 2]
            + translation
        )


def pack_lidar_spin(
    xyz_buffer: bytes,
    n_points: int,
    timestamp_us: int,
    rig_to_sensor: Pose,
) -> bytes:
    """Pack the raw float32 buffers of a ``LidarRenderReturn``."""
    if len(xyz_buffer) != n_points * 12:
        raise ValueError(
            f"xyz buffer has {len(xyz_buffer)} bytes, expected {n_points * 12}"
        )
    header = _HEADER.pack(
        timestamp_us,
        n_points,
        *rig_to_sensor.vec3.tolist(),
        *rig_to_sensor.quat.tolist(),
    )
    return MAGIC + header + xyz_buffer


def unpack_lidar_spin(data: bytes) -> LidarSpin:
    """Inverse of :func:`pack_lidar_spin`."""
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError(f"renderer_data is not a {MAGIC.decode()} payload")
    if len(data) < HEADER_SIZE:
        raise ValueError(f"{MAGIC.decode()} payload is shorter than its header")
    timestamp_us, n_points, *mount = _HEADER.unpack_from(data, len(MAGIC))
    expected_size = HEADER_SIZE + n_points * 12
    if len(data) != expected_size:
        raise ValueError(
            f"{MAGIC.decode()} payload has {len(data)} bytes, expected {expected_size}"
        )
    xyz = np.frombuffer(data, dtype="<f4", count=n_points * 3, offset=HEADER_SIZE)
    return LidarSpin(
        timestamp_us=int(timestamp_us),
        xyz=xyz.reshape(n_points, 3).copy(),
        rig_to_sensor=Pose(
            np.array(mount[:3], dtype=np.float32), np.array(mount[3:], dtype=np.float32)
        ),
    )
