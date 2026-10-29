# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Binary radar scan payload forwarded from the runtime to the driver.

One scan holds the detections of every simulated radar at one timestamp,
already moved into the rig frame so the driver needs no radar mounts.
Layout::

    8 bytes    magic b"RADAR001"
    8 bytes    timestamp_us (uint64, little-endian)
    4 bytes    n_detections (uint32, little-endian)
    N*20 bytes detections, float32 per detection:
               x, y            detection position in the rig frame (m)
               range_rate      d(range)/dt seen by the radar (m/s, < 0 closing)
               los_x, los_y    unit line of sight radar -> detection, rig frame
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"RADAR001"
_HEADER = struct.Struct("<QI")
HEADER_SIZE = len(MAGIC) + _HEADER.size
_FIELDS = 5


@dataclass(frozen=True)
class RadarScan:
    timestamp_us: int
    xy: np.ndarray  # (N, 2) float32, detection positions in the rig frame
    range_rate: np.ndarray  # (N,) float32, m/s, negative when closing
    line_of_sight: np.ndarray  # (N, 2) float32, unit vectors in the rig frame


def pack_radar_scan(scan: RadarScan) -> bytes:
    n = len(scan.xy)
    detections = np.column_stack([scan.xy, scan.range_rate, scan.line_of_sight]).astype(
        "<f4"
    )
    if detections.shape != (n, _FIELDS):
        raise ValueError(f"radar detections have shape {detections.shape}")
    return MAGIC + _HEADER.pack(scan.timestamp_us, n) + detections.tobytes()


def unpack_radar_scan(data: bytes) -> RadarScan:
    """Inverse of :func:`pack_radar_scan`."""
    if data[: len(MAGIC)] != MAGIC:
        raise ValueError(f"payload is not a {MAGIC.decode()} payload")
    timestamp_us, n = _HEADER.unpack_from(data, len(MAGIC))
    expected_size = HEADER_SIZE + n * _FIELDS * 4
    if len(data) != expected_size:
        raise ValueError(
            f"{MAGIC.decode()} payload has {len(data)} bytes, expected {expected_size}"
        )
    detections = np.frombuffer(
        data, dtype="<f4", count=n * _FIELDS, offset=HEADER_SIZE
    ).reshape(n, _FIELDS)
    return RadarScan(
        timestamp_us=int(timestamp_us),
        xy=detections[:, 0:2].copy(),
        range_rate=detections[:, 2].copy(),
        line_of_sight=detections[:, 3:5].copy(),
    )
