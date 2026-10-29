# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Container for the sensor payloads the runtime forwards to the driver.

``DriveRequest.renderer_data`` holds one container; each section is a
self-describing payload whose first 8 bytes are its magic (e.g. ``LIDAR002``
from :mod:`alpasim_utils.lidar_payload`, ``RADAR001`` from
:mod:`alpasim_utils.radar_payload`).  Layout::

    8 bytes  magic b"SENSORS1"
    4 bytes  n_sections (uint32, little-endian)
    per section:
      4 bytes  length (uint32, little-endian)
      length   payload
"""

from __future__ import annotations

import struct

MAGIC = b"SENSORS1"
_COUNT = struct.Struct("<I")
_MAGIC_SIZE = 8


def pack_sensor_payloads(payloads: list[bytes]) -> bytes:
    """Bundle payloads with distinct magics into one container."""
    magics = [payload[:_MAGIC_SIZE] for payload in payloads]
    if len(set(magics)) != len(magics):
        raise ValueError(f"duplicate sensor payloads: {magics}")
    parts = [MAGIC, _COUNT.pack(len(payloads))]
    for payload in payloads:
        parts += [_COUNT.pack(len(payload)), payload]
    return b"".join(parts)


def unpack_sensor_payloads(data: bytes) -> dict[bytes, bytes]:
    """Return the container's payloads keyed by their magic."""
    if data[:_MAGIC_SIZE] != MAGIC:
        raise ValueError(f"renderer_data is not a {MAGIC.decode()} container")
    (count,) = _COUNT.unpack_from(data, _MAGIC_SIZE)
    offset = _MAGIC_SIZE + _COUNT.size
    payloads: dict[bytes, bytes] = {}
    for _ in range(count):
        (length,) = _COUNT.unpack_from(data, offset)
        offset += _COUNT.size
        payload = data[offset : offset + length]
        if len(payload) != length:
            raise ValueError(f"{MAGIC.decode()} container is truncated")
        payloads[payload[:_MAGIC_SIZE]] = payload
        offset += length
    if offset != len(data):
        raise ValueError(f"{MAGIC.decode()} container has trailing bytes")
    return payloads
