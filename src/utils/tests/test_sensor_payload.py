# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest
from alpasim_utils.sensor_payload import pack_sensor_payloads, unpack_sensor_payloads


def test_round_trip_keys_payloads_by_magic() -> None:
    lidar, radar = b"LIDAR002" + b"spin", b"RADAR001" + b"scan"

    assert unpack_sensor_payloads(pack_sensor_payloads([lidar, radar])) == {
        b"LIDAR002": lidar,
        b"RADAR001": radar,
    }


def test_rejects_duplicate_payload_kinds() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        pack_sensor_payloads([b"RADAR001a", b"RADAR001b"])


def test_rejects_truncated_container() -> None:
    data = pack_sensor_payloads([b"RADAR001" + b"scan"])

    with pytest.raises(ValueError, match="truncated"):
        unpack_sensor_payloads(data[:-1])


def test_rejects_bare_payload() -> None:
    with pytest.raises(ValueError):
        unpack_sensor_payloads(b"LIDAR002" + b"\0" * 32)
