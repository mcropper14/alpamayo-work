# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_utils.radar_payload import RadarScan, pack_radar_scan, unpack_radar_scan


def _scan(n: int) -> RadarScan:
    return RadarScan(
        timestamp_us=42,
        xy=np.arange(2 * n, dtype=np.float32).reshape(n, 2),
        range_rate=np.linspace(-5, 5, n, dtype=np.float32),
        line_of_sight=np.tile(np.array([1.0, 0.0], dtype=np.float32), (n, 1)),
    )


@pytest.mark.parametrize("n", [0, 3])
def test_round_trip(n: int) -> None:
    scan = _scan(n)

    unpacked = unpack_radar_scan(pack_radar_scan(scan))

    assert unpacked.timestamp_us == 42
    np.testing.assert_array_equal(unpacked.xy, scan.xy)
    np.testing.assert_array_equal(unpacked.range_rate, scan.range_rate)
    np.testing.assert_array_equal(unpacked.line_of_sight, scan.line_of_sight)


def test_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError):
        unpack_radar_scan(pack_radar_scan(_scan(2))[:-1])
