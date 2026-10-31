# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
from alpasim_runtime.sensor_weather import (
    WEATHER_EFFECTS,
    degrade_lidar_payload,
    degrade_radar,
)
from alpasim_utils.geometry import Pose
from alpasim_utils.lidar_payload import pack_lidar_spin, unpack_lidar_spin
from alpasim_utils.radar_payload import RadarScan

_MOUNT = Pose(
    np.array([1.2, 0.0, 1.9], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
)


def _spin(ranges_m: np.ndarray) -> bytes:
    """Points straight ahead of the sensor at the given ranges."""
    xyz = np.column_stack([ranges_m, np.zeros_like(ranges_m), np.zeros_like(ranges_m)])
    xyz = xyz.astype(np.float32)
    return pack_lidar_spin(xyz.tobytes(), len(xyz), 7, _MOUNT)


def _degraded(weather: str, payload: bytes, seed: int = 0):
    return unpack_lidar_spin(
        degrade_lidar_payload(
            payload, WEATHER_EFFECTS[weather], np.random.default_rng(seed)
        )
    )


def test_clear_weather_leaves_spin_unchanged() -> None:
    payload = _spin(np.linspace(3.0, 80.0, 500))

    assert _degraded("clear", payload).xyz.tolist() == (
        unpack_lidar_spin(payload).xyz.tolist()
    )


def test_fog_removes_distant_returns_and_adds_near_backscatter() -> None:
    far = _spin(np.full(10_000, 60.0))
    effects = WEATHER_EFFECTS["fog"]

    spin = _degraded("fog", far)

    ranges = np.linalg.norm(spin.xyz, axis=1)
    surviving_far = int((ranges > 50.0).sum())
    assert surviving_far < 0.01 * 10_000  # exp(-2 * 0.06 * 60) ~ 0.07 %
    _, far_edge = effects.lidar_clutter_range_m
    # Everything else is backscatter near the sensor (patches spread a little
    # beyond the band edge).
    backscatter = ranges[ranges <= 50.0]
    assert len(backscatter) > effects.lidar_noise_points
    assert backscatter.max() < far_edge + 2.0
    # Timestamp and mount survive the round trip.
    assert spin.timestamp_us == 7
    np.testing.assert_allclose(spin.rig_to_sensor.vec3, _MOUNT.vec3)


def test_rain_keeps_most_near_returns() -> None:
    near = _spin(np.full(10_000, 10.0))

    spin = _degraded("rain", near)

    at_ten_m = np.isclose(np.linalg.norm(spin.xyz, axis=1), 10.0, atol=1e-3)
    assert at_ten_m.sum() > 0.75 * 10_000  # exp(-2 * 0.01 * 10) ~ 82 %


def test_degradation_is_reproducible_from_seed() -> None:
    payload = _spin(np.linspace(3.0, 80.0, 500))

    first = _degraded("snow", payload, seed=3).xyz
    again = _degraded("snow", payload, seed=3).xyz

    np.testing.assert_array_equal(first, again)


def test_radar_loses_few_detections_in_fog() -> None:
    n = 10_000
    scan = RadarScan(
        timestamp_us=0,
        xy=np.tile([[50.0, 0.0]], (n, 1)).astype(np.float32),
        range_rate=np.zeros(n, dtype=np.float32),
        line_of_sight=np.tile([[1.0, 0.0]], (n, 1)).astype(np.float32),
    )

    degraded = degrade_radar(scan, WEATHER_EFFECTS["fog"], np.random.default_rng(0))

    assert 0.85 * n < len(degraded.xy) < n  # exp(-2 * 0.001 * 50) ~ 90 %
    assert len(degraded.range_rate) == len(degraded.line_of_sight) == len(degraded.xy)
