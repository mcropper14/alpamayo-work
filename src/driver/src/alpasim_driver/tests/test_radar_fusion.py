# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_driver.radar_fusion import fuse_radar
from alpasim_utils.radar_payload import RadarScan

_EGO_SPEED = 10.0


def _scan(*detections: tuple[float, float, float]) -> RadarScan:
    """Detections as (x, y, range_rate), seen from a radar at the rig origin."""
    xy = np.array([d[:2] for d in detections], dtype=np.float32).reshape(-1, 2)
    norms = np.linalg.norm(xy, axis=1, keepdims=True)
    return RadarScan(
        timestamp_us=0,
        xy=xy,
        range_rate=np.array([d[2] for d in detections], dtype=np.float32),
        line_of_sight=(xy / np.where(norms > 0, norms, 1.0)).astype(np.float32),
    )


def _lidar(x: float, y: float, vx: float) -> dict:
    return {
        "id": 1,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": 0.0,
        "confidence": 0.8,
        "source": "lidar",
    }


def _fuse(obstacles: list[dict], scan: RadarScan) -> list[dict]:
    return fuse_radar(
        obstacles,
        scan,
        _EGO_SPEED,
        association_gate_m=2.0,
        radar_only_confidence=0.5,
    )


def test_radar_range_rate_sets_speed_of_matched_lidar_obstacles() -> None:
    # The LiDAR detector split one stopped car into two clusters; the tracker's
    # speed guess is off.  Range rate -10 m/s at ego 10 m/s means stopped.
    obstacles = [_lidar(20.0, 0.0, vx=4.0), _lidar(21.5, 0.0, vx=3.0)]

    fused = _fuse(obstacles, _scan((20.2, 0.0, -10.0)))

    assert [o["vx"] for o in fused] == [pytest.approx(0.0, abs=1e-5)] * 2
    assert {o["source"] for o in fused} == {"lidar+radar"}


def test_unmatched_radar_detection_becomes_radar_only_obstacle() -> None:
    fused = _fuse([_lidar(20.0, 0.0, vx=0.0)], _scan((60.0, 0.0, -4.0)))

    (radar_only,) = [o for o in fused if o["source"] == "radar"]
    assert radar_only["x"] == pytest.approx(60.0)
    assert radar_only["vx"] == pytest.approx(6.0, abs=1e-5)
    assert radar_only["confidence"] == 0.5


def test_crossing_detection_speed_is_unobservable() -> None:
    """Straight to the side, range rate says nothing about x-speed: the LiDAR
    speed is kept, and a radar-only obstacle is treated as stationary."""
    lidar = _lidar(0.0, 5.0, vx=7.0)

    fused = _fuse([lidar], _scan((0.0, 5.0, 0.0), (0.0, -9.0, 0.0)))

    assert fused[0] == lidar
    assert fused[1]["source"] == "radar"
    assert fused[1]["vx"] == 0.0


def test_empty_scan_keeps_lidar_obstacles() -> None:
    obstacles = [_lidar(20.0, 0.0, vx=0.0)]

    assert _fuse(obstacles, _scan()) is obstacles
