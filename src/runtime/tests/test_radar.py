# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import numpy as np
import pytest
from alpasim_runtime.config import PoseConfig, RadarConfig
from alpasim_runtime.radar import simulate_radar_scan
from alpasim_utils.geometry import Pose, Trajectory
from alpasim_utils.scenario import AABB, TrafficObject, TrafficObjects

_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_TIMES_US = [0, 100_000, 200_000, 300_000]
_NOW_US = 200_000
_EGO_SPEED = 10.0


def _moving(x0: float, y: float, speed: float) -> Trajectory:
    return Trajectory.from_poses(
        timestamps=np.array(_TIMES_US, dtype=np.uint64),
        poses=[
            Pose(np.array([x0 + speed * t * 1e-6, y, 0.0], dtype=np.float32), _IDENTITY)
            for t in _TIMES_US
        ],
    )


def _car(track_id: str, x0: float, y: float = 0.0, speed: float = 0.0) -> TrafficObject:
    return TrafficObject(
        track_id=track_id,
        aabb=AABB(x=4.0, y=2.0, z=1.5),
        trajectory=_moving(x0, y, speed),
        is_static=False,
        label_class="automobile",
    )


def _radar(**overrides) -> RadarConfig:
    cfg = RadarConfig(
        name="front",
        rig_to_radar=PoseConfig(
            translation_m=(4.0, 0.0, 0.5), rotation_xyzw=(0, 0, 0, 1)
        ),
        max_range_m=200.0,
        half_fov_deg=10.0,
        range_noise_m=0.0,
        azimuth_noise_deg=0.0,
        range_rate_noise_mps=0.0,
        detection_probability=1.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _scan(objs: list[TrafficObject], **radar_overrides):
    # Ego at x = 2 m at _NOW_US, driving +x at _EGO_SPEED.
    return simulate_radar_scan(
        [_radar(**radar_overrides)],
        _moving(0.0, 0.0, _EGO_SPEED),
        TrafficObjects({obj.track_id: obj for obj in objs}),
        _NOW_US,
        np.random.default_rng(0),
    )


def test_detects_rear_of_stopped_car_with_closing_range_rate() -> None:
    # Car centre at local x = 24 m, ego at x = 2 m: the centre is 22 m ahead of
    # the rig, so its rear face is at rig x = 20 m.
    scan = _scan([_car("lead", x0=24.0)])

    assert scan.xy.tolist() == [pytest.approx([20.0, 0.0], abs=1e-4)]
    assert scan.range_rate.tolist() == [pytest.approx(-_EGO_SPEED, abs=1e-3)]
    np.testing.assert_allclose(scan.line_of_sight, [[1.0, 0.0]], atol=1e-6)


def test_range_rate_of_car_moving_at_ego_speed_is_zero() -> None:
    scan = _scan([_car("lead", x0=24.0, speed=_EGO_SPEED)])

    assert scan.range_rate.tolist() == [pytest.approx(0.0, abs=1e-3)]


def test_ignores_cars_outside_field_of_view_and_range() -> None:
    beside = _car("beside", x0=12.0, y=8.0)  # ~40 deg off boresight
    far = _car("far", x0=300.0)

    assert len(_scan([beside, far]).xy) == 0


def test_car_hidden_behind_nearer_car_is_not_detected() -> None:
    scan = _scan([_car("near", x0=24.0), _car("hidden", x0=44.0)])

    assert scan.xy[:, 0].tolist() == [pytest.approx(20.0, abs=1e-4)]


def test_static_car_is_detected_outside_its_track_window() -> None:
    parked = TrafficObject(
        track_id="parked",
        aabb=AABB(x=4.0, y=2.0, z=1.5),
        trajectory=Trajectory.from_poses(
            timestamps=np.array([0, 50_000], dtype=np.uint64),
            poses=[Pose(np.array([24.0, 0.0, 0.0], dtype=np.float32), _IDENTITY)] * 2,
        ),
        is_static=True,
        label_class="automobile",
    )
    scan = _scan([parked])

    assert scan.xy.tolist() == [pytest.approx([20.0, 0.0], abs=1e-4)]
    assert scan.range_rate.tolist() == [pytest.approx(-_EGO_SPEED, abs=1e-3)]


def test_moving_car_outside_or_before_its_track_is_not_detected() -> None:
    lead = _car("lead", x0=24.0)
    gone = TrafficObject(
        track_id="gone",
        aabb=lead.aabb,
        trajectory=lead.trajectory.clip(0, 100_000),
        is_static=False,
        label_class="automobile",
    )
    not_yet_simulated = TrafficObject(
        track_id="not_yet",
        aabb=lead.aabb,
        trajectory=Trajectory.create_empty(),
        is_static=False,
        label_class="automobile",
    )

    assert len(_scan([gone, not_yet_simulated]).xy) == 0


def test_detection_probability_drops_detections() -> None:
    assert len(_scan([_car("lead", x0=24.0)], detection_probability=0.0).xy) == 0


def test_false_alarms_look_like_static_clutter() -> None:
    scan = _scan([], false_alarms_per_scan=20.0)

    assert len(scan.xy) > 0
    # A static object closes at the ego speed projected on the line of sight.
    np.testing.assert_allclose(
        scan.range_rate, -_EGO_SPEED * scan.line_of_sight[:, 0], atol=1e-3
    )
