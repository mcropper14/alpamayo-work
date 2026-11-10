# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import numpy as np
import pytest
from alpasim_runtime.config import LeadBrakeConfig
from alpasim_runtime.lead_brake import apply_lead_brake
from alpasim_utils.geometry import Pose, Trajectory
from alpasim_utils.scenario import AABB, TrafficObject, TrafficObjects

_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_TIMES_US = np.arange(0, 10_000_001, 100_000)


def _driving(x0: float, y: float, speed: float) -> Trajectory:
    return Trajectory.from_poses(
        timestamps=_TIMES_US.astype(np.uint64),
        poses=[
            Pose(np.array([x0 + speed * t * 1e-6, y, 0.0], dtype=np.float32), _IDENTITY)
            for t in _TIMES_US
        ],
    )


def _vehicle(x0: float, y: float = 0.0, label: str = "automobile") -> TrafficObject:
    return TrafficObject(
        track_id="",
        aabb=AABB(x=4.5, y=1.9, z=1.5),
        trajectory=_driving(x0, y, 10.0),
        is_static=False,
        label_class=label,
    )


def _brake(traffic: dict[str, TrafficObject], cfg: LeadBrakeConfig = LeadBrakeConfig()):
    return apply_lead_brake(
        TrafficObjects(**traffic), _driving(0.0, 0.0, 10.0), 2_000_000, 10_000_000, cfg
    )


def test_nearest_vehicle_ahead_in_lane_brakes_to_a_stop_along_its_path() -> None:
    """Ego and traffic at 10 m/s; from 2 s the lead brakes at 6 m/s^2 for 10/6 s."""
    traffic, lead, brake_us = _brake(
        {"far": _vehicle(35.0), "lead": _vehicle(20.0), "next_lane": _vehicle(10.0, y=3.5)}
    )

    assert (lead, brake_us) == ("lead", 2_000_000)
    braked = traffic["lead"].trajectory
    stop_x = 20.0 + 10.0 * 2.0 + 10.0**2 / (2 * 6.0)
    assert braked.interpolate_pose(1_000_000).vec3[0] == pytest.approx(30.0, abs=1e-3)
    assert braked.interpolate_pose(4_000_000).vec3[0] == pytest.approx(stop_x, abs=0.05)
    assert braked.interpolate_pose(10_000_000).vec3[0] == pytest.approx(stop_x, abs=0.05)
    assert traffic["far"].trajectory is not braked


def test_no_lead_vehicle_leaves_the_traffic_unchanged() -> None:
    assert _brake({"pedestrian": _vehicle(20.0, label="pedestrian"), "beside": _vehicle(0.0, y=3.5)}) is None
