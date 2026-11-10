# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Make the vehicle ahead of the ego brake hard to a stop.

A scenario the recordings rarely contain, after the car-to-car braking tests of
emergency-braking ratings: once the closed loop has run for a while, the
nearest vehicle ahead of the recorded ego, in its lane, brakes at a constant
deceleration along its own recorded path and stays stopped to the end.

Only the vehicle's recorded trajectory is replaced, before the rollout starts,
so the renderer, radar, physics and evaluation all see the braking vehicle.
The ego that reaches it is the policy's, not the recording's.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
from alpasim_runtime.config import LeadBrakeConfig
from alpasim_utils.geometry import Trajectory
from alpasim_utils.scenario import TrafficObjects

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = ("automobile", "heavy_truck", "bus", "trailer")
_SEARCH_STEP_US = 500_000
_TRACK_STEP_US = 100_000
# Velocities are differences over this interval of a trajectory.
_VELOCITY_DT_US = 100_000


def _speed(trajectory: Trajectory, time_us: int) -> float:
    start = max(trajectory.time_range_us.start, time_us - _VELOCITY_DT_US)
    end = min(trajectory.time_range_us.stop - 1, time_us + _VELOCITY_DT_US)
    if end <= start:
        return 0.0
    delta = trajectory.interpolate_pose(end).vec3 - trajectory.interpolate_pose(start).vec3
    return float(np.linalg.norm(delta[:2]) / ((end - start) * 1e-6))


def find_lead(
    traffic: TrafficObjects, ego: Trajectory, time_us: int, cfg: LeadBrakeConfig
) -> str | None:
    """The nearest moving vehicle ahead of ``ego`` in its lane at ``time_us``."""
    ego_from_local = np.linalg.inv(ego.interpolate_pose(time_us).as_se3())
    lead, lead_distance = None, cfg.max_distance_m
    for track_id, obj in traffic.items():
        if obj.is_static or obj.label_class not in VEHICLE_CLASSES:
            continue
        if time_us not in obj.trajectory.time_range_us:
            continue
        x, y = (ego_from_local @ obj.trajectory.interpolate_pose(time_us).as_se3())[:2, 3]
        if 0.0 < x <= lead_distance and abs(y) <= cfg.lane_half_width_m:
            lead, lead_distance = track_id, x
    return lead


def braking_trajectory(
    trajectory: Trajectory, start_us: int, end_us: int, decel: float
) -> Trajectory:
    """``trajectory`` up to ``start_us``, then braking at ``decel`` along its own
    path to a stop, held until ``end_us``."""
    start_speed = _speed(trajectory, start_us)
    # The recorded path from start_us on, by distance travelled along it.
    path_times = np.arange(start_us, trajectory.time_range_us.stop, _TRACK_STEP_US)
    path_xy = np.array([trajectory.interpolate_pose(int(t)).vec3[:2] for t in path_times])
    path_length = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))]
    )

    times = np.arange(start_us, max(end_us, start_us) + 1, _TRACK_STEP_US)
    elapsed = np.minimum((times - start_us) * 1e-6, start_speed / decel)
    travelled = start_speed * elapsed - 0.5 * decel * elapsed**2
    # Where the recorded vehicle had travelled as far; past the end of its
    # track, it stops where the track ends.
    recorded_times = np.interp(travelled, path_length, path_times)
    braking = [trajectory.interpolate_pose(int(t)) for t in recorded_times]

    before = trajectory.timestamps_us < start_us
    poses = [trajectory.get_pose(i) for i in np.flatnonzero(before)] + braking
    timestamps = np.concatenate([trajectory.timestamps_us[before], times]).astype(np.uint64)
    return Trajectory.from_poses(timestamps=timestamps, poses=poses)


def apply_lead_brake(
    traffic: TrafficObjects,
    ego: Trajectory,
    earliest_us: int,
    end_us: int,
    cfg: LeadBrakeConfig,
) -> tuple[TrafficObjects, str, int] | None:
    """``traffic`` with the lead vehicle braking, the lead's ID and the brake time.

    The brake starts at the first time from ``earliest_us`` on, every
    _SEARCH_STEP_US, at which the recorded ego drives at least
    ``cfg.min_ego_speed_mps`` behind a lead vehicle.  None if there is none.
    """
    for time_us in range(earliest_us, end_us, _SEARCH_STEP_US):
        if time_us not in ego.time_range_us or _speed(ego, time_us) < cfg.min_ego_speed_mps:
            continue
        lead = find_lead(traffic, ego, time_us, cfg)
        if lead is None:
            continue
        obj = traffic[lead]
        braked = replace(
            obj, trajectory=braking_trajectory(obj.trajectory, time_us, end_us, cfg.decel_mps2)
        )
        return TrafficObjects(**{**traffic, lead: braked}), lead, time_us
    return None
