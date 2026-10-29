# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Fuse a radar scan into the LiDAR obstacles for the safety validator.

Radar measures range rate directly, while the LiDAR tracker can only
difference positions between spins.  Each LiDAR obstacle with a radar
detection within the association gate takes its speed from the nearest such
detection; one detection may serve several obstacles, since the LiDAR detector
can split one vehicle into several clusters.  Detections no LiDAR obstacle
explains become radar-only obstacles with a fixed, lower confidence.

Everything is in the rig frame, ego at the origin moving along +x.  Like the
LiDAR obstacles, a radar obstacle is assumed to move along x.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from alpasim_utils.radar_payload import RadarScan

# Below this |line of sight x|, the range rate says little about x-speed.
_MIN_LOS_X = 0.5


def _forward_speeds(scan: RadarScan, ego_speed: float) -> np.ndarray:
    """Obstacle x-speed per detection, NaN where unobservable.

    range_rate = (v_obstacle - v_ego) . line_of_sight with both velocities
    along x, so v_obstacle = v_ego + range_rate / los_x.
    """
    los_x = scan.line_of_sight[:, 0].astype(np.float64)
    observable = np.abs(los_x) >= _MIN_LOS_X
    speeds = np.full(len(los_x), np.nan)
    speeds[observable] = ego_speed + scan.range_rate[observable] / los_x[observable]
    return speeds


def fuse_radar(
    lidar_obstacles: list[dict[str, Any]],
    scan: RadarScan,
    ego_speed: float,
    association_gate_m: float,
    radar_only_confidence: float,
) -> list[dict[str, Any]]:
    """Return the LiDAR obstacles with radar speeds, plus radar-only obstacles."""
    if len(scan.xy) == 0:
        return lidar_obstacles
    speeds = _forward_speeds(scan, ego_speed)
    explained = np.zeros(len(scan.xy), dtype=bool)

    fused = []
    for obstacle in lidar_obstacles:
        distances = np.linalg.norm(scan.xy - [obstacle["x"], obstacle["y"]], axis=1)
        explained |= distances <= association_gate_m
        nearest = int(np.argmin(distances))
        if distances[nearest] <= association_gate_m and not np.isnan(speeds[nearest]):
            obstacle = {
                **obstacle,
                "vx": float(speeds[nearest]),
                "source": "lidar+radar",
            }
        fused.append(obstacle)

    for i in np.flatnonzero(~explained):
        fused.append(
            {
                "id": f"radar_{i}",
                "x": float(scan.xy[i, 0]),
                "y": float(scan.xy[i, 1]),
                # Treat an obstacle whose speed is unobservable as stationary.
                "vx": 0.0 if np.isnan(speeds[i]) else float(speeds[i]),
                "vy": 0.0,
                "confidence": radar_only_confidence,
                "source": "radar",
            }
        )
    return fused
