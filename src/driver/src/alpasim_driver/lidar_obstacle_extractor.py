# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Turn a forwarded LiDAR spin into obstacles for the safety validator.

Runs the ``physical_ai_av_lidar`` pipeline (preprocess -> DBSCAN detection ->
tracking) on the points expressed around the sensor but levelled to the rig
orientation, then moves the detections into the rig frame (x forward, y left),
the same frame as the model's candidate waypoints.

Output obstacle format (consumed by ``SafetyValidator``)::

    {"id", "x", "y", "vx", "vy", "confidence", "source": "lidar"}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from alpasim_utils.lidar_payload import LidarSpin
from physical_ai_av_lidar.detection import (
    DetectionConfig,
    LidarDetector,
    SimpleObjectTracker,
)
from physical_ai_av_lidar.loader import SpinFrame
from physical_ai_av_lidar.preprocessing import LidarPreprocessor, PreprocessConfig
from physical_ai_av_lidar.schema import LidarExtrinsics

logger = logging.getLogger(__name__)


class LidarObstacleExtractor:
    """Stateful per-session extractor; the tracker associates across spins."""

    def __init__(
        self, min_confidence: float, min_range_m: float, max_range_m: float
    ) -> None:
        self._min_confidence = min_confidence
        self._min_range_m = min_range_m
        self._max_range_m = max_range_m
        self._detector = LidarDetector(config=DetectionConfig())
        self._tracker = SimpleObjectTracker(max_association_dist=3.0)

    def extract(
        self,
        spin: LidarSpin,
        ego_speed: float,
    ) -> list[dict[str, Any]]:
        """Detect obstacles in ``spin`` and return them in the rig frame."""
        n = len(spin.xyz)
        if n == 0:
            return []

        # The pipeline's ground filter is a flat z cut around the sensor, so a
        # sensor tilted by even a degree would turn distant road into obstacles.
        # Rotate into the rig orientation but keep the sensor as the origin.
        sensor_in_rig = spin.rig_to_sensor.vec3.astype(np.float64)
        xyz_levelled = spin.sensor_to_rig(spin.xyz) - sensor_in_rig

        # SpinFrame columns: [x, y, z, intensity, ring, return_type]; the
        # payload has none of the last three (the detector classifies and
        # scores clusters by geometry only).
        points = np.zeros((n, 6), dtype=np.float32)
        points[:, :3] = xyz_levelled
        frame = SpinFrame(
            scan_index=0,
            timestamp_us=spin.timestamp_us,
            points=points,
            ego_pose=None,
        )
        detections = self._tracker.update(
            self._detector.detect(self._preprocessor(spin).process(frame))
        )

        centers_in_rig = (
            np.array([obj.bbox.center for obj in detections.objects]).reshape(-1, 3)
            + sensor_in_rig
        )
        obstacles = []
        for obj, center_in_rig in zip(detections.objects, centers_in_rig):
            if obj.confidence < self._min_confidence:
                continue
            if not self._min_range_m <= obj.range_m <= self._max_range_m:
                continue

            # relative_velocity > 0 means the object is approaching, so its
            # forward speed is the ego speed minus the closing speed.
            obs_fwd_speed = ego_speed - (obj.relative_velocity or 0.0)
            obstacles.append(
                {
                    "id": obj.object_id,
                    "x": float(center_in_rig[0]),
                    "y": float(center_in_rig[1]),
                    "vx": obs_fwd_speed,
                    "vy": 0.0,
                    "confidence": float(obj.confidence),
                    "source": "lidar",
                }
            )

        logger.debug("LiDAR: %d points -> %d obstacles", n, len(obstacles))
        return obstacles

    def _preprocessor(self, spin: LidarSpin) -> LidarPreprocessor:
        """Preprocess around the sensor, the origin the pipeline's ground and
        ego-shadow thresholds are written for.  The ground threshold assumes a
        sensor ``LidarExtrinsics().tz`` above the road; shift it so ground is cut
        at the same height above the road for this sensor's actual mount.
        """
        mount_height_m = float(spin.rig_to_sensor.vec3[2])
        defaults = PreprocessConfig()
        return LidarPreprocessor(
            config=PreprocessConfig(
                range_min=self._min_range_m,
                range_max=self._max_range_m,
                ground_z_threshold=defaults.ground_z_threshold
                + LidarExtrinsics().tz
                - mount_height_m,
                apply_motion_compensation=False,  # payload carries no ego pose
                apply_extrinsic_transform=False,  # detections are moved to rig below
                remove_ego_vehicle_shadow=True,
            )
        )
