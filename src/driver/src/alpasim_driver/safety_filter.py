# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""LiDAR/radar safety filter over the model's sampled trajectory candidates.

Obstacles come from whichever sensors the runtime forwards: LiDAR detections,
radar detections, or LiDAR detections with radar speeds (``radar_fusion``).

Candidates are checked in model-preference order (the model's own selection
first).  The first one the ``SafetyValidator`` accepts is driven as-is.  If
none pass, the candidate with the highest validated safe speed is driven with
its waypoints scaled down to that speed; its direction is preserved.

Everything runs in the rig frame of the model's t0: ego at the origin with
heading 0, obstacles and candidate waypoints in that same frame.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from typing import Any

import numpy as np
from alpasim_utils import lidar_payload, radar_payload
from alpasim_utils.sensor_payload import unpack_sensor_payloads

from .lidar_obstacle_extractor import LidarObstacleExtractor
from .models.base import ModelPrediction
from .radar_fusion import fuse_radar
from .safety_validator import SafetyValidator
from .schema import SafetyValidatorConfig

logger = logging.getLogger(__name__)


def _action_from_rig_xy(xy: np.ndarray, current_speed: float) -> dict[str, float]:
    """Steering is the heading change between the first two waypoint segments."""
    heading0 = math.atan2(float(xy[0, 1]), float(xy[0, 0]))
    steering_angle = 0.0
    if len(xy) >= 2:
        dx, dy = xy[1] - xy[0]
        steering_angle = math.atan2(float(dy), float(dx)) - heading0
    return {"target_speed": max(current_speed, 0.0), "steering_angle": steering_angle}


def _scale_to_speed(
    positions: np.ndarray, target_speed: float, output_frequency_hz: int
) -> np.ndarray:
    """Shrink (T, 3) rig-frame waypoints so their implied speed is target_speed."""
    if target_speed <= 0.0:
        return np.zeros_like(positions)
    xy = np.vstack([[0.0, 0.0], positions[:, :2]])
    implied_speed = (
        float(np.mean(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        * output_frequency_hz
    )
    if implied_speed < 0.05:
        return positions
    return positions * min(target_speed / implied_speed, 1.0)


class SafetyFilter:
    """Per-session state: the LiDAR tracker and the validator's hold-stop latch."""

    def __init__(self, cfg: SafetyValidatorConfig, output_frequency_hz: int) -> None:
        self._cfg = cfg
        self._output_frequency_hz = output_frequency_hz
        self._extractor = LidarObstacleExtractor(
            min_confidence=cfg.min_confidence,
            min_range_m=cfg.lidar_min_range_m,
            max_range_m=cfg.lidar_max_range_m,
        )
        self._validator = SafetyValidator(
            lane_half_width=cfg.lane_half_width,
            prediction_horizon=cfg.prediction_horizon,
            reaction_time=cfg.reaction_time,
            max_decel=cfg.max_decel,
            min_ttc=cfg.min_ttc,
            emergency_ttc=cfg.emergency_ttc,
            safety_buffer=cfg.safety_buffer,
            min_confidence=cfg.min_confidence,
            steering_path_gain=cfg.steering_path_gain,
            confidence_weight_power=cfg.confidence_weight_power,
            rear_min_ttc=cfg.rear_min_ttc,
            rear_emergency_ttc=cfg.rear_emergency_ttc,
            rear_prediction_horizon=cfg.rear_prediction_horizon,
        )

    def apply(
        self, prediction: ModelPrediction, renderer_data: bytes, speed: float
    ) -> tuple[ModelPrediction, dict[str, Any]]:
        """Return the prediction to drive and a debug summary of the decision."""
        if not renderer_data:
            # The runtime forwards sensor data once per camera render; drive
            # calls that outnumber renders (or force-GT cache replays) carry none.
            logger.warning("Safety validator enabled but no sensor data received")
            return prediction, {"reason": "no_sensor_data"}

        obstacles, sensor_debug = self._obstacles(renderer_data, speed)
        ego_state = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": speed}

        k = len(prediction.candidate_positions)
        order = [prediction.selected_index] + [
            i for i in range(k) if i != prediction.selected_index
        ]
        # The validator carries a hold-stop latch across steps.  Every candidate
        # is judged from the same pre-step latch state, and only the driven
        # candidate's resulting state is kept.
        results = {}
        validators = {}
        for i in order:
            action = _action_from_rig_xy(
                prediction.candidate_positions[i, :, :2], speed
            )
            validators[i] = copy.copy(self._validator)
            result = validators[i].validate(ego_state, action, obstacles)
            results[i] = result
            if result["is_safe"]:
                self._validator = validators[i]
                return dataclasses.replace(prediction, selected_index=i), {
                    "reason": "safe",
                    "model_index": prediction.selected_index,
                    "selected_index": i,
                    "num_obstacles": len(obstacles),
                    **sensor_debug,
                    "validator_result": result,
                }

        best_i = max(
            order, key=lambda i: results[i]["validated_action"]["target_speed"]
        )
        self._validator = validators[best_i]
        safe_speed = results[best_i]["validated_action"]["target_speed"]
        positions = prediction.candidate_positions.copy()
        positions[best_i] = _scale_to_speed(
            positions[best_i], safe_speed, self._output_frequency_hz
        )
        logger.info(
            "No safe candidate among %d; driving %d at %.2f m/s (violations=%s)",
            k,
            best_i,
            safe_speed,
            results[best_i]["violations"],
        )
        return dataclasses.replace(
            prediction, candidate_positions=positions, selected_index=best_i
        ), {
            "reason": "speed_override",
            "model_index": prediction.selected_index,
            "selected_index": best_i,
            "safe_speed": safe_speed,
            "num_obstacles": len(obstacles),
            **sensor_debug,
            "validator_result": results[best_i],
        }

    def _obstacles(
        self, renderer_data: bytes, speed: float
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Rig-frame obstacles from the forwarded sensors, plus debug counts."""
        payloads = unpack_sensor_payloads(renderer_data)
        obstacles: list[dict[str, Any]] = []
        debug: dict[str, Any] = {}
        if lidar_payload.MAGIC in payloads:
            spin = lidar_payload.unpack_lidar_spin(payloads[lidar_payload.MAGIC])
            obstacles = self._extractor.extract(spin, speed)
            debug["num_lidar_obstacles"] = len(obstacles)
        if radar_payload.MAGIC in payloads:
            scan = radar_payload.unpack_radar_scan(payloads[radar_payload.MAGIC])
            obstacles = fuse_radar(
                obstacles,
                scan,
                speed,
                association_gate_m=self._cfg.radar_association_gate_m,
                radar_only_confidence=self._cfg.radar_only_confidence,
            )
            debug["num_radar_detections"] = len(scan.xy)
        return obstacles, debug
