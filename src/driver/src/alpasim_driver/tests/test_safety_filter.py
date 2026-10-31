# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_driver.models.base import ModelPrediction
from alpasim_driver.safety_filter import SafetyFilter
from alpasim_driver.scene_labels import SensorMode
from alpasim_driver.schema import SafetyValidatorConfig
from alpasim_utils.geometry import Pose
from alpasim_utils.lidar_payload import pack_lidar_spin
from alpasim_utils.radar_payload import RadarScan, pack_radar_scan
from alpasim_utils.sensor_payload import pack_sensor_payloads, unpack_sensor_payloads

_HZ = 10
_T = 40


# Roof LiDAR, as calibrated on the Hyperion rigs.
_LIDAR_IN_RIG = np.array([1.17, 0.0, 1.93])
_RIG_TO_LIDAR = Pose(
    _LIDAR_IN_RIG.astype(np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
)


def _payload(points_in_rig: np.ndarray, timestamp_us: int = 0) -> bytes:
    """Pack rig-frame points the way sensorsim returns them: in the sensor frame."""
    xyz = (points_in_rig - _LIDAR_IN_RIG).astype(np.float32)
    return pack_sensor_payloads(
        [
            pack_lidar_spin(
                xyz.tobytes(),
                len(xyz),
                timestamp_us,
                _RIG_TO_LIDAR,
            )
        ]
    )


def _car_ahead(distance_m: float, timestamp_us: int = 0) -> bytes:
    """Dense box of points, car-sized, centred on the ego path, in the rig frame."""
    rng = np.random.default_rng(0)
    n = 800
    return _payload(
        np.column_stack(
            [
                rng.uniform(distance_m, distance_m + 4.0, n),
                rng.uniform(-0.9, 0.9, n),
                rng.uniform(0.3, 1.8, n),
            ]
        ),
        timestamp_us,
    )


def _prediction(paths: list[np.ndarray]) -> ModelPrediction:
    positions = np.stack(paths)
    rotations = np.broadcast_to(np.eye(3), (*positions.shape[:2], 3, 3)).copy()
    return ModelPrediction(candidate_positions=positions, candidate_rotations=rotations)


def _straight(speed: float) -> np.ndarray:
    t = np.arange(1, _T + 1) / _HZ
    return np.column_stack([speed * t, np.zeros(_T), np.zeros(_T)])


def _swerve_left(speed: float) -> np.ndarray:
    """Turns hard left (~73 deg) after the first waypoint, out of the corridor."""
    path = _straight(speed)
    path[1:, 0] = path[0, 0] + 0.3 * np.arange(1, _T)
    path[1:, 1] = np.arange(1, _T, dtype=float)
    return path


def _filter() -> SafetyFilter:
    return SafetyFilter(SafetyValidatorConfig(enabled=True), output_frequency_hz=_HZ)


def _step(
    safety_filter: SafetyFilter,
    prediction: ModelPrediction,
    renderer_data: bytes,
    speed: float,
    mode: SensorMode = SensorMode.FUSED,
) -> tuple[ModelPrediction, dict]:
    """One drive step: sense the forwarded sensors, then judge the candidates."""
    return safety_filter.apply(
        prediction, safety_filter.sense(renderer_data, speed, mode), speed
    )


def test_clear_road_keeps_model_choice() -> None:
    prediction = _prediction([_straight(10.0), _swerve_left(10.0)])

    result, debug = _step(_filter(), prediction, _payload(np.zeros((0, 3))), 10.0)

    assert debug["reason"] == "safe"
    assert result.selected_index == 0
    np.testing.assert_array_equal(result.selected_positions, _straight(10.0))


def test_obstacle_ahead_slows_the_only_candidate() -> None:
    prediction = _prediction([_straight(10.0)])

    result, debug = _step(_filter(), prediction, _car_ahead(8.0), 10.0)

    assert debug["reason"] == "speed_override"
    assert debug["num_obstacles"] >= 1
    driven = np.linalg.norm(result.selected_positions[-1, :2])
    assert driven < np.linalg.norm(_straight(10.0)[-1, :2])


def test_obstacle_ahead_prefers_candidate_that_avoids_it() -> None:
    prediction = _prediction([_straight(10.0), _swerve_left(10.0)])
    safety_filter = _filter()
    # The tracker needs two spins to see that the car is stationary: ego at
    # 10 m/s closes 1 m per 0.1 s.
    _step(safety_filter, prediction, _car_ahead(16.0, timestamp_us=0), 10.0)

    result, debug = _step(
        safety_filter, prediction, _car_ahead(15.0, timestamp_us=100_000), 10.0
    )

    assert debug["model_index"] == 0
    assert result.selected_index == 1


def test_missing_sensor_data_passes_prediction_through() -> None:
    prediction = _prediction([_straight(10.0)])

    result, debug = _step(_filter(), prediction, b"", 10.0)

    assert debug == {"reason": "no_sensor_data"}
    assert result is prediction


def test_radar_only_stopped_car_ahead_slows_down() -> None:
    """Without LiDAR, a radar detection of a stopped car in the path (closing
    at the ego speed) still makes the validator slow down."""
    radar = RadarScan(
        timestamp_us=0,
        xy=np.array([[8.0, 0.0]], dtype=np.float32),
        range_rate=np.array([-10.0], dtype=np.float32),
        line_of_sight=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    payload = pack_sensor_payloads([pack_radar_scan(radar)])

    _, debug = _step(_filter(), _prediction([_straight(10.0)]), payload, 10.0)

    assert debug["reason"] == "speed_override"
    assert debug["num_radar_detections"] == 1
    assert "num_lidar_obstacles" not in debug


_NO_RADAR_DETECTIONS = RadarScan(
    timestamp_us=0,
    xy=np.zeros((0, 2), dtype=np.float32),
    range_rate=np.zeros(0, dtype=np.float32),
    line_of_sight=np.zeros((0, 2), dtype=np.float32),
)


def _with_empty_radar(lidar_payload: bytes) -> bytes:
    """Add a radar scan without detections to a LiDAR-only payload."""
    return pack_sensor_payloads(
        [
            *unpack_sensor_payloads(lidar_payload).values(),
            pack_radar_scan(_NO_RADAR_DETECTIONS),
        ]
    )


def test_sensors_off_keeps_model_plan_despite_obstacle() -> None:
    prediction = _prediction([_straight(10.0)])

    result, debug = _step(
        _filter(), prediction, _car_ahead(8.0), 10.0, mode=SensorMode.NONE
    )

    assert debug["reason"] == "sensors_off"
    assert result is prediction


def test_radar_mode_ignores_lidar_obstacle_that_fused_mode_brakes_for() -> None:
    payload = _with_empty_radar(_car_ahead(8.0))
    prediction = _prediction([_straight(10.0)])

    _, fused = _step(_filter(), prediction, payload, 10.0, mode=SensorMode.FUSED)
    _, radar = _step(_filter(), prediction, payload, 10.0, mode=SensorMode.RADAR)

    assert fused["reason"] == "speed_override"
    assert radar["reason"] == "safe"
    assert "num_lidar_obstacles" not in radar


def test_policy_sensor_missing_from_payload_fails() -> None:
    radar_only = pack_sensor_payloads([pack_radar_scan(_NO_RADAR_DETECTIONS)])

    with pytest.raises(ValueError, match="LIDAR"):
        _filter().sense(radar_only, 10.0, SensorMode.LIDAR)
