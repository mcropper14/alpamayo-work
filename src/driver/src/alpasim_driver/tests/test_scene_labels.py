# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from concurrent.futures import Future
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from alpasim_driver.frame_cache import FrameEntry
from alpasim_driver.main import DriveJob, EgoDriverService, Session
from alpasim_driver.models import DriveCommand
from alpasim_driver.models.base import BaseTrajectoryModel, CameraFrame
from alpasim_driver.scene_labels import (
    SCENARIO,
    UNKNOWN,
    LabelEstimate,
    SensorMode,
    parse_sensor_policy,
)
from alpasim_driver.schema import SafetyValidatorConfig, SceneLabelConfig


def _answer(label: str) -> dict[str, float]:
    probabilities = dict.fromkeys(("clear", "rain", "fog", "snow"), 0.1)
    probabilities[label] = 0.7
    return probabilities


def test_label_is_unknown_until_first_answer() -> None:
    weather = LabelEstimate(SceneLabelConfig(history=3))

    assert weather.label == UNKNOWN
    assert weather.query_due(0)


def test_label_is_majority_of_recent_answers() -> None:
    weather = LabelEstimate(SceneLabelConfig(history=3))
    for time_us, label in enumerate(["rain", "clear", "rain"]):
        weather.update(time_us, _answer(label))

    assert weather.label == "rain"


def test_old_answers_leave_the_window() -> None:
    weather = LabelEstimate(SceneLabelConfig(history=2))
    for time_us, label in enumerate(["rain", "rain", "fog", "fog"]):
        weather.update(time_us, _answer(label))

    assert weather.label == "fog"


def test_tie_goes_to_most_recent_answer() -> None:
    weather = LabelEstimate(SceneLabelConfig(history=2))
    weather.update(0, _answer("clear"))
    weather.update(1, _answer("rain"))

    assert weather.label == "rain"


def test_next_query_waits_for_interval() -> None:
    weather = LabelEstimate(SceneLabelConfig(query_interval_s=2.0))
    weather.update(1_000_000, _answer("clear"))

    assert not weather.query_due(2_999_999)
    assert weather.query_due(3_000_000)


def test_default_policy_turns_safety_off_only_in_clear_weather() -> None:
    policy = parse_sensor_policy(SafetyValidatorConfig().weather_sensor_policy)

    assert policy["clear"] is SensorMode.NONE
    assert policy["rain"] is SensorMode.FUSED
    assert policy["fog"] is SensorMode.RADAR
    assert policy[UNKNOWN] is SensorMode.FUSED


def test_policy_must_cover_every_label() -> None:
    with pytest.raises(ValueError, match="snow"):
        parse_sensor_policy(
            {"unknown": "fused", "clear": "none", "rain": "fused", "fog": "radar"}
        )


def test_policy_rejects_unknown_sensor_mode() -> None:
    policy = dict(SafetyValidatorConfig().weather_sensor_policy, fog="sonar")

    with pytest.raises(ValueError, match="sonar"):
        parse_sensor_policy(policy)


class _IntersectionModel:
    def __init__(self) -> None:
        self.asked: list[tuple[str, int, str]] = []

    def classify(
        self,
        camera_id: str,
        frame: CameraFrame,
        question: str,
        labels: tuple[str, ...],
    ) -> dict[str, float]:
        self.asked.append((camera_id, frame.timestamp_us, question))
        return {label: 1.0 if label == "intersection" else 0.0 for label in labels}


class _FrameCache:
    def latest_frame_entries(self, count: int) -> list[FrameEntry]:
        assert count == 1
        return [FrameEntry(timestamp_us=42, image=np.zeros((2, 2, 3), np.uint8))]


def test_worker_labels_scene_from_latest_frame_of_configured_camera() -> None:
    cfg = SceneLabelConfig(enabled=True, camera="camera_front_wide_120fov")
    service = EgoDriverService.__new__(EgoDriverService)
    service._device = torch.device("cpu")
    model = _IntersectionModel()
    service._model = cast(BaseTrajectoryModel, model)
    scenario = LabelEstimate(cfg)
    session = SimpleNamespace(frame_caches={cfg.camera: _FrameCache()})
    job = DriveJob(
        session_id="session",
        session=cast(Session, session),
        command=DriveCommand.STRAIGHT,
        pose=None,
        timestamp_us=1_000,
        result=Future(),
        enqueued_at=0.0,
    )

    service._query_label(job, SCENARIO, cfg.camera, scenario)

    assert model.asked == [(cfg.camera, 42, SCENARIO.question)]
    assert scenario.label == "intersection"
    assert not scenario.query_due(1_000)
    assert "scenario_label_ms" in job.timings
