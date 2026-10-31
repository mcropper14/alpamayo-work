# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Per-session scene labels from the driving model's own VQA answers.

Each label task (``WEATHER``, ``SCENARIO``) is a multiple-choice question.
Every ``query_interval_s`` of simulation time the inference worker asks the
model to score each answer for the latest frame of the task's camera
(``BaseTrajectoryModel.classify``).  Single answers are noisy, so a session's
label is the most frequent of the last ``history`` answers, the most recent
winning ties, and ``UNKNOWN`` before the first answer.

The weather label picks the safety validator's sensors through its
``weather_sensor_policy``.  The scenario label is logged and shown in videos.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
from collections.abc import Mapping

from .schema import SceneLabelConfig

UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class LabelTask:
    name: str
    question: str
    labels: tuple[str, ...]


def _task(name: str, question: str, labels: tuple[str, ...]) -> LabelTask:
    return LabelTask(
        name, f"{question} Answer with one of: {', '.join(labels)}.", labels
    )


WEATHER = _task(
    "weather", "What is the weather in this scene?", ("clear", "rain", "fog", "snow")
)
SCENARIO = _task(
    "scenario",
    "What kind of driving situation is this?",
    (
        "lane following",
        "intersection",
        "construction zone",
        "pedestrians nearby",
        "lane change",
    ),
)


class SensorMode(enum.StrEnum):
    NONE = "none"  # skip the safety check
    LIDAR = "lidar"
    RADAR = "radar"
    FUSED = "fused"


def parse_sensor_policy(policy: Mapping[str, str]) -> dict[str, SensorMode]:
    """Validate a weather -> sensor policy; every label and ``unknown`` needed."""
    missing = {UNKNOWN, *WEATHER.labels} - policy.keys()
    if missing:
        raise ValueError(f"weather_sensor_policy has no entry for {sorted(missing)}")
    return {label: SensorMode(mode) for label, mode in policy.items()}


class LabelEstimate:
    def __init__(self, cfg: SceneLabelConfig) -> None:
        if cfg.history < 1:
            raise ValueError(f"history must be at least 1, got {cfg.history}")
        self._interval_us = int(cfg.query_interval_s * 1e6)
        self._answers: collections.deque[str] = collections.deque(maxlen=cfg.history)
        self._last_query_us: int | None = None
        self.last_probabilities: dict[str, float] | None = None

    def query_due(self, time_us: int) -> bool:
        return (
            self._last_query_us is None
            or time_us - self._last_query_us >= self._interval_us
        )

    def update(self, time_us: int, probabilities: Mapping[str, float]) -> None:
        self._last_query_us = time_us
        self.last_probabilities = dict(probabilities)
        self._answers.append(max(probabilities, key=probabilities.__getitem__))

    @property
    def label(self) -> str:
        if not self._answers:
            return UNKNOWN
        counts = collections.Counter(self._answers)
        top = max(counts.values())
        return next(a for a in reversed(self._answers) if counts[a] == top)
