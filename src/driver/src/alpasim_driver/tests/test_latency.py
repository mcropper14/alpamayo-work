# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest

from ..latency import StageTimestamps


def test_stage_durations_partition_the_inference() -> None:
    stages = StageTimestamps(
        vlm_first_start=1.0,
        vision_start=1.1,
        vision_end=1.3,
        vlm_first_end=1.5,
        vlm_last_end=2.5,
        vlm_forward_count=11,
    )

    durations = stages.stage_durations_ms(start=0.5, end=3.0)

    assert durations["preprocess_ms"] == pytest.approx(500.0)
    assert durations["vision_encoder_ms"] == pytest.approx(200.0)
    assert durations["llm_prefill_ms"] == pytest.approx(300.0)
    assert durations["cot_decode_ms"] == pytest.approx(1000.0)
    assert durations["trajectory_ms"] == pytest.approx(500.0)
    assert durations["cot_decode_steps"] == 10
    stage_sum = sum(v for k, v in durations.items() if k.endswith("_ms"))
    assert stage_sum == pytest.approx(2500.0)


def test_stage_durations_empty_without_vlm_hooks() -> None:
    assert StageTimestamps().stage_durations_ms(start=0.0, end=1.0) == {}
