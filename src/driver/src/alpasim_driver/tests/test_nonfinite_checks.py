# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..latency import StageRecorder
from ..main import DriveJob, EgoDriverService, _release_traceback_locals
from ..models.alpamayo_base import _raise_if_nonfinite
from ..models.base import DriveCommand, NonFiniteModelOutputError


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_values_name_the_stage(bad: float) -> None:
    with pytest.raises(NonFiniteModelOutputError, match="VLM logits"):
        _raise_if_nonfinite("VLM logits", torch.tensor([0.0, bad]))


def test_finite_values_pass() -> None:
    _raise_if_nonfinite("VLM logits", torch.tensor([0.0, -1e30]))


class _NaNModel:
    def predict_batch(self, inputs):
        raise NonFiniteModelOutputError("VLM logits produced NaN or inf values")


def test_run_batch_saves_inputs_of_a_nonfinite_inference(tmp_path) -> None:
    service = EgoDriverService.__new__(EgoDriverService)
    service._cfg = SimpleNamespace(output_dir=str(tmp_path))
    service._model = _NaNModel()
    service._device = torch.device("cpu")
    service._stage_recorder = StageRecorder(service._device)
    service._inference_index = 0
    service._get_speed_and_acceleration = lambda session: (0.0, 0.0)
    # The frame cache hands out plain (timestamp, image) tuples.
    service._prepare_camera_images = lambda session: {
        "cam": [(5, np.zeros((2, 2, 3), np.uint8))]
    }
    session = SimpleNamespace(
        seed=7,
        inference_count=0,
        poses=[],
        last_selected_plan=None,
        route=None,
        frames_trail_request_warned=False,
        debug_scene_id="scene-a",
    )
    job = DriveJob(
        session_id="session-a",
        session=session,
        command=DriveCommand.STRAIGHT,
        pose=None,
        timestamp_us=123,
        result=None,  # type: ignore[arg-type]
        enqueued_at=0.0,
    )

    with pytest.raises(NonFiniteModelOutputError):
        service._run_batch([job])

    saved = torch.load(
        tmp_path / "nonfinite_inputs" / "session-a_123.pt", weights_only=False
    )
    assert saved["scene_ids"] == ["scene-a"]
    (prediction_input,) = saved["inputs"]
    assert prediction_input.inference_seed == 7
    ((timestamp_us, image),) = prediction_input.camera_images["cam"]
    assert timestamp_us == 5
    assert isinstance(image, torch.Tensor)


def test_release_traceback_locals_frees_tensors_held_by_failed_frames() -> None:
    import weakref

    def failing_inference():
        activations = torch.zeros(4)
        ref = weakref.ref(activations)
        raise RuntimeError(ref)

    try:
        failing_inference()
    except RuntimeError as exc:
        error = exc
    (ref,) = error.args
    assert ref() is not None

    _release_traceback_locals(error)

    assert ref() is None
