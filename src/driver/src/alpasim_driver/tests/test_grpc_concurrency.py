# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import DynamicState, PoseAtTime
from alpasim_grpc.v0.egodriver_pb2 import DriveRequest, RolloutCameraImage
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceStub,
    add_EgodriverServiceServicer_to_server,
)
from PIL import Image

import grpc

from ..main import DriveJob, EgoDriverService, Session
from ..models import DriveCommand
from ..models.base import BaseTrajectoryModel, ModelPrediction
from ..scene_labels import SensorMode
from ..schema import DriverConfig


class _DebugConfig:
    plot_debug_images = False
    output_dir = ""

    class model:
        image_decode_device = "cpu"


class _ConcurrentImageSession:
    debug_scene_id = "test-scene"

    def __init__(self) -> None:
        self._image_barrier = threading.Barrier(2)
        self._frames_lock = threading.Lock()
        self.frame_caches = {"camera-left": object(), "camera-right": object()}
        self.received_frames: list[tuple[str, int]] = []

    def rectify_image(self, logical_id: str, image: Image.Image) -> Image.Image:
        return image

    def add_image(
        self, logical_id: str, image_tensor: np.ndarray, timestamp_us: int
    ) -> None:
        self._image_barrier.wait(timeout=5)
        with self._frames_lock:
            self.received_frames.append((logical_id, timestamp_us))


class _ReadySession:
    poses = [PoseAtTime()]
    current_command = DriveCommand.STRAIGHT

    def all_cameras_ready(self) -> bool:
        return True


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_observation_rpcs_run_concurrently() -> None:
    service = EgoDriverService.__new__(EgoDriverService)
    service._cfg = cast(DriverConfig, _DebugConfig())
    service._model = cast(BaseTrajectoryModel, SimpleNamespace(MIN_FRAME_HW=None))
    session = _ConcurrentImageSession()
    service._sessions = {"session": cast(Session, session)}

    server_executor = ThreadPoolExecutor(max_workers=2)
    server = grpc.server(server_executor)
    add_EgodriverServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")

    try:
        grpc.channel_ready_future(channel).result(timeout=5)
        stub = EgodriverServiceStub(channel)
        requests = [
            RolloutCameraImage(
                session_uuid="session",
                camera_image=RolloutCameraImage.CameraImage(
                    logical_id=logical_id,
                    frame_end_us=index,
                    image_bytes=_image_bytes(),
                ),
            )
            for index, logical_id in enumerate(("camera-left", "camera-right"), start=1)
        ]

        with ThreadPoolExecutor(max_workers=2) as client_executor:
            responses = [
                client_executor.submit(stub.submit_image_observation, request)
                for request in requests
            ]
            for response in responses:
                response.result(timeout=5)

        assert set(session.received_frames) == {
            ("camera-left", 1),
            ("camera-right", 2),
        }
    finally:
        channel.close()
        server.stop(grace=None).wait(timeout=5)
        server_executor.shutdown(wait=True)


def test_drive_rejects_work_after_worker_starts_stopping() -> None:
    service = EgoDriverService.__new__(EgoDriverService)
    service._sessions = {"session": cast(Session, _ReadySession())}
    service._worker_lifecycle_lock = threading.Lock()
    service._worker_stop = threading.Event()
    service._worker_stop.set()
    service._job_queue = queue.Queue[DriveJob | object]()

    with pytest.raises(RuntimeError, match="worker is stopping"):
        service.drive(DriveRequest(session_uuid="session"), None)

    assert service._job_queue.empty()


class _SafetyRun(Exception):
    """Ends drive() once the safety filter has judged the prediction."""


class _RecordingSafetyFilter:
    def __init__(self) -> None:
        self.sensed = threading.Event()

    def sense(self, renderer_data: bytes, speed: float, mode: SensorMode) -> str:
        self.sensed.set()
        return "obstacles"

    def apply(self, prediction: object, sensed: str, speed: float) -> None:
        assert sensed == "obstacles"
        raise _SafetyRun


class _SafetySession(_ReadySession):
    def __init__(self) -> None:
        self.safety_filter = _RecordingSafetyFilter()
        self.dynamic_states = [(0, DynamicState())]


def test_drive_senses_obstacles_while_model_infers() -> None:
    service = EgoDriverService.__new__(EgoDriverService)
    session = _SafetySession()
    service._sessions = {"session": cast(Session, session)}
    service._worker_lifecycle_lock = threading.Lock()
    service._worker_stop = threading.Event()
    service._job_queue = queue.Queue[DriveJob | object]()
    service._sensor_policy = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        drive = executor.submit(
            service.drive, DriveRequest(session_uuid="session"), None
        )
        # Play the inference worker: hold the job until sensing has happened.
        job = service._job_queue.get(timeout=5)
        assert isinstance(job, DriveJob)
        assert session.safety_filter.sensed.wait(timeout=5)
        assert not job.result.done()
        job.result.set_result(cast(ModelPrediction, SimpleNamespace(selected_index=0)))

        with pytest.raises(_SafetyRun):
            drive.result(timeout=5)
