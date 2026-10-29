# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for SensorsimService batch (batch_render_rgb) response mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from alpasim_grpc.v0.sensorsim_pb2 import (
    BatchRGBRenderReturnItem,
    LidarRenderReturn,
    RGBRenderReturn,
)
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_utils.geometry import Pose, Trajectory, pose_from_grpc
from alpasim_utils.lidar_payload import unpack_lidar_spin


def _item(camera_name: str, image: bytes = b"img", success: bool = True, err: str = ""):
    return BatchRGBRenderReturnItem(
        camera_name=camera_name,
        result=RGBRenderReturn(image_bytes=image),
        success=success,
        error_message=err,
    )


def _triggers() -> dict[str, Clock.Trigger]:
    return {
        "cam_a": Clock.Trigger(range(0, 33_000), sequential_idx=0),
        "cam_b": Clock.Trigger(range(100_000, 133_000), sequential_idx=1),
    }


@pytest.mark.asyncio
async def test_sensorsim_service_uses_large_grpc_message_limits(monkeypatch):
    captured: dict[str, object] = {}
    fake_channel = object()

    def fake_insecure_channel(address, options=None):
        captured["address"] = address
        captured["options"] = options
        return fake_channel

    class FakeStub:
        def __init__(self, channel):
            captured["channel"] = channel

    monkeypatch.setattr(
        "alpasim_runtime.services.service_base.grpc.aio.insecure_channel",
        fake_insecure_channel,
    )
    monkeypatch.setattr(
        "alpasim_runtime.services.sensorsim_service.SensorsimServiceStub",
        FakeStub,
    )

    service = SensorsimService("localhost:50051", False, MagicMock())
    await service._open_connection()

    assert captured["address"] == "localhost:50051"
    assert captured["channel"] is fake_channel
    options = dict(captured["options"])
    assert options.keys() == {
        "grpc.max_receive_message_length",
        "grpc.max_send_message_length",
    }
    assert options["grpc.max_receive_message_length"] > 4 * 1024 * 1024
    assert (
        options["grpc.max_send_message_length"]
        == options["grpc.max_receive_message_length"]
    )


def test_batch_return_maps_images_by_camera_name():
    items = [_item("cam_a", b"a"), _item("cam_b", b"b")]

    images = SensorsimService._batch_return_to_images(items, _triggers())

    by_cam = {img.camera_logical_id: img for img in images}
    assert set(by_cam) == {"cam_a", "cam_b"}
    assert by_cam["cam_a"].image_bytes == b"a"
    # Timestamps come from the request-side trigger, not the response.
    assert by_cam["cam_a"].start_timestamp_us == 0
    assert by_cam["cam_a"].end_timestamp_us == 33_000
    assert by_cam["cam_b"].start_timestamp_us == 100_000


def test_batch_return_raises_on_failed_item():
    items = [
        _item("cam_a"),
        _item("cam_b", success=False, err="actor editing disabled"),
    ]

    with pytest.raises(RuntimeError, match=r"cam_b.*actor editing disabled"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_unknown_camera():
    items = [_item("cam_a"), _item("cam_UNEXPECTED")]

    with pytest.raises(RuntimeError, match=r"unknown camera 'cam_UNEXPECTED'"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_duplicate_camera():
    # cam_b would be missing the next loop, but the duplicate must fail first.
    items = [_item("cam_a"), _item("cam_a")]

    with pytest.raises(RuntimeError, match=r"duplicate camera 'cam_a'"):
        SensorsimService._batch_return_to_images(items, _triggers())


def test_batch_return_raises_on_missing_camera():
    # Requested cam_a and cam_b, but NRE only returned cam_a.
    items = [_item("cam_a")]

    with pytest.raises(RuntimeError, match=r"omitted requested camera.*cam_b"):
        SensorsimService._batch_return_to_images(items, _triggers())


@pytest.mark.asyncio
async def test_batch_render_skip_mode_returns_empty_images():
    """In skip mode batch_render returns placeholder frames like render()."""
    svc = SensorsimService("addr:0", skip=True, camera_catalog=MagicMock())
    cam = RuntimeCamera(
        logical_id="cam_a",
        render_resolution_hw=(2, 2),
        clock=Clock(interval_us=100_000, duration_us=33_000, start_us=0),
    )
    trigger = cam.clock.ith_trigger(0)

    images, driver_data = await svc.batch_render(
        [(cam, trigger)],
        ego_trajectory=MagicMock(),
        traffic_trajectories={},
        scene_id="scene",
        image_format=MagicMock(),
    )

    assert driver_data is None
    assert [img.camera_logical_id for img in images] == ["cam_a"]
    assert images[0].image_bytes == b""


@pytest.mark.asyncio
async def test_render_lidar_payload_places_sensor_at_lidar_mount():
    """NRE raycasts from the sensor pose; it must be the roof mount, not the rig
    origin on the road."""
    svc = SensorsimService("addr:0", skip=False, camera_catalog=MagicMock())
    svc.stub = MagicMock()
    requests = []

    async def fake_rpc(method_name, stub_method, request):
        requests.append(request)
        xyz = np.array([[10.0, 0.0, -1.9]], dtype=np.float32)
        return LidarRenderReturn(
            num_points=1,
            point_xyzs_buffer=xyz.tobytes(),
        )

    svc._sensorsim_rpc = fake_rpc
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    ego_trajectory = Trajectory.from_poses(
        timestamps=np.array([0, 100_000], dtype=np.uint64),
        poses=[
            Pose(np.array([x, 5.0, 0.0], dtype=np.float32), identity)
            for x in (100.0, 101.0)
        ],
    )
    rig_to_lidar = Pose(np.array([1.2, 0.0, 1.9], dtype=np.float32), identity)

    payload = await svc.render_lidar_payload(
        ego_trajectory=ego_trajectory,
        traffic_trajectories={},
        trigger=Clock.Trigger(range(0, 100_000), sequential_idx=0),
        scene_id="scene",
        rig_to_lidar=rig_to_lidar,
    )

    (request,) = requests
    np.testing.assert_allclose(
        pose_from_grpc(request.sensor_pose.start_pose).vec3, [101.2, 5.0, 1.9]
    )
    np.testing.assert_allclose(
        pose_from_grpc(request.sensor_pose.end_pose).vec3, [102.2, 5.0, 1.9]
    )
    spin = unpack_lidar_spin(payload)
    np.testing.assert_allclose(spin.rig_to_sensor.vec3, [1.2, 0.0, 1.9])
    np.testing.assert_allclose(
        spin.sensor_to_rig(spin.xyz), [[11.2, 0.0, 0.0]], atol=1e-5
    )
