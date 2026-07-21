from __future__ import annotations

import pytest

from football_analytics.multicamera import CameraConfig, MultiCameraSetup


def test_manual_offset_defaults_to_offset_seconds():
    camera = CameraConfig(camera_id="cam1", video_path="a.mp4", offset_seconds=2.5)
    assert camera.manual_offset_seconds == 2.5


def test_explicit_manual_offset_wins():
    camera = CameraConfig(
        camera_id="cam1", video_path="a.mp4", offset_seconds=2.5, manual_offset_seconds=1.0
    )
    assert camera.manual_offset_seconds == 1.0


def test_setup_defaults_reference_to_first_camera():
    setup = MultiCameraSetup(
        cameras=[
            CameraConfig(camera_id="cam1", video_path="a.mp4"),
            CameraConfig(camera_id="cam2", video_path="b.mp4"),
        ]
    )
    assert setup.reference_camera_id == "cam1"
    assert "cam2" in setup
    assert setup.camera("cam2").video_path == "b.mp4"


def test_setup_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        MultiCameraSetup(
            cameras=[
                CameraConfig(camera_id="cam1", video_path="a.mp4"),
                CameraConfig(camera_id="cam1", video_path="b.mp4"),
            ]
        )


def test_setup_rejects_unknown_reference():
    with pytest.raises(KeyError):
        MultiCameraSetup(
            cameras=[CameraConfig(camera_id="cam1", video_path="a.mp4")],
            reference_camera_id="ghost",
        )
