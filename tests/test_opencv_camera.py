"""Tests for the OpenCV camera source."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pupil_tracker.camera import CameraError, OpenCVCamera


class FakeVideoCapture:
    def __init__(self, camera_id: int | str) -> None:
        self.camera_id = camera_id
        self.opened = True
        self.released = False
        self.properties: dict[int, float] = {}
        self.frame = np.zeros((4, 5, 3), dtype=np.uint8)
        self.read_success = True

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop_id: int, value: float) -> bool:
        self.properties[prop_id] = value
        return True

    def read(self) -> tuple[bool, Any]:
        return self.read_success, self.frame

    def release(self) -> None:
        self.released = True
        self.opened = False


def test_constructor_stores_camera_id_and_optional_dimensions() -> None:
    camera = OpenCVCamera(camera_id="external", width=1280, height=720)

    assert camera.camera_id == "external"
    assert camera.width == 1280
    assert camera.height == 720
    assert not camera.is_open


def test_read_before_open_raises_clear_camera_error() -> None:
    camera = OpenCVCamera(camera_id=0)

    with pytest.raises(CameraError, match="not open"):
        camera.read()


def test_open_configures_capture_and_read_returns_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: list[FakeVideoCapture] = []

    def fake_video_capture(camera_id: int | str) -> FakeVideoCapture:
        capture = FakeVideoCapture(camera_id)
        captures.append(capture)
        return capture

    monkeypatch.setattr("pupil_tracker.camera.opencv_camera.cv2.VideoCapture", fake_video_capture)
    camera = OpenCVCamera(camera_id=1, width=640, height=480)

    camera.open()
    frame = camera.read()

    assert captures[0].camera_id == 1
    assert captures[0].properties[3] == 640
    assert captures[0].properties[4] == 480
    assert frame.image.shape == (4, 5, 3)
    assert frame.metadata.camera_id == 1
    assert frame.metadata.width == 5
    assert frame.metadata.height == 4
    assert frame.metadata.channels == 3
    assert frame.metadata.timestamp > 0.0


def test_open_raises_camera_error_when_capture_does_not_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_video_capture(camera_id: int | str) -> FakeVideoCapture:
        capture = FakeVideoCapture(camera_id)
        capture.opened = False
        return capture

    monkeypatch.setattr("pupil_tracker.camera.opencv_camera.cv2.VideoCapture", fake_video_capture)
    camera = OpenCVCamera(camera_id=0)

    with pytest.raises(CameraError, match="failed to open"):
        camera.open()


def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: list[FakeVideoCapture] = []

    def fake_video_capture(camera_id: int | str) -> FakeVideoCapture:
        capture = FakeVideoCapture(camera_id)
        captures.append(capture)
        return capture

    monkeypatch.setattr("pupil_tracker.camera.opencv_camera.cv2.VideoCapture", fake_video_capture)
    camera = OpenCVCamera(camera_id=0)
    camera.open()

    camera.close()
    camera.close()

    assert captures[0].released
    assert not camera.is_open
