"""OpenCV-backed camera source."""

from __future__ import annotations

import time
from typing import Any

import cv2

from pupil_tracker.logging_config import get_logger
from pupil_tracker.models import FrameMetadata
from pupil_tracker.tracking import Frame

_LOGGER = get_logger("camera")


class CameraError(RuntimeError):
    """Raised when camera lifecycle or frame capture fails."""


class OpenCVCamera:
    """Read camera frames through OpenCV behind the runtime camera interface."""

    def __init__(
        self,
        camera_id: int | str = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._capture: Any | None = None

    @property
    def is_open(self) -> bool:
        """Return whether the camera currently has an open capture."""

        return self._capture is not None and bool(self._capture.isOpened())

    def open(self) -> None:
        """Open the OpenCV capture and apply optional dimensions."""

        if self.is_open:
            return

        capture = cv2.VideoCapture(self.camera_id)
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        if not capture.isOpened():
            capture.release()
            msg = f"failed to open camera {self.camera_id!r}"
            raise CameraError(msg)

        self._capture = capture
        _LOGGER.info("opened camera %r", self.camera_id)

    def read(self) -> Frame:
        """Read the next frame from the camera."""

        if not self.is_open or self._capture is None:
            msg = "camera is not open"
            raise CameraError(msg)

        success, image = self._capture.read()
        if not success or image is None:
            msg = f"failed to read frame from camera {self.camera_id!r}"
            raise CameraError(msg)

        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        metadata = FrameMetadata(
            timestamp=time.monotonic(),
            camera_id=self.camera_id,
            width=int(width),
            height=int(height),
            channels=int(channels),
        )
        return Frame(image=image, metadata=metadata)

    def close(self) -> None:
        """Release the OpenCV capture if one is open."""

        if self._capture is None:
            return

        self._capture.release()
        self._capture = None
        _LOGGER.info("closed camera %r", self.camera_id)
