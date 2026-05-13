"""Tests for the pluggable tracker backend contract."""

import numpy as np

from pupil_tracker.models import FrameMetadata, RawObservation
from pupil_tracker.tracking import Frame, TrackerBackend


class FakeBackend:
    """Simple backend proving the public protocol shape."""

    name = "fake"

    def __init__(self) -> None:
        self.closed = False

    def process(self, frame: Frame) -> RawObservation:
        return RawObservation(
            timestamp=frame.metadata.timestamp,
            valid=True,
            confidence=0.75,
            feature_vector=(0.1, 0.2),
        )

    def close(self) -> None:
        self.closed = True


def _frame() -> Frame:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    metadata = FrameMetadata(timestamp=1.25, camera_id=0, width=3, height=2, channels=3)
    return Frame(image=image, metadata=metadata)


def test_frame_stores_image_and_metadata() -> None:
    frame = _frame()

    assert frame.image.shape == (2, 3, 3)
    assert frame.metadata.width == 3
    assert frame.metadata.height == 2


def test_backend_protocol_exposes_name() -> None:
    backend: TrackerBackend = FakeBackend()

    assert backend.name == "fake"


def test_backend_process_returns_raw_observation() -> None:
    backend: TrackerBackend = FakeBackend()

    observation = backend.process(_frame())

    assert observation.timestamp == 1.25
    assert observation.valid
    assert observation.confidence == 0.75
    assert observation.feature_vector == (0.1, 0.2)


def test_backend_close_can_be_called_safely() -> None:
    backend = FakeBackend()

    backend.close()

    assert backend.closed
