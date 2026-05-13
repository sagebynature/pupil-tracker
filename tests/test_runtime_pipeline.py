"""Tests for the synchronous runtime gaze pipeline."""

import numpy as np

from pupil_tracker.models import FrameMetadata, GazeSample, RawObservation
from pupil_tracker.runtime import RuntimePipeline
from pupil_tracker.tracking import Frame


class FakeCamera:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame
        self.read_count = 0

    def read(self) -> Frame:
        self.read_count += 1
        return self.frame


class FakeBackend:
    name = "fake"

    def __init__(self, observation: RawObservation) -> None:
        self.observation = observation
        self.processed_frames: list[Frame] = []

    def process(self, frame: Frame) -> RawObservation:
        self.processed_frames.append(frame)
        return self.observation

    def close(self) -> None:
        pass


class FakeCalibrationModel:
    def __init__(self, gaze: GazeSample) -> None:
        self.gaze = gaze
        self.observations: list[RawObservation] = []

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        self.observations.append(observation)
        return self.gaze


class FakeSmoother:
    def __init__(self) -> None:
        self.samples: list[GazeSample] = []

    def update(self, sample: GazeSample) -> GazeSample:
        self.samples.append(sample)
        return GazeSample(
            timestamp=sample.timestamp,
            x=sample.x + 10,
            y=sample.y + 5,
            confidence=sample.confidence,
            valid=sample.valid,
            region_id=sample.region_id,
        )


def _frame() -> Frame:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    metadata = FrameMetadata(timestamp=1.0, camera_id=0, width=10, height=10, channels=3)
    return Frame(image=image, metadata=metadata)


def test_step_returns_observation_and_gaze_sample() -> None:
    observation = RawObservation(
        timestamp=1.0,
        valid=True,
        confidence=0.8,
        feature_vector=(0.1, 0.2),
    )
    gaze = GazeSample(timestamp=1.0, x=50.0, y=50.0, confidence=0.8, valid=True)
    camera = FakeCamera(_frame())
    backend = FakeBackend(observation)
    calibration_model = FakeCalibrationModel(gaze)
    smoother = FakeSmoother()
    pipeline = RuntimePipeline(
        camera=camera,
        backend=backend,
        calibration_model=calibration_model,
        smoother=smoother,
        screen_width=300,
        screen_height=300,
    )

    result = pipeline.step()

    assert result.observation == observation
    assert result.gaze_sample is not None
    assert camera.read_count == 1
    assert backend.processed_frames == [camera.frame]


def test_invalid_observation_returns_no_gaze_sample() -> None:
    observation = RawObservation.invalid(timestamp=2.0, reason="no face")
    camera = FakeCamera(_frame())
    backend = FakeBackend(observation)
    calibration_model = FakeCalibrationModel(
        GazeSample(timestamp=2.0, x=0.0, y=0.0, confidence=0.0, valid=False)
    )
    smoother = FakeSmoother()
    pipeline = RuntimePipeline(
        camera=camera,
        backend=backend,
        calibration_model=calibration_model,
        smoother=smoother,
        screen_width=300,
        screen_height=300,
    )

    result = pipeline.step()

    assert result.observation == observation
    assert result.gaze_sample is None
    assert calibration_model.observations == []
    assert smoother.samples == []


def test_valid_observation_runs_calibration_smoothing_and_region_mapping() -> None:
    observation = RawObservation(
        timestamp=3.0,
        valid=True,
        confidence=0.9,
        feature_vector=(0.2, 0.4),
    )
    gaze = GazeSample(timestamp=3.0, x=140.0, y=145.0, confidence=0.9, valid=True)
    calibration_model = FakeCalibrationModel(gaze)
    smoother = FakeSmoother()
    pipeline = RuntimePipeline(
        camera=FakeCamera(_frame()),
        backend=FakeBackend(observation),
        calibration_model=calibration_model,
        smoother=smoother,
        screen_width=300,
        screen_height=300,
    )

    result = pipeline.step()

    assert calibration_model.observations == [observation]
    assert smoother.samples == [gaze]
    assert result.gaze_sample is not None
    assert result.gaze_sample.x == 150.0
    assert result.gaze_sample.y == 150.0
    assert result.gaze_sample.region_id == "middle_center"
