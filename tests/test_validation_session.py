"""Tests for post-calibration validation session collection."""

from __future__ import annotations

import sys
from pathlib import Path

from pupil_tracker.calibration import TimedCalibrationConfig, validation_pattern
from pupil_tracker.models import GazeSample

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def gaze(*, x: float, y: float, valid: bool = True) -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=x,
        y=y,
        confidence=0.9,
        valid=valid,
        region_id=None,
    )


def config() -> TimedCalibrationConfig:
    return TimedCalibrationConfig(
        settle_seconds=1.0,
        capture_seconds=1.0,
        min_samples_per_target=1,
        min_confidence=0.0,
    )


def test_validation_ignores_samples_during_settle() -> None:
    from desktop_demo.validation_session import ValidationSession, ValidationSessionState

    clock = FakeClock()
    session = ValidationSession(
        targets=validation_pattern()[:1],
        screen_width=1000.0,
        screen_height=800.0,
        timing_config=config(),
        clock=clock,
    )
    session.start()

    assert session.state is ValidationSessionState.SETTLING
    assert session.capture(gaze(x=250.0, y=200.0)) is False
    assert session.samples_for_current_target() == ()


def test_validation_collects_valid_gaze_during_capture() -> None:
    from desktop_demo.validation_session import ValidationSession, ValidationSessionState

    clock = FakeClock()
    session = ValidationSession(
        targets=validation_pattern()[:1],
        screen_width=1000.0,
        screen_height=800.0,
        timing_config=config(),
        clock=clock,
    )
    session.start()
    clock.advance(1.0)

    assert session.capture(gaze(x=250.0, y=200.0)) is False

    assert session.state is ValidationSessionState.CAPTURING
    assert len(session.samples_for_current_target()) == 1
    assert session.accepted_for_current_target == 1


def test_validation_computes_metrics_after_all_targets() -> None:
    from desktop_demo.validation_session import ValidationSession, ValidationSessionState

    clock = FakeClock()
    targets = validation_pattern()[:2]
    session = ValidationSession(
        targets=targets,
        screen_width=1000.0,
        screen_height=800.0,
        timing_config=config(),
        clock=clock,
    )
    session.start()

    clock.advance(1.0)
    assert session.capture(gaze(x=250.0, y=200.0)) is False
    clock.advance(1.1)
    assert session.capture(gaze(x=250.0, y=200.0)) is True

    assert session.state is ValidationSessionState.SETTLING
    assert session.current_target == targets[1]

    clock.advance(1.0)
    assert session.capture(gaze(x=750.0, y=200.0)) is False
    clock.advance(1.1)
    assert session.capture(gaze(x=750.0, y=200.0)) is True

    assert session.state is ValidationSessionState.COMPLETE
    assert session.metrics is not None
    assert session.metrics.sample_count == 2
    assert session.metrics.mean_error_px == 0.0
    assert session.metrics.recommendation == "excellent"
