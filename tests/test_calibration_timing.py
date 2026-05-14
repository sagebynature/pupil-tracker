"""Tests for accuracy-first timed calibration phases."""

from __future__ import annotations

from typing import Any

import pytest

from pupil_tracker.calibration import (
    CalibrationPhase,
    TimedCalibrationConfig,
    TimedTargetTimer,
)


def test_timed_state_starts_in_settling_phase() -> None:
    timer = TimedTargetTimer(TimedCalibrationConfig())

    state = timer.start(now_seconds=10.0)

    assert state.phase is CalibrationPhase.SETTLING
    assert state.target_started_at == 10.0
    assert state.capture_started_at is None
    assert state.accepted_count == 0
    assert state.rejected_count == 0
    assert state.progress == 0.0


def test_timed_state_moves_to_capturing_after_settle_duration() -> None:
    timer = TimedTargetTimer(TimedCalibrationConfig(settle_seconds=1.0))
    state = timer.start(now_seconds=10.0)

    state = timer.update(state, now_seconds=11.0)

    assert state.phase is CalibrationPhase.CAPTURING
    assert state.capture_started_at == 11.0
    assert state.progress == 0.0


def test_timed_state_reports_capture_progress_and_review_phase() -> None:
    timer = TimedTargetTimer(
        TimedCalibrationConfig(settle_seconds=1.0, capture_seconds=2.0)
    )
    state = timer.start(now_seconds=10.0)
    state = timer.update(state, now_seconds=11.0)

    state = timer.update(
        state,
        now_seconds=12.0,
        accepted_count=12,
        rejected_count=3,
    )

    assert state.phase is CalibrationPhase.CAPTURING
    assert state.progress == pytest.approx(0.5)
    assert state.accepted_count == 12
    assert state.rejected_count == 3

    state = timer.update(state, now_seconds=13.0)

    assert state.phase is CalibrationPhase.REVIEWING
    assert state.progress == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"settle_seconds": 0.0}, "settle_seconds must be positive"),
        ({"capture_seconds": 0.0}, "capture_seconds must be positive"),
        ({"min_samples_per_target": 0}, "min_samples_per_target must be positive"),
        ({"min_confidence": -0.1}, "min_confidence must be between 0 and 1"),
        ({"min_confidence": 1.1}, "min_confidence must be between 0 and 1"),
    ],
)
def test_timed_config_rejects_invalid_thresholds(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TimedCalibrationConfig(**kwargs)
