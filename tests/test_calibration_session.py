"""Tests for the desktop demo live calibration session."""

from __future__ import annotations

import sys
from pathlib import Path

from pupil_tracker.calibration import (
    CalibrationFitResult,
    CalibrationPhase,
    TimedCalibrationConfig,
)
from pupil_tracker.models import CalibrationSample, RawObservation

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


class FakeCalibrationModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.fit_calls = 0
        self.fit_samples: tuple[CalibrationSample, ...] = ()
        self.fit_screen_size: tuple[float, float] | None = None

    def fit(
        self,
        samples: tuple[CalibrationSample, ...],
        screen_width: float,
        screen_height: float,
    ) -> CalibrationFitResult:
        self.fit_calls += 1
        self.fit_samples = samples
        self.fit_screen_size = (screen_width, screen_height)
        if self.fail:
            msg = "synthetic fit failure"
            raise ValueError(msg)
        return CalibrationFitResult(
            sample_count=len(samples),
            mean_error_px=2.5,
            max_error_px=5.0,
        )


def valid_observation(timestamp: float = 1.0, confidence: float = 0.9) -> RawObservation:
    return RawObservation(
        timestamp=timestamp,
        valid=True,
        confidence=confidence,
        feature_vector=(timestamp, timestamp + 0.1),
    )


def invalid_observation() -> RawObservation:
    return RawObservation.invalid(timestamp=99.0, reason="no face")


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def timed_config() -> TimedCalibrationConfig:
    return TimedCalibrationConfig(
        settle_seconds=1.0,
        capture_seconds=1.0,
        min_samples_per_target=2,
        min_confidence=0.6,
    )


def test_session_starts_idle_and_capture_noops_until_started() -> None:
    from desktop_demo.calibration_session import CalibrationSession, CalibrationSessionState
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    flow = CalibrationFlowState(samples_per_target=1)
    model = FakeCalibrationModel()
    session = CalibrationSession(flow=flow, model=model, screen_width=1000, screen_height=800)

    assert session.state is CalibrationSessionState.IDLE
    assert session.capture(valid_observation()) is False
    assert flow.all_samples() == ()
    assert model.fit_calls == 0


def test_session_captures_only_while_collecting_and_ignores_invalid_observations() -> None:
    from desktop_demo.calibration_session import CalibrationSession, CalibrationSessionState
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    flow = CalibrationFlowState(samples_per_target=2)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    session.start()

    assert session.state is CalibrationSessionState.COLLECTING
    assert session.capture(invalid_observation()) is False
    assert flow.all_samples() == ()

    assert session.capture(valid_observation()) is False
    assert len(flow.samples_for_current_target()) == 1
    assert session.state is CalibrationSessionState.COLLECTING


def test_session_fits_model_after_all_targets_have_samples() -> None:
    from desktop_demo.calibration_session import CalibrationSession, CalibrationSessionState
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    model = FakeCalibrationModel()
    flow = CalibrationFlowState(samples_per_target=1)
    session = CalibrationSession(flow=flow, model=model, screen_width=1000, screen_height=800)
    session.start()

    for index, _target in enumerate(flow.targets):
        session.capture(valid_observation(timestamp=float(index)))

    assert session.state is CalibrationSessionState.COMPLETE
    assert session.is_complete is True
    assert model.fit_calls == 1
    assert len(model.fit_samples) == 9
    assert model.fit_screen_size == (1000, 800)
    assert session.fit_result == CalibrationFitResult(
        sample_count=9,
        mean_error_px=2.5,
        max_error_px=5.0,
    )


def test_session_start_resets_previous_flow_samples() -> None:
    from desktop_demo.calibration_session import CalibrationSession, CalibrationSessionState
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    flow = CalibrationFlowState(samples_per_target=1)
    assert flow.capture_observation(valid_observation())
    assert flow.current_index == 1

    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
    )
    session.start()

    assert session.state is CalibrationSessionState.COLLECTING
    assert flow.current_index == 0
    assert flow.all_samples() == ()


def test_session_moves_to_failed_when_model_fit_fails() -> None:
    from desktop_demo.calibration_session import CalibrationSession, CalibrationSessionState
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    flow = CalibrationFlowState(samples_per_target=1)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(fail=True),
        screen_width=1000,
        screen_height=800,
    )
    session.start()

    for index, _target in enumerate(flow.targets):
        session.capture(valid_observation(timestamp=float(index)))

    assert session.state is CalibrationSessionState.FAILED
    assert session.fit_result is None
    assert session.error_message == "synthetic fit failure"


def test_timed_settle_phase_ignores_valid_observations() -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    clock = FakeClock(now=0.0)
    flow = CalibrationFlowState(samples_per_target=2)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
        timing_config=timed_config(),
        clock=clock,
    )
    session.start()

    assert session.phase is CalibrationPhase.SETTLING
    assert session.capture(valid_observation()) is False
    assert flow.all_samples() == ()
    assert session.accepted_for_current_target == 0


def test_timed_capture_phase_accepts_high_quality_observations() -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    clock = FakeClock(now=0.0)
    flow = CalibrationFlowState(samples_per_target=2)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
        timing_config=timed_config(),
        clock=clock,
    )
    session.start()
    clock.advance(1.0)

    assert session.capture(valid_observation(timestamp=1.0)) is False

    assert session.phase is CalibrationPhase.CAPTURING
    assert len(flow.samples_for_current_target()) == 1
    assert session.accepted_for_current_target == 1
    assert session.rejected_for_current_target == 0


def test_timed_low_quality_target_retries_instead_of_advancing() -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    clock = FakeClock(now=0.0)
    flow = CalibrationFlowState(samples_per_target=2)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
        timing_config=timed_config(),
        clock=clock,
    )
    session.start()
    clock.advance(1.0)
    session.capture(valid_observation(timestamp=1.0))
    session.capture(valid_observation(timestamp=1.1, confidence=0.5))
    clock.advance(1.1)

    assert session.capture(valid_observation(timestamp=2.2)) is True

    assert flow.current_index == 0
    assert flow.samples_for_current_target() == ()
    assert session.target_quality is not None
    assert session.target_quality.recommendation == "retry"
    assert session.phase is CalibrationPhase.SETTLING


def test_timed_high_quality_target_advances_after_capture_duration() -> None:
    from desktop_demo.calibration_session import CalibrationSession
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    clock = FakeClock(now=0.0)
    flow = CalibrationFlowState(samples_per_target=2)
    session = CalibrationSession(
        flow=flow,
        model=FakeCalibrationModel(),
        screen_width=1000,
        screen_height=800,
        timing_config=timed_config(),
        clock=clock,
    )
    session.start()
    clock.advance(1.0)
    session.capture(valid_observation(timestamp=1.0))
    session.capture(valid_observation(timestamp=1.1))
    clock.advance(1.1)

    assert session.capture(valid_observation(timestamp=2.2)) is True

    assert flow.current_index == 1
    assert session.target_quality is not None
    assert session.target_quality.recommendation == "advance"
    assert session.phase is CalibrationPhase.SETTLING
