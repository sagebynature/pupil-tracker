"""Tests for the pure desktop demo state machine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


def test_state_starts_stopped() -> None:
    from desktop_demo.state import DemoMode, DemoStateMachine

    state = DemoStateMachine()

    assert state.mode is DemoMode.STOPPED
    assert state.error_message is None


def test_state_transitions_from_calibrating_to_tracking_after_fit() -> None:
    from desktop_demo.state import DemoMode, DemoStateMachine

    state = DemoStateMachine()
    state.camera_started()
    state.calibration_started()
    state.calibration_completed()

    assert state.mode is DemoMode.TRACKING
    assert state.error_message is None


def test_camera_stop_resets_any_active_mode() -> None:
    from desktop_demo.state import DemoMode, DemoStateMachine

    state = DemoStateMachine()
    state.camera_started()
    state.calibration_started()

    state.camera_stopped()

    assert state.mode is DemoMode.STOPPED
    assert state.error_message is None


def test_failures_move_to_error_with_message() -> None:
    from desktop_demo.state import DemoMode, DemoStateMachine

    state = DemoStateMachine()
    state.camera_failed("camera unavailable")

    assert state.mode is DemoMode.ERROR
    assert state.error_message == "camera unavailable"


def test_calibration_requires_previewing_or_tracking_mode() -> None:
    from desktop_demo.state import DemoStateMachine, InvalidDemoStateTransition

    state = DemoStateMachine()

    with pytest.raises(InvalidDemoStateTransition, match="cannot start calibration"):
        state.calibration_started()


def test_tracking_completion_requires_calibrating_mode() -> None:
    from desktop_demo.state import DemoStateMachine, InvalidDemoStateTransition

    state = DemoStateMachine()
    state.camera_started()

    with pytest.raises(InvalidDemoStateTransition, match="cannot complete calibration"):
        state.calibration_completed()


def test_error_can_recover_when_camera_restarts() -> None:
    from desktop_demo.state import DemoMode, DemoStateMachine

    state = DemoStateMachine()
    state.camera_failed("read failed")
    state.camera_started()

    assert state.mode is DemoMode.PREVIEWING
    assert state.error_message is None
