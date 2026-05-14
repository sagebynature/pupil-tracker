"""Pure state machine for the desktop demo runtime."""

from __future__ import annotations

from enum import Enum


class DemoMode(Enum):
    """High-level modes for the desktop demo."""

    STOPPED = "stopped"
    PREVIEWING = "previewing"
    CALIBRATING = "calibrating"
    VALIDATING = "validating"
    TRACKING = "calibrated_tracking"
    ERROR = "error"


class InvalidDemoStateTransition(RuntimeError):
    """Raised when a demo event is invalid for the current mode."""


class DemoStateMachine:
    """Explicit state transitions for preview, calibration, tracking, and errors."""

    def __init__(self) -> None:
        self.mode = DemoMode.STOPPED
        self.error_message: str | None = None

    def camera_started(self) -> None:
        """Move to previewing after camera startup or error recovery."""

        self.mode = DemoMode.PREVIEWING
        self.error_message = None

    def camera_stopped(self) -> None:
        """Stop all live activity and clear any error."""

        self.mode = DemoMode.STOPPED
        self.error_message = None

    def camera_failed(self, message: str) -> None:
        """Record a camera failure and move to error mode."""

        self.mode = DemoMode.ERROR
        self.error_message = message

    def calibration_started(self) -> None:
        """Move from preview/tracking into calibration collection."""

        if self.mode not in {DemoMode.PREVIEWING, DemoMode.TRACKING, DemoMode.VALIDATING}:
            msg = f"cannot start calibration while {self.mode.value}"
            raise InvalidDemoStateTransition(msg)
        self.mode = DemoMode.CALIBRATING
        self.error_message = None

    def calibration_completed(self) -> None:
        """Move from calibration collection into calibrated tracking."""

        if self.mode is not DemoMode.CALIBRATING:
            msg = f"cannot complete calibration while {self.mode.value}"
            raise InvalidDemoStateTransition(msg)
        self.mode = DemoMode.VALIDATING
        self.error_message = None

    def validation_passed(self) -> None:
        """Move from validation into trusted calibrated tracking."""

        if self.mode is not DemoMode.VALIDATING:
            msg = f"cannot pass validation while {self.mode.value}"
            raise InvalidDemoStateTransition(msg)
        self.mode = DemoMode.TRACKING
        self.error_message = None

    def validation_failed(self, message: str) -> None:
        """Record validation failure and require calibration/validation recovery."""

        if self.mode is not DemoMode.VALIDATING:
            msg = f"cannot fail validation while {self.mode.value}"
            raise InvalidDemoStateTransition(msg)
        self.mode = DemoMode.ERROR
        self.error_message = message

    def calibration_failed(self, message: str) -> None:
        """Record a calibration failure and move to error mode."""

        self.mode = DemoMode.ERROR
        self.error_message = message
