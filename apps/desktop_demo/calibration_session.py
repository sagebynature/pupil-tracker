"""Live calibration session controller for the desktop demo."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from desktop_demo.ui.calibration_view import CalibrationFlowState
from pupil_tracker.calibration import CalibrationFitResult
from pupil_tracker.models import CalibrationSample, RawObservation


class CalibrationModelLike(Protocol):
    """Calibration model surface used by the desktop calibration session."""

    def fit(
        self,
        samples: tuple[CalibrationSample, ...],
        screen_width: float,
        screen_height: float,
    ) -> CalibrationFitResult:
        """Fit from collected samples and return fit metrics."""


class CalibrationSessionState(Enum):
    """Lifecycle states for a live calibration session."""

    IDLE = "idle"
    COLLECTING = "collecting"
    COMPLETE = "complete"
    FAILED = "failed"


class CalibrationSession:
    """Capture live observations and fit a calibration model when complete."""

    def __init__(
        self,
        *,
        flow: CalibrationFlowState,
        model: CalibrationModelLike,
        screen_width: float,
        screen_height: float,
    ) -> None:
        if screen_width <= 0 or screen_height <= 0:
            msg = "screen dimensions must be positive"
            raise ValueError(msg)
        self.flow = flow
        self.model = model
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.state = CalibrationSessionState.IDLE
        self.fit_result: CalibrationFitResult | None = None
        self.error_message: str | None = None

    @property
    def is_collecting(self) -> bool:
        """Return whether this session is actively collecting observations."""

        return self.state is CalibrationSessionState.COLLECTING

    @property
    def is_complete(self) -> bool:
        """Return whether this session completed successfully."""

        return self.state is CalibrationSessionState.COMPLETE

    def start(self) -> None:
        """Reset flow state and begin collecting observations."""

        self.flow.reset()
        self.fit_result = None
        self.error_message = None
        self.state = CalibrationSessionState.COLLECTING

    def capture(self, observation: RawObservation) -> bool:
        """Capture an observation when collecting.

        Returns True when capture advances the flow or completes calibration.
        """

        if self.state is not CalibrationSessionState.COLLECTING:
            return False

        advanced = self.flow.capture_observation(observation)
        if not advanced:
            return False

        if self.flow.is_complete:
            self._fit_completed_flow()
        return True

    def _fit_completed_flow(self) -> None:
        """Fit the calibration model from completed flow samples."""

        try:
            self.fit_result = self.model.fit(
                self.flow.all_samples(),
                self.screen_width,
                self.screen_height,
            )
        except ValueError as error:
            self.fit_result = None
            self.error_message = str(error)
            self.state = CalibrationSessionState.FAILED
            return
        self.error_message = None
        self.state = CalibrationSessionState.COMPLETE
