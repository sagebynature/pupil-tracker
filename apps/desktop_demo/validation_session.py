"""Post-calibration validation session controller."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from time import monotonic

from pupil_tracker.calibration import (
    TimedCalibrationConfig,
    TimedTargetState,
    TimedTargetTimer,
    ValidationMetrics,
    ValidationSample,
    ValidationTarget,
    compute_validation_metrics,
)
from pupil_tracker.models import GazeSample


class ValidationSessionState(Enum):
    """Lifecycle states for validation capture."""

    IDLE = "idle"
    SETTLING = "settling"
    CAPTURING = "capturing"
    COMPLETE = "complete"


class ValidationSession:
    """Collect predicted gaze against known targets and compute validation metrics."""

    def __init__(
        self,
        *,
        targets: Sequence[ValidationTarget],
        screen_width: float,
        screen_height: float,
        timing_config: TimedCalibrationConfig,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not targets:
            msg = "at least one validation target is required"
            raise ValueError(msg)
        if screen_width <= 0 or screen_height <= 0:
            msg = "screen dimensions must be positive"
            raise ValueError(msg)
        self.targets = tuple(targets)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.timing_config = timing_config
        self.clock = clock if clock is not None else monotonic
        self.state = ValidationSessionState.IDLE
        self.current_index = 0
        self.accepted_for_current_target = 0
        self.capture_progress = 0.0
        self.metrics: ValidationMetrics | None = None
        self._samples: list[ValidationSample] = []
        self._timer = TimedTargetTimer(timing_config)
        self._timing_state: TimedTargetState | None = None

    @property
    def current_target(self) -> ValidationTarget | None:
        """Return the active validation target."""

        if self.current_index >= len(self.targets):
            return None
        return self.targets[self.current_index]

    def start(self) -> None:
        """Start validation at the first target."""

        self.current_index = 0
        self.accepted_for_current_target = 0
        self.capture_progress = 0.0
        self.metrics = None
        self._samples.clear()
        self._restart_target()

    def capture(self, gaze_sample: GazeSample) -> bool:
        """Capture one gaze sample and return True when a target/session advances."""

        if self.state in {
            ValidationSessionState.IDLE,
            ValidationSessionState.COMPLETE,
        }:
            return False
        self._update_timing_state()
        if self._is_reviewing_target():
            return self._advance_or_complete()
        if self.state is ValidationSessionState.SETTLING:
            return False
        if self.state is not ValidationSessionState.CAPTURING:
            return False
        if gaze_sample.valid:
            target = self.current_target
            if target is not None:
                self._samples.append(ValidationSample(target=target, gaze_sample=gaze_sample))
                self.accepted_for_current_target += 1
        self._update_timing_state()
        if self._is_reviewing_target():
            return self._advance_or_complete()
        if self.state is ValidationSessionState.CAPTURING:
            return False
        return self._advance_or_complete()

    def samples_for_current_target(self) -> tuple[ValidationSample, ...]:
        """Return samples collected for the active validation target."""

        target = self.current_target
        if target is None:
            return ()
        return tuple(sample for sample in self._samples if sample.target.id == target.id)

    def _restart_target(self) -> None:
        self.accepted_for_current_target = 0
        self.capture_progress = 0.0
        self._timing_state = self._timer.start(now_seconds=self.clock())
        self.state = ValidationSessionState.SETTLING

    def _update_timing_state(self) -> None:
        if self._timing_state is None:
            return
        self._timing_state = self._timer.update(
            self._timing_state,
            now_seconds=self.clock(),
            accepted_count=self.accepted_for_current_target,
        )
        self.capture_progress = self._timing_state.progress
        if self._timing_state.phase.value == "settling":
            self.state = ValidationSessionState.SETTLING
        elif self._timing_state.phase.value == "capturing":
            self.state = ValidationSessionState.CAPTURING
        else:
            self.state = ValidationSessionState.COMPLETE

    def _is_reviewing_target(self) -> bool:
        return (
            self._timing_state is not None
            and self._timing_state.phase.value == "reviewing"
        )

    def _advance_or_complete(self) -> bool:
        self.current_index += 1
        if self.current_index >= len(self.targets):
            self._timing_state = None
            self.state = ValidationSessionState.COMPLETE
            self.metrics = compute_validation_metrics(
                self._samples,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
            )
            return True
        self._restart_target()
        return True
