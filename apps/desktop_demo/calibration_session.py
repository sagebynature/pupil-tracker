"""Live calibration session controller for the desktop demo."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from time import monotonic
from typing import Literal, Protocol

from pupil_tracker.calibration import (
    CalibrationFitResult,
    CalibrationPhase,
    CalibrationQualityFilter,
    TargetQualitySummary,
    TimedCalibrationConfig,
    TimedTargetState,
    TimedTargetTimer,
    summarize_target_quality,
)
from pupil_tracker.models import CalibrationSample, CalibrationTarget, RawObservation

CalibrationSampleWindow = Literal["all", "early", "middle", "late"]


def select_calibration_samples_by_window(
    samples: Sequence[CalibrationSample],
    *,
    window: CalibrationSampleWindow,
) -> tuple[CalibrationSample, ...]:
    """Select the same capture window from each calibration target."""

    if window == "all":
        return tuple(samples)

    indices_by_target: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        indices_by_target.setdefault(sample.target.id, []).append(index)

    selected_indices: set[int] = set()
    for indices in indices_by_target.values():
        start, stop = _sample_window_bounds(len(indices), window=window)
        selected_indices.update(indices[start:stop])

    return tuple(
        sample for index, sample in enumerate(samples) if index in selected_indices
    )


def _sample_window_bounds(
    sample_count: int,
    *,
    window: CalibrationSampleWindow,
) -> tuple[int, int]:
    if sample_count <= 0:
        return (0, 0)
    window_size = max(1, sample_count // 3)
    if window == "early":
        return (0, window_size)
    if window == "middle":
        start = (sample_count - window_size) // 2
        return (start, start + window_size)
    if window == "late":
        return (sample_count - window_size, sample_count)
    return (0, sample_count)


class CalibrationFlowLike(Protocol):
    """Calibration flow surface used by the session controller."""

    @property
    def is_complete(self) -> bool:
        """Return whether all calibration targets have enough samples."""

    @property
    def current_target(self) -> CalibrationTarget | None:
        """Return the active calibration target, if any."""

    def reset(self) -> None:
        """Reset collected samples and return to the first target."""
        ...

    def capture_observation(self, observation: RawObservation) -> bool:
        """Capture one observation and return whether the flow advanced."""
        ...

    def add_current_target_sample(self, observation: RawObservation) -> bool:
        """Store one observation for the active target without advancing."""
        ...

    def advance_target(self) -> bool:
        """Advance to the next target."""
        ...

    def clear_current_target_samples(self) -> None:
        """Clear active target samples."""
        ...

    def samples_for_current_target(self) -> tuple[CalibrationSample, ...]:
        """Return samples for the active target."""
        ...

    def all_samples(self) -> tuple[CalibrationSample, ...]:
        """Return every valid collected calibration sample."""
        ...


class CalibrationModelLike(Protocol):
    """Calibration model surface used by the desktop calibration session."""

    def fit(
        self,
        samples: tuple[CalibrationSample, ...],
        screen_width: float,
        screen_height: float,
    ) -> CalibrationFitResult:
        """Fit from collected samples and return fit metrics."""
        ...


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
        flow: CalibrationFlowLike,
        model: CalibrationModelLike,
        screen_width: float,
        screen_height: float,
        timing_config: TimedCalibrationConfig | None = None,
        clock: Callable[[], float] | None = None,
        quality_filter: CalibrationQualityFilter | None = None,
        calibration_sample_window: CalibrationSampleWindow = "all",
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
        self.timing_config = timing_config
        self.clock = clock if clock is not None else monotonic
        self.quality_filter = quality_filter
        self.calibration_sample_window: CalibrationSampleWindow = calibration_sample_window
        if self.timing_config is not None and self.quality_filter is None:
            self.quality_filter = CalibrationQualityFilter(
                min_confidence=self.timing_config.min_confidence
            )
        self._target_timer = (
            TimedTargetTimer(self.timing_config)
            if self.timing_config is not None
            else None
        )
        self._target_timing_state: TimedTargetState | None = None
        self.phase = CalibrationPhase.COMPLETE
        self.target_quality: TargetQualitySummary | None = None
        self.accepted_for_current_target = 0
        self.rejected_for_current_target = 0
        self.capture_progress = 0.0

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
        self.target_quality = None
        if self._target_timer is not None:
            self._restart_timed_target(clear_quality=True)
        else:
            self.phase = CalibrationPhase.COMPLETE
            self._target_timing_state = None
            self.accepted_for_current_target = 0
            self.rejected_for_current_target = 0
            self.capture_progress = 0.0

    def capture(self, observation: RawObservation) -> bool:
        """Capture an observation when collecting.

        Returns True when capture advances the flow or completes calibration.
        """

        if self.state is not CalibrationSessionState.COLLECTING:
            return False

        if self._target_timer is not None:
            return self._capture_timed(observation)

        advanced = self.flow.capture_observation(observation)
        if not advanced:
            return False

        if self.flow.is_complete:
            self._fit_completed_flow()
        return True

    def _capture_timed(self, observation: RawObservation) -> bool:
        """Capture an observation with timed, quality-gated target windows."""

        if self._target_timer is None or self.timing_config is None:
            return False
        if self._target_timing_state is None:
            self._restart_timed_target(clear_quality=True)

        self._update_timed_state()
        if self.phase is CalibrationPhase.SETTLING:
            return False
        if self.phase is CalibrationPhase.REVIEWING:
            return self._review_timed_target()
        if self.phase is not CalibrationPhase.CAPTURING:
            return False

        quality_filter = self.quality_filter
        if quality_filter is None:
            return False
        current_samples = self.flow.samples_for_current_target()
        reference_features = (
            current_samples[0].observation.feature_vector if current_samples else None
        )
        decision = quality_filter.decide(
            observation,
            reference_features=reference_features,
        )
        if decision.accepted and self.flow.add_current_target_sample(observation):
            self.accepted_for_current_target += 1
        else:
            self.rejected_for_current_target += 1

        self._update_timed_state()
        if self.phase is CalibrationPhase.REVIEWING:
            return self._review_timed_target()
        return False

    def _update_timed_state(self) -> None:
        """Refresh public timed phase/progress fields from the fakeable clock."""

        if self._target_timer is None or self._target_timing_state is None:
            return
        self._target_timing_state = self._target_timer.update(
            self._target_timing_state,
            now_seconds=self.clock(),
            accepted_count=self.accepted_for_current_target,
            rejected_count=self.rejected_for_current_target,
        )
        self.phase = self._target_timing_state.phase
        self.capture_progress = self._target_timing_state.progress

    def _review_timed_target(self) -> bool:
        """Advance or retry the active target based on capture quality."""

        if self.timing_config is None:
            return False
        target = self.flow.current_target
        if target is None:
            return False
        self.target_quality = summarize_target_quality(
            target_id=target.id,
            accepted_observations=tuple(
                sample.observation for sample in self.flow.samples_for_current_target()
            ),
            rejected_count=self.rejected_for_current_target,
            min_samples=self.timing_config.min_samples_per_target,
        )
        if self.target_quality.recommendation == "retry":
            self.flow.clear_current_target_samples()
            self._restart_timed_target(clear_quality=False)
            return True

        self.flow.advance_target()
        if self.flow.is_complete:
            self.phase = CalibrationPhase.COMPLETE
            self.capture_progress = 1.0
            self._target_timing_state = None
            self._fit_completed_flow()
            return True
        self._restart_timed_target(clear_quality=False)
        return True

    def _restart_timed_target(self, *, clear_quality: bool) -> None:
        """Reset timed counters for the current target."""

        self.accepted_for_current_target = 0
        self.rejected_for_current_target = 0
        self.capture_progress = 0.0
        if clear_quality:
            self.target_quality = None
        if self._target_timer is None or self.flow.current_target is None:
            self.phase = CalibrationPhase.COMPLETE
            self._target_timing_state = None
            return
        self._target_timing_state = self._target_timer.start(now_seconds=self.clock())
        self.phase = self._target_timing_state.phase

    def _fit_completed_flow(self) -> None:
        """Fit the calibration model from completed flow samples."""

        try:
            samples = select_calibration_samples_by_window(
                self.flow.all_samples(),
                window=self.calibration_sample_window,
            )
            self.fit_result = self.model.fit(
                samples,
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
