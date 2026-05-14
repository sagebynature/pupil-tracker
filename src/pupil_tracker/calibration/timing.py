"""Timed calibration target phase helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CalibrationPhase(Enum):
    """Timed calibration phase for one target."""

    SETTLING = "settling"
    CAPTURING = "capturing"
    REVIEWING = "reviewing"
    COMPLETE = "complete"


@dataclass(frozen=True)
class TimedCalibrationConfig:
    """Accuracy-first timing and sample thresholds for calibration."""

    settle_seconds: float = 1.0
    capture_seconds: float = 2.0
    min_samples_per_target: int = 20
    min_confidence: float = 0.60

    def __post_init__(self) -> None:
        if self.settle_seconds <= 0:
            msg = "settle_seconds must be positive"
            raise ValueError(msg)
        if self.capture_seconds <= 0:
            msg = "capture_seconds must be positive"
            raise ValueError(msg)
        if self.min_samples_per_target <= 0:
            msg = "min_samples_per_target must be positive"
            raise ValueError(msg)
        if not 0.0 <= self.min_confidence <= 1.0:
            msg = "min_confidence must be between 0 and 1"
            raise ValueError(msg)


@dataclass(frozen=True)
class TimedTargetState:
    """Current timed state for one calibration target."""

    phase: CalibrationPhase
    target_started_at: float
    capture_started_at: float | None
    accepted_count: int
    rejected_count: int
    progress: float


class TimedTargetTimer:
    """Pure phase calculator for one calibration target."""

    def __init__(self, config: TimedCalibrationConfig) -> None:
        self.config = config

    def start(self, *, now_seconds: float) -> TimedTargetState:
        """Start timing a target in settle phase."""

        return TimedTargetState(
            phase=CalibrationPhase.SETTLING,
            target_started_at=now_seconds,
            capture_started_at=None,
            accepted_count=0,
            rejected_count=0,
            progress=0.0,
        )

    def update(
        self,
        state: TimedTargetState,
        *,
        now_seconds: float,
        accepted_count: int | None = None,
        rejected_count: int | None = None,
    ) -> TimedTargetState:
        """Return updated phase/progress for a target at `now_seconds`."""

        accepted = state.accepted_count if accepted_count is None else accepted_count
        rejected = state.rejected_count if rejected_count is None else rejected_count
        if state.phase is CalibrationPhase.SETTLING:
            elapsed = now_seconds - state.target_started_at
            if elapsed < self.config.settle_seconds:
                return TimedTargetState(
                    phase=CalibrationPhase.SETTLING,
                    target_started_at=state.target_started_at,
                    capture_started_at=None,
                    accepted_count=accepted,
                    rejected_count=rejected,
                    progress=self._progress(elapsed, self.config.settle_seconds),
                )
            return TimedTargetState(
                phase=CalibrationPhase.CAPTURING,
                target_started_at=state.target_started_at,
                capture_started_at=now_seconds,
                accepted_count=accepted,
                rejected_count=rejected,
                progress=0.0,
            )
        if state.phase is CalibrationPhase.CAPTURING:
            capture_started_at = state.capture_started_at
            if capture_started_at is None:
                capture_started_at = now_seconds
            elapsed = now_seconds - capture_started_at
            progress = self._progress(elapsed, self.config.capture_seconds)
            phase = (
                CalibrationPhase.REVIEWING
                if elapsed >= self.config.capture_seconds
                else CalibrationPhase.CAPTURING
            )
            return TimedTargetState(
                phase=phase,
                target_started_at=state.target_started_at,
                capture_started_at=capture_started_at,
                accepted_count=accepted,
                rejected_count=rejected,
                progress=progress,
            )
        return TimedTargetState(
            phase=state.phase,
            target_started_at=state.target_started_at,
            capture_started_at=state.capture_started_at,
            accepted_count=accepted,
            rejected_count=rejected,
            progress=state.progress,
        )

    @staticmethod
    def _progress(elapsed_seconds: float, duration_seconds: float) -> float:
        return max(0.0, min(1.0, elapsed_seconds / duration_seconds))
