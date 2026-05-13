"""Filters for stabilizing gaze samples."""

from __future__ import annotations

from dataclasses import replace

from pupil_tracker.models import GazeSample


class EmaGazeSmoother:
    """Exponential moving average smoother for gaze samples."""

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        self._alpha = alpha
        self._last_valid: GazeSample | None = None

    def update(self, sample: GazeSample) -> GazeSample:
        """Update smoother state and return a smoothed sample."""

        if not sample.valid:
            if self._last_valid is None:
                return sample
            return replace(
                sample,
                x=self._last_valid.x,
                y=self._last_valid.y,
                confidence=0.0,
                valid=False,
            )

        if self._last_valid is None:
            self._last_valid = sample
            return sample

        smoothed = replace(
            sample,
            x=self._blend(previous=self._last_valid.x, current=sample.x),
            y=self._blend(previous=self._last_valid.y, current=sample.y),
            confidence=self._blend(
                previous=self._last_valid.confidence,
                current=sample.confidence,
            ),
        )
        self._last_valid = smoothed
        return smoothed

    def reset(self) -> None:
        """Clear all smoothing state."""

        self._last_valid = None

    def _blend(self, previous: float, current: float) -> float:
        return previous + (self._alpha * (current - previous))
