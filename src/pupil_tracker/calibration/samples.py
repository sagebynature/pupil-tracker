"""Calibration sample collection helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping

from pupil_tracker.models import CalibrationSample


class CalibrationSampleCollector:
    """Collect valid calibration samples while preserving insertion order."""

    def __init__(self) -> None:
        self._samples: list[CalibrationSample] = []
        self._samples_by_target: MutableMapping[str, list[CalibrationSample]] = defaultdict(list)

    def add(self, sample: CalibrationSample) -> bool:
        """Add `sample` if its observation is valid.

        Returns `True` when the sample is stored and `False` when it is skipped.
        """

        if not sample.observation.valid:
            return False

        self._samples.append(sample)
        self._samples_by_target[sample.target.id].append(sample)
        return True

    def samples_for(self, target_id: str) -> tuple[CalibrationSample, ...]:
        """Return valid samples collected for `target_id` in insertion order."""

        return tuple(self._samples_by_target[target_id])

    def all_samples(self) -> tuple[CalibrationSample, ...]:
        """Return all valid samples in insertion order."""

        return tuple(self._samples)

    def clear(self) -> None:
        """Remove every collected sample."""

        self._samples.clear()
        self._samples_by_target.clear()
