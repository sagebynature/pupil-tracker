"""Post-calibration validation targets and metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from statistics import median
from typing import Literal

from pupil_tracker.models import GazeSample

ValidationRecommendation = Literal["excellent", "good", "usable", "retry"]


@dataclass(frozen=True)
class ValidationTarget:
    """A normalized target used to validate fitted gaze calibration."""

    id: str
    x: float
    y: float


@dataclass(frozen=True)
class ValidationSample:
    """One predicted gaze sample captured for a known validation target."""

    target: ValidationTarget
    gaze_sample: GazeSample


@dataclass(frozen=True)
class ValidationMetrics:
    """Aggregate post-calibration validation error metrics."""

    sample_count: int
    mean_error_px: float
    median_error_px: float
    max_error_px: float
    per_target_error_px: Mapping[str, float]
    recommendation: ValidationRecommendation


def validation_pattern() -> tuple[ValidationTarget, ...]:
    """Return stable intermediate validation targets distinct from training corners."""

    return (
        ValidationTarget(id="v0", x=0.25, y=0.25),
        ValidationTarget(id="v1", x=0.75, y=0.25),
        ValidationTarget(id="v2", x=0.50, y=0.50),
        ValidationTarget(id="v3", x=0.25, y=0.75),
        ValidationTarget(id="v4", x=0.75, y=0.75),
    )


def compute_validation_metrics(
    samples: Sequence[ValidationSample],
    *,
    screen_width: float,
    screen_height: float,
) -> ValidationMetrics:
    """Compute validation gaze error against known normalized target positions."""

    if screen_width <= 0 or screen_height <= 0:
        msg = "screen dimensions must be positive"
        raise ValueError(msg)

    errors: list[float] = []
    errors_by_target: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if not sample.gaze_sample.valid:
            continue
        target_x = sample.target.x * screen_width
        target_y = sample.target.y * screen_height
        error_px = hypot(sample.gaze_sample.x - target_x, sample.gaze_sample.y - target_y)
        errors.append(error_px)
        errors_by_target[sample.target.id].append(error_px)

    if not errors:
        msg = "at least one valid validation sample is required"
        raise ValueError(msg)

    mean_error = sum(errors) / len(errors)
    return ValidationMetrics(
        sample_count=len(errors),
        mean_error_px=mean_error,
        median_error_px=median(errors),
        max_error_px=max(errors),
        per_target_error_px={
            target_id: sum(target_errors) / len(target_errors)
            for target_id, target_errors in errors_by_target.items()
        },
        recommendation=_recommend(mean_error),
    )


def _recommend(mean_error_px: float) -> ValidationRecommendation:
    if mean_error_px < 75.0:
        return "excellent"
    if mean_error_px < 125.0:
        return "good"
    if mean_error_px < 200.0:
        return "usable"
    return "retry"
