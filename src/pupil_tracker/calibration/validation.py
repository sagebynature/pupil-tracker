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
    mean_abs_x_error_px: float
    mean_abs_y_error_px: float
    mean_signed_y_error_px: float
    per_target_signed_y_error_px: Mapping[str, float]
    grid_cell_accuracy: float
    per_target_grid_cell_accuracy: Mapping[str, float]
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
    abs_x_errors: list[float] = []
    abs_y_errors: list[float] = []
    signed_y_errors: list[float] = []
    grid_cell_correct: list[bool] = []
    errors_by_target: dict[str, list[float]] = defaultdict(list)
    signed_y_errors_by_target: dict[str, list[float]] = defaultdict(list)
    grid_cell_correct_by_target: dict[str, list[bool]] = defaultdict(list)
    for sample in samples:
        if not sample.gaze_sample.valid:
            continue
        target_x = sample.target.x * screen_width
        target_y = sample.target.y * screen_height
        dx = sample.gaze_sample.x - target_x
        dy = sample.gaze_sample.y - target_y
        error_px = hypot(dx, dy)
        errors.append(error_px)
        abs_x_errors.append(abs(dx))
        abs_y_errors.append(abs(dy))
        signed_y_errors.append(dy)
        errors_by_target[sample.target.id].append(error_px)
        signed_y_errors_by_target[sample.target.id].append(dy)
        target_cell = _grid_cell_id(target_x, target_y, screen_width, screen_height)
        gaze_cell = _grid_cell_id(
            sample.gaze_sample.x,
            sample.gaze_sample.y,
            screen_width,
            screen_height,
        )
        matches_target_cell = target_cell == gaze_cell
        grid_cell_correct.append(matches_target_cell)
        grid_cell_correct_by_target[sample.target.id].append(matches_target_cell)

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
        mean_abs_x_error_px=sum(abs_x_errors) / len(abs_x_errors),
        mean_abs_y_error_px=sum(abs_y_errors) / len(abs_y_errors),
        mean_signed_y_error_px=sum(signed_y_errors) / len(signed_y_errors),
        per_target_signed_y_error_px={
            target_id: sum(target_errors) / len(target_errors)
            for target_id, target_errors in signed_y_errors_by_target.items()
        },
        grid_cell_accuracy=_fraction_true(grid_cell_correct),
        per_target_grid_cell_accuracy={
            target_id: _fraction_true(target_matches)
            for target_id, target_matches in grid_cell_correct_by_target.items()
        },
        recommendation=_recommend(mean_error),
    )


def _grid_cell_id(
    x: float,
    y: float,
    screen_width: float,
    screen_height: float,
    *,
    columns: int = 3,
    rows: int = 3,
) -> str:
    clamped_x = min(max(x, 0.0), screen_width)
    clamped_y = min(max(y, 0.0), screen_height)
    column = min(int(clamped_x / (screen_width / columns)), columns - 1)
    row = min(int(clamped_y / (screen_height / rows)), rows - 1)
    return f"r{row}c{column}"


def _fraction_true(values: Sequence[bool]) -> float:
    return sum(1 for value in values if value) / len(values)


def _recommend(mean_error_px: float) -> ValidationRecommendation:
    if mean_error_px < 75.0:
        return "excellent"
    if mean_error_px < 125.0:
        return "good"
    if mean_error_px < 200.0:
        return "usable"
    return "retry"
