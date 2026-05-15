"""Analyze scalar separability between top-left calibration cluster and v0 validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot, inf, sqrt
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

FEATURE_NAMES: tuple[str, ...] = (
    "left iris face-relative X",
    "left iris face-relative Y",
    "right iris face-relative X",
    "right iris face-relative Y",
    "iris midpoint face-relative X",
    "iris midpoint face-relative Y",
    "left iris eye-relative X",
    "left iris eye-relative Y",
    "right iris eye-relative X",
    "right iris eye-relative Y",
    "left eye aperture",
    "right eye aperture",
    "eye-relative Y midpoint",
    "left-right eye-relative Y delta",
    "face center X",
    "face center Y",
    "face width",
    "face height",
    "face aspect ratio",
    "interocular distance",
    "roll proxy",
    "yaw proxy",
    "pitch proxy",
)


@dataclass(frozen=True)
class RunRange:
    """Inclusive line range identifying one run inside a JSONL telemetry file."""

    start_line: int
    end_line: int
    label: str


@dataclass(frozen=True)
class FeatureRow:
    """One scalar feature-vector row for a calibration or validation target."""

    target_id: str
    target_x: float
    target_y: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class ValidationPredictionRow:
    """One validation prediction row with predicted grid-cell metadata."""

    target_id: str
    target_x: float
    target_y: float
    predicted_cell: str
    matches_target_cell: bool


@dataclass(frozen=True)
class FeatureDistribution:
    """Scalar distribution summary for a group of feature vectors."""

    sample_count: int
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    feature_min: tuple[float, ...]
    feature_max: tuple[float, ...]


@dataclass(frozen=True)
class FeatureSeparabilityDelta:
    """Difference between validation and calibration feature distributions."""

    feature_index: int
    feature_name: str
    calibration_mean: float
    validation_mean: float
    signed_delta: float
    pooled_std: float
    normalized_delta: float
    calibration_range: tuple[float, float]
    validation_range: tuple[float, float]


@dataclass(frozen=True)
class TopLeftSeparabilityAnalysis:
    """Top-left calibration-cluster versus held-out v0 validation analysis."""

    label: str
    start_line: int
    end_line: int
    cluster_center: tuple[float, float]
    cluster_radius: float
    calibration_target_ids: tuple[str, ...]
    validation_target_id: str
    calibration_summary: FeatureDistribution
    validation_summary: FeatureDistribution
    dominant_feature_deltas: tuple[FeatureSeparabilityDelta, ...]
    validation_predicted_cell_counts: Mapping[str, int]
    validation_grid_accuracy: float
    assessment: str


class _TargetedRow(Protocol):
    @property
    def target_id(self) -> str: ...


TargetedRowT = TypeVar("TargetedRowT", bound=_TargetedRow)


def parse_run_range(value: str, *, label: str | None = None) -> RunRange:
    """Parse an inclusive `start:end` line range."""

    parts = value.split(":", maxsplit=1)
    if len(parts) != 2:
        msg = "run range must use START:END"
        raise ValueError(msg)
    start_line = _parse_positive_int(parts[0], field="start line")
    end_line = _parse_positive_int(parts[1], field="end line")
    if end_line < start_line:
        msg = "run range end must be greater than or equal to start"
        raise ValueError(msg)
    return RunRange(
        start_line=start_line,
        end_line=end_line,
        label=label if label is not None else f"lines {start_line}-{end_line}",
    )


def analyze_top_left_separability_log(
    path: Path,
    *,
    run_range: RunRange,
    screen_width: float,
    screen_height: float,
    grid_columns: int = 4,
    grid_rows: int = 3,
    cluster_center: tuple[float, float] = (0.25, 0.25),
    cluster_radius: float = 0.10,
    validation_target_id: str = "v0",
    dominant_feature_count: int = 8,
) -> TopLeftSeparabilityAnalysis:
    """Compare top-left cluster calibration features with held-out v0 validation."""

    if screen_width <= 0 or screen_height <= 0:
        msg = "screen dimensions must be positive"
        raise ValueError(msg)
    if grid_columns <= 0 or grid_rows <= 0:
        msg = "grid dimensions must be positive"
        raise ValueError(msg)
    if cluster_radius <= 0:
        msg = "cluster radius must be positive"
        raise ValueError(msg)

    calibration_rows: list[FeatureRow] = []
    validation_feature_rows: list[FeatureRow] = []
    validation_prediction_rows: list[ValidationPredictionRow] = []
    metrics_window: tuple[int, tuple[str, ...]] | None = None

    with path.open(encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if line_number < run_range.start_line:
                continue
            if line_number > run_range.end_line:
                break
            event = _parse_event(line, line_number=line_number)
            event_type = event.get("event_type")
            raw_payload = event.get("payload")
            if not isinstance(raw_payload, Mapping):
                continue
            payload = cast(Mapping[object, object], raw_payload)
            if event_type == "calibration_replay_sample":
                row = _parse_feature_row(payload, line_number=line_number)
                if row is None or not _calibration_sample_is_accepted(payload):
                    continue
                if _is_inside_cluster(row, center=cluster_center, radius=cluster_radius):
                    calibration_rows.append(row)
                continue
            if event_type == "validation_replay_sample":
                row = _parse_feature_row(payload, line_number=line_number)
                if row is not None:
                    validation_feature_rows.append(row)
                continue
            if event_type == "validation_sample":
                row = _parse_validation_prediction_row(
                    payload,
                    line_number=line_number,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    grid_columns=grid_columns,
                    grid_rows=grid_rows,
                )
                if row is not None:
                    validation_prediction_rows.append(row)
                continue
            if event_type == "validation_metrics":
                sample_count = _as_int(payload.get("sample_count"), field="sample_count")
                metrics_window = (sample_count, _metrics_target_ids(payload))

    if not calibration_rows:
        msg = "no accepted calibration replay samples found inside top-left cluster"
        raise ValueError(msg)
    selected_validation_features = [
        row
        for row in _select_metrics_sample_window(validation_feature_rows, metrics_window)
        if row.target_id == validation_target_id
    ]
    selected_validation_predictions = [
        row
        for row in _select_metrics_sample_window(validation_prediction_rows, metrics_window)
        if row.target_id == validation_target_id
    ]
    if not selected_validation_features:
        msg = f"no validation replay samples found for {validation_target_id!r}"
        raise ValueError(msg)
    if len(selected_validation_predictions) != len(selected_validation_features):
        msg = (
            f"validation prediction/sample count mismatch for {validation_target_id!r}: "
            f"{len(selected_validation_predictions)} predictions vs "
            f"{len(selected_validation_features)} feature rows"
        )
        raise ValueError(msg)

    calibration_summary = _summarize_features(calibration_rows)
    validation_summary = _summarize_features(selected_validation_features)
    deltas = _feature_deltas(calibration_summary, validation_summary)
    predicted_cell_counts = Counter(row.predicted_cell for row in selected_validation_predictions)
    validation_grid_accuracy = sum(
        1 for row in selected_validation_predictions if row.matches_target_cell
    ) / len(selected_validation_predictions)

    return TopLeftSeparabilityAnalysis(
        label=run_range.label,
        start_line=run_range.start_line,
        end_line=run_range.end_line,
        cluster_center=cluster_center,
        cluster_radius=cluster_radius,
        calibration_target_ids=tuple(sorted({row.target_id for row in calibration_rows})),
        validation_target_id=validation_target_id,
        calibration_summary=calibration_summary,
        validation_summary=validation_summary,
        dominant_feature_deltas=tuple(
            sorted(
                deltas,
                key=lambda delta: (
                    -abs(delta.normalized_delta),
                    -abs(delta.signed_delta),
                    delta.feature_index,
                ),
            )[:dominant_feature_count]
        ),
        validation_predicted_cell_counts=dict(
            sorted(predicted_cell_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        validation_grid_accuracy=validation_grid_accuracy,
        assessment=_assess_separability(
            calibration_summary,
            validation_summary,
            deltas,
        ),
    )


def format_top_left_separability_report(analysis: TopLeftSeparabilityAnalysis) -> str:
    """Format top-left separability diagnostics as compact Markdown."""

    lines = [
        "## Top-left separability",
        "",
        f"run: {analysis.label} ({analysis.start_line}:{analysis.end_line})",
        f"cluster_center: ({analysis.cluster_center[0]:.3f}, {analysis.cluster_center[1]:.3f})",
        f"cluster_radius: {analysis.cluster_radius:.3f}",
        f"calibration_cluster_samples: {analysis.calibration_summary.sample_count}",
        "calibration_targets: " + ", ".join(analysis.calibration_target_ids),
        f"validation_target: {analysis.validation_target_id}",
        f"validation_samples: {analysis.validation_summary.sample_count}",
        f"validation_grid_accuracy: {analysis.validation_grid_accuracy:.1%}",
        "validation_predicted_cells: "
        + _format_counts(analysis.validation_predicted_cell_counts),
        f"assessment: {analysis.assessment}",
        "",
        "| Feature | Calibration Mean | Validation Mean | Signed Δ | Normalized Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for delta in analysis.dominant_feature_deltas:
        lines.append(
            "| "
            f"{delta.feature_index} {delta.feature_name} | "
            f"{delta.calibration_mean:.6f} | "
            f"{delta.validation_mean:.6f} | "
            f"{delta.signed_delta:+.6f} | "
            f"{delta.normalized_delta:+.2f} |"
        )
    return "\n".join(lines)


def _parse_event(line: str, *, line_number: int) -> Mapping[str, Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as exc:
        msg = f"line {line_number}: invalid JSON"
        raise ValueError(msg) from exc
    if not isinstance(event, Mapping):
        msg = f"line {line_number}: event must be an object"
        raise ValueError(msg)
    return event


def _parse_feature_row(
    payload: Mapping[object, object],
    *,
    line_number: int,
) -> FeatureRow | None:
    if not _as_bool(payload.get("valid"), field="valid"):
        return None
    features = _parse_float_vector(payload.get("features"), field="features")
    if not features:
        return None
    feature_count = _as_int(payload.get("feature_count"), field="feature_count")
    if feature_count != len(features):
        msg = f"line {line_number}: feature_count does not match features length"
        raise ValueError(msg)
    return FeatureRow(
        target_id=str(payload.get("target_id", "")),
        target_x=_as_float(payload.get("target_x"), field="target_x"),
        target_y=_as_float(payload.get("target_y"), field="target_y"),
        features=features,
    )


def _parse_validation_prediction_row(
    payload: Mapping[object, object],
    *,
    line_number: int,
    screen_width: float,
    screen_height: float,
    grid_columns: int,
    grid_rows: int,
) -> ValidationPredictionRow | None:
    if not _as_bool(payload.get("valid"), field="valid"):
        return None
    target_x = _as_float(payload.get("target_x"), field="target_x")
    target_y = _as_float(payload.get("target_y"), field="target_y")
    x = _as_float(payload.get("x"), field="x")
    y = _as_float(payload.get("y"), field="y")
    expected_cell = _grid_cell(
        target_x * screen_width,
        target_y * screen_height,
        screen_width=screen_width,
        screen_height=screen_height,
        grid_columns=grid_columns,
        grid_rows=grid_rows,
    )
    predicted_cell = _grid_cell(
        x,
        y,
        screen_width=screen_width,
        screen_height=screen_height,
        grid_columns=grid_columns,
        grid_rows=grid_rows,
    )
    if not predicted_cell:
        msg = f"line {line_number}: could not compute predicted grid cell"
        raise ValueError(msg)
    return ValidationPredictionRow(
        target_id=str(payload.get("target_id", "")),
        target_x=target_x,
        target_y=target_y,
        predicted_cell=predicted_cell,
        matches_target_cell=predicted_cell == expected_cell,
    )


def _calibration_sample_is_accepted(payload: Mapping[object, object]) -> bool:
    sample_accepted = payload.get("sample_accepted")
    if sample_accepted is None:
        return True
    return _as_bool(sample_accepted, field="sample_accepted")


def _is_inside_cluster(
    row: FeatureRow,
    *,
    center: tuple[float, float],
    radius: float,
) -> bool:
    return hypot(row.target_x - center[0], row.target_y - center[1]) <= radius


def _select_metrics_sample_window(
    rows: Sequence[TargetedRowT],
    metrics_window: tuple[int, tuple[str, ...]] | None,
) -> tuple[TargetedRowT, ...]:
    if metrics_window is None:
        return tuple(rows)
    sample_count, target_ids = metrics_window
    if sample_count <= 0:
        msg = "validation metrics sample_count must be positive"
        raise ValueError(msg)
    if not target_ids:
        return tuple(rows[-sample_count:])
    per_target_count = sample_count // len(target_ids)
    if per_target_count <= 0:
        return tuple(rows[-sample_count:])
    selected: list[TargetedRowT] = []
    for target_id in target_ids:
        target_rows = [row for row in rows if row.target_id == target_id]
        selected.extend(target_rows[-per_target_count:])
    return tuple(selected)


def _metrics_target_ids(payload: Mapping[object, object]) -> tuple[str, ...]:
    for field in (
        "per_target_grid_cell_accuracy",
        "per_target_error_px",
        "per_target_signed_y_error_px",
    ):
        raw_value = payload.get(field)
        if isinstance(raw_value, Mapping):
            return tuple(str(target_id) for target_id in raw_value)
    return ()


def _summarize_features(rows: Sequence[FeatureRow]) -> FeatureDistribution:
    if not rows:
        msg = "feature rows must not be empty"
        raise ValueError(msg)
    feature_count = len(rows[0].features)
    for row in rows:
        if len(row.features) != feature_count:
            msg = "feature rows have inconsistent lengths"
            raise ValueError(msg)
    means = tuple(
        sum(row.features[index] for row in rows) / len(rows)
        for index in range(feature_count)
    )
    return FeatureDistribution(
        sample_count=len(rows),
        feature_mean=means,
        feature_std=tuple(
            sqrt(sum((row.features[index] - mean) ** 2 for row in rows) / len(rows))
            for index, mean in enumerate(means)
        ),
        feature_min=tuple(
            min(row.features[index] for row in rows) for index in range(feature_count)
        ),
        feature_max=tuple(
            max(row.features[index] for row in rows) for index in range(feature_count)
        ),
    )


def _feature_deltas(
    calibration: FeatureDistribution,
    validation: FeatureDistribution,
) -> tuple[FeatureSeparabilityDelta, ...]:
    if len(calibration.feature_mean) != len(validation.feature_mean):
        msg = "calibration and validation feature counts differ"
        raise ValueError(msg)
    deltas: list[FeatureSeparabilityDelta] = []
    for index, (calibration_mean, validation_mean) in enumerate(
        zip(calibration.feature_mean, validation.feature_mean, strict=True)
    ):
        signed_delta = validation_mean - calibration_mean
        pooled_std = sqrt(
            calibration.feature_std[index] ** 2 + validation.feature_std[index] ** 2
        )
        if pooled_std == 0.0:
            normalized_delta = inf if signed_delta else 0.0
        else:
            normalized_delta = signed_delta / pooled_std
        deltas.append(
            FeatureSeparabilityDelta(
                feature_index=index,
                feature_name=_feature_name(index),
                calibration_mean=calibration_mean,
                validation_mean=validation_mean,
                signed_delta=signed_delta,
                pooled_std=pooled_std,
                normalized_delta=normalized_delta,
                calibration_range=(
                    calibration.feature_min[index],
                    calibration.feature_max[index],
                ),
                validation_range=(
                    validation.feature_min[index],
                    validation.feature_max[index],
                ),
            )
        )
    return tuple(deltas)


def _assess_separability(
    calibration: FeatureDistribution,
    validation: FeatureDistribution,
    deltas: Sequence[FeatureSeparabilityDelta],
) -> str:
    if calibration.sample_count < 2 or validation.sample_count < 2:
        return "inconclusive"
    max_normalized_delta = max(abs(delta.normalized_delta) for delta in deltas)
    if max_normalized_delta >= 2.0:
        return "separable"
    return "overlapping"


def _grid_cell(
    x: float,
    y: float,
    *,
    screen_width: float,
    screen_height: float,
    grid_columns: int,
    grid_rows: int,
) -> str:
    clamped_x = min(max(x, 0.0), screen_width - 1e-9)
    clamped_y = min(max(y, 0.0), screen_height - 1e-9)
    column = int(clamped_x / (screen_width / grid_columns))
    row = int(clamped_y / (screen_height / grid_rows))
    return f"r{row}c{column}"


def _feature_name(index: int) -> str:
    if index < len(FEATURE_NAMES):
        return FEATURE_NAMES[index]
    return f"feature {index}"


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(
        f"{cell}={count}"
        for cell, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _parse_float_vector(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"{field} must be a numeric sequence"
        raise ValueError(msg)
    return tuple(_as_float(item, field=field) for item in value)


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, str | int | float):
        return float(value)
    msg = f"{field} must be numeric"
    raise ValueError(msg)


def _as_int(value: object, *, field: str) -> int:
    if isinstance(value, str | int):
        return int(value)
    msg = f"{field} must be an integer"
    raise ValueError(msg)


def _as_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    msg = f"{field} must be a boolean"
    raise ValueError(msg)


def _parse_positive_int(value: str, *, field: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        msg = f"{field} must be an integer"
        raise ValueError(msg) from exc
    if number <= 0:
        msg = f"{field} must be positive"
        raise ValueError(msg)
    return number


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", type=Path, help="Path to metrics/demo.jsonl")
    parser.add_argument("--run", required=True, help="Inclusive run range START:END")
    parser.add_argument("--screen-width", required=True, type=float)
    parser.add_argument("--screen-height", required=True, type=float)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--cluster-x", type=float, default=0.25)
    parser.add_argument("--cluster-y", type=float, default=0.25)
    parser.add_argument("--cluster-radius", type=float, default=0.10)
    parser.add_argument("--validation-target", default="v0")
    args = parser.parse_args()

    analysis = analyze_top_left_separability_log(
        args.jsonl_path,
        run_range=parse_run_range(args.run),
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        grid_columns=args.grid_columns,
        grid_rows=args.grid_rows,
        cluster_center=(args.cluster_x, args.cluster_y),
        cluster_radius=args.cluster_radius,
        validation_target_id=args.validation_target,
    )
    print(format_top_left_separability_report(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
