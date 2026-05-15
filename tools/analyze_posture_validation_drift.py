"""Analyze scalar posture drift between accepted calibration and validation samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import hypot, inf, isinf, sqrt
from pathlib import Path
from typing import NoReturn, Protocol, TypeVar, cast

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
POSTURE_FEATURE_INDICES: tuple[int, ...] = (20, 21, 22)
CONTEXT_FEATURE_INDICES: tuple[int, ...] = (20, 21, 22, 14, 15, 16, 17, 18)
MATERIAL_SIGNED_DELTA_THRESHOLD = 0.005
MATERIAL_NORMALIZED_DELTA_THRESHOLD = 2.0


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
    """Scalar feature distribution summary for a group of feature vectors."""

    sample_count: int
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]
    feature_min: tuple[float, ...]
    feature_max: tuple[float, ...]


@dataclass(frozen=True)
class FeatureDriftDelta:
    """Difference between validation and calibration feature distributions."""

    feature_index: int
    feature_name: str
    calibration_mean: float
    calibration_std: float
    calibration_range: tuple[float, float]
    validation_mean: float
    validation_std: float
    validation_range: tuple[float, float]
    signed_delta: float
    normalized_delta: float


@dataclass(frozen=True)
class TargetPostureValidationDrift:
    """Calibration-vs-validation drift for one validation target."""

    target_id: str
    target_x: float
    target_y: float
    nearest_calibration_target_id: str
    calibration_target_ids: tuple[str, ...]
    calibration_summary: FeatureDistribution
    validation_summary: FeatureDistribution
    validation_grid_accuracy: float
    predicted_cell_counts: Mapping[str, int]
    context_feature_deltas: tuple[FeatureDriftDelta, ...]
    dominant_feature_deltas: tuple[FeatureDriftDelta, ...]
    flags: tuple[str, ...]


@dataclass(frozen=True)
class PostureValidationDriftAnalysis:
    """All scalar posture-vs-validation drift diagnostics for one run."""

    label: str
    start_line: int
    end_line: int
    targets: Mapping[str, TargetPostureValidationDrift]
    screen_width: float
    screen_height: float
    grid_columns: int
    grid_rows: int


class _TargetedRow(Protocol):
    @property
    def target_id(self) -> str: ...


TargetedRowT = TypeVar("TargetedRowT", bound=_TargetedRow)
MetricsWindow = tuple[int, int, int, tuple[str, ...]]


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


def analyze_posture_validation_drift_log(
    path: Path,
    *,
    run_range: RunRange,
    screen_width: float,
    screen_height: float,
    grid_columns: int = 4,
    grid_rows: int = 3,
    dominant_feature_count: int = 6,
) -> PostureValidationDriftAnalysis:
    """Compare accepted calibration features against validation replay windows."""

    if screen_width <= 0 or screen_height <= 0:
        msg = "screen dimensions must be positive"
        raise ValueError(msg)
    if grid_columns <= 0 or grid_rows <= 0:
        msg = "grid dimensions must be positive"
        raise ValueError(msg)
    if dominant_feature_count <= 0:
        msg = "dominant feature count must be positive"
        raise ValueError(msg)

    calibration_rows: list[FeatureRow] = []
    validation_feature_rows: list[FeatureRow] = []
    validation_prediction_rows: list[ValidationPredictionRow] = []
    metrics_window: MetricsWindow | None = None

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
                if row is not None and _calibration_sample_is_accepted(payload):
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
                sample_count = _as_int(
                    payload.get("sample_count"), field="sample_count", line_number=line_number
                )
                metrics_window = (
                    len(validation_feature_rows),
                    len(validation_prediction_rows),
                    sample_count,
                    _metrics_target_ids(payload),
                )

    if not calibration_rows:
        msg = "no accepted calibration replay samples found"
        raise ValueError(msg)

    selected_validation_features = _select_metrics_sample_window(
        validation_feature_rows,
        metrics_window,
        row_count_index=0,
    )
    selected_validation_predictions = _select_metrics_sample_window(
        validation_prediction_rows,
        metrics_window,
        row_count_index=1,
    )
    if not selected_validation_predictions:
        msg = "no validation samples found in metrics window"
        raise ValueError(msg)

    calibration_rows_by_target = _group_by_target(calibration_rows)
    validation_features_by_target = _group_by_target(selected_validation_features)
    validation_predictions_by_target = _group_by_target(selected_validation_predictions)

    targets: dict[str, TargetPostureValidationDrift] = {}
    for target_id, prediction_rows in sorted(validation_predictions_by_target.items()):
        feature_rows = validation_features_by_target.get(target_id, [])
        if not feature_rows:
            msg = f"no validation replay samples found for {target_id!r}"
            raise ValueError(msg)
        if len(feature_rows) != len(prediction_rows):
            msg = (
                f"validation prediction/sample count mismatch for {target_id!r}: "
                f"{len(prediction_rows)} predictions vs {len(feature_rows)} feature rows"
            )
            raise ValueError(msg)
        targets[target_id] = _analyze_target_drift(
            target_id=target_id,
            validation_feature_rows=feature_rows,
            validation_prediction_rows=prediction_rows,
            calibration_rows_by_target=calibration_rows_by_target,
            dominant_feature_count=dominant_feature_count,
        )

    return PostureValidationDriftAnalysis(
        label=run_range.label,
        start_line=run_range.start_line,
        end_line=run_range.end_line,
        targets=targets,
        screen_width=screen_width,
        screen_height=screen_height,
        grid_columns=grid_columns,
        grid_rows=grid_rows,
    )


def format_posture_validation_drift_report(analysis: PostureValidationDriftAnalysis) -> str:
    """Format posture/validation drift analysis as scalar Markdown tables."""

    lines = [
        "## Posture/validation drift",
        "",
        f"run: {analysis.label}",
        f"lines: {analysis.start_line}-{analysis.end_line}",
        f"screen: {analysis.screen_width:.0f}x{analysis.screen_height:.0f}",
        f"grid: {analysis.grid_columns}x{analysis.grid_rows}",
        "",
        "| Target | Calibration Cluster | Cal Samples | Validation Samples | "
        "Grid Accuracy | Predicted Cells | Flags |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for target in analysis.targets.values():
        lines.append(
            "| "
            f"{target.target_id} | "
            f"{', '.join(target.calibration_target_ids)} | "
            f"{target.calibration_summary.sample_count} | "
            f"{target.validation_summary.sample_count} | "
            f"{target.validation_grid_accuracy:.1%} | "
            f"{_format_counts(target.predicted_cell_counts)} | "
            f"{_format_flags(target.flags)} |"
        )
    lines.extend(
        (
            "",
            "### Posture/context feature drift",
            "",
            "| Target | Feature | Cal Mean | Cal Std | Cal Range | Val Mean | Val Std | "
            "Val Range | Signed Δ | Normalized Δ |",
            "|---|---|---:|---:|---|---:|---:|---|---:|---:|",
        )
    )
    for target in analysis.targets.values():
        for delta in target.context_feature_deltas:
            lines.append(_format_feature_delta_row(target.target_id, delta))
    lines.extend(
        (
            "",
            "### Dominant scalar feature drift",
            "",
            "| Target | Feature | Cal Mean | Cal Std | Cal Range | Val Mean | Val Std | "
            "Val Range | Signed Δ | Normalized Δ |",
            "|---|---|---:|---:|---|---:|---:|---|---:|---:|",
        )
    )
    for target in analysis.targets.values():
        for delta in target.dominant_feature_deltas:
            lines.append(_format_feature_delta_row(target.target_id, delta))
    return "\n".join(lines)


def format_posture_validation_drift_run_comparison(
    analyses: Sequence[PostureValidationDriftAnalysis],
) -> str:
    """Format a compact scalar comparison of target outliers across runs."""

    if not analyses:
        msg = "at least one posture validation drift analysis is required"
        raise ValueError(msg)
    lines = [
        "## Posture/validation drift run comparison",
        "",
        "| Run | Lines | Target Outlier | Grid Accuracy | Predicted Cells | Flags | "
        "Top Context Drift |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for analysis in analyses:
        outliers = tuple(
            target for target in analysis.targets.values() if _is_target_outlier(target)
        )
        if not outliers:
            lines.append(
                f"| {analysis.label} | {analysis.start_line}-{analysis.end_line} | "
                "- | - | - | - | - |"
            )
            continue
        for target in outliers:
            lines.append(
                "| "
                f"{analysis.label} | "
                f"{analysis.start_line}-{analysis.end_line} | "
                f"{target.target_id} | "
                f"{target.validation_grid_accuracy:.1%} | "
                f"{_format_counts(target.predicted_cell_counts)} | "
                f"{_format_flags(target.flags)} | "
                f"{_format_top_context_drift(target.context_feature_deltas)} |"
            )
    return "\n".join(lines)


def _is_target_outlier(target: TargetPostureValidationDrift) -> bool:
    return (
        "posture-drift-grid-collapse" in target.flags
        or target.validation_grid_accuracy < 0.5
    )


def _format_top_context_drift(deltas: Sequence[FeatureDriftDelta]) -> str:
    selected = next((delta for delta in deltas if _is_material_drift(delta)), None)
    if selected is None and deltas:
        selected = deltas[0]
    if selected is None:
        return "-"
    return (
        f"{selected.feature_index} {selected.feature_name} "
        f"{selected.signed_delta:+.6f} "
        f"({_format_normalized_delta(selected.normalized_delta)})"
    )


def _analyze_target_drift(
    *,
    target_id: str,
    validation_feature_rows: Sequence[FeatureRow],
    validation_prediction_rows: Sequence[ValidationPredictionRow],
    calibration_rows_by_target: Mapping[str, Sequence[FeatureRow]],
    dominant_feature_count: int,
) -> TargetPostureValidationDrift:
    target_x = validation_feature_rows[0].target_x
    target_y = validation_feature_rows[0].target_y
    nearest_target_id, calibration_cluster_rows = _nearest_calibration_cluster(
        target_x,
        target_y,
        calibration_rows_by_target,
    )
    calibration_summary = _summarize_features(calibration_cluster_rows)
    validation_summary = _summarize_features(validation_feature_rows)
    deltas = _feature_deltas(calibration_summary, validation_summary)
    context_deltas = tuple(
        sorted(
            (delta for delta in deltas if delta.feature_index in CONTEXT_FEATURE_INDICES),
            key=_context_drift_sort_key,
        )
    )
    dominant_deltas = tuple(sorted(deltas, key=_drift_sort_key)[:dominant_feature_count])
    predicted_cell_counts = Counter(row.predicted_cell for row in validation_prediction_rows)
    validation_grid_accuracy = sum(
        1 for row in validation_prediction_rows if row.matches_target_cell
    ) / len(validation_prediction_rows)
    calibration_target_ids = tuple(sorted({row.target_id for row in calibration_cluster_rows}))
    flags = _target_flags(validation_grid_accuracy, context_deltas)
    return TargetPostureValidationDrift(
        target_id=target_id,
        target_x=target_x,
        target_y=target_y,
        nearest_calibration_target_id=nearest_target_id,
        calibration_target_ids=calibration_target_ids,
        calibration_summary=calibration_summary,
        validation_summary=validation_summary,
        validation_grid_accuracy=validation_grid_accuracy,
        predicted_cell_counts=dict(sorted(predicted_cell_counts.items())),
        context_feature_deltas=context_deltas,
        dominant_feature_deltas=dominant_deltas,
        flags=flags,
    )


def _nearest_calibration_cluster(
    target_x: float,
    target_y: float,
    rows_by_target: Mapping[str, Sequence[FeatureRow]],
) -> tuple[str, tuple[FeatureRow, ...]]:
    candidates: list[tuple[float, str, float, float]] = []
    for calibration_target_id, rows in rows_by_target.items():
        if not rows:
            continue
        row = rows[0]
        candidates.append(
            (
                hypot(row.target_x - target_x, row.target_y - target_y),
                calibration_target_id,
                row.target_x,
                row.target_y,
            )
        )
    if not candidates:
        msg = "no calibration target clusters found"
        raise ValueError(msg)
    _, nearest_target_id, nearest_x, nearest_y = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    cluster_rows = tuple(
        row
        for rows in rows_by_target.values()
        for row in rows
        if row.target_x == nearest_x and row.target_y == nearest_y
    )
    return nearest_target_id, cluster_rows


def _target_flags(
    validation_grid_accuracy: float,
    context_deltas: Sequence[FeatureDriftDelta],
) -> tuple[str, ...]:
    has_material_posture_drift = any(
        delta.feature_index in POSTURE_FEATURE_INDICES and _is_material_drift(delta)
        for delta in context_deltas
    )
    if not has_material_posture_drift:
        return ()
    flags: list[str] = []
    if validation_grid_accuracy == 0.0:
        flags.append("posture-drift-grid-collapse")
    flags.append("posture-drift")
    return tuple(flags)


def _is_material_drift(delta: FeatureDriftDelta) -> bool:
    return (
        abs(delta.signed_delta) >= MATERIAL_SIGNED_DELTA_THRESHOLD
        and abs(delta.normalized_delta) >= MATERIAL_NORMALIZED_DELTA_THRESHOLD
    )


def _feature_deltas(
    calibration: FeatureDistribution,
    validation: FeatureDistribution,
) -> tuple[FeatureDriftDelta, ...]:
    if len(calibration.feature_mean) != len(validation.feature_mean):
        msg = "calibration and validation feature counts differ"
        raise ValueError(msg)
    deltas: list[FeatureDriftDelta] = []
    for index, (calibration_mean, validation_mean) in enumerate(
        zip(calibration.feature_mean, validation.feature_mean, strict=True)
    ):
        signed_delta = validation_mean - calibration_mean
        pooled_std = sqrt(
            calibration.feature_std[index] ** 2 + validation.feature_std[index] ** 2
        )
        normalized_delta = _normalized_delta(signed_delta, pooled_std)
        deltas.append(
            FeatureDriftDelta(
                feature_index=index,
                feature_name=_feature_name(index),
                calibration_mean=calibration_mean,
                calibration_std=calibration.feature_std[index],
                calibration_range=(calibration.feature_min[index], calibration.feature_max[index]),
                validation_mean=validation_mean,
                validation_std=validation.feature_std[index],
                validation_range=(validation.feature_min[index], validation.feature_max[index]),
                signed_delta=signed_delta,
                normalized_delta=normalized_delta,
            )
        )
    return tuple(deltas)


def _normalized_delta(signed_delta: float, pooled_std: float) -> float:
    if pooled_std == 0.0:
        return inf if signed_delta else 0.0
    return signed_delta / pooled_std


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
        sum(row.features[index] for row in rows) / len(rows) for index in range(feature_count)
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


def _group_by_target(rows: Sequence[TargetedRowT]) -> Mapping[str, list[TargetedRowT]]:
    grouped: dict[str, list[TargetedRowT]] = defaultdict(list)
    for row in rows:
        grouped[row.target_id].append(row)
    return grouped


def _select_metrics_sample_window(
    rows: Sequence[TargetedRowT],
    metrics_window: MetricsWindow | None,
    *,
    row_count_index: int,
) -> tuple[TargetedRowT, ...]:
    if metrics_window is None:
        return tuple(rows)
    feature_row_count, prediction_row_count, sample_count, target_ids = metrics_window
    if sample_count <= 0:
        msg = "validation metrics sample_count must be positive"
        raise ValueError(msg)
    row_count_at_metric = (feature_row_count, prediction_row_count)[row_count_index]
    metric_rows = tuple(rows[:row_count_at_metric])
    if not target_ids:
        return tuple(metric_rows[-sample_count:])
    per_target_count = sample_count // len(target_ids)
    if per_target_count <= 0:
        return tuple(metric_rows[-sample_count:])
    selected: list[TargetedRowT] = []
    for target_id in target_ids:
        target_rows = [row for row in metric_rows if row.target_id == target_id]
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


def _parse_feature_row(
    payload: Mapping[object, object],
    *,
    line_number: int,
) -> FeatureRow | None:
    if not _as_bool(payload.get("valid"), field="valid", line_number=line_number):
        return None
    features = _parse_features(payload.get("features"), line_number=line_number)
    feature_count = _as_int(
        payload.get("feature_count"),
        field="feature_count",
        line_number=line_number,
    )
    if len(features) != feature_count:
        msg = f"line {line_number}: feature_count does not match features length"
        raise ValueError(msg)
    if not features:
        return None
    return FeatureRow(
        target_id=_as_str(payload.get("target_id"), field="target_id", line_number=line_number),
        target_x=_as_float(payload.get("target_x"), field="target_x", line_number=line_number),
        target_y=_as_float(payload.get("target_y"), field="target_y", line_number=line_number),
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
    if not _as_bool(payload.get("valid"), field="valid", line_number=line_number):
        return None
    target_x = _as_float(payload.get("target_x"), field="target_x", line_number=line_number)
    target_y = _as_float(payload.get("target_y"), field="target_y", line_number=line_number)
    predicted_x = _as_float(payload.get("x"), field="x", line_number=line_number)
    predicted_y = _as_float(payload.get("y"), field="y", line_number=line_number)
    target_cell = _grid_cell(
        target_x * screen_width,
        target_y * screen_height,
        screen_width=screen_width,
        screen_height=screen_height,
        grid_columns=grid_columns,
        grid_rows=grid_rows,
    )
    predicted_cell = _grid_cell(
        predicted_x,
        predicted_y,
        screen_width=screen_width,
        screen_height=screen_height,
        grid_columns=grid_columns,
        grid_rows=grid_rows,
    )
    return ValidationPredictionRow(
        target_id=_as_str(payload.get("target_id"), field="target_id", line_number=line_number),
        target_x=target_x,
        target_y=target_y,
        predicted_cell=predicted_cell,
        matches_target_cell=target_cell == predicted_cell,
    )


def _parse_features(value: object, *, line_number: int) -> tuple[float, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        msg = f"line {line_number}: features must be a list of numbers"
        raise ValueError(msg)
    return tuple(_as_float(item, field="features", line_number=line_number) for item in value)


def _calibration_sample_is_accepted(payload: Mapping[object, object]) -> bool:
    sample_accepted = payload.get("sample_accepted")
    if sample_accepted is None:
        return True
    return _as_bool(sample_accepted, field="sample_accepted", line_number=0)


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


def _drift_sort_key(delta: FeatureDriftDelta) -> tuple[float, float, int]:
    normalized = inf if isinf(delta.normalized_delta) else abs(delta.normalized_delta)
    return (-normalized, -abs(delta.signed_delta), delta.feature_index)


def _context_drift_sort_key(delta: FeatureDriftDelta) -> tuple[int, float, float, int]:
    group = 0 if delta.feature_index in POSTURE_FEATURE_INDICES else 1
    normalized = inf if isinf(delta.normalized_delta) else abs(delta.normalized_delta)
    return (group, -normalized, -abs(delta.signed_delta), delta.feature_index)


def _feature_name(index: int) -> str:
    if index < len(FEATURE_NAMES):
        return FEATURE_NAMES[index]
    return f"feature {index}"


def _format_feature_delta_row(target_id: str, delta: FeatureDriftDelta) -> str:
    return (
        "| "
        f"{target_id} | "
        f"{delta.feature_index} {delta.feature_name} | "
        f"{delta.calibration_mean:.6f} | "
        f"{delta.calibration_std:.6f} | "
        f"{_format_range(delta.calibration_range)} | "
        f"{delta.validation_mean:.6f} | "
        f"{delta.validation_std:.6f} | "
        f"{_format_range(delta.validation_range)} | "
        f"{delta.signed_delta:+.6f} | "
        f"{_format_normalized_delta(delta.normalized_delta)} |"
    )


def _format_range(value_range: tuple[float, float]) -> str:
    return f"{value_range[0]:.6f}..{value_range[1]:.6f}"


def _format_normalized_delta(value: float) -> str:
    if isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:+.2f}"


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{cell}={count}" for cell, count in sorted(counts.items()))


def _format_flags(flags: Sequence[str]) -> str:
    return ", ".join(flags) if flags else "-"


def _parse_event(line: str, *, line_number: int) -> Mapping[str, object]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        msg = f"line {line_number}: invalid JSON"
        raise ValueError(msg) from error
    if not isinstance(event, Mapping):
        msg = f"line {line_number}: event must be an object"
        raise ValueError(msg)
    return cast(Mapping[str, object], event)


def _parse_positive_int(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"{field} must be an integer"
        raise ValueError(msg) from error
    if parsed <= 0:
        msg = f"{field} must be positive"
        raise ValueError(msg)
    return parsed


def _as_float(value: object, *, field: str, line_number: int) -> float:
    if not isinstance(value, int | float):
        msg = f"line {line_number}: {field} must be numeric"
        raise ValueError(msg)
    return float(value)


def _as_int(value: object, *, field: str, line_number: int) -> int:
    if not isinstance(value, int):
        msg = f"line {line_number}: {field} must be an integer"
        raise ValueError(msg)
    return value


def _as_str(value: object, *, field: str, line_number: int) -> str:
    if not isinstance(value, str):
        msg = f"line {line_number}: {field} must be a string"
        raise ValueError(msg)
    return value


def _as_bool(value: object, *, field: str, line_number: int) -> bool:
    if not isinstance(value, bool):
        msg = f"line {line_number}: {field} must be a boolean"
        raise ValueError(msg)
    return value


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to metrics/demo.jsonl")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Inclusive run range START:END; repeat to compare runs",
    )
    parser.add_argument("--screen-width", type=float, required=True)
    parser.add_argument("--screen-height", type=float, required=True)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--dominant-feature-count", type=int, default=6)
    args = parser.parse_args(argv)

    try:
        analyses = tuple(
            analyze_posture_validation_drift_log(
                args.log_path,
                run_range=parse_run_range(run),
                screen_width=args.screen_width,
                screen_height=args.screen_height,
                grid_columns=args.grid_columns,
                grid_rows=args.grid_rows,
                dominant_feature_count=args.dominant_feature_count,
            )
            for run in args.run
        )
    except ValueError as error:
        _die(str(error))
    if len(analyses) == 1:
        print(format_posture_validation_drift_report(analyses[0]))
        return 0
    print(format_posture_validation_drift_run_comparison(analyses))
    for analysis in analyses:
        print()
        print(format_posture_validation_drift_report(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
