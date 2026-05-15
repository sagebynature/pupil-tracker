"""Evaluate calibration model variants from scalar replay telemetry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, cast

from pupil_tracker.calibration import (
    LinearRidgeCalibrationModel,
    PolynomialRidgeCalibrationModel,
    ValidationSample,
    ValidationTarget,
    compute_validation_metrics,
)
from pupil_tracker.models import CalibrationSample, CalibrationTarget, GazeSample, RawObservation

EvaluationObjective = Literal["error", "grid"]
SampleWindow = Literal["all", "early", "middle", "late"]
TargetWeightingPolicy = Literal["none", "vertical_edges", "screen_edges", "corners"]
ResidualRow = tuple[float, float, float, float, float, bool | None]


class _ReplayCalibrationModel(Protocol):
    def fit(
        self,
        samples: Sequence[CalibrationSample],
        screen_width: float,
        screen_height: float,
    ) -> Any: ...

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample: ...


class _CoordinateCorrector(Protocol):
    def fit(
        self,
        predicted_coordinates: Sequence[tuple[float, float]],
        expected_coordinates: Sequence[tuple[float, float]],
    ) -> Any: ...

    def predict(self, coordinates: Sequence[tuple[float, float]]) -> Any: ...


@dataclass(frozen=True)
class ReplayModelCandidate:
    """Evaluator-only model candidate plus optional calibration weighting policy."""

    name: str
    model: _ReplayCalibrationModel
    weighting_policy: TargetWeightingPolicy = "none"


@dataclass(frozen=True)
class ReplayValidationObservation:
    """A validation target paired with the original scalar feature observation."""

    target: ValidationTarget
    observation: RawObservation


@dataclass(frozen=True)
class ReplayDataset:
    """Replayable scalar calibration and validation observations."""

    calibration_samples: tuple[CalibrationSample, ...]
    validation_observations: tuple[ReplayValidationObservation, ...]
    feature_count: int


@dataclass(frozen=True)
class TargetResidualSummary:
    """Per-target replay residual summary for calibration or validation samples."""

    target_id: str
    target_x: float
    target_y: float
    sample_count: int
    mean_error_px: float
    mean_abs_x_error_px: float
    mean_abs_y_error_px: float
    mean_signed_x_error_px: float
    mean_signed_y_error_px: float
    grid_cell_accuracy: float | None = None


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Metrics for one model variant evaluated against replay validation samples."""

    model_name: str
    metrics: Any
    calibration_target_residuals: tuple[TargetResidualSummary, ...] = ()
    validation_target_residuals: tuple[TargetResidualSummary, ...] = ()


class BiasCorrectedCalibrationModel:
    """Apply a constant residual correction learned from calibration predictions."""

    def __init__(self, base_model: _ReplayCalibrationModel) -> None:
        self._base_model = base_model
        self._bias_x = 0.0
        self._bias_y = 0.0

    def fit(
        self,
        samples: Sequence[CalibrationSample],
        screen_width: float,
        screen_height: float,
    ) -> Any:
        result = self._base_model.fit(samples, screen_width, screen_height)
        residuals = _calibration_prediction_residuals(
            self._base_model,
            samples,
            screen_width,
            screen_height,
        )
        self._bias_x = sum(residual[0] for residual in residuals) / len(residuals)
        self._bias_y = sum(residual[1] for residual in residuals) / len(residuals)
        return result

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        sample = self._base_model.predict(observation, screen_width, screen_height)
        if not sample.valid:
            return sample
        return GazeSample(
            timestamp=sample.timestamp,
            x=sample.x + self._bias_x,
            y=sample.y + self._bias_y,
            confidence=sample.confidence,
            valid=True,
            region_id=sample.region_id,
        )


class PerBandCorrectedCalibrationModel:
    """Apply band-specific Y residual corrections learned from calibration predictions."""

    def __init__(self, base_model: _ReplayCalibrationModel, *, bands: int = 3) -> None:
        if bands <= 0:
            msg = "bands must be positive"
            raise ValueError(msg)
        self._base_model = base_model
        self._bands = bands
        self._band_biases: tuple[float, ...] = ()
        self._global_bias = 0.0
        self._fitted = False

    def fit(
        self,
        samples: Sequence[CalibrationSample],
        screen_width: float,
        screen_height: float,
    ) -> Any:
        result = self._base_model.fit(samples, screen_width, screen_height)
        residuals_by_band: list[list[float]] = [[] for _ in range(self._bands)]
        all_residuals: list[float] = []
        for sample in samples:
            if not sample.observation.valid:
                continue
            prediction = self._base_model.predict(
                sample.observation,
                screen_width,
                screen_height,
            )
            if not prediction.valid:
                continue
            residual_y = (sample.target.y * screen_height) - prediction.y
            band = self._band_index(prediction.y, screen_height)
            residuals_by_band[band].append(residual_y)
            all_residuals.append(residual_y)
        if not all_residuals:
            msg = "per-band correction requires at least 1 valid calibration prediction"
            raise ValueError(msg)
        self._global_bias = sum(all_residuals) / len(all_residuals)
        self._band_biases = tuple(
            (sum(residuals) / len(residuals)) if residuals else self._global_bias
            for residuals in residuals_by_band
        )
        self._fitted = True
        return result

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        if not self._fitted:
            msg = "per-band-corrected model is not fitted"
            raise RuntimeError(msg)
        sample = self._base_model.predict(observation, screen_width, screen_height)
        if not sample.valid:
            return sample
        band = self._band_index(sample.y, screen_height)
        return GazeSample(
            timestamp=sample.timestamp,
            x=sample.x,
            y=sample.y + self._band_biases[band],
            confidence=sample.confidence,
            valid=True,
            region_id=sample.region_id,
        )

    def _band_index(self, y: float, screen_height: float) -> int:
        normalized_y = y / screen_height
        clamped = min(max(normalized_y, 0.0), 0.999999)
        return int(clamped * self._bands)


class VerticalBiasCorrectedCalibrationModel:
    """Apply a one-dimensional Y residual correction learned from calibration predictions."""

    def __init__(self, base_model: _ReplayCalibrationModel) -> None:
        self._base_model = base_model
        self._slope = 0.0
        self._intercept = 0.0
        self._fitted = False

    def fit(
        self,
        samples: Sequence[CalibrationSample],
        screen_width: float,
        screen_height: float,
    ) -> Any:
        result = self._base_model.fit(samples, screen_width, screen_height)
        rows: list[tuple[float, float]] = []
        for sample in samples:
            if not sample.observation.valid:
                continue
            prediction = self._base_model.predict(
                sample.observation,
                screen_width,
                screen_height,
            )
            if not prediction.valid:
                continue
            predicted_y_norm = prediction.y / screen_height
            expected_y = sample.target.y * screen_height
            rows.append((predicted_y_norm, expected_y - prediction.y))
        if len(rows) < 2:
            msg = "vertical bias correction requires at least 2 valid calibration predictions"
            raise ValueError(msg)
        mean_x = sum(row[0] for row in rows) / len(rows)
        mean_y = sum(row[1] for row in rows) / len(rows)
        denominator = sum((row[0] - mean_x) ** 2 for row in rows)
        if denominator == 0:
            self._slope = 0.0
        else:
            self._slope = sum((row[0] - mean_x) * (row[1] - mean_y) for row in rows) / denominator
        self._intercept = mean_y - self._slope * mean_x
        self._fitted = True
        return result

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        if not self._fitted:
            msg = "vertical-bias-corrected model is not fitted"
            raise RuntimeError(msg)
        sample = self._base_model.predict(observation, screen_width, screen_height)
        if not sample.valid:
            return sample
        correction_y = self._intercept + self._slope * (sample.y / screen_height)
        return GazeSample(
            timestamp=sample.timestamp,
            x=sample.x,
            y=sample.y + correction_y,
            confidence=sample.confidence,
            valid=True,
            region_id=sample.region_id,
        )


class AsymmetricRegionCorrectedCalibrationModel:
    """Apply quadrant-specific residual corrections learned from calibration predictions."""

    def __init__(self, base_model: _ReplayCalibrationModel) -> None:
        self._base_model = base_model
        self._region_biases: dict[str, tuple[float, float]] = {}
        self._global_bias = (0.0, 0.0)
        self._fitted = False

    def fit(
        self,
        samples: Sequence[CalibrationSample],
        screen_width: float,
        screen_height: float,
    ) -> Any:
        result = self._base_model.fit(samples, screen_width, screen_height)
        residuals_by_region: dict[str, list[tuple[float, float]]] = defaultdict(list)
        all_residuals: list[tuple[float, float]] = []
        for sample in samples:
            if not sample.observation.valid:
                continue
            prediction = self._base_model.predict(
                sample.observation,
                screen_width,
                screen_height,
            )
            if not prediction.valid:
                continue
            residual = (
                sample.target.x * screen_width - prediction.x,
                sample.target.y * screen_height - prediction.y,
            )
            region = _asymmetric_region_id(sample.target.x, sample.target.y)
            residuals_by_region[region].append(residual)
            all_residuals.append(residual)
        if not all_residuals:
            msg = "asymmetric correction requires valid calibration predictions"
            raise ValueError(msg)
        self._global_bias = _mean_pair(all_residuals)
        self._region_biases = {
            region: _mean_pair(residuals)
            for region, residuals in residuals_by_region.items()
            if residuals
        }
        self._fitted = True
        return result

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        if not self._fitted:
            msg = "asymmetric-corrected model is not fitted"
            raise RuntimeError(msg)
        sample = self._base_model.predict(observation, screen_width, screen_height)
        if not sample.valid:
            return sample
        region = _asymmetric_region_id(sample.x / screen_width, sample.y / screen_height)
        bias_x, bias_y = self._region_biases.get(region, self._global_bias)
        return GazeSample(
            timestamp=sample.timestamp,
            x=sample.x + bias_x,
            y=sample.y + bias_y,
            confidence=sample.confidence,
            valid=True,
            region_id=sample.region_id,
        )


class AffineCorrectedCalibrationModel:
    """Apply a 2D affine correction learned from calibration predictions."""

    def __init__(self, base_model: _ReplayCalibrationModel) -> None:
        self._base_model = base_model
        self._corrector: _CoordinateCorrector | None = None

    def fit(
        self,
        samples: Sequence[CalibrationSample],
        screen_width: float,
        screen_height: float,
    ) -> Any:
        result = self._base_model.fit(samples, screen_width, screen_height)
        predicted_coordinates: list[tuple[float, float]] = []
        expected_coordinates: list[tuple[float, float]] = []
        for sample in samples:
            if not sample.observation.valid:
                continue
            prediction = self._base_model.predict(
                sample.observation,
                screen_width,
                screen_height,
            )
            predicted_coordinates.append((prediction.x, prediction.y))
            expected_coordinates.append(
                (sample.target.x * screen_width, sample.target.y * screen_height)
            )
        if len(predicted_coordinates) < 3:
            msg = "affine correction requires at least 3 valid calibration samples"
            raise ValueError(msg)
        from sklearn.linear_model import LinearRegression

        corrector = cast(_CoordinateCorrector, LinearRegression())
        corrector.fit(predicted_coordinates, expected_coordinates)
        self._corrector = corrector
        return result

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        if self._corrector is None:
            msg = "affine-corrected model is not fitted"
            raise RuntimeError(msg)
        sample = self._base_model.predict(observation, screen_width, screen_height)
        if not sample.valid:
            return sample
        prediction = self._corrector.predict([(sample.x, sample.y)])[0]
        return GazeSample(
            timestamp=sample.timestamp,
            x=float(prediction[0]),
            y=float(prediction[1]),
            confidence=sample.confidence,
            valid=True,
            region_id=sample.region_id,
        )


def load_replay_dataset(path: Path) -> ReplayDataset:
    """Load scalar replay samples from a JSONL telemetry file."""

    calibration_samples: list[CalibrationSample] = []
    validation_observations: list[ReplayValidationObservation] = []
    expected_feature_count: int | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = _parse_event(line, line_number=line_number)
        event_type = event.get("event_type")
        raw_payload = event.get("payload")
        if not isinstance(raw_payload, Mapping):
            continue
        payload = cast(Mapping[object, object], raw_payload)
        if event_type == "calibration_replay_sample":
            sample = _parse_calibration_sample(payload, line_number=line_number)
            expected_feature_count = _check_feature_count(
                sample.observation.feature_vector,
                expected_feature_count,
                line_number=line_number,
            )
            calibration_samples.append(sample)
        elif event_type == "validation_replay_sample":
            replay_observation = _parse_validation_observation(
                payload,
                line_number=line_number,
            )
            expected_feature_count = _check_feature_count(
                replay_observation.observation.feature_vector,
                expected_feature_count,
                line_number=line_number,
            )
            validation_observations.append(replay_observation)
    if not calibration_samples:
        msg = "replay log has no calibration_replay_sample events"
        raise ValueError(msg)
    if not validation_observations:
        msg = "replay log has no validation_replay_sample events"
        raise ValueError(msg)
    if expected_feature_count is None:
        msg = "replay log has no feature vectors"
        raise ValueError(msg)
    return ReplayDataset(
        calibration_samples=tuple(calibration_samples),
        validation_observations=tuple(validation_observations),
        feature_count=expected_feature_count,
    )


def filter_calibration_samples_by_window(
    samples: Sequence[CalibrationSample],
    *,
    window: SampleWindow,
) -> tuple[CalibrationSample, ...]:
    """Keep the same early/middle/late capture window for each target id."""

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


def _sample_window_bounds(sample_count: int, *, window: SampleWindow) -> tuple[int, int]:
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


def apply_target_weighting(
    samples: Sequence[CalibrationSample],
    *,
    policy: TargetWeightingPolicy,
) -> tuple[CalibrationSample, ...]:
    """Expand calibration samples according to an evaluator-only target weighting policy."""

    if policy == "none":
        return tuple(samples)
    weighted: list[CalibrationSample] = []
    for sample in samples:
        weighted.extend([sample] * _target_weight(sample.target, policy=policy))
    return tuple(weighted)


def _target_weight(target: CalibrationTarget, *, policy: TargetWeightingPolicy) -> int:
    x_is_edge = target.x <= 0.25 or target.x >= 0.75
    y_is_edge = target.y <= 0.25 or target.y >= 0.75
    if policy == "vertical_edges" and y_is_edge:
        return 3
    if policy == "screen_edges" and (x_is_edge or y_is_edge):
        return 3
    if policy == "corners" and x_is_edge and y_is_edge:
        return 3
    return 1


def summarize_calibration_target_residuals(
    model: _ReplayCalibrationModel,
    samples: Sequence[CalibrationSample],
    *,
    screen_width: float,
    screen_height: float,
) -> tuple[TargetResidualSummary, ...]:
    """Summarize fitted-model residuals grouped by calibration target."""

    rows_by_target: dict[str, list[ResidualRow]] = defaultdict(list)
    for sample in samples:
        if not sample.observation.valid:
            continue
        prediction = model.predict(sample.observation, screen_width, screen_height)
        if not prediction.valid:
            continue
        target_x = sample.target.x * screen_width
        target_y = sample.target.y * screen_height
        dx = prediction.x - target_x
        dy = prediction.y - target_y
        rows_by_target[sample.target.id].append(
            (sample.target.x, sample.target.y, dx, dy, hypot(dx, dy), None)
        )
    return _summarize_residual_rows(rows_by_target)


def summarize_validation_target_residuals(
    samples: Sequence[ValidationSample],
    *,
    screen_width: float,
    screen_height: float,
    grid_columns: int,
    grid_rows: int,
) -> tuple[TargetResidualSummary, ...]:
    """Summarize predicted validation residuals grouped by validation target."""

    rows_by_target: dict[str, list[ResidualRow]] = defaultdict(list)
    for sample in samples:
        if not sample.gaze_sample.valid:
            continue
        target_x = sample.target.x * screen_width
        target_y = sample.target.y * screen_height
        dx = sample.gaze_sample.x - target_x
        dy = sample.gaze_sample.y - target_y
        target_cell = _grid_cell_id(
            target_x,
            target_y,
            screen_width,
            screen_height,
            columns=grid_columns,
            rows=grid_rows,
        )
        gaze_cell = _grid_cell_id(
            sample.gaze_sample.x,
            sample.gaze_sample.y,
            screen_width,
            screen_height,
            columns=grid_columns,
            rows=grid_rows,
        )
        rows_by_target[sample.target.id].append(
            (
                sample.target.x,
                sample.target.y,
                dx,
                dy,
                hypot(dx, dy),
                target_cell == gaze_cell,
            )
        )
    return _summarize_residual_rows(rows_by_target)


def _summarize_residual_rows(
    rows_by_target: Mapping[str, Sequence[ResidualRow]],
) -> tuple[TargetResidualSummary, ...]:
    summaries: list[TargetResidualSummary] = []
    for target_id, rows in rows_by_target.items():
        if not rows:
            continue
        signed_x_errors = [row[2] for row in rows]
        signed_y_errors = [row[3] for row in rows]
        errors = [row[4] for row in rows]
        grid_matches = [row[5] for row in rows if row[5] is not None]
        summaries.append(
            TargetResidualSummary(
                target_id=target_id,
                target_x=rows[0][0],
                target_y=rows[0][1],
                sample_count=len(rows),
                mean_error_px=sum(errors) / len(errors),
                mean_abs_x_error_px=sum(abs(value) for value in signed_x_errors)
                / len(signed_x_errors),
                mean_abs_y_error_px=sum(abs(value) for value in signed_y_errors)
                / len(signed_y_errors),
                mean_signed_x_error_px=sum(signed_x_errors) / len(signed_x_errors),
                mean_signed_y_error_px=sum(signed_y_errors) / len(signed_y_errors),
                grid_cell_accuracy=(
                    sum(1 for value in grid_matches if value) / len(grid_matches)
                    if grid_matches
                    else None
                ),
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (-summary.mean_error_px, summary.target_id),
        )
    )


def _grid_cell_id(
    x: float,
    y: float,
    screen_width: float,
    screen_height: float,
    *,
    columns: int,
    rows: int,
) -> str:
    clamped_x = min(max(x, 0.0), screen_width)
    clamped_y = min(max(y, 0.0), screen_height)
    column = min(int(clamped_x / (screen_width / columns)), columns - 1)
    row = min(int(clamped_y / (screen_height / rows)), rows - 1)
    return f"r{row}c{column}"


def _asymmetric_region_id(normalized_x: float, normalized_y: float) -> str:
    horizontal = "left" if normalized_x < 0.5 else "right"
    vertical = "top" if normalized_y < 0.5 else "bottom"
    return f"{vertical}_{horizontal}"


def _mean_pair(values: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(value[0] for value in values) / len(values),
        sum(value[1] for value in values) / len(values),
    )


def evaluate_replay_models(
    dataset: ReplayDataset,
    *,
    screen_width: float,
    screen_height: float,
    grid_columns: int,
    grid_rows: int,
    objective: EvaluationObjective = "error",
    calibration_sample_window: SampleWindow = "all",
) -> tuple[ModelEvaluationResult, ...]:
    """Fit candidate models on replay calibration samples and score validation samples."""

    results: list[ModelEvaluationResult] = []
    calibration_samples = filter_calibration_samples_by_window(
        dataset.calibration_samples,
        window=calibration_sample_window,
    )
    for candidate in _candidate_models():
        model = candidate.model
        weighted_calibration_samples = apply_target_weighting(
            calibration_samples,
            policy=candidate.weighting_policy,
        )
        model.fit(weighted_calibration_samples, screen_width, screen_height)
        validation_samples = tuple(
            ValidationSample(
                target=replay_observation.target,
                gaze_sample=model.predict(
                    replay_observation.observation,
                    screen_width,
                    screen_height,
                ),
            )
            for replay_observation in dataset.validation_observations
        )
        metrics = compute_validation_metrics(
            validation_samples,
            screen_width=screen_width,
            screen_height=screen_height,
            grid_columns=grid_columns,
            grid_rows=grid_rows,
        )
        calibration_residuals = summarize_calibration_target_residuals(
            model,
            calibration_samples,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        validation_residuals = summarize_validation_target_residuals(
            validation_samples,
            screen_width=screen_width,
            screen_height=screen_height,
            grid_columns=grid_columns,
            grid_rows=grid_rows,
        )
        results.append(
            ModelEvaluationResult(
                model_name=candidate.name,
                metrics=metrics,
                calibration_target_residuals=calibration_residuals,
                validation_target_residuals=validation_residuals,
            )
        )
    return sort_model_results(results, objective=objective)


def sort_model_results(
    results: Sequence[ModelEvaluationResult],
    *,
    objective: EvaluationObjective,
) -> tuple[ModelEvaluationResult, ...]:
    """Sort model results by the chosen objective."""

    if objective == "grid":
        return tuple(
            sorted(
                results,
                key=lambda result: (
                    -result.metrics.grid_cell_accuracy,
                    result.metrics.mean_error_px,
                ),
            )
        )
    return tuple(sorted(results, key=lambda result: result.metrics.mean_error_px))


def format_model_evaluation_report(results: Sequence[ModelEvaluationResult]) -> str:
    """Format model evaluation metrics as a compact Markdown table."""

    lines = [
        "| Model | Mean Error | Mean X | Mean Y | Signed Y | Grid Accuracy | Recommendation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        metrics = result.metrics
        lines.append(
            "| "
            f"{result.model_name} | "
            f"{metrics.mean_error_px:.2f} px | "
            f"{metrics.mean_abs_x_error_px:.2f} px | "
            f"{metrics.mean_abs_y_error_px:.2f} px | "
            f"{metrics.mean_signed_y_error_px:+.2f} px | "
            f"{metrics.grid_cell_accuracy:.1%} | "
            f"{metrics.recommendation} |"
        )
    return "\n".join(lines)


def format_target_residual_report(result: ModelEvaluationResult) -> str:
    """Format per-target residual summaries for one model as Markdown."""

    return "\n\n".join(
        (
            f"## Target Residuals: {result.model_name}",
            "### Calibration Residuals\n"
            + _format_residual_table(result.calibration_target_residuals),
            "### Validation Residuals\n"
            + _format_residual_table(result.validation_target_residuals),
        )
    )


def _format_residual_table(summaries: Sequence[TargetResidualSummary]) -> str:
    lines = [
        "| Target | Target X | Target Y | Samples | Mean Error | Mean X | Mean Y | "
        "Signed X | Signed Y | Grid Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        grid_accuracy = (
            f"{summary.grid_cell_accuracy:.1%}"
            if summary.grid_cell_accuracy is not None
            else "N/A"
        )
        lines.append(
            "| "
            f"{summary.target_id} | "
            f"{summary.target_x:.2f} | "
            f"{summary.target_y:.2f} | "
            f"{summary.sample_count} | "
            f"{summary.mean_error_px:.2f} px | "
            f"{summary.mean_abs_x_error_px:.2f} px | "
            f"{summary.mean_abs_y_error_px:.2f} px | "
            f"{summary.mean_signed_x_error_px:+.2f} px | "
            f"{summary.mean_signed_y_error_px:+.2f} px | "
            f"{grid_accuracy} |"
        )
    return "\n".join(lines)


def _calibration_prediction_residuals(
    model: _ReplayCalibrationModel,
    samples: Sequence[CalibrationSample],
    screen_width: float,
    screen_height: float,
) -> tuple[tuple[float, float], ...]:
    residuals: list[tuple[float, float]] = []
    for sample in samples:
        if not sample.observation.valid:
            continue
        prediction = model.predict(sample.observation, screen_width, screen_height)
        residuals.append(
            (
                sample.target.x * screen_width - prediction.x,
                sample.target.y * screen_height - prediction.y,
            )
        )
    if not residuals:
        msg = "correction requires valid calibration predictions"
        raise ValueError(msg)
    return tuple(residuals)


def _candidate_models() -> tuple[ReplayModelCandidate, ...]:
    linear_1 = LinearRidgeCalibrationModel(alpha=1.0)
    poly_1 = PolynomialRidgeCalibrationModel(degree=2, alpha=1.0)
    base_candidates = (
        ReplayModelCandidate(
            "linear-alpha-0.1",
            LinearRidgeCalibrationModel(alpha=0.1),
        ),
        ReplayModelCandidate(
            "linear-alpha-0.1-asymmetric-corrected",
            AsymmetricRegionCorrectedCalibrationModel(LinearRidgeCalibrationModel(alpha=0.1)),
        ),
        ReplayModelCandidate(
            "linear-alpha-0.1-per-band-corrected",
            PerBandCorrectedCalibrationModel(LinearRidgeCalibrationModel(alpha=0.1)),
        ),
        ReplayModelCandidate(
            "linear-alpha-0.1-vertical-bias-corrected",
            VerticalBiasCorrectedCalibrationModel(LinearRidgeCalibrationModel(alpha=0.1)),
        ),
        ReplayModelCandidate("linear-alpha-1.0", linear_1),
        ReplayModelCandidate(
            "linear-alpha-1.0-asymmetric-corrected",
            AsymmetricRegionCorrectedCalibrationModel(LinearRidgeCalibrationModel(alpha=1.0)),
        ),
        ReplayModelCandidate(
            "linear-alpha-1.0-per-band-corrected",
            PerBandCorrectedCalibrationModel(LinearRidgeCalibrationModel(alpha=1.0)),
        ),
        ReplayModelCandidate(
            "linear-alpha-1.0-vertical-bias-corrected",
            VerticalBiasCorrectedCalibrationModel(LinearRidgeCalibrationModel(alpha=1.0)),
        ),
        ReplayModelCandidate(
            "linear-alpha-1.0-bias-corrected",
            BiasCorrectedCalibrationModel(LinearRidgeCalibrationModel(alpha=1.0)),
        ),
        ReplayModelCandidate(
            "linear-alpha-1.0-affine-corrected",
            AffineCorrectedCalibrationModel(LinearRidgeCalibrationModel(alpha=1.0)),
        ),
        ReplayModelCandidate("linear-alpha-10.0", LinearRidgeCalibrationModel(alpha=10.0)),
        ReplayModelCandidate(
            "poly2-alpha-0.1",
            PolynomialRidgeCalibrationModel(degree=2, alpha=0.1),
        ),
        ReplayModelCandidate("poly2-alpha-1.0", poly_1),
        ReplayModelCandidate(
            "poly2-alpha-1.0-asymmetric-corrected",
            AsymmetricRegionCorrectedCalibrationModel(
                PolynomialRidgeCalibrationModel(degree=2, alpha=1.0)
            ),
        ),
        ReplayModelCandidate(
            "poly2-alpha-1.0-per-band-corrected",
            PerBandCorrectedCalibrationModel(
                PolynomialRidgeCalibrationModel(degree=2, alpha=1.0)
            ),
        ),
        ReplayModelCandidate(
            "poly2-alpha-1.0-vertical-bias-corrected",
            VerticalBiasCorrectedCalibrationModel(
                PolynomialRidgeCalibrationModel(degree=2, alpha=1.0)
            ),
        ),
        ReplayModelCandidate(
            "poly2-alpha-1.0-bias-corrected",
            BiasCorrectedCalibrationModel(
                PolynomialRidgeCalibrationModel(degree=2, alpha=1.0)
            ),
        ),
        ReplayModelCandidate(
            "poly2-alpha-1.0-affine-corrected",
            AffineCorrectedCalibrationModel(
                PolynomialRidgeCalibrationModel(degree=2, alpha=1.0)
            ),
        ),
        ReplayModelCandidate(
            "poly2-alpha-10.0",
            PolynomialRidgeCalibrationModel(degree=2, alpha=10.0),
        ),
        ReplayModelCandidate(
            "poly2-alpha-10.0-asymmetric-corrected",
            AsymmetricRegionCorrectedCalibrationModel(
                PolynomialRidgeCalibrationModel(degree=2, alpha=10.0)
            ),
        ),
        ReplayModelCandidate(
            "poly2-alpha-10.0-per-band-corrected",
            PerBandCorrectedCalibrationModel(
                PolynomialRidgeCalibrationModel(degree=2, alpha=10.0)
            ),
        ),
        ReplayModelCandidate(
            "poly2-alpha-10.0-vertical-bias-corrected",
            VerticalBiasCorrectedCalibrationModel(
                PolynomialRidgeCalibrationModel(degree=2, alpha=10.0)
            ),
        ),
    )
    weighting_policies: tuple[TargetWeightingPolicy, ...] = (
        "vertical_edges",
        "screen_edges",
        "corners",
    )
    weighted_candidates = tuple(
        candidate
        for policy in weighting_policies
        for candidate in (
            ReplayModelCandidate(
                f"linear-alpha-0.1-weight-{policy}",
                LinearRidgeCalibrationModel(alpha=0.1),
                weighting_policy=policy,
            ),
            ReplayModelCandidate(
                f"poly2-alpha-1.0-weight-{policy}",
                PolynomialRidgeCalibrationModel(degree=2, alpha=1.0),
                weighting_policy=policy,
            ),
        )
    )
    return base_candidates + weighted_candidates


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


def _parse_calibration_sample(
    payload: Mapping[object, object],
    *,
    line_number: int,
) -> CalibrationSample:
    observation = _parse_observation(payload, line_number=line_number)
    return CalibrationSample(
        target=CalibrationTarget(
            id=_as_str(payload.get("target_id"), field="target_id", line_number=line_number),
            x=_as_float(payload.get("target_x"), field="target_x", line_number=line_number),
            y=_as_float(payload.get("target_y"), field="target_y", line_number=line_number),
        ),
        observation=observation,
    )


def _parse_validation_observation(
    payload: Mapping[object, object],
    *,
    line_number: int,
) -> ReplayValidationObservation:
    observation = _parse_observation(payload, line_number=line_number)
    return ReplayValidationObservation(
        target=ValidationTarget(
            id=_as_str(payload.get("target_id"), field="target_id", line_number=line_number),
            x=_as_float(payload.get("target_x"), field="target_x", line_number=line_number),
            y=_as_float(payload.get("target_y"), field="target_y", line_number=line_number),
        ),
        observation=observation,
    )


def _parse_observation(payload: Mapping[object, object], *, line_number: int) -> RawObservation:
    features = _parse_features(payload.get("features"), line_number=line_number)
    feature_count = _as_int(
        payload.get("feature_count"),
        field="feature_count",
        line_number=line_number,
    )
    if len(features) != feature_count:
        msg = f"line {line_number}: feature_count does not match features length"
        raise ValueError(msg)
    return RawObservation(
        timestamp=_as_float(payload.get("timestamp"), field="timestamp", line_number=line_number),
        valid=_as_bool(payload.get("valid"), field="valid", line_number=line_number),
        confidence=_as_float(
            payload.get("confidence"),
            field="confidence",
            line_number=line_number,
        ),
        feature_vector=features,
    )


def _parse_features(value: object, *, line_number: int) -> tuple[float, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        msg = f"line {line_number}: features must be a list of numbers"
        raise ValueError(msg)
    return tuple(
        _as_float(item, field="features", line_number=line_number) for item in value
    )


def _check_feature_count(
    features: tuple[float, ...],
    expected_feature_count: int | None,
    *,
    line_number: int,
) -> int:
    if not features:
        msg = f"line {line_number}: replay features must not be empty"
        raise ValueError(msg)
    if expected_feature_count is None:
        return len(features)
    if len(features) != expected_feature_count:
        msg = f"line {line_number}: replay feature lengths must match"
        raise ValueError(msg)
    return expected_feature_count


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
    parser = argparse.ArgumentParser(
        description="Evaluate calibration models from scalar replay telemetry."
    )
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--screen-width", type=float, required=True)
    parser.add_argument("--screen-height", type=float, required=True)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument(
        "--objective",
        choices=("grid", "error"),
        default="grid",
        help="Rank models by grid-cell accuracy or mean pixel error.",
    )
    parser.add_argument(
        "--calibration-sample-window",
        choices=("all", "early", "middle", "late"),
        default="all",
        help="Fit models with all or one third of each target's calibration samples.",
    )
    parser.add_argument(
        "--include-target-residuals",
        action="store_true",
        help="Append calibration and validation residual tables for the top-ranked model.",
    )
    args = parser.parse_args(argv)
    try:
        dataset = load_replay_dataset(args.log_path)
        results = evaluate_replay_models(
            dataset,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            grid_columns=args.grid_columns,
            grid_rows=args.grid_rows,
            objective=args.objective,
            calibration_sample_window=args.calibration_sample_window,
        )
    except ValueError as error:
        _die(str(error))
    print(format_model_evaluation_report(results))
    if args.include_target_residuals and results:
        print()
        print(format_target_residual_report(results[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
