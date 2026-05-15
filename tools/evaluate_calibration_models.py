"""Evaluate calibration model variants from scalar replay telemetry."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from pupil_tracker.calibration import (
    LinearRidgeCalibrationModel,
    PolynomialRidgeCalibrationModel,
    ValidationSample,
    ValidationTarget,
    compute_validation_metrics,
)
from pupil_tracker.models import CalibrationSample, CalibrationTarget, RawObservation


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
class ModelEvaluationResult:
    """Metrics for one model variant evaluated against replay validation samples."""

    model_name: str
    metrics: Any


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


def evaluate_replay_models(
    dataset: ReplayDataset,
    *,
    screen_width: float,
    screen_height: float,
    grid_columns: int,
    grid_rows: int,
) -> tuple[ModelEvaluationResult, ...]:
    """Fit candidate models on replay calibration samples and score validation samples."""

    results: list[ModelEvaluationResult] = []
    for model_name, model in _candidate_models():
        model.fit(dataset.calibration_samples, screen_width, screen_height)
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
        results.append(ModelEvaluationResult(model_name=model_name, metrics=metrics))
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


def _candidate_models() -> tuple[
    tuple[str, LinearRidgeCalibrationModel | PolynomialRidgeCalibrationModel], ...
]:
    return (
        ("linear-alpha-0.1", LinearRidgeCalibrationModel(alpha=0.1)),
        ("linear-alpha-1.0", LinearRidgeCalibrationModel(alpha=1.0)),
        ("linear-alpha-10.0", LinearRidgeCalibrationModel(alpha=10.0)),
        ("poly2-alpha-0.1", PolynomialRidgeCalibrationModel(degree=2, alpha=0.1)),
        ("poly2-alpha-1.0", PolynomialRidgeCalibrationModel(degree=2, alpha=1.0)),
        ("poly2-alpha-10.0", PolynomialRidgeCalibrationModel(degree=2, alpha=10.0)),
    )


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
    args = parser.parse_args(argv)
    try:
        dataset = load_replay_dataset(args.log_path)
        results = evaluate_replay_models(
            dataset,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            grid_columns=args.grid_columns,
            grid_rows=args.grid_rows,
        )
    except ValueError as error:
        _die(str(error))
    print(format_model_evaluation_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
