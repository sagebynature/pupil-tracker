"""Tests for offline calibration model replay evaluation."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_calibration_models import (  # noqa: E402
    ModelEvaluationResult,
    evaluate_replay_models,
    filter_calibration_samples_by_window,
    format_model_evaluation_report,
    format_target_residual_report,
    load_replay_dataset,
    sort_model_results,
    summarize_calibration_target_residuals,
    summarize_validation_target_residuals,
)
from pupil_tracker.calibration import ValidationSample, ValidationTarget  # noqa: E402
from pupil_tracker.models import (  # noqa: E402
    CalibrationSample,
    CalibrationTarget,
    GazeSample,
    RawObservation,
)


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def _replay_event(
    event_type: str,
    *,
    target_id: str,
    target_x: float,
    target_y: float,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "timestamp": 1.0,
        "payload": {
            "target_id": target_id,
            "target_x": target_x,
            "target_y": target_y,
            "timestamp": 10.0,
            "valid": True,
            "confidence": 0.9,
            "feature_count": 2,
            "features": [target_x, target_y],
        },
    }


class FeatureCoordinateModel:
    def fit(
        self,
        samples: Sequence[CalibrationSample],
        screen_width: float,
        screen_height: float,
    ) -> None:
        return None

    def predict(
        self,
        observation: RawObservation,
        screen_width: float,
        screen_height: float,
    ) -> GazeSample:
        return GazeSample(
            timestamp=observation.timestamp,
            x=observation.feature_vector[0],
            y=observation.feature_vector[1],
            confidence=observation.confidence,
            valid=observation.valid,
        )


def _calibration_sample(target_id: str, sample_index: int) -> CalibrationSample:
    return CalibrationSample(
        target=CalibrationTarget(
            id=target_id,
            x=0.25 if target_id == "left" else 0.75,
            y=0.5,
        ),
        observation=RawObservation(
            timestamp=float(sample_index),
            valid=True,
            confidence=0.9,
            feature_vector=(float(sample_index),),
        ),
    )


def test_load_replay_dataset_reads_calibration_and_validation_samples(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _replay_event("calibration_replay_sample", target_id="c0", target_x=0.0, target_y=0.0),
            _replay_event("calibration_replay_sample", target_id="c1", target_x=1.0, target_y=0.0),
            _replay_event("validation_replay_sample", target_id="v0", target_x=0.5, target_y=0.5),
            {"event_type": "raw_observation", "payload": {"valid": True}},
        ],
    )

    dataset = load_replay_dataset(log_path)

    assert len(dataset.calibration_samples) == 2
    assert len(dataset.validation_observations) == 1
    assert dataset.feature_count == 2
    assert dataset.calibration_samples[0].target.id == "c0"
    assert dataset.validation_observations[0].target.id == "v0"
    assert dataset.validation_observations[0].observation.feature_vector == (0.5, 0.5)


def test_filter_calibration_samples_by_window_keeps_same_window_per_target() -> None:
    samples = tuple(
        _calibration_sample(target_id, sample_index)
        for sample_index in range(6)
        for target_id in ("left", "right")
    )

    all_samples = filter_calibration_samples_by_window(samples, window="all")
    early = filter_calibration_samples_by_window(samples, window="early")
    middle = filter_calibration_samples_by_window(samples, window="middle")
    late = filter_calibration_samples_by_window(samples, window="late")

    assert all_samples == samples
    assert [sample.observation.feature_vector[0] for sample in early] == [0.0, 0.0, 1.0, 1.0]
    assert [sample.observation.feature_vector[0] for sample in middle] == [2.0, 2.0, 3.0, 3.0]
    assert [sample.observation.feature_vector[0] for sample in late] == [4.0, 4.0, 5.0, 5.0]


def test_target_residual_summaries_sort_worst_targets_first() -> None:
    model = FeatureCoordinateModel()
    calibration_samples = (
        CalibrationSample(
            target=CalibrationTarget(id="c_good", x=0.5, y=0.5),
            observation=RawObservation(
                timestamp=1.0,
                valid=True,
                confidence=0.9,
                feature_vector=(50.0, 50.0),
            ),
        ),
        CalibrationSample(
            target=CalibrationTarget(id="c_bad", x=0.5, y=0.5),
            observation=RawObservation(
                timestamp=2.0,
                valid=True,
                confidence=0.9,
                feature_vector=(80.0, 20.0),
            ),
        ),
    )
    validation_samples = (
        ValidationSample(
            target=ValidationTarget(id="v_good", x=0.25, y=0.25),
            gaze_sample=GazeSample(
                timestamp=3.0,
                x=26.0,
                y=26.0,
                confidence=0.9,
                valid=True,
            ),
        ),
        ValidationSample(
            target=ValidationTarget(id="v_bad", x=0.75, y=0.75),
            gaze_sample=GazeSample(
                timestamp=4.0,
                x=20.0,
                y=90.0,
                confidence=0.9,
                valid=True,
            ),
        ),
    )

    calibration = summarize_calibration_target_residuals(
        model,
        calibration_samples,
        screen_width=100.0,
        screen_height=100.0,
    )
    validation = summarize_validation_target_residuals(
        validation_samples,
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=4,
    )

    assert [summary.target_id for summary in calibration] == ["c_bad", "c_good"]
    assert calibration[0].sample_count == 1
    assert calibration[0].mean_signed_x_error_px == 30.0
    assert calibration[0].mean_signed_y_error_px == -30.0
    assert calibration[0].grid_cell_accuracy is None
    assert [summary.target_id for summary in validation] == ["v_bad", "v_good"]
    assert validation[0].mean_signed_x_error_px == -55.0
    assert validation[0].mean_signed_y_error_px == 15.0
    assert validation[0].grid_cell_accuracy == 0.0
    assert validation[1].grid_cell_accuracy == 1.0


def test_target_residual_report_formats_markdown_tables(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    events = [
        _replay_event("calibration_replay_sample", target_id="c0", target_x=0.0, target_y=0.0),
        _replay_event("calibration_replay_sample", target_id="c1", target_x=1.0, target_y=1.0),
        _replay_event("calibration_replay_sample", target_id="c2", target_x=0.0, target_y=1.0),
        _replay_event("validation_replay_sample", target_id="v0", target_x=0.25, target_y=0.25),
        _replay_event("validation_replay_sample", target_id="v1", target_x=0.75, target_y=0.75),
    ]
    _write_jsonl(log_path, events)
    dataset = load_replay_dataset(log_path)
    result = evaluate_replay_models(
        dataset,
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=4,
        objective="grid",
    )[0]

    report = format_target_residual_report(result)

    assert f"## Target Residuals: {result.model_name}" in report
    assert "### Calibration Residuals" in report
    assert "### Validation Residuals" in report
    residual_header = (
        "| Target | Target X | Target Y | Samples | Mean Error | Mean X | Mean Y | "
        "Signed X | Signed Y | Grid Accuracy |"
    )
    assert residual_header in report
    assert "N/A" in report
    assert "v0" in report


def test_sort_model_results_supports_grid_first_objective() -> None:
    results = (
        ModelEvaluationResult(
            model_name="low-error-low-grid",
            metrics=SimpleNamespace(mean_error_px=10.0, grid_cell_accuracy=0.25),
        ),
        ModelEvaluationResult(
            model_name="high-error-high-grid",
            metrics=SimpleNamespace(mean_error_px=50.0, grid_cell_accuracy=0.75),
        ),
        ModelEvaluationResult(
            model_name="tie-grid-better-error",
            metrics=SimpleNamespace(mean_error_px=40.0, grid_cell_accuracy=0.75),
        ),
    )

    grid_sorted = sort_model_results(results, objective="grid")
    error_sorted = sort_model_results(results, objective="error")

    assert [result.model_name for result in grid_sorted] == [
        "tie-grid-better-error",
        "high-error-high-grid",
        "low-error-low-grid",
    ]
    assert [result.model_name for result in error_sorted] == [
        "low-error-low-grid",
        "tie-grid-better-error",
        "high-error-high-grid",
    ]


def test_corrected_candidates_can_reduce_regularized_mapping_error(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    events = [
        _replay_event("calibration_replay_sample", target_id="c0", target_x=0.0, target_y=0.0),
        _replay_event("calibration_replay_sample", target_id="c1", target_x=1.0, target_y=0.0),
        _replay_event("calibration_replay_sample", target_id="c2", target_x=0.0, target_y=1.0),
        _replay_event("calibration_replay_sample", target_id="c3", target_x=1.0, target_y=1.0),
        _replay_event("calibration_replay_sample", target_id="c4", target_x=0.5, target_y=0.5),
        _replay_event("validation_replay_sample", target_id="v0", target_x=0.25, target_y=0.25),
        _replay_event("validation_replay_sample", target_id="v1", target_x=0.75, target_y=0.75),
    ]
    _write_jsonl(log_path, events)
    dataset = load_replay_dataset(log_path)

    results = evaluate_replay_models(
        dataset,
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=4,
        objective="error",
    )
    by_name = {result.model_name: result for result in results}

    assert "linear-alpha-1.0-affine-corrected" in by_name
    assert "poly2-alpha-1.0-bias-corrected" in by_name
    assert (
        by_name["linear-alpha-1.0-affine-corrected"].metrics.mean_error_px
        < by_name["linear-alpha-1.0"].metrics.mean_error_px
    )


def test_evaluate_replay_models_ranks_models_on_same_samples(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    events = [
        _replay_event("calibration_replay_sample", target_id="c0", target_x=0.0, target_y=0.0),
        _replay_event("calibration_replay_sample", target_id="c1", target_x=1.0, target_y=0.0),
        _replay_event("calibration_replay_sample", target_id="c2", target_x=0.0, target_y=1.0),
        _replay_event("calibration_replay_sample", target_id="c3", target_x=1.0, target_y=1.0),
        _replay_event("validation_replay_sample", target_id="v0", target_x=0.25, target_y=0.25),
        _replay_event("validation_replay_sample", target_id="v1", target_x=0.75, target_y=0.75),
    ]
    _write_jsonl(log_path, events)
    dataset = load_replay_dataset(log_path)

    results = evaluate_replay_models(
        dataset,
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=4,
    )

    assert {result.model_name for result in results} >= {
        "linear-alpha-0.1",
        "linear-alpha-1.0",
        "poly2-alpha-1.0",
    }
    best = min(results, key=lambda result: result.metrics.mean_error_px)
    assert best.metrics.mean_error_px < 10.0
    report = format_model_evaluation_report(results)
    assert "Model" in report
    assert "Mean Error" in report
    assert best.model_name in report
