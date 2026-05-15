"""Tests for offline calibration model replay evaluation."""

from __future__ import annotations

import json
import sys
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
    load_replay_dataset,
    sort_model_results,
)
from pupil_tracker.models import CalibrationSample, CalibrationTarget, RawObservation  # noqa: E402


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
