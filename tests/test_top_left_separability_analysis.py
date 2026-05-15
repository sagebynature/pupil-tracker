"""Tests for top-left calibration/validation feature separability analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from analyze_top_left_separability import (  # noqa: E402
    analyze_top_left_separability_log,
    format_top_left_separability_report,
    parse_run_range,
)


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def _calibration_replay_sample(
    target_id: str,
    *,
    target_x: float,
    target_y: float,
    features: tuple[float, ...],
    valid: bool = True,
    sample_accepted: bool = True,
) -> dict[str, object]:
    return {
        "event_type": "calibration_replay_sample",
        "payload": {
            "target_id": target_id,
            "target_x": target_x,
            "target_y": target_y,
            "timestamp": 1.0,
            "valid": valid,
            "confidence": 0.9,
            "feature_count": len(features),
            "features": list(features),
            "sample_accepted": sample_accepted,
        },
    }


def _validation_replay_sample(
    target_id: str,
    *,
    target_x: float,
    target_y: float,
    features: tuple[float, ...],
    valid: bool = True,
) -> dict[str, object]:
    return {
        "event_type": "validation_replay_sample",
        "payload": {
            "target_id": target_id,
            "target_x": target_x,
            "target_y": target_y,
            "timestamp": 2.0,
            "valid": valid,
            "confidence": 0.9,
            "feature_count": len(features),
            "features": list(features),
        },
    }


def _validation_sample(
    target_id: str,
    *,
    target_x: float,
    target_y: float,
    x: float,
    y: float,
    valid: bool = True,
) -> dict[str, object]:
    return {
        "event_type": "validation_sample",
        "payload": {
            "target_id": target_id,
            "target_x": target_x,
            "target_y": target_y,
            "timestamp": 2.0,
            "valid": valid,
            "confidence": 0.9,
            "x": x,
            "y": y,
        },
    }


def _validation_metrics(
    sample_count: int,
    target_ids: tuple[str, ...] = ("v0",),
) -> dict[str, object]:
    return {
        "event_type": "validation_metrics",
        "payload": {
            "sample_count": sample_count,
            "per_target_grid_cell_accuracy": {target_id: 0.0 for target_id in target_ids},
        },
    }


def test_top_left_separability_compares_cluster_to_v0_validation_samples(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "tl_settling",
                target_x=0.25,
                target_y=0.25,
                features=(100.0, 100.0, 100.0),
                sample_accepted=False,
            ),
            _calibration_replay_sample(
                "tl_center", target_x=0.25, target_y=0.25, features=(1.0, 10.0, 100.0)
            ),
            _calibration_replay_sample(
                "tl_upper", target_x=0.25, target_y=0.18, features=(9.0, 12.0, 110.0)
            ),
            _calibration_replay_sample(
                "mid_center", target_x=0.50, target_y=0.50, features=(50.0, 50.0, 50.0)
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=20.0, y=40.0),
            _validation_replay_sample(
                "v0",
                target_x=0.25,
                target_y=0.25,
                features=(5.0, 9.0, 130.0),
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=35.0, y=40.0),
            _validation_replay_sample(
                "v0",
                target_x=0.25,
                target_y=0.25,
                features=(7.0, 13.0, 170.0),
            ),
            _validation_metrics(sample_count=2),
        ],
    )

    analysis = analyze_top_left_separability_log(
        log_path,
        run_range=parse_run_range("1:9", label="unit-run"),
        screen_width=100.0,
        screen_height=90.0,
        grid_columns=4,
        grid_rows=3,
        cluster_center=(0.25, 0.25),
        cluster_radius=0.10,
        validation_target_id="v0",
    )

    assert analysis.label == "unit-run"
    assert analysis.calibration_summary.sample_count == 2
    assert analysis.calibration_target_ids == ("tl_center", "tl_upper")
    assert analysis.calibration_summary.feature_mean == pytest.approx((5.0, 11.0, 105.0))
    assert analysis.calibration_summary.feature_min == pytest.approx((1.0, 10.0, 100.0))
    assert analysis.calibration_summary.feature_max == pytest.approx((9.0, 12.0, 110.0))
    assert analysis.validation_summary.sample_count == 2
    assert analysis.validation_summary.feature_mean == pytest.approx((6.0, 11.0, 150.0))
    assert analysis.validation_predicted_cell_counts == {"r1c0": 1, "r1c1": 1}
    assert analysis.validation_grid_accuracy == 0.0
    assert analysis.assessment == "separable"

    dominant = analysis.dominant_feature_deltas[0]
    assert dominant.feature_index == 2
    assert dominant.feature_name == "right iris face-relative X"
    assert dominant.signed_delta == pytest.approx(45.0)
    assert dominant.normalized_delta > 2.0


def test_top_left_separability_uses_latest_metrics_window_for_validation(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "tl_center", target_x=0.25, target_y=0.25, features=(1.0, 1.0)
            ),
            _calibration_replay_sample(
                "tl_lower", target_x=0.25, target_y=0.32, features=(2.0, 2.0)
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=95.0, y=80.0),
            _validation_replay_sample("v0", target_x=0.25, target_y=0.25, features=(99.0, 99.0)),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=20.0, y=40.0),
            _validation_replay_sample("v0", target_x=0.25, target_y=0.25, features=(3.0, 3.0)),
            _validation_metrics(sample_count=1),
        ],
    )

    analysis = analyze_top_left_separability_log(
        log_path,
        run_range=parse_run_range("1:7"),
        screen_width=100.0,
        screen_height=90.0,
        grid_columns=4,
        grid_rows=3,
    )

    assert analysis.validation_summary.sample_count == 1
    assert analysis.validation_summary.feature_mean == pytest.approx((3.0, 3.0))
    assert analysis.validation_predicted_cell_counts == {"r1c0": 1}


def test_top_left_separability_report_is_scalar_and_copyable(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "tl_center", target_x=0.25, target_y=0.25, features=(1.0, 1.0)
            ),
            _calibration_replay_sample(
                "tl_lower", target_x=0.25, target_y=0.32, features=(2.0, 2.0)
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=20.0, y=40.0),
            _validation_replay_sample("v0", target_x=0.25, target_y=0.25, features=(3.0, 5.0)),
            _validation_metrics(sample_count=1),
        ],
    )

    report = format_top_left_separability_report(
        analyze_top_left_separability_log(
            log_path,
            run_range=parse_run_range("1:5"),
            screen_width=100.0,
            screen_height=90.0,
            grid_columns=4,
            grid_rows=3,
        )
    )

    assert "## Top-left separability" in report
    assert "calibration_cluster_samples: 2" in report
    assert "validation_target: v0" in report
    assert "validation_predicted_cells: r1c0=1" in report
    assert "| Feature | Calibration Mean | Validation Mean | Signed Δ | Normalized Δ |" in report
    assert "MEDIA:" not in report
