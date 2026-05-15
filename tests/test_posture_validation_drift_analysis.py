"""Tests for posture/validation feature drift analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from analyze_posture_validation_drift import (  # noqa: E402
    analyze_posture_envelope_replay_log,
    analyze_posture_validation_drift_log,
    format_posture_envelope_replay_report,
    format_posture_validation_drift_report,
    format_posture_validation_drift_run_comparison,
    main,
    parse_run_range,
)


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def _features(*, base: float = 0.0, overrides: dict[int, float] | None = None) -> tuple[float, ...]:
    values = [base] * 23
    for index, value in (overrides or {}).items():
        values[index] = value
    return tuple(values)


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


def _validation_metrics(sample_count: int, target_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "event_type": "validation_metrics",
        "payload": {
            "sample_count": sample_count,
            "per_target_grid_cell_accuracy": {target_id: 0.0 for target_id in target_ids},
        },
    }


def test_posture_validation_drift_compares_each_validation_target_to_nearest_calibration(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0_a",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={22: 0.0, 21: 1.0, 14: 0.40}),
            ),
            _calibration_replay_sample(
                "cal_v0_b",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={22: 2.0, 21: 3.0, 14: 0.42}),
            ),
            _calibration_replay_sample(
                "cal_v1",
                target_x=0.75,
                target_y=0.25,
                features=_features(overrides={22: 10.0, 21: 0.0, 14: 0.50}),
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=55.0, y=55.0),
            _validation_replay_sample(
                "v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={22: 12.0, 21: -2.0, 14: 0.60}),
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=58.0, y=55.0),
            _validation_replay_sample(
                "v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={22: 14.0, 21: -4.0, 14: 0.62}),
            ),
            _validation_sample("v1", target_x=0.75, target_y=0.25, x=75.0, y=25.0),
            _validation_replay_sample(
                "v1",
                target_x=0.75,
                target_y=0.25,
                features=_features(overrides={22: 10.001, 21: 0.0, 14: 0.50}),
            ),
            _validation_metrics(sample_count=3, target_ids=("v0", "v1")),
        ],
    )

    analysis = analyze_posture_validation_drift_log(
        log_path,
        run_range=parse_run_range("1:10", label="unit-run"),
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=4,
    )

    v0 = analysis.targets["v0"]
    assert v0.validation_summary.sample_count == 1
    assert v0.nearest_calibration_target_id == "cal_v0_a"
    assert v0.calibration_target_ids == ("cal_v0_a", "cal_v0_b")
    assert v0.predicted_cell_counts == {"r2c2": 1}
    assert v0.validation_grid_accuracy == 0.0
    assert v0.flags == ("posture-drift-grid-collapse", "posture-drift")
    assert v0.context_feature_deltas[0].feature_index == 22
    assert v0.context_feature_deltas[0].feature_name == "pitch proxy"
    assert v0.context_feature_deltas[0].signed_delta == pytest.approx(13.0)
    assert v0.context_feature_deltas[0].normalized_delta > 10.0

    v1 = analysis.targets["v1"]
    assert v1.nearest_calibration_target_id == "cal_v1"
    assert v1.validation_grid_accuracy == 1.0
    assert v1.flags == ()


def test_posture_validation_drift_uses_latest_validation_metrics_window(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features()
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=95.0, y=95.0),
            _validation_replay_sample(
                "v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 99.0})
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=25.0, y=25.0),
            _validation_replay_sample(
                "v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 3.0})
            ),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    analysis = analyze_posture_validation_drift_log(
        log_path,
        run_range=parse_run_range("1:6"),
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=4,
    )

    v0 = analysis.targets["v0"]
    assert v0.validation_summary.sample_count == 1
    assert v0.validation_summary.feature_mean[22] == 3.0
    assert v0.predicted_cell_counts == {"r1c1": 1}
    assert v0.validation_grid_accuracy == 1.0


def test_posture_validation_drift_rejects_feature_length_mismatch(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features()
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=25.0, y=25.0),
            _validation_replay_sample(
                "v0", target_x=0.25, target_y=0.25, features=_features()[:-1]
            ),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    with pytest.raises(ValueError, match="feature counts differ"):
        analyze_posture_validation_drift_log(
            log_path,
            run_range=parse_run_range("1:4"),
            screen_width=100.0,
            screen_height=100.0,
            grid_columns=4,
            grid_rows=4,
        )


def test_posture_validation_drift_requires_matching_validation_replay_samples(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features()
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=25.0, y=25.0),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    with pytest.raises(ValueError, match="no validation replay samples found for 'v0'"):
        analyze_posture_validation_drift_log(
            log_path,
            run_range=parse_run_range("1:3"),
            screen_width=100.0,
            screen_height=100.0,
            grid_columns=4,
            grid_rows=4,
        )


def test_posture_validation_drift_requires_accepted_calibration_samples(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(),
                sample_accepted=False,
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=25.0, y=25.0),
            _validation_replay_sample("v0", target_x=0.25, target_y=0.25, features=_features()),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    with pytest.raises(ValueError, match="no accepted calibration replay samples found"):
        analyze_posture_validation_drift_log(
            log_path,
            run_range=parse_run_range("1:4"),
            screen_width=100.0,
            screen_height=100.0,
            grid_columns=4,
            grid_rows=4,
        )


def test_posture_validation_drift_report_is_scalar_and_flags_collapses(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 0.0})
            ),
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 1.0})
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=55.0, y=55.0),
            _validation_replay_sample(
                "v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 12.0})
            ),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    report = format_posture_validation_drift_report(
        analyze_posture_validation_drift_log(
            log_path,
            run_range=parse_run_range("1:5", label="report-run"),
            screen_width=100.0,
            screen_height=100.0,
            grid_columns=4,
            grid_rows=4,
        )
    )

    assert "## Posture/validation drift" in report
    assert "run: report-run" in report
    assert "| v0 | cal_v0 | 2 | 1 | 0.0% | r2c2=1 |" in report
    assert "posture-drift-grid-collapse" in report
    assert "pitch proxy" in report
    assert "MEDIA:" not in report
    assert "features=[" not in report


def test_posture_validation_drift_run_comparison_lists_target_outliers(tmp_path: Path) -> None:
    first_log_path = tmp_path / "first.jsonl"
    second_log_path = tmp_path / "second.jsonl"
    _write_jsonl(
        first_log_path,
        [
            _calibration_replay_sample(
                "cal_v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={22: 0.0}),
            ),
            _calibration_replay_sample(
                "cal_v1",
                target_x=0.75,
                target_y=0.25,
                features=_features(overrides={22: 0.0}),
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=55.0, y=55.0),
            _validation_replay_sample(
                "v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={22: 0.03}),
            ),
            _validation_sample("v1", target_x=0.75, target_y=0.25, x=75.0, y=25.0),
            _validation_replay_sample(
                "v1", target_x=0.75, target_y=0.25, features=_features()
            ),
            _validation_metrics(sample_count=2, target_ids=("v0", "v1")),
        ],
    )
    _write_jsonl(
        second_log_path,
        [
            _calibration_replay_sample(
                "cal_v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={22: 0.0}),
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=25.0, y=25.0),
            _validation_replay_sample(
                "v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={22: 0.0}),
            ),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    comparison = format_posture_validation_drift_run_comparison(
        (
            analyze_posture_validation_drift_log(
                first_log_path,
                run_range=parse_run_range("1:7", label="first-run"),
                screen_width=100.0,
                screen_height=100.0,
                grid_columns=4,
                grid_rows=4,
            ),
            analyze_posture_validation_drift_log(
                second_log_path,
                run_range=parse_run_range("1:4", label="second-run"),
                screen_width=100.0,
                screen_height=100.0,
                grid_columns=4,
                grid_rows=4,
            ),
        )
    )

    assert "## Posture/validation drift run comparison" in comparison
    assert (
        "| first-run | 1-7 | v0 | 0.0% | r2c2=1 | "
        "posture-drift-grid-collapse, posture-drift |"
    ) in comparison
    assert "22 pitch proxy +0.030000 (+inf)" in comparison
    assert "| second-run | 1-4 | - | - | - | - | - |" in comparison
    assert "| first-run | 1-7 | v1 |" not in comparison
    assert "MEDIA:" not in comparison
    assert "features=[" not in comparison


def test_posture_envelope_replay_flags_validation_samples_outside_calibration_envelope(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={20: 0.10, 21: 0.20, 22: 0.00}),
            ),
            _calibration_replay_sample(
                "cal_v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={20: 0.12, 21: 0.22, 22: 0.02}),
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=55.0, y=55.0),
            _validation_replay_sample(
                "v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={20: 0.11, 21: 0.21, 22: 0.03}),
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=58.0, y=55.0),
            _validation_replay_sample(
                "v0",
                target_x=0.25,
                target_y=0.25,
                features=_features(overrides={20: 0.11, 21: 0.21, 22: 0.10}),
            ),
            _validation_metrics(sample_count=2, target_ids=("v0",)),
        ],
    )

    analysis = analyze_posture_envelope_replay_log(
        log_path,
        run_range=parse_run_range("1:7", label="envelope-run"),
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=4,
        envelope_feature_indices=(20, 21, 22),
        envelope_padding=0.0,
    )

    v0 = analysis.targets["v0"]
    assert v0.validation_sample_count == 2
    assert v0.outside_envelope_count == 2
    assert v0.outside_envelope_rate == 1.0
    assert v0.validation_grid_accuracy == 0.0
    assert v0.flags == ("posture-envelope-grid-collapse", "posture-outside-envelope")
    assert v0.feature_breaches[0].feature_index == 22
    assert v0.feature_breaches[0].outside_sample_count == 2
    assert v0.feature_breaches[0].max_excess == pytest.approx(0.08)


def test_posture_envelope_replay_uses_latest_validation_metrics_window(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 0.0})
            ),
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 0.02})
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=95.0, y=95.0),
            _validation_replay_sample(
                "v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 99.0})
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=25.0, y=25.0),
            _validation_replay_sample(
                "v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 0.01})
            ),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    analysis = analyze_posture_envelope_replay_log(
        log_path,
        run_range=parse_run_range("1:7"),
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=4,
        envelope_feature_indices=(22,),
    )

    v0 = analysis.targets["v0"]
    assert v0.validation_sample_count == 1
    assert v0.outside_envelope_count == 0
    assert v0.predicted_cell_counts == {"r1c1": 1}
    assert v0.validation_grid_accuracy == 1.0
    assert v0.flags == ()


def test_posture_envelope_report_is_scalar_only(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 0.0})
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=55.0, y=55.0),
            _validation_replay_sample(
                "v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 0.10})
            ),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    report = format_posture_envelope_replay_report(
        analyze_posture_envelope_replay_log(
            log_path,
            run_range=parse_run_range("1:4", label="report-run"),
            screen_width=100.0,
            screen_height=100.0,
            grid_columns=4,
            grid_rows=4,
            envelope_feature_indices=(22,),
        )
    )

    assert "## Posture envelope replay" in report
    assert "run: report-run" in report
    assert "| v0 | cal_v0 | 1 | 1/1 | 100.0% | 0.0% | r2c2=1 |" in report
    assert "posture-envelope-grid-collapse" in report
    assert "22 pitch proxy" in report
    assert "MEDIA:" not in report
    assert "features=[" not in report


def test_posture_envelope_replay_rejects_invalid_feature_index(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features()
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=25.0, y=25.0),
            _validation_replay_sample("v0", target_x=0.25, target_y=0.25, features=_features()),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    with pytest.raises(ValueError, match="envelope feature index 23 is out of range"):
        analyze_posture_envelope_replay_log(
            log_path,
            run_range=parse_run_range("1:4"),
            screen_width=100.0,
            screen_height=100.0,
            grid_columns=4,
            grid_rows=4,
            envelope_feature_indices=(23,),
        )


def test_cli_can_emit_posture_envelope_replay_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _calibration_replay_sample(
                "cal_v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 0.0})
            ),
            _validation_sample("v0", target_x=0.25, target_y=0.25, x=55.0, y=55.0),
            _validation_replay_sample(
                "v0", target_x=0.25, target_y=0.25, features=_features(overrides={22: 0.10})
            ),
            _validation_metrics(sample_count=1, target_ids=("v0",)),
        ],
    )

    exit_code = main(
        (
            str(log_path),
            "--run",
            "1:4",
            "--screen-width",
            "100",
            "--screen-height",
            "100",
            "--grid-columns",
            "4",
            "--grid-rows",
            "4",
            "--posture-envelope",
            "--envelope-feature-index",
            "22",
        )
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "## Posture envelope replay" in output
    assert "22 pitch proxy" in output
    assert "## Posture/validation drift" not in output
