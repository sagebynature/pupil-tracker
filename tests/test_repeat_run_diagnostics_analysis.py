"""Tests for repeat-run target diagnostics analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from analyze_repeat_run_diagnostics import (  # noqa: E402
    analyze_repeat_run_diagnostics_log,
    format_repeat_run_diagnostics_report,
    parse_run_range,
)


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


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
            "timestamp": 1.0,
            "x": x,
            "y": y,
            "confidence": 0.9,
            "valid": valid,
        },
    }


def test_parse_run_range_accepts_inclusive_line_ranges() -> None:
    run_range = parse_run_range("10:20")

    assert run_range.label == "lines 10-20"
    assert run_range.start_line == 10
    assert run_range.end_line == 20


def test_repeat_run_diagnostics_compare_per_target_validation_behavior(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _validation_sample("v0", target_x=0.20, target_y=0.20, x=20.0, y=20.0),
            _validation_sample("v1", target_x=0.80, target_y=0.20, x=80.0, y=20.0),
            _validation_sample("v0", target_x=0.20, target_y=0.20, x=80.0, y=85.0),
            _validation_sample("v1", target_x=0.80, target_y=0.20, x=82.0, y=22.0),
        ],
    )

    diagnostics = analyze_repeat_run_diagnostics_log(
        log_path,
        run_ranges=(parse_run_range("1:2", label="run-a"), parse_run_range("3:4", label="run-b")),
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=3,
    )

    assert [run.label for run in diagnostics.runs] == ["run-a", "run-b"]
    first_v0 = diagnostics.runs[0].targets["v0"]
    second_v0 = diagnostics.runs[1].targets["v0"]
    assert first_v0.sample_count == 1
    assert first_v0.grid_accuracy == 1.0
    assert first_v0.predicted_cell_counts == {"r0c0": 1}
    assert second_v0.grid_accuracy == 0.0
    assert second_v0.mean_signed_x_error_px == 60.0
    assert second_v0.mean_signed_y_error_px == 65.0
    assert second_v0.predicted_cell_counts == {"r2c3": 1}

    delta = diagnostics.target_deltas["v0"]
    assert delta.signed_y_delta_px == 65.0
    assert delta.grid_accuracy_delta == -1.0
    assert delta.flags == ("grid-collapse", "signed-y-shift")


def test_repeat_run_diagnostics_uses_latest_metrics_sample_window(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _validation_sample("v0", target_x=0.20, target_y=0.20, x=90.0, y=90.0),
            _validation_sample("v0", target_x=0.20, target_y=0.20, x=20.0, y=20.0),
            {
                "event_type": "validation_metrics",
                "payload": {
                    "sample_count": 1,
                    "per_target_grid_cell_accuracy": {"v0": 1.0},
                },
            },
        ],
    )

    diagnostics = analyze_repeat_run_diagnostics_log(
        log_path,
        run_ranges=(parse_run_range("1:3", label="run-a"),),
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=3,
    )

    target = diagnostics.runs[0].targets["v0"]
    assert target.sample_count == 1
    assert target.mean_error_px == 0.0
    assert target.grid_accuracy == 1.0
    assert target.predicted_cell_counts == {"r0c0": 1}


def test_repeat_run_diagnostics_uses_metrics_window_per_target(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _validation_sample("v0", target_x=0.20, target_y=0.20, x=90.0, y=90.0),
            _validation_sample("v0", target_x=0.20, target_y=0.20, x=20.0, y=20.0),
            _validation_sample("v1", target_x=0.80, target_y=0.20, x=10.0, y=90.0),
            _validation_sample("v1", target_x=0.80, target_y=0.20, x=80.0, y=20.0),
            {
                "event_type": "validation_metrics",
                "payload": {
                    "sample_count": 2,
                    "per_target_grid_cell_accuracy": {"v0": 1.0, "v1": 1.0},
                },
            },
        ],
    )

    diagnostics = analyze_repeat_run_diagnostics_log(
        log_path,
        run_ranges=(parse_run_range("1:5", label="run-a"),),
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=3,
    )

    assert diagnostics.runs[0].targets["v0"].sample_count == 1
    assert diagnostics.runs[0].targets["v0"].mean_error_px == 0.0
    assert diagnostics.runs[0].targets["v1"].sample_count == 1
    assert diagnostics.runs[0].targets["v1"].mean_error_px == 0.0


def test_repeat_run_diagnostics_report_is_markdown_and_copyable(tmp_path: Path) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            _validation_sample("v0", target_x=0.20, target_y=0.20, x=20.0, y=20.0),
            _validation_sample("v0", target_x=0.20, target_y=0.20, x=80.0, y=85.0),
        ],
    )

    diagnostics = analyze_repeat_run_diagnostics_log(
        log_path,
        run_ranges=(parse_run_range("1:1", label="run-a"), parse_run_range("2:2", label="run-b")),
        screen_width=100.0,
        screen_height=100.0,
        grid_columns=4,
        grid_rows=3,
    )

    report = format_repeat_run_diagnostics_report(diagnostics)

    assert "## Repeat-run target diagnostics" in report
    header = (
        "| Run | Target | Samples | Mean Error | Signed X | Signed Y | "
        "Grid Accuracy | Predicted Cells |"
    )
    assert header in report
    assert "| run-b | v0 | 1 | 88.46 px | +60.00 px | +65.00 px | 0.0% | r2c3=1 |" in report
    assert "| v0 | +65.00 px | -100.0% | grid-collapse, signed-y-shift |" in report
