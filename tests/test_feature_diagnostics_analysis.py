"""Tests for feature diagnostics telemetry analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from analyze_feature_diagnostics import (  # noqa: E402
    analyze_feature_diagnostics_log,
    format_feature_diagnostics_report,
)


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def test_analyze_feature_diagnostics_uses_latest_diagnostic_event(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "event_type": "calibration_feature_diagnostics",
                "payload": {"feature_count": 1, "targets": {}},
            },
            {"event_type": "validation_metrics", "payload": {"mean_error_px": 100.0}},
            {
                "event_type": "calibration_feature_diagnostics",
                "payload": {
                    "feature_count": 2,
                    "targets": {
                        "top": {
                            "target_id": "top",
                            "target_x": 0.5,
                            "target_y": 0.1,
                            "accepted_count": 2,
                            "feature_mean": [1.0, 10.0],
                            "feature_std": [0.1, 0.2],
                        },
                        "center": {
                            "target_id": "center",
                            "target_x": 0.5,
                            "target_y": 0.5,
                            "accepted_count": 2,
                            "feature_mean": [2.0, 20.0],
                            "feature_std": [0.3, 0.4],
                        },
                        "bottom": {
                            "target_id": "bottom",
                            "target_x": 0.5,
                            "target_y": 0.9,
                            "accepted_count": 2,
                            "feature_mean": [4.0, 25.0],
                            "feature_std": [0.5, 0.6],
                        },
                    },
                },
            },
        ],
    )

    analysis = analyze_feature_diagnostics_log(log_path)

    assert analysis.latest_line_number == 3
    assert analysis.feature_count == 2
    assert analysis.targets["top"].feature_mean == (1.0, 10.0)
    assert analysis.targets["top"].feature_std == (0.1, 0.2)
    assert analysis.top_target_id == "top"
    assert analysis.center_target_id == "center"
    assert analysis.bottom_target_id == "bottom"
    assert analysis.top_vs_center_delta == (-1.0, -10.0)
    assert analysis.bottom_vs_center_delta == (2.0, 5.0)


def test_format_feature_diagnostics_report_is_compact_and_copyable(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "demo.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "event_type": "calibration_feature_diagnostics",
                "payload": {
                    "feature_count": 2,
                    "targets": {
                        "top": {
                            "target_id": "top",
                            "target_x": 0.5,
                            "target_y": 0.1,
                            "accepted_count": 2,
                            "feature_mean": [1.0, 10.0],
                            "feature_std": [0.1, 0.2],
                        },
                        "center": {
                            "target_id": "center",
                            "target_x": 0.5,
                            "target_y": 0.5,
                            "accepted_count": 2,
                            "feature_mean": [2.0, 20.0],
                            "feature_std": [0.3, 0.4],
                        },
                        "bottom": {
                            "target_id": "bottom",
                            "target_x": 0.5,
                            "target_y": 0.9,
                            "accepted_count": 2,
                            "feature_mean": [4.0, 25.0],
                            "feature_std": [0.5, 0.6],
                        },
                    },
                },
            }
        ],
    )

    report = format_feature_diagnostics_report(analyze_feature_diagnostics_log(log_path))

    assert "latest_line: 1" in report
    assert "feature_count: 2" in report
    assert "top: y=0.100 samples=2 mean=[1.000000, 10.000000] std=[0.100000, 0.200000]" in report
    assert "top_vs_center_delta: [-1.000000, -10.000000]" in report
    assert "bottom_vs_center_delta: [2.000000, 5.000000]" in report
