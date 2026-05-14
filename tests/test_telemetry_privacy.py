"""Tests for privacy-conscious demo telemetry controls."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pupil_tracker.calibration import TargetQualitySummary, ValidationMetrics
from pupil_tracker.models import CalibrationTarget, GazeSample, Rect, WindowCandidate
from pupil_tracker.telemetry import (
    JsonlLogger,
    calibration_event_payload,
    calibration_target_quality_payload,
    gaze_event_payload,
    validation_metrics_payload,
    window_candidate_payload,
)

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


def _read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def qt_app() -> Iterator[QApplication]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    app = cast(QApplication, existing) if existing is not None else QApplication([])
    yield app


def test_gaze_event_payload_omits_frame_image_arrays() -> None:
    sample = GazeSample(
        timestamp=1.5,
        x=320.0,
        y=240.0,
        confidence=0.8,
        valid=True,
        region_id="middle_center",
    )

    payload = gaze_event_payload(sample, frame_image=np.zeros((2, 2, 3), dtype=np.uint8))

    assert payload == {
        "timestamp": 1.5,
        "x": 320.0,
        "y": 240.0,
        "confidence": 0.8,
        "valid": True,
        "region_id": "middle_center",
    }
    assert "frame_image" not in payload
    assert "image" not in payload


def test_calibration_gaze_and_window_events_serialize_as_json(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    target = CalibrationTarget(id="r1c1", x=0.5, y=0.5)
    gaze_sample = GazeSample(timestamp=2.0, x=100.0, y=150.0, confidence=0.7, valid=True)
    window = WindowCandidate(
        app_name="DemoApp",
        title="Demo Window",
        bounds=Rect(x=10.0, y=20.0, width=300.0, height=200.0),
        score=4.0,
    )
    quality = TargetQualitySummary(
        target_id="r1c1",
        accepted_count=20,
        rejected_count=2,
        mean_confidence=0.82,
        meets_min_samples=True,
        recommendation="advance",
    )
    metrics = ValidationMetrics(
        sample_count=5,
        mean_error_px=40.0,
        median_error_px=35.0,
        max_error_px=80.0,
        per_target_error_px={"v0": 20.0, "v1": 60.0},
        mean_abs_x_error_px=15.0,
        mean_abs_y_error_px=25.0,
        mean_signed_y_error_px=-10.0,
        per_target_signed_y_error_px={"v0": -5.0, "v1": -15.0},
        recommendation="excellent",
    )

    with JsonlLogger(log_path) as logger:
        logger.write_event("calibration_sample", calibration_event_payload(target, sample_count=3))
        logger.write_event(
            "calibration_target_quality",
            calibration_target_quality_payload(quality),
        )
        logger.write_event("gaze_sample", gaze_event_payload(gaze_sample))
        logger.write_event("validation_metrics", validation_metrics_payload(metrics))
        logger.write_event("window_candidate", window_candidate_payload(window))

    events = _read_events(log_path)

    assert [event["event_type"] for event in events] == [
        "calibration_sample",
        "calibration_target_quality",
        "gaze_sample",
        "validation_metrics",
        "window_candidate",
    ]
    assert events[0]["payload"] == {
        "target_id": "r1c1",
        "target_x": 0.5,
        "target_y": 0.5,
        "sample_count": 3,
    }
    assert events[1]["payload"] == {
        "target_id": "r1c1",
        "accepted_count": 20,
        "rejected_count": 2,
        "mean_confidence": 0.82,
        "meets_min_samples": True,
        "recommendation": "advance",
    }
    assert events[3]["payload"] == {
        "sample_count": 5,
        "mean_error_px": 40.0,
        "median_error_px": 35.0,
        "max_error_px": 80.0,
        "per_target_error_px": {"v0": 20.0, "v1": 60.0},
        "mean_abs_x_error_px": 15.0,
        "mean_abs_y_error_px": 25.0,
        "mean_signed_y_error_px": -10.0,
        "per_target_signed_y_error_px": {"v0": -5.0, "v1": -15.0},
        "recommendation": "excellent",
    }
    assert events[4]["payload"] == {
        "app_name": "DemoApp",
        "title": "Demo Window",
        "bounds": {"x": 10.0, "y": 20.0, "width": 300.0, "height": 200.0},
        "score": 4.0,
    }
    event_json = json.dumps(events)
    assert "image" not in event_json
    assert "frame" not in event_json
    assert "feature_vector" not in event_json


def test_demo_start_stop_logging_flushes_file(qt_app: QApplication, tmp_path: Path) -> None:
    from desktop_demo.ui.main_window import MainWindow

    del qt_app
    log_path = tmp_path / "demo.jsonl"
    window = MainWindow(telemetry_path=log_path)

    assert window.telemetry_logger is None
    assert window.start_logging_button.text() == "Start Logging"
    assert window.stop_logging_button.text() == "Stop Logging"

    window.start_logging()
    window.log_telemetry_event("manual", {"ok": True})
    window.stop_logging()

    assert window.telemetry_logger is None
    assert _read_events(log_path)[0]["payload"] == {"ok": True}
