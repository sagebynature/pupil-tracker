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

from pupil_tracker.models import CalibrationTarget, GazeSample, Rect, WindowCandidate
from pupil_tracker.telemetry import (
    JsonlLogger,
    calibration_event_payload,
    gaze_event_payload,
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

    with JsonlLogger(log_path) as logger:
        logger.write_event("calibration_sample", calibration_event_payload(target, sample_count=3))
        logger.write_event("gaze_sample", gaze_event_payload(gaze_sample))
        logger.write_event("window_candidate", window_candidate_payload(window))

    events = _read_events(log_path)

    assert [event["event_type"] for event in events] == [
        "calibration_sample",
        "gaze_sample",
        "window_candidate",
    ]
    assert events[0]["payload"] == {
        "target_id": "r1c1",
        "target_x": 0.5,
        "target_y": 0.5,
        "sample_count": 3,
    }
    assert events[2]["payload"] == {
        "app_name": "DemoApp",
        "title": "Demo Window",
        "bounds": {"x": 10.0, "y": 20.0, "width": 300.0, "height": 200.0},
        "score": 4.0,
    }


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
