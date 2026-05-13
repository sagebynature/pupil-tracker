"""Tests for JSONL telemetry logging."""

import json

import numpy as np
import pytest

from pupil_tracker.telemetry import JsonlLogger


def _read_json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writing_two_events_produces_two_json_lines(tmp_path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    logger = JsonlLogger(log_path)

    logger.write_event("calibration_started", {"target_count": 9})
    logger.write_event("sample_collected", {"target_id": "r1c1"})
    logger.close()

    events = _read_json_lines(log_path)
    assert [event["event_type"] for event in events] == [
        "calibration_started",
        "sample_collected",
    ]
    assert events[0]["payload"] == {"target_count": 9}
    assert events[1]["payload"] == {"target_id": "r1c1"}


def test_event_contains_event_type_and_timestamp(tmp_path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    logger = JsonlLogger(log_path)

    logger.write_event("runtime_step", {"confidence": 0.8})
    logger.close()

    event = _read_json_lines(log_path)[0]
    assert event["event_type"] == "runtime_step"
    assert isinstance(event["timestamp"], float)
    assert event["timestamp"] > 0.0


def test_payload_must_be_json_serializable(tmp_path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    logger = JsonlLogger(log_path)

    logger.write_event("serializable", {"values": [1, 2, 3], "ok": True})
    logger.close()

    event = _read_json_lines(log_path)[0]
    assert event["payload"] == {"values": [1, 2, 3], "ok": True}


def test_numpy_arrays_are_rejected_by_default(tmp_path) -> None:
    log_path = tmp_path / "metrics.jsonl"
    logger = JsonlLogger(log_path)

    with pytest.raises(TypeError, match="JSON serializable"):
        logger.write_event("frame", {"image": np.zeros((2, 2), dtype=np.uint8)})

    logger.close()
    assert log_path.read_text(encoding="utf-8") == ""


def test_context_manager_closes_file(tmp_path) -> None:
    log_path = tmp_path / "metrics.jsonl"

    with JsonlLogger(log_path) as logger:
        logger.write_event("done", {})

    assert _read_json_lines(log_path)[0]["event_type"] == "done"
