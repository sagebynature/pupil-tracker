"""JSON Lines telemetry logger."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from pupil_tracker.calibration import (
    FeatureDiagnosticsSummary,
    TargetQualitySummary,
    ValidationMetrics,
    ValidationTarget,
)
from pupil_tracker.logging_config import get_logger
from pupil_tracker.models import CalibrationTarget, GazeSample, RawObservation, WindowCandidate

_LOGGER = get_logger("telemetry")


class JsonlLogger:
    """Write non-video telemetry events as one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        _LOGGER.debug("opened JSONL telemetry log at %s", path)

    def write_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Write a JSON-serializable telemetry event."""

        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "payload": dict(payload),
        }
        line = self._serialize_event(event)
        self._file.write(f"{line}\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying log file."""

        if not self._file.closed:
            self._file.close()
            _LOGGER.debug("closed JSONL telemetry log at %s", self._path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _serialize_event(event: Mapping[str, Any]) -> str:
        try:
            return json.dumps(event, sort_keys=True)
        except TypeError as error:
            msg = "telemetry event payload must be JSON serializable"
            raise TypeError(msg) from error


def raw_observation_event_payload(observation: RawObservation) -> dict[str, Any]:
    """Serialize raw tracker observation status without frame or feature vectors."""

    return {
        "timestamp": observation.timestamp,
        "valid": observation.valid,
        "confidence": observation.confidence,
        "reason": observation.reason,
    }


def gaze_event_payload(sample: GazeSample, *, frame_image: Any | None = None) -> dict[str, Any]:
    """Serialize a gaze sample without frame/image payloads."""

    del frame_image
    return {
        "timestamp": sample.timestamp,
        "x": sample.x,
        "y": sample.y,
        "confidence": sample.confidence,
        "valid": sample.valid,
        "region_id": sample.region_id,
    }


def calibration_config_payload(
    *,
    calibration_path: str,
    targets: Sequence[CalibrationTarget],
    model_name: str,
    calibration_sample_window: str,
    screen_width: float,
    screen_height: float,
    posture_stability_max_delta: float | None,
    posture_feature_indices: Sequence[int],
    stability_gate_name: str | None = None,
    stability_gate_max_delta: float | None = None,
    stability_gate_feature_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """Serialize active calibration configuration for replay/run comparisons."""

    resolved_gate_name = stability_gate_name
    if resolved_gate_name is None:
        resolved_gate_name = "posture" if posture_stability_max_delta is not None else "none"
    resolved_gate_max_delta = (
        stability_gate_max_delta
        if stability_gate_max_delta is not None
        else posture_stability_max_delta
    )
    resolved_gate_indices = (
        tuple(stability_gate_feature_indices)
        if stability_gate_feature_indices
        else tuple(posture_feature_indices)
        if resolved_gate_name == "posture"
        else ()
    )
    return {
        "calibration_path": calibration_path,
        "target_count": len(targets),
        "target_ids": [target.id for target in targets],
        "model_name": model_name,
        "calibration_sample_window": calibration_sample_window,
        "screen_width": screen_width,
        "screen_height": screen_height,
        "posture_stability_max_delta": posture_stability_max_delta,
        "posture_feature_indices": list(posture_feature_indices),
        "stability_gate_name": resolved_gate_name,
        "stability_gate_max_delta": resolved_gate_max_delta,
        "stability_gate_feature_indices": list(resolved_gate_indices),
    }


def calibration_event_payload(
    target: CalibrationTarget,
    *,
    sample_count: int,
) -> dict[str, Any]:
    """Serialize calibration progress for a target without raw frame data."""

    return {
        "target_id": target.id,
        "target_x": target.x,
        "target_y": target.y,
        "sample_count": sample_count,
    }


def calibration_replay_sample_payload(
    target: CalibrationTarget,
    observation: RawObservation,
    *,
    capture_phase: str | None = None,
    sample_accepted: bool | None = None,
    decision_reason: str | None = None,
) -> dict[str, Any]:
    """Serialize one scalar calibration sample for offline replay."""

    payload = _replay_sample_payload(target, observation)
    if capture_phase is not None:
        payload["capture_phase"] = capture_phase
    if sample_accepted is not None:
        payload["sample_accepted"] = sample_accepted
    if decision_reason is not None:
        payload["decision_reason"] = decision_reason
    return payload


def validation_replay_sample_payload(
    target: ValidationTarget,
    observation: RawObservation,
) -> dict[str, Any]:
    """Serialize one scalar validation observation for offline replay."""

    return _replay_sample_payload(target, observation)


def _replay_sample_payload(
    target: CalibrationTarget | ValidationTarget,
    observation: RawObservation,
) -> dict[str, Any]:
    return {
        "target_id": target.id,
        "target_x": target.x,
        "target_y": target.y,
        "timestamp": observation.timestamp,
        "valid": observation.valid,
        "confidence": observation.confidence,
        "feature_count": len(observation.feature_vector),
        "features": list(observation.feature_vector),
    }


def calibration_target_quality_payload(
    quality: TargetQualitySummary,
) -> dict[str, Any]:
    """Serialize target-level calibration quality without observations/features."""

    return {
        "target_id": quality.target_id,
        "accepted_count": quality.accepted_count,
        "rejected_count": quality.rejected_count,
        "mean_confidence": quality.mean_confidence,
        "meets_min_samples": quality.meets_min_samples,
        "recommendation": quality.recommendation,
    }


def feature_diagnostics_payload(summary: FeatureDiagnosticsSummary) -> dict[str, Any]:
    """Serialize scalar calibration feature diagnostics without raw samples."""

    return {
        "feature_count": summary.feature_count,
        "targets": {
            target_id: {
                "target_id": target_summary.target_id,
                "target_x": target_summary.target_x,
                "target_y": target_summary.target_y,
                "accepted_count": target_summary.accepted_count,
                "feature_mean": list(target_summary.feature_mean),
                "feature_std": list(target_summary.feature_std),
            }
            for target_id, target_summary in summary.target_summaries.items()
        },
    }


def validation_metrics_payload(metrics: ValidationMetrics) -> dict[str, Any]:
    """Serialize post-calibration validation metrics."""

    return {
        "sample_count": metrics.sample_count,
        "mean_error_px": metrics.mean_error_px,
        "median_error_px": metrics.median_error_px,
        "max_error_px": metrics.max_error_px,
        "per_target_error_px": dict(metrics.per_target_error_px),
        "mean_abs_x_error_px": metrics.mean_abs_x_error_px,
        "mean_abs_y_error_px": metrics.mean_abs_y_error_px,
        "mean_signed_y_error_px": metrics.mean_signed_y_error_px,
        "per_target_signed_y_error_px": dict(metrics.per_target_signed_y_error_px),
        "grid_cell_accuracy": metrics.grid_cell_accuracy,
        "per_target_grid_cell_accuracy": dict(metrics.per_target_grid_cell_accuracy),
        "grid_columns": metrics.grid_columns,
        "grid_rows": metrics.grid_rows,
        "recommendation": metrics.recommendation,
    }


def validation_sample_payload(
    target: ValidationTarget,
    sample: GazeSample,
) -> dict[str, Any]:
    """Serialize one validation sample without raw frame/image/feature payloads."""

    return {
        "target_id": target.id,
        "target_x": target.x,
        "target_y": target.y,
        "timestamp": sample.timestamp,
        "x": sample.x,
        "y": sample.y,
        "confidence": sample.confidence,
        "valid": sample.valid,
    }


def window_candidate_payload(candidate: WindowCandidate | None) -> dict[str, Any]:
    """Serialize a likely window candidate without changing window focus."""

    if candidate is None:
        return {"candidate": None}
    return {
        "app_name": candidate.app_name,
        "title": candidate.title,
        "bounds": {
            "x": candidate.bounds.x,
            "y": candidate.bounds.y,
            "width": candidate.bounds.width,
            "height": candidate.bounds.height,
        },
        "score": candidate.score,
    }
