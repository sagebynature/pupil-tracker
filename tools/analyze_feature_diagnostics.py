"""Analyze scalar calibration feature diagnostics telemetry."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class TargetDiagnostics:
    """Feature diagnostics for one calibration target."""

    target_id: str
    target_x: float
    target_y: float
    accepted_count: int
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]


@dataclass(frozen=True)
class FeatureDiagnosticsAnalysis:
    """Latest feature diagnostics and top/center/bottom separability deltas."""

    latest_line_number: int
    feature_count: int
    targets: Mapping[str, TargetDiagnostics]
    top_target_id: str
    center_target_id: str
    bottom_target_id: str
    top_vs_center_delta: tuple[float, ...]
    bottom_vs_center_delta: tuple[float, ...]


def analyze_feature_diagnostics_log(path: Path) -> FeatureDiagnosticsAnalysis:
    """Read a JSONL log and analyze the latest feature diagnostics event."""

    latest_line_number: int | None = None
    latest_payload: Mapping[str, Any] | None = None
    with path.open(encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            event = json.loads(line)
            if event.get("event_type") != "calibration_feature_diagnostics":
                continue
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                continue
            latest_line_number = line_number
            latest_payload = payload
    if latest_line_number is None or latest_payload is None:
        msg = f"no calibration_feature_diagnostics events found in {path}"
        raise ValueError(msg)
    return _analyze_payload(latest_payload, latest_line_number=latest_line_number)


def format_feature_diagnostics_report(analysis: FeatureDiagnosticsAnalysis) -> str:
    """Format diagnostics as compact, copyable text."""

    lines = [
        f"latest_line: {analysis.latest_line_number}",
        f"feature_count: {analysis.feature_count}",
        f"top_target: {analysis.top_target_id}",
        f"center_target: {analysis.center_target_id}",
        f"bottom_target: {analysis.bottom_target_id}",
        f"top_vs_center_delta: {_format_vector(analysis.top_vs_center_delta)}",
        f"bottom_vs_center_delta: {_format_vector(analysis.bottom_vs_center_delta)}",
        "targets:",
    ]
    for target in sorted(
        analysis.targets.values(),
        key=lambda item: (item.target_y, item.target_x, item.target_id),
    ):
        lines.append(
            f"  {target.target_id}: y={target.target_y:.3f} "
            f"samples={target.accepted_count} "
            f"mean={_format_vector(target.feature_mean)} "
            f"std={_format_vector(target.feature_std)}"
        )
    return "\n".join(lines)


def _analyze_payload(
    payload: Mapping[str, Any],
    *,
    latest_line_number: int,
) -> FeatureDiagnosticsAnalysis:
    feature_count = int(payload.get("feature_count", 0))
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        msg = "feature diagnostics payload must contain targets"
        raise ValueError(msg)
    targets = {
        str(target_id): _parse_target(str(target_id), raw_target)
        for target_id, raw_target in raw_targets.items()
    }
    top = _select_edge_target(targets.values(), edge="top")
    center = min(
        targets.values(),
        key=lambda target: (
            abs(target.target_y - 0.5) + abs(target.target_x - 0.5),
            target.target_id,
        ),
    )
    bottom = _select_edge_target(targets.values(), edge="bottom")
    return FeatureDiagnosticsAnalysis(
        latest_line_number=latest_line_number,
        feature_count=feature_count,
        targets=targets,
        top_target_id=top.target_id,
        center_target_id=center.target_id,
        bottom_target_id=bottom.target_id,
        top_vs_center_delta=_vector_delta(top.feature_mean, center.feature_mean),
        bottom_vs_center_delta=_vector_delta(bottom.feature_mean, center.feature_mean),
    )


def _parse_target(target_id: str, raw_target: object) -> TargetDiagnostics:
    if not isinstance(raw_target, Mapping):
        msg = f"target {target_id} must be an object"
        raise ValueError(msg)
    raw_target = cast(Mapping[str, object], raw_target)
    feature_mean = _parse_float_vector(raw_target.get("feature_mean"), field="feature_mean")
    feature_std = _parse_float_vector(raw_target.get("feature_std"), field="feature_std")
    if len(feature_mean) != len(feature_std):
        msg = f"target {target_id} has mismatched mean/std lengths"
        raise ValueError(msg)
    return TargetDiagnostics(
        target_id=str(raw_target.get("target_id", target_id)),
        target_x=_as_float(raw_target.get("target_x"), field="target_x"),
        target_y=_as_float(raw_target.get("target_y"), field="target_y"),
        accepted_count=_as_int(raw_target.get("accepted_count"), field="accepted_count"),
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, str | int | float):
        return float(value)
    msg = f"{field} must be numeric"
    raise ValueError(msg)


def _as_int(value: object, *, field: str) -> int:
    if isinstance(value, str | int):
        return int(value)
    msg = f"{field} must be an integer"
    raise ValueError(msg)


def _parse_float_vector(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"{field} must be a numeric sequence"
        raise ValueError(msg)
    values: list[float] = []
    for item in value:
        values.append(_as_float(item, field=field))
    return tuple(values)


def _select_edge_target(
    targets: Iterable[TargetDiagnostics],
    *,
    edge: str,
) -> TargetDiagnostics:
    target_list = tuple(targets)
    if edge == "top":
        edge_y = min(target.target_y for target in target_list)
    else:
        edge_y = max(target.target_y for target in target_list)
    edge_targets = [target for target in target_list if target.target_y == edge_y]
    return min(edge_targets, key=lambda target: (abs(target.target_x - 0.5), target.target_id))


def _vector_delta(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    if len(left) != len(right):
        msg = "feature vectors must have equal lengths"
        raise ValueError(msg)
    return tuple(
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _format_vector(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", type=Path, help="Path to metrics/demo.jsonl")
    args = parser.parse_args()
    analysis = analyze_feature_diagnostics_log(args.jsonl_path)
    print(format_feature_diagnostics_report(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
