"""Analyze target-level validation diagnostics across repeated live runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import NoReturn, cast


@dataclass(frozen=True)
class RunRange:
    """Inclusive line range identifying one run inside a JSONL telemetry file."""

    start_line: int
    end_line: int
    label: str


@dataclass(frozen=True)
class TargetRunSummary:
    """Validation behavior for one target inside one run."""

    target_id: str
    target_x: float
    target_y: float
    sample_count: int
    mean_error_px: float
    mean_abs_x_error_px: float
    mean_abs_y_error_px: float
    mean_signed_x_error_px: float
    mean_signed_y_error_px: float
    grid_accuracy: float
    predicted_cell_counts: Mapping[str, int]


@dataclass(frozen=True)
class RunDiagnostics:
    """All target diagnostics for one line-bounded run."""

    label: str
    start_line: int
    end_line: int
    sample_count: int
    targets: Mapping[str, TargetRunSummary]


@dataclass(frozen=True)
class TargetRunDelta:
    """Per-target delta between the first two requested runs."""

    target_id: str
    signed_y_delta_px: float
    grid_accuracy_delta: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class RepeatRunDiagnostics:
    """Repeat-run diagnostics plus first-vs-second target deltas."""

    runs: tuple[RunDiagnostics, ...]
    target_deltas: Mapping[str, TargetRunDelta]
    screen_width: float
    screen_height: float
    grid_columns: int
    grid_rows: int


@dataclass(frozen=True)
class _ValidationRow:
    target_id: str
    target_x: float
    target_y: float
    dx: float
    dy: float
    error_px: float
    matches_target_cell: bool
    predicted_cell: str


def parse_run_range(value: str, *, label: str | None = None) -> RunRange:
    """Parse an inclusive `start:end` line range."""

    parts = value.split(":", maxsplit=1)
    if len(parts) != 2:
        msg = "run range must use START:END"
        raise ValueError(msg)
    start_line = _parse_positive_int(parts[0], field="start line")
    end_line = _parse_positive_int(parts[1], field="end line")
    if end_line < start_line:
        msg = "run range end must be greater than or equal to start"
        raise ValueError(msg)
    return RunRange(
        start_line=start_line,
        end_line=end_line,
        label=label if label is not None else f"lines {start_line}-{end_line}",
    )


def analyze_repeat_run_diagnostics_log(
    path: Path,
    *,
    run_ranges: Sequence[RunRange],
    screen_width: float,
    screen_height: float,
    grid_columns: int = 4,
    grid_rows: int = 3,
) -> RepeatRunDiagnostics:
    """Read validation samples from requested line ranges and compare targets."""

    if not run_ranges:
        msg = "at least one run range is required"
        raise ValueError(msg)
    if screen_width <= 0 or screen_height <= 0:
        msg = "screen dimensions must be positive"
        raise ValueError(msg)
    if grid_columns <= 0 or grid_rows <= 0:
        msg = "grid dimensions must be positive"
        raise ValueError(msg)

    run_rows: list[list[_ValidationRow]] = [[] for _ in run_ranges]
    metrics_windows: list[tuple[int, int, tuple[str, ...]] | None] = [None for _ in run_ranges]
    with path.open(encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            matching_indices = [
                index
                for index, run_range in enumerate(run_ranges)
                if run_range.start_line <= line_number <= run_range.end_line
            ]
            if not matching_indices:
                continue
            event = _parse_event(line, line_number=line_number)
            event_type = event.get("event_type")
            payload = event.get("payload")
            if event_type == "validation_metrics" and isinstance(payload, Mapping):
                sample_count = _as_int(
                    cast(Mapping[object, object], payload).get("sample_count"),
                    field="sample_count",
                    line_number=line_number,
                )
                target_ids = _metrics_target_ids(cast(Mapping[object, object], payload))
                for index in matching_indices:
                    metrics_windows[index] = (len(run_rows[index]), sample_count, target_ids)
                continue
            if event_type != "validation_sample" or not isinstance(payload, Mapping):
                continue
            row = _parse_validation_row(
                cast(Mapping[object, object], payload),
                line_number=line_number,
                screen_width=screen_width,
                screen_height=screen_height,
                grid_columns=grid_columns,
                grid_rows=grid_rows,
            )
            if row is None:
                continue
            for index in matching_indices:
                run_rows[index].append(row)

    trimmed_run_rows = tuple(
        _select_metrics_sample_window(rows, metrics_window)
        for rows, metrics_window in zip(run_rows, metrics_windows, strict=True)
    )
    runs = tuple(
        _summarize_run(run_range, rows)
        for run_range, rows in zip(run_ranges, trimmed_run_rows, strict=True)
    )
    return RepeatRunDiagnostics(
        runs=runs,
        target_deltas=_compare_first_two_runs(runs),
        screen_width=screen_width,
        screen_height=screen_height,
        grid_columns=grid_columns,
        grid_rows=grid_rows,
    )


def format_repeat_run_diagnostics_report(diagnostics: RepeatRunDiagnostics) -> str:
    """Format repeat-run diagnostics as compact Markdown tables."""

    lines = [
        "## Repeat-run target diagnostics",
        "",
        f"Screen: {diagnostics.screen_width:.0f}x{diagnostics.screen_height:.0f}; "
        f"grid: {diagnostics.grid_columns}x{diagnostics.grid_rows}",
        "",
        "| Run | Target | Samples | Mean Error | Signed X | Signed Y | "
        "Grid Accuracy | Predicted Cells |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in diagnostics.runs:
        for target in sorted(run.targets.values(), key=lambda item: item.target_id):
            lines.append(
                "| "
                f"{run.label} | "
                f"{target.target_id} | "
                f"{target.sample_count} | "
                f"{target.mean_error_px:.2f} px | "
                f"{target.mean_signed_x_error_px:+.2f} px | "
                f"{target.mean_signed_y_error_px:+.2f} px | "
                f"{target.grid_accuracy:.1%} | "
                f"{_format_cell_counts(target.predicted_cell_counts)} |"
            )
    if diagnostics.target_deltas:
        lines.extend(
            (
                "",
                "### First-vs-second target deltas",
                "",
                "| Target | Signed Y Δ | Grid Accuracy Δ | Flags |",
                "|---|---:|---:|---|",
            )
        )
        for delta in sorted(
            diagnostics.target_deltas.values(),
            key=lambda item: ("grid-collapse" not in item.flags, item.target_id),
        ):
            flags = ", ".join(delta.flags) if delta.flags else "-"
            lines.append(
                "| "
                f"{delta.target_id} | "
                f"{delta.signed_y_delta_px:+.2f} px | "
                f"{delta.grid_accuracy_delta:+.1%} | "
                f"{flags} |"
            )
    return "\n".join(lines)


def _select_metrics_sample_window(
    rows: Sequence[_ValidationRow],
    metrics_window: tuple[int, int, tuple[str, ...]] | None,
) -> Sequence[_ValidationRow]:
    if metrics_window is None:
        return rows
    row_count_at_metric, sample_count, target_ids = metrics_window
    if sample_count <= 0:
        msg = "validation metrics sample_count must be positive"
        raise ValueError(msg)
    metric_rows = rows[:row_count_at_metric]
    if not target_ids:
        return metric_rows[-sample_count:]
    per_target_count = sample_count // len(target_ids)
    if per_target_count <= 0:
        return metric_rows[-sample_count:]
    selected: list[_ValidationRow] = []
    for target_id in target_ids:
        target_rows = [row for row in metric_rows if row.target_id == target_id]
        selected.extend(target_rows[-per_target_count:])
    return selected


def _metrics_target_ids(payload: Mapping[object, object]) -> tuple[str, ...]:
    for field in (
        "per_target_grid_cell_accuracy",
        "per_target_error_px",
        "per_target_signed_y_error_px",
    ):
        raw_value = payload.get(field)
        if isinstance(raw_value, Mapping):
            return tuple(str(target_id) for target_id in raw_value)
    return ()


def _summarize_run(run_range: RunRange, rows: Sequence[_ValidationRow]) -> RunDiagnostics:
    rows_by_target: dict[str, list[_ValidationRow]] = defaultdict(list)
    for row in rows:
        rows_by_target[row.target_id].append(row)
    targets = {
        target_id: _summarize_target(target_id, target_rows)
        for target_id, target_rows in rows_by_target.items()
    }
    return RunDiagnostics(
        label=run_range.label,
        start_line=run_range.start_line,
        end_line=run_range.end_line,
        sample_count=sum(summary.sample_count for summary in targets.values()),
        targets=targets,
    )


def _summarize_target(
    target_id: str,
    rows: Sequence[_ValidationRow],
) -> TargetRunSummary:
    if not rows:
        msg = "target rows must not be empty"
        raise ValueError(msg)
    signed_x_errors = [row.dx for row in rows]
    signed_y_errors = [row.dy for row in rows]
    errors = [row.error_px for row in rows]
    predicted_cell_counts = Counter(row.predicted_cell for row in rows)
    return TargetRunSummary(
        target_id=target_id,
        target_x=rows[0].target_x,
        target_y=rows[0].target_y,
        sample_count=len(rows),
        mean_error_px=sum(errors) / len(errors),
        mean_abs_x_error_px=sum(abs(value) for value in signed_x_errors) / len(signed_x_errors),
        mean_abs_y_error_px=sum(abs(value) for value in signed_y_errors) / len(signed_y_errors),
        mean_signed_x_error_px=sum(signed_x_errors) / len(signed_x_errors),
        mean_signed_y_error_px=sum(signed_y_errors) / len(signed_y_errors),
        grid_accuracy=sum(1 for row in rows if row.matches_target_cell) / len(rows),
        predicted_cell_counts=dict(sorted(predicted_cell_counts.items())),
    )


def _compare_first_two_runs(
    runs: Sequence[RunDiagnostics],
) -> Mapping[str, TargetRunDelta]:
    if len(runs) < 2:
        return {}
    first, second = runs[0], runs[1]
    shared_targets = set(first.targets) & set(second.targets)
    deltas: dict[str, TargetRunDelta] = {}
    for target_id in shared_targets:
        first_target = first.targets[target_id]
        second_target = second.targets[target_id]
        signed_y_delta = (
            second_target.mean_signed_y_error_px - first_target.mean_signed_y_error_px
        )
        grid_delta = second_target.grid_accuracy - first_target.grid_accuracy
        deltas[target_id] = TargetRunDelta(
            target_id=target_id,
            signed_y_delta_px=signed_y_delta,
            grid_accuracy_delta=grid_delta,
            flags=_target_delta_flags(
                first_target,
                second_target,
                signed_y_delta=signed_y_delta,
                grid_delta=grid_delta,
            ),
        )
    return deltas


def _target_delta_flags(
    first: TargetRunSummary,
    second: TargetRunSummary,
    *,
    signed_y_delta: float,
    grid_delta: float,
) -> tuple[str, ...]:
    flags: list[str] = []
    if first.grid_accuracy >= 0.5 and second.grid_accuracy == 0.0:
        flags.append("grid-collapse")
    elif first.grid_accuracy == 0.0 and second.grid_accuracy >= 0.5:
        flags.append("grid-recovery")
    elif abs(grid_delta) >= 0.5:
        flags.append("grid-shift")
    if abs(signed_y_delta) >= 50.0:
        flags.append("signed-y-shift")
    return tuple(flags)


def _parse_validation_row(
    payload: Mapping[object, object],
    *,
    line_number: int,
    screen_width: float,
    screen_height: float,
    grid_columns: int,
    grid_rows: int,
) -> _ValidationRow | None:
    if not _as_bool(payload.get("valid"), field="valid", line_number=line_number):
        return None
    target_x_normalized = _as_float(
        payload.get("target_x"),
        field="target_x",
        line_number=line_number,
    )
    target_y_normalized = _as_float(
        payload.get("target_y"),
        field="target_y",
        line_number=line_number,
    )
    target_x = target_x_normalized * screen_width
    target_y = target_y_normalized * screen_height
    predicted_x = _as_float(payload.get("x"), field="x", line_number=line_number)
    predicted_y = _as_float(payload.get("y"), field="y", line_number=line_number)
    dx = predicted_x - target_x
    dy = predicted_y - target_y
    target_cell = _grid_cell_id(
        target_x,
        target_y,
        screen_width,
        screen_height,
        columns=grid_columns,
        rows=grid_rows,
    )
    predicted_cell = _grid_cell_id(
        predicted_x,
        predicted_y,
        screen_width,
        screen_height,
        columns=grid_columns,
        rows=grid_rows,
    )
    return _ValidationRow(
        target_id=_as_str(payload.get("target_id"), field="target_id", line_number=line_number),
        target_x=target_x_normalized,
        target_y=target_y_normalized,
        dx=dx,
        dy=dy,
        error_px=hypot(dx, dy),
        matches_target_cell=target_cell == predicted_cell,
        predicted_cell=predicted_cell,
    )


def _grid_cell_id(
    x: float,
    y: float,
    screen_width: float,
    screen_height: float,
    *,
    columns: int,
    rows: int,
) -> str:
    clamped_x = min(max(x, 0.0), screen_width)
    clamped_y = min(max(y, 0.0), screen_height)
    column = min(int(clamped_x / (screen_width / columns)), columns - 1)
    row = min(int(clamped_y / (screen_height / rows)), rows - 1)
    return f"r{row}c{column}"


def _format_cell_counts(cell_counts: Mapping[str, int]) -> str:
    return ", ".join(f"{cell}={count}" for cell, count in sorted(cell_counts.items()))


def _parse_event(line: str, *, line_number: int) -> Mapping[str, object]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as error:
        msg = f"line {line_number}: invalid JSON"
        raise ValueError(msg) from error
    if not isinstance(event, Mapping):
        msg = f"line {line_number}: event must be an object"
        raise ValueError(msg)
    return cast(Mapping[str, object], event)


def _parse_positive_int(value: str, *, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"{field} must be an integer"
        raise ValueError(msg) from error
    if parsed <= 0:
        msg = f"{field} must be positive"
        raise ValueError(msg)
    return parsed


def _as_float(value: object, *, field: str, line_number: int) -> float:
    if not isinstance(value, int | float):
        msg = f"line {line_number}: {field} must be numeric"
        raise ValueError(msg)
    return float(value)


def _as_int(value: object, *, field: str, line_number: int) -> int:
    if not isinstance(value, int):
        msg = f"line {line_number}: {field} must be an integer"
        raise ValueError(msg)
    return value


def _as_str(value: object, *, field: str, line_number: int) -> str:
    if not isinstance(value, str):
        msg = f"line {line_number}: {field} must be a string"
        raise ValueError(msg)
    return value


def _as_bool(value: object, *, field: str, line_number: int) -> bool:
    if not isinstance(value, bool):
        msg = f"line {line_number}: {field} must be a boolean"
        raise ValueError(msg)
    return value


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare per-target validation diagnostics across repeated run ranges."
    )
    parser.add_argument("log_path", type=Path)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Inclusive run line range as START:END. Pass multiple times.",
    )
    parser.add_argument("--screen-width", type=float, required=True)
    parser.add_argument("--screen-height", type=float, required=True)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--grid-rows", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        run_ranges = tuple(
            parse_run_range(value, label=f"run-{index}")
            for index, value in enumerate(args.run, start=1)
        )
        diagnostics = analyze_repeat_run_diagnostics_log(
            args.log_path,
            run_ranges=run_ranges,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            grid_columns=args.grid_columns,
            grid_rows=args.grid_rows,
        )
    except ValueError as error:
        _die(str(error))
    print(format_repeat_run_diagnostics_report(diagnostics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
