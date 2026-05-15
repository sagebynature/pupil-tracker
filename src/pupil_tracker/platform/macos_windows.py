"""macOS visible-window parsing and gaze target scoring.

This module only enumerates and scores windows. It never activates, raises,
focuses, clicks, or otherwise mutates window state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Final, cast

from pupil_tracker.models import Point2D, Rect, WindowCandidate

_OWNER_NAME_KEY: Final = "kCGWindowOwnerName"
_WINDOW_NAME_KEY: Final = "kCGWindowName"
_BOUNDS_KEY: Final = "kCGWindowBounds"
_LAYER_KEY: Final = "kCGWindowLayer"
_ALPHA_KEY: Final = "kCGWindowAlpha"
_ONSCREEN_KEY: Final = "kCGWindowIsOnscreen"
_OWNER_PID_KEY: Final = "kCGWindowOwnerPID"


RawWindowRecord = Mapping[str, Any]


def visible_window_candidates(records: Sequence[RawWindowRecord]) -> tuple[WindowCandidate, ...]:
    """Parse raw macOS window records into visible candidates.

    Records earlier in the sequence are treated as frontmost and receive a higher
    score. Hidden/minimized-like records are filtered out.
    """

    candidates: list[WindowCandidate] = []
    total = len(records)
    for index, record in enumerate(records):
        candidate = _parse_visible_window_record(record, score=float(total - index))
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def candidate_at_point(
    point: Point2D,
    candidates: Sequence[WindowCandidate],
) -> WindowCandidate | None:
    """Return the highest-scoring candidate containing `point`, if any."""

    matches = [candidate for candidate in candidates if candidate.bounds.contains(point)]
    if not matches:
        return None
    return max(matches, key=lambda candidate: candidate.score)


def list_visible_windows() -> tuple[WindowCandidate, ...]:
    """Enumerate visible macOS windows through CoreGraphics when available."""

    quartz = cast(Any, import_module("Quartz"))
    raw_records = quartz.CGWindowListCopyWindowInfo(
        quartz.kCGWindowListOptionOnScreenOnly,
        quartz.kCGNullWindowID,
    )
    records = cast(Sequence[RawWindowRecord], raw_records)
    return visible_window_candidates(records)


def _parse_visible_window_record(
    record: RawWindowRecord,
    *,
    score: float,
) -> WindowCandidate | None:
    if not _is_visible_record(record):
        return None

    bounds = _parse_bounds(record.get(_BOUNDS_KEY))
    if bounds is None:
        return None

    app_name = str(record.get(_OWNER_NAME_KEY) or "Unknown")
    title = str(record.get(_WINDOW_NAME_KEY) or "")
    process_id = _int_value(record, _OWNER_PID_KEY)
    return WindowCandidate(
        app_name=app_name,
        title=title,
        bounds=bounds,
        score=score,
        process_id=process_id,
    )


def _is_visible_record(record: RawWindowRecord) -> bool:
    if not bool(record.get(_ONSCREEN_KEY, False)):
        return False
    if int(record.get(_LAYER_KEY, 0) or 0) != 0:
        return False
    if float(record.get(_ALPHA_KEY, 1.0) or 0.0) <= 0.0:
        return False
    return bool(record.get(_OWNER_NAME_KEY))


def _parse_bounds(raw_bounds: Any) -> Rect | None:
    if not isinstance(raw_bounds, Mapping):
        return None

    x = _float_value(raw_bounds, "X")
    y = _float_value(raw_bounds, "Y")
    width = _float_value(raw_bounds, "Width")
    height = _float_value(raw_bounds, "Height")
    if x is None or y is None or width is None or height is None:
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return Rect(x=x, y=y, width=width, height=height)


def _float_value(mapping: Mapping[Any, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    return float(value)


def _int_value(mapping: Mapping[Any, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    return int(value)
