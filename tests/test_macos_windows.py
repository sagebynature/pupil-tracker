"""Tests for macOS visible-window parsing and candidate scoring."""

from __future__ import annotations

from typing import Any

import pytest

from pupil_tracker.models import Point2D
from pupil_tracker.platform import macos_windows
from pupil_tracker.platform.macos_windows import candidate_at_point, visible_window_candidates


def _record(
    *,
    app_name: str = "DemoApp",
    title: str = "Demo Window",
    x: float = 0.0,
    y: float = 0.0,
    width: float = 100.0,
    height: float = 100.0,
    layer: int = 0,
    alpha: float = 1.0,
    onscreen: bool = True,
) -> dict[str, Any]:
    return {
        "kCGWindowOwnerName": app_name,
        "kCGWindowName": title,
        "kCGWindowBounds": {
            "X": x,
            "Y": y,
            "Width": width,
            "Height": height,
        },
        "kCGWindowLayer": layer,
        "kCGWindowAlpha": alpha,
        "kCGWindowIsOnscreen": onscreen,
    }


def test_point_inside_one_window_returns_that_candidate() -> None:
    candidates = visible_window_candidates([
        _record(app_name="Notes", title="Plan", x=10, y=20, width=200, height=100)
    ])

    candidate = candidate_at_point(Point2D(50, 50), candidates)

    assert candidate is not None
    assert candidate.app_name == "Notes"
    assert candidate.title == "Plan"
    assert candidate.process_id is None


def test_window_record_process_id_is_preserved_when_available() -> None:
    record = _record(app_name="Notes", title="Plan")
    record["kCGWindowOwnerPID"] = 4242

    candidates = visible_window_candidates([record])

    assert candidates[0].process_id == 4242


def test_list_visible_windows_returns_empty_when_quartz_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_quartz(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(macos_windows, "import_module", missing_quartz)

    assert macos_windows.list_visible_windows() == ()


def test_overlapping_windows_choose_higher_frontmost_score() -> None:
    candidates = visible_window_candidates([
        _record(app_name="FrontApp", title="Front", x=0, y=0, width=100, height=100),
        _record(app_name="BackApp", title="Back", x=0, y=0, width=100, height=100),
    ])

    candidate = candidate_at_point(Point2D(25, 25), candidates)

    assert candidate is not None
    assert candidate.app_name == "FrontApp"
    assert candidates[0].score > candidates[1].score


def test_off_window_point_returns_none() -> None:
    candidates = visible_window_candidates([
        _record(app_name="Notes", title="Plan", x=10, y=20, width=200, height=100)
    ])

    assert candidate_at_point(Point2D(500, 500), candidates) is None


def test_hidden_or_minimized_like_records_are_filtered_by_parser_rules() -> None:
    candidates = visible_window_candidates([
        _record(app_name="Visible", title="Window"),
        _record(app_name="Hidden", title="Offscreen", onscreen=False),
        _record(app_name="Transparent", title="Invisible", alpha=0.0),
        _record(app_name="Overlay", title="NonNormalLayer", layer=1),
        _record(app_name="Collapsed", title="NoSize", width=0.0),
    ])

    assert len(candidates) == 1
    assert candidates[0].app_name == "Visible"
