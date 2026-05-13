"""Tests for confidence-aware gaze overlay state."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pupil_tracker.models import GazeSample

if TYPE_CHECKING:
    from desktop_demo.ui.overlay import OverlayState

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


def _overlay(
    *,
    dot_radius: float = 6.0,
    min_halo_radius: float = 14.0,
    max_halo_radius: float = 48.0,
    max_trail_length: int = 30,
) -> OverlayState:
    from desktop_demo.ui.overlay import OverlayState

    return OverlayState(
        dot_radius=dot_radius,
        min_halo_radius=min_halo_radius,
        max_halo_radius=max_halo_radius,
        max_trail_length=max_trail_length,
    )


def _sample(
    confidence: float,
    *,
    valid: bool = True,
    x: float = 100.0,
    y: float = 200.0,
) -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=x,
        y=y,
        confidence=confidence,
        valid=valid,
        region_id="middle_center",
    )


def test_high_confidence_produces_smaller_halo_than_low_confidence() -> None:
    overlay = _overlay()

    high = overlay.render_state_for(_sample(0.95))
    low = overlay.render_state_for(_sample(0.2))

    assert high.visible
    assert low.visible
    assert high.halo_radius < low.halo_radius
    assert high.dot_radius == low.dot_radius


def test_invalid_sample_hides_cursor() -> None:
    overlay = _overlay()

    state = overlay.render_state_for(_sample(0.0, valid=False))

    assert not state.visible
    assert state.opacity == 0.0
    assert state.halo_radius == 0.0


def test_confidence_is_clamped_for_rendering() -> None:
    overlay = _overlay(min_halo_radius=12.0, max_halo_radius=40.0)

    overconfident = overlay.render_state_for(_sample(2.0))
    underconfident = overlay.render_state_for(_sample(-1.0))

    assert overconfident.halo_radius == 12.0
    assert underconfident.halo_radius == 40.0


def test_debug_trail_keeps_bounded_history_of_valid_samples() -> None:
    overlay = _overlay(max_trail_length=3)

    overlay.update(_sample(0.9, x=1.0, y=1.0))
    overlay.update(_sample(0.9, x=2.0, y=2.0))
    overlay.update(_sample(0.9, x=3.0, y=3.0))
    overlay.update(_sample(0.0, valid=False, x=4.0, y=4.0))
    overlay.update(_sample(0.9, x=5.0, y=5.0))

    assert [(point.x, point.y) for point in overlay.trail] == [(2.0, 2.0), (3.0, 3.0), (5.0, 5.0)]
