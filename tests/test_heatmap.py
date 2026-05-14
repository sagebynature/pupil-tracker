"""Tests for gaze heatmap accumulation."""

from __future__ import annotations

import pytest

from pupil_tracker.models import GazeSample
from pupil_tracker.screen import GazeHeatmap, HeatmapConfig


def sample(*, x: float, y: float, valid: bool = True) -> GazeSample:
    return GazeSample(
        timestamp=1.0,
        x=x,
        y=y,
        confidence=0.9,
        valid=valid,
    )


def test_valid_gaze_increments_expected_cell() -> None:
    heatmap = GazeHeatmap(
        HeatmapConfig(screen_width=100.0, screen_height=100.0, cols=10, rows=10)
    )

    heatmap.add(sample(x=25.0, y=35.0))

    cells = heatmap.normalized_cells()
    assert cells[3][2] == 1.0
    assert sum(sum(row) for row in cells) == 1.0


def test_invalid_gaze_is_ignored() -> None:
    heatmap = GazeHeatmap(
        HeatmapConfig(screen_width=100.0, screen_height=100.0, cols=10, rows=10)
    )

    heatmap.add(sample(x=25.0, y=35.0, valid=False))

    assert sum(sum(row) for row in heatmap.normalized_cells()) == 0.0


def test_decay_reduces_intensity() -> None:
    heatmap = GazeHeatmap(
        HeatmapConfig(
            screen_width=100.0,
            screen_height=100.0,
            cols=10,
            rows=10,
            decay=0.5,
        )
    )
    heatmap.add(sample(x=25.0, y=35.0))

    heatmap.decay()

    assert heatmap.normalized_cells()[3][2] == pytest.approx(0.5)


def test_normalized_output_max_is_never_above_one() -> None:
    heatmap = GazeHeatmap(
        HeatmapConfig(screen_width=100.0, screen_height=100.0, cols=10, rows=10)
    )

    heatmap.add(sample(x=25.0, y=35.0))
    heatmap.add(sample(x=25.0, y=35.0))
    heatmap.add(sample(x=75.0, y=75.0))

    cells = heatmap.normalized_cells()

    assert cells[3][2] == 1.0
    assert cells[7][7] == pytest.approx(0.5)
    assert max(max(row) for row in cells) <= 1.0


def test_out_of_bounds_samples_are_clamped_to_screen_edges() -> None:
    heatmap = GazeHeatmap(
        HeatmapConfig(screen_width=100.0, screen_height=100.0, cols=10, rows=10)
    )

    heatmap.add(sample(x=-50.0, y=150.0))

    assert heatmap.normalized_cells()[9][0] == 1.0


def test_config_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="screen dimensions must be positive"):
        HeatmapConfig(screen_width=0.0, screen_height=100.0)

    with pytest.raises(ValueError, match="grid dimensions must be positive"):
        HeatmapConfig(screen_width=100.0, screen_height=100.0, cols=0)

    with pytest.raises(ValueError, match="decay must be between 0 and 1"):
        HeatmapConfig(screen_width=100.0, screen_height=100.0, decay=1.5)
