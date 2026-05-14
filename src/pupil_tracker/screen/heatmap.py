"""Lightweight gaze heatmap accumulation without frame/image payloads."""

from __future__ import annotations

from dataclasses import dataclass

from pupil_tracker.models import GazeSample


@dataclass(frozen=True)
class HeatmapConfig:
    """Screen and grid configuration for a gaze heatmap."""

    screen_width: float
    screen_height: float
    cols: int = 64
    rows: int = 36
    decay: float = 0.95

    def __post_init__(self) -> None:
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("screen dimensions must be positive")
        if self.cols <= 0 or self.rows <= 0:
            raise ValueError("grid dimensions must be positive")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")


class GazeHeatmap:
    """Accumulate valid gaze samples into a decaying grid."""

    def __init__(self, config: HeatmapConfig) -> None:
        self.config = config
        self._cells = [
            [0.0 for _ in range(config.cols)] for _ in range(config.rows)
        ]

    def add(self, sample: GazeSample) -> None:
        """Add a valid gaze sample to the heatmap, clamping to screen edges."""

        if not sample.valid:
            return
        row, col = self._cell_for_sample(sample)
        self._cells[row][col] += 1.0

    def decay(self) -> None:
        """Apply configured decay to all accumulated cells."""

        for row_index, row in enumerate(self._cells):
            self._cells[row_index] = [value * self.config.decay for value in row]

    def clear(self) -> None:
        """Reset all accumulated heatmap values."""

        for row_index in range(self.config.rows):
            self._cells[row_index] = [0.0 for _ in range(self.config.cols)]

    def normalized_cells(self) -> tuple[tuple[float, ...], ...]:
        """Return grid values normalized to a maximum of one."""

        max_value = max((max(row) for row in self._cells), default=0.0)
        denominator = max(1.0, max_value)
        return tuple(
            tuple(value / denominator for value in row)
            for row in self._cells
        )

    def _cell_for_sample(self, sample: GazeSample) -> tuple[int, int]:
        x = _clamp(sample.x, 0.0, self.config.screen_width)
        y = _clamp(sample.y, 0.0, self.config.screen_height)
        col = min(
            self.config.cols - 1,
            int((x / self.config.screen_width) * self.config.cols),
        )
        row = min(
            self.config.rows - 1,
            int((y / self.config.screen_height) * self.config.rows),
        )
        return row, col


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
