"""Calibration target pattern generation."""

from pupil_tracker.models import CalibrationTarget


def grid_pattern(rows: int, cols: int, margin: float = 0.1) -> list[CalibrationTarget]:
    """Generate a normalized row-major grid of calibration targets.

    Coordinates are normalized to the inclusive range `[0, 1]`. The `margin`
    controls the inset from the screen edges for the outermost targets.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    if cols <= 0:
        raise ValueError("cols must be positive")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must satisfy 0 <= margin < 0.5")

    targets: list[CalibrationTarget] = []
    for row in range(rows):
        y = _coordinate(index=row, count=rows, margin=margin)
        for col in range(cols):
            x = _coordinate(index=col, count=cols, margin=margin)
            targets.append(CalibrationTarget(id=f"r{row}c{col}", x=x, y=y))
    return targets


def _coordinate(index: int, count: int, margin: float) -> float:
    if count == 1:
        return 0.5
    usable_range = 1.0 - (2.0 * margin)
    return margin + (usable_range * index / (count - 1))
