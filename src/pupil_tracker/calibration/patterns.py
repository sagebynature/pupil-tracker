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


def vertical_grid_pattern(margin: float = 0.1) -> list[CalibrationTarget]:
    """Return a denser 3-column by 5-row pattern for vertical accuracy checks."""

    return grid_pattern(rows=5, cols=3, margin=margin)


def edge_dense_calibration_pattern(margin: float = 0.1) -> list[CalibrationTarget]:
    """Return an experimental edge-dense pattern for corner/edge validation.

    The layout preserves inset screen edges while adding extra top/bottom edge
    samples plus validation-like upper/lower quadrant points. It is intended as
    an opt-in live calibration geometry experiment, not the default pattern.
    """

    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must satisfy 0 <= margin < 0.5")

    top_targets = [
        CalibrationTarget(
            id=f"top{index}",
            x=_coordinate(index=index, count=5, margin=margin),
            y=margin,
        )
        for index in range(5)
    ]
    bottom_targets = [
        CalibrationTarget(
            id=f"bottom{index}",
            x=_coordinate(index=index, count=5, margin=margin),
            y=1.0 - margin,
        )
        for index in range(5)
    ]
    return [
        *top_targets,
        CalibrationTarget(id="upper_left", x=0.25, y=0.25),
        CalibrationTarget(id="upper_right", x=0.75, y=0.25),
        CalibrationTarget(id="mid_left", x=margin, y=0.5),
        CalibrationTarget(id="mid_center", x=0.5, y=0.5),
        CalibrationTarget(id="mid_right", x=1.0 - margin, y=0.5),
        CalibrationTarget(id="lower_left", x=0.25, y=0.75),
        CalibrationTarget(id="lower_right", x=0.75, y=0.75),
        *bottom_targets,
    ]


def _coordinate(index: int, count: int, margin: float) -> float:
    if count == 1:
        return 0.5
    usable_range = 1.0 - (2.0 * margin)
    return margin + (usable_range * index / (count - 1))
