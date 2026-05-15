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

    top_targets = _horizontal_edge_targets(prefix="top", y=margin, margin=margin)
    bottom_targets = _horizontal_edge_targets(
        prefix="bottom",
        y=1.0 - margin,
        margin=margin,
    )
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


def top_left_focus_calibration_pattern(margin: float = 0.1) -> list[CalibrationTarget]:
    """Return an experimental top-left-focused pattern for persistent v0 collapse.

    The layout keeps broad top/bottom edge coverage and adds a 3x3 local cluster
    around the top-left validation target at `(0.25, 0.25)`. It is intentionally
    opt-in and should only be promoted after live repeat-run validation.
    """

    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must satisfy 0 <= margin < 0.5")

    top_targets = _horizontal_edge_targets(prefix="top", y=margin, margin=margin)
    top_left_focus_targets = [
        CalibrationTarget(id="tl_upper_left", x=0.18, y=0.18),
        CalibrationTarget(id="tl_upper_mid", x=0.25, y=0.18),
        CalibrationTarget(id="tl_upper_right", x=0.32, y=0.18),
        CalibrationTarget(id="tl_center_left", x=0.18, y=0.25),
        CalibrationTarget(id="tl_center", x=0.25, y=0.25),
        CalibrationTarget(id="tl_center_right", x=0.32, y=0.25),
        CalibrationTarget(id="tl_lower_left", x=0.18, y=0.32),
        CalibrationTarget(id="tl_lower_mid", x=0.25, y=0.32),
        CalibrationTarget(id="tl_lower_right", x=0.32, y=0.32),
    ]
    bottom_targets = _horizontal_edge_targets(
        prefix="bottom",
        y=1.0 - margin,
        margin=margin,
    )
    return [
        *top_targets,
        *top_left_focus_targets,
        CalibrationTarget(id="upper_right", x=0.75, y=0.25),
        CalibrationTarget(id="mid_left", x=margin, y=0.5),
        CalibrationTarget(id="mid_center", x=0.5, y=0.5),
        CalibrationTarget(id="mid_right", x=1.0 - margin, y=0.5),
        CalibrationTarget(id="lower_left", x=0.25, y=0.75),
        CalibrationTarget(id="lower_right", x=0.75, y=0.75),
        *bottom_targets,
    ]


def top_row_focus_calibration_pattern(margin: float = 0.1) -> list[CalibrationTarget]:
    """Return an experimental top-row-focused pattern for v0/v1 collapse.

    The layout keeps broad top/bottom edge coverage and adds two 3x3 local
    clusters around the held-out top validation targets `(0.25, 0.25)` and
    `(0.75, 0.25)`. It is intentionally opt-in and should only be promoted
    after repeated live validation improves top-row grid accuracy without
    moving the collapse to lower validation targets.
    """

    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must satisfy 0 <= margin < 0.5")

    top_targets = _horizontal_edge_targets(prefix="top", y=margin, margin=margin)
    top_left_focus_targets = _local_cluster_targets(
        prefix="tl",
        center_x=0.25,
        center_y=0.25,
    )
    top_right_focus_targets = _local_cluster_targets(
        prefix="tr",
        center_x=0.75,
        center_y=0.25,
    )
    bottom_targets = _horizontal_edge_targets(
        prefix="bottom",
        y=1.0 - margin,
        margin=margin,
    )
    return [
        *top_targets,
        *top_left_focus_targets,
        *top_right_focus_targets,
        CalibrationTarget(id="mid_left", x=margin, y=0.5),
        CalibrationTarget(id="mid_center", x=0.5, y=0.5),
        CalibrationTarget(id="mid_right", x=1.0 - margin, y=0.5),
        CalibrationTarget(id="lower_left", x=0.25, y=0.75),
        CalibrationTarget(id="lower_right", x=0.75, y=0.75),
        *bottom_targets,
    ]


def _horizontal_edge_targets(
    *,
    prefix: str,
    y: float,
    margin: float,
) -> list[CalibrationTarget]:
    return [
        CalibrationTarget(
            id=f"{prefix}{index}",
            x=_coordinate(index=index, count=5, margin=margin),
            y=y,
        )
        for index in range(5)
    ]


def _local_cluster_targets(
    *,
    prefix: str,
    center_x: float,
    center_y: float,
    spacing: float = 0.07,
) -> list[CalibrationTarget]:
    positions = (
        ("upper_left", center_x - spacing, center_y - spacing),
        ("upper_mid", center_x, center_y - spacing),
        ("upper_right", center_x + spacing, center_y - spacing),
        ("center_left", center_x - spacing, center_y),
        ("center", center_x, center_y),
        ("center_right", center_x + spacing, center_y),
        ("lower_left", center_x - spacing, center_y + spacing),
        ("lower_mid", center_x, center_y + spacing),
        ("lower_right", center_x + spacing, center_y + spacing),
    )
    return [
        CalibrationTarget(id=f"{prefix}_{name}", x=x, y=y)
        for name, x, y in positions
    ]


def _coordinate(index: int, count: int, margin: float) -> float:
    if count == 1:
        return 0.5
    usable_range = 1.0 - (2.0 * margin)
    return margin + (usable_range * index / (count - 1))
