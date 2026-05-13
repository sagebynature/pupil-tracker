import pytest

from pupil_tracker.calibration.patterns import grid_pattern


def test_grid_pattern_returns_stable_nine_point_grid() -> None:
    targets = grid_pattern(rows=3, cols=3, margin=0.1)

    assert len(targets) == 9
    assert [target.id for target in targets] == [
        "r0c0",
        "r0c1",
        "r0c2",
        "r1c0",
        "r1c1",
        "r1c2",
        "r2c0",
        "r2c1",
        "r2c2",
    ]


def test_grid_pattern_coordinates_are_normalized_with_center_target() -> None:
    targets = grid_pattern(rows=3, cols=3, margin=0.1)

    assert all(0.0 <= target.x <= 1.0 for target in targets)
    assert all(0.0 <= target.y <= 1.0 for target in targets)
    assert targets[0].x == pytest.approx(0.1)
    assert targets[0].y == pytest.approx(0.1)
    assert targets[4].x == pytest.approx(0.5)
    assert targets[4].y == pytest.approx(0.5)
    assert targets[8].x == pytest.approx(0.9)
    assert targets[8].y == pytest.approx(0.9)


def test_grid_pattern_supports_single_row_or_column() -> None:
    horizontal = grid_pattern(rows=1, cols=3, margin=0.2)
    vertical = grid_pattern(rows=3, cols=1, margin=0.2)

    assert [(target.x, target.y) for target in horizontal] == [
        pytest.approx((0.2, 0.5)),
        pytest.approx((0.5, 0.5)),
        pytest.approx((0.8, 0.5)),
    ]
    assert [(target.x, target.y) for target in vertical] == [
        pytest.approx((0.5, 0.2)),
        pytest.approx((0.5, 0.5)),
        pytest.approx((0.5, 0.8)),
    ]


@pytest.mark.parametrize(
    ("rows", "cols", "margin"),
    [
        (0, 3, 0.1),
        (3, 0, 0.1),
        (-1, 3, 0.1),
        (3, -1, 0.1),
        (3, 3, -0.1),
        (3, 3, 0.5),
        (3, 3, 0.75),
    ],
)
def test_grid_pattern_rejects_invalid_dimensions_or_margin(
    rows: int,
    cols: int,
    margin: float,
) -> None:
    with pytest.raises(ValueError):
        grid_pattern(rows=rows, cols=cols, margin=margin)
