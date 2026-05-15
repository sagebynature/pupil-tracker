import pytest

from pupil_tracker.calibration.patterns import (
    edge_dense_calibration_pattern,
    grid_pattern,
    top_left_focus_calibration_pattern,
    top_row_focus_calibration_pattern,
    vertical_grid_pattern,
)


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


def test_vertical_grid_pattern_returns_inset_three_by_five_targets() -> None:
    targets = vertical_grid_pattern(margin=0.1)

    assert len(targets) == 15
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
        "r3c0",
        "r3c1",
        "r3c2",
        "r4c0",
        "r4c1",
        "r4c2",
    ]
    assert [(target.x, target.y) for target in targets[0:3]] == [
        pytest.approx((0.1, 0.1)),
        pytest.approx((0.5, 0.1)),
        pytest.approx((0.9, 0.1)),
    ]
    assert [(target.x, target.y) for target in targets[-3:]] == [
        pytest.approx((0.1, 0.9)),
        pytest.approx((0.5, 0.9)),
        pytest.approx((0.9, 0.9)),
    ]
    assert targets[6].y == pytest.approx(0.5)


def test_edge_dense_calibration_pattern_adds_top_and_edge_coverage() -> None:
    targets = edge_dense_calibration_pattern(margin=0.1)

    assert len(targets) == 17
    assert [target.id for target in targets] == [
        "top0",
        "top1",
        "top2",
        "top3",
        "top4",
        "upper_left",
        "upper_right",
        "mid_left",
        "mid_center",
        "mid_right",
        "lower_left",
        "lower_right",
        "bottom0",
        "bottom1",
        "bottom2",
        "bottom3",
        "bottom4",
    ]
    assert [(target.x, target.y) for target in targets[0:5]] == [
        pytest.approx((0.1, 0.1)),
        pytest.approx((0.3, 0.1)),
        pytest.approx((0.5, 0.1)),
        pytest.approx((0.7, 0.1)),
        pytest.approx((0.9, 0.1)),
    ]
    assert [(targets[5].x, targets[5].y), (targets[6].x, targets[6].y)] == [
        pytest.approx((0.25, 0.25)),
        pytest.approx((0.75, 0.25)),
    ]
    assert [(targets[10].x, targets[10].y), (targets[11].x, targets[11].y)] == [
        pytest.approx((0.25, 0.75)),
        pytest.approx((0.75, 0.75)),
    ]
    assert [(target.x, target.y) for target in targets[-5:]] == [
        pytest.approx((0.1, 0.9)),
        pytest.approx((0.3, 0.9)),
        pytest.approx((0.5, 0.9)),
        pytest.approx((0.7, 0.9)),
        pytest.approx((0.9, 0.9)),
    ]


def test_top_left_focus_calibration_pattern_adds_local_v0_geometry() -> None:
    targets = top_left_focus_calibration_pattern(margin=0.1)

    assert len(targets) == 25
    assert [target.id for target in targets] == [
        "top0",
        "top1",
        "top2",
        "top3",
        "top4",
        "tl_upper_left",
        "tl_upper_mid",
        "tl_upper_right",
        "tl_center_left",
        "tl_center",
        "tl_center_right",
        "tl_lower_left",
        "tl_lower_mid",
        "tl_lower_right",
        "upper_right",
        "mid_left",
        "mid_center",
        "mid_right",
        "lower_left",
        "lower_right",
        "bottom0",
        "bottom1",
        "bottom2",
        "bottom3",
        "bottom4",
    ]
    assert [(target.x, target.y) for target in targets[0:5]] == [
        pytest.approx((0.1, 0.1)),
        pytest.approx((0.3, 0.1)),
        pytest.approx((0.5, 0.1)),
        pytest.approx((0.7, 0.1)),
        pytest.approx((0.9, 0.1)),
    ]
    assert [(target.x, target.y) for target in targets[5:14]] == [
        pytest.approx((0.18, 0.18)),
        pytest.approx((0.25, 0.18)),
        pytest.approx((0.32, 0.18)),
        pytest.approx((0.18, 0.25)),
        pytest.approx((0.25, 0.25)),
        pytest.approx((0.32, 0.25)),
        pytest.approx((0.18, 0.32)),
        pytest.approx((0.25, 0.32)),
        pytest.approx((0.32, 0.32)),
    ]
    assert targets[14].id == "upper_right"
    assert (targets[14].x, targets[14].y) == pytest.approx((0.75, 0.25))
    assert [(target.x, target.y) for target in targets[-5:]] == [
        pytest.approx((0.1, 0.9)),
        pytest.approx((0.3, 0.9)),
        pytest.approx((0.5, 0.9)),
        pytest.approx((0.7, 0.9)),
        pytest.approx((0.9, 0.9)),
    ]


def test_top_row_focus_calibration_pattern_adds_local_v0_and_v1_geometry() -> None:
    targets = top_row_focus_calibration_pattern(margin=0.1)

    assert len(targets) == 33
    assert [target.id for target in targets] == [
        "top0",
        "top1",
        "top2",
        "top3",
        "top4",
        "tl_upper_left",
        "tl_upper_mid",
        "tl_upper_right",
        "tl_center_left",
        "tl_center",
        "tl_center_right",
        "tl_lower_left",
        "tl_lower_mid",
        "tl_lower_right",
        "tr_upper_left",
        "tr_upper_mid",
        "tr_upper_right",
        "tr_center_left",
        "tr_center",
        "tr_center_right",
        "tr_lower_left",
        "tr_lower_mid",
        "tr_lower_right",
        "mid_left",
        "mid_center",
        "mid_right",
        "lower_left",
        "lower_right",
        "bottom0",
        "bottom1",
        "bottom2",
        "bottom3",
        "bottom4",
    ]
    assert [(target.x, target.y) for target in targets[5:14]] == [
        pytest.approx((0.18, 0.18)),
        pytest.approx((0.25, 0.18)),
        pytest.approx((0.32, 0.18)),
        pytest.approx((0.18, 0.25)),
        pytest.approx((0.25, 0.25)),
        pytest.approx((0.32, 0.25)),
        pytest.approx((0.18, 0.32)),
        pytest.approx((0.25, 0.32)),
        pytest.approx((0.32, 0.32)),
    ]
    assert [(target.x, target.y) for target in targets[14:23]] == [
        pytest.approx((0.68, 0.18)),
        pytest.approx((0.75, 0.18)),
        pytest.approx((0.82, 0.18)),
        pytest.approx((0.68, 0.25)),
        pytest.approx((0.75, 0.25)),
        pytest.approx((0.82, 0.25)),
        pytest.approx((0.68, 0.32)),
        pytest.approx((0.75, 0.32)),
        pytest.approx((0.82, 0.32)),
    ]
    assert [target.id for target in targets[23:28]] == [
        "mid_left",
        "mid_center",
        "mid_right",
        "lower_left",
        "lower_right",
    ]
    assert [(target.x, target.y) for target in targets[-5:]] == [
        pytest.approx((0.1, 0.9)),
        pytest.approx((0.3, 0.9)),
        pytest.approx((0.5, 0.9)),
        pytest.approx((0.7, 0.9)),
        pytest.approx((0.9, 0.9)),
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
