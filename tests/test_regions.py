import pytest

from pupil_tracker.screen.regions import region_3x3


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0, 0, "top_left"),
        (50, 0, "top_left"),
        (150, 50, "top_center"),
        (299, 50, "top_right"),
        (50, 150, "middle_left"),
        (150, 150, "middle_center"),
        (299, 150, "middle_right"),
        (50, 299, "bottom_left"),
        (150, 299, "bottom_center"),
        (299, 299, "bottom_right"),
    ],
)
def test_region_3x3_maps_points_to_named_regions(x: float, y: float, expected: str) -> None:
    assert region_3x3(x=x, y=y, width=300, height=300) == expected


def test_region_3x3_clamps_out_of_bounds_coordinates() -> None:
    assert region_3x3(x=-50, y=-1, width=300, height=300) == "top_left"
    assert region_3x3(x=350, y=999, width=300, height=300) == "bottom_right"


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0, 100),
        (100, 0),
        (-1, 100),
        (100, -1),
    ],
)
def test_region_3x3_rejects_invalid_screen_dimensions(width: float, height: float) -> None:
    with pytest.raises(ValueError):
        region_3x3(x=0, y=0, width=width, height=height)
