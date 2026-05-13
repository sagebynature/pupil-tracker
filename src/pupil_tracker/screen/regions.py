"""Map calibrated gaze points to named screen regions."""

_ROW_NAMES = ("top", "middle", "bottom")
_COL_NAMES = ("left", "center", "right")


def region_3x3(x: float, y: float, width: float, height: float) -> str:
    """Return the named 3x3 region containing a screen point.

    Out-of-bounds points are clamped to the nearest screen edge before mapping.
    """

    if width <= 0:
        raise ValueError("width must be positive")
    if height <= 0:
        raise ValueError("height must be positive")

    clamped_x = _clamp(value=x, lower=0.0, upper=width)
    clamped_y = _clamp(value=y, lower=0.0, upper=height)
    col = _bucket(value=clamped_x, size=width)
    row = _bucket(value=clamped_y, size=height)
    return f"{_ROW_NAMES[row]}_{_COL_NAMES[col]}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _bucket(value: float, size: float) -> int:
    bucket = int(value / (size / 3.0))
    return min(bucket, 2)
