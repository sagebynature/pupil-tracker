import pytest

from pupil_tracker.models import GazeSample
from pupil_tracker.smoothing.filters import EmaGazeSmoother


def _sample(
    x: float,
    y: float,
    confidence: float = 1.0,
    valid: bool = True,
    timestamp: float = 1.0,
    region_id: str | None = None,
) -> GazeSample:
    return GazeSample(
        timestamp=timestamp,
        x=x,
        y=y,
        confidence=confidence,
        valid=valid,
        region_id=region_id,
    )


def test_first_valid_sample_initializes_smoother() -> None:
    smoother = EmaGazeSmoother(alpha=0.5)

    result = smoother.update(_sample(100, 200, confidence=0.8, region_id="middle_center"))

    assert result.x == 100
    assert result.y == 200
    assert result.confidence == 0.8
    assert result.region_id == "middle_center"


def test_second_valid_sample_is_blended_by_alpha() -> None:
    smoother = EmaGazeSmoother(alpha=0.25)

    smoother.update(_sample(100, 200, confidence=0.8, timestamp=1.0))
    result = smoother.update(_sample(200, 100, confidence=0.4, timestamp=2.0))

    assert result.x == pytest.approx(125)
    assert result.y == pytest.approx(175)
    assert result.confidence == pytest.approx(0.7)
    assert result.timestamp == 2.0


def test_invalid_sample_preserves_invalid_status_without_resetting_last_valid_point() -> None:
    smoother = EmaGazeSmoother(alpha=0.5)

    smoother.update(_sample(10, 20, confidence=1.0, timestamp=1.0))
    invalid = smoother.update(_sample(999, 999, confidence=0.0, valid=False, timestamp=2.0))
    valid_after_invalid = smoother.update(_sample(20, 40, confidence=1.0, timestamp=3.0))

    assert not invalid.valid
    assert invalid.x == 10
    assert invalid.y == 20
    assert invalid.confidence == 0.0
    assert valid_after_invalid.x == pytest.approx(15)
    assert valid_after_invalid.y == pytest.approx(30)


def test_reset_clears_smoother_state() -> None:
    smoother = EmaGazeSmoother(alpha=0.5)

    smoother.update(_sample(100, 200))
    smoother.reset()
    result = smoother.update(_sample(300, 400))

    assert result.x == 300
    assert result.y == 400


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1])
def test_smoother_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        EmaGazeSmoother(alpha=alpha)
