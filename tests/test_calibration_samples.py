"""Tests for collecting calibration samples by target."""

from pupil_tracker.calibration import CalibrationSampleCollector
from pupil_tracker.models import CalibrationSample, CalibrationTarget, RawObservation


def _target(target_id: str = "r1c1") -> CalibrationTarget:
    return CalibrationTarget(id=target_id, x=0.5, y=0.5)


def _valid_sample(target_id: str = "r1c1", timestamp: float = 1.0) -> CalibrationSample:
    return CalibrationSample(
        target=_target(target_id),
        observation=RawObservation(
            timestamp=timestamp,
            valid=True,
            confidence=0.9,
            feature_vector=(0.1, 0.2, 0.3),
        ),
    )


def test_collector_starts_empty() -> None:
    collector = CalibrationSampleCollector()

    assert collector.all_samples() == ()
    assert collector.samples_for("r1c1") == ()


def test_add_valid_sample_increments_count_for_target() -> None:
    collector = CalibrationSampleCollector()
    sample = _valid_sample("r0c0")

    assert collector.add(sample)

    assert len(collector.samples_for("r0c0")) == 1
    assert collector.samples_for("r0c0") == (sample,)


def test_add_invalid_sample_is_skipped() -> None:
    collector = CalibrationSampleCollector()
    invalid_sample = CalibrationSample(
        target=_target("r0c0"),
        observation=RawObservation.invalid(timestamp=2.0, reason="no face"),
    )

    assert not collector.add(invalid_sample)

    assert collector.all_samples() == ()
    assert collector.samples_for("r0c0") == ()


def test_samples_for_returns_stable_samples_for_target() -> None:
    collector = CalibrationSampleCollector()
    first = _valid_sample("r0c0", timestamp=1.0)
    second = _valid_sample("r1c1", timestamp=2.0)
    third = _valid_sample("r0c0", timestamp=3.0)

    collector.add(first)
    collector.add(second)
    collector.add(third)

    assert collector.samples_for("r0c0") == (first, third)
    assert collector.samples_for("r1c1") == (second,)


def test_all_samples_returns_samples_in_insertion_order() -> None:
    collector = CalibrationSampleCollector()
    first = _valid_sample("r0c0", timestamp=1.0)
    second = _valid_sample("r1c1", timestamp=2.0)
    third = _valid_sample("r2c2", timestamp=3.0)

    collector.add(first)
    collector.add(second)
    collector.add(third)

    assert collector.all_samples() == (first, second, third)


def test_clear_removes_all_samples() -> None:
    collector = CalibrationSampleCollector()
    collector.add(_valid_sample("r0c0"))
    collector.add(_valid_sample("r1c1"))

    collector.clear()

    assert collector.all_samples() == ()
    assert collector.samples_for("r0c0") == ()
    assert collector.samples_for("r1c1") == ()
