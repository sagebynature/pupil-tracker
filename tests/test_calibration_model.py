"""Tests for polynomial/ridge gaze calibration."""

import pytest

from pupil_tracker.calibration.model import PolynomialRidgeCalibrationModel
from pupil_tracker.models import CalibrationSample, CalibrationTarget, RawObservation


def _sample(target_id: str, x: float, y: float) -> CalibrationSample:
    return CalibrationSample(
        target=CalibrationTarget(id=target_id, x=x, y=y),
        observation=RawObservation(
            timestamp=x + y,
            valid=True,
            confidence=0.95,
            feature_vector=(x, y),
        ),
    )


def _grid_samples() -> list[CalibrationSample]:
    return [
        _sample("r0c0", 0.0, 0.0),
        _sample("r0c1", 0.5, 0.0),
        _sample("r0c2", 1.0, 0.0),
        _sample("r1c0", 0.0, 0.5),
        _sample("r1c1", 0.5, 0.5),
        _sample("r1c2", 1.0, 0.5),
        _sample("r2c0", 0.0, 1.0),
        _sample("r2c1", 0.5, 1.0),
        _sample("r2c2", 1.0, 1.0),
    ]


def test_fit_returns_sample_count_and_pixel_error_metrics() -> None:
    model = PolynomialRidgeCalibrationModel(degree=1, alpha=0.0)

    result = model.fit(_grid_samples(), screen_width=1000, screen_height=500)

    assert result.sample_count == 9
    assert result.mean_error_px == pytest.approx(0.0, abs=1e-6)
    assert result.max_error_px == pytest.approx(0.0, abs=1e-6)


def test_predict_maps_synthetic_observation_to_screen_coordinates() -> None:
    model = PolynomialRidgeCalibrationModel(degree=1, alpha=0.0)
    model.fit(_grid_samples(), screen_width=1000, screen_height=500)

    prediction = model.predict(
        RawObservation(
            timestamp=10.0,
            valid=True,
            confidence=0.8,
            feature_vector=(0.25, 0.75),
        ),
        screen_width=1000,
        screen_height=500,
    )

    assert prediction.valid
    assert prediction.timestamp == 10.0
    assert prediction.confidence == 0.8
    assert prediction.x == pytest.approx(250.0, abs=1e-6)
    assert prediction.y == pytest.approx(375.0, abs=1e-6)


def test_predict_before_fit_raises_clear_error() -> None:
    model = PolynomialRidgeCalibrationModel()

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(
            RawObservation(
                timestamp=1.0,
                valid=True,
                confidence=0.8,
                feature_vector=(0.5, 0.5),
            ),
            screen_width=1000,
            screen_height=500,
        )


def test_fit_with_too_few_valid_samples_raises_value_error() -> None:
    model = PolynomialRidgeCalibrationModel(degree=1, alpha=0.0)
    samples = [
        _sample("r0c0", 0.0, 0.0),
        CalibrationSample(
            target=CalibrationTarget(id="invalid", x=0.5, y=0.5),
            observation=RawObservation.invalid(timestamp=2.0, reason="no face"),
        ),
    ]

    with pytest.raises(ValueError, match="at least 3 valid samples"):
        model.fit(samples, screen_width=1000, screen_height=500)
