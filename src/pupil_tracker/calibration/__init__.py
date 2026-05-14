"""Calibration utilities for target generation, samples, and gaze mapping."""

from pupil_tracker.calibration.model import CalibrationFitResult, PolynomialRidgeCalibrationModel
from pupil_tracker.calibration.patterns import grid_pattern
from pupil_tracker.calibration.quality import (
    CalibrationQualityFilter,
    CalibrationSampleDecision,
    TargetQualitySummary,
    summarize_target_quality,
)
from pupil_tracker.calibration.samples import CalibrationSampleCollector
from pupil_tracker.calibration.timing import (
    CalibrationPhase,
    TimedCalibrationConfig,
    TimedTargetState,
    TimedTargetTimer,
)

__all__ = [
    "CalibrationFitResult",
    "CalibrationPhase",
    "CalibrationQualityFilter",
    "CalibrationSampleCollector",
    "CalibrationSampleDecision",
    "PolynomialRidgeCalibrationModel",
    "TargetQualitySummary",
    "TimedCalibrationConfig",
    "TimedTargetState",
    "TimedTargetTimer",
    "grid_pattern",
    "summarize_target_quality",
]
