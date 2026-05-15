"""Calibration utilities for target generation, samples, and gaze mapping."""

from pupil_tracker.calibration.feature_diagnostics import (
    FeatureDiagnosticsSummary,
    TargetFeatureSummary,
    summarize_feature_diagnostics,
)
from pupil_tracker.calibration.model import (
    CalibrationFitResult,
    LinearRidgeCalibrationModel,
    PolynomialRidgeCalibrationModel,
)
from pupil_tracker.calibration.patterns import (
    edge_dense_calibration_pattern,
    grid_pattern,
    top_left_focus_calibration_pattern,
    vertical_grid_pattern,
)
from pupil_tracker.calibration.quality import (
    CalibrationQualityFilter,
    CalibrationSampleDecision,
    FeatureStabilityConfig,
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
from pupil_tracker.calibration.validation import (
    ValidationMetrics,
    ValidationSample,
    ValidationTarget,
    compute_validation_metrics,
    validation_pattern,
)

__all__ = [
    "CalibrationFitResult",
    "CalibrationPhase",
    "CalibrationQualityFilter",
    "CalibrationSampleCollector",
    "CalibrationSampleDecision",
    "FeatureDiagnosticsSummary",
    "FeatureStabilityConfig",
    "LinearRidgeCalibrationModel",
    "PolynomialRidgeCalibrationModel",
    "TargetFeatureSummary",
    "TargetQualitySummary",
    "TimedCalibrationConfig",
    "TimedTargetState",
    "TimedTargetTimer",
    "ValidationMetrics",
    "ValidationSample",
    "ValidationTarget",
    "compute_validation_metrics",
    "edge_dense_calibration_pattern",
    "grid_pattern",
    "summarize_feature_diagnostics",
    "summarize_target_quality",
    "top_left_focus_calibration_pattern",
    "validation_pattern",
    "vertical_grid_pattern",
]
