"""Scalar diagnostics for calibration feature separability."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

from pupil_tracker.models import CalibrationSample


@dataclass(frozen=True)
class TargetFeatureSummary:
    """Feature-vector summary for one calibration target."""

    target_id: str
    target_x: float
    target_y: float
    accepted_count: int
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]


@dataclass(frozen=True)
class FeatureDiagnosticsSummary:
    """Scalar feature diagnostics grouped by calibration target."""

    feature_count: int
    target_summaries: Mapping[str, TargetFeatureSummary]


def summarize_feature_diagnostics(
    samples: Sequence[CalibrationSample],
) -> FeatureDiagnosticsSummary:
    """Summarize valid non-empty calibration feature vectors by target.

    Invalid observations and observations without feature vectors are skipped.
    All included feature vectors must have the same length so feature indices can
    be compared across targets in later telemetry analysis.
    """

    grouped: OrderedDict[str, list[tuple[float, ...]]] = OrderedDict()
    targets_by_id: dict[str, tuple[float, float]] = {}
    feature_count = 0

    for sample in samples:
        observation = sample.observation
        features = observation.feature_vector
        if not observation.valid or not features:
            continue
        if feature_count == 0:
            feature_count = len(features)
        elif len(features) != feature_count:
            msg = f"feature vector length {len(features)} != expected {feature_count}"
            raise ValueError(msg)

        target = sample.target
        existing_target = targets_by_id.get(target.id)
        target_position = (target.x, target.y)
        if existing_target is not None and existing_target != target_position:
            msg = f"target {target.id!r} has inconsistent coordinates"
            raise ValueError(msg)
        targets_by_id[target.id] = target_position
        grouped.setdefault(target.id, []).append(tuple(float(value) for value in features))

    if feature_count == 0:
        return FeatureDiagnosticsSummary(feature_count=0, target_summaries={})

    target_summaries: dict[str, TargetFeatureSummary] = {}
    for target_id, feature_vectors in grouped.items():
        target_x, target_y = targets_by_id[target_id]
        means = _feature_means(feature_vectors, feature_count)
        target_summaries[target_id] = TargetFeatureSummary(
            target_id=target_id,
            target_x=target_x,
            target_y=target_y,
            accepted_count=len(feature_vectors),
            feature_mean=means,
            feature_std=_feature_std(feature_vectors, means),
        )

    return FeatureDiagnosticsSummary(
        feature_count=feature_count,
        target_summaries=target_summaries,
    )


def _feature_means(
    feature_vectors: Sequence[tuple[float, ...]],
    feature_count: int,
) -> tuple[float, ...]:
    count = len(feature_vectors)
    return tuple(
        sum(vector[index] for vector in feature_vectors) / count
        for index in range(feature_count)
    )


def _feature_std(
    feature_vectors: Sequence[tuple[float, ...]],
    means: tuple[float, ...],
) -> tuple[float, ...]:
    count = len(feature_vectors)
    return tuple(
        sqrt(sum((vector[index] - mean) ** 2 for vector in feature_vectors) / count)
        for index, mean in enumerate(means)
    )
