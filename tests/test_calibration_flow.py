"""Tests for the desktop calibration flow state."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pupil_tracker.models import CalibrationTarget, RawObservation

if TYPE_CHECKING:
    from desktop_demo.ui.calibration_view import CalibrationFlowState

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))


def _flow(samples_per_target: int) -> CalibrationFlowState:
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    return CalibrationFlowState(samples_per_target=samples_per_target)


def _valid_observation(timestamp: float = 1.0) -> RawObservation:
    return RawObservation(
        timestamp=timestamp,
        valid=True,
        confidence=0.9,
        feature_vector=(0.1, 0.2, 0.3, 0.4, 0.2, 0.3),
    )


def test_initial_target_is_first_nine_point_target() -> None:
    flow = _flow(samples_per_target=2)
    target = flow.current_target

    assert target is not None
    assert target.id == "r0c0"
    assert target.x == 0.1
    assert target.y == 0.1
    assert flow.current_index == 0
    assert not flow.is_complete


def test_flow_accepts_custom_calibration_targets() -> None:
    from desktop_demo.ui.calibration_view import CalibrationFlowState

    targets = [
        CalibrationTarget(id="top", x=0.5, y=0.1),
        CalibrationTarget(id="bottom", x=0.5, y=0.9),
    ]
    flow = CalibrationFlowState(samples_per_target=1, targets=targets)

    assert flow.targets == tuple(targets)
    assert flow.current_target == targets[0]
    assert flow.capture_observation(_valid_observation())
    assert flow.current_target == targets[1]


def test_advancing_after_required_valid_samples_moves_to_next_target() -> None:
    flow = _flow(samples_per_target=2)
    first_target = flow.current_target

    assert not flow.capture_observation(_valid_observation(timestamp=1.0))
    assert flow.current_target == first_target

    assert flow.capture_observation(_valid_observation(timestamp=2.0))
    target = flow.current_target
    assert target is not None
    assert target.id == "r0c1"
    assert flow.current_index == 1


def test_completion_exposes_all_collected_samples() -> None:
    flow = _flow(samples_per_target=1)

    for index in range(9):
        assert flow.capture_observation(_valid_observation(timestamp=float(index)))

    assert flow.is_complete
    assert flow.current_target is None
    samples = flow.all_samples()
    assert len(samples) == 9
    assert [sample.target.id for sample in samples] == [
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


def test_insufficient_valid_samples_keeps_current_target() -> None:
    flow = _flow(samples_per_target=2)
    target = flow.current_target

    assert not flow.capture_observation(RawObservation.invalid(timestamp=1.0, reason="blink"))
    assert flow.current_target == target
    assert flow.samples_for_current_target() == ()

    assert not flow.capture_observation(_valid_observation(timestamp=2.0))
    assert flow.current_target == target
    assert len(flow.samples_for_current_target()) == 1


def test_add_current_target_sample_stores_without_advancing() -> None:
    flow = _flow(samples_per_target=1)
    target = flow.current_target

    assert flow.add_current_target_sample(_valid_observation())

    assert flow.current_target == target
    assert len(flow.samples_for_current_target()) == 1
    assert not flow.is_complete


def test_advance_target_moves_to_next_target() -> None:
    flow = _flow(samples_per_target=2)

    assert flow.advance_target()

    target = flow.current_target
    assert target is not None
    assert target.id == "r0c1"
    assert flow.current_index == 1


def test_clear_current_target_samples_preserves_previous_targets() -> None:
    flow = _flow(samples_per_target=5)
    first_observation = _valid_observation(timestamp=1.0)
    second_observation = _valid_observation(timestamp=2.0)

    assert flow.add_current_target_sample(first_observation)
    assert flow.advance_target()
    assert flow.add_current_target_sample(second_observation)

    flow.clear_current_target_samples()

    current_target = flow.current_target
    assert current_target is not None
    assert flow.samples_for_current_target() == ()
    all_samples = flow.all_samples()
    assert len(all_samples) == 1
    assert all_samples[0].target.id == "r0c0"
    assert all_samples[0].observation == first_observation
