"""Calibration flow state and view widgets."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from pupil_tracker.calibration import CalibrationSampleCollector, grid_pattern
from pupil_tracker.models import CalibrationSample, CalibrationTarget, RawObservation


class CalibrationFlowState:
    """Pure state machine for the 9-point calibration flow."""

    def __init__(self, samples_per_target: int = 5) -> None:
        if samples_per_target <= 0:
            msg = "samples_per_target must be positive"
            raise ValueError(msg)
        self.samples_per_target = samples_per_target
        self.targets = grid_pattern(3, 3)
        self.current_index = 0
        self._collector = CalibrationSampleCollector()

    @property
    def is_complete(self) -> bool:
        """Return whether all calibration targets have enough samples."""

        return self.current_index >= len(self.targets)

    @property
    def current_target(self) -> CalibrationTarget | None:
        """Return the active target or None after completion."""

        if self.is_complete:
            return None
        return self.targets[self.current_index]

    def capture_observation(self, observation: RawObservation) -> bool:
        """Capture an observation for the current target.

        Returns True when the flow advances to another target or completes.
        Invalid observations are skipped.
        """

        target = self.current_target
        if target is None:
            return False

        stored = self._collector.add(CalibrationSample(target=target, observation=observation))
        if not stored:
            return False

        if len(self._collector.samples_for(target.id)) < self.samples_per_target:
            return False

        self.current_index += 1
        return True

    def samples_for_current_target(self) -> tuple[CalibrationSample, ...]:
        """Return valid samples collected for the current target."""

        target = self.current_target
        if target is None:
            return ()
        return self._collector.samples_for(target.id)

    def all_samples(self) -> tuple[CalibrationSample, ...]:
        """Return every valid collected calibration sample in insertion order."""

        return self._collector.all_samples()


class CalibrationView(QWidget):
    """Simple calibration UI placeholder backed by CalibrationFlowState."""

    def __init__(self, flow: CalibrationFlowState | None = None) -> None:
        super().__init__()
        self.flow = flow if flow is not None else CalibrationFlowState()
        self.title_label = QLabel("9-point calibration")
        self.target_label = QLabel()
        self.status_label = QLabel()
        self.start_button = QPushButton("Start Calibration")

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.target_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.start_button)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        """Refresh labels from the current flow state."""

        target = self.flow.current_target
        if target is None:
            self.target_label.setText("Calibration complete")
            self.status_label.setText(f"Collected samples: {len(self.flow.all_samples())}")
            return

        self.target_label.setText(
            f"Target {self.flow.current_index + 1}/9: {target.id} ({target.x:.2f}, {target.y:.2f})"
        )
        sample_count = len(self.flow.samples_for_current_target())
        self.status_label.setText(
            f"Samples: {sample_count}/{self.flow.samples_per_target}"
        )
