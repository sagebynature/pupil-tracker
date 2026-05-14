"""Calibration flow state and view widgets."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
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

        stored = self.add_current_target_sample(observation)
        if not stored:
            return False

        if len(self._collector.samples_for(target.id)) < self.samples_per_target:
            return False

        return self.advance_target()

    def add_current_target_sample(self, observation: RawObservation) -> bool:
        """Store a valid observation for the active target without advancing."""

        target = self.current_target
        if target is None:
            return False
        return self._collector.add(CalibrationSample(target=target, observation=observation))

    def advance_target(self) -> bool:
        """Advance to the next target and return whether the flow advanced."""

        if self.current_target is None:
            return False
        self.current_index += 1
        return True

    def clear_current_target_samples(self) -> None:
        """Clear samples for the active target while preserving previous targets."""

        target = self.current_target
        if target is None:
            return
        retained_samples = [
            sample for sample in self._collector.all_samples() if sample.target.id != target.id
        ]
        self._collector = CalibrationSampleCollector()
        for sample in retained_samples:
            self._collector.add(sample)

    def samples_for_current_target(self) -> tuple[CalibrationSample, ...]:
        """Return valid samples collected for the current target."""

        target = self.current_target
        if target is None:
            return ()
        return self._collector.samples_for(target.id)

    def all_samples(self) -> tuple[CalibrationSample, ...]:
        """Return every valid collected calibration sample in insertion order."""

        return self._collector.all_samples()

    def reset(self) -> None:
        """Reset collected samples and return to the first target."""

        self.current_index = 0
        self._collector.clear()


class CalibrationTargetWidget(QWidget):
    """Widget that draws the active normalized calibration target."""

    def __init__(self, flow: CalibrationFlowState) -> None:
        super().__init__()
        self.flow = flow
        self.setMinimumSize(320, 220)
        self.setStyleSheet("background: #050505;")

    def current_target_position(self) -> tuple[float, float] | None:
        """Return the active target as normalized widget coordinates."""

        target = self.flow.current_target
        if target is None:
            return None
        return (target.x, target.y)

    def current_target_pixel(self) -> tuple[int, int] | None:
        """Return the active target center in widget pixel coordinates."""

        position = self.current_target_position()
        if position is None:
            return None
        x, y = position
        return (round(self.width() * x), round(self.height() * y))

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the calibration target."""

        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#050505"))
        point = self.current_target_pixel()
        if point is None:
            painter.setPen(QPen(QColor("#cfcfcf")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Calibration complete")
            return

        x, y = point
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(x - 18, y, x + 18, y)
        painter.drawLine(x, y - 18, x, y + 18)
        painter.setBrush(QColor("#00d1ff"))
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawEllipse(x - 8, y - 8, 16, 16)


class CalibrationView(QWidget):
    """Simple calibration UI placeholder backed by CalibrationFlowState."""

    def __init__(self, flow: CalibrationFlowState | None = None) -> None:
        super().__init__()
        self.flow = flow if flow is not None else CalibrationFlowState()
        self.title_label = QLabel("9-point calibration")
        self.target_label = QLabel()
        self.status_label = QLabel()
        self.target_widget = CalibrationTargetWidget(self.flow)
        self.start_button = QPushButton("Start Calibration")

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.target_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.target_widget)
        layout.addWidget(self.start_button)
        self.setLayout(layout)
        self.refresh()

    def current_target_position(self) -> tuple[float, float] | None:
        """Return the active target as normalized widget coordinates."""

        return self.target_widget.current_target_position()

    def refresh(self) -> None:
        """Refresh labels from the current flow state."""

        target = self.flow.current_target
        if target is None:
            self.target_label.setText("Calibration complete")
            self.status_label.setText(f"Collected samples: {len(self.flow.all_samples())}")
            self.target_widget.update()
            return

        self.target_label.setText(
            f"Target {self.flow.current_index + 1}/9: {target.id} ({target.x:.2f}, {target.y:.2f})"
        )
        sample_count = len(self.flow.samples_for_current_target())
        self.status_label.setText(
            f"Samples: {sample_count}/{self.flow.samples_per_target}"
        )
        self.target_widget.update()
