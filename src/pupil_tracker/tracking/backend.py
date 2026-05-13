"""Pluggable tracker backend contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pupil_tracker.models import FrameMetadata, RawObservation


@dataclass(frozen=True)
class Frame:
    """Camera frame image payload plus metadata."""

    image: NDArray[np.uint8]
    metadata: FrameMetadata


class TrackerBackend(Protocol):
    """Protocol for tracker backends that turn frames into raw observations."""

    name: str

    def process(self, frame: Frame) -> RawObservation:
        """Process one frame and return a raw tracker observation."""

    def close(self) -> None:
        """Release backend resources."""
