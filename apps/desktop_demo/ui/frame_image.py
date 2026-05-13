"""Helpers for displaying OpenCV camera frames in Qt widgets."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage


def bgr_ndarray_to_qimage(image: np.ndarray) -> QImage:
    """Convert a 3-channel uint8 OpenCV BGR image to an owning RGB QImage."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        msg = "expected a 3-channel uint8 BGR image"
        raise ValueError(msg)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_image.shape
    bytes_per_line = width * channels
    qimage = QImage(
        rgb_image.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    return qimage.copy()
