"""Tests for converting OpenCV frames into Qt images."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from desktop_demo.ui.frame_image import bgr_ndarray_to_qimage  # noqa: E402


def test_bgr_frame_converts_to_rgb_qimage() -> None:
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)  # BGR

    qimage = bgr_ndarray_to_qimage(image)

    assert qimage.width() == 1
    assert qimage.height() == 1
    assert qimage.pixelColor(0, 0).red() == 30
    assert qimage.pixelColor(0, 0).green() == 20
    assert qimage.pixelColor(0, 0).blue() == 10


def test_bgr_frame_conversion_copies_image_data() -> None:
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)

    qimage = bgr_ndarray_to_qimage(image)
    image[0, 0] = [200, 210, 220]

    assert qimage.pixelColor(0, 0).red() == 30
    assert qimage.pixelColor(0, 0).green() == 20
    assert qimage.pixelColor(0, 0).blue() == 10


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
    ],
)
def test_bgr_frame_conversion_rejects_unsupported_shapes_or_types(
    image: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="3-channel uint8 BGR"):
        bgr_ndarray_to_qimage(image)
