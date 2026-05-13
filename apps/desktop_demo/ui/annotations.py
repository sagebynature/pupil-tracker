"""Preview annotation helpers for tracker observations."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from pupil_tracker.models import Point2D, RawObservation, Rect

_COLOR_FACE = (0, 255, 0)
_COLOR_IRIS = (0, 255, 255)
_COLOR_INVALID = (0, 0, 255)


def _point_tuple(point: Point2D) -> tuple[int, int]:
    return (round(point.x), round(point.y))


def _rect_points(rect: Rect) -> tuple[tuple[int, int], tuple[int, int]]:
    top_left = (round(rect.x), round(rect.y))
    bottom_right = (round(rect.x + rect.width), round(rect.y + rect.height))
    return top_left, bottom_right


def annotate_observation(
    image: NDArray[np.uint8],
    observation: RawObservation,
) -> NDArray[np.uint8]:
    """Return a copy of `image` annotated with tracker observation details."""

    annotated = np.array(image, copy=True)
    if observation.valid:
        if observation.face_bounds is not None:
            top_left, bottom_right = _rect_points(observation.face_bounds)
            cv2.rectangle(annotated, top_left, bottom_right, _COLOR_FACE, thickness=2)
        for iris in (observation.left_iris, observation.right_iris):
            if iris is not None:
                cv2.circle(annotated, _point_tuple(iris), radius=4, color=_COLOR_IRIS, thickness=-1)
        return annotated

    message = observation.reason or "invalid"
    cv2.putText(
        annotated,
        message,
        org=(8, 24),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=_COLOR_INVALID,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return annotated
