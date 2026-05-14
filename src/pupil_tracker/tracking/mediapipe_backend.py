"""MediaPipe FaceMesh tracker backend."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import cv2
import mediapipe as mp

from pupil_tracker.logging_config import get_logger
from pupil_tracker.models import Point2D, RawObservation, Rect
from pupil_tracker.tracking.backend import Frame
from pupil_tracker.tracking.features import FeatureExtractionError, eye_geometry_feature_vector

_LOGGER = get_logger("tracking.mediapipe")


class MediaPipeTrackerBackend:
    """Tracker backend that extracts iris gaze features with MediaPipe FaceMesh."""

    LEFT_IRIS_INDICES: ClassVar[tuple[int, ...]] = (468, 469, 470, 471, 472)
    RIGHT_IRIS_INDICES: ClassVar[tuple[int, ...]] = (473, 474, 475, 476, 477)
    LEFT_EYE_INDICES: ClassVar[tuple[int, ...]] = (33, 133, 159, 145)
    RIGHT_EYE_INDICES: ClassVar[tuple[int, ...]] = (362, 263, 386, 374)

    name = "mediapipe"

    def __init__(self, face_mesh: Any | None = None, model_asset_path: str | None = None) -> None:
        self._face_mesh = (
            face_mesh if face_mesh is not None else self._create_face_mesh(model_asset_path)
        )

    def process(self, frame: Frame) -> RawObservation:
        """Process a frame and return a raw observation for calibration."""

        rgb_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb_image)
        faces = getattr(result, "multi_face_landmarks", None)
        if not faces:
            return RawObservation.invalid(
                timestamp=frame.metadata.timestamp,
                reason="no face detected",
            )

        landmarks = faces[0].landmark
        face_bounds = self._face_bounds(landmarks, frame.metadata.width, frame.metadata.height)
        left_iris = self._iris_center(
            landmarks,
            self.LEFT_IRIS_INDICES,
            frame.metadata.width,
            frame.metadata.height,
        )
        right_iris = self._iris_center(
            landmarks,
            self.RIGHT_IRIS_INDICES,
            frame.metadata.width,
            frame.metadata.height,
        )
        left_eye_bounds = self._landmark_bounds(
            landmarks,
            self.LEFT_EYE_INDICES,
            frame.metadata.width,
            frame.metadata.height,
        )
        right_eye_bounds = self._landmark_bounds(
            landmarks,
            self.RIGHT_EYE_INDICES,
            frame.metadata.width,
            frame.metadata.height,
        )

        try:
            features = eye_geometry_feature_vector(
                face_bounds=face_bounds,
                left_iris=left_iris,
                right_iris=right_iris,
                left_eye_bounds=left_eye_bounds,
                right_eye_bounds=right_eye_bounds,
            )
        except FeatureExtractionError as error:
            _LOGGER.debug("failed to extract MediaPipe features: %s", error)
            return RawObservation.invalid(
                timestamp=frame.metadata.timestamp,
                reason=str(error),
            )

        return RawObservation(
            timestamp=frame.metadata.timestamp,
            valid=True,
            confidence=1.0,
            face_bounds=face_bounds,
            left_iris=left_iris,
            right_iris=right_iris,
            feature_vector=features,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""

        self._face_mesh.close()

    @staticmethod
    def _create_face_mesh(model_asset_path: str | None) -> Any:
        if model_asset_path is None:
            msg = "MediaPipe FaceLandmarker model_asset_path is required"
            raise RuntimeError(msg)

        mp_any = cast(Any, mp)
        options = mp_any.tasks.vision.FaceLandmarkerOptions(
            base_options=mp_any.tasks.BaseOptions(model_asset_path=model_asset_path),
            running_mode=mp_any.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp_any.tasks.vision.FaceLandmarker.create_from_options(options)
        return _FaceLandmarkerAdapter(landmarker)

    @staticmethod
    def _face_bounds(landmarks: Any, frame_width: int, frame_height: int) -> Rect:
        xs = [landmark.x * frame_width for landmark in landmarks]
        ys = [landmark.y * frame_height for landmark in landmarks]
        min_x = min(xs)
        min_y = min(ys)
        return Rect(
            x=min_x,
            y=min_y,
            width=max(xs) - min_x,
            height=max(ys) - min_y,
        )

    @staticmethod
    def _iris_center(
        landmarks: Any,
        indices: tuple[int, ...],
        frame_width: int,
        frame_height: int,
    ) -> Point2D:
        points = [landmarks[index] for index in indices]
        return Point2D(
            x=sum(point.x for point in points) * frame_width / len(points),
            y=sum(point.y for point in points) * frame_height / len(points),
        )

    @staticmethod
    def _landmark_bounds(
        landmarks: Any,
        indices: tuple[int, ...],
        frame_width: int,
        frame_height: int,
    ) -> Rect:
        points = [landmarks[index] for index in indices]
        xs = [point.x * frame_width for point in points]
        ys = [point.y * frame_height for point in points]
        min_x = min(xs)
        min_y = min(ys)
        return Rect(
            x=min_x,
            y=min_y,
            width=max(xs) - min_x,
            height=max(ys) - min_y,
        )


class _FaceLandmarkerAdapter:
    """Adapt MediaPipe Tasks FaceLandmarker to the FaceMesh-like test seam."""

    def __init__(self, landmarker: Any) -> None:
        self._landmarker = landmarker

    def process(self, rgb_image: Any) -> Any:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._landmarker.detect(mp_image)
        return _FaceLandmarkerResultAdapter(result)

    def close(self) -> None:
        self._landmarker.close()


class _FaceLandmarkerResultAdapter:
    """Expose Tasks face landmarks under the FaceMesh-style attribute name."""

    def __init__(self, result: Any) -> None:
        self.multi_face_landmarks = [
            _FaceLandmarksAdapter(landmarks) for landmarks in result.face_landmarks
        ]


class _FaceLandmarksAdapter:
    """Expose a Tasks landmark list under the FaceMesh-style `.landmark` name."""

    def __init__(self, landmarks: Any) -> None:
        self.landmark = landmarks
