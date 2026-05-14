"""Tests for MediaPipe model download helper documentation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_makefile_exposes_download_model_target() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "download-model" in makefile
    assert "MODEL_DIR ?= models" in makefile
    assert "FACE_LANDMARKER_MODEL_URL ?= https://storage.googleapis.com" in makefile
    assert "face_landmarker/face_landmarker/float16/latest/face_landmarker.task" in makefile
    assert "$(MODEL_DIR)/face_landmarker.task" in makefile
    assert "curl -L --fail" in makefile


def test_readme_documents_download_model_and_launch_flow() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "make download-model" in readme
    assert "export PUPIL_TRACKER_MEDIAPIPE_MODEL=$(pwd)/models/face_landmarker.task" in readme
    launch_command = (
        "PUPIL_TRACKER_MEDIAPIPE_MODEL=/absolute/path/to/face_landmarker.task "
        "make run-demo"
    )
    assert launch_command in readme
