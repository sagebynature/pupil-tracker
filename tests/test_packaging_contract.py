from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_includes_desktop_calibration_demo_package() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert "src/pupil_tracker" in wheel["packages"]
    assert "apps/desktop_demo" in wheel["packages"]
