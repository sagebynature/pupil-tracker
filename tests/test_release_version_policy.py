"""Tests for release version policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from validate_release_version import (  # noqa: E402
    ReleaseVersionError,
    validate_release_version,
    version_from_ref,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_pyproject(path: Path, version: str) -> Path:
    pyproject = path / "pyproject.toml"
    pyproject.write_text(
        f"""
[project]
name = "example"
version = "{version}"
""".lstrip(),
        encoding="utf-8",
    )
    return pyproject


def test_project_version_uses_stable_semver() -> None:
    version = validate_release_version(PROJECT_ROOT / "pyproject.toml")

    major, minor, patch = version.split(".")
    assert major.isdecimal()
    assert minor.isdecimal()
    assert patch.isdecimal()


def test_release_tag_must_match_project_version(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path, "1.2.3")

    assert validate_release_version(pyproject, ref_name="v1.2.3", require_tag_match=True) == "1.2.3"


@pytest.mark.parametrize("version", ["1.2", "01.2.3", "1.2.3.4", "1.2.3-dev"])
def test_rejects_non_stable_semver_project_versions(tmp_path: Path, version: str) -> None:
    pyproject = _write_pyproject(tmp_path, version)

    with pytest.raises(ReleaseVersionError):
        validate_release_version(pyproject)


def test_rejects_release_tag_that_does_not_match_project_version(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path, "1.2.3")

    with pytest.raises(ReleaseVersionError, match="does not match"):
        validate_release_version(pyproject, ref_name="v1.2.4", require_tag_match=True)


@pytest.mark.parametrize("ref_name", ["1.2.3", "v1.2", "release-1.2.3"])
def test_rejects_non_semver_release_tags(ref_name: str) -> None:
    with pytest.raises(ReleaseVersionError):
        version_from_ref(ref_name)
