"""Tests for release version policy."""

from __future__ import annotations

import sys
import tomllib
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


def test_project_metadata_marks_core_library_cross_platform() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert "Operating System :: OS Independent" in project["classifiers"]
    assert "Operating System :: MacOS :: MacOS X" in project["classifiers"]
    assert "pyobjc-framework-quartz>=10.0; sys_platform == 'darwin'" in project["dependencies"]


def test_ci_workflow_runs_checks_on_ubuntu_without_macos_gate() -> None:
    workflows_dir = PROJECT_ROOT / ".github" / "workflows"
    workflow = (workflows_dir / "ci.yml").read_text(encoding="utf-8")

    assert not (workflows_dir / "publish.yml").exists()
    assert "name: CI" in workflow
    assert "pull_request:" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "run: make check" in workflow
    assert "macos-latest" not in workflow


def test_release_workflow_semantic_releases_and_publishes_from_ubuntu() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "name: Release & Publish" in workflow
    assert "branches: [main]" in workflow
    assert "python-semantic-release/python-semantic-release@" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "uses: pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "macos-latest" not in workflow


def test_semantic_release_updates_project_version() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    semantic_release = pyproject["tool"]["semantic_release"]
    assert semantic_release["version_toml"] == ["pyproject.toml:project.version"]
    assert semantic_release["upload_to_pypi"] is False


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
