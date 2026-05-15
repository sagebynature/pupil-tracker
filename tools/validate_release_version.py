"""Validate PyPI release version policy.

The project publishes stable SemVer releases to PyPI from git tags named
``vMAJOR.MINOR.PATCH``. PyPI package metadata must use the same
``MAJOR.MINOR.PATCH`` value.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from pathlib import Path

STABLE_SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)$"
)
TAG_PATTERN = re.compile(r"^v(?P<version>(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*))$")


class ReleaseVersionError(ValueError):
    """Raised when release metadata violates the versioning policy."""


def load_project_version(pyproject_path: Path) -> str:
    """Return the static project version from pyproject.toml."""
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not isinstance(version, str):
        msg = "pyproject.toml must define [project].version as a static string"
        raise ReleaseVersionError(msg)
    return version


def validate_stable_semver(version: str) -> None:
    """Validate the project version is stable SemVer core: MAJOR.MINOR.PATCH."""
    if STABLE_SEMVER_PATTERN.fullmatch(version) is None:
        msg = f"project version must be stable SemVer MAJOR.MINOR.PATCH; got {version!r}"
        raise ReleaseVersionError(msg)


def version_from_ref(ref_name: str) -> str:
    """Extract the PyPI version from a release tag name."""
    match = TAG_PATTERN.fullmatch(ref_name)
    if match is None:
        msg = f"release tags must use vMAJOR.MINOR.PATCH; got {ref_name!r}"
        raise ReleaseVersionError(msg)
    return match.group("version")


def validate_release_version(
    pyproject_path: Path,
    *,
    ref_name: str | None = None,
    require_tag_match: bool = False,
) -> str:
    """Validate project release metadata and optional git tag consistency."""
    version = load_project_version(pyproject_path)
    validate_stable_semver(version)

    if require_tag_match:
        if not ref_name:
            msg = "release validation requires a git tag name"
            raise ReleaseVersionError(msg)
        tag_version = version_from_ref(ref_name)
        if tag_version != version:
            msg = f"release tag {ref_name!r} does not match project version {version!r}"
            raise ReleaseVersionError(msg)

    return version


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml.",
    )
    parser.add_argument(
        "--ref-name",
        default=os.environ.get("GITHUB_REF_NAME"),
        help="Git ref name to compare against [default: GITHUB_REF_NAME].",
    )
    parser.add_argument(
        "--require-tag-match",
        action="store_true",
        help=(
            "Require --ref-name/GITHUB_REF_NAME to be vMAJOR.MINOR.PATCH and match project.version."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        version = validate_release_version(
            args.pyproject,
            ref_name=args.ref_name,
            require_tag_match=args.require_tag_match,
        )
    except ReleaseVersionError as error:
        print(f"release version check failed: {error}", file=sys.stderr)
        return 1

    print(f"release version check passed: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
