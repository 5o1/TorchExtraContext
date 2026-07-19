#!/usr/bin/env python3
"""Validate the project version and, optionally, the release tag.

The package version is intentionally single-sourced from pyproject.toml. Release
workflows can additionally require tags in the form vX.Y.Z, matching the
pyproject.toml version exactly.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import tomllib as TOML_LIB
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as TOML_LIB
    except ModuleNotFoundError:  # pragma: no cover
        TOML_LIB = None


SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _read_project_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text(encoding="utf-8")
    if TOML_LIB is not None:
        pyproject = TOML_LIB.loads(text)
        return pyproject["project"]["version"]

    in_project_table = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project_table = line == "[project]"
            continue
        if in_project_table:
            match = re.fullmatch(r'version\s*=\s*"([^"]+)"(?:\s*#.*)?', line)
            if match:
                return match.group(1)

    raise RuntimeError("Could not read project.version from pyproject.toml.")


def main(argv: list[str] | None = None) -> int:
    """Run version validation from CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-tag",
        action="store_true",
        help="Require the current GitHub tag to be vX.Y.Z and match project.version.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    pyproject_path = root / "pyproject.toml"
    version = _read_project_version(pyproject_path)

    errors: list[str] = []
    if not SEMVER_RE.fullmatch(version):
        errors.append(
            f"pyproject.toml version must be plain SemVer X.Y.Z; got {version!r}."
        )

    ref_type = os.environ.get("GITHUB_REF_TYPE")
    ref_name = os.environ.get("GITHUB_REF_NAME")
    if args.release_tag and ref_type == "tag":
        expected_tag = f"v{version}"
        if ref_name != expected_tag:
            errors.append(f"release tag must be {expected_tag!r}; got {ref_name!r}.")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Validated version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
