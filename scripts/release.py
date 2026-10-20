# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Prepare and check the releases of the project.

The changelog follows Keep a Changelog: the changes are written by hand under
``## [Unreleased]``, and this script turns that section into a release.

Usage::

    python scripts/release.py prepare 0.2.0  # before the release, then commit
    python scripts/release.py check v0.2.0  # run by the release workflow
    python scripts/release.py notes v0.2.0  # notes of the GitHub release
"""

import argparse
import datetime
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
CITATION = ROOT / "CITATION.cff"

UNRELEASED = "## [Unreleased]"
VERSION = re.compile(r"\d+\.\d+\.\d+")
# Link of the [Unreleased] heading, at the bottom of the changelog: it points to
# the commits since the previous release, or to all the commits before the first one
UNRELEASED_LINK = re.compile(
    r"^\[unreleased\]: (?P<base>\S+?)/(?:compare/(?P<previous>\S+)\.\.\.HEAD|commits/HEAD)$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_version(tag: str) -> str:
    """Return the version of a tag such as ``v0.2.0``, or of a bare version."""
    version = tag.removeprefix("v")
    if not VERSION.fullmatch(version):
        msg = f"{tag!r} is not a version such as 0.2.0 or v0.2.0"
        raise SystemExit(f"error: {msg}")
    return version


def section(changelog: str, heading: str) -> str | None:
    """Return the body of the section starting with ``heading``, or None if absent."""
    match = re.search(
        rf"^{re.escape(heading)}[^\n]*\n(?P<body>.*?)(?=^## |^\[[^\]]+\]: |\Z)",
        changelog,
        re.DOTALL | re.MULTILINE,
    )
    return match["body"].strip() if match else None


def prepare_changelog(changelog: str, version: str, date: datetime.date) -> str:
    """Turn the unreleased changes of ``changelog`` into the release ``version``."""
    if changelog.count(UNRELEASED) != 1:
        raise SystemExit(f"error: the changelog must contain one {UNRELEASED!r} heading")
    if section(changelog, f"## [{version}]") is not None:
        raise SystemExit(f"error: the changelog already contains version {version}")
    if not section(changelog, UNRELEASED):
        raise SystemExit("error: nothing to release, the [Unreleased] section is empty")
    changelog = changelog.replace(
        UNRELEASED, f"{UNRELEASED}\n\n## [{version}] - {date.isoformat()}", 1
    )
    # Searched after the new heading is inserted, so that its position is still right
    link = UNRELEASED_LINK.search(changelog)
    if link is None:
        raise SystemExit("error: the [unreleased] link is missing from the changelog")

    base, previous = link["base"], link["previous"]
    if previous:
        version_link = f"[{version}]: {base}/compare/{previous}...v{version}"
    else:
        version_link = f"[{version}]: {base}/releases/tag/v{version}"
    return (
        changelog[: link.start()]
        + f"[unreleased]: {base}/compare/v{version}...HEAD\n{version_link}"
        + changelog[link.end() :]
    )


def prepare_citation(citation: str, version: str, date: datetime.date) -> str:
    """Set the version and release date of ``citation``, adding them if missing."""
    for key, value in (("date-released", date.isoformat()), ("version", version)):
        line = f'{key}: "{value}"'
        citation, count = re.subn(rf"^{key}:.*$", line, citation, flags=re.MULTILINE)
        if count == 0:
            # Right after the title, where readers of the file expect them
            citation = re.sub(
                r"^(title:.*)$", rf"\1\n{line}", citation, count=1, flags=re.MULTILINE
            )
    return citation


def check(changelog: str, citation: str, version: str) -> list[str]:
    """Return the problems that prevent the release of ``version``, if any."""
    problems = []
    heading = re.search(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", changelog, re.MULTILINE
    )
    if heading is None:
        problems.append(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section")
    elif not section(changelog, f"## [{version}]"):
        problems.append(f"the [{version}] section of CHANGELOG.md is empty")
    if not re.search(rf'^version: "?{re.escape(version)}"?$', citation, re.MULTILINE):
        problems.append(f"CITATION.cff does not have version {version}")
    if heading and not re.search(rf'^date-released: "?{heading[1]}"?$', citation, re.MULTILINE):
        problems.append(
            f"the date-released of CITATION.cff is not {heading[1]}, the date of the changelog"
        )
    return problems


def main(argv: list[str] | None = None) -> None:
    """Run the command given on the command line."""
    parser = argparse.ArgumentParser(description="Prepare and check the releases of the project.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("prepare", "turn the unreleased changes into a release"),
        ("check", "check that the changelog and the citation are ready"),
        ("notes", "print the changelog section of a release"),
    ):
        command = commands.add_parser(name, help=help_)
        command.add_argument("version", help="version or tag, such as 0.2.0 or v0.2.0")
    args = parser.parse_args(argv)
    version = parse_version(args.version)
    changelog = CHANGELOG.read_text(encoding="utf-8")
    citation = CITATION.read_text(encoding="utf-8")

    if args.command == "prepare":
        today = datetime.date.today()
        # newline="\n": keep the LF line endings of the repository, even on Windows
        CHANGELOG.write_text(
            prepare_changelog(changelog, version, today), encoding="utf-8", newline="\n"
        )
        CITATION.write_text(
            prepare_citation(citation, version, today), encoding="utf-8", newline="\n"
        )
        print(f"Prepared version {version}: review the changes, commit them, then tag v{version}.")
    elif args.command == "check":
        problems = check(changelog, citation, version)
        if problems:
            raise SystemExit("\n".join(f"error: {problem}" for problem in problems))
        print(f"Version {version} is ready to be released.")
    else:
        print(section(changelog, f"## [{version}]") or "")


if __name__ == "__main__":
    main()
