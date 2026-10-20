# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Tests of scripts/release.py, which prepares the changelog and the citation."""

import datetime
import importlib.util
from pathlib import Path

import pytest

# scripts/ is not a package: load the script from its path
_spec = importlib.util.spec_from_file_location(
    "release", Path(__file__).resolve().parent.parent / "scripts" / "release.py"
)
assert _spec is not None
assert _spec.loader is not None
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

DATE = datetime.date(2026, 10, 20)
BASE = "https://github.com/owner/repo"

FIRST = f"""# Changelog

## [Unreleased]

### Added

- A first feature.

[unreleased]: {BASE}/commits/HEAD
"""

CITATION = """cff-version: 1.2.0
title: "Project"
authors:
    - family-names: "Doe"
"""


def test_first_release_links_to_its_tag():
    changelog = release.prepare_changelog(FIRST, "0.1.0", DATE)
    assert "## [Unreleased]\n\n## [0.1.0] - 2026-10-20\n\n### Added\n" in changelog
    assert release.section(changelog, "## [0.1.0]") == "### Added\n\n- A first feature."
    assert changelog.endswith(
        f"[unreleased]: {BASE}/compare/v0.1.0...HEAD\n[0.1.0]: {BASE}/releases/tag/v0.1.0\n"
    )


def test_next_release_compares_with_the_previous_one():
    changelog = release.prepare_changelog(FIRST, "0.1.0", DATE)
    fixed = "## [Unreleased]\n\n### Fixed\n\n- A bug.\n"
    changelog = changelog.replace("## [Unreleased]\n", fixed, 1)
    changelog = release.prepare_changelog(changelog, "0.2.0", DATE)
    assert f"[unreleased]: {BASE}/compare/v0.2.0...HEAD\n" in changelog
    assert f"[0.2.0]: {BASE}/compare/v0.1.0...v0.2.0\n" in changelog
    assert f"[0.1.0]: {BASE}/releases/tag/v0.1.0\n" in changelog
    assert release.section(changelog, "## [0.2.0]") == "### Fixed\n\n- A bug."
    assert release.section(changelog, "## [0.1.0]") == "### Added\n\n- A first feature."


def test_nothing_to_release():
    empty = FIRST.replace("### Added\n\n- A first feature.\n\n", "")
    with pytest.raises(SystemExit, match="nothing to release"):
        release.prepare_changelog(empty, "0.1.0", DATE)


def test_version_released_twice():
    changelog = release.prepare_changelog(FIRST, "0.1.0", DATE)
    with pytest.raises(SystemExit, match=r"already contains version 0\.1\.0"):
        release.prepare_changelog(changelog, "0.1.0", DATE)


def test_citation_gets_the_version_and_date():
    citation = release.prepare_citation(CITATION, "0.1.0", DATE)
    assert 'title: "Project"\nversion: "0.1.0"\ndate-released: "2026-10-20"\n' in citation
    citation = release.prepare_citation(citation, "0.2.0", datetime.date(2026, 11, 2))
    assert citation.count("version: ") == 2  # cff-version and version
    assert 'version: "0.2.0"\ndate-released: "2026-11-02"\n' in citation


def test_check_accepts_a_prepared_release():
    changelog = release.prepare_changelog(FIRST, "0.1.0", DATE)
    citation = release.prepare_citation(CITATION, "0.1.0", DATE)
    assert release.check(changelog, citation, "0.1.0") == []


def test_check_reports_what_was_forgotten():
    problems = release.check(FIRST, CITATION, "0.1.0")
    assert problems == [
        "CHANGELOG.md has no '## [0.1.0] - YYYY-MM-DD' section",
        "CITATION.cff does not have version 0.1.0",
    ]
    changelog = release.prepare_changelog(FIRST, "0.1.0", DATE)
    citation = release.prepare_citation(CITATION, "0.1.0", datetime.date(2026, 1, 1))
    assert release.check(changelog, citation, "0.1.0") == [
        "the date-released of CITATION.cff is not 2026-10-20, the date of the changelog"
    ]


@pytest.mark.parametrize(("tag", "version"), [("v1.2.3", "1.2.3"), ("1.2.3", "1.2.3")])
def test_parse_version(tag, version):
    assert release.parse_version(tag) == version


@pytest.mark.parametrize("tag", ["v1.2", "v1.2.3rc1", "latest"])
def test_parse_version_rejects_other_tags(tag):
    with pytest.raises(SystemExit, match="is not a version"):
        release.parse_version(tag)
