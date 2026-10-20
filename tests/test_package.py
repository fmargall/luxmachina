# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Smoke tests for the luxmachina package."""

import luxmachina


def test_version_is_resolved():
    assert luxmachina.__version__ != "0+unknown"
