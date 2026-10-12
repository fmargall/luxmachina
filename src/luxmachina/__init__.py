# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall
"""Differentiable GPU light tracer built on NVIDIA Warp."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("luxmachina")
except PackageNotFoundError:  # pragma: no cover
    # Fallback when the source tree is imported without install
    # Tests always run on an installed package, so this line is
    # excluded from coverage measurement thanks to the pragma.
    __version__ = "0+unknown"
