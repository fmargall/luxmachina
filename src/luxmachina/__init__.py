# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall
"""Differentiable GPU light tracer built on NVIDIA Warp."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("luxmachina")
except PackageNotFoundError:
    __version__ = "?.?.?"
