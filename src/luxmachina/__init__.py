# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("luxmachina")
except PackageNotFoundError:
    __version__ = "?.?.?"
