# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Shared fixtures: every test runs on each device and in each precision."""

import pytest
import warp as wp

from luxmachina._precision import REALS

# Show warnings and errors only: hides the banner printed at initialization
wp.config.log_level = wp.LOG_WARNING
wp.init()


@pytest.fixture(params=["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def device(request):
    """Warp device; CUDA is skipped, not removed, when no GPU is available."""
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return wp.get_device(request.param)


@pytest.fixture(params=REALS, ids=[real.__name__ for real in REALS])
def dtype(request):
    """Real type of a trace (ADR-0003: the precision is a parameter)."""
    return request.param
