# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

# pyright: reportInvalidTypeForm=false
# Warp kernel annotations such as wp.array(dtype=...) are calls, which pyright rejects

"""Spike for ADR-0003: one generic kernel, both precisions, differentiable."""

from typing import Any

import numpy as np
import warp as wp


@wp.kernel
def cube(x: wp.array(dtype=Any), y: wp.array(dtype=Any)):
    i = wp.tid()
    y[i] = x[i] * x[i] * x[i]


def test_generic_kernel_gradient(device, dtype):
    x = wp.array([0.5, 1.0, 2.0], dtype=dtype, device=device, requires_grad=True)
    y = wp.zeros(3, dtype=dtype, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(cube, dim=3, inputs=[x], outputs=[y], device=device)
    tape.backward(grads={y: wp.ones(3, dtype=dtype, device=device)})

    rtol = 1e-6 if dtype == wp.float32 else 1e-12
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), 3.0 * x.numpy() ** 2, rtol=rtol)
