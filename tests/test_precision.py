# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Real types of a trace, and generic kernels in both precisions (ADR-0003)."""

from typing import Any

import numpy as np
import pytest
import warp as wp

from luxmachina._precision import REALS, check_real, vec3


def test_single_precision_is_the_default():
    assert REALS[0] is wp.float32


def test_check_real_accepts_supported_types(dtype):
    assert check_real(dtype) is dtype


@pytest.mark.parametrize("real", [wp.float16, float, np.float32])
def test_check_real_rejects_other_types(real):
    with pytest.raises(TypeError, match="unsupported real type"):
        check_real(real)


def test_vec3_matches_the_precision():
    assert vec3(wp.float32) is wp.vec3f
    assert vec3(wp.float64) is wp.vec3d


# The tests below check Warp itself: a generic kernel is specialized, runs and
# differentiates in both precisions. They go once kernels of the package cover it.


@wp.kernel
def cubic(x: wp.array(dtype=Any), y: wp.array(dtype=Any)):
    i = wp.tid()
    v = x[i]
    # The literal is built from the type of v: a bare 2.0 would be a float32
    y[i] = v * v * v + type(v)(2.0) * v


@wp.func
def sphere_far_root(o: Any, d: Any, c: Any):
    # Origin o on the sphere of center c, unit direction d (ADR-0002, known root)
    s = wp.dot(d, o - c)
    return -type(s)(2.0) * s


@wp.kernel
def far_root(
    o: wp.array(dtype=Any), d: wp.array(dtype=Any), c: wp.array(dtype=Any), t: wp.array(dtype=Any)
):
    i = wp.tid()
    t[i] = sphere_far_root(o[i], d[i], c[i])


def test_generic_kernel_gradient(device, dtype):
    x = wp.array([0.5, 1.0, 2.0], dtype=dtype, device=device, requires_grad=True)
    y = wp.zeros(3, dtype=dtype, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(cubic, dim=3, inputs=[x], outputs=[y], device=device)
    tape.backward(grads={y: wp.ones(3, dtype=dtype, device=device)})

    rtol = 1e-6 if dtype == wp.float32 else 1e-12
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), 3.0 * x.numpy() ** 2 + 2.0, rtol=rtol)


def test_float64_is_not_rounded_to_float32(device):
    # In float32, 1 + 1e-10 rounds to 1 and the result would be exactly 3
    x = wp.array([1.0 + 1e-10], dtype=wp.float64, device=device)
    y = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(cubic, dim=1, inputs=[x], outputs=[y], device=device)

    assert abs(y.numpy()[0] - (3.0 + 5e-10)) < 1e-14


def test_generic_vector_func_gradient(device, dtype):
    # Unit sphere at the origin; both rays leave its south pole
    o = wp.array([[0.0, 0.0, -1.0]] * 2, dtype=vec3(dtype), device=device)
    d = wp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=vec3(dtype), device=device)
    c = wp.zeros(2, dtype=vec3(dtype), device=device, requires_grad=True)
    t = wp.zeros(2, dtype=dtype, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(far_root, dim=2, inputs=[o, d, c], outputs=[t], device=device)
    tape.backward(grads={t: wp.ones(2, dtype=dtype, device=device)})

    # Inwards: the far side is 2 away; outwards: negative root, no hit
    np.testing.assert_allclose(t.numpy(), [2.0, -2.0])
    # dt/dc = 2 d
    assert c.grad is not None
    np.testing.assert_allclose(c.grad.numpy(), 2.0 * d.numpy())
