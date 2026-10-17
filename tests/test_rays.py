# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""State of the rays traced at one depth (ADR-0003)."""

from typing import Any

import numpy as np
import pytest
import warp as wp

from luxmachina._precision import vec3
from luxmachina.rays import Status, _rays_type, allocate_rays

REAL_FIELDS = ("origin", "direction", "power", "wavelength", "opl")
INTEGER_FIELDS = ("medium", "surface", "status")


def test_fields_have_the_types_of_the_precision(device, dtype):
    rays = allocate_rays(5, dtype, device)

    assert rays.origin.dtype is vec3(dtype)
    assert rays.direction.dtype is vec3(dtype)
    for name in ("power", "wavelength", "opl"):
        assert getattr(rays, name).dtype is dtype
    for name in INTEGER_FIELDS:
        assert getattr(rays, name).dtype is wp.int32


def test_fields_hold_one_value_per_ray_on_the_device(device, dtype):
    rays = allocate_rays(5, dtype, device)

    for name in REAL_FIELDS + INTEGER_FIELDS:
        field = getattr(rays, name)
        assert field.shape == (5,)
        assert field.device == device


def test_new_rays_are_inactive_and_zero(device):
    rays = allocate_rays(3, device=device)

    assert np.all(rays.status.numpy() == Status.INACTIVE)
    for name in REAL_FIELDS + INTEGER_FIELDS:
        assert not np.any(getattr(rays, name).numpy())


def test_single_precision_is_the_default():
    assert allocate_rays(1).power.dtype is wp.float32


@pytest.mark.parametrize("requires_grad", [False, True])
def test_only_real_fields_keep_a_gradient(device, requires_grad):
    rays = allocate_rays(2, device=device, requires_grad=requires_grad)

    for name in REAL_FIELDS:
        assert getattr(rays, name).requires_grad is requires_grad
    for name in INTEGER_FIELDS:
        assert getattr(rays, name).requires_grad is False


def test_struct_is_created_once_per_precision():
    assert _rays_type(wp.float32) is _rays_type(wp.float32)
    assert _rays_type(wp.float32) is not _rays_type(wp.float64)


def test_empty_buffer_is_allowed():
    assert allocate_rays(0).origin.shape == (0,)


def test_negative_count_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        allocate_rays(-1)


def test_unsupported_precision_is_rejected():
    with pytest.raises(TypeError, match="unsupported real type"):
        allocate_rays(1, float)


@wp.kernel
def advance(rays: Any, t: Any, moved: Any):
    # Move each ray by t along its direction, in a medium of index 1
    i = wp.tid()
    moved.origin[i] = rays.origin[i] + t * rays.direction[i]
    moved.opl[i] = rays.opl[i] + t


@wp.kernel
def loss(rays: Any, total: wp.array(dtype=Any)):
    i = wp.tid()
    z = rays.origin[i][2]
    wp.atomic_add(total, 0, z * z + rays.opl[i])


def test_gradients_flow_through_the_struct(device, dtype):
    rays = allocate_rays(2, dtype, device, requires_grad=True)
    moved = allocate_rays(2, dtype, device, requires_grad=True)
    rays.origin.assign(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]))
    rays.direction.assign(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
    total = wp.zeros(1, dtype=dtype, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(advance, dim=2, inputs=[rays, dtype(0.5)], outputs=[moved], device=device)
        wp.launch(loss, dim=2, inputs=[moved], outputs=[total], device=device)
    tape.backward(loss=total)

    # total = sum((z + t)^2 + opl + t), with t = 0.5
    assert rays.origin.grad is not None
    assert rays.direction.grad is not None
    assert rays.opl.grad is not None
    np.testing.assert_allclose(rays.origin.grad.numpy()[:, 2], [3.0, 5.0])  # 2 (z + t)
    np.testing.assert_allclose(rays.direction.grad.numpy()[:, 2], [1.5, 2.5])  # 2 (z + t) t
    np.testing.assert_allclose(rays.opl.grad.numpy(), [1.0, 1.0])
