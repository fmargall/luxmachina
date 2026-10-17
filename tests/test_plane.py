# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Plane: intersections and normal (ADR-0002)."""

from typing import Any

import numpy as np
import pytest
import warp as wp

from luxmachina._precision import vec3
from luxmachina.shapes.plane import (
    _plane_type,
    intersect_plane,
    intersect_plane_from,
    make_planes,
    plane_normal,
)


@wp.kernel
def distances(
    o: wp.array(dtype=Any),
    d: wp.array(dtype=Any),
    planes: wp.array(dtype=Any),
    t: wp.array(dtype=Any),
):
    i = wp.tid()
    t[i] = intersect_plane(o[i], d[i], planes[i])


@wp.kernel
def distances_from(
    o: wp.array(dtype=Any),
    d: wp.array(dtype=Any),
    planes: wp.array(dtype=Any),
    t: wp.array(dtype=Any),
):
    i = wp.tid()
    t[i] = intersect_plane_from(o[i], d[i], planes[i])


@wp.kernel
def normals(p: wp.array(dtype=Any), planes: wp.array(dtype=Any), n: wp.array(dtype=Any)):
    i = wp.tid()
    n[i] = plane_normal(p[i], planes[i])


def trace(kernel, origins, directions, plane_origins, plane_normals, real, device):
    """Launch `kernel` on one ray per plane, return the distances."""
    count = len(origins)
    o = wp.array(np.asarray(origins), dtype=vec3(real), device=device)
    d = wp.array(np.asarray(directions), dtype=vec3(real), device=device)
    planes = make_planes(plane_origins, plane_normals, real, device)
    t = wp.zeros(count, dtype=real, device=device)
    wp.launch(kernel, dim=count, inputs=[o, d, planes], outputs=[t], device=device)
    return t.numpy()


def tolerance(real):
    return 1e-6 if real == wp.float32 else 1e-14


@pytest.mark.parametrize(
    ("direction", "normal", "expected"),
    [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 5.0),  # perpendicular
        ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), 5.0),  # the side of the normal does not matter
        ((0.0, 0.6, 0.8), (0.0, 0.0, 1.0), 6.25),  # oblique: 5 / cos
        ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), np.inf),  # plane behind
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.inf),  # parallel
    ],
)
def test_intersection(device, dtype, direction, normal, expected):
    t = trace(distances, [(0.0, 0.0, 0.0)], [direction], [(1.0, 2.0, 5.0)], [normal], dtype, device)

    np.testing.assert_allclose(t, [expected], rtol=tolerance(dtype))


def test_intersection_from_the_plane_is_always_a_miss(device, dtype):
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(100, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = np.column_stack([rng.uniform(-5.0, 5.0, (100, 2)), np.full(100, 5.0)])
    t = trace(
        distances_from,
        points,
        directions,
        [(0.0, 0.0, 5.0)] * 100,
        [(0.0, 0.0, 1.0)] * 100,
        dtype,
        device,
    )

    assert np.all(np.isinf(t))


def test_normal_is_the_normalized_normal(device, dtype):
    p = wp.array([(3.0, -1.0, 5.0)], dtype=vec3(dtype), device=device)
    planes = make_planes([(0.0, 0.0, 5.0)], [(0.0, 3.0, 4.0)], dtype, device)
    n = wp.zeros(1, dtype=vec3(dtype), device=device)
    wp.launch(normals, dim=1, inputs=[p, planes], outputs=[n], device=device)

    np.testing.assert_allclose(n.numpy(), [(0.0, 0.6, 0.8)], rtol=tolerance(dtype))


def test_gradients_match_the_analytic_derivatives(device):
    # t = (p0 - o).n / (d.n)
    o, d = np.array([0.1, -0.2, 0.3]), np.array([0.0, 0.6, 0.8])
    p0, n = np.array([1.0, 2.0, 5.0]), np.array([0.0, 0.0, 1.0])
    rays_o = wp.array([o], dtype=wp.vec3d, device=device, requires_grad=True)
    rays_d = wp.array([d], dtype=wp.vec3d, device=device, requires_grad=True)
    planes = make_planes([p0], [n], wp.float64, device, requires_grad=True)
    t = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(distances, dim=1, inputs=[rays_o, rays_d, planes], outputs=[t], device=device)
    tape.backward(grads={t: wp.ones(1, dtype=wp.float64, device=device)})

    distance = (p0 - o) @ n / (d @ n)
    assert rays_o.grad is not None and rays_d.grad is not None and planes.grad is not None
    plane = planes.grad.numpy()[0]
    np.testing.assert_allclose(rays_o.grad.numpy()[0], -n / (d @ n))
    np.testing.assert_allclose(rays_d.grad.numpy()[0], -distance * n / (d @ n))
    np.testing.assert_allclose(plane["origin"], n / (d @ n))
    np.testing.assert_allclose(plane["normal"], (p0 - o) / (d @ n) - distance * d / (d @ n))


def test_make_planes_builds_the_array(device, dtype):
    planes = make_planes(
        [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)], [(0.0, 0.0, 2.0), (1.0, 0.0, 0.0)], dtype, device
    )

    assert planes.shape == (2,)
    assert planes.dtype is _plane_type(dtype)
    assert planes.device == device
    np.testing.assert_allclose(
        [plane["normal"] for plane in planes.numpy()], [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
    )


def test_struct_is_created_once_per_precision():
    assert _plane_type(wp.float32) is _plane_type(wp.float32)
    assert _plane_type(wp.float32) is not _plane_type(wp.float64)


def test_make_planes_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="2 origins for 1 normals"):
        make_planes([(0.0, 0.0, 0.0)] * 2, [(0.0, 0.0, 1.0)])


def test_make_planes_rejects_zero_normals():
    with pytest.raises(ValueError, match="cannot be zero"):
        make_planes([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])
