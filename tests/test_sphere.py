# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Sphere: intersections, known root and normal (ADR-0002)."""

from typing import Any

import numpy as np
import pytest
import warp as wp

from luxmachina._precision import vec3
from luxmachina.shapes.sphere import (
    _sphere_type,
    intersect_sphere,
    intersect_sphere_from,
    make_spheres,
    sphere_normal,
)

CENTER = (1.0, 2.0, 3.0)
RADIUS = 2.0


@wp.kernel
def distances(
    o: wp.array(dtype=Any),
    d: wp.array(dtype=Any),
    spheres: wp.array(dtype=Any),
    t: wp.array(dtype=Any),
):
    i = wp.tid()
    t[i] = intersect_sphere(o[i], d[i], spheres[i])


@wp.kernel
def distances_from(
    o: wp.array(dtype=Any),
    d: wp.array(dtype=Any),
    spheres: wp.array(dtype=Any),
    t: wp.array(dtype=Any),
):
    i = wp.tid()
    t[i] = intersect_sphere_from(o[i], d[i], spheres[i])


@wp.kernel
def normals(p: wp.array(dtype=Any), spheres: wp.array(dtype=Any), n: wp.array(dtype=Any)):
    i = wp.tid()
    n[i] = sphere_normal(p[i], spheres[i])


def trace(kernel, origins, directions, centers, radii, real, device):
    """Launch `kernel` on one ray per sphere, return the distances."""
    count = len(origins)
    o = wp.array(np.asarray(origins), dtype=vec3(real), device=device)
    d = wp.array(np.asarray(directions), dtype=vec3(real), device=device)
    spheres = make_spheres(centers, radii, real, device)
    t = wp.zeros(count, dtype=real, device=device)
    wp.launch(kernel, dim=count, inputs=[o, d, spheres], outputs=[t], device=device)
    return t.numpy()


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def tolerance(real):
    return 1e-5 if real == wp.float32 else 1e-12


@pytest.mark.parametrize(
    ("offset", "direction", "expected"),
    [
        ((0.0, 0.0, -5.0), (0.0, 0.0, 1.0), 3.0),  # outside, towards the center: near side
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0),  # at the center: one radius
        ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), 3.0),  # inside: far side
        ((3.0, 0.0, -5.0), (0.0, 0.0, 1.0), np.inf),  # passes beside
        ((0.0, 0.0, 5.0), (0.0, 0.0, 1.0), np.inf),  # sphere behind
    ],
)
def test_intersection(device, dtype, offset, direction, expected):
    origin = np.add(CENTER, offset)
    t = trace(distances, [origin], [direction], [CENTER], [RADIUS], dtype, device)

    np.testing.assert_allclose(t, [expected], rtol=tolerance(dtype))


def test_intersections_lie_on_the_sphere(device, dtype):
    rng = np.random.default_rng(0)
    origins = np.add(CENTER, rng.uniform(-4.0, 4.0, (1000, 3)))
    directions = unit(rng.normal(size=(1000, 3)))
    t = trace(distances, origins, directions, [CENTER] * 1000, [RADIUS] * 1000, dtype, device)

    hit = np.isfinite(t)
    assert 0 < hit.sum() < 1000  # both hits and misses are covered
    assert np.all(t[hit] > 0.0)
    radii = np.linalg.norm(origins[hit] + t[hit, None] * directions[hit] - CENTER, axis=1)
    np.testing.assert_allclose(radii, RADIUS, rtol=10 * tolerance(dtype))


def test_intersection_from_the_sphere(device, dtype):
    # Rays leaving points of the sphere in every direction
    rng = np.random.default_rng(1)
    points = np.add(CENTER, RADIUS * unit(rng.normal(size=(1000, 3))))
    directions = unit(rng.normal(size=(1000, 3)))
    t = trace(distances_from, points, directions, [CENTER] * 1000, [RADIUS] * 1000, dtype, device)

    inwards = np.sum(directions * (points - CENTER), axis=1) < 0.0
    assert np.all(np.isinf(t[~inwards]))
    # The chord of a ray going inwards ends on the sphere
    ends = points[inwards] + t[inwards, None] * directions[inwards]
    np.testing.assert_allclose(
        np.linalg.norm(ends - CENTER, axis=1), RADIUS, rtol=10 * tolerance(dtype)
    )


def test_intersection_from_the_sphere_never_finds_its_origin(device, dtype):
    # Grazing rays: the regular intersection may find a root close to 0, never this one
    points = np.add(CENTER, [(0.0, 0.0, -RADIUS)] * 3)
    directions = unit([(1.0, 0.0, 1e-3), (1.0, 0.0, 0.0), (1.0, 0.0, -1e-3)])
    t = trace(distances_from, points, directions, [CENTER] * 3, [RADIUS] * 3, dtype, device)

    assert t[0] > 0.0 and np.isfinite(t[0])
    np.testing.assert_allclose(t[0], 2.0 * RADIUS * 1e-3 / np.sqrt(1.0 + 1e-6), rtol=1e-3)
    assert np.all(np.isinf(t[1:]))


def test_normal_is_outward_and_unit(device, dtype):
    rng = np.random.default_rng(2)
    outward = unit(rng.normal(size=(100, 3)))
    p = wp.array(np.add(CENTER, RADIUS * outward), dtype=vec3(dtype), device=device)
    spheres = make_spheres([CENTER] * 100, [RADIUS] * 100, dtype, device)
    n = wp.zeros(100, dtype=vec3(dtype), device=device)
    wp.launch(normals, dim=100, inputs=[p, spheres], outputs=[n], device=device)

    np.testing.assert_allclose(n.numpy(), outward, atol=10 * tolerance(dtype))


def gradients(kernel, origin, direction, center, radius, device):
    """Return dt/d(origin, direction, center, radius) of one float64 ray, from the tape."""
    o = wp.array([origin], dtype=wp.vec3d, device=device, requires_grad=True)
    d = wp.array([direction], dtype=wp.vec3d, device=device, requires_grad=True)
    spheres = make_spheres([center], [radius], wp.float64, device, requires_grad=True)
    t = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=1, inputs=[o, d, spheres], outputs=[t], device=device)
    tape.backward(grads={t: wp.ones(1, dtype=wp.float64, device=device)})
    assert o.grad is not None and d.grad is not None and spheres.grad is not None
    sphere = spheres.grad.numpy()[0]
    return o.grad.numpy()[0], d.grad.numpy()[0], sphere["center"], sphere["radius"]


def finite_differences(kernel, origin, direction, center, radius, device, step=1e-6):
    """Return the same derivatives by central finite differences."""

    def distance(o, d, c, r):
        return trace(kernel, [o], [d], [c], [r], wp.float64, device)[0]

    def derivative(index, axis):
        values = [np.array(origin), np.array(direction), np.array(center), np.array([radius])]
        values[index][axis] += step
        forward = distance(values[0], values[1], values[2], values[3][0])
        values[index][axis] -= 2.0 * step
        backward = distance(values[0], values[1], values[2], values[3][0])
        return (forward - backward) / (2.0 * step)

    return [
        np.array([derivative(index, axis) for axis in range(size)])
        for index, size in enumerate((3, 3, 3, 1))
    ]


@pytest.mark.parametrize(
    ("kernel", "origin", "direction"),
    [
        (distances, (0.5, 1.5, -1.0), unit((0.1, 0.2, 1.0))),  # from outside
        (distances, (1.2, 2.3, 2.5), unit((0.3, -0.2, 1.0))),  # from inside
        (distances_from, np.add(CENTER, RADIUS * unit((0.2, 0.1, -1.0))), unit((0.1, 0.3, 1.0))),
    ],
    ids=["outside", "inside", "from the sphere"],
)
def test_gradients_match_finite_differences(device, kernel, origin, direction):
    tape = gradients(kernel, origin, direction, CENTER, RADIUS, device)
    reference = finite_differences(kernel, origin, direction, CENTER, RADIUS, device)

    for computed, expected in zip(tape, reference, strict=True):
        np.testing.assert_allclose(np.ravel(computed), expected, rtol=1e-6, atol=1e-8)


def test_make_spheres_builds_the_array(device, dtype):
    spheres = make_spheres([CENTER, (0.0, 0.0, 0.0)], [RADIUS, 1.0], dtype, device)

    assert spheres.shape == (2,)
    assert spheres.dtype is _sphere_type(dtype)
    assert spheres.device == device
    assert [float(sphere["radius"]) for sphere in spheres.numpy()] == [RADIUS, 1.0]


def test_struct_is_created_once_per_precision():
    assert _sphere_type(wp.float32) is _sphere_type(wp.float32)
    assert _sphere_type(wp.float32) is not _sphere_type(wp.float64)


def test_make_spheres_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="2 centers for 1 radii"):
        make_spheres([CENTER, CENTER], [RADIUS])


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_make_spheres_rejects_non_positive_radii(radius):
    with pytest.raises(ValueError, match="must be positive"):
        make_spheres([CENTER], [radius])


def test_make_spheres_rejects_unsupported_precisions():
    with pytest.raises(TypeError, match="unsupported real type"):
        make_spheres([CENTER], [RADIUS], float)
