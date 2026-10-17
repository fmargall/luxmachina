# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Sphere, from its center and its radius."""

import functools
from typing import Any

import numpy as np
import numpy.typing as npt
import warp as wp

from luxmachina._precision import check_real, vec3


@functools.cache
def _sphere_type(real: type) -> Any:
    """Return the Warp struct of a sphere for the real type `real`, created once."""
    real_vec3 = vec3(real)

    @wp.struct
    class Sphere:
        """Sphere of center `center` and radius `radius`."""

        #: Center of the sphere.
        center: real_vec3
        #: Radius of the sphere, positive.
        radius: real

    return Sphere


def make_spheres(
    centers: npt.ArrayLike,
    radii: npt.ArrayLike,
    real: type = wp.float32,
    device: wp.DeviceLike = None,
    requires_grad: bool = False,
) -> wp.array:
    """Return an array of spheres, from their centers and radii.

    Parameters
    ----------
    centers
        ``(N, 3)`` centers of the spheres.
    radii
        ``(N,)`` radii of the spheres, positive.
    real
        Real type of the trace, ``wp.float32`` (the default) or ``wp.float64``.
    device
        Warp device of the array; the default device of Warp when omitted.
    requires_grad
        Whether the array keeps a gradient, for the backward pass.

    Returns
    -------
    wp.array
        Array of ``N`` Warp structs ``Sphere``.
    """
    sphere_type = _sphere_type(check_real(real))
    points = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    lengths = np.asarray(radii, dtype=np.float64).reshape(-1)
    if len(points) != len(lengths):
        raise ValueError(f"got {len(points)} centers for {len(lengths)} radii")
    if np.any(lengths <= 0.0):
        raise ValueError("the radius of a sphere must be positive")
    spheres = []
    for point, length in zip(points, lengths, strict=True):
        sphere = sphere_type()
        sphere.center = vec3(real)(*point)
        sphere.radius = real(length)
        spheres.append(sphere)
    return wp.array(spheres, dtype=sphere_type, device=device, requires_grad=requires_grad)


@wp.func
def intersect_sphere(o: Any, d: Any, sphere: Any) -> Any:
    """Return the distance from `o` along `d` to `sphere`, or ``+inf`` if the ray misses it."""
    # |o + t d - c|^2 = r^2 with |d| = 1 gives t^2 + 2 b t + q = 0
    oc = o - sphere.center
    b = wp.dot(d, oc)
    q = wp.dot(oc, oc) - sphere.radius * sphere.radius
    zero = type(b)(0.0)
    t = type(b)(wp.inf)
    discriminant = b * b - q
    if discriminant >= zero:
        # Stable roots: the one of largest magnitude, then q / root (product of the roots)
        root = -b - wp.sign(b) * wp.sqrt(discriminant)
        if root != zero:
            near = wp.min(root, q / root)
            far = wp.max(root, q / root)
            if near > zero:
                t = near
            elif far > zero:
                t = far
    return t


@wp.func
def intersect_sphere_from(o: Any, d: Any, sphere: Any) -> Any:
    """Return the distance from `o`, a point of `sphere`, along `d` to `sphere`, or ``+inf``.

    Once the root ``t = 0`` is removed, the other root is ``-2 d.(o - c)``. It
    is positive only when the ray goes into the sphere.
    """
    t = type(sphere.radius)(-2.0) * wp.dot(d, o - sphere.center)
    if t <= type(t)(0.0):
        t = type(t)(wp.inf)
    return t


@wp.func
def sphere_normal(p: Any, sphere: Any) -> Any:
    """Return the outward unit normal of `sphere` at its point `p`."""
    return (p - sphere.center) / sphere.radius
