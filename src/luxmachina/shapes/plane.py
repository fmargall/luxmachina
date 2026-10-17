# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Plane through a point, with a unit normal."""

import functools
from typing import Any

import numpy as np
import numpy.typing as npt
import warp as wp

from luxmachina._precision import check_real, vec3


@functools.cache
def _plane_type(real: type) -> Any:
    """Return the Warp struct of a plane for the real type `real`, created once."""
    real_vec3 = vec3(real)

    @wp.struct
    class Plane:
        """Plane through `origin`, of unit normal `normal`."""

        #: A point of the plane.
        origin: real_vec3
        #: Unit normal, which defines the front side of the plane.
        normal: real_vec3

    return Plane


def make_planes(
    origins: npt.ArrayLike,
    normals: npt.ArrayLike,
    real: type = wp.float32,
    device: wp.DeviceLike = None,
    requires_grad: bool = False,
) -> wp.array:
    """Return an array of planes, from their points and normals.

    Parameters
    ----------
    origins, normals
        ``(N, 3)`` points of the planes and their normals. The normals are
        normalized.
    real
        Real type of the trace, ``wp.float32`` (the default) or ``wp.float64``.
    device
        Warp device of the array; the default device of Warp when omitted.
    requires_grad
        Whether the array keeps a gradient, for the backward pass.

    Returns
    -------
    wp.array
        Array of ``N`` Warp structs ``Plane``.
    """
    plane_type = _plane_type(check_real(real))
    points = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(points) != len(directions):
        raise ValueError(f"got {len(points)} origins for {len(directions)} normals")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("the normal of a plane cannot be zero")
    planes = []
    for point, direction in zip(points, directions / lengths[:, None], strict=True):
        plane = plane_type()
        plane.origin = vec3(real)(*point)
        plane.normal = vec3(real)(*direction)
        planes.append(plane)
    return wp.array(planes, dtype=plane_type, device=device, requires_grad=requires_grad)


@wp.func
def intersect_plane(o: Any, d: Any, plane: Any) -> Any:
    """Return the distance from `o` along `d` to `plane`, or ``+inf`` if the ray misses it."""
    cosine = wp.dot(d, plane.normal)
    t = type(cosine)(wp.inf)
    if cosine != type(cosine)(0.0):
        distance = wp.dot(plane.origin - o, plane.normal) / cosine
        if distance > type(cosine)(0.0):
            t = distance
    return t


@wp.func
def intersect_plane_from(o: Any, d: Any, plane: Any) -> Any:
    """Return ``+inf``: a ray leaving a plane never meets it again."""
    return type(wp.dot(d, plane.normal))(wp.inf)


@wp.func
def plane_normal(p: Any, plane: Any) -> Any:
    """Return the unit normal of `plane`, the same at every point `p`."""
    return plane.normal
