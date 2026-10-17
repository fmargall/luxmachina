# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Local frames of the surfaces (ADR-0004).

The local frame of a surface has its origin at the vertex of the surface and
its ``z`` axis along the local optical axis; its ``x`` and ``y`` axes span the
plane tangent to the surface at the vertex, on which apertures are expressed.
A frame is right-handed and orthonormal. It is stored as a rotation, whose
columns are the local axes in global coordinates, and as its origin.
"""

import functools
from typing import Any

import numpy as np
import numpy.typing as npt
import warp as wp

from luxmachina._precision import check_real, mat33, vec3


@functools.cache
def _frame_type(real: type) -> Any:
    """Return the Warp struct of a frame for the real type `real`, created once."""
    real_vec3 = vec3(real)
    real_mat33 = mat33(real)

    @wp.struct
    class Frame:
        """Right-handed orthonormal frame, from local to global coordinates."""

        #: Local x, y and z axes, as the columns of the matrix, in global coordinates.
        rotation: real_mat33
        #: Origin of the frame, the vertex of the surface, in global coordinates.
        origin: real_vec3

    return Frame


def _perpendicular_axes(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y axes completing the unit axes `z` into right-handed frames.

    Duff et al., "Building an Orthonormal Basis, Revisited", JCGT 6(1), 2017:
    defined for every axis, it only jumps where the z component of the axis
    changes sign. The global frame is returned for ``z = (0, 0, 1)``.
    """
    sign = np.copysign(1.0, z[:, 2])
    a = -1.0 / (sign + z[:, 2])
    b = z[:, 0] * z[:, 1] * a
    x = np.column_stack([1.0 + sign * z[:, 0] ** 2 * a, sign * b, -sign * z[:, 0]])
    y = np.column_stack([b, sign + z[:, 1] ** 2 * a, -z[:, 1]])
    return x, y


def make_frames(
    origins: npt.ArrayLike,
    axes: npt.ArrayLike,
    x_axes: npt.ArrayLike | None = None,
    real: type = wp.float32,
    device: wp.DeviceLike = None,
    requires_grad: bool = False,
) -> wp.array:
    """Return an array of local frames, from their origins and optical axes.

    Parameters
    ----------
    origins
        ``(N, 3)`` origins of the frames, the vertices of the surfaces, in meters.
    axes
        ``(N, 3)`` local optical axes, the ``z`` axes of the frames; they are
        normalized.
    x_axes
        ``(N, 3)`` directions of the ``x`` axes, which matter for apertures
        that are not rotationally symmetric. Each is projected on the plane
        perpendicular to its axis, then normalized. When omitted, the ``x``
        axes are derived from the optical axes, and the frame of the axis
        ``(0, 0, 1)`` is the global frame.
    real
        Real type of the trace, ``wp.float32`` (the default) or ``wp.float64``.
    device
        Warp device of the array; the default device of Warp when omitted.
    requires_grad
        Whether the array keeps a gradient, for the backward pass.

    Returns
    -------
    wp.array
        Array of ``N`` Warp structs ``Frame``.
    """
    frame_type = _frame_type(check_real(real))
    points = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    z = np.asarray(axes, dtype=np.float64).reshape(-1, 3)
    if len(points) != len(z):
        raise ValueError(f"got {len(points)} origins for {len(z)} axes")
    lengths = np.linalg.norm(z, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("the axis of a frame cannot be zero")
    z = z / lengths[:, None]

    if x_axes is None:
        x, y = _perpendicular_axes(z)
    else:
        x = np.asarray(x_axes, dtype=np.float64).reshape(-1, 3)
        if len(x) != len(z):
            raise ValueError(f"got {len(x)} x axes for {len(z)} axes")
        x = x - np.sum(x * z, axis=1, keepdims=True) * z
        lengths = np.linalg.norm(x, axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("the x axis of a frame must not be parallel to its axis")
        x = x / lengths[:, None]
        y = np.cross(z, x)

    frames = []
    for point, rotation in zip(points, np.stack([x, y, z], axis=2), strict=True):
        frame = frame_type()
        frame.rotation = mat33(real)(*rotation.ravel())
        frame.origin = vec3(real)(*point)
        frames.append(frame)
    return wp.array(frames, dtype=frame_type, device=device, requires_grad=requires_grad)


@wp.func
def to_local_point(p: Any, frame: Any) -> Any:
    """Return the local coordinates of the point `p`, given in global coordinates."""
    return wp.transpose(frame.rotation) @ (p - frame.origin)


@wp.func
def to_global_point(q: Any, frame: Any) -> Any:
    """Return the global coordinates of the point `q`, given in local coordinates."""
    return frame.rotation @ q + frame.origin


@wp.func
def to_local_direction(d: Any, frame: Any) -> Any:
    """Return the local components of the direction `d`, given in global coordinates."""
    return wp.transpose(frame.rotation) @ d


@wp.func
def to_global_direction(v: Any, frame: Any) -> Any:
    """Return the global components of the direction `v`, given in local coordinates."""
    return frame.rotation @ v
