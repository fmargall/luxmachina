# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""State of the rays traced at one depth (ADR-0003).

The rays are stored as a structure of arrays: a Warp struct whose fields are
arrays, one value per ray. Kernels receive it as a single argument and read
only the fields they need, for example ``rays.origin[i]``. The tracer keeps
one such struct per depth, so that no buffer is overwritten during a trace
(ADR-0002).
"""

import enum
import functools
from typing import Any

import warp as wp

from luxmachina._precision import check_real, vec3


class Status(enum.IntEnum):
    """What happened to a ray, stored in the ``status`` field."""

    #: Not traced: never emitted, or padding of the last batch (the value of a new buffer).
    INACTIVE = 0
    #: Still traveling through the scene.
    ALIVE = 1
    #: Absorbed by a surface or by the medium it travels in.
    ABSORBED = 2
    #: Left the scene without hitting any surface.
    ESCAPED = 3
    #: Recorded by a detector, which stopped it.
    RECORDED = 4


@functools.cache
def _rays_type(real: type) -> Any:
    """Return the Warp struct of the rays for the real type `real`, created once."""
    real_vec3 = vec3(real)

    @wp.struct
    class Rays:
        """Rays traced at one depth, one array element per ray."""

        #: Start point of the current segment.
        origin: wp.array(dtype=real_vec3)
        #: Unit direction of the current segment.
        direction: wp.array(dtype=real_vec3)
        #: Power carried by the ray.
        power: wp.array(dtype=real)
        #: Vacuum wavelength.
        wavelength: wp.array(dtype=real)
        #: Optical path length from the source.
        opl: wp.array(dtype=real)
        #: Index of the material the ray travels in.
        medium: wp.array(dtype=wp.int32)
        #: Index of the surface the ray leaves, -1 at the source.
        surface: wp.array(dtype=wp.int32)
        #: A Status value.
        status: wp.array(dtype=wp.int32)

    return Rays


def allocate_rays(
    count: int,
    real: type = wp.float32,
    device: wp.DeviceLike = None,
    requires_grad: bool = False,
) -> Any:
    """Allocate the state of `count` rays, all inactive.

    Parameters
    ----------
    count
        Number of rays.
    real
        Real type of the trace, ``wp.float32`` (the default) or ``wp.float64``.
    device
        Warp device of the arrays; the default device of Warp when omitted.
    requires_grad
        Whether the real fields keep a gradient, for the backward pass. The
        integer fields never do.

    Returns
    -------
    Rays
        Warp struct of arrays of length `count`, filled with zeros: every ray
        is ``Status.INACTIVE`` until a source emits it.
    """
    if count < 0:
        raise ValueError(f"the number of rays must be non-negative, got {count}")
    real = check_real(real)
    real_vec3 = vec3(real)

    rays = _rays_type(real)()
    rays.origin = wp.zeros(count, dtype=real_vec3, device=device, requires_grad=requires_grad)
    rays.direction = wp.zeros(count, dtype=real_vec3, device=device, requires_grad=requires_grad)
    rays.power = wp.zeros(count, dtype=real, device=device, requires_grad=requires_grad)
    rays.wavelength = wp.zeros(count, dtype=real, device=device, requires_grad=requires_grad)
    rays.opl = wp.zeros(count, dtype=real, device=device, requires_grad=requires_grad)
    rays.medium = wp.zeros(count, dtype=wp.int32, device=device)
    rays.surface = wp.zeros(count, dtype=wp.int32, device=device)
    rays.status = wp.zeros(count, dtype=wp.int32, device=device)
    return rays
