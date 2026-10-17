# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Real types of a trace: single or double precision (ADR-0003).

Kernels and functions are written once, generic over the real type, with
``typing.Any`` as Warp documents it; Warp specializes them at launch from the
types of their arguments. Inside them, literals are built from the type of a
value, as in ``type(x)(2.0)``: a bare ``2.0`` is a float32 for Warp and does
not mix with float64 values.
"""

import warp as wp

# Supported real types and their 3-vector type, single precision (the default) first
_VEC3: dict[type, type] = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}
# Their 3x3 matrix type
_MAT33: dict[type, type] = {wp.float32: wp.mat33f, wp.float64: wp.mat33d}

#: Real types a trace can use, single precision (the default) first.
REALS: tuple[type, ...] = tuple(_VEC3)


def check_real(real: type) -> type:
    """Return `real` if a trace can use it, raise a `TypeError` otherwise."""
    if real not in _VEC3:
        expected = ", ".join(r.__name__ for r in REALS)
        raise TypeError(f"unsupported real type {real!r}, expected one of: {expected}")
    return real


def vec3(real: type) -> type:
    """Return the Warp 3-vector type of the real type `real` (vec3f or vec3d)."""
    return _VEC3[check_real(real)]


def mat33(real: type) -> type:
    """Return the Warp 3x3 matrix type of the real type `real` (mat33f or mat33d)."""
    return _MAT33[check_real(real)]
