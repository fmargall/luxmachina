# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

# Warp device code: the type stubs of Warp do not describe operations on its
# scalars and vectors (indexing, shifts, constructors), Warp checks them itself
# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportArgumentType=false

"""Counter-based random numbers (ADR-0003).

Every random number is a pure function of its coordinates: the seed of the
trace, the batch, the ray, the depth and the purpose of the draw (emission
position, wavelength, Fresnel choice...). Nothing is stored between draws, so
the same coordinates always give the same numbers, whatever the order of the
kernels or the device.

The generator is Philox-4x32-10 (Salmon et al., "Parallel Random Numbers: As
Easy as 1, 2, 3", SC'11, 2011), the one of cuRAND, JAX and PyTorch: it maps a
128-bit counter and a 64-bit key to 128 random bits, using only integer
operations, so the bits are identical on every device. Each call gives four
32-bit words, that is four single-precision or two double-precision numbers.

The numbers are uniform in the open interval (0, 1), so that a logarithm or an
inverse cumulative distribution never meets 0 or 1. They are the centers of
2**23 (single) or 2**52 (double) equal bins: one bit fewer than the mantissa,
so that every center is exactly representable.
"""

import operator
from typing import SupportsIndex

import warp as wp

# Philox-4x32 multipliers and Weyl key increments (golden ratio, sqrt(3) - 1)
_M0 = wp.constant(wp.uint32(0xD2511F53))
_M1 = wp.constant(wp.uint32(0xCD9E8D57))
_W0 = wp.constant(wp.uint32(0x9E3779B9))
_W1 = wp.constant(wp.uint32(0xBB67AE85))


def seed_key(seed: SupportsIndex) -> wp.vec2ui:
    """Return the Philox key of a 64-bit `seed`, as (low word, high word)."""
    value = operator.index(seed)
    if not 0 <= value < 1 << 64:
        raise ValueError(f"the seed must be in [0, 2**64), got {value}")
    return wp.vec2ui(value & 0xFFFFFFFF, value >> 32)


@wp.func
def philox(counter: wp.vec4ui, key: wp.vec2ui) -> wp.vec4ui:
    """Return the 128 random bits of Philox-4x32-10 for `counter` and `key`."""
    c = counter
    k = key
    for _ in range(10):
        p0 = wp.uint64(_M0) * wp.uint64(c[0])
        p1 = wp.uint64(_M1) * wp.uint64(c[2])
        c = wp.vec4ui(
            wp.uint32(p1 >> wp.uint64(32)) ^ c[1] ^ k[0],
            wp.uint32(p1),
            wp.uint32(p0 >> wp.uint64(32)) ^ c[3] ^ k[1],
            wp.uint32(p0),
        )
        k = wp.vec2ui(k[0] + _W0, k[1] + _W1)
    return c


@wp.func
def random_bits(key: wp.vec2ui, batch: int, ray: int, depth: int, purpose: int) -> wp.vec4ui:
    """Return the 128 random bits of one draw, from its coordinates."""
    counter = wp.vec4ui(wp.uint32(ray), wp.uint32(depth), wp.uint32(purpose), wp.uint32(batch))
    return philox(counter, key)


@wp.func
def unit_float32(bits: wp.uint32) -> wp.float32:
    """Return a single-precision number uniform in (0, 1), from 23 of the 32 `bits`."""
    return (wp.float32(bits >> wp.uint32(9)) + wp.float32(0.5)) * wp.float32(1.1920928955078125e-07)


@wp.func
def unit_float64(high: wp.uint32, low: wp.uint32) -> wp.float64:
    """Return a double-precision number uniform in (0, 1), from 52 of the 64 bits."""
    a = wp.float64(high >> wp.uint32(6))  # 26 bits
    b = wp.float64(low >> wp.uint32(6))  # 26 bits
    return (a * wp.float64(67108864.0) + b + wp.float64(0.5)) * wp.float64(2.220446049250313e-16)
