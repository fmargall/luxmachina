# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

# Warp device code: the type stubs of Warp do not describe operations on its
# scalars and vectors (indexing, shifts, constructors), Warp checks them itself
# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportArgumentType=false

"""Counter-based random numbers (ADR-0003)."""

import numpy as np
import pytest
import warp as wp

from luxmachina._random import philox, random_bits, seed_key, unit_float32, unit_float64

# Known-answer tests of Philox-4x32-10, from the kat_vectors file of Random123
KNOWN_ANSWERS = [
    # counter, key, expected output
    ([0, 0, 0, 0], [0, 0], [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
    ([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2, [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]),
    (
        [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
        [0xA4093822, 0x299F31D0],
        [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1],
    ),
]


@wp.kernel
def apply_philox(
    counters: wp.array(dtype=wp.vec4ui),
    keys: wp.array(dtype=wp.vec2ui),
    bits: wp.array(dtype=wp.vec4ui),
):
    i = wp.tid()
    bits[i] = philox(counters[i], keys[i])


@wp.kernel
def draw_grid(
    key: wp.vec2ui, batch: int, depths: int, purposes: int, bits: wp.array(dtype=wp.vec4ui)
):
    # One draw per (ray, depth, purpose), flattened in this order
    i = wp.tid()
    ray = i // (depths * purposes)
    depth = (i // purposes) % depths
    purpose = i % purposes
    bits[i] = random_bits(key, batch, ray, depth, purpose)


@wp.kernel
def to_unit(
    bits: wp.array(dtype=wp.vec4ui),
    singles: wp.array(dtype=wp.float32),
    doubles: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    singles[i] = unit_float32(bits[i][0])
    doubles[i] = unit_float64(bits[i][0], bits[i][1])


def draw(device, count, seed=0, batch=0, depths=1, purposes=1):
    bits = wp.zeros(count, dtype=wp.vec4ui, device=device)
    wp.launch(
        draw_grid,
        dim=count,
        inputs=[seed_key(seed), batch, depths, purposes],
        outputs=[bits],
        device=device,
    )
    return bits


def unit(bits):
    singles = wp.zeros(bits.shape[0], dtype=wp.float32, device=bits.device)
    doubles = wp.zeros(bits.shape[0], dtype=wp.float64, device=bits.device)
    wp.launch(
        to_unit, dim=bits.shape[0], inputs=[bits], outputs=[singles, doubles], device=bits.device
    )
    return singles.numpy(), doubles.numpy()


def test_philox_matches_the_known_answers(device):
    counters, keys, expected = (
        np.array(column, dtype=np.uint32) for column in zip(*KNOWN_ANSWERS, strict=True)
    )
    bits = wp.zeros(len(KNOWN_ANSWERS), dtype=wp.vec4ui, device=device)
    wp.launch(
        apply_philox,
        dim=len(KNOWN_ANSWERS),
        inputs=[
            wp.array(counters, dtype=wp.vec4ui, device=device),
            wp.array(keys, dtype=wp.vec2ui, device=device),
        ],
        outputs=[bits],
        device=device,
    )
    np.testing.assert_array_equal(bits.numpy(), expected)


@pytest.mark.gpu
def test_cpu_and_cuda_draw_the_same_bits():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    np.testing.assert_array_equal(
        draw("cpu", 4096, seed=7, depths=4, purposes=4).numpy(),
        draw("cuda:0", 4096, seed=7, depths=4, purposes=4).numpy(),
    )


def test_every_coordinate_changes_the_bits(device):
    reference = draw(device, 1).numpy()[0]
    for changed in ({"seed": 1}, {"batch": 1}):
        assert not np.array_equal(draw(device, 1, **changed).numpy()[0], reference)
    # Ray, depth and purpose: all the draws of a grid differ from each other
    bits = draw(device, 4096 * 4 * 4, depths=4, purposes=4).numpy()
    assert len(np.unique(bits, axis=0)) == len(bits)


def test_same_coordinates_give_the_same_bits(device):
    np.testing.assert_array_equal(
        draw(device, 256, seed=3).numpy(), draw(device, 256, seed=3).numpy()
    )


def test_unit_numbers_are_uniform(device):
    singles, doubles = unit(draw(device, 1 << 20, seed=11))
    for values in (singles, doubles):
        # Mean 1/2 and variance 1/12, within about 7 standard errors
        assert abs(values.mean() - 0.5) < 2e-3
        assert abs(values.var() - 1.0 / 12.0) < 1e-3
        # Chi-squared over 64 bins: 63 degrees of freedom, mean 63, standard deviation 11
        counts = np.histogram(values, bins=64, range=(0.0, 1.0))[0]
        expected = len(values) / 64
        assert ((counts - expected) ** 2 / expected).sum() < 63 + 6 * 11


def test_unit_numbers_use_all_their_bits(device):
    bits = draw(device, 1024, seed=5).numpy()
    singles, doubles = unit(wp.array(bits, dtype=wp.vec4ui, device=device))
    high, low = bits[:, 0].astype(np.int64), bits[:, 1].astype(np.int64)
    np.testing.assert_array_equal(singles, ((high >> 9) + 0.5) / 2.0**23)
    np.testing.assert_array_equal(doubles, (((high >> 6) << 26) + (low >> 6) + 0.5) / 2.0**52)


def test_unit_numbers_exclude_zero_and_one(device):
    extremes = wp.array([[0, 0, 0, 0], [0xFFFFFFFF] * 4], dtype=wp.vec4ui, device=device)
    singles, doubles = unit(extremes)
    np.testing.assert_array_equal(singles, [2.0**-24, 1.0 - 2.0**-24])
    np.testing.assert_array_equal(doubles, [2.0**-53, 1.0 - 2.0**-53])


@pytest.mark.parametrize(
    ("seed", "words"),
    [
        (0, (0, 0)),
        (0x0123456789ABCDEF, (0x89ABCDEF, 0x01234567)),
        ((1 << 64) - 1, (0xFFFFFFFF,) * 2),
    ],
)
def test_seed_key_splits_the_seed(seed, words):
    assert tuple(seed_key(seed)) == words


def test_seed_key_accepts_numpy_integers():
    assert tuple(seed_key(np.uint64(5))) == (5, 0)


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_seed_key_rejects_out_of_range_seeds(seed):
    with pytest.raises(ValueError, match="seed must be in"):
        seed_key(seed)


def test_seed_key_rejects_non_integers():
    with pytest.raises(TypeError):
        seed_key(1.5)
