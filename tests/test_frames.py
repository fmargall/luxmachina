# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Local frames of the surfaces (ADR-0004)."""

from typing import Any

import numpy as np
import pytest
import warp as wp

from luxmachina._precision import vec3
from luxmachina.frames import (
    _frame_type,
    make_frames,
    to_global_direction,
    to_global_point,
    to_local_direction,
    to_local_point,
)


@wp.kernel
def transform(
    vectors: wp.array(dtype=Any),
    frames: wp.array(dtype=Any),
    local_points: wp.array(dtype=Any),
    global_points: wp.array(dtype=Any),
    local_directions: wp.array(dtype=Any),
    global_directions: wp.array(dtype=Any),
):
    i = wp.tid()
    local_points[i] = to_local_point(vectors[i], frames[i])
    global_points[i] = to_global_point(vectors[i], frames[i])
    local_directions[i] = to_local_direction(vectors[i], frames[i])
    global_directions[i] = to_global_direction(vectors[i], frames[i])


def apply(vectors, frames, real, device):
    """Return the four transformations of `vectors`, one frame each."""
    count = len(vectors)
    inputs = wp.array(np.asarray(vectors), dtype=vec3(real), device=device)
    outputs = [wp.zeros(count, dtype=vec3(real), device=device) for _ in range(4)]
    wp.launch(transform, dim=count, inputs=[inputs, frames], outputs=outputs, device=device)
    return [output.numpy() for output in outputs]


def rotations(frames):
    return np.array([frame["rotation"] for frame in frames.numpy()])


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def test_default_frame_of_the_z_axis_is_the_global_frame(device, dtype):
    frames = make_frames([(1.0, 2.0, 3.0)], [(0.0, 0.0, 2.0)], real=dtype, device=device)

    np.testing.assert_array_equal(rotations(frames)[0], np.eye(3))
    np.testing.assert_array_equal(frames.numpy()[0]["origin"], (1.0, 2.0, 3.0))


@pytest.mark.parametrize("axes", ["random", "near -z", "in the xy plane"])
def test_default_frames_are_orthonormal_and_right_handed(axes):
    rng = np.random.default_rng(0)
    z = {
        "random": unit(rng.normal(size=(1000, 3))),
        "near -z": unit(np.column_stack([rng.normal(scale=1e-6, size=(100, 2)), -np.ones(100)])),
        "in the xy plane": unit(np.column_stack([rng.normal(size=(100, 2)), np.zeros(100)])),
    }[axes]
    matrices = rotations(make_frames(np.zeros_like(z), z, real=wp.float64))

    np.testing.assert_allclose(
        matrices @ matrices.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), matrices.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.det(matrices), 1.0, rtol=1e-12)
    np.testing.assert_allclose(matrices[:, :, 2], z, atol=1e-12)


def test_given_x_axes_are_projected_and_normalized():
    frames = make_frames([(0.0, 0.0, 0.0)], [(0.0, 0.0, 1.0)], [(2.0, 2.0, 5.0)], real=wp.float64)
    matrix = rotations(frames)[0]

    np.testing.assert_allclose(matrix[:, 0], unit((1.0, 1.0, 0.0)), atol=1e-15)
    np.testing.assert_allclose(matrix[:, 1], unit((-1.0, 1.0, 0.0)), atol=1e-15)


def test_transformations_of_a_known_frame(device, dtype):
    # Optical axis along global x, local x along global y, vertex at (1, 2, 3)
    frames = make_frames([(1.0, 2.0, 3.0)], [(1.0, 0.0, 0.0)], [(0.0, 1.0, 0.0)], dtype, device)
    local_point, global_point, local_direction, global_direction = apply(
        [(1.0, 0.0, 0.0)], frames, dtype, device
    )

    # Local axes in global coordinates: x = (0, 1, 0), y = cross(z, x) = (0, 0, 1), z = (1, 0, 0)
    np.testing.assert_allclose(local_point, [(-2.0, -3.0, 0.0)])
    np.testing.assert_allclose(global_point, [(1.0, 3.0, 3.0)])
    np.testing.assert_allclose(local_direction, [(0.0, 0.0, 1.0)])
    np.testing.assert_allclose(global_direction, [(0.0, 1.0, 0.0)])


def test_local_and_global_transformations_are_inverse(device, dtype):
    rng = np.random.default_rng(1)
    frames = make_frames(
        rng.normal(size=(100, 3)), rng.normal(size=(100, 3)), real=dtype, device=device
    )
    points = rng.normal(size=(100, 3))
    local_point, _, local_direction, _ = apply(points, frames, dtype, device)
    _, back_point, _, _ = apply(local_point, frames, dtype, device)
    _, _, _, back_from_direction = apply(local_direction, frames, dtype, device)

    tolerance = 1e-5 if dtype == wp.float32 else 1e-12
    np.testing.assert_allclose(back_point, points, atol=tolerance)
    np.testing.assert_allclose(back_from_direction, points, atol=tolerance)
    # Directions keep their length
    np.testing.assert_allclose(
        np.linalg.norm(local_direction, axis=1), np.linalg.norm(points, axis=1), rtol=tolerance
    )


def test_vertex_and_axis_are_the_local_origin_and_z(device, dtype):
    vertex, axis = (0.3, -0.2, 1.5), unit((0.2, -0.4, 1.0))
    frames = make_frames([vertex], [axis], real=dtype, device=device)
    local_vertex, _, _, _ = apply([vertex], frames, dtype, device)
    _, _, local_axis, _ = apply([axis], frames, dtype, device)

    tolerance = 1e-6 if dtype == wp.float32 else 1e-15
    np.testing.assert_allclose(local_vertex, [(0.0, 0.0, 0.0)], atol=tolerance)
    np.testing.assert_allclose(local_axis, [(0.0, 0.0, 1.0)], atol=tolerance)


@wp.kernel
def local_points_only(
    points: wp.array(dtype=Any), frames: wp.array(dtype=Any), local: wp.array(dtype=Any)
):
    i = wp.tid()
    local[i] = to_local_point(points[i], frames[i])


def test_gradient_with_respect_to_the_origin(device):
    # q = R^T (p - origin): dq/dorigin = -R^T, so the gradient of sum(q) is -R (1, 1, 1)
    frames = make_frames(
        [(0.0, 0.0, 0.0)], [(1.0, 2.0, 2.0)], real=wp.float64, device=device, requires_grad=True
    )
    points = wp.array([(0.5, 0.1, -0.3)], dtype=wp.vec3d, device=device, requires_grad=True)
    local = wp.zeros(1, dtype=wp.vec3d, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(local_points_only, dim=1, inputs=[points, frames], outputs=[local], device=device)
    tape.backward(grads={local: wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3d, device=device)})

    matrix = rotations(frames)[0]
    assert frames.grad is not None and points.grad is not None
    np.testing.assert_allclose(frames.grad.numpy()[0]["origin"], -matrix @ np.ones(3), atol=1e-15)
    np.testing.assert_allclose(points.grad.numpy()[0], matrix @ np.ones(3), atol=1e-15)


def test_struct_is_created_once_per_precision():
    assert _frame_type(wp.float32) is _frame_type(wp.float32)
    assert _frame_type(wp.float32) is not _frame_type(wp.float64)


def test_make_frames_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="2 origins for 1 axes"):
        make_frames([(0.0, 0.0, 0.0)] * 2, [(0.0, 0.0, 1.0)])
    with pytest.raises(ValueError, match="2 x axes for 1 axes"):
        make_frames([(0.0, 0.0, 0.0)], [(0.0, 0.0, 1.0)], [(1.0, 0.0, 0.0)] * 2)


def test_make_frames_rejects_zero_axes():
    with pytest.raises(ValueError, match="cannot be zero"):
        make_frames([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])


def test_make_frames_rejects_x_axes_parallel_to_the_axis():
    with pytest.raises(ValueError, match="must not be parallel"):
        make_frames([(0.0, 0.0, 0.0)], [(0.0, 0.0, 1.0)], [(0.0, 0.0, -3.0)])
