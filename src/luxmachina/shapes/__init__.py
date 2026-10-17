# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 François Margall

"""Shapes: the equations of the surfaces of a scene (ADR-0002).

A shape only describes where a surface lies. Its aperture (the part of the
shape that is kept), its interface and its materials are described apart.

Every shape provides three Warp functions, generic over the real type:

- its intersection with a ray ``o + t d``, where ``d`` has unit length;
- its intersection from one of its points, for the surface a ray leaves;
- its unit normal at one of its points.

The intersections return the smallest distance ``t > 0``, or ``+inf`` when the
ray misses the shape. The intersection from a point of the shape removes the
root ``t = 0`` analytically, so that no offset is ever applied to the origin
of a ray. The normal defines the two sides of the surface, to which the
interface and the materials refer.

The shapes of a scene are stored as one array per kind of shape, each of its
own Warp struct, created once per precision.
"""
