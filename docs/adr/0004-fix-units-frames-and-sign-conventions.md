---
status: accepted
date: 2026-10-17
decision-makers: François Margall
---

# 0004. Express lengths in meters and describe surfaces in local frames with signed curvatures

## Context

The shapes of [ADR-0002](0002-trace-rays-as-a-non-sequential-wavefront.md) are
described in the global frame: a plane by a point and a normal, a sphere by its
center and its radius. That is enough to intersect them, but not to describe
an optical system:

- an aperture (a disk, an annulus, a rectangle) is a domain *on* a surface,
  which needs coordinates attached to the surface;
- a lens surface is a part of a sphere, and the two points of a sphere at the
  same distance from its axis must be told apart;
- aspheres are only defined by their sag `z(r)` along a local axis;
- an interface needs to know which side of the surface is which, to pick the
  material on each side;
- [ADR-0003](0003-define-the-ray-state.md) left the unit of length to a later
  record.

Optical design software describes a surface by its vertex, its local axis and
its curvature, with a sign convention. Whatever we choose is hard to change
once scenes, files and users depend on it. Which unit of length, which frames
and which sign conventions do we use?

## Considered options

### Unit of length

- **Meters**: the SI unit, consistent with the radiometric quantities (watts,
  square meters) and with the previous code base.
- **Millimeters**: the habit of optical design software, but it mixes with SI
  radiometry and makes wavelengths read as `5.5e-4`.
- **No fixed unit**, left to the user: flexible, but every constant that
  carries a unit (wavelengths, dispersion formulas, absorption coefficients)
  then becomes ambiguous.

### Parameter of a curved surface

- **Radius of curvature `R`**: what users read on drawings, but a flat surface
  is `R = ∞`, so an optimization cannot go through flat.
- **Curvature `c = 1 / R`**: a flat surface is `c = 0`, the parameter is
  continuous and differentiable through flat, and it is the parameter of the
  sag of aspheres.

## Decision

Chosen options: "meters" and "curvature", because SI units keep every physical
constant unambiguous, and the curvature is continuous through flat surfaces,
which a differentiable tracer needs.

We will follow these rules.

### Units

- **Lengths are in meters**, including wavelengths and optical path lengths.
- Angles are in radians, powers in watts.
- Conversions from other units (millimeters, nanometers, degrees) happen only
  at the boundary of the library, in the functions that read user input,
  never inside the tracer.

### Frames

- The **global frame** is right-handed and has no preferred axis. Models of
  instruments (cameras) document their own frame conventions.
- Each surface has a **local frame**, right-handed and orthonormal:
  - its origin is the **vertex** of the surface;
  - its `z` axis is the **local optical axis**;
  - its `x` and `y` axes span the plane tangent to the surface at the vertex.
- **Apertures** are expressed in the local frame, on the `x` and `y`
  coordinates of a point.
- A local frame is stored as a rotation matrix and a translation, from local
  to global coordinates.

### Sign conventions

- A curved surface is described by its **curvature** `c = 1 / R`; `c = 0` is
  flat. Users may give a radius `R`, converted at the boundary.
- **`c > 0` when the center of curvature is on the `+z` side** of the vertex.
  The sag `z(r)` of a surface then has the sign of `c`: it bends towards `+z`
  when `c > 0`.
- A spherical surface of curvature `c ≠ 0` and vertex `v` has its center at
  `v + z / c` and its radius `1 / |c|`, so it reuses the sphere of the shapes.
- The **orientation of a surface** is the side its local `+z` axis points to.
  The normal used by interfaces points to that side: at the vertex, it is `+z`,
  whatever the sign of the curvature. For a sphere it is `c (center − p)` at a
  point `p`, the outward normal of the shape when `c < 0` and the inward one
  when `c > 0`.
- Each surface has a material on its `−z` side and one on its `+z` side. A ray
  crosses from one to the other according to the sign of `d · n`.

### Consequences

#### Positive

- Physical constants and material data have a single, unambiguous unit.
- Apertures, caps and aspheres share one description, the one of optical
  design software, so systems can be transcribed from drawings and catalogs.
- A curvature can be optimized through flat, with continuous gradients.
- The shapes of ADR-0002 are reused as they are: the local description is a
  layer above them.

#### Neutral

- Users who think in millimeters convert at input; the documentation and the
  examples must show it.
- The global frame imposes no optical axis: a system is placed by the local
  frames of its surfaces.

#### Negative

- A local frame adds a rotation and a translation to every surface, and each
  aperture test first transforms the hit point into local coordinates.
- The orientation of a sphere's normal depends on the sign of its curvature,
  a frequent source of sign errors that only tests can catch.

### Confirmation

- Tests convert a vertex and a curvature into a center and a radius for both
  signs of the curvature, and check that the oriented normal at the vertex is
  `+z` in both cases.
- Tests check that the sag of a surface has the sign of its curvature.
- Docstrings of public functions state the unit of every physical quantity.
- Reviewers reject unit conversions inside kernels.

## More information

- [ADR-0002](0002-trace-rays-as-a-non-sequential-wavefront.md): shapes and
  known-root intersections.
- [ADR-0003](0003-define-the-ray-state.md): ray state, whose lengths are in
  meters from now on.
- Warren J. Smith, "Modern Optical Engineering", 4th edition, McGraw-Hill
  (2008), for the sign conventions of optical design.
