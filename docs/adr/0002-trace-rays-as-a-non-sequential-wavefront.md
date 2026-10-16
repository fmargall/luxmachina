---
status: accepted
date: 2026-10-16
decision-makers: François Margall
---

# 0002. Trace rays as a non-sequential wavefront, one bounce at a time, without self-intersection epsilon

## Context

luxmachina is a differentiable light tracer written with NVIDIA Warp: rays are
emitted by the light sources, bounce on the surfaces of the scene (lenses,
blockers, diffusers) and are recorded by detectors. Gradients with respect to
the scene parameters (poses, shapes, materials) are obtained with `wp.Tape`.

The previous code base launched, from a Python loop, one kernel generating the
rays, then for every depth one kernel for the intersections, one for the camera
accumulation and one for the propagation. It worked, but showed several
limits:

- each intersection record embedded a full copy of the hit primitive, which
  costs memory and bandwidth, twice when gradients are kept;
- the closest-hit loop over the primitives had to be unrolled to be
  differentiated, which capped the scene size at `wp.config.max_unroll`;
- a single ray buffer overwritten in place produced wrong gradients, because the
  backward pass replayed earlier kernels with later data;
- the camera behind a Lambertian plate was hard-coded in its own kernel, which
  re-intersected the plate to check occlusion: adding a second detector meant
  writing a new kernel;
- every new ray started `5e-7` m away from its hit point along its direction, an
  arbitrary offset that depends on the units, shifts the optical path and may
  skip a close surface.

The library is meant to grow for years: spectral rendering, thin-film
coatings, polarization, diffraction at apertures, scattering surfaces. How
should rays be traced so that the forward pass is fast, the gradients are
correct, any number of detectors can be added, and no geometric approximation
is introduced at each bounce?

## Considered options

### Execution model

- **Megakernel**: one kernel, each thread follows its ray through every bounce
  in an inner loop. No launch overhead and the ray state stays in registers.
  But `wp.Tape` does not record the intermediate values of a dynamic loop, so
  correct gradients require unrolling `maxDepth × number of surfaces` copies of
  the code, with long compile times and high register pressure.
- **Wavefront with one kernel per stage** (the previous design): generate,
  intersect, accumulate, propagate. Simple and differentiable, but the
  intersection records are large, detectors are special kernels, and the many
  short launches leave the GPU waiting for Python.
- **Wavefront with one bounce per step, surfaces carrying their behavior,
  captured in a CUDA graph.** Each bounce reads a ray buffer and writes the
  next one; detectors are surfaces; the whole launch sequence is recorded once
  and replayed in a single submission.

### Self-intersection

- **Fixed epsilon offset** along the new direction: simple, but arbitrary,
  scale-dependent and wrong on the optical path.
- **Adaptive offset along the normal** (Wächter and Binder, 2019): offsets the
  origin by a few units in the last place, so it is minimal and scale-aware,
  but it is still an offset.
- **Skip the surface the ray leaves**: no offset, but wrong on curved
  surfaces, since a ray refracted into a sphere must hit that same sphere
  again on the far side.
- **Known root**: the surface the ray leaves is intersected with a dedicated
  routine that knows the origin lies on it and removes the root `t = 0`
  analytically.

## Decision

Chosen options: "wavefront with one bounce per step, surfaces carrying their
behavior, captured in a CUDA graph" and "known root", because together they
keep the gradients of `wp.Tape` correct without unrolling, remove launch
overhead, let detectors be added as data rather than code, and remove the only
systematic geometric approximation of the previous tracer.

We will follow these rules.

### Tracing

- Tracing is **non-sequential**: at every bounce, a ray is tested against every
  surface of the scene. A sequential mode, if ever needed, will reuse the same
  surfaces and behaviors.
- The host loops over the depth. Bounce `d` reads the ray buffer `d` and writes
  the ray buffer `d + 1`: **no buffer is overwritten during a trace**, so the
  tape can replay every kernel with its own inputs.
- Each bounce has two stages, kept as separate modules:
  - **Intersection**: a dynamic, non-differentiated loop finds the index of the
    closest surface; the intersection with that single surface is then
    recomputed in a differentiable way. The hit record is compact: surface
    index, distance, normal.
  - **Interaction**: the ray is first attenuated along the segment it has just
    crossed, according to the medium it travels in; then the behavior of the
    hit surface (refraction and reflection, absorption, diffusion, recording)
    produces the next ray.

  They start as two kernels; fusing them into one is an optimization to decide
  later by profiling, without changing the design.
- Ray data is stored as a **structure of arrays**, one Warp array per field, so
  that a field can be added without touching the kernels that do not use it.
- Rays are traced by **batches of fixed size**, each with its own seed, so that
  memory does not depend on the number of rays. Gradients are obtained by
  replaying each batch under `wp.Tape` from the adjoint of the output, and
  summing the batch gradients.
- The launch sequence of a batch, forward and backward, is **capturable in a
  CUDA graph** (`wp.ScopedCapture`): every value that changes between replays
  (seed, batch index, scene parameters) lives in a Warp array, launch
  dimensions are fixed (the last batch is padded with dead rays), and no host
  synchronization happens during a trace.

### Scene

- A surface is made of a **geometry** (how to intersect it, its normal), an
  **interface** (what happens to a ray that hits it) and references to the
  **materials** on each side.
- **Detectors are surfaces** whose interface records the ray into the
  detector's own buffer, then absorbs it or lets it pass. Occlusion between
  surfaces and detectors therefore comes from the intersection stage, and any
  number of detectors can be added without a new kernel.
- **Cameras are connected detectors**: at every hit on a scattering surface, the
  ray contributes to each camera through the surface's scattering function
  evaluated towards the camera, provided a visibility test finds no surface in
  between. A camera looking at a Lambertian plate is one such case. Scattering
  interfaces therefore provide both a way to sample an outgoing direction and
  a way to evaluate their scattering function for a given pair of directions.

### Geometric robustness

- **No offset is applied to a new ray.** Each ray stores the index of the
  surface it leaves. Every geometry provides a regular intersection and an
  intersection from a point of the surface, which excludes the root `t = 0`:
  - planes, disks and parallelograms: no other intersection;
  - quadrics (sphere, cylinder, cone, conic): the remaining root of the
    quadratic, in closed form (for a sphere of center `c` and a unit
    direction `d`, `t = −2 d·(o − c)`, kept only when positive);
  - surfaces solved iteratively (aspheres): a root search that excludes `t = 0`
    by construction, by restricting the bracket or deflating the known root.
- **A cemented interface is a single surface** with a material on each side,
  never two coincident surfaces.
- **Normals are analytic**, and the hit point is projected back onto the
  surface when its equation allows it.

### Consequences

#### Positive

- Gradients stay correct without unrolling, and the number of surfaces is no
  longer limited by `wp.config.max_unroll`.
- Adding a detector, a surface type or an interface behavior touches one
  module, not the trace loop.
- Spectral rendering, coatings, polarization and diffraction models can be
  added as new ray fields, materials and interfaces.
- Hit points and optical paths are free of any arbitrary offset, which matters
  for phase-sensitive extensions.

#### Neutral

- Each new geometry must provide two intersection routines instead of one.
- Graph capture shapes the host code: parameters go through arrays rather than
  Python scalars.

#### Negative

- Rays still go through global memory between stages, which a megakernel
  avoids; the data must stay compact for this to be cheap.
- A ray leaving a surface very close to an edge shared with another surface
  (for example where a lens cap meets its rim) can still self-intersect that
  neighboring surface. This rare case is accepted and documented rather than
  hidden by an offset everywhere.
- The known-root search for iterative surfaces is harder to write than an
  offset, especially for surfaces with inflection points.

### Confirmation

- Unit tests for every geometry: a ray leaving a plane never hits it again; a
  ray refracted into a sphere hits its far side at the analytic distance; a ray
  leaving a convex surface outwards does not hit it again.
- Gradient tests against finite differences on small scenes.
- A test, marked for GPU runners, traces a scene through a captured CUDA graph
  and compares the result with the uncaptured trace.
- Reviewers reject any constant offset added to a ray origin.

## More information

- Samuli Laine, Tero Karras and Timo Aila, "Megakernels Considered Harmful:
  Wavefront Path Tracing on GPUs", High-Performance Graphics (2013).
- Carsten Wächter and Nikolaus Binder, "A Fast and Robust Method for Avoiding
  Self-Intersection", Ray Tracing Gems, chapter 6 (2019).
- Warp documentation, differentiability and CUDA graph capture:
  <https://nvidia.github.io/warp/>
- The ray state (wavelength, optical path length, polarization, current medium)
  and the gradient estimators for discrete events (Fresnel choice, edges) will
  be decided in later records.
