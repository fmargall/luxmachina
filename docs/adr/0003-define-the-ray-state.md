---
status: proposed
date: 2026-10-16
decision-makers: François Margall
---

# 0003. Define the ray state for spectral, phase-aware and path-aware tracing, with a configurable precision

## Context

[ADR-0002](0002-trace-rays-as-a-non-sequential-wavefront.md) stores the rays as
a structure of arrays, one buffer per depth. It leaves open what a ray carries.
The previous code base used a struct with an origin, a direction, a depth, an
energy and an alive flag. That was enough for a monochromatic, unpolarized
beam on a Lambertian plate, but not for what luxmachina should grow into:

- spectral rendering, with dispersive materials and detectors that have a
  spectral response;
- absorption in the bulk of materials (Beer-Lambert), which needs to know the
  medium a ray travels in;
- thin-film coatings, whose reflectance depends on polarization;
- wavefronts, point spread functions and diffraction, which need the phase of
  each ray, that is its optical path length;
- stray-light analysis, which needs to know which sequence of surfaces a ray
  went through (for example to isolate second-order ghosts).

Fields that every kernel reads are costly to add once the code exists. The
precision of the computation is also at stake: in single precision, a
coordinate carries a relative error of about `6e-8`, so 60 nm on a 1 m scene,
a sizeable fraction of a wavelength. The phase is then meaningless far from the
origin. Double precision fixes that, but runs up to 64 times slower on consumer
GPUs. Accumulating only the optical path in double precision does not help,
since each distance it adds is computed from single-precision positions.

What does a ray carry, in which precision, and how are its random numbers drawn?

## Considered options

### Spectral sampling

- **RGB triplets**: cheap, but neither dispersion nor a spectral detector
  response can be modeled correctly.
- **One wavelength per ray**, sampled from the spectrum of its source: exact
  in the Monte Carlo sense, and a dispersive material simply splits rays of
  different wavelengths.
- **Hero wavelength** (several wavelengths per ray): less spectral noise, but
  the secondary wavelengths must be dropped at the first dispersive interface,
  which is most interfaces in an optical system.

### Precision

- **Single precision only**: simple and fast, but the phase is only reliable
  for compact scenes.
- **Double precision only**: accurate, but far too slow on consumer GPUs for
  everyday use.
- **Configurable precision**: kernels are generated for either precision, single
  by default. More code paths to test, but each use gets the precision it
  needs.

### Polarization

- **Unpolarized only**, Fresnel coefficients averaged over s and p: what the
  previous code did. Wrong as soon as light is partially polarized by a first
  reflection and then meets a coating.
- **Stokes vector and Mueller matrices**: suits incoherent light, four values
  per ray.
- **Polarization ray tracing matrices** (Chipman): the reference in optical
  design, heavier.

### Random numbers

- **A random generator state stored per ray**: one more field, and the draws
  depend on the order in which kernels consume them.
- **Counter-based draws**: each random number is a pure function of its
  coordinates, nothing is stored.

## Decision

Chosen options: "one wavelength per ray", "configurable precision", "counter-based
draws", and for polarization an interface contract that keeps both Stokes and
polarization ray tracing matrices open, without choosing between them yet. They
give every planned extension a place in the ray state from the first version,
for the cost of a few scalars per ray.

We will follow these rules.

### Fields

Each ray carries the following fields, each stored as its own Warp array:

| Field | Type | Meaning |
| --- | --- | --- |
| `origin` | vector of 3 reals | start point of the current segment |
| `direction` | vector of 3 reals | unit direction of the current segment |
| `power` | real | power carried by the ray; the source power divided by the number of rays at emission |
| `wavelength` | real | vacuum wavelength, sampled from the spectrum of the source |
| `opl` | real | optical path length from the source, the sum of `n · t` over the segments |
| `medium` | integer | index of the material the ray travels in |
| `surface` | integer | index of the surface the ray leaves, `-1` at the source (see ADR-0002) |
| `status` | integer | alive, absorbed, escaped from the scene or recorded |

- **The depth is not stored**: it is the index of the buffer the ray lives in.
- Lengths, including the wavelength and the optical path, are expressed in the
  length unit of the scene, to be fixed by a later record.
- A field is added to this table by a new record that supersedes this one.

### Precision

- **The precision is a parameter of a trace**, single by default. All the reals
  of a trace (rays, surfaces, materials, detectors) use that precision; kernels
  never mix the two.
- Kernels are written once, generic over the real type, and specialized for
  each precision.
- Gradient tests against finite differences run in double precision, where
  finite differences are meaningful.

### Polarization

- The first version traces unpolarized light, and `power` is a scalar.
- Every interface already returns its reflection and transmission
  coefficients **separately for s and p**, so that a polarization state can
  later replace the scalar `power` without rewriting the interfaces.

### Media

- When a ray crosses an interface, its `medium` becomes the material on the
  other side of the interface.
- If the medium of a ray differs from the material on the side it comes from,
  the scene is inconsistent (for example an overlap of two solids). These
  events are counted and reported, never silently ignored.
- Materials are described by a complex refractive index `n(λ) + i·k(λ)`: `n`
  drives refraction and Fresnel coefficients, `k` the absorption in the bulk
  (`α = 4πk/λ`) and in coatings. The bulk attenuation is applied
  deterministically to `power`, never by sampling an absorption distance.

### Path history

- The sequence of surfaces hit by a ray is available within a batch from the
  `surface` field of each depth buffer; it is not stored in the ray.
- Detectors may filter what they record on that history, for example by number
  of reflections or by sequence of surfaces (ghost and stray-light analysis).

### Random numbers

- Each random number is a pure function of `(seed, batch, ray, depth, purpose)`,
  where `purpose` distinguishes the draws made at the same bounce (emission
  position, emission direction, wavelength, Fresnel choice, scattering).
- Two traces with the same seed therefore draw the same numbers (common random
  numbers): the loss is a deterministic and nearly smooth function of the
  scene parameters, and the forward and backward passes of a batch see the
  same rays.

### Consequences

#### Positive

- Spectral rendering, bulk absorption, coatings, phase and stray-light
  analysis can be added without changing the layout of the rays.
- Single precision keeps everyday tracing fast; double precision is available
  when the phase matters or to check gradients.
- Results are reproducible from a seed, regardless of the order of the kernels.

#### Neutral

- A ray carries a few more scalars than before; with a structure of arrays,
  kernels that do not use a field do not read it.
- The unit of length is left to a later record.

#### Negative

- Every kernel must be generic over the real type, and the test suite must
  cover both precisions, which roughly doubles the compilation time of the
  tests.
- Double precision remains very slow on consumer GPUs; it is a tool for
  accuracy, not the default.
- Unpolarized tracing is wrong for partially polarized light until a
  polarization state is added.

### Confirmation

- Tests run the same scene in both precisions and check that the results
  agree within the tolerance of single precision.
- A test traces the same scene twice with the same seed and checks that the
  outputs are identical.
- Tests check that interfaces return distinct s and p coefficients at oblique
  incidence.
- A test with overlapping solids checks that the medium inconsistencies are
  reported.

## More information

- [ADR-0002](0002-trace-rays-as-a-non-sequential-wavefront.md): tracing
  pipeline, known-root intersection and surfaces.
- Alexander Wilkie et al., "Hero Wavelength Spectral Sampling", Computer
  Graphics Forum (2014).
- Russell A. Chipman, Wai Sze Tiffany Lam and Garam Young, "Polarized Light and
  Optical Systems", CRC Press (2018).
- John K. Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3",
  SC'11 (2011), for counter-based random number generators.
