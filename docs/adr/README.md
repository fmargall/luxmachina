# Architecture decision records

This folder records the significant decisions taken for luxmachina, the context
in which they were taken and the options that were rejected. The process is
described in [ADR-0001](0001-record-architecture-decisions.md).

To propose a new decision, copy [`template.md`](template.md) to the next free
number, fill it in, add it to the table below and open a pull request.

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions in the repository with a MADR-based template | Accepted |
| [0002](0002-trace-rays-as-a-non-sequential-wavefront.md) | Trace rays as a non-sequential wavefront, one bounce at a time, without self-intersection epsilon | Accepted |
| [0003](0003-define-the-ray-state.md) | Define the ray state for spectral, phase-aware and path-aware tracing, with a configurable precision | Accepted |
