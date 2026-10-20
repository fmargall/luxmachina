---
status: accepted
date: 2026-10-20
decision-makers: "François Margall"
---

# 0001. Record architecture decisions in the repository with a MADR-based template

## Context

Several structuring decisions are built into Lux Machina from the start
(pull request only workflow, actions pinned by commit SHA, a single aggregate CI
check, a minimum Python version following SPEC 0), and many more will follow
(docstring convention, documentation tooling, dependencies). Without a record,
their rationale lives only in commit messages, pull request descriptions and
memory.

Without a written record, a future contributor (or the author, months later)
cannot tell whether a choice was deliberate or accidental, and risks either
reverting a sound decision or keeping an outdated one.

Whatever we choose must keep the rationale next to the code and versioned with
it, be cheap enough that it is actually done, be readable on GitHub without any
tooling, and keep superseded decisions visible instead of silently rewriting
them. How should we keep track of these decisions and of why they were taken?

## Considered options

- **ADRs with a template derived from MADR, stored in `docs/adr/`.**
- **ADRs with the original Nygard template** (context, decision, status,
  consequences): minimal, but rejected alternatives are not part of the
  structure and are often left out.
- **A wiki page or GitHub Discussions**: easy to edit, but not versioned with the
  code, not reviewed, and lost if the repository moves away from GitHub.
- **No dedicated record** (commit messages and pull requests only): costs
  nothing, but the rationale stays scattered and hard to find, which is the
  problem we want to solve.

## Decision

Chosen option: "ADRs with a template derived from MADR, stored in `docs/adr/`",
because it keeps decisions versioned with the code and reviewed in pull requests
like any other change. Its list of considered options records the alternatives
we rejected, not only the one we picked, while the Nygard-style consequences
(positive, neutral, negative) keep the trade-off explicit.

We will follow these rules:

- One decision per file, named `NNNN-title-in-kebab-case.md`, numbered
  sequentially from `0001` and never renumbered.
- New records start from [`template.md`](template.md); sections marked optional
  may be removed.
- Each record is added through its own pull request, usually together with the
  change it decides, and listed in the [index](README.md).
- Once accepted, a record is not rewritten. Changing a decision means writing a
  new record and setting the old one's status to `superseded by ADR-NNNN`. Fixing
  typos or links is allowed.
- A decision deserves a record when it is hard to reverse, affects the whole
  project, or would make a newcomer ask "why is it done this way?".

### Consequences

#### Positive

- The reasoning behind the project survives in the repository and can be linked
  from code, issues and pull requests.
- Each decision goes through review before being accepted.

#### Neutral

- The template is slightly longer than Nygard's; its optional sections can be
  dropped for small decisions.

#### Negative

- Writing a record takes time; the threshold above keeps this for decisions that
  matter.
- Records can drift from reality if a decision is changed without a new record.

### Confirmation

Reviewers ask for a new or superseding record when a pull request changes a
decision covered by the criteria above.

## More information

- Michael Nygard, [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) (2011).
- [MADR](https://adr.github.io/madr/), the Markdown Architectural Decision Records.
