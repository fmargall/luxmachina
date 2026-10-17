<!--
The title of the pull request follows gitmoji and Conventional Commits, for
example "✨ feat(shapes): add spheres". It becomes the commit message when the
pull request is squashed, and it feeds the release notes.
-->

## Summary

<!-- What does this change, and why? Link the issue or the ADR it implements. -->

## Checklist

- [ ] Tests cover the change, on every device and in both precisions where it applies.
- [ ] Hand-written gradients, if any, are checked against finite differences.
- [ ] Docstrings state the unit of every physical quantity (ADR-0004).
- [ ] `CHANGELOG.md` is updated, or the change is not visible to users.
- [ ] A decision that deserves a record has one in `docs/adr/`.
- [ ] Every ADR changed or added here has its final status, in its file and in the index.

<!--
Merge method: "Rebase and merge" when the commits are atomic and their
messages final; "Squash and merge" otherwise, after checking the title above.
-->
