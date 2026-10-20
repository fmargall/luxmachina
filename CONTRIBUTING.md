# Contributing to Lux Machina

Thank you for your interest in Lux Machina! Everyone taking part in the project is
expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting Bugs and Requesting Features

Open an [issue](https://github.com/fmargall/luxmachina/issues), with a minimal example reproducing the bug when
there is one. Security vulnerabilities must not be reported in public issues: see the
[Security Policy](SECURITY.md) instead.

## Setting Up a Development Environment

The project is managed with [uv](https://docs.astral.sh/uv/). Clone the repository, then
create the development environment and install the pre-commit hooks:

```bash
git clone https://github.com/fmargall/luxmachina.git
cd luxmachina
uv sync
uv run pre-commit install
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

`uv sync` creates a `.venv` with the package installed in editable mode and the `dev`
dependency group, at the exact versions of `uv.lock`. The last command makes
`git blame` skip the formatting-only commits listed in `.git-blame-ignore-revs`.

Without uv, the same environment can be created with pip, but without the versions of
`uv.lock`. Dependency groups require pip 25.1 or later, hence the upgrade of pip first:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip  # .venv\Scripts\python on Windows
.venv/bin/python -m pip install -e . --group dev
```

The pre-commit hook of pyright runs it through `uv run`: without uv, skip it with
`SKIP=pyright git commit`, and run `pyright` from the environment instead.

## Running the Checks

```bash
uv run pytest --cov                # tests, with coverage
uv run pyright                     # type checking
uv run pre-commit run --all-files  # linting, formatting and all the other hooks
```

The pre-commit hooks also run on every commit, on the staged files.

## Managing Dependencies

Add a dependency with `uv add <package>`, or a development tool with
`uv add --group dev <tool>`. Both commands update `pyproject.toml` and `uv.lock`: commit
the two files together.

Upgrade the locked versions with `uv lock --upgrade` (or
`uv lock --upgrade-package <package>` for a single one).

## Submitting Changes

1. Create a branch from `main`.
2. Make your changes, with tests for any new behavior.
3. Describe user-visible changes in [`CHANGELOG.md`](CHANGELOG.md), under `[Unreleased]`.
4. Record a significant decision as an ADR in [`docs/adr/`](docs/adr/README.md).
5. Check that all the checks pass, then open a pull request, and fill in its checklist.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/),
prefixed with the matching [gitmoji](https://gitmoji.dev/), for example:

```text
✨ feat: add a command-line interface
🐛 fix: handle empty input files
📝 docs: document the installation on Windows
```

## Releasing

The version comes from the git tag (`vX.Y.Z`, following
[Semantic Versioning](https://semver.org/)): pushing a tag runs the release workflow,
which publishes the package to TestPyPI then PyPI, and creates the GitHub release.

1. Prepare the release on a branch: `[Unreleased]` in `CHANGELOG.md` becomes the new version, and `CITATION.cff` gets its version and date.

```bash
git switch -c release/vX.Y.Z
uv run scripts/release.py prepare X.Y.Z
git commit -am "🔖 chore(release): prepare vX.Y.Z"
```

2. Open a pull request, and merge it once the checks pass.
3. Tag the merge commit on `main`, then push the tag:

```bash
git switch main
git pull
git tag -s vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

The release workflow first runs `scripts/release.py check`: a forgotten step fails it
before anything is published. The changelog section of the version becomes the notes of
the GitHub release.

## License

By contributing, you agree that your contributions will be licensed under the
[AGPL-3.0-or-later](LICENSE.txt) license of the project.
