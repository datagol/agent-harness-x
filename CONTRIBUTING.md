# Contributing to HarnessX

Thanks for helping. This page covers the setup, the checks a change must pass,
and how changes reach a release.

## Set up

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/datagol/agent-harness-x.git
cd agent-harness-x
uv sync --all-extras        # SDK in editable mode, every integration, dev tools
uv run pytest -q            # runs offline; service-backed tests skip
```

Copy `.env.example` values you need into a root `.env` if you want to run the
live examples. The suite itself never contacts a model or a service.

## Make a change

1. Branch from `main`.
2. Keep the change focused. A bug fix and an unrelated cleanup are two pull
   requests.
3. Add or update tests. Provider changes get a scripted-provider test, runtime
   changes get a durable-runtime test, and anything a user calls gets covered
   by `tests/test_sdk_contracts.py` or a typing fixture under `tests/typing/`.
4. Run the checks below.
5. Add a line under **Unreleased** in `CHANGELOG.md` when the change is visible
   to users.

## Checks

```bash
uv run ruff check .          # lint
uv run mypy                  # type-check the SDK, tests, and examples
uv run pytest -q             # unit tests
```

The integration job in CI runs `tests/test_postgres_runtime.py` and
`tests/test_temporal_runtime.py` against real PostgreSQL, Redis, and Temporal.
Locally they skip unless you export `HARNESS_TEST_POSTGRES_DSN`,
`HARNESS_TEST_REDIS`, and `HARNESS_TEST_TEMPORAL`. Docker plus the
`temporal` CLI is enough to run them; see the job in
`.github/workflows/tests.yml` for the exact services.

CI also builds a wheel and installs it into a fresh interpreter for every
extras combination, so keep optional dependencies imported lazily. A new
provider or backend must not be imported by `harnessx/__init__.py` at
module load unless its dependency is in the core set.

## Pull requests

- Open the pull request against `main`. The **Runtime checks** workflow must
  pass; a maintainer reviews after that.
- Describe what changed and why, and how you verified it. The template asks
  for this.
- Squash or rebase as you like; the merge is what gets tagged.
- Pull requests from forks run the same checks. They cannot publish anything,
  because publishing happens only from a `v*` tag on this repository.

## Style

- `ruff` enforces the baseline; there is no separate formatter step.
- Prefer small functions with explicit types. `mypy` runs with
  `check_untyped_defs`.
- Docstrings say what a thing guarantees, not how it is implemented.
- Examples must run offline by default or say clearly which key they need.

## Releases

Maintainers release by tagging `main`:

```bash
git tag v0.4.0 && git push origin v0.4.0
```

The publish workflow re-runs the checks, verifies the tag matches
`pyproject.toml`, and uploads to PyPI through Trusted Publishing. Nothing
reaches PyPI from a pull request or from a push to `main`.

## Where things live

- `harnessx/` is the package. `examples/` and `harness-web/` ship in the
  repository, not in the wheel.
- The documentation site is a separate repository; the `README.md` here is
  what PyPI renders.
- Design notes are not kept in this repository.

## Reporting problems

Bugs and feature requests go to
[issues](https://github.com/datagol/agent-harness-x/issues) using the
templates. Security problems go through the process in `SECURITY.md`, not a
public issue.
