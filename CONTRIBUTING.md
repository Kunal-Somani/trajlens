# Contributing to trajlens

## Setup

```bash
git clone https://github.com/yourusername/trajlens
cd trajlens

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dev dependencies
uv venv
source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[dev,hub]"

# Install pre-commit hooks
pre-commit install
```

## Running checks locally

```bash
# Format + lint
ruff format src/ tests/
ruff check src/ tests/

# Type check (must pass with zero errors)
mypy src/

# Tests
pytest

# All three in one go (same as CI)
ruff format src/ tests/ && ruff check src/ tests/ && mypy src/ && pytest
```

## Before opening a PR

- [ ] `ruff format`, `ruff check`, `mypy --strict` all pass with zero errors
- [ ] Tests added: happy path + two failure modes + one edge case
- [ ] No new dependency added without justification in the PR description
- [ ] Commit message follows Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`)
- [ ] One logical change per commit — the diff is readable in a single sitting

## Adding a new check

1. Add the check entry to `docs/04_CHECK_CATALOG.md` first (id, severity, detection logic, FP notes)
2. Create the check class in `src/trajlens/checks/` implementing the `Check` Protocol
3. Register it via the `@register_check` decorator
4. Add a fixture dataset to `tests/fixtures/` — one that passes and one that fails the check
5. Write unit tests referencing both fixtures
6. A check without both fixtures is not done

## Code standards

See `docs/05_ENGINEERING_STANDARDS.md` for the full rules. Short version:

- Python 3.11+, full type hints, `mypy --strict` passes
- `ruff format` + `ruff check` with zero warnings — line length 100
- No `print` — use `structlog.get_logger()`
- Comments explain *why*, not *what*
- Typed exceptions only — never bare `Exception`
- No silent failures — a check that cannot run reports ERROR, never PASS

## Security

trajlens processes untrusted dataset files. Any code that touches file paths,
parses metadata, or decodes media must follow `docs/06_SECURITY_AND_THREAT_MODEL.md`.
Key rules:

- All path construction from dataset-derived parts goes through `safe_join()`
- Every loop over dataset-declared sizes has a hard ceiling
- No `eval`, `exec`, `pickle.load`, or `shell=True` subprocess — ever

## good-first-issue

Check the `good-first-issue` label on GitHub for well-scoped tasks suitable for
first contributions. Each issue describes the check to implement, links to the
catalog entry, and notes what fixtures to create.
