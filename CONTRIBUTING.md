# Contributing to ISOSTER

Thank you for your interest in contributing to ISOSTER! This guide covers the development workflow and conventions used in this project.

## Development Setup

ISOSTER uses [uv](https://docs.astral.sh/uv/) exclusively for environment and dependency management.

```bash
# Clone the repository
git clone https://github.com/MassiveSeaOtters/isoster.git
cd isoster

# Sync environment with core + development + docs tools
uv sync --extra dev --extra docs

# Verify the setup
uv run pytest --collect-only -q
uv run mkdocs --version
```

## Code Style

- **Naming**: Use `snake_case` for all identifiers. Never use `camelCase`.
- **Names**: Use specific, complete words that are clear to those unfamiliar with the codebase.
- **Docstrings**: Provide informative and concise docstrings for all public functions and classes.
- **Comments**: Add inline comments where logic is not self-evident. Avoid redundant comments that restate function or variable names.
- **Linting**: The project uses [Ruff](https://docs.astral.sh/ruff/) via `pre-commit` hooks.

## Branch Workflow

1. Always create a new branch from `main` for your work.
2. Use descriptive branch names: `feat/description`, `fix/description`, `chore/description`.
3. Do not push directly to `main`.

## Pull Request Process

1. Ensure all tests pass: `uv run pytest tests/ -q`
2. Ensure docs build cleanly: `uv run mkdocs build`
3. Write a clear PR description summarizing the changes.
4. Link related issues if applicable.

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/unit/test_fitting.py

# Run with verbose output
uv run pytest tests/ -v

# Run real-data tests (requires example data)
uv run pytest tests/real_data -m real_data -v -s
```

Tests and benchmarks should be quantitative, with explicit statistics. See `docs/05-testing.md` for detailed benchmark and mock-validation conventions.

## Reporting Issues

When reporting a bug or requesting a feature, please include:

- A minimal reproducible example (if applicable).
- The version of Python and key dependencies (`numpy`, `scipy`, `astropy`).
- The full traceback for errors.
- Expected vs. actual behavior.

Open issues at: <https://github.com/MassiveSeaOtters/isoster/issues>
