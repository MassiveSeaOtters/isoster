# CLAUDE.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

ISOSTER (ISOphote on STERoid) is an accelerated Python library for elliptical isophote fitting in galaxy images. It runs tens of times faster than `photutils.isophote` (median 45x on the synthetic Sersic sweep in `benchmarks/performance/`; archived in `reference_speedup.json`) using vectorized path-based sampling via scipy's `map_coordinates`.

## Non-negotiable Rules for developing

- Always create a new branch for new features and new development. Do not merge back into the main branch unless I approve it.
- Use clear, plain English in progress updates, plans, and final summaries. Assume the reader has an astrophysics background, not a professional software-engineering or project-management background. Avoid jargon when a plain phrase works; for example, say "run a realistic test that writes output" instead of "write-mode smoke refresh", "copy one galaxy's campaign folder and test there so the original data is not changed" instead of "on a copied single-galaxy campaign tree", and "find/list the matching galaxy folders" instead of "enumerate galaxies". If a technical term is needed, define it the first time.
- It is essential to provide informative and concise docstrings and inline comments.
- Warn the users when the context window has <30% left. Remind the users to save the conversation and start fresh. Also propose ways to compact the conversation and save the current progress in files.
- Keep record of development progress, important lessons, and critical decisions in markdown files in `docs/agent/journal/`.
- Keep `docs/04-architecture.md` updated for architecture and interface changes.
- Keep active execution checklist and end-of-phase review in `docs/agent/todo.md`.
- Use lowercase kebab-case for markdown file names (for example, `stop-codes.md`).
- Generated artifacts must be written under `outputs/` and not mixed into source folders.
- **Folder-name shorthand**: `benchmark_xxx` always means `benchmarks/xxx/`, `example_xxx` always means `examples/xxx/`, `test_xxx` always means `tests/xxx/`. Never create a sibling `benchmark_xxx/` at repo root — drop straight under the corresponding parent. Generated artifacts go under `outputs/benchmark_xxx/` etc., which stays gitignored.
- Use `uv` as the default tool for dependency management and environment execution.
- **Surface brightness convention**: whenever intensity (per pixel) is converted to surface brightness in mag/arcsec², the formula is always `μ = -2.5·log10(I_per_pix / pixarea) + zp` where `pixarea = pixel_scale_arcsec**2`. The photometric zeropoint `zp` and the pixel scale `pixel_scale_arcsec` must be passed to the plotting/conversion code as **two separate inputs** and never pre-combined into a single effective zeropoint. Reference values: HSC coadd `zp=27.0`, `pixel_scale_arcsec=0.168`. LegacySurvey (DECaLS/BASS/MzLS) `zp=22.5`, pixel scale from the image header (typically 0.262″ for DECaLS). Callers that pass one of the two must pass both; passing only one is a hard error.
- **Harmonic-coefficient convention**: all plotting *and* analysis of third- and fourth-order deviation amplitudes `A_n`, `B_n` must use the Bender-normalized form `A_n_norm = -A_n / (a · dI/da)` (and analogously for `B_n`), where `a` is the isophote's semi-major axis and `dI/da` is the local intensity gradient. Raw `A_n`/`B_n` values are **not** interpretable — their magnitudes scale with the local flux and change across a galaxy's profile even when the underlying geometric deviation is identical. Normalized values are scale-invariant and directly comparable across arms, tools, and galaxies. This rule applies uniformly to (a) `isoster.plotting` (used by every exhausted-benchmark tool: isoster, photutils, autoprof — `plot_qa_summary` calls `_normalize_harmonic_for_plot`), and (b) `benchmarks/exhausted/analysis/scenario_summary.py` (audit pipeline — uses `normalize_harmonic` + `effective_grad`). When a tool ships no usable `grad` column (autoprof), fall back to `np.gradient(intens, sma)` via the helpers; never plot or score raw `A_n`. When an arm intentionally does not fit harmonics (`harm_disabled`, or `harmonic_orders` sweeps that replace `[3, 4]`), the audit marks Prior 2 `prior2_applicable=False` and the QA plot's harmonic panel is auto-skipped — do not re-introduce a code path that silently evaluates raw `A_n`.

## Context and Memory Preservation

When a task is long or the context window is becoming constrained:

1. Write a concise status snapshot to `docs/agent/journal/` (what was done, what remains, blockers).
2. Update `docs/agent/todo.md` checklist status and review notes.
3. Update `docs/04-architecture.md` if architecture or interfaces changed.
4. Update `docs/agent/lessons.md` when a correction yields a reusable lesson.
5. Before ending, include exact file paths and commands needed to resume.

## Build and Test Commands

```bash
# Sync project environment (core + development + docs tooling)
uv sync --extra dev --extra docs

# Run all tests (main package)
uv run pytest tests/

# Run a single test file
uv run pytest tests/unit/test_fitting.py

# Run reference implementation tests (photutils-compatible)
uv run pytest reference/tests/

# Run with verbose output
uv run pytest tests/ -v

# CLI usage
uv run isoster image.fits --output isophotes.fits --config config.yaml

# Build docs
uv run mkdocs serve
```

## uv Workflow Rules

Follow these rules for all Python environment and dependency work in this repository:

1. Use `uv` as the single workflow for environment and dependency management.
2. Do not use `pip`, `poetry`, or `conda` commands for project dependency changes.
3. Install/sync environment with:
   - `uv sync --extra dev --extra docs`
4. Run project commands through the managed environment:
   - `uv run pytest ...`
   - `uv run python ...`
   - `uv run isoster ...`
   - `uv run mkdocs ...`
5. When adding/removing dependencies, update `pyproject.toml` and then run:
   - `uv lock`
   - `uv sync --extra dev --extra docs`
6. Keep `uv.lock` committed and up to date with dependency changes.
7. For tools that are not compatible with all Python versions in `requires-python`,
   use dependency markers in `pyproject.toml` (for example `python_version >= '3.9'`).
8. Minimum verification after dependency changes:
   - `uv run pytest --collect-only -q`
   - `uv run mkdocs --version`

## Documentation Index

### Public docs (tracked, served by mkdocs)

| Document | Description |
|----------|-------------|
| `docs/index.md` | Documentation home page and map |
| `docs/archive/` | Retired dated reports (pre-publication review, QA refresh). Tracked so the history survives, excluded from the published site. |
| `docs/01-user-guide.md` | Usage guidance, stop-code reference, public API |
| `docs/02-configuration-reference.md` | All configuration parameters and guidelines |
| `docs/03-algorithm.md` | Fitting and sampling implementation notes |
| `docs/04-architecture.md` | Architecture, interfaces, and design decisions |
| `docs/05-testing.md` | Testing and benchmark directives |
| `docs/06-qa-functions.md` | QA plotting functions: usage, options, and examples |
| `docs/07-lsb-features.md` | Design + implementation notes for LSB auto-lock and outer-region center regularization |
| `docs/08-outer-regularization.md` | Publication-grade reference for the outer-region Tikhonov regularization: math, algorithm, config, benchmarks |
| `docs/09-exhausted-benchmark.md` | Exhausted benchmark campaign framework: YAML schema, arm sentinels, output layout, composite score, adapter recipe |
| `docs/10-multiband.md` | Multi-band isoster (experimental): joint free fit replacing forced photometry; API, Schema 1 column reference, joint-solver math, demo |
| `docs/technical/1.0`–`1.6` | Long-form technical chapter. **Tracked and published.** Every timing in it is produced by `benchmarks/draft_timings/run_draft_timings.py` and checked against `reference_timings.json` by `check_draft_numbers.py`, which gates the docs CI job. Do not hand-edit a quoted number: re-archive and let the checker print the replacements. |

### Agent-internal docs (untracked, in `docs/agent/`)

| Document | Description |
|----------|-------------|
| `docs/agent/todo.md` | Active execution checklist and review notes |
| `docs/agent/lessons.md` | Development lessons and process guardrails |
| `docs/agent/future.md` | Long-term upgrades and research roadmap |
| `docs/agent/qa-figures.md` | QA figure layout and style conventions |
| `docs/agent/journal/` | Chronological project journal notes |


## Testing and Benchmark Directives

See `docs/05-testing.md`.

## QA Figure Rules

See `docs/agent/qa-figures.md`.

## Architecture and Key Concepts

See `docs/04-architecture.md` and `docs/03-algorithm.md`.

## Public API and Configuration

See `docs/01-user-guide.md` and `docs/02-configuration-reference.md`.

## Mock Generation

For high-fidelity mock generation with PSF convolution and realistic noise, use the external `mockgal.py` workflow referenced by `benchmarks/utils/mockgal_adapter.py`. Force `--engine libprofit` and do not rely on astropy fallback rendering.
