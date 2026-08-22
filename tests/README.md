# Tests

Automated correctness tests for isoster. Organized into five categories:
`unit`, `integration`, `validation`, `real_data`, and `multiband`.

**Current collection (2026-08-23):** pytest discovers 1141 tests and deselects
5, leaving 1136 selected; a full run reports 1135 passed and 1 skipped. By
category: `unit` 528, `integration` 158, `validation` 8, `multiband` 443,
`real_data` 4 (deselected by default; they need bundled real images and, for
the figure tests, LaTeX fonts).

Regenerate these numbers rather than trusting them:

```bash
uv run pytest tests/ --collect-only -q | tail -2
uv run pytest tests/multiband --collect-only -q | tail -2
```

## Overview Table

| File | Lines | Tests | What it covers |
|------|------:|------:|----------------|
| **unit/** | | | |
| `test_config_validation.py` | 225 | 18 | `IsosterConfig` field validation, deprecated aliases, template API |
| `test_convergence.py` | 146 | 8 | Convergence stop conditions, max-iter limits, gradient-error paths |
| `test_driver.py` | 293 | 13 | `fit_image()` API, stop-code propagation, centroid freezing |
| `test_fitting.py` | 1506 | 51 | All `fitting.py` primitives — sigma-clip, harmonic LSQ, error estimation, deviations, higher harmonics (EA + simultaneous), `fit_isophote()` |
| `test_huang2013_campaign_fault_tolerance.py` | 811 | 20 | Fault-tolerance and status-check logic for the Huang2013 campaign runner |
| `test_model.py` | 294 | 11 | `build_isoster_model()`, pixel assignment, model completeness |
| `test_public_api.py` | 187 | 4 | Public surface (`fit_image`, `isophote_results_to_fits`, `build_isoster_model`) |
| `test_sampling.py` | 49 | 2 | `sample_isophote()` — basic path sampling smoke tests |
| **integration/** | | | |
| `test_cli.py` | 69 | 1 | CLI entry-point (`isoster` command) basic invocation |
| `test_edge_cases.py` | 424 | 19 | Edge cases: empty images, all-masked inputs, very small/large SMA, degenerate geometries |
| `test_isofit_integration.py` | 447 | 6 | ISOFIT (EA mode) integration on synthetic Sérsic galaxy |
| `test_numba_validation.py` | 465 | 24 | Numba kernel correctness vs pure-Python reference, JIT warmup |
| `test_sersic_accuracy.py` | 446 | 3 | End-to-end accuracy on analytic Sérsic model — intensity, eps, PA |
| `test_template_forced.py` | 648 | 31 | Unified template-forced photometry API (R26-05): single/multi-band, freeze modes |
| **validation/** | | | |
| `test_model_residuals.py` | 228 | 1 | 2D model reconstruction residuals vs photutils on synthetic data; deprecated `build_ellipse_model` coverage |
| `test_photutils_comparison.py` | 236 | 1 | 1D isophote accuracy vs photutils on synthetic Sérsic |
| **real_data/** | | | |
| `test_m51.py` | 116 | 2 | Fitting on real M51 galaxy (`data/m51/M51.fits`); convergence rate check |
| `test_ea_harmonics_comparison.py` | 840 | 2 | Multi-method comparison on ESO 243-49 and NGC 3610: PA mode, EA mode, extended harmonics, vs photutils reference |

**Total**: see the collection figures at the top of this file. The per-file
counts in the table above are indicative and drift as tests are added; the
`--collect-only` commands are authoritative.

The table does not enumerate `tests/multiband/`, which is the largest single
category at 443 tests. It covers the joint solver, the geometry
parameterisation and `band_scale` invariant, per-band error propagation,
validity policy, the Schema-1 writer, and the multi-band CLI.

---

## Categories

### unit/

Low-level module behavior and API contracts. Fast, no I/O, no external dependencies.

- **`test_fitting.py`** — The most thorough file. Covers every public function in `isoster/fitting.py`,
  including both `unittest.TestCase` and `pytest` styles. Exercises normal paths, degenerate inputs
  (too-few points, singular matrices), and the `fflag` sigma-clip mechanism.
- **`test_config_validation.py`** — Exercises `IsosterConfig` Pydantic validation: required fields,
  range checks, deprecated aliases (`template_isophotes`), and template API consistency.
- **`test_driver.py`** — Exercises `fit_image()` behavior: stop-code propagation, center-freeze flag,
  negative-error detection, and the `stop_code=2` acceptable-failure contract.
- **`test_model.py`** — Verifies `build_isoster_model()` fills the correct pixels and produces a
  finite model everywhere an isophote was fitted.
- **`test_huang2013_campaign_fault_tolerance.py`** — Regression suite for the fault-tolerance
  logic in `examples/example_huang2013/run_huang2013_campaign.py`. Uses JSON fixtures; not a unit test of
  core fitting code.
- **`test_sampling.py`** — Minimal smoke tests for `sampling.py`. Only two tests; sampling is
  primarily exercised indirectly via `test_fitting.py` and integration tests.

### integration/

Cross-module behavior on synthetic data. Exercises multiple modules together.

- **`test_sersic_accuracy.py`** — Generates an analytic Sérsic image and verifies that
  `fit_image()` recovers intensity, ellipticity, and PA within quantitative tolerances.
- **`test_numba_validation.py`** — Verifies that Numba JIT-compiled kernels (`numba_kernels.py`)
  agree with pure-Python reference implementations to machine precision.
- **`test_template_forced.py`** — Tests the R26-05 unified template API in depth: single-band
  forced photometry, multi-band, freeze modes (center/eps/pa), and interoperability with
  `isophote_results_to_fits`.
- **`test_isofit_integration.py`** — Exercises the ISOFIT/EA mode end-to-end on a synthetic
  galaxy with higher-order harmonics enabled.
- **`test_edge_cases.py`** — Adversarial inputs: blank images, fully-masked data, SMA outside
  image bounds, `minsma > sma0`, NaN-filled arrays.
- **`test_cli.py`** — Basic smoke test that the `isoster` CLI entry-point runs and exits cleanly.

### validation/

Method-level accuracy validation against external reference behavior.

- **`test_photutils_comparison.py`** — Compares isoster 1D profile (intensity, eps, PA) against
  `photutils.isophote` on the same synthetic Sérsic image. Quantitative tolerance assertions.
- **`test_model_residuals.py`** — Compares 2D model reconstruction residuals between isoster and
  photutils. Also covers the deprecated `build_ellipse_model()` wrapper.
  **Maintenance note**: `build_ellipse_model` coverage will be removed at 2.0.0.

### real_data/

Real-galaxy tests. Require FITS data files from `data/`. Marked `@pytest.mark.real_data` and
excluded from the default run.

- **`test_m51.py`** — Canonical basic real-data test. Verifies `fit_image()` runs on M51 and
  achieves >50% convergence rate. Requires `data/m51/M51.fits`.
- **`test_ea_harmonics_comparison.py`** — Comprehensive comparison of PA mode, EA mode with [3,4]
  harmonics, and EA mode with [3..10] harmonics on ESO 243-49 and NGC 3610. Generates QA figures
  and JSON results under `outputs/tests_real_data/`. Requires photutils.

---

## Run Commands

```bash
# Default: all tests except real_data (fast, ~50s)
uv run pytest tests/ -q

# Unit tests only
uv run pytest tests/unit -q

# Integration tests only
uv run pytest tests/integration -q

# Validation tests only
uv run pytest tests/validation -q

# Real-data tests (requires data/ FITS files and photutils)
uv run pytest tests/real_data -m real_data -v -s

# Specific file
uv run pytest tests/unit/test_fitting.py -v

# Run a specific test by name
uv run pytest tests/unit/test_fitting.py -k test_sigma_clip -v

# Collection sanity check (no test execution)
uv run pytest --collect-only -q
```

---

## Output Policy

Tests that generate artifacts write under `outputs/`, not under `tests/`.
Override the output root with the `ISOSTER_OUTPUT_ROOT` environment variable.

Naming convention:
- `outputs/tests_unit/<run_id>/...`
- `outputs/tests_integration/<run_id>/...`
- `outputs/tests_validation/<run_id>/...`
- `outputs/tests_real_data/<run_id>/...`

---

## Coverage Gaps

- **`test_sampling.py`** is sparse (2 tests). Sampling is exercised indirectly; direct unit
  coverage of `sampling.py` paths (different step modes, sector-area sampling) is thin.
- **`test_cli.py`** is a minimal smoke test. Config-file parsing, multi-extension FITS output,
  and error handling are not covered.
- **`test_m51.py`** requires the external file `data/m51/M51.fits`. If it is missing, the test is
  skipped (not failed).

---

## Maintenance Notes

- `test_model_residuals.py` includes coverage of the deprecated `build_ellipse_model()` function.
  This coverage will be removed when `build_ellipse_model` is deleted at 2.0.0.
- `test_fitting.py` uses `unittest.TestCase` style for legacy reasons; new tests in that file
  should use pytest style.
- `conftest.py` (at `tests/real_data/conftest.py`) registers the `real_data` marker so that
  `uv run pytest tests/` does not run real-data tests by default.
