# Testing and Benchmarks

This page has two parts: a **reference** for what the suite contains and how to
run it, and the **directives** that govern how new tests and benchmarks are
written.

## 1. Suite reference

Five categories under `tests/`:

| Directory | Tests | Scope |
|---|---:|---|
| `unit/` | 319 | Module behaviour and API contracts; fast, no I/O |
| `integration/` | 148 | Multi-module paths through `fit_image` and the drivers |
| `validation/` | 8 | Accuracy against analytic truth and against `photutils` |
| `multiband/` | 443 | Joint solver, `band_scale` invariant, per-band errors, validity policy, Schema-1 writer, CLI |
| `real_data/` | 4 | Real images; **deselected by default** (needs bundled data, and LaTeX fonts for the figure tests) |

Total collection is 922 (917 collected + 5 deselected) as of 2026-08-20.
These figures drift — regenerate rather than trusting them:

```bash
uv run pytest tests/ --collect-only -q | tail -2
uv run pytest tests/multiband --collect-only -q | tail -2
```

Common invocations:

```bash
uv run pytest tests/                      # full suite
uv run pytest tests/unit/test_fitting.py  # one file
uv run pytest tests/ -v                   # verbose
uv run pytest reference/tests/            # photutils-compatible reference implementation
```

Per-file descriptions live in [`tests/README.md`](https://github.com/MassiveSeaOtters/isoster/blob/main/tests/README.md).
Benchmarks are separate from tests and live under `benchmarks/`; see
[`09-exhausted-benchmark.md`](09-exhausted-benchmark.md) for the campaign
framework and `benchmarks/FRAMEWORK.md` for the conventions every benchmark
script follows.

Documentation numbers quoted in the technical chapter are guarded by
`benchmarks/draft_timings/check_draft_numbers.py`, which runs in the docs CI
job. It is not part of the pytest suite.

## 2. Directives

### General

- The canonical basic real-data dataset is `data/m51/M51.fits`; the corresponding
  test module is `tests/real_data/test_m51.py` (standard pytest `test_*.py`
  naming — an earlier version of this page called for `m51_test`).
- For Huang2013 workflows, use a fixed default initial SMA of `6.0` pixels (`sma0`) instead of deriving it from `RE_PX1`.
- For high-fidelity mock generation, use the external `mockgal.py` workflow referenced by `benchmarks/utils/mockgal_adapter.py` when PSF convolution and realistic background-noise controls are required.
- For `mockgal.py` benchmark/test workflows, force `--engine libprofit` and do not rely on astropy fallback rendering.
- For noiseless single-Sersic validation without PSF convolution, compare against an analytic 1-D Sersic truth using accurate `b_n` evaluation (e.g., `scipy.special.gammaincinv`) rather than low-accuracy approximations.
- Tests and benchmarks must be quantitative, with explicit statistics from 1-D profile deviations and 2-D residual diagnostics.
- Treat 2-D residual metrics as system-level diagnostics (profile extraction + model reconstruction combined), not as an isolated extraction-only metric.

### Mock Single-Sersic Model Tests

- The mock galaxy model shall be centered with the image array.
- The half-size of the mock image shall be at least 10 times of the effective radius of the mock model; if the mock model's effective radius is not very large, 15x would be even better.
- Pay attention to the oversampling ratio, high-Sersic index and high ellipticity often require higher oversampling in the central region.
- When comparing results between the truth or the reference profile:
  - Ignore the region smaller than 3 pixels when there is no PSF convolution because sampling the central region often has numerical issues; ignore the region `<= 2 * psf_fwhm` due to PSF convolution.
  - Ignore the region in the outskirt where problematic data points (using stop code) begin to appear or the intensity error bars become huge.
- Metrics to evaluate the results:
  - Median or maximum difference of a property between `0.5 * r_effective` (or 3 pixels, whichever is larger) to `8 * r_effective` is good for noiseless mocks; and to `5 * r_effective` is good for mocks with noise.
  - Using median or maximum absolute difference is a more strict standard.
  - `isoster` can provide the curve of growth measurement, the relative difference of the curve of growth values at a few typical radius could be useful metrics.

### Build Ellipse Model Tests

#### Key Residual Statistics

- Fractional residual level: `100.0 * (model - data) / data`
- Fractional absolute residual level: `100.0 * |model - data| / data`
- Chi-Square statistics: `(model - data) ** 2.0 / (sigma ** 2)`

#### Key Metrics

1. Statistics of the fractional residual level (e.g., median or maximum values) within different radial ranges. This works best for noiseless mocks.
2. Statistics of the integrated values of the fractional absolute residual level within different radial ranges. This works best for noiseless mocks.
3. Integrated Chi-square statistics within different radial ranges. This works best for noise-added mocks or real images.

#### Radial Ranges

- < 0.5 Re (effective radius)
- 0.5-4 Re
- 4-8 Re
