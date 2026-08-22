# Testing and Benchmarks

This page has two parts: a **reference** for what the suite contains and how to
run it, and the **directives** that govern how new tests and benchmarks are
written.

## 1. Suite reference

Five categories under `tests/`:

| Directory | Tests | Scope |
|---|---:|---|
| `unit/` | 528 | Module behaviour and API contracts; fast, no I/O |
| `integration/` | 158 | Multi-module paths through `fit_image` and the drivers |
| `validation/` | 8 | Accuracy against analytic truth and against `photutils` |
| `multiband/` | 443 | Joint solver, `band_scale` invariant, per-band errors, validity policy, Schema-1 writer, CLI |
| `real_data/` | 4 | Real images; **deselected by default** (needs bundled data, and LaTeX fonts for the figure tests) |

As of 2026-08-23 pytest discovers 1141 tests and deselects 5, leaving 1136
selected; a full run reports 1135 passed and 1 skipped. These figures drift —
regenerate rather than trusting them:

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

The cross-tool harmonic-scale calibration is guarded the same way, by
`benchmarks/harmonic_scale/check_harmonic_scale.py`, the third gate in that
job. It covers **every** archived campaign — there are two, one per galaxy —
and checks each archive's fingerprint against the fixture that archive says it
used. It reads only the committed archive and the committed prose, so it needs
no AutoProf — which matters, because CI has none. What it enforces is a
pre-registration: the tolerances in `frozen_tolerances.json` were derived from
a pilot run on one noise-seed block and committed *before* the validation run
on a disjoint block that they judge. Choosing a tolerance after seeing the
result it will judge is the failure that procedure exists to prevent.

Two rules follow for anyone touching that archive. Do not hand-edit a number
in it or in the prose it guards — re-run
`run_harmonic_scale.py --mode validation --archive` and let the checker print
the replacements. And do not change the grid, the fixture or the planted modes
without re-freezing: those inputs are covered by a fixture fingerprint, and a
changed fingerprint means the archived numbers describe a different
experiment, which the checker reports as a redefinition rather than as drift.
Archiving refuses to run from a dirty working tree, checked both when the run
was made and when the archive is written.

A fourth gate, `benchmarks/harmonic_scale/check_gradient_reconstruction.py`,
guards the two A4 Track 2 archives on the same terms, and adds two things the
third does not need.

First, every Track 2 quantity is archived under **two reductions over rings**,
because a max and a median over five rings answer different questions and
neither substitutes for the other:

- `worst_ring_*` is the max. It is the statistic the acceptance criteria were
  pre-registered on, and the conservative one for a licence, since a licence
  is a claim about the worst ring a reader might quote. It is also unstable
  under noise — it reports whichever ring happened to be worst in this seed
  block — so it reproduces loosely.
- `typical_ring_*` is the median over the same values. No verdict rests on it,
  but it reproduces roughly forty times more tightly between seed blocks, so
  it is what actually constrains the gate.

The licensing criteria read the `worst_ring` family and nothing else. Choosing
a reduction after seeing which verdict it produces is exactly what a
pre-registration forbids, and the two do differ: a blanket switch to the
median was measured to flip criterion 2 on the n = 2 fixture's reference case,
which would have unlicensed Track 2 there.

Second, each tolerance is **measured by bootstrapping that claim's own
definition** — the same reduction over the same columns — across resampled
realizations of the pilot. An earlier version used the largest single-ring
standard error as a stand-in, which is only correct when the claim is one
ring's median; a max additionally varies through *which* ring wins, so its
true spread is larger, and the gate failed a validation run whose measurement
was fine. Because a tolerance is now tied to a specific claim definition, the
frozen file also carries a fingerprint of those definitions, and the gate
fails if the definitions have moved since the freeze. Without it, editing a
reduction would silently compare a validation value computed one way against a
pilot value computed another.

The gate reports two kinds of weak claim separately, and the distinction
matters. A *measured* tolerance larger than half its own value means the claim
is genuinely unstable and constrains little. A claim sitting under the
deterministic floor is merely small: it reproduces exactly, and the floor is a
fixed allowance rather than a statement about spread. Reporting both as one
line would train the reader to skip it.

Run `check_harmonic_scale.py --self-test` after changing the checker. It
corrupts the archived values and requires every claim, and every prose
sentence quoting one, to fail. That is not ceremony: the prose gate once
matched documents on a stem that contained the guarded number, so corrupting
the number made the check *dormant* rather than failing, and seven of nine
guarded figures could be edited without complaint. A stem must identify a
claim by something stable — the campaign label, or the radius — never by the
value it guards. `check_gradient_reconstruction.py --self-test` does the
equivalent for Track 2: it corrupts the archived medians and requires both the
claims and the recomputed licensing verdict to trip. Both self-tests run in
docs CI, so a gate that has gone quiet fails the build rather than passing it.

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
