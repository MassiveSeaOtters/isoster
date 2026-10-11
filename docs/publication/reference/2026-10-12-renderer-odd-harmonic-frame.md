# Renderer fix: odd harmonics must follow the unwrapped position angle (1.1.1)

Date: 2026-10-12. Branch `feature/renderer-odd-harmonic-unwrap-20261012`.
Found during the SGA-2020 bm1 round-2 smoke in `sga_isoster`; fixed in
`isoster/model.py` and in the benchmark photutils wrapper
`benchmarks/benchmark_baseline/baseline_shared.py`. Exposure of the round-2 mock
campaigns measured with `benchmarks/exhausted/analysis/renderer_frame_fix.py`
(census, paired rescoring, summary); outputs under
`outputs/benchmark_exhausted/renderer_frame_fix_20261012/`.

## What was wrong

`build_isoster_model` interpolates every ring quantity over `sma`. Because the
position angle has period π, it first unwraps `pa` with `np.unwrap(period=np.pi)`
so a jump from 175° to 5° interpolates through 180° rather than 90°. Every ring
outward of a jump larger than π/2 is thereby rotated by π (or an odd multiple).
The ellipse is unchanged, but the harmonic angle θ measured from the rotated
frame shifts by π, and

    sin(n(θ + π)) = (−1)^n sin(nθ),   cos(n(θ + π)) = (−1)^n cos(nθ),

so the odd-order coefficients `a3, b3, a5, b5, ...` stored in each ring's own
frame change sign in the rotated frame while the even orders `a4, b4, ...` do
not. The renderer interpolated the stored coefficients unchanged, so one `pa`
jump rendered the `m = 3` term with the wrong sign over the whole galaxy outside
the jump. The jump is typically in the sub-pixel inner rings of a round galaxy,
where the position angle is ill-defined, which is why the fit itself was fine
and the error only showed in the 2-D model. The photutils model wrapper used
by the benchmarks applied the same unwrap to photutils rows and had the same
flaw (photutils splines `pa` and the harmonics over `sma` and measures `phi`
from the splined `pa`).

The fix negates the odd-order coefficients of every ring whose unwrap shift is an
odd number of half-turns before building the interpolators; even orders are
left alone. Regression test:
`tests/unit/test_model.py::test_pa_unwrap_flips_odd_harmonics_with_the_frame`
renders the same isophotes once with all rings at `pa` and once with the inner
four rings written as `pa + π` with `a3, b3, a5, b5` negated; the models must
agree to 1e-9. Before the fix they differed by several per cent of the
intensity.

## How it was found

On UGC02446 (SGA-2020 demo, elliptical, MAJORAXIS 125 px) the cubic and bilinear
fits agree to 0.001 in every ring parameter (`intens`, `eps`, `pa`, `x0`, `y0`,
`a3..b4`, `grad`) beyond 3 px, yet the campaign's harmonics-on renders gave an
inner-zone data residual of 0.97 σ for the cubic fit and 0.79 σ for the bilinear
one. The cubic fit's five sub-pixel rings sit at `pa` 2.91 to 3.07 rad, the rest
at 0.14; the bilinear fit's inner rings sit at 1.60 (a jump of 1.48 < π/2), so
only the cubic render was rotated.

Mean |image − model| / σ at 12 to 20 px:

| render | cubic fit | bilinear fit |
|---|---|---|
| harmonics on, as shipped (1.1.0) | 2.151 | 1.158 |
| harmonics off | 1.512 | 1.633 |
| `m = 3` only | 2.204 | 1.253 |
| rings with `sma > 3` px only | 1.026 | 1.147 |
| inner rings rewritten as `pa − π` with `a3, b3` negated | 1.026 | (no jump) 1.158 |

The model difference between the two as-shipped renders is 2.1 σ rms in that
annulus, dominated by `m = 3` (2.4) with `m = 1` and `m = 5` sidebands; with
harmonics off it is 0.44. With the frame corrected the cubic fit beats the
bilinear one, as the spot check predicted.

## What is affected

- Any residual or model metric computed from a harmonics-on render of a fit
  with a > π/2 `pa` jump between consecutive rings. Profile-based metrics
  (truth error of `intens`, geometry, convergence, timing) are untouched.
- Round-2 mock campaigns: the headline renderer `log_off` is harmonics-off and
  unaffected. The `native` renderer rows and the sentences built on them (the
  native renderer "much better on the disks", "harmonics are signal on bars")
  are exposed; the exposure is measured below.
- Round-1 records (`reference/round1-no-psf/`) quote native renders and were
  already retired for the PSF omission.
- SGA-2020 bm1 v3 in `sga_isoster`: the campaign's `model.fits` and every
  residual metric derived from it; 37 % of the bm1_batch01 `ref_default` fits
  (150 of 400) carry such a jump, median position 5 % of `maxsma`. The models
  must be re-rendered and the afterburner re-run before those metrics are quoted.
- The stored odd coefficients themselves are frame-dependent by construction;
  an `a3(sma)` profile plotted across a `pa` jump shows a sign flip. That is a
  presentation caveat, not a bug.

## Exposure of the round-2 mock campaigns

Pending: filled from `outputs/benchmark_exhausted/renderer_frame_fix_20261012/summary.md`.
