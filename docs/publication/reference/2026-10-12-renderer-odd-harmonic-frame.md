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
- Not affected: the fit itself (every ring's harmonics are solved in its own
  frame), the forced-photometry and multi-band paths (same geometry per ring),
  and the QA contour overlays in `isoster/plotting.py`, which draw each
  isophote from its own `pa` and coefficients without interpolating across
  rings. The two interpolating renderers were the only places where rings in
  different frames were combined.

## Exposure of the round-2 mock campaigns

Measured with `renderer_frame_fix.py` on 2026-10-12 (census 66,000 run records at
8 workers in 10 min; 11,075 affected fits re-rendered in 8 min). Full tables for
every arm: `outputs/benchmark_exhausted/renderer_frame_fix_20261012/summary.md`.

**Census.** About one ok fit in five has at least one ring rotated by the unwrap,
at every depth, noiseless included:

| dataset | tool | ok fits | affected | first rotated ring, sma/R_e (p10, median, p90) | max odd coefficient on rotated rings (median, p90) |
|---|---|---:|---:|---|---|
| Huang2013 | isoster | 14,229 | 2,819 (19.8 %) | 0.09, 0.79, 3.18 | 0.031, 0.091 |
| Huang2013 | photutils | 3,289 | 688 (20.9 %) | 0.03, 0.19, 2.01 | 0.028, 0.74 |
| S4G | isoster | 30,600 | 6,118 (20.0 %) | 0.15, 0.49, 1.69 | 0.004, 0.048 |
| S4G | photutils | 6,910 | 1,450 (21.0 %) | 0.03, 0.29, 1.12 | 0.009, 0.055 |

Affected fraction by depth, isoster: Huang2013 14 % noiseless, 17 % deep, 24 % wide;
S4G 19, 20, 21 %. The rotations are not confined to sub-pixel rings: on the round
Huang2013 ellipticals the position angle is ill-defined at every radius, and the
median first rotation sits near 0.8 R_e.

**Paired rescoring of the native renderer.** For every affected fit the native model
was rendered with the old behaviour (emulated exactly) and with the fix, and both were
scored against the saved truth on the campaign zones. The median paired change in truth
RMS is 0.000 percentage points in every zone for every arm quoted in the records; the
10th to 90th percentile spread is within ±0.03 points in the inner and mid zones and
within ±0.35 points in the noisy outer zone (both signs). Campaign medians of the
native renderer, archived against patched by the paired shift (truth RMS in per cent):

| dataset | arm | all | inner | mid | outer |
|---|---|---|---|---|---|
| Huang2013 | `ref_default` | 1.324 → 1.324 | 1.097 → 1.097 | 1.403 → 1.403 | 4.539 → 4.517 |
| Huang2013 | `samp_cubic` | 0.697 → 0.697 | 0.250 → 0.250 | 0.811 → 0.811 | 4.932 → 4.956 |
| Huang2013 | `samp_cubic_ea` | 0.691 → 0.694 | 0.243 → 0.243 | 0.795 → 0.795 | 4.993 → 4.994 |
| Huang2013 | `samp_cubic_snr200` | 0.656 → 0.656 | 0.250 → 0.250 | 0.808 → 0.808 | 4.521 → 4.521 |
| Huang2013 | photutils `baseline_median` | 2.294 → 2.291 | 2.086 → 2.086 | 1.904 → 1.905 | 4.877 → 4.849 |
| S4G | `ref_default` | 1.264 → 1.266 | 1.102 → 1.102 | 1.216 → 1.216 | 1.514 → 1.514 |
| S4G | `samp_cubic` | 0.800 → 0.800 | 0.354 → 0.353 | 0.969 → 0.969 | 1.490 → 1.489 |
| S4G | `samp_cubic_ea` | 0.752 → 0.752 | 0.306 → 0.306 | 0.896 → 0.896 | 1.490 → 1.492 |
| S4G | `samp_cubic_snr200` | 0.781 → 0.781 | 0.357 → 0.357 | 0.973 → 0.973 | 1.475 → 1.474 |
| S4G | photutils `baseline_median` | 1.923 → 1.923 | 1.998 → 1.998 | 1.271 → 1.271 | 1.249 → 1.246 |

**Conclusion.** The round-2 records stand. Every native-renderer number quoted in the
Huang2013 and S4G records changes by at most 0.003 percentage points in the full
aperture and the inner and mid zones, and by at most 0.03 points in the outer zone,
below the precision the records quote and far below the differences they discuss
(0.80 against 1.31 % on the disks). The reason is physical: the mocks are symmetric,
so the `m = 3` coefficients that the bug flips are noise on the ellipticals and nearly
zero on the barred disks, whose structure is even (`m = 2, 4`). On real galaxies with
genuine lopsidedness the same flip is large: UGC02446 carries `b3 ≈ −0.013` at 12 to
20 px and its residual doubled. The SGA-2020 campaign, not the mock campaigns, is where
the fix changes numbers, and `sga_isoster` must re-render before quoting them.

The old-behaviour emulation reproduces the archived values exactly at the median; the
90th-percentile difference (0.34 points on Huang2013, 0.06 on S4G) is the support
difference described above, in the noisy outskirts, and cancels in the paired shift.
