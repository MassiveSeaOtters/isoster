# Round-2 Huang2013 full campaign: cubic sampling on the whole sample (2026-10-10)

The round-2 small-sample campaign (20 galaxies) showed that cubic ring sampling removes
Isoster's bilinear read bias and closes the gap to AutoProf. This report repeats the
measurement on the whole Huang2013 mock sample with the full round-1 arm roster, so the
paper's tables can be rebuilt from one generation of PSF-convolved images.

## Design and provenance

- Campaign `round2_huang2013_2026_10_10`, 93 galaxies, 9 scenarios
  (noiseless at z = 0.05; wide and deep at z = 0.05, 0.20, 0.35, 0.50): 837 inputs.
  Images: round-2 PSF-convolved mocks, generator commit `8632e1f`.
- 25 arms. Isoster (17): the round-1 core (`ref_default`, `reg_outer_damp`, `stack_all`)
  and afterburner arms (`int_median`, `geom_simul`, `lsb_autolock`, `geom_ea`,
  `geom_simul_ea`, `harm_simul_ea`), the small-sample sampling arms (`samp_cubic`,
  `samp_cubic_ea`, `samp_cubic_ea_min4`, `samp_cubic_ea_median`,
  `samp_cubic_ea_outer_damp`) and the outer-ring fallback arms (`samp_cubic_outer2re`,
  the oracle that reads bilinearly beyond 2 R_ref; `samp_cubic_rms010` and
  `samp_cubic_snr200`, the self-contained switches). `ols_noweight` was left out because
  it is a no-op without a variance map. Photutils (4): `baseline_median`,
  `aggressive_clip`, `fixed_center`, `default_bilinear`. AutoProf (4, zero background):
  `baseline`, `deep`, `high_regularization`, `fix_center`.
- Isoster code at commit `da060d8`; campaign files in `14dbe76`. Gate on NGC2325 first.
- Measurement: `benchmarks/exhausted/analysis/round2_small_sample.py` (unchanged from the
  small sample). The truth image of every input was regenerated with the frozen generator
  and matched the campaign image exactly in all 837 cases. Each arm is rendered with four
  harmonic-off radial interpolators and the tool's native model, scored by truth RMS in
  percent on one common pixel support per input with the 2-pixel inner cut, in four
  zones: all, inner (< 0.5 R_ref), mid, outer (≥ 2 R_ref).
- Headline renderer: the log-linear harmonic-off renderer (`log_off`). It gives the lowest
  RMS for the cubic Isoster arms and for AutoProf, and the ranking of tools and arms is
  the same under every renderer (table below). Win fractions are paired: an arm is scored
  only on inputs where both it and the reference produced a model.
- Outputs: `/Volumes/galaxy/isophote/publication_single_band_round2_2026_10_10/analysis/huang2013_full_2026_10_10/`
  (`fit_metrics.csv`, `summary.md` with every arm, renderer and zone; curated
  `report_log_off.md` and `report_linear_off.md` from
  `benchmarks/exhausted/analysis/round2_full_report.py`) and
  `.../analysis/huang2013_switch_radius_2026_10_10/` (switch radii from
  `benchmarks/exhausted/analysis/round2_switch_radius.py`).

## Fit outcomes and cost

Isoster: 0 failures in 14,229 fits. AutoProf: 49 errors, 46 of them on three galaxies
(IC2311, NGC1379, NGC1407) in every arm except `deep`. Photutils: 59 failures, 58 of them
`baseline_median` and `fixed_center` on noiseless images (median mode on noise-free pixels)
plus a few 900 s timeouts.

| arm | median fit time [s] | relative to `ref_default` |
|---|---:|---:|
| isoster `ref_default` | 0.065 | 1.00 |
| isoster `samp_cubic` | 0.066 | 1.01 |
| isoster `samp_cubic_rms010` / `samp_cubic_snr200` | 0.065 | 1.01 |
| isoster `samp_cubic_ea` | 0.067 | 1.03 |
| photutils `baseline_median` | 6.6 | 102 |
| AutoProf `baseline` | 9.7 | 150 |

## Headline: full aperture, log-linear renderer

| arm | n | median RMS [%] | 90th pct [%] | beats `ref_default` | beats AutoProf `baseline` |
|---|---:|---:|---:|---:|---:|
| isoster `ref_default` (bilinear) | 837 | 1.265 | 2.24 | – | 0.31 |
| isoster `samp_cubic` | 837 | 0.647 | 1.95 | 0.95 | 0.80 |
| isoster `samp_cubic_ea` | 837 | 0.639 | 1.88 | 0.96 | 0.84 |
| isoster `samp_cubic_rms010` | 837 | 0.614 | 1.80 | 0.98 | 0.84 |
| isoster `samp_cubic_snr200` | 837 | 0.613 | 1.82 | 0.98 | 0.84 |
| isoster `samp_cubic_outer2re` (oracle) | 837 | 0.618 | 1.90 | 0.97 | 0.81 |
| isoster `samp_cubic_ea_min4` | 837 | 1.122 | 2.18 | 0.87 | 0.36 |
| photutils `baseline_median` | 806 | 1.384 | 2.37 | 0.19 | 0.28 |
| photutils `default_bilinear` | 837 | 1.350 | 2.36 | 0.27 | 0.28 |
| AutoProf `baseline` | 820 | 1.153 | 2.39 | 0.70 | – |
| AutoProf `deep` | 836 | 1.187 | 2.54 | 0.63 | 0.37 |

Cubic sampling halves the full-aperture RMS of the default (1.265 → 0.647 %) and turns
a 31 % win rate against AutoProf into 80 %. The two self-contained switch arms are the
best arms overall (0.61 %, 84 % against AutoProf) and beat plain cubic in two thirds of
the inputs. The bilinear inner fallback below 4 px (`samp_cubic_ea_min4`) gives most of
the gain away on these PSF-convolved images and should not be used.

## By zone

Median truth RMS in percent, log-linear renderer; win fractions against AutoProf `baseline`
in parentheses.

| arm | inner (< 0.5 R_ref) | mid | outer (≥ 2 R_ref) |
|---|---:|---:|---:|
| isoster `ref_default` | 1.097 (0.28) | 1.337 (0.42) | 3.574 (0.59) |
| isoster `samp_cubic` | 0.239 (0.91) | 0.868 (0.62) | 3.841 (0.51) |
| isoster `samp_cubic_ea` | 0.238 (0.93) | 0.855 (0.71) | 3.720 (0.55) |
| isoster `samp_cubic_rms010` | 0.238 (0.90) | 0.864 (0.64) | 3.522 (0.58) |
| isoster `samp_cubic_snr200` | 0.238 (0.91) | 0.858 (0.66) | 3.495 (0.59) |
| isoster `samp_cubic_outer2re` | 0.238 (0.90) | 0.870 (0.62) | 3.521 (0.58) |
| photutils `baseline_median` | 1.176 (0.27) | 1.414 (0.26) | 4.044 (0.54) |
| AutoProf `baseline` | 0.780 | 1.066 | 3.952 |

The inner zone is where the bilinear bias lives: cubic cuts the RMS by a factor 4.6 and
beats AutoProf nine times out of ten. The mid zone improves by a third. The outer zone
shows the noise penalty of the cubic read seen in the small sample (3.84 against 3.57 %
for bilinear, 7 % worse), and both switches remove it: they are better than bilinear,
better than the 2 R_ref oracle, and better than AutoProf there.

Outer-zone median RMS by image depth:

| depth | `ref_default` | `samp_cubic` | `samp_cubic_rms010` | `samp_cubic_snr200` | oracle 2 R_ref | AutoProf `baseline` |
|---|---:|---:|---:|---:|---:|---:|
| noiseless | 0.361 | 0.358 | 0.358 | 0.358 | 0.361 | 0.489 |
| deep | 3.037 | 3.149 | 2.979 | 2.959 | 3.038 | 3.267 |
| wide | 5.868 | 6.308 | 5.865 | 5.858 | 5.870 | 6.434 |

## Regressions

Ratio of full-aperture RMS to `ref_default`, log-linear renderer:

| arm | 10 / 50 / 90 % quantiles | worse by > 20 % |
|---|---|---:|
| `samp_cubic` | 0.22 / 0.54 / 0.91 | 3 of 837 (all shallow z ≥ 0.35 images; worst 1.28) |
| `samp_cubic_rms010` | 0.21 / 0.52 / 0.86 | 0 of 837 (worst 1.10) |

## Where the switch fired

Every refit of the switch arms reproduced the saved campaign profile exactly. Switch radius
in units of R_ref (10 / 50 / 90 % quantiles) and the fraction of inputs where the switch
fired:

| arm | depth | fired | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|
| `samp_cubic_rms010` | noiseless | 0.00 | – | – | – |
| | deep | 1.00 | 1.7 | 2.8 | 4.1 |
| | wide | 1.00 | 1.1 | 1.9 | 3.1 |
| `samp_cubic_snr200` | noiseless | 0.02 | 0.9 | 2.3 | 3.7 |
| | deep | 0.99 | 1.2 | 2.3 | 4.3 |
| | wide | 1.00 | 0.7 | 1.5 | 3.0 |

The scatter rule never fires on a noise-free image and fires later on deep than on shallow
images, as intended; the S/N rule fires about 0.4 R_ref earlier. At z = 0.05 a third of the
inputs never reach the threshold before the last ring. Median ring S/N at 2 R_ref is 650
(deep) and 270 (wide) at z = 0.05, falling to 80 and 40 at z = 0.5.

## Renderer dependence

Median full-aperture RMS in percent of the primary arms under each renderer:

| renderer | isoster `ref_default` | isoster `samp_cubic` | AutoProf `baseline` | photutils `baseline_median` |
|---|---:|---:|---:|---:|
| linear, harmonics off | 1.322 | 0.690 | 1.164 | 1.452 |
| cubic, harmonics off | 1.227 | 0.661 | 1.180 | 1.363 |
| log-linear, harmonics off | 1.265 | 0.647 | 1.153 | 1.384 |
| photutils spline painter | 2.107 | 1.211 | 1.541 | 2.230 |
| native model | 1.324 | 0.697 | 1.217 | 2.294 |

The ranking is the same under every renderer. The linear renderer used in round 1 gives the
same conclusions with slightly larger numbers (`report_linear_off.md`).

*Note added 2026-10-12.* The "native model" row was rendered with isoster 1.1.0, whose
renderer applied `a3, b3` with the wrong sign outward of any > π/2 position-angle jump
(20 % of these fits; fixed in 1.1.1). Re-scoring the affected fits with the fixed renderer
moves the medians in this row by at most 0.003 percentage points (`samp_cubic_ea` 0.691 →
0.694, photutils 2.294 → 2.291); the harmonic-off rows are unaffected by construction. See
`2026-10-12-renderer-odd-harmonic-frame.md`.

## Integrator

Median integration is worse than the clipped mean on the whole sample, as in the small
sample: `int_median` 1.323 against 1.265 % with bilinear reads, `samp_cubic_ea_median`
0.655 against 0.639 % with cubic reads (beats the mean in 48 % of inputs). The mean stays.

## Recommendation for the default

- Make cubic ring sampling the default. On the whole sample it halves the error, never
  fails, costs 1 % in fit time, and regresses by more than 20 % in 3 of 837 inputs, all
  shallow high-redshift images where the extra read noise dominates.
- Keep the self-contained switch available and documented, off by default until a real-data
  check. On the mocks `sampling_cubic_max_relative_rms = 0.1` is a free improvement
  (no input worse than the default by more than 10 %), and its failure mode on real data is
  benign: an early switch returns the rings beyond it to today's bilinear behaviour.
- Do not use `sampling_cubic_min_sma = 4` on PSF-convolved images; keep its default of 0.

## Next

The S4G campaign (300 galaxies, 6 scenarios, same arms) ran after this one and is reported
in [2026-10-11-round2-s4g-full-campaign.md](2026-10-11-round2-s4g-full-campaign.md). The
measurement script gained a `--dataset` switch for it (commit `971b9ed`).
