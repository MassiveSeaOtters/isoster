# Round-2 S4G full campaign: cubic sampling on barred disk galaxies (2026-10-11)

The Huang2013 sample tests the tools on single-Sersic ellipticals, where the isophotes
are ellipses and the only question is how well each tool reads them. The S4G sample asks
the harder question: what happens when the galaxy is a bulge, an exponential disk and a
bar with their own position angles and ellipticities, so that the isophotes twist and are
not ellipses at all. This report scores the same 25 arms on that sample.

## Design and provenance

- Campaign `round2_s4g_2026_10_10`, 300 galaxies, 6 scenarios (noiseless, wide and deep
  at z = 0.05 and 0.10): 1800 inputs. Images: round-2 PSF-convolved mocks, generator
  commit `8632e1f`, 0.7 arcsec Gaussian PSF, HSC i-band pixel scale and depths.
- Models: 300 late-type disks from the Salo et al. (2015) S4G GALFIT decompositions
  (sizes from Comerón et al. 2025), chosen to be at least 75 px across at z = 0.1 with a
  preference for complex models. Components: Sersic bulge and disk (523), Ferrer bar (260
  galaxies), unresolved nucleus (130). 9 galaxies have two components, 269 three, 22 four.
- Arms, tools and Isoster code identical to the Huang2013 full campaign (25 arms; campaign
  file `campaign.round2_s4g_2026_10_10.yaml` in commit `14dbe76`; Isoster at `da060d8`).
- Measurement: `benchmarks/exhausted/analysis/round2_small_sample.py --dataset s4g`
  (commit `971b9ed`), otherwise unchanged: the truth image of every input was regenerated
  with the frozen generator and matched the campaign image exactly in all 1800 cases; four
  harmonic-off radial interpolators plus the tool's native model; truth RMS in percent on
  one common pixel support per input with the 2-pixel inner cut; zones all, inner
  (< 0.5 R_ref), mid, outer (≥ 2 R_ref); paired win fractions.
- Headline renderer: the log-linear harmonic-off renderer (`log_off`), as for Huang2013.
  The harmonics-on renderer tells a different story on this sample and gets its own
  section below.
- Outputs: `/Volumes/galaxy/isophote/publication_single_band_round2_2026_10_10/analysis/s4g_full_2026_10_11/`
  (`fit_metrics.csv`, `summary.md`, `report_log_off.md`, `report_linear_off.md`) and
  `.../analysis/s4g_switch_radius_2026_10_11/`.
- Cost: 6.5 h wall for 45,000 fits (22:59 to 05:27, eight galaxies in parallel).

## Fit outcomes and cost

Isoster: 0 failures in 30,600 fits. AutoProf: 85 errors (`Process_Image returned 1`) on
13 galaxies, 15 of the 27 `baseline` errors on noiseless images; the `deep` arm fails on
2 inputs only. Photutils: 290 failures, 282 of them the median-mode arms
(`baseline_median`, `fixed_center`) on noiseless images (NaN conversion or a 900 s
timeout), the rest an empty isophote list.

| arm | median fit time [s] | relative to `ref_default` |
|---|---:|---:|
| isoster `ref_default` | 0.066 | 1.00 |
| isoster `samp_cubic` | 0.068 | 1.03 |
| isoster `samp_cubic_rms010` / `samp_cubic_snr200` | 0.068–0.069 | 1.03–1.04 |
| photutils `baseline_median` | 8.4 | 126 |
| AutoProf `baseline` | 7.7 | 116 |

## Headline: full aperture, log-linear renderer

| arm | n | median RMS [%] | 90th pct [%] | beats `ref_default` | beats AutoProf `baseline` |
|---|---:|---:|---:|---:|---:|
| isoster `ref_default` (bilinear) | 1800 | 1.601 | 4.58 | – | 0.54 |
| isoster `samp_cubic` | 1800 | 1.310 | 4.62 | 0.84 | 0.80 |
| isoster `samp_cubic_ea` | 1800 | 1.306 | 5.24 | 0.71 | 0.85 |
| isoster `samp_cubic_rms010` | 1800 | 1.304 | 4.61 | 0.85 | 0.82 |
| isoster `samp_cubic_snr200` | 1800 | 1.304 | 4.55 | 0.88 | 0.82 |
| isoster `samp_cubic_outer2re` (oracle) | 1800 | 1.306 | 4.64 | 0.86 | 0.82 |
| isoster `samp_cubic_ea_min4` | 1800 | 1.560 | 5.29 | 0.60 | 0.54 |
| photutils `baseline_median` | 1658 | 1.779 | 5.89 | 0.27 | 0.42 |
| photutils `default_bilinear` | 1798 | 1.839 | 5.63 | 0.33 | 0.45 |
| AutoProf `baseline` | 1773 | 1.850 | 6.73 | 0.46 | – |
| AutoProf `deep` | 1798 | 1.938 | 6.83 | 0.39 | 0.33 |

Cubic sampling lowers the median error by 18 % (1.60 → 1.31 %) instead of halving it as
on the ellipticals, and the 90th percentile does not move: 4.6 % for every Isoster arm,
5.6 to 6.8 % for the other tools. On this sample the error is dominated by what a single
ellipse cannot follow, the bar and the bulge-disk twist, not by how the ring is read. The
bilinear default already beats AutoProf here (54 %); cubic takes that to 80 %.

The error barely depends on depth (cubic: 1.26 % noiseless, 1.30 deep, 1.39 wide) and is
larger at z = 0.05 (1.72 %) than at z = 0.10 (0.97 %), where the bars are less resolved.
Both are signs of model error rather than noise. By component count on the noisy inputs,
the cubic median is 1.27 % for the nine two-component galaxies, 1.30 % for the 269
three-component galaxies and 1.61 % for the 22 four-component galaxies, whose mid zone is
at 4 % for every tool.

## By zone

Median truth RMS in percent, log-linear renderer; win fractions against AutoProf `baseline`
in parentheses.

| arm | inner (< 0.5 R_ref) | mid | outer (≥ 2 R_ref) |
|---|---:|---:|---:|
| isoster `ref_default` | 1.274 (0.46) | 1.702 (0.60) | 0.662 (0.82) |
| isoster `samp_cubic` | 0.678 (0.75) | 1.604 (0.72) | 0.725 (0.76) |
| isoster `samp_cubic_ea` | 0.661 (0.81) | 1.542 (0.79) | 0.703 (0.80) |
| isoster `samp_cubic_rms010` | 0.677 (0.75) | 1.590 (0.72) | 0.712 (0.79) |
| isoster `samp_cubic_snr200` | 0.682 (0.75) | 1.572 (0.72) | 0.687 (0.81) |
| isoster `samp_cubic_outer2re` | 0.679 (0.75) | 1.614 (0.72) | 0.654 (0.82) |
| photutils `baseline_median` | 1.372 (0.39) | 1.813 (0.45) | 1.023 (0.56) |
| AutoProf `baseline` | 1.316 | 1.746 | 1.096 |

The inner zone (bulge, nucleus and bar centre) gets the factor-two gain, as on the
ellipticals. The mid zone, where the bar lives, improves by 6 % only, and its 90th
percentile is 8 to 10 % for every tool. The outer zone is the exponential disk: cubic is
10 % worse than bilinear there, and the switches recover a third to a half of the loss;
the 2 R_ref oracle ties bilinear.

Outer-zone median RMS by image depth:

| depth | `ref_default` | `samp_cubic` | `samp_cubic_rms010` | `samp_cubic_snr200` | oracle 2 R_ref | AutoProf `baseline` |
|---|---:|---:|---:|---:|---:|---:|
| noiseless | 0.197 | 0.126 | 0.126 | 0.135 | 0.189 | 0.621 |
| deep | 0.631 | 0.704 | 0.694 | 0.678 | 0.624 | 1.085 |
| wide | 1.549 | 1.737 | 1.621 | 1.586 | 1.528 | 2.285 |

On noisy inputs the outer-zone win fraction over bilinear is 0.15 for plain cubic, 0.31 to
0.34 for the S/N switch and 0.49 to 0.53 for the oracle. The reason is the profile shape.
The outer disks are exponential with scale lengths of tens of pixels, so the bilinear bias
is small there (0.20 against 0.13 % on noise-free images) and the extra read noise of the
cubic kernel wins from 2 R_ref outward. The rings at 2 R_ref still have a median S/N of
1030 (deep) and 385 (wide) at z = 0.05, so the self-contained rules fire later than the
oracle (next section) and switch too late to protect the outer zone. This is the opposite
of the early-firing worry raised for faint galaxies: the threshold measures noise, and
what decides the kernel is noise against bilinear bias, which here means profile curvature
in pixel units. A criterion that weighs both is a later refinement. In the full aperture
the difference is immaterial: all cubic variants sit at 1.30 to 1.31 %.

## Harmonics matter on this sample

The native Isoster renderer paints the fitted a3, a4, b3, b4 harmonics into the model
(with linear radial interpolation). On Huang2013 it was slightly worse than the
harmonic-off renderers; on S4G it is much better in the inner and mid zones, because the
harmonics carry the bar.

| zone | renderer | `ref_default` | `samp_cubic` | `samp_cubic_ea` | `samp_cubic_snr200` | photutils `baseline_median` | AutoProf `baseline` |
|---|---|---:|---:|---:|---:|---:|---:|
| all | harmonics off, log-linear | 1.601 | 1.310 | 1.306 | 1.304 | 1.779 | 1.850 |
| all | native (harmonics on) | 1.264 | 0.800 | 0.752 | 0.781 | 1.923 | 1.743 |
| inner | harmonics off, log-linear | 1.274 | 0.678 | 0.661 | 0.682 | 1.372 | 1.316 |
| inner | native | 1.102 | 0.354 | 0.306 | 0.357 | 1.998 | 1.291 |
| mid | harmonics off, log-linear | 1.702 | 1.604 | 1.542 | 1.572 | 1.813 | 1.746 |
| mid | native | 1.216 | 0.969 | 0.896 | 0.973 | 1.271 | 1.788 |
| outer | harmonics off, log-linear | 0.662 | 0.725 | 0.703 | 0.687 | 1.023 | 1.096 |
| outer | native | 1.514 | 1.490 | 1.490 | 1.475 | 1.249 | 1.191 |

With harmonics on, the cubic Isoster arms reach 0.75 to 0.80 % over the full aperture
(0.31 to 0.36 % inner, 0.90 to 0.97 % mid) and `samp_cubic_ea` becomes the best arm; the
paired win fraction of `samp_cubic` over AutoProf's own model is 0.77 and over the
photutils model 0.82. The native renderer loses in the outer zone, but that loss is the
linear radial interpolation, not the harmonics: the harmonic-off linear renderer gives
1.44 % there against 1.51 % for native and 0.66 % for log-linear, because linear
interpolation between rings spaced at 10 % of the radius overshoots an exponential whose
scale length is a fraction of the radius. The renderer set has log-linear interpolation
or harmonics, never both. A log-linear harmonics-on renderer would be the right headline
for Isoster on disk galaxies and should be added before the S4G tables go into the paper.

## Regressions

Ratio of full-aperture RMS to `ref_default`, log-linear renderer:

| arm | 10 / 50 / 90 % quantiles | worse by > 20 % |
|---|---|---:|
| `samp_cubic` | 0.34 / 0.87 / 1.02 | 17 of 1800 (worst 1.45, ESO506-029 deep z = 0.05) |
| `samp_cubic_rms010` | 0.32 / 0.86 / 1.01 | 8 of 1800 (worst 1.56) |

Relative to plain cubic the switch arms are neutral at the median (ratio 1.000, 10 / 90 %
quantiles 0.97 / 1.01): `samp_cubic_rms010` is worse by more than 20 % in 11 inputs and
better by more than 20 % in 20; `samp_cubic_snr200` 15 and 14; the oracle 16 and 14. The
worst switch case is NGC1566 deep z = 0.10 with the S/N rule (2.0 times plain cubic, 1.5
times the default, switch at 1.8 R_ref).

## Where the switch fired

All 3600 refits of the switch arms reproduced the saved campaign profiles exactly. Switch
radius in units of R_ref (10 / 50 / 90 % quantiles) and the fraction of inputs where the
switch fired:

| arm | depth | fired | p10 | p50 | p90 |
|---|---|---:|---:|---:|---:|
| `samp_cubic_rms010` | noiseless | 0.08 | 0.7 | 1.3 | 4.1 |
| | deep | 0.94 | 2.0 | 3.5 | 4.6 |
| | wide | 1.00 | 1.4 | 2.8 | 3.8 |
| `samp_cubic_snr200` | noiseless | 0.28 | 0.3 | 0.6 | 1.3 |
| | deep | 0.95 | 0.5 | 2.9 | 4.5 |
| | wide | 1.00 | 0.5 | 2.2 | 3.6 |

On a noise-free image the rules can only fire on structure, and they do: the scatter rule
on 46 of 600 noiseless inputs, the S/N rule on 167, typically between 0.3 and 1.3 R_ref,
where the ring crosses the bar. It costs nothing: on those inputs the ratio to plain cubic
is 0.98 to 1.00 and no input is worse by 20 %. The S/N rule fired before 1 R_ref on 458
inputs in all; the median ratio to plain cubic is 0.997 and to the bilinear default 0.975,
with 5 inputs worse than cubic by more than 20 %. An early switch is benign because it
returns the outer rings to the default behaviour; the price of the switch on this sample
is the late firing described above, not early firing.

## Renderer dependence

Median full-aperture RMS in percent of the primary arms under each renderer:

| renderer | isoster `ref_default` | isoster `samp_cubic` | AutoProf `baseline` | photutils `baseline_median` |
|---|---:|---:|---:|---:|
| linear, harmonics off | 1.637 | 1.321 | 1.854 | 1.822 |
| cubic, harmonics off | 1.587 | 1.320 | 1.863 | 1.758 |
| log-linear, harmonics off | 1.601 | 1.310 | 1.850 | 1.779 |
| photutils spline painter | 2.151 | 1.495 | 1.909 | 2.270 |
| native model | 1.264 | 0.800 | 1.743 | 1.923 |

The Isoster ordering (cubic arms, then the bilinear default, then the other two tools)
holds under every renderer. AutoProf and photutils swap places between the harmonic-off
renderers and the native models, so their relative ranking on this sample is
renderer-dependent and should not be quoted without the renderer.

*Note added 2026-10-12.* Every "native" number in this record (the table above and the
zone table in "Harmonics matter on this sample") was rendered with isoster 1.1.0, whose
renderer applied `a3, b3` with the wrong sign outward of any > π/2 position-angle jump
(20 % of these fits; fixed in 1.1.1). Re-scoring the affected fits with the fixed renderer
changes the quoted medians by at most 0.003 percentage points (for example `samp_cubic`
inner 0.354 → 0.353, photutils outer 1.249 → 1.246): the bars are even structure and the
odd coefficients are small here. The harmonic-off rows are unaffected by construction. See
`2026-10-12-renderer-odd-harmonic-frame.md`.

## Integrator

Median integration is worse than the clipped mean again: `int_median` 1.663 against
1.601 % with bilinear reads, `samp_cubic_ea_median` 1.309 against 1.306 % with cubic reads
(beats the mean in 43 % of inputs). The mean stays.

## What changes relative to the Huang2013 conclusions

- The recommendation to make cubic sampling the default stands: 18 % lower median error on
  barred disks, 0 failures, 3 % in fit time, worse than bilinear by more than 20 % in 0.9 %
  of inputs.
- The self-contained switch stays opt-in. On disks it is neutral at the median and its
  outer-zone gain is partial because the rules fire beyond 2 R_ref; the oracle shows the
  gain available in the outer zone is modest anyway (0.65 against 0.73 %).
- `sampling_cubic_min_sma = 4` is bad on this sample too (1.56 %); keep the default of 0.
- The paper's S4G tables need a log-linear harmonics-on renderer. On disk galaxies the
  harmonics are signal, not noise, and the headline harmonic-off number understates
  Isoster's model by 40 %.

## Next

- Add the log-linear harmonics-on renderer to `round2_small_sample.py` and rescore both
  datasets with it (about 15 min for S4G, longer for Huang2013).
- Figures for the paper from the full campaigns, including the switch arms.
- Real-data spot check (SGA or HSC galaxies) before the default flip.
