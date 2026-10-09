# Round-2 small-sample campaign: does cubic ring sampling close the gap?

Three tools, 13 arms, 20 Huang2013 galaxies, all nine scenarios, on the PSF-convolved
round-2 mocks. Branch `review/isoster-autoprof-gap-20261009`; campaign
`/Volumes/galaxy/isophote/publication_single_band_round2_2026_10_10/fits/round2_small_sample_2026_10_10`,
analysis `.../analysis/small_sample_2026_10_10` (`summary.md`, `fit_metrics.csv`, two figures).

## Verdict

- **Cubic ring sampling closes the gap and reverses it.** Full-aperture median truth RMS
  (Isoster's linear harmonic-off renderer, the published baseline mode): Isoster default
  1.286 % → cubic 0.666 % (−48 %); AutoProf baseline 1.142 %; Photutils 1.416 %. Cubic
  beats the Isoster default in 95 % of the 180 inputs and AutoProf in 86 %. Inner zone:
  1.08 % → 0.28 % (AutoProf 0.72 %), beating AutoProf in 94 % of inputs. Every one of the
  20 galaxies improves (per-galaxy median ratio 0.31–0.86).
- **No new failure mode.** All 1 620 Isoster fits succeeded. The 24 failures are AutoProf's
  known small-image crash (8 inputs × 2 arms) and Photutils' known noiseless NaN failure
  (7) plus one 900 s timeout. No input gets worse than 1.10× the default, and those few
  are the smallest, noisiest images (`wide_z035`, `wide_z050`).
- **Flux bias.** With the linear renderer the median absolute bias falls from 0.53 % to
  0.42 % (AutoProf 0.41 %). Rendering with cubic radial interpolation removes the rest:
  Isoster cubic −0.007 % signed / 0.17 % absolute, AutoProf −0.03 % / 0.18 %. The
  remaining bias of the linear renderer is the renderer's, not the fit's.
- **Cost.** Median fit time ratio cubic / default 1.04 (90th percentile 1.17); cubic + EA
  1.06 (1.20). Isoster 0.06 s per fit versus AutoProf 9.6 s and Photutils 6.9 s.
- **Eccentric-anomaly sampling adds nothing measurable** on these galaxies (cubic 0.666 %
  vs cubic + EA 0.672 %; EA alone 1.290 % vs 1.286 %). The median statistic adds nothing.
  The inner bilinear fallback (`samp_cubic_ea_min4`) throws most of the gain away
  (1.116 %) and should not be used on PSF-limited images.
- **One caveat: the noise-dominated outer zone.** Cubic reads are slightly noisier than a
  bilinear average, and outer-zone RMS rises from 3.88 % to 4.06 % (cubic beats the
  default there in only 36 % of inputs; AutoProf is at 4.57 %). The outer-damping arm
  makes this worse (6.16 %), so it is not the remedy.

## Sample and arms

Galaxies: the 15 drawn at random for the earlier population test (seed 20261009) plus 5
drawn at random from the 20 galaxies with R_ref > 60 px (seed 20261010): NGC1172,
NGC1379, NGC1404, NGC1549, NGC2305, NGC2325, NGC3091, NGC3904, NGC4696, NGC5898,
NGC6851, NGC6909, NGC7029, NGC7144, NGC7145, ESO185-G054, IC4296, IC4329, NGC1395,
NGC4786. R_ref quantiles 10/50/90: 21/34/106 px (all 93 galaxies: 20/40/98 px);
components 2/3/4: 1/14/5 galaxies (all: 5/70/18). Scenarios: noiseless z = 0.05 plus
wide/deep at z = 0.05, 0.20, 0.35, 0.50 (180 inputs). None of the five earlier
"AutoProf-advantage" cases was chosen deliberately; ESO185-G054 entered the sample by chance.

| Tool | Arms |
|---|---|
| Isoster | `ref_default` (bilinear, polar, mean); `geom_ea`; `int_median`; `samp_cubic`; `samp_cubic_ea`; `samp_cubic_ea_min4`; `samp_cubic_ea_median`; `samp_cubic_ea_outer_damp` |
| Photutils | `baseline_median`; `default_bilinear` |
| AutoProf | `baseline`, `deep`, `fix_center`, all with the prepared zero background fixed (`autoprof_arms_zero_background.yaml`) and the corrected initial PA |

2 340 fits in 34 min (8 inputs in parallel). Configs:
`benchmarks/exhausted/configs/campaign.round2_small_sample_2026_10_10.yaml` (one-galaxy
gate variant alongside). Truth images were regenerated from the frozen generator
`8632e1f` (kept at `.../round2.../source/isophote_test_8632e1f`) and every campaign image
was reproduced exactly, noise seed included. Evaluation: fixed 2-pixel elliptical inner
cut, one matched pixel support per input across all arms and renderers, zones
inner < 0.5 R_ref, mid, outer ≥ 2 R_ref. Analysis code:
`benchmarks/exhausted/analysis/round2_small_sample.py` and `round2_small_sample_figures.py`.

## Results (medians over inputs; truth RMS and signed flux bias in percent)

Linear harmonic-off renderer (Isoster's; applied to every tool):

| Arm | all RMS | inner | mid | outer | all bias | beats default | beats AutoProf |
|---|---:|---:|---:|---:|---:|---:|---:|
| isoster ref_default | 1.286 | 1.082 | 1.427 | 3.878 | +0.51 | — | 0.31 |
| isoster geom_ea | 1.290 | 1.088 | 1.396 | 3.825 | +0.55 | 0.61 | 0.31 |
| isoster samp_cubic | 0.666 | 0.280 | 0.970 | 4.057 | +0.40 | 0.95 | 0.84 |
| isoster samp_cubic_ea | 0.672 | 0.277 | 0.952 | 4.052 | +0.39 | 0.95 | 0.86 |
| isoster samp_cubic_ea_min4 | 1.116 | 1.010 | — | — | +0.31 | 0.86 | 0.38 |
| photutils baseline_median | 1.416 | 1.192 | 1.519 | 4.306 | +0.78 | 0.20 | 0.26 |
| autoprof baseline | 1.142 | 0.722 | 1.209 | 4.565 | +0.39 | 0.69 | — |
| autoprof deep | 1.147 | 0.729 | 1.153 | 4.323 | +0.08 | 0.67 | 0.52 |

Cubic harmonic-off renderer (the fair counterpart of AutoProf's spline-in-magnitude model):

| Arm | all RMS | inner | outer | all bias (signed / abs) |
|---|---:|---:|---:|---:|
| isoster ref_default | 1.180 | 1.048 | 3.807 | +0.08 / 0.20 |
| isoster samp_cubic | 0.634 | 0.246 | 4.215 | −0.01 / 0.17 |
| isoster samp_cubic_ea | 0.653 | 0.244 | 4.243 | −0.01 / 0.16 |
| autoprof baseline | 1.143 | 0.730 | 4.778 | −0.03 / 0.18 |

Native models (Isoster harmonics on, linear; AutoProf `genmodel`; Photutils additive
harmonics): Isoster default 1.330 %, cubic 0.718 %, AutoProf 1.175 %, Photutils 2.24 %.

By scenario, full RMS with the linear renderer (default → cubic + EA; AutoProf):
noiseless 0.77 → 0.29 (0.49); deep_z005 0.77 → 0.29 (0.58); deep_z020 1.19 → 0.42 (0.98);
deep_z035 1.42 → 0.72 (1.31); deep_z050 1.58 → 0.95 (1.39); wide_z005 0.78 → 0.31 (0.68);
wide_z020 1.31 → 0.74 (1.14); wide_z035 1.78 → 1.36 (1.52); wide_z050 2.45 → 2.15 (2.41).

## Why this works (fitting and rendering)

The fit: bilinear reads bias the second harmonic in a grid-phased way, so the inner
ellipticity and position angle settle in the wrong place; cubic reads remove that, and
the inner-zone RMS drops fourfold. On PSF-limited cores there is no ringing, so the
inner fallback is unnecessary. The rendering: linear-in-intensity interpolation between
isophotes overestimates a convex profile by a few tenths of a percent regardless of the
fit; cubic (or log-linear) radial interpolation removes it, which is why AutoProf's
native model looked unbiased. Both halves are needed to compare tools fairly.

## Recommendations

1. Make `sampling_interpolation="cubic"` the Isoster default. Keep `use_eccentric_anomaly`
   optional (no measured gain here; harmless at +2 % time). Drop `sampling_cubic_min_sma`
   from any default. Leave `integrator="mean"`.
2. Report Isoster's model images with cubic (or log-linear) radial interpolation in the
   benchmark, and keep the linear renderer only as the historical control.
3. Next campaign on round 2 (all 93 Huang2013 galaxies, then S4G): Isoster `ref_default`
   (bilinear control), `samp_cubic` (candidate default) and `samp_cubic_ea`; Photutils
   `baseline_median`; AutoProf `baseline` and `deep` with zero background. Then rerun the
   standardized two-pixel analysis with both renderers.
4. Follow-up for the outer zone: the cubic kernel's extra noise sensitivity in the
   outskirts (+5 % outer RMS) is the one remaining regression; test a radius- or
   signal-to-noise-dependent switch back to bilinear, or noise-aware outer regularization,
   before the S4G run. Not a blocker for the Huang2013 campaign.
