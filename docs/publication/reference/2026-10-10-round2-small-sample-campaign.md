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
- **One caveat: the noise-dominated outer zone.** Cubic reads are noisier than a
  bilinear average, and outer-zone RMS rises from 3.88 % to 4.06 % (cubic beats the
  default there in only 36 % of inputs; AutoProf is at 4.57 %). The outer-damping arm
  makes this worse (6.16 %), so it is not the remedy. The cause and a tested remedy are
  in the follow-up section below.

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
4. Outer zone: the cubic kernel's extra noise is understood and measured (follow-up
   below). A switch to bilinear reads once the ring's own scatter says it is
   noise-dominated (`sampling_cubic_max_relative_rms = 0.1`, arm `samp_cubic_rms010`)
   recovers it with no size prior and matches the 2 Re oracle; carry it, the
   signal-to-noise variant (`samp_cubic_snr200`) and the oracle (`samp_cubic_outer2re`)
   in the full campaign, and decide the default from that run.

## Follow-up: the integrator and the outer-zone penalty (2026-10-10)

Two questions had to be settled before changing a default: is the `mean` integrator the
right choice in both the inner and the outer zone, and why does cubic sampling lose in
the outer zone. Three measurements answer them; outputs under
`.../analysis/{outer_zone_decomposition,kernel_switch_refit}_2026_10_10` and
`outputs/benchmark_exhausted/ring_noise_response`.

**Noise response of a ring (pure white noise, 600 realizations;
`benchmarks/exhausted/analysis/ring_noise_response.py`).** A bilinear read averages the
four surrounding pixels, so a sample carries 0.66 sigma of noise; an interpolating cubic
spline passes through the pixel values and a sample carries 0.87 sigma. Averaged along the
ring, the cubic ring mean is 1.20× noisier than the bilinear one at every radius from 5
to 150 px, and the four harmonic coefficients that drive the centre and shape updates are
1.18–1.23× noisier. Relative to the same number of independent pixels, bilinear is at
0.85–0.93 (it smooths a little) and cubic at 1.01–1.13 (no smoothing). Integrators on the
same rings: the one-pass 3-sigma clipped mean (the actual Isoster default, `nclip=1`)
costs < 1 % over the plain mean, the 10 % trimmed mean 1–2 %, the biweight location
2–3 %, and the median 5–15 % (less than the textbook 25 % because neighbouring samples
are correlated).

**Where the outer-zone error comes from
(`benchmarks/exhausted/analysis/round2_outer_zone_decomposition.py`).** Every Isoster
arm of the 160 noisy inputs was refitted on the noiseless truth image with its own saved
configuration, and the saved isophotes were compared one by one with that noise-free fit
and with the truth averaged along the saved ellipse.

| Outer zone, medians over noisy inputs | ref_default (bilinear) | samp_cubic | samp_cubic_ea_median |
|---|---:|---:|---:|
| truth RMS of the saved fit (`full`) | 4.68 % | 4.95 % | 5.06 % |
| truth RMS of the noise-free fit (`noise_free`) | 1.40 % | 1.11 % | 1.11 % |
| intensity noise along the saved ellipse | 2.05 % | 2.43 % | 2.76 % |
| ellipticity scatter vs noise-free fit | 0.0106 | 0.0118 | 0.0110 |
| position-angle scatter vs noise-free fit | 1.60° | 1.85° | 1.56° |
| centre scatter vs noise-free fit | 0.50 px | 0.48 px | 0.48 px |
| rings held by the gradient check (`stop_code = -1`) | 15.5 % | 20.1 % | 27.7 % |

Without noise, cubic is the better kernel in the outer zone too (1.11 % against 1.40 %),
so the penalty is entirely noise. Per ring the cubic intensity is 18 % noisier and the
position angle 16 % noisier, as the white-noise test predicts, and the noisier gradient
trips the `maxgerr` check more often, which freezes the geometry of more outer rings. The
penalty is largest in the shallow `wide` images (7.19 → 8.24 %) and small in the `deep`
ones (3.36 → 3.45 %). The same table shows what bilinear costs inside: the bilinear
intensity sits 0.49 % off the truth along its own ellipse in the inner zone and 0.51 % in
the mid zone, where the cubic read is off by 0.02 % and 0.21 %; that offset is the read
bias that produced the gap, not noise.

**Integrator.** On these mocks the mean is the right statistic in every zone, but the
margin is small: with bilinear reads the median arm is worse everywhere (all 1.35 vs
1.29 %, inner 1.15 vs 1.08 %, mid 1.47 vs 1.43 %, outer 3.92 vs 3.88 %; better than the
default in 7 % of inputs); with cubic + EA the median is a wash (0.65 vs 0.67 %, inner
0.285 vs 0.277 %, mid 0.953 vs 0.952 %, outer 3.99 vs 4.05 %), and per ring its outer
intensity is 20 % noisier. The mocks carry no unmasked stars or neighbours, so they cannot
measure what the median is for, robustness on real images; that must be tested on the
SGA sample or on contaminant-injected mocks. The available integrators are the clipped
mean (default), `median`, `adaptive` (mean inside `lsb_sma_threshold`, median beyond) and
the LSB auto-lock switch; a biweight or trimmed mean would give most of the median's
robustness at a 1–3 % noise cost and is the natural candidate if the real-image tests
show the clipped mean failing.

**Remedy for the outer zone
(`benchmarks/exhausted/analysis/round2_kernel_switch_refit.py`).** New option
`sampling_cubic_max_sma`: rings beyond it return to bilinear reads. All 180 inputs were
refitted from the saved `samp_cubic` configuration (the `bilinear` and `cubic` variants
reproduce the saved `ref_default` and `samp_cubic` profiles exactly, 180/180). Medians,
linear renderer:

| switch radius | all | inner | mid | outer | outer: beats cubic | outer: beats bilinear | fit time / bilinear |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (bilinear) | 1.289 | 1.082 | 1.427 | 3.924 | 64 % | — | 1.00 |
| none (cubic) | 0.663 | 0.280 | 0.970 | 4.227 | — | 36 % | 1.05 |
| 3 PSF FWHM (12.5 px) | 0.640 | 0.282 | 0.969 | 3.926 | 67 % | 52 % | 1.02 |
| 5 PSF FWHM (20.8 px) | 0.631 | 0.279 | 0.956 | 4.004 | 66 % | 49 % | 1.01 |
| 10 PSF FWHM (41.7 px) | 0.658 | 0.279 | 0.958 | 3.976 | 45 % | 42 % | 1.02 |
| 1 Re | 0.716 | 0.281 | 1.037 | 3.924 | 66 % | 40 % | 1.01 |
| 2 Re | 0.630 | 0.280 | 0.966 | 3.797 | 72 % | 62 % | 1.01 |
| 4 Re | 0.651 | 0.279 | 0.966 | 3.929 | 64 % | 40 % | 1.01 |

Switching at 2 Re keeps the inner and mid gains untouched, turns the outer-zone loss into
a gain (3.80 % against 3.92 % bilinear and 4.23 % cubic; the `wide` images go from 8.25 %
to 6.70 %), and brings the fit time back to +1 %. Switching at 1 Re is too early (bias
returns in the mid zone, worst input 2.9× cubic there); PSF-based radii help the small
high-redshift images but clip the gain of the large ones. The right switch radius is a
galaxy size, which is unknown before the fit on a real image, so the radius itself cannot
be the default rule.

**Self-contained switch (same script, second run,
`.../analysis/kernel_switch_snr_refit_2026_10_10`).** The ring already carries the
information: during outward growth, once two consecutive rings look noise-dominated, all
remaining outward rings are read bilinearly (a one-way latch like the LSB auto-lock; the
inward rings and the switch radius are reported in the result). Two triggers were tested
on the same 180 inputs: the relative scatter of the samples along the ring, `rms / intens`
(`sampling_cubic_max_relative_rms`), and the signal-to-noise of the ring mean,
`intens / intens_err` (`sampling_cubic_min_snr`). The two are related by the sample count,
`rms / intens = sqrt(n) × intens_err / intens`, so the scatter rule is a signal-to-noise
rule whose threshold grows with ring size, which follows the bias–variance trade-off
(the bilinear bias shrinks with radius while the ring mean gets more samples).

| trigger | all: ratio to cubic (beats cubic) | mid: worst input | outer: ratio to cubic (beats cubic / bilinear) | switch radius / Re, median (10–90 %) | switched | fit time / bilinear |
|---|---:|---:|---:|---:|---:|---:|
| none (cubic) | 1.000 | 1.00 | 1.000 (— / 36 %) | — | — | 1.07 |
| oracle 2 Re | 0.996 (64 %) | 1.05 | 0.943 (71 % / 61 %) | 2.0 | 100 % | 1.04 |
| `rms / intens > 0.1` | 0.993 (63 %) | 1.05 | 0.962 (68 % / 54 %) | 2.2 (1.3–3.6) | 88 % | 1.01 |
| `rms / intens > 0.2` | 0.999 (56 %) | 1.07 | 0.990 (57 % / 43 %) | 2.9 (1.7–4.3) | 87 % | 1.02 |
| `S/N < 200` | 0.993 (63 %) | 1.05 | 0.968 (63 % / 50 %) | 1.7 (0.9–3.6) | 89 % | 1.02 |
| `S/N < 100` | 0.997 (57 %) | 1.05 | 0.977 (62 % / 46 %) | 2.5 (1.3–4.6) | 87 % | 1.02 |
| `S/N < 50` | 1.000 (52 %) | 1.07 | 0.996 (53 % / 42 %) | 3.2 (1.8–5.1) | 81 % | 1.04 |

Inner-zone results are identical for every row (0.279–0.281 %). The scatter rule at 0.1
(per-sample signal-to-noise below 10) is the best self-contained trigger and sits between
the oracle and the plain cubic fit in the outer zone (median RMS 3.81 % against 3.77 %
oracle, 3.91 % bilinear, 4.22 % cubic); the signal-to-noise rule at 200 is equivalent
within the noise of 180 inputs. Both fire later on deep images (2.6 and 2.2 Re) than on
shallow ones (1.8 and 1.5 Re), which is the intended behaviour: the switch follows the
noise, not the galaxy. When the signal-to-noise rule fires inside 1 Re (10 % of inputs,
all shallow and small) the mid zone is not hurt (worst input 1.05× cubic), because a
ring whose mean is that noisy gains nothing from an unbiased read. Neither trigger ever
fires on the noiseless images (except `S/N < 200` on 2 of 20, through the numerical
scatter of the samples). Arms `samp_cubic_rms010` and `samp_cubic_snr200` carry the two
triggers into the full campaign; `samp_cubic_outer2re` stays as the oracle reference.
