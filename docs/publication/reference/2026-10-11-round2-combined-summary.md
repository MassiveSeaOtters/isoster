# Round-2 benchmark summary: Huang2013 and S4G (2026-10-11)

This is the main publication record of the round-2 benchmark. It puts the two full
campaigns side by side and covers every metric the campaign produces, so the paper's
tables and claims can be built from one document. The per-dataset reports
([Huang2013](2026-10-10-round2-huang2013-full-campaign.md),
[S4G](2026-10-11-round2-s4g-full-campaign.md)) hold the details and the switch-radius
studies; the [gap review](2026-10-09-isoster-autoprof-gap-review.md) holds the diagnosis
that led to cubic sampling. Every round-1 result (PSF-free mocks) is superseded.

Every number below is copied from
`/Volumes/galaxy/isophote/publication_single_band_round2_2026_10_10/analysis/combined_2026_10_11/combined_summary.md`,
written by `benchmarks/exhausted/analysis/round2_combined_summary.py` from the two
`fit_metrics.csv` files (`round2_small_sample.py`, commit `971b9ed`) and the two
`switch_radius.csv` files (`round2_switch_radius.py`). Arms not listed in a table are
summarised in words from the per-dataset `summary.md` files.

## The two campaigns

| | Huang2013 | S4G |
|---|---|---|
| galaxies | 93 single-Sersic ellipticals (Huang et al. 2013 fits) | 300 late-type disks: Sersic bulge and disk, Ferrer bar in 260, unresolved nucleus in 130 (Salo et al. 2015) |
| what is tested | reading an ellipse correctly | fitting ellipses to isophotes that twist and are not ellipses |
| scenarios | noiseless at z = 0.05; wide and deep at z = 0.05, 0.20, 0.35, 0.50 | noiseless, wide and deep at z = 0.05 and 0.10 |
| inputs | 837 | 1800 |
| images | HSC i-band pixel scale and depths, 0.7 arcsec Gaussian PSF, generator commit `8632e1f`; every truth image regenerated and matched exactly | same |
| arms | 25: Isoster 17, photutils 4, AutoProf 4 (zero background, corrected PA) | same |
| fits | 20,925 | 45,000 |
| wall time | 3.9 h | 6.5 h |
| headline renderer | log-linear, harmonics off (`log_off`) | same |

Zones: inner (< 0.5 R_ref), mid, outer (≥ 2 R_ref) on one common pixel support per input
with the 2-pixel inner cut. Win fractions are paired: an arm is scored only on inputs where
both it and the reference produced a model.

## Conclusions, metric by metric

**Reliability.** Isoster did not fail once: 0 failures in 44,829 fits over 17 arms. Photutils
failed 349 times, almost all its median-mode arms on noiseless images (a NaN in the median
estimator) plus 900 s timeouts. AutoProf returned an error 134 times on 20 galaxies, 19 of
which fail in at least three of its four arms; the `deep` arm fails on 3 inputs in all.

**Speed.** Isoster fits a galaxy in 0.065 s at the median on both samples. Cubic sampling
costs 1 to 3 % of that, the kernel switch 1 to 4 %, the median integrator 5 to 17 %.
Photutils is 100 to 137 times slower, AutoProf 116 to 152 times.

| arm | median [s], Huang2013 | median [s], S4G | relative to Isoster default |
|---|---:|---:|---:|
| isoster `ref_default` | 0.065 | 0.066 | 1.00 |
| isoster `samp_cubic` | 0.066 | 0.068 | 1.01 / 1.03 |
| isoster `samp_cubic_rms010` | 0.065 | 0.069 | 1.01 / 1.04 |
| photutils `baseline_median` | 6.6 | 8.4 | 102 / 126 |
| AutoProf `baseline` | 9.7 | 7.7 | 150 / 116 |

**Accuracy: truth RMS.** Cubic ring sampling halves the full-aperture error on the
ellipticals (1.26 → 0.65 %) and lowers it by 18 % on the disks (1.60 → 1.31 %). The inner
zone, where the bilinear read bias lives, improves by a factor 4.6 and 1.9; the mid zone by
35 % and 6 %; the outer zone gets 7 % and 10 % worse because the cubic read is noisier,
and the self-contained switch repairs that on the ellipticals and partly on the disks. The
90th percentile falls from 2.24 to 1.95 % on the ellipticals and stays at 4.6 % on the
disks, where the bar and the bulge-disk twist set the tail for every tool.

Median truth RMS in percent (90th percentile for the full aperture in parentheses):

| arm | all, Huang2013 | inner | mid | outer | all, S4G | inner | mid | outer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| isoster `ref_default` (bilinear) | 1.265 (2.24) | 1.097 | 1.337 | 3.574 | 1.601 (4.58) | 1.274 | 1.702 | 0.662 |
| isoster `samp_cubic` | 0.647 (1.95) | 0.239 | 0.868 | 3.841 | 1.310 (4.62) | 0.678 | 1.604 | 0.725 |
| isoster `samp_cubic_ea` | 0.639 (1.88) | 0.238 | 0.855 | 3.720 | 1.306 (5.24) | 0.661 | 1.542 | 0.703 |
| isoster `samp_cubic_rms010` | 0.614 (1.80) | 0.238 | 0.864 | 3.522 | 1.304 (4.61) | 0.677 | 1.590 | 0.712 |
| isoster `samp_cubic_snr200` | 0.613 (1.82) | 0.238 | 0.858 | 3.495 | 1.304 (4.55) | 0.682 | 1.572 | 0.687 |
| isoster `samp_cubic_outer2re` (oracle) | 0.618 (1.89) | 0.238 | 0.870 | 3.521 | 1.306 (4.64) | 0.679 | 1.614 | 0.654 |
| isoster `samp_cubic_ea_min4` | 1.122 (2.18) | 1.013 | 1.026 | 3.720 | 1.560 (5.28) | 1.170 | 1.540 | 0.703 |
| isoster `int_median` | 1.323 (2.30) | 1.169 | 1.375 | 3.616 | 1.663 (4.61) | 1.336 | 1.745 | 0.677 |
| photutils `baseline_median` | 1.384 (2.37) | 1.176 | 1.414 | 4.044 | 1.779 (5.89) | 1.372 | 1.813 | 1.023 |
| photutils `default_bilinear` | 1.350 (2.36) | 1.156 | 1.354 | 3.797 | 1.839 (5.63) | 1.381 | 1.762 | 0.719 |
| AutoProf `baseline` | 1.153 (2.39) | 0.780 | 1.066 | 3.952 | 1.850 (6.73) | 1.316 | 1.746 | 1.096 |
| AutoProf `deep` | 1.187 (2.54) | 0.815 | 1.158 | 4.141 | 1.938 (6.83) | 1.309 | 1.813 | 1.193 |

The other Isoster arms, the geometry and harmonic afterburners (`geom_ea`, `geom_simul`,
`geom_simul_ea`, `harm_simul_ea`), `lsb_autolock`, `stack_all` and `reg_outer_damp`, all
read bilinearly and sit within 6 % of the default on both samples (1.25 to 1.34 % and 1.59
to 1.70 %). Outer damping on top of cubic + EA costs accuracy (0.739 against 0.639 %, 1.377
against 1.306 %). AutoProf's `high_regularization` is its best arm on the ellipticals
(1.122 %) and its worst on the disks (1.957 %). Photutils `aggressive_clip` fails on the
disks (5.17 %).

**Accuracy against the other tools.** With the bilinear default Isoster beats AutoProf in
31 % of elliptical inputs and 54 % of disk inputs. With cubic sampling it beats AutoProf in
80 % of both, and the switch arms in 82 to 84 %. Photutils `baseline_median` beats plain
cubic in 5 % (ellipticals) and 8 % (disks) of inputs. In the inner zone cubic Isoster beats
AutoProf in 90 % and 75 %; in the outer zone the bilinear default already beats AutoProf in
59 % and 82 %.

**Flux bias.** The bilinear default overestimates the flux of the ellipticals by 0.17 %
at the median (0.22 % absolute); cubic brings that to 0.06 % (0.15 %). The inner-zone
absolute bias falls from 0.25 to 0.07 % on the ellipticals and from 0.28 to 0.08 % on the
disks. On the disks the signed bias of cubic is −0.08 % (0.12 % absolute). Photutils
`baseline_median` has the largest bias of any arm, +0.45 % on the ellipticals with +0.95 %
in the outer zone; AutoProf is unbiased on the ellipticals (+0.02 %) and −0.17 % on the
disks. The outer-zone absolute bias is 0.5 to 0.6 % for every Isoster and AutoProf arm on
the ellipticals, set by noise.

**Data residual in noise units.** The RMS of model minus noisy image divided by the injected
pixel noise, over noisy inputs. One means the model is as close to the data as the truth
is; nothing above the noise is left. On the ellipticals the cubic arms are at 1.00 in every
zone (inner 0.97), while the bilinear default is at 1.04 with 3.4 in the inner zone, the
read bias standing 3.4 sigma above the noise per pixel. On the disks every tool is above the
noise: cubic Isoster 1.55 (inner 5.1, mid 2.0, outer 1.00), the bilinear default 2.08
(inner 12.6), AutoProf 2.16, photutils 2.28 to 2.30. What remains on the disks is the
ellipse model's failure on the bar, not the ring read.

**Geometry: centre and convergence.** Isoster's rings sit on the true centre to 0.02 px
(ellipticals) and 0.002 px (disks) at the median; the largest offset among the rings of a
fit is 1.3 px and 0.4 px at the median (4.5 and 2.7 px at the 90th percentile), from the
outermost noisy rings. Cubic sampling leaves the geometry unchanged (1.30 against 1.38 px).
Photutils is similar (0.9 to 1.0 px). AutoProf holds one centre for every ring and reports
the true centre to 0.01 px; the mocks are placed on the image centre, so this is a trivial
target. Isoster converges (stop code 0) on 98 % and 100 % of its rings at the median,
photutils on 84 to 88 % and 93 %. Isoster and photutils fit 53 to 55 rings per galaxy,
AutoProf `baseline` 49 to 52 and `deep` 96 to 101.

**Robustness: regressions against the default.** Ratio of full-aperture RMS to the bilinear
default per input, and the count of inputs worse by more than 20 %:

| arm | Huang2013 10 / 50 / 90 % | worse > 20 % | S4G 10 / 50 / 90 % | worse > 20 % |
|---|---|---:|---|---:|
| `samp_cubic` | 0.22 / 0.54 / 0.91 | 3 of 837 (worst 1.28) | 0.34 / 0.87 / 1.02 | 17 of 1800 (worst 1.45) |
| `samp_cubic_ea` | 0.22 / 0.54 / 0.90 | 3 (worst 1.33) | 0.33 / 0.84 / 1.18 | 155 (worst 1.76) |
| `samp_cubic_rms010` | 0.21 / 0.52 / 0.86 | 0 (worst 1.10) | 0.32 / 0.86 / 1.01 | 8 (worst 1.56) |
| `samp_cubic_snr200` | 0.21 / 0.52 / 0.86 | 0 (worst 1.10) | 0.32 / 0.87 / 1.00 | 10 (worst 1.56) |
| `samp_cubic_outer2re` | 0.21 / 0.53 / 0.89 | 1 (worst 1.29) | 0.32 / 0.86 / 1.01 | 12 (worst 1.85) |

Plain cubic regresses in 0.4 % and 0.9 % of inputs. Cubic with eccentric-anomaly sampling
(EA, `use_eccentric_anomaly`) is the best arm at the median on both samples but has a
tail on the disks: worse than the default by more than 20 % in 155 inputs and worse than
plain cubic by more than 20 % in 152 (8 of 837 on the ellipticals). The tail is in the
inner and mid zones, 121 of the 155 at z = 0.05 and 49 on noiseless images, so it is driven
by bar structure, not noise. EA beats plain cubic in 61 % of elliptical and 47 % of disk
inputs. Plain cubic, not cubic + EA, is therefore the default candidate; this supersedes the
small-sample recommendation. The inner bilinear fallback below 4 px (`samp_cubic_ea_min4`)
gives most of the gain away on both samples.

**Depth and redshift.** On the ellipticals the error grows with redshift and shallowness
and the cubic gain is largest where the data are good: at z = 0.05 the default's 0.77 %
becomes 0.23 %, at z = 0.5 1.91 % becomes 1.56 %. On the disks the error does not depend on
depth (cubic 1.26 % noiseless, 1.30 deep, 1.39 wide) and is larger at z = 0.05 (1.72 %)
than at z = 0.10 (0.97 %), where the bars are less resolved. Outer-zone medians by depth:

| depth | Huang2013 default | cubic | rms010 | snr200 | oracle | S4G default | cubic | rms010 | snr200 | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noiseless | 0.361 | 0.358 | 0.358 | 0.358 | 0.361 | 0.197 | 0.126 | 0.126 | 0.135 | 0.189 |
| deep | 3.037 | 3.149 | 2.979 | 2.959 | 3.038 | 0.631 | 0.704 | 0.694 | 0.678 | 0.624 |
| wide | 5.868 | 6.308 | 5.865 | 5.858 | 5.870 | 1.549 | 1.737 | 1.621 | 1.586 | 1.528 |

**Renderer dependence.** The order of the Isoster arms (switch arms, plain cubic, bilinear
default) is the same under every renderer on both samples, and so is Isoster-above-the-others.
The order of AutoProf and photutils flips on the disks between the harmonic-off renderers and
the native models, so that comparison must always quote the renderer. The native Isoster
renderer (harmonics on, linear radial interpolation) is slightly worse than the headline on
the ellipticals (0.70 against 0.65 %) and much better on the disks (0.80 against 1.31 %;
inner 0.35 against 0.68, mid 0.97 against 1.60): the a3, a4, b3, b4 harmonics are noise on
ellipticals and signal on bars. Its outer-zone loss on the disks (1.49 against 0.73 %) comes
from the linear interpolation, not from the harmonics. No renderer in the set combines
log-linear interpolation with harmonics; one should be added before the S4G tables go into
the paper.

**Kernel switch.** Both self-contained rules fire on 99 to 100 % of noisy elliptical inputs
and 94 to 100 % of noisy disk inputs, and every one of the 5,274 refits reproduced the saved
campaign profile exactly. Where they fire, in units of R_ref (median): scatter rule 2.8
(deep) and 1.9 (wide) on the ellipticals, 3.5 and 2.8 on the disks; S/N rule 2.3 and 1.5,
2.9 and 2.2. The scatter rule never fires on a noise-free elliptical; on noise-free disks it
fires on 8 % of inputs and the S/N rule on 28 %, on bar structure between 0.3 and 1.3 R_ref,
at no cost. On the ellipticals the switch arms beat the 2 R_ref oracle in the outer zone; on
the disks the oracle wins (0.654 against 0.687 %) because the exponential outer disk has
little bilinear bias and the rules fire too late there. Ring S/N at 2 R_ref (deep / wide,
z = 0.05): 650 / 270 on the ellipticals, 1030 / 385 on the disks.

**Integrator.** Median integration is worse than the clipped mean on both samples and with
both kernels: 1.32 against 1.26 % and 1.66 against 1.60 % with bilinear reads; 0.655
against 0.639 % and 1.309 against 1.307 % with cubic + EA reads, where the median beats the
mean in 37 % and 36 % of inputs. The mean stays.

## What this means for the paper and the code

1. Cubic ring sampling should become the default. It halves the error on ellipticals,
   lowers it by 18 % on barred disks, never fails, costs 1 to 3 % in time, leaves the
   geometry unchanged and regresses by more than 20 % in under 1 % of inputs on both
   samples. Plain cubic, without eccentric-anomaly sampling.
2. Eccentric-anomaly sampling stays opt-in until its tail on barred isophotes is understood.
3. The self-contained kernel switch stays opt-in and documented. It is a free improvement on
   the ellipticals (no input worse than the default by more than 10 %) and neutral on the
   disks; its remaining weakness is late firing on high-S/N exponential disks, and the right
   criterion would weigh bilinear bias against noise rather than noise alone.
4. The inner bilinear fallback (`sampling_cubic_min_sma`) must stay at 0 on PSF-convolved
   images; it is a knob for unconvolved cusps only.
5. The clipped mean remains the integrator.
6. The headline renderer for the paper's S4G tables should interpolate log-linearly and
   include the harmonics; the current set has one or the other.
7. Before the default flip: a real-data spot check on a few SGA or HSC galaxies.
