# Why Isoster trails AutoProf on the Huang2013 mocks: code review and refit tests

Branch `review/isoster-autoprof-gap-20261009` (worktree `~/code/isoster-gap-review`,
from `main` at `9597a87`). Nothing was merged or pushed; no campaign data was changed.

## Summary

1. **The main cause is bilinear image sampling inside the geometry fit.** Isoster reads
   each ring by bilinear interpolation. Between pixel centres that overestimates a steep,
   convex profile, and because the error depends on where each sample falls relative to
   the pixel grid, it leaks into the second harmonic that sets ellipticity and position
   angle. The innermost ellipticity profile therefore wobbles (NGC5328: 0.24 → 0.35 →
   0.28 → 0.36 between 0.5 and 3.4 px), and that wrong inner geometry, not the
   intensity read, carries most of the inner-zone error. Earlier work replaced the
   intensities on the *saved* ellipses and found no gain; the gain only appears when
   the geometry is refitted with a better kernel.
2. **Refitting with cubic-spline sampling plus eccentric-anomaly sampling closes the gap
   on resolved cores and removes the flux bias.** On the five "AutoProf-advantage"
   special cases, full-aperture truth RMS falls from 0.78–2.93 % to 0.31–0.87 %, equal
   to or better than AutoProf (0.31–0.91 %), at +9 % fit time. Signed flux bias drops
   from about +0.5 to +1.1 % to within ±0.05 % when the model is also rendered with
   cubic radial interpolation.
3. **The mock images were never convolved with the PSF.** `profit-cli -P psf.fits`
   only convolves profiles that carry `convolve=1`, and MockGal's profile string does
   not. Every Huang2013 and S4G mock (and every regenerated truth image) is a bare
   Sérsic cusp: the pixel next to the centre is 17–50 % of the central pixel, whereas a
   4.17-pixel-FWHM Gaussian cannot give less than 85 %. This exaggerates every inner-zone
   number in the benchmark for all three tools, makes bilinear bias larger than it would
   be on real images, and makes any interpolating kernel (cubic, Lanczos) ring on the
   cusp. This needs a decision before the benchmark is used for publication.
4. **Cubic sampling rings on those cusps.** On 20 of 55 population cases the plain
   cubic refit is worse (up to 5×) because rings at 1.7–3.1 px see the ringing and the
   ellipticity runs away. Reading rings below 4 px bilinearly removes the failures
   (worst case 1.18× worse, median 7.6 % better, 69 % of cases improved) but is a
   work-around for the unconvolved cores, not a principled choice.
5. **Rendering adds a separate positive flux bias** that is independent of the tool:
   linear-in-intensity radial interpolation on a factor-1.1 grid overestimates by
   0.3–0.6 %. AutoProf's own renderer interpolates a cubic spline in magnitudes, which
   is why its native flux bias is ≈0. Isoster's renderer now offers `log_linear`
   (exact for an exponential) next to the existing `cubic`.

## 1. What the benchmark actually shows

Accepted final statistics (`outputs/huang2013_final_statistics_20260923/metrics.csv`,
designated primaries, 820 Isoster/AutoProf pairs, 2-pixel inner cut), Isoster's
linear harmonic-off renderer for every tool:

| Quantity | Isoster | Photutils | AutoProf |
|---|---:|---:|---:|
| Median full-aperture truth RMS [%] | 2.93 | 3.27 | 4.66 |
| Median inner-zone truth RMS [%] | 2.78 | 3.04 | 4.12 |
| Median signed full flux bias [%] | +0.97 | +1.57 | +0.42 |
| Median signed inner flux bias [%] | +1.53 | +1.67 | +0.25 |
| Fraction of inputs with positive full bias | 0.98 | — | 0.81 |

Isoster has the lower full RMS in 73 % of pairs, but the ratio Isoster/AutoProf has a
long tail: the 90th percentile is 1.57 and 5 % of inputs exceed 2 (8 % in the inner
zone). Under the Photutils spline renderer the picture flips (median 6.52 % vs 2.78 %,
inner bias +3.7 %), and in native mode AutoProf's flux bias is +0.03 %. So "the gap" is
three things: a systematic positive flux bias, a tail of inputs with a badly wrong inner
model, and inner profiles that spline renderers cannot use.

## 2. Code comparison (Isoster `main` vs installed AutoProf 1.3.4)

| Step | Isoster | AutoProf | Consequence |
|---|---|---|---|
| Reading a ring | Bilinear (`map_coordinates(order=1)`), `max(64, 2πa)` samples uniform in position angle φ | Geometry fit: bicubic spline (`RectBivariateSpline`) below 30 px, **nearest pixel** beyond; extraction: Lanczos-5 below 5 PSF (20 px), nearest pixel beyond; `max(15, 0.9·2πa)` samples uniform in eccentric anomaly | Bilinear smooths a convex profile upward and the error is grid-phased; uniform-EA sampling is area-weighted, uniform-φ is not |
| Ring statistic | Harmonic constant term (≈ mean), symmetric 3σ clip once | Median, upper 5σ clip (10 iterations), top 15 % clipped before the FFT in the geometry loss | AutoProf is insensitive to a few high samples (ringing, cusps) |
| Geometry solver | Per ring: Jedrzejewski updates of x0, y0, ε, PA from the 1st/2nd harmonics divided by the local gradient; damping 0.7; rings only coupled by seeding the next one | Global centre fixed; ε and PA at every radius perturbed stochastically to minimise \|FFT₂\|/median with a penalty on ε/PA differences between neighbours; a factor-1.2 grid from max(1, PSF/2) | Isoster's inner rings can wander independently; AutoProf's cannot |
| Gradient | Finite difference between rings at a and 1.1a, same kernel | Not needed (loss is normalised by the median) | A biased kernel also biases Isoster's step size |
| Model image | Linear interpolation of intensity between isophotes (factor 1.1) | Cubic spline of surface brightness in mag vs radius, evaluated at each pixel's elliptical radius | Linear-in-intensity overestimates by 0.3–0.6 % for a Sérsic-like profile |

Two small code findings: the central-pixel row written by `build_isoster_model` was
immediately overwritten by the interpolated model (dead code, now applied last), and
`tests/unit/test_inner_profile_sampling.py` (uncommitted, other worktree) imports
pandas, which is not a project dependency, so a plain `uv run pytest tests/` cannot
collect it.

## 3. Evidence for the sampling mechanism

### Inner geometry on NGC5328 (noiseless, saved configuration)

| sma [px] | ε bilinear (saved) | ε cubic refit | ε AutoProf |
|---:|---:|---:|---:|
| 0.50 | 0.243 | 0.201 | — |
| 1.08 | 0.281 | 0.255 | 0.377 |
| 1.44 | 0.333 | 0.313 | 0.377 |
| 1.74 | 0.320 | 0.357 | 0.377 |
| 2.10 | 0.279 | 0.394 | 0.387 |
| 2.55 | 0.311 | 0.406 | 0.387 |
| 3.08 | 0.355 | 0.370 | 0.367 |
| 3.73 | 0.346 | 0.354 | 0.361 |

The bilinear profile is not monotonic; the cubic one is smooth and matches AutoProf.
The bilinear intensity at 0.5 px is also 11 % low (576 vs 646), because bilinear
interpolation flattens the peak.

### Refits of the seven recorded cases

`benchmarks/exhausted/analysis/sampling_refit_experiment.py` refits each input from
its saved `config.yaml`, changing only the named fields. The `default` variant
reproduces every saved profile exactly (sma, centre, ε, PA, intensity, stop code).
Full-aperture truth RMS [%] on the accepted common support, 2-pixel cut:

| Case | Isoster default | cubic | cubic + EA | AutoProf (saved) | AutoProf, log-spline render |
|---|---:|---:|---:|---:|---:|
| ESO185-G054 / deep_z020 | 1.800 | 0.653 | 0.551 | 0.420 | 0.360 |
| NGC5328 / noiseless | 1.395 | 0.610 | 0.383 | 0.402 | 0.340 |
| NGC1600 / deep_z020 | 1.403 | 0.571 | 0.420 | 0.432 | 0.371 |
| NGC3557 / deep_z020 | 2.930 | 0.927 | 0.865 | 0.905 | 0.864 |
| NGC3268 / wide_z005 | 0.781 | 0.315 | 0.314 | 0.309 | 0.235 |
| NGC1521 / noiseless | 1.048 | 0.596 | 0.573 | 0.722 | 0.682 |
| IC2311 / noiseless | 1.390 | 7.781 | 5.393 | failed | — |

With Isoster's cubic radial renderer the cubic + EA fits reach 0.21–0.74 % (NGC5328
0.254 %) and signed flux bias within ±0.05 % except in the noisy outskirts. Eccentric
anomaly alone (bilinear) or the median statistic alone do nothing here; the kernel is
the lever, EA adds a further 10–40 % once the kernel is right.

### 55-case population sample (15 galaxies × 4 scenarios, seed 20261009)

Medians of full-aperture truth RMS [%] with the linear renderer, and the fraction of
cases where each variant beats the saved default:

| Variant | all | inner | mid | outer | beats default (all) | beats AutoProf (all) | median \|bias\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| default (bilinear, φ) | 3.09 | 2.00 | 1.36 | 2.95 | — | 0.78 | 0.77 |
| EA only | 2.97 | 1.89 | 1.30 | 2.73 | 0.55 | — | — |
| cubic + EA | 3.37 | 3.18 | 1.01 | 2.97 | 0.42 | 0.65 | 0.55 |
| cubic + EA, bilinear below 3 px | 2.92 | 1.65 | 1.01 | 2.97 | 0.64 | 0.87 | 0.66 |
| cubic + EA, bilinear below 4 px | 2.86 | 1.83 | 1.03 | 2.97 | 0.69 | 0.82 | 0.68 |
| AutoProf (saved) | 4.37 | 3.70 | 1.22 | 4.12 | — | — | 0.47 |

Plain cubic + EA is worse on 20/55 cases, by up to 5.1×, all with core ratios
I(1 px)/I(0) ≤ 0.27 and ε jumps at 1.7–3.1 px. Restricted to the 15 cases whose core
ratio is ≥ 0.4 (the least cusp-dominated), it beats the default in 87 % and AutoProf in
93 % of cases (median 0.68 % vs 1.04 % vs 1.02 %). The 4-px fallback keeps the gains
(ratio to default: 10th–90th percentile 0.88–1.08, worst 1.18, all in `wide_z050`).
Fit time: median +9 %, 90th percentile +19 %, worst +45 % (small images, where the
once-per-image prefilter is a larger share); on the largest Huang2013 image (2363²,
0.81 s) +11 %. The outer zone is unchanged by the kernel and slightly improved by EA.

## 4. The mocks have no PSF

`profit-cli -w 41 -H 41 -m 27 -p "sersic:...:re=1.9:nser=3.5:..." -t -P psf.fits`
returns exactly the same image as without `-P` (same total flux 61893.7, neighbour /
centre = 0.219). Only `...:convolve=1` convolves (neighbour / centre = 0.892, as a
4.17-px Gaussian requires). `MockGal.to_libprofit_spec` (`isophote_test/mockgal.py`)
never emits `convolve=1`, so the production images and the regenerated truth images are
unconvolved, although their headers say `PSF=True, PSFFWHM=0.7`. In the 55 population
cases the neighbour/centre ratio is 0.17–0.82 (median 0.30).

Consequences: all inner-zone metrics in the published analysis describe fits to bare
Sérsic cusps; AutoProf's robust-median, regularised geometry copes with cusps better,
while Isoster's bilinear bias is amplified; the IC2311-type failures of cubic sampling
are a response to those cusps; the S4G mocks share the generator. A separate check
(`benchmarks/exhausted/analysis/psf_convolved_refit.py`) convolves the noiseless images
with the intended Gaussian and refits; see section 5.

## 5. Behaviour on PSF-convolved images

`psf_convolved_refit.py` convolves the 22 noiseless images (15 population galaxies plus
the seven recorded cases) with the intended 4.17-px Gaussian, optionally adds the
campaign's `wide_z005` noise (seed 20261009), refits with the saved configuration and
scores against the convolved noiseless image on the accepted support. After
convolution the neighbour/centre ratio is 0.91–0.96, as it should be. Medians of
full-aperture truth RMS [%] over the 22 galaxies:

| Condition, renderer | default | cubic + EA | cubic + EA, bilinear < 3 px | cubic + EA beats default |
|---|---:|---:|---:|---:|
| noiseless, linear | 0.784 | 0.287 | 0.640 | 22 / 22 |
| noiseless, cubic | 0.695 | 0.166 | 0.598 | 22 / 22 |
| wide noise, linear | 0.788 | 0.308 | 0.648 | 22 / 22 |
| wide noise, cubic | 0.706 | 0.213 | 0.611 | 22 / 22 |

Inner-zone RMS with the cubic renderer drops from 0.72 % to 0.08 % (noiseless) and
0.10 % (wide noise); median absolute flux bias from 0.061 % to 0.003 %. IC2311, the
worst cubic failure on the unconvolved image, improves from 0.95 % to 0.26 %. The
per-galaxy ratio cubic + EA / default is 0.27–0.86 in both conditions. Only the noisy
outer zone is marginally worse (1.84 → 1.97 % with the cubic renderer), consistent
with cubic reads carrying slightly more noise than a bilinear average. The inner
bilinear fallback is unnecessary once the cores are PSF-limited and costs most of the
gain, so it should stay a diagnostic option rather than a default.

## 6. What was changed in Isoster (all opt-in, defaults unchanged)

- `IsosterConfig.sampling_interpolation = "bilinear" | "cubic"`; cubic prefilters the
  image once (`scipy.ndimage.spline_filter`, 9 ms at 731², 0.37 s at 4001²) and reads
  rings with `map_coordinates(order=3, prefilter=False)`.
- `IsosterConfig.sampling_cubic_min_sma` (default 0): rings below this sma are read
  bilinearly in cubic mode.
- `build_isoster_model(interp_kind="log_linear")`; central-pixel row applied last.
- Tests: `tests/unit/test_sampling_interpolation.py`. Full suite: 1274 passed.
- Not done: NaN/mask handling for the cubic prefilter (non-finite pixels are zeroed;
  the mask must exclude samples near them), forced-photometry path still bilinear,
  variance map still sampled bilinearly.

## 7. Recommendations and open decisions

1. Fix MockGal (`convolve=1`) and regenerate both mock sets before any publication use;
   until then treat inner-zone results as fits to unconvolved cusps.
2. Re-run this refit comparison (and AutoProf) on convolved mocks and on the SGA real
   images, where a PSF is always present, before changing Isoster's default kernel.
3. If adopted: cubic + EA sampling with the inner bilinear fallback, plus `log_linear`
   or `cubic` radial rendering for Isoster's model images (the latter is the fair
   counterpart to AutoProf's spline-in-magnitude renderer).
4. Longer term, give the inner rings the same protection AutoProf has: couple
   neighbouring ε/PA or clip the top samples before the harmonic fit, so a cusp or a
   star cannot drive one ring away.
