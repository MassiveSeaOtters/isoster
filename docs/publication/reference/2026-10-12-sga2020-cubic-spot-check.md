# Cubic ring read on SGA-2020 cutouts: real-data spot check

**Date:** 2026-10-12. **Branch:** `feature/sga2020-cubic-spot-check-20261011`.
**Script:** `benchmarks/sga2020/cubic_spot_check.py`. **Outputs:**
`outputs/benchmark_sga2020/cubic_spot_check/` (per-galaxy ring CSVs, `summary.json`,
`summary.md`; gitignored, the tables below are copied from `summary.md`).

## Question

The 1.1.0 default switched every ring read from bilinear to a prefiltered cubic
B-spline on the strength of two truth-based mock campaigns (see the
[combined round-2 summary](2026-10-11-round2-combined-summary.md)). Before the
default is used in production, does it behave on real Legacy Survey cutouts the way
it did on mocks: no failures, the same rings, a measurable gain where the profile is
steep, nothing in the noise-dominated outskirts, and no new sensitivity to masked
or uncovered pixels?

This is a spot check from the isoster side, not a run of the `sga_isoster`
pipeline. Nothing was written into the data folders.

## Setup

- **Data.** Twelve galaxies from `benchmark_sample1` of the SGA-2020 benchmark set
  (`/Volumes/galaxy/isophote/sga2020/data/benchmark_sample1/`, r band, 0.262 arcsec
  per pixel, nanomaggies). Among the first 400 galaxies of the sample, four were
  drawn from each tercile of the SGA `MAJORAXIS`: the two with the most pixels of
  zero inverse variance (survey edges, saturated stars) and the two with none. The
  scan of those 400 found no non-finite image pixels at all; 212 galaxies have
  zero-inverse-variance pixels (median 145 of them), and the campaign mask covers
  6.5 % of a cutout at the median.
- **Configuration.** Mirrors the `sga_isoster` campaign fitter: centre, ellipticity
  and position angle from the SGA ellipse table (astronomical PA converted to the
  math convention), `sma0 = max(8, 0.05 MAJORAXIS)`, `maxsma = min(ny, nx) // 2`,
  the campaign mask (`> 0` is bad), the variance map `1 / invvar` (infinite where
  `invvar <= 0`, which isoster excludes), `debug`, `full_photometry` and
  `compute_cog` on, twelve first-isophote retries, everything else at the library
  defaults (harmonics on).
- **Arms.** `bilinear` (`sampling_interpolation="bilinear"`, the pre-1.1.0 read),
  `cubic` (the default), and `cubic_nan_masked`: the cubic read on a copy of the
  image whose masked or uncovered pixels are set to NaN, so that the prefilter sees
  the nearest unmasked value instead of the masked pixel's own value. The
  difference to `cubic` isolates what masked pixels leak into the spline
  coefficients of their neighbours.
- **Metrics.** Ring-by-ring differences at the same semi-major axis (stop code 0 on
  both sides), in per cent and in units of the bilinear ring's own intensity error;
  geometry differences; wall time, best of two runs; and the campaign's data
  residual, the RMS of `(image - model) / sigma` over unmasked covered pixels,
  with the model rendered from each arm's profile by the native renderer. Zones are
  `sma / MAJORAXIS` (inner < 0.25, mid 0.25 to 1, outer > 1); the near-mask split
  takes pixels within 3 px of a masked or uncovered pixel.

## Results

| galaxy | side px | MAJORAXIS px | rings bil / cubic | stop!=0 bil / cubic | wall bil / cubic s | cubic-bil median % inner / mid | inner median abs sigma | rings touching mask | residual RMS sigma inner bil / cubic | residual RMS sigma mid bil / cubic |
|---|---|---|---|---|---|---|---|---|---|---|
| 2MASXJ15313041+0615387 | 487 | 65 | 66 / 66 | 11 / 10 | 0.12 / 0.12 | 0.28 / 0.26 | 4.64 | 2 | 0.912 / 0.782 | 1.082 / 1.055 |
| 2MASXJ12312909+3409146 | 315 | 47 | 62 / 62 | 13 / 12 | 0.08 / 0.07 | 0.77 / 0.34 | 2.70 | 0 | 0.649 / 0.601 | 0.896 / 0.918 |
| 2MASXJ10510760+1951209 | 251 | 35 | 59 / 59 | 16 / 13 | 0.08 / 0.08 | 0.59 / 0.21 | 2.67 | 0 | 0.857 / 0.772 | 1.171 / 1.160 |
| 2MASXJ16012223+4137029 | 259 | 36 | 60 / 60 | 12 / 12 | 0.06 / 0.07 | 1.32 / 0.15 | 4.44 | 0 | 0.427 / 0.362 | 0.538 / 0.535 |
| IC2535 | 587 | 122 | 68 / 68 | 8 / 7 | 0.15 / 0.15 | 0.20 / -0.13 | 2.14 | 7 | 1.777 / 1.704 | 0.687 / 0.701 |
| NGC7406 | 637 | 125 | 68 / 68 | 9 / 5 | 0.16 / 0.20 | 0.33 / -0.00 | 2.22 | 10 | 10.826 / 9.429 | 3.546 / 3.445 |
| ESO364-032 | 601 | 108 | 69 / 69 | 8 / 7 | 0.13 / 0.18 | 0.19 / -0.17 | 1.91 | 5 | 6.493 / 6.667 | 8.687 / 8.610 |
| ESO476-027 | 657 | 109 | 69 / 69 | 5 / 5 | 0.18 / 0.17 | 0.46 / 0.15 | 3.66 | 8 | 1.720 / 1.643 | 3.100 / 3.021 |
| NGC6289 | 755 | 181 | 71 / 71 | 0 / 0 | 0.32 / 0.34 | 0.30 / -0.03 | 1.85 | 29 | 0.430 / 0.424 | 0.505 / 0.504 |
| IC1256 | 877 | 190 | 72 / 72 | 7 / 6 | 0.31 / 0.28 | 0.09 / 0.12 | 1.75 | 7 | 2.439 / 2.552 | 1.859 / 1.978 |
| NGC3767 | 677 | 180 | 70 / 70 | 2 / 2 | 0.23 / 0.23 | 0.16 / -0.03 | 1.90 | 4 | 1.620 / 1.606 | 2.072 / 2.150 |
| IC5258 | 689 | 187 | 69 / 69 | 2 / 2 | 0.20 / 0.18 | 0.04 / -0.00 | 0.98 | 21 | 0.834 / 0.823 | 1.008 / 1.014 |

### Across the sample

- 12 galaxies, 803 bilinear rings; ring counts identical across arms in 12 of 12
  galaxies; non-zero stop codes: bilinear 93, cubic 81.
- Wall time, best of repeats, summed over the sample: bilinear 2.02 s, cubic 2.07 s
  (ratio 1.03; per-galaxy ratio median 1.00).

Ring-by-ring, cubic minus bilinear:

| zone (sma / MAJORAXIS) | rings | median % | RMS % | median abs in sigma | rings beyond 3 sigma | median d eps | median d PA deg | median centre shift px |
|---|---|---|---|---|---|---|---|---|
| inner (0-0.25) | 479 | +0.399 | 1.05 | 2.63 | 48 % | +0.0013 | -0.00 | 0.011 |
| mid (0.25-1) | 176 | +0.041 | 1.96 | 0.62 | 15 % | +0.0006 | -0.01 | 0.078 |
| outer (1-inf) | 39 | -1.235 | 16.62 | 1.46 | 18 % | -0.0052 | -0.13 | 2.059 |

Masked pixels filled before the prefilter (cubic with NaN-masked pixels minus plain
cubic), rings inside MAJORAXIS:

- rings touching masked pixels (> 5 % of samples): 66 rings, median -0.001 %, RMS
  1.96 %, max |d| 14.92 %.
- clean rings: 588 rings, median +0.000 %, RMS 0.38 %, max |d| 7.06 %.

Data residual, RMS of `(image - model) / sigma` over unmasked covered pixels
(medians over galaxies of each galaxy's RMS):

| region | bilinear | cubic | cubic, masked pixels filled | galaxies where cubic beats bilinear |
|---|---|---|---|---|
| inner | 1.266 | 1.215 | 1.224 | 10 of 12 |
| mid | 1.126 | 1.107 | 1.119 | 7 of 12 |
| outer | 0.920 | 0.920 | 0.919 | 4 of 12 |
| near_mask | 1.628 | 1.616 | 1.607 | 5 of 9 |
| away_from_mask | 1.111 | 1.088 | 1.099 | 7 of 12 |

## Reading

- **Nothing breaks.** Every arm produced the same number of rings on every galaxy;
  the cubic read left 81 rings with a non-zero stop code against 93 for bilinear,
  all in the noise-dominated outskirts where the rings reach the frame edge.
- **No measurable cost on real cutouts.** With the campaign flags on, the cubic
  read costs 3 % over the sample and nothing at the per-galaxy median; the
  prefilter of a 0.1 to 0.8 Mpix cutout is a few milliseconds of a 0.1 to 0.3 s fit.
- **The cores move, and towards the pixels.** Inside a quarter of MAJORAXIS the cubic
  read is 0.4 % brighter at the median than the bilinear one, 1 % RMS, and the shift
  is significant ring by ring (median 2.6 sigma, half the rings beyond 3 sigma). The
  data residual says which read describes the pixels better: the inner-zone RMS
  falls from 1.27 to 1.22 sigma and cubic wins on 10 of 12 galaxies. This is the
  real-data counterpart of the mocks' inner-zone gain.
- **Mid zone slightly better, outskirts untouched.** Mid-zone differences are at the
  0.6 sigma level with a small residual gain (7 of 12); beyond MAJORAXIS the two
  reads differ by 16 % RMS in relative terms but only 1.5 sigma, the rings there
  wander by 2 px in centre, and the residual is identical to three decimals.
  Geometry is unchanged where it matters: ellipticity differs by 0.001, position
  angle by less than 0.01 degree and the centre by 0.01 px in the inner zone.
- **Masked pixels do leak into the coefficients, but filling them does not help.**
  Rings whose samples touch masked pixels move by 2 % RMS when the masked pixels
  are replaced by their nearest unmasked neighbour before the prefilter, against
  0.4 % for clean rings, so the leak is real. The data residual near masks improves
  only from 1.616 to 1.607 sigma while it worsens away from masks (1.088 to 1.099)
  and in the inner zone (1.215 to 1.224). The library therefore keeps filling
  non-finite pixels only (the fix shipped with the multi-band kernel) and leaves
  masked pixels in the coefficients, as the bilinear read also reads them within
  one pixel of a mask edge.

## Decision

The cubic default stands for the SGA-2020 production run: it is at least as robust
as the bilinear read on these cutouts, costs nothing measurable, and reproduces the
pixels better where the profile is steep. The `sga_isoster` campaign should treat
its historical `ref_default` arm as bilinear and record which kernel each new run
uses. Filling masked pixels before the prefilter is not adopted; the evidence above
is the reason, and the `cubic_nan_masked` arm of the script is kept so the question
can be reopened on a larger sample.
