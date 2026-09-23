# Huang2013: fixed two-pixel evaluation and reconstruction diagnostic

## Decision and scope

On 2026-09-23 the author adopted a fixed **2-pixel elliptical inner radial
cut** for publication benchmark evaluation and QA, including future S4G and
real-galaxy tests. It replaces the former one-PSF-FWHM cut (4.166667 pixels
for these mocks). This is an evaluation change, not a fit, PSF, or input-image
change. The outer radius, reference geometry, radial-zone boundaries and
all-successful-arm finite-support intersection are unchanged. Inner, middle,
outer zones remain r < 0.5 R_ref, 0.5 R_ref <= r < 2 R_ref, and r >= 2 R_ref,
each intersected with the two-pixel common aperture.

Shared benchmark model-evaluation defaults now use 2 pixels, and fitter
wrappers no longer substitute their tool-specific first fitted radius.
Huang2013 records `inner_cut_pix=2` and
`reconstruction=isoster_linear_no_harmonics`; the physical `psf_pix` is
unchanged. Ring and centroid diagnostics also use the fixed cut. Native
metrics copied from the old run records remain historical, not recalculated
claims. The native v1.1 zone boundaries (0.5/1.5 R_ref) remain distinct from
the scientific truth-analysis zones (0.5/2 R_ref).

## Execution and safety

Active checkout outside Dropbox:
`/Users/shuang/code/isoster-autoprof-zero-background-20260921-remote`, branch
`feature/huang-two-pixel-analysis-20260923`. Previous uncommitted QA work was
preserved. No merge, push, or modification of fitting products was performed.

Analysis root:
`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/`.

- Gate: `huang2013_two_pixel_gate_2026_09_23`, IC1459, all nine conditions,
  153 records; measured galaxy processing time 7.1703 seconds.
- Full: `huang2013_scientific_analysis_two_pixel_2026_09_23`, 8 workers with
  BLAS/OpenMP/Accelerate each limited to one thread. All 93 galaxies finished.
- Previous `huang2013_scientific_analysis_zero_background_2026_09_21` remains
  intact and is the historical PSF-cut comparison, not the current aperture.

All 14,229 selected outcomes, source paths, statuses, physical PSF and
reference radii match the previous selection. There are 13,283 successes,
51 retained AutoProf errors, 58 Photutils failures, and 837 Isoster skips.
All 837 regenerated input/noise checks match exactly at float32 precision.
The 30,862 unique source hashes agree before/after the new analysis and
also with the previous analysis. No fit was repeated. Inner RMS is unavailable
for 1,213 successful rows because the corresponding aperture is empty;
middle RMS is available for every successful row. Empty is not zero.

## Updated primary-arm comparison

Primaries remain Isoster `ref_default`, Photutils `baseline_median`, and
AutoProf `baseline`, with fixed zero AutoProf background. Only matched finite
successful triples enter each row. A win is a strict minimum, with numerical
ties checked at rtol=1e-9, atol=1e-12; no ties were found. Flux bias ranks its
absolute magnitude. Medians and individual win counts answer different questions.

| Metric | Zone | Matched inputs | Isoster wins | Photutils wins | AutoProf wins |
|---|---|---:|---:|---:|---:|
| Truth RMS | All | 794 | 402 | 184 | 208 |
| Truth RMS | Inner | 719 | 343 | 157 | 219 |
| Truth RMS | Middle | 794 | 313 | 112 | 369 |
| Truth RMS | Outer | 794 | 272 | 252 | 270 |
| Absolute flux bias | All | 794 | 126 | 38 | 630 |
| Absolute flux bias | Inner | 719 | 76 | 64 | 579 |
| Absolute flux bias | Middle | 794 | 143 | 60 | 591 |
| Absolute flux bias | Outer | 794 | 324 | 108 | 362 |

| Metric | Zone | Isoster median [%] | Photutils median [%] | AutoProf median [%] |
|---|---|---:|---:|---:|
| Truth RMS | All | 3.0026 | 3.2655 | 4.8019 |
| Truth RMS | Inner | 2.8494 | 3.0409 | 4.4206 |
| Truth RMS | Middle | 1.7548 | 1.8871 | 1.6109 |
| Truth RMS | Outer | 4.1478 | 4.8080 | 4.8089 |
| Absolute flux bias | All | 1.0027 | 1.6050 | 0.5044 |
| Absolute flux bias | Inner | 1.6538 | 1.6809 | 0.4853 |
| Absolute flux bias | Middle | 0.7104 | 0.8319 | 0.4061 |
| Absolute flux bias | Outer | 0.9673 | 1.8988 | 0.9178 |

The change from 151 to 402 Isoster full-aperture RMS wins is an aperture
sensitivity result using unchanged fits, not a new fitting improvement.
Some middle/outer pixels also change for compact galaxies whose zone
boundaries lie inside the old 4.166667-pixel exclusion.

EA's previous full-aperture advantage also depends on the cut. `geom_ea`
now improves RMS in 51.02% of 837 pairs and a majority of scenarios for
47/93 galaxies; `geom_simul_ea` improves it in 53.05% and for 51/93 galaxies.
Neither wins a majority in all nine scenario groups. Their median absolute
RMS differences have galaxy-block bootstrap intervals spanning zero.
Absolute flux-bias wins are 40.50% and 40.98%, respectively. There is no
new-default fitting recommendation from these measurements.

## Harmonics: what the flag actually means

The public `isoster.model.build_isoster_model` defaults to
`use_harmonics=True`; it is not globally disabled. The scientific comparison
explicitly uses `False` as a common elliptical-profile baseline. This
baseline cannot establish the full benefit of harmonic reconstruction.

Schema-2 AutoProf profiles deliberately store Bender columns a3/b3/a4/b4 as
NaN while retaining native FFT coefficients and b0. The current Isoster
renderer maps nonfinite harmonics to zero. Therefore simply enabling its
flag would add Isoster/Photutils terms but effectively no AutoProf terms.
The audited baseline AutoProf profiles use polar angle from the image x axis.
Recover raw amplitudes S=-2|b0|a and C=2|b0|b and rotate by n*PA into the
major-axis polar frame. For Isoster/Photutils recover raw amplitudes with
sma*|gradient| times the stored Bender coefficients. These conversions reuse
the existing `benchmarks/harmonic_scale/conventions.py` helpers.

## Bounded reconstruction experiment

Script: `benchmarks/exhausted/analysis/reconstruction_diagnostic.py`.
Output: `outputs/huang2013_reconstruction_diagnostic_20260923_v3`.
Inputs: IC1459 and NGC4697, wide_z005 and wide_z050, three primary arms.
All 48 models were produced; `unavailable.json` is empty. Inputs are hashed.

For each input, hold pixels fixed across all tools and all four variants:

1. Shared Isoster renderer, linear interpolation, harmonics off.
2. Shared Isoster renderer, cubic intensity interpolation, harmonics off.
3. Shared Photutils spline/rasterization renderer, harmonics off.
4. The same Photutils renderer, raw polar n=3,4 harmonics added.

The fourth variant uses a synthetic gradient of -1 and carrier coefficients
raw/sma so Photutils' multiplication by -gradient*sma recovers the exact raw
intensity amplitude. This is an interface adapter, **not an inferred physical
AutoProf gradient**. The synthetic regression test checks amplitude, phase,
sign and agreement of both source conventions, including rejection of missing
or EA coefficients. Polar and EA bases cannot be converted by same-order PA
rotation. This experiment does not yet test the EA arms' harmonic reconstruction.

Example full-aperture Truth RMS [%], IC1459/wide_z050:

| Shared reconstruction | Isoster | Photutils | AutoProf |
|---|---:|---:|---:|
| Isoster linear, no harmonics | 4.1610 | 4.3700 | 4.9446 |
| Isoster cubic, no harmonics | 3.8175 | 4.0122 | 4.8063 |
| Photutils spline, no harmonics | 9.0418 | 9.1071 | 3.1543 |
| Photutils spline, raw harmonics | 9.3436 | 9.5297 | 5.8006 |

These diagnostic pixels intersect only the primary tools and reconstruction
variants, not all 16 executable campaign arms, so the numbers differ slightly
from the production QA. They must not replace production scores silently.
Changing the shared renderer can reverse the ranking. Adding harmonics does
not automatically improve truth accuracy: noise, estimator conventions and
reconstruction matter even for symmetric multicomponent mocks.

The script also records truth sampled on each fitted ellipse, with both a
common polar-mean statistic and a native-statistic proxy. These avoid 2-D
rendering but remain conditional on the fitted geometry and native radius
grid. AutoProf's finite-width extraction/clipping is not reproduced exactly;
do not interpret them as a standalone tool ranking or proof of a fitting bug.

Next work: verify central sampling, geometry interpolation and pixel integration
on analytic fixtures; extend the synthetic harmonic check to flattened/EA
bases and test more representative inputs before a harmonic-inclusive campaign
ranking. No conclusion that one fitting algorithm is intrinsically inferior
is supported by this bounded diagnostic.

## Products and validation

- `outputs/huang2013_scientific_products_two_pixel_20260923`: six summary
  PNG/PDF pairs and eight selected-case atlas pairs, using the approved QA
  design, numerical CSVs and metric checks. The case roster is recomputed from
  the new measurements, so it need not match the old ten-page atlas.
- `outputs/huang2013_individual_qa_two_pixel_20260923`: eight PNG/PDF pairs
  for the two review examples, captions, statistics and 29 metric checks;
  all 70 source hashes unchanged. Inner ellipses now indicate 2 pixels.
- 658 tests passed (657 unit plus one actual campaign integration test), with
  10 existing warnings. Ruff and whitespace checks passed.
- `outputs/huang2013_two_pixel_summary_20260923/cross_tool_summary.csv`
  stores the primary-tool win counts, matched denominators and median values
  above, with a measurement digest and tie tolerances in `provenance.json`.
- Old analyses, QA and fitting products were neither overwritten nor deleted.

Reproduction: use the full analysis command in the new `provenance.json`,
changing `--output` to a new directory. Default `--inner-cut-pix` is 2.
Run the existing `publication_huang` plotting CLI and `individual_qa_demo`
CLI on that measurement directory, each with a new output directory.
The latter retains its historical module name but is now the approved layout.
