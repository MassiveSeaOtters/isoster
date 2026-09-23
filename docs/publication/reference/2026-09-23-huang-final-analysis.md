# Huang2013 final saved-fit reconstruction analysis

The complete retained population has been analysed and audited: 93 galaxies,
837 inputs, 17 arms, 14,229 accepted fit outcomes, 56,916 arm/mode outcomes,
227,664 radial measurement rows and 8,370 QA page pairs. No fits were repeated.
The harmonic-off baseline, all failures, intentional skips and unsupported
reconstructions remain explicit. **The ranking depends strongly on reconstruction
and radial zone; these results do not establish a single overall tool winner.**

[Browse QA and statistics](../../../outputs/huang2013_final_statistics_20260923/index.html).
The QA directory is on the mounted `/Volumes/galaxy` volume; the local statistics
index links to it. All machine-readable tables are in
`outputs/huang2013_final_statistics_20260923` in the isolated checkout.

## Accepted sources and preservation

Accepted selection:
`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/huang2013_scientific_analysis_two_pixel_2026_09_23`.
The saved roster preserves the zero-background AutoProf selection, accepted recovery
joins and corrected PA convention. We reused its audited PSF-convolved truth images.
There was no source reselection, refitting, extrapolated truth filling, merge or push.

Final reconstruction/QA directory:
`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/huang2013_final_reconstruction_2026_09_23_v2`.
The earlier directory without `_v2` was stopped after a missing-model QA gate
exposed removal of valid 1-D curves; its files remain preserved. All historical
campaigns and outputs remain intact. The final run contains a source archive,
uncommitted diff, code hashes, package versions, branch and HEAD provenance.

Original fit coverage is unchanged: Isoster 7,533 successful and 837 intentional
no-variance-map skips; Photutils 2,453 successful and 58 failed; AutoProf 3,297
successful and 51 errors. Primary arms remain `ref_default`, `baseline_median`
and `baseline`; a failed primary is never replaced by another arm.

## Reconstruction definitions

| Mode | Renderer and harmonics | Scope |
|---|---|---|
| `shared_baseline` | Isoster linear, harmonics off | All successful saved profiles; original-support measurements reproduced before extensions |
| `shared_spline_off` | Photutils spline/rasterizer, harmonics off | All successful profiles, including EA geometry |
| `shared_spline_on` | Same spline/rasterizer, calibrated raw polar intensity n=3,4 | Compatible finite polar profiles only; no EA-to-polar substitution |
| `native` | Isoster basis-aware radial shifts n=3,4; Photutils native additive n=3,4; AutoProf saved ellipse model | Tool-native alternatives; not a common harmonic-on experiment |

All audited stored higher-order fields are n=3,4. Each outcome records source
basis, available and rendered orders, renderer, rendered basis and reason for
unavailability. The raw adapter recovers Bender amplitudes using
`sma * abs(gradient)`, converts AutoProf FFT fields with their calibrated sign and
factor of two, and rotates sky-polar coefficients by n times PA. Its synthetic
carrier gradient of -1 transports amplitudes through Photutils; it is not a
physical AutoProf gradient estimate. Native Photutils interpolates normalized
coefficients and gradient separately, so it need not equal the raw-carrier model.

**Native AutoProf is not intensity-harmonic-on.** We loaded only the accepted
`raw/<galaxy>__<scenario>_genmodel.fits`, verifying saved options for zero background
and absent fitted Fourier shape modes. Generic fallback `model.fits` files were
not used. Native AutoProf zero-valued exterior pixels are unavailable model support.
All 3,297 successful AutoProf fits had verified native ellipse models.

Real `geom_ea`, `geom_simul_ea` and `harm_simul_ea` profiles were validated against
explicit EA reconstruction; they matched exactly and differed from deliberately
incorrect polar reconstruction. Independent synthetic checks test the angular
basis and raw harmonic sign/phase/normalization. EA is supported in native Isoster
and explicitly unsupported in the shared polar harmonic-on mode. No new shared-EA
algorithm or modified reconstruction algorithm was introduced.

## Apertures, values and missing measurements

Elliptical radii use the frozen reference geometry, an inner cut of exactly 2 pixels
and the recorded outer radius. The physical PSF and fitted radii are unchanged.
Zones are inner <0.5 R_ref, middle 0.5–2 R_ref and outer >=2 R_ref. Every numerical
metric intersects its zone with the recorded common finite support.

The original baseline uses the intersection of all successful harmonic-off arms.
The new matched aperture further intersects every available arm/mode. A missing
model contributes no invented zeros and is unavailable for comparisons. All new
mode/arm differences use this same matched aperture. Relative to original support,
the median retained full-aperture fraction is 96.826%, with minimum 32.0175%; the
outer-zone median is 96.3448%, minimum 31.4159%. Inner and middle support are
unchanged where nonempty. Seventy-six inputs have empty inner zones; no full,
middle or outer aperture is empty. See `support_audit.csv` for every count.

Truth RMS is `sqrt(sum((M-T)^2)/sum(T^2))`. Signed flux bias is
`sum(M-T)/sum(T)`, and absolute flux bias is its absolute value, not the sum of
absolute residuals. Data residual RMS is `sqrt(mean((D-M)^2))/injected_sigma`.
CSVs contain fractions; figure tables display percentages. Noise-normalized RMS
is undefined for noiseless images; their accepted `injected_sigma` field is NaN,
not a measured positive noise scale. `metric_unavailability.csv` records this
explicitly, along with empty zones and missing/failed models. No missing value
has been scored as zero. Original native fit-record metrics remain historical
columns in `accepted_rows.csv` and are not substituted for these measurements.

Matched three-tool wins require all designated primaries and finite measurements.
Ties use rtol=1e-9 and atol=1e-12. Flux-bias wins compare absolute magnitude while
QA prints the signed bias. No ties occurred in the primary tables below.

## Coverage of reconstruction modes

| Mode | Available models | New unavailable models | Retained fit failures/errors/skips |
|---|---:|---:|---:|
| Shared linear off | 13,283 | 0 | 946 |
| Shared spline off | 13,283 | 0 | 946 |
| Shared spline on | 10,711 | 2,572 | 946 |
| Native | 13,224 | 59 | 946 |

Shared-spline-on exclusions comprise 2,511 EA outcomes, 57 missing raw-harmonic
amplitude outcomes and four nonfinite geometry/intensity outcomes. The latter
include two Isoster `stack_all` and two Photutils outcomes. Native exclusions are
59 Photutils outcomes: 58 missing a3 and one missing gradient. These are failures
of supported reconstruction from otherwise accepted fits, not newly failed fits.
All rows and detailed reasons remain in `outcomes.csv` and `coverage.csv`.

## Primary-tool results on matched support

The order of each triplet is **Isoster / Photutils / AutoProf**. ALL denotes the
full evaluation aperture, not every displayed pixel. Mode denominators differ;
compare harmonic effects using the paired-control results below rather than
subtracting win totals from different samples.

| Mode | Zone | Matched inputs | Truth RMS wins | Absolute flux-bias wins |
|---|---|---:|---|---|
| shared_baseline | ALL | 794 | 402 / 184 / 208 | 126 / 41 / 627 |
| shared_baseline | INNER | 719 | 343 / 157 / 219 | 76 / 64 / 579 |
| shared_baseline | MID | 794 | 313 / 112 / 369 | 143 / 60 / 591 |
| shared_baseline | OUTER | 794 | 277 / 248 / 269 | 314 / 105 / 375 |
| shared_spline_off | ALL | 794 | 38 / 18 / 738 | 52 / 14 / 728 |
| shared_spline_off | INNER | 719 | 48 / 19 / 652 | 13 / 14 / 692 |
| shared_spline_off | MID | 794 | 221 / 40 / 533 | 166 / 33 / 595 |
| shared_spline_off | OUTER | 794 | 256 / 289 / 249 | 368 / 100 / 326 |
| shared_spline_on | ALL | 772 | 61 / 38 / 673 | 55 / 8 / 709 |
| shared_spline_on | INNER | 697 | 41 / 41 / 615 | 15 / 19 / 663 |
| shared_spline_on | MID | 772 | 368 / 98 / 306 | 162 / 26 / 584 |
| shared_spline_on | OUTER | 772 | 394 / 325 / 53 | 378 / 82 / 312 |
| native | ALL | 772 | 637 / 2 / 133 | 90 / 12 / 670 |
| native | INNER | 697 | 553 / 0 / 144 | 166 / 2 / 529 |
| native | MID | 772 | 320 / 126 / 326 | 76 / 57 / 639 |
| native | OUTER | 772 | 109 / 284 / 379 | 177 / 149 / 446 |

Full-aperture median errors on each mode’s matched three-tool sample:

| Mode | Truth RMS [%], I/P/A | Absolute flux bias [%], I/P/A |
|---|---|---|
| shared_baseline | 2.996 / 3.265 / 4.798 | 1.015 / 1.574 / 0.502 |
| shared_spline_off | 6.641 / 6.702 / 2.861 | 1.688 / 2.281 / 0.403 |
| shared_spline_on | 6.863 / 6.916 / 4.015 | 1.760 / 2.428 / 0.391 |
| native | 2.604 / 6.979 / 5.039 | 1.052 / 2.434 / 0.313 |

The original-support baseline is preserved separately: full RMS wins remain
402/184/208 of 794, and absolute flux-bias wins are 126/38/630. Recomputing the
baseline on the new support leaves those RMS counts unchanged but changes the
bias counts to 126/41/627. Do not overwrite the original bias result with the
matched-aperture result. Inner denominators are 719 for the off modes and 697
for the harmonic-on/native modes; 22 additional primary comparisons are unavailable
in the latter modes. These additional losses occur in the noiseless condition.

Shared-spline-off gives AutoProf the most full-aperture RMS wins in each of the
nine scenarios; shared-spline-on does too. Native gives Isoster the most full RMS
wins in each scenario. These are reconstruction-dependent rankings, not proof
that the underlying fitting algorithms exchange intrinsic accuracy. The original
linear and shared-spline renderers interact differently with native radial sampling,
particularly in the inner region. Native models also use different harmonic
operations, including AutoProf's absence of measured intensity harmonics.

## Paired harmonic, renderer and arm effects

Within each tool, harmonic-on versus its identical shared-spline-off control gives:

| Primary tool | Full RMS lower / pairs | Median RMS change [percentage points] | Outer RMS lower / pairs | Median outer RMS change [percentage points] | Galaxies with lower median full RMS / 93 |
|---|---:|---:|---:|---:|---:|
| Isoster | 284 / 837 | +0.06546 | 162 / 837 | +0.92416 | 29 |
| Photutils | 249 / 788 | +0.08075 | 146 / 788 | +0.79000 | 27 |
| AutoProf | 169 / 820 | +0.54654 | 139 / 820 | +1.22731 | 14 |

Thus adding the measured raw n=3,4 terms does not generally improve truth RMS
under this shared additive renderer. Full absolute-flux-bias medians change by
+0.02718, +0.03779 and -0.01560 percentage points for Isoster, Photutils and
AutoProf, respectively. Harmonic conclusions must retain the radial trade-offs.

Native Isoster versus its linear harmonic-off control reduces full RMS in
723/837 pairs, median change -0.32898 percentage points; 90/93 galaxies have a
negative median change across available conditions. Inner RMS decreases in
706/761 pairs, median -0.44281 points. Outer RMS decreases in only 160/837 pairs,
median change +1.16080 points. Absolute full flux bias improves in 495/837 pairs,
but outer absolute bias improves in only 200/837. The full-aperture improvement
therefore does not imply improved outer reconstruction.

Changing only the off renderer from linear to spline has larger median full-RMS
changes than the shared harmonic effect: +3.36201, +3.31943 and -1.19152 percentage
points for Isoster, Photutils and AutoProf. Native Photutils versus shared linear
has median full-RMS change +3.59863 points, combining renderer and harmonic effects;
it is not an isolated harmonic test. Native AutoProf versus shared linear has
median full-RMS change +0.10126 points and absolute-bias change -0.30644 points;
this is an ellipse-renderer comparison, not an intensity-harmonic comparison.

On the new matched linear baseline, `geom_ea` and `geom_simul_ea` beat the Isoster
primary in 428/837 and 446/837 full-RMS pairs. With native harmonics enabled, those
counts fall to 291/837 and 300/837, with median changes +0.0229 and +0.0227
percentage points. Only 26/93 and 32/93 galaxies have negative median native
arm-minus-default RMS. `harm_simul_ea` wins 184/837 native pairs and has negative
median changes in 16/93 galaxies. No EA-default recommendation follows.

Large unfavorable native simultaneous-EA outcomes remain in all summaries. For
example, IC4765/wide_z005 has truth-relative RMS 65.733907 and signed flux bias
128.356817 (fractions, i.e. 6573.3907% and 12835.6817%). These reflect the saved
large finite coefficients passed unchanged to the native radial-shift renderer.
They were not clipped, refitted or removed. The full machine-readable tables retain
extremes as well as medians; a handful of selected diagnostic images is not the
basis for the population conclusions.

`paired_changes.csv` contains every arm/default and mode/control pair, including
unavailable pairs. `paired_summary.csv` gives roster sizes, finite denominators,
lower-value counts, ties and median differences. Lower signed bias is not
necessarily better; use absolute bias for accuracy. `galaxy_consistency.csv` groups
conditions within galaxies. `scenario_summary.csv` and scenario rows of
`primary_wins.csv` retain the nine conditions separately. Per-tool descriptive
summaries use each tool's available outcomes; three-tool tables explicitly match
participants. No new uncertainty intervals or independent-condition significance
claims are made. Historical galaxy-block bootstrap results remain labelled with
their original baseline aperture.

## QA, execution and audit

Every input has four primary cross-tool pages, one for each mode, plus six
cross-arm pages (three tools, linear baseline and native), in both PNG and PDF,
with separate captions and exported display tables. Cross-tool residuals show full
finite Data-minus-model coverage without evaluation-cut masks. Cross-arm residual
thumbnails retain the common metric aperture. Cross-tool color limits pool finite
residuals inside the matched aperture over all available modes/arms for that input.
Ellipses label inner/outer limits; there is no bad-pixel overlay on ideal mocks or
I=0 surface-brightness reference. Valid native profiles and uncertainties remain
visible when only a reconstruction is unavailable. Captions identify all failures.

Validation included ordinary, difficult, failed-primary, empty-inner-zone, real EA
and separately copied missing-coefficient cases. Individual gate outputs took less
than a minute; the four-input gate produced 40 page pairs with 244 reproduced
baseline zone rows. Synthetic angular/sign/normalization checks and the complete
unit suite passed: 665 tests, ten existing warnings. An independent recomputation of 1,282 nonempty arm/mode/zone measurements
matched the final files across six inputs, including an extreme simultaneous-EA
case, a failed primary and a missing primary harmonic input.

The final audit verified all 53,132 original baseline zone measurements,
all display-table metric values, 33,480 per-input product hashes,
36,153 current source hashes and 30,862 historical source hashes. The full delivery
inventory also includes completion records and run-root provenance. The run manifest
freezes the roster, modes, common-support policy, page coverage and code hashes.

The final run used 12 process workers with one numerical-library thread per worker.
Measured manifest-to-index time was 1,658 seconds; per-input wall times ranged from
15.4395 to 163.4550 seconds. Maximum recorded worker lifetime peak memory was
8,409,546,752 bytes. Per-input products total 7,924,009,243 bytes. Arrays were kept
in memory; matched-support FITS, numerical tables, figures and hashes were saved,
while full model FITS can be reproduced from the unchanged source profiles.
The earlier interrupted run and all gate outputs remain preserved.

`inherited_fit_times.csv` reports accepted parallel-campaign fit times. Neither
those times nor reconstruction/load durations replace controlled timing benchmarks.
Python 3.12.11, numpy 2.3.5, scipy 1.17.0, photutils 2.3.0, astropy 7.2.0,
matplotlib 3.10.8 and pandas 2.3.3 were recorded. The isolated feature branch is
`feature/huang-final-analysis-20260923`; inherited HEAD remains `9a04906`.

Limitations remain smooth prepared mock images, one noise draw per condition,
correlated conditions within each galaxy, native sampling/estimator differences,
finite support and unrecorded historical AutoProf optimizer seeds. Shared EA
intensity-harmonic reconstruction remains explicitly unsupported; native EA was
validated and included. No fitting defaults, physical PSF, reconstruction algorithm,
S4G campaign or controlled speed benchmark was changed.
