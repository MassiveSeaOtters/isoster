# Default Isoster versus AutoProf: center and radial extent

This saved-fit investigation covers 837 Huang2013 inputs on
`analysis/autoprof-comparison-20260923`. It identifies measured opportunities,
not approved default changes. No fitting, historical-result modification,
merge or push was performed. S4G remains deferred.

## Main findings

AutoProf estimates one global center, then fixes it during geometry fitting and
intensity extraction. Isoster allows each isophote to move. AutoProf generally
**extracts farther but fits geometry less far**: among 820 successful primary
pairs, the median AutoProf/Isoster saved-profile radius ratio is 1.4924, whereas
the geometry-fit/Isoster-attempted-radius ratio is 0.6548. These are different
quantities, not competing measures of successful geometry fitting.

All 837 Isoster profiles reach the last configured radial step. Outer failures
do not terminate the driver. The radial difference is primarily a difference
between the campaign's Isoster radius cap and AutoProf's extended extraction,
not evidence of premature Isoster termination on a gradient failure.

**Correction to the handover:** `ap_truncate_evaluation=True` and
`ap_extractfull=False` belong to the conditional retry, not every baseline run.
706 accepted AutoProf primaries use ordinary options; 114 successful primaries
use retry options; 17 retry cases remain errors. All 837 pass a truth-centered
initial guess, none pass `ap_set_center`, and all fix the prepared background
to zero. The ordinary installed defaults are truncation off and extract-full off.

## Evidence and source identity

- Accepted selection: `huang2013_scientific_analysis_two_pixel_2026_09_23/fit_metrics.csv`
  under the publication campaign's `analysis` folder. Every designated primary
  is retained, including errors. No successful recovery arm replaces a primary.
- All Isoster primary records come from `publication_single_band_round1`, whose
  environment records Isoster 1.0.0, Python 3.12.11 and commit `4176f72`.
  `isoster/{driver,fitting,config,sampling}.py` are byte-identical between that
  commit and this checkout. Saved configurations, not just current defaults,
  establish the campaign choices below. Every input has no variance map and
  no bad-pixel mask, so these are unweighted fits despite variance support.
- AutoProf campaign YAML selects `~/.venvs/autoprof_venv/bin/python`. The exact
  installed package is 1.3.4 in that environment. The worker explicitly chooses
  background, PSF, center, initialization, geometry fit, extraction, checkfit,
  ellipse model, and profile writing in that order.
- Current installed AutoProf sources and metadata are copied and SHA-256 hashed
  in `outputs/autoprof_source_audit_20260923/`. The campaign's environment file
  describes its parent process, not the external AutoProf interpreter. Historical
  byte identity of that external installation is **not established** by this
  audit; current installed semantics are corroborated by saved options and outputs.
  Do not describe the source copy as a recovered September 21 binary snapshot.
- Population diagnostics took 119.8685 seconds after a three-input, actual-output
  gate took 1.5156 seconds. These are analysis times, not fitting benchmarks.
  All 9,140 consulted input-file hashes were unchanged after the population run;
  4,989 overlap the final accepted analysis's source hashes and match exactly.

## Source-linked behavior

AutoProf links below refer to the inspected installation; frozen copies are in
the source-audit folder named above. Line numbers refer to that inspected source.

| Topic | Isoster accepted primary | AutoProf accepted primary |
|---|---|---|
| Initial center | Adapter's `(width-1)/2, (height-1)/2`; campaign passes it into the fit. [Adapter](../../../benchmarks/exhausted/adapters/huang2013.py#L122), [wrapper](../../../benchmarks/exhausted/fitters/isoster_fitter.py#L265). | Same truth-centered guess; no truth-fixed constraint. [Options](../../../benchmarks/exhausted/fitters/autoprof_fitter.py#L523), [worker](../../../benchmarks/exhausted/fitters/autoprof_worker.py#L47). |
| Center optimization | `fix_center=False`, largest-harmonic update; first-order sine/cosine terms move the center along the ellipse axes. Damping 0.7, shift clipped to 5 pixels per iteration; this is not an absolute bound on accumulated drift. [Updates](../../../isoster/fitting.py#L1869). | `Center_HillClimb`: circular rings, clipped FFT phase direction, hill climbing, then Nelder–Mead refinement; one center returned. Geometry fitter copies it and never optimizes it. [Center](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Center.py:735), [fit](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Isophote_Fit.py:581). |
| Geometry propagation / freeze | Codes 0, 1, 2 propagate; failed gradient/too-few-point rows remain, but do not replace the previous accepted seed. No automatic lock, outer regularization, permissive geometry or fixed center in any primary. [Driver](../../../isoster/driver.py#L694). | Global center fixed; neighboring ellipticity/PA solutions coupled through regularization (scale 1). Stochastic perturbations update geometry across radii. Native optimizer reseeds inside `Process_Image`; fixed input-noise seeds do not fix this optimizer. [Loss](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Isophote_Fit.py:231), [pipeline](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/Pipeline.py:152). |
| Initial failures | Start at 6 pixels; up to three perturbation retries and initial growth probes. Codes 1/2 are acceptable for propagation, not proof of convergence. [Driver](../../../isoster/driver.py#L561). | On recognized failure, retry once with image-size-dependent centering-ring count, truncation on, extract-full off. Saved `.attempt1` logs remain evidence. [Retry](../../../benchmarks/exhausted/fitters/autoprof_fitter.py#L209). |
| Geometry radii | Geometric factor 1.1, inward to 0.5 pixels plus central sample, outward to explicit `maxsma`; cap comes from input preparation. [Driver](../../../isoster/driver.py#L549). | Geometry grid starts at `max(1, PSF/2)`; default `ap_scale=0.2` gives factor 1.2, reduced when fewer than 15 radii. Stops grid construction below twice estimated background noise only beyond initialization's minimum radius, or at half the maximum image dimension. Threshold-crossing/overshooting endpoint is included. [Grid](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Isophote_Fit.py:498). |
| Local stopping | Default max 50/min 6 iterations, gradient-relative-error threshold 0.5, unreliable outward gradient on repeated checks yields -1. This exits one isophote fit, not outward growth. [Gradient](../../../isoster/fitting.py#L1545), [growth](../../../isoster/driver.py#L703). | Iterations stop after sufficient unchanged trials or installed default 1000 cycles. Do not substitute the different value in the function's documentation. Standardized profile `stop_code=0` is wrapper-generated, not native convergence evidence. [Loop](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Isophote_Fit.py:573), [adapter](../../../benchmarks/exhausted/fitters/autoprof_fitter.py#L751). |
| Extraction extent | Intensity and geometry stored on the same attempted radial grid, including failed rows. Driver does not silently enter a separate frozen-geometry extraction phase. | Separate factor-1.1 grid; ordinarily grows until 3 times fit radius or maximum image dimension divided by sqrt(2). Checks precede appending, so endpoint can overshoot. Ellipticity/PA splines use constant endpoints beyond fit support. Retry truncation counts two non-positive samples cumulatively, not consecutively. Harmonic interpolation occurs before truncation and can fail on an empty ring. [Extraction](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Isophote_Extract.py:718), [truncation](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Isophote_Extract.py:200). |
| Extraction statistic | Mean, symmetric 3-sigma clipping once; no variance map. Lazy gradient evaluation is enabled. | Median, 5-sigma clipping; installed iteration default 10. At low previous-ring signal and sufficient ring width, uses an annular band (half-width 0.025 radius) rather than a thin ring. [Extraction](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Isophote_Extract.py:89). |
| Saved center / radius precision | Per-row floating-point centers retained. | Auxiliary center and geometry-fit endpoint are printed to two decimals. Wrapper stamps the rounded center on every row. Raw `.prof` radius is converted from arcsec to pixels; rows with `SB >= 90` are dropped. [Parser](../../../benchmarks/exhausted/fitters/autoprof_fitter.py#L694), [center stamping](../../../benchmarks/exhausted/fitters/autoprof_fitter.py#L317). |
| Reconstruction support | Shared linear renderer accepts finite positive-radius geometry/intensity, including finite failed-fit rows; fills outside with NaN in this analysis. [Model](../../../isoster/model.py#L108). | Same renderer for this diagnostic. Native AutoProf zero-filled exterior is a different support convention and native model remains ellipse-only; measured intensity harmonics are not applied. [Native exterior](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/Ellipse_Model.py:230). |

AutoProf's pipeline uses an assumed PSF scale of 4 pixels unless overridden;
this is an algorithmic length scale, not a measurement of the physical mock PSF.
See [PSF_Assumed](/Users/shuang/.venvs/autoprof_venv/lib/python3.12/site-packages/autoprof/pipeline_steps/PSF.py:121).
The evaluation cut stays at 2 pixels and never follows that scale.

## Center measurements

Offsets below are Euclidean pixel distances from the injected center. Per-profile
median/max use rows at SMA >= 2 pixels, including finite failed rows. AutoProf
has one rounded center for all rows. These are descriptive summaries, not matched
precision estimates or measurements of an unrecorded optimization trajectory.

| Statistic | Isoster (837 successes) | AutoProf (820 successes) |
|---|---:|---:|
| Median of per-profile median offsets, pixels | 0.02188 | 0.00000 |
| 95th percentile of median offsets, pixels | 0.14719 | 0.04472 |
| Median of per-profile maximum offsets, pixels | 1.36322 | 0.00000 |
| 95th percentile of maximum offsets, pixels | 5.39038 | 0.04472 |
| Median last-row offset, pixels | 1.08705 | 0.00000 |

528 AutoProf successes have a saved zero offset; sub-0.01-pixel behavior cannot
be resolved from the rounded auxiliary output. Isoster's maximum offset exceeds
1 pixel in 541 inputs. Its noiseless median maximum offset is only 0.00143 pixels,
versus 2.77460 pixels in `wide_z005`: outer freedom reacts strongly to noise.

This does **not** demonstrate that centering dominates the RMS difference.
Within each noisy scenario, Spearman correlations between Isoster maximum offset
and outer RMS range from -0.0645 to +0.3070; correlations with full RMS are negative
(-0.5817 to -0.2197). Galaxy size, profile shape and relative-error normalization
confound these associations. Noiseless full-RMS correlation is +0.4714. The saved
AutoProf fixed-center alternative improves full RMS in 429/819 successful pairs,
but its median full RMS worsens from 4.6479% to 4.8342%; it also changes the
stochastic optimizer realization. Neither observation isolates a causal center effect.

## Three distinct radial extents

1. Geometry endpoint: AutoProf's rounded auxiliary fit limit; Isoster's last
   attempted isophote. Isoster's last converged and last propagation-accepted
   radii are separately retained in the table.
2. Intensity endpoint: raw extracted radius and standardized positive-SB profile
   endpoint are both retained for AutoProf. 305 successful profiles lose 463
   raw rows in the SB filter. Isoster retains its attempted profile rows.
3. Reconstructed endpoint: maximum **truth-frame elliptical radius** of finite
   pixels in the shared linear harmonic-off model, over the full image. This is
   not a local fitted SMA, a complete circular aperture, or a quality score.
   Finite pixel count is also saved. Native and spline endpoints are not measured
   here; the four-mode accepted support/quality tables remain authoritative.

Among 820 successful pairs, AutoProf has the larger standardized profile radius
in 780 cases, larger geometry endpoint in 104, and larger shared harmonic-off
finite endpoint in 758. Median ratios are respectively 1.4924, 0.6548 and 1.4205.
Every successful AutoProf raw extraction extends beyond its geometry endpoint;
median raw-extraction/geometry ratio is 2.3412.

| Scenario | Successful pairs | Median profile-radius ratio | Median geometry/attempted-radius ratio |
|---|---:|---:|---:|
| noiseless_z005 | 91 | 1.4924 | 0.9274 |
| wide_z005 | 91 | 1.4924 | 0.7665 |
| deep_z005 | 91 | 1.4924 | 1.0034 |
| wide_z020 | 93 | 1.4924 | 0.6154 |
| deep_z020 | 90 | 1.4924 | 0.8329 |
| wide_z035 | 92 | 1.3567 | 0.4879 |
| deep_z035 | 92 | 1.4924 | 0.7145 |
| wide_z050 | 92 | 1.1212 | 0.4033 |
| deep_z050 | 88 | 1.4924 | 0.5906 |

Isoster's last row is at the final allowed factor-1.1 grid point in all 837 cases;
median last-radius/maxsma is 0.94797. Last-row codes are 0 in 512, -1 in 316 and
2 in nine. Thus 325 profiles continue beyond the last converged row. Do not call
all 837 terminal geometries successful just because all run records say `ok`.

Residual associations and saved-arm comparisons use the accepted original
harmonic-off common support with the fixed 2-pixel cut. No residual measurements
were extended to the additional AutoProf-only pixels. Greater extent has not
been shown to provide better photometry there.

## Measured opportunities and limits

**Separate a geometry-quality endpoint from an extraction endpoint.** The current
driver already continues after geometry failures, so removing a nonexistent early
stop would solve nothing. A possible future experiment would preserve the default
profile and separately extract with explicitly frozen, quality-qualified geometry
beyond its cap. Existing `lsb_autolock` is the first implementation to study, not
a reason to add another locking framework. This requires a fitting/extraction
experiment decision; no quality or runtime gain is established by extent alone.

**Investigate center stabilization selectively.** Noise-related movement is real,
but always freezing the center is not supported by the comparisons. Existing
outer damping improves outer RMS in 477/837 pairs, while the population median
outer RMS increases from 3.9705% to 4.4320%. Its median paired change is -0.1597
percentage points: paired-change medians and differences of population medians
answer different questions. Preserve both benefits and regressions. Investigate
which cases benefit before changing any trigger or default.

**Isolate radial interpolation bias before changing the extraction statistic.**
A controlled analytic test uses exact pixel-center exponential intensities,
fixed ellipticity 0.3, no noise, no fitting, no harmonics, and identical 2-to-35
pixel support (2,690 pixels). With radial scales of 2 and 8 pixels, shared linear
rendering on factor-1.1 grids produces signed flux biases of +0.6320% and +0.3216%.
Halving the fractional spacing to 0.05 reduces them to +0.1580% and +0.08468%.
The corresponding full RMS values fall from 0.3640%/0.2229% to 0.08841%/0.05923%.
The assertion-based test confirms nonnegative interpolation residuals for these
convex profiles. It demonstrates one rendering contribution without center or
extraction error; it does not explain the full campaign bias, PSF convolution,
pixel integration, or harmonic radial shifts. Denser *synthetic* samples here
are exact new information; interpolating existing sparse samples cannot reproduce
that information. Do not infer that increasing real fitting density is free.

Changing Isoster from mean to median is not a demonstrated general solution:
the saved median arm improves full RMS in 480/837 pairs but outer RMS in only
392/837 and absolute flux bias in 413/837. Matched-support renderer rankings
already reverse in the accepted four-mode report. Next isolate extracted intensity
against truth on the same saved ellipses and cumulative flux before selecting
a renderer or estimator change.

Acceptance for any later proposal: retain the original default and each renderer's
harmonic-off control; compare common-pixel ALL/INNER/MIDDLE/OUTER RMS and signed
flux bias; retain empty zones and failures; report per-scenario and per-galaxy
benefits/regressions; measure repeated runtime on a controlled small sample.
Require evidence of improved target accuracy without hiding coverage loss or
other-zone regressions. No equal-quality speedup claim follows from this analysis.

## Artifacts, validation and remaining work

- Implementation: `benchmarks/exhausted/analysis/autoprof_center_extent.py`;
  one unit check retains failed outer rows and verifies the 2-pixel center cut.
- `outputs/autoprof_center_extent_gate_20260923/`: three-input gate.
- `outputs/autoprof_center_extent_population_20260923/`: 1,674 rows, paired
  radii, scenario/galaxy medians, correlation tables, 17 failed primaries,
  selected cases, saved-alternative comparisons, summary and input hash audit.
- `outputs/autoprof_source_audit_20260923/`: exact inspected source copies,
  hashes, environment provenance and Isoster historical source comparison.
- `outputs/autoprof_analytic_interpolation_20260923/`: runnable analytic check
  and measurements. Pixel-center truth deliberately excludes pixel integration.

Recorded follow-up cases: ordinary noiseless NGC1521, large terminal Isoster
offset NGC3258/wide_z005 (9.2451 pixels), largest saved AutoProf offset
NGC6673/wide_z050 (0.22361 pixels), and retained AutoProf error
IC2311/noiseless_z005. Their selection is numerical; detailed visual diagnosis
and causal saved-ellipse tests remain open.

This completes the initial center/stopping trace and population measurement.
Native/spline extent measurement, cumulative flux decomposition, controlled
same-ellipse extraction, outer harmonic-gradient analysis and implementation
decisions remain open. Shared polar EA-on remains unsupported; native EA and all
historical failed/unsupported outcomes are preserved. S4G is not ready to resume.
