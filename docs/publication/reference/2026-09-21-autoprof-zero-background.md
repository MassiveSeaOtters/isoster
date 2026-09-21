# Huang2013 AutoProf zero-background replacement and analysis

## Scope and current status

The user approved fixing AutoProf's background to the known injected value,
`ap_set_background=0.0`, and repeating the full Huang2013 scientific analysis.
This compares extraction from prepared images, not an automatic sky-estimation
pipeline. S4G will receive the same policy in a later campaign.

The small fitting and analysis-environment gates passed. All 3,348 requested
Huang2013 AutoProf fits have finished and the final audit passed at
2026-09-21 15:49:06 +0800. The full scientific reanalysis and figure export
are complete. This is the current prepared-background Huang2013 reference;
the [September 12 analysis](2026-09-12-huang2013-scientific-analysis.md)
remains a historical estimated-background comparison, not the primary result.

Replacement means explicit selection of the new AutoProf campaign. All old
images, fitted profiles, auxiliary files and analyses remain unchanged.
Isoster and Photutils are not refitted. The fixed-center AutoProf arm reads
copies of the accepted Isoster reference profiles.

## Conditions and reproducibility

The campaign retains the previous 93 galaxies, nine scenarios, four AutoProf
arms (`baseline`, `deep`, `high_regularization`, `fix_center`) and corrected
PA conversion: 837 images and 3,348 requested fits. Requested fitting options
other than background and filesystem paths must equal their PA-corrected
predecessors. The pre-existing conditional retry is verified separately:
its image-size-dependent values and first-attempt failure signature must
match the recorded retry. A changed sky can change which cases trigger it.
The noise-estimation algorithm is unchanged; its measured
output is not required to equal the old estimate.

Code is isolated from Dropbox in
`/Users/shuang/code/isoster-autoprof-zero-background-20260921-remote`, branch
`benchmark/autoprof-zero-background`, implementation commit `6a4099a`.
The independent uv environment is `/Users/shuang/.venvs/isoster-zero-background`
(Python 3.12.11); AutoProf still uses
`/Users/shuang/.venvs/autoprof_venv/bin/python`. Eight galaxy workers are used,
with BLAS, OpenMP, MKL, Accelerate and NumExpr threads each limited to one.
`caffeinate -i` prevents idle sleep. These are throughput runs, not a repeat
of the controlled Stage 4 timing experiment.

The measured host is an Apple M1 Ultra with 20 logical CPUs and 128 GiB RAM,
running macOS 15.7.3 (24G419). Monitored thermal checks reported no warning
and swap usage remained zero. AutoProf's recorded Python/package metadata
exactly matches the previous PA-corrected campaign. Differences in the outer
environment are absent documentation/web-support packages, not changed
scientific-library versions.

All directories below are children of
`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28`:

- Source: `fits/publication_autoprof_pa_huang_2026_09_11`.
- Gate: `fits/publication_autoprof_zero_gate_2026_09_21`.
- Gate figures: `analysis/autoprof_zero_gate_qa_2026_09_21`.
- Full replacement: `fits/publication_autoprof_zero_huang_2026_09_21`.
- Environment-only analysis check (old PA selection, not new science):
  `analysis/huang2013_zero_background_analysis_environment_gate_2026_09_21`.

Invoke the existing correction entry point with the tracked new configuration:

```bash
uv run python -u -m benchmarks.exhausted.campaigns.run_autoprof_pa_correction \
  benchmarks/exhausted/configs/campaign.publication_autoprof_zero_huang_2026_09_21.yaml \
  --zero-background \
  --source-campaign /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/fits/publication_autoprof_pa_huang_2026_09_11
```

Set the environment and thread limits described above before invocation.
The runner rejects an existing destination. Its `correction_audit` directory
records resolved options, source hashes, dependency centers, environments,
accepted outcomes and completion verdicts. Scientific failures are retained;
no best-of-run selection is allowed.

The first full-run audit implementation incorrectly compared final retry
options as if retry triggers had to remain identical. This was caught during
monitoring: IC2311/wide_z020/fix_center activated the unchanged retry in the
zero-sky run only. The corrected audit checks the requested recipe and retry
policy separately. The already-running fitting process is unchanged and
finished with its original strict audit error; then `--audit-only` on the
same command verified all retained records without refitting. It rejects an
already completed audit, a changed configuration or a changed source roster.
In audit-only mode, elapsed fitting time is left null rather than fabricated.

The retained final audit was executed at commit `e45f91d`. All 14,230 source
hashes match the original before-run values. All 837 copied reference centers
match; all requested AutoProf options except sky/paths match; saved PA and
fixed-center checks pass; all saved backgrounds and all successful auxiliary
backgrounds are zero. Retry triggers agree for 3,330 records (2,951 neither,
379 both); 16 newly trigger the same verified retry and two no longer do.

### Complete fitting outcome

| AutoProf arm | Requested | Successful | Retained errors |
|---|---:|---:|---:|
| `baseline` | 837 | 820 | 17 |
| `deep` | 837 | 835 | 2 |
| `high_regularization` | 837 | 821 | 16 |
| `fix_center` | 837 | 821 | 16 |
| Total | 3,348 | 3,297 | 51 |

All errors retain the native `error` status; they are not skipped, silently
repaired or replaced by the previous successful estimated-background fit.
There are 48 final `isophoteextract` empty-sample errors. Three final
`isophotefit` initialization errors occur for NGC4742/wide_z050 in baseline,
deep and high-regularization arms; its fixed-center arm succeeds. First-attempt
retry signatures are not necessarily the final failure stage.

Failures by galaxy are IC2311 (11), NGC1379 (13), NGC1407 (3), NGC2434 (3),
NGC4742 (3), NGC7144 (6), NGC7145 (3), NGC7192 (3) and NGC7507 (6). Six
errors are noiseless and 45 are noisy. Full scenario/arm/stage/log details are
in `correction_audit/failure_details.csv`. The earlier PA-corrected campaign
had 3,348 successes; the execution difference is part of this result, not
removed from the scientific denominator.

Early full-run failures in IC2311 are real AutoProf extraction failures:
an outer ellipse has no unmasked image samples and interpolation raises
`array of sample points is empty`, including after its existing retry.
They are retained as failures, not replaced by another arm or tuned rerun.
Inspection of installed AutoProf 1.3.4 explains why the retry is not a full
remedy: `ap_truncate_evaluation` stops after two non-positive intensity
samples, not when an ellipse leaves the image, and harmonic interpolation
of an empty ring happens before that test. With zero adopted sky, positive
galaxy wings need not encounter that stop condition. We corrected the
wrapper's misleading comment but did not change the retry or disable
harmonic extraction. This is an extraction-boundary failure, not evidence
that subtracting a positive sky is scientifically appropriate.

## Gate results

All four arms succeeded for IC2597/wide_z005, NGC1209/noiseless_z005 and
NGC4742/deep_z050: 12/12 successful. All saved and reported backgrounds are
zero, reference centers match exactly, all other fitting options match, and
source hashes are unchanged. The recorded fitting/audit phase took 55.564 s,
excluding source-selection startup. Forty-four focused regression tests passed.

The three PNG/PDF before/after diagnostics reuse the existing PA-correction
QA function with accurate background labels, shared residual scales and a
common no-harmonic reconstruction aperture. Visual inspection found no new
execution pathology. The compact high-redshift case retains large ellipticity
changes, not silently discarded as bad data. Zero background is the known
input condition, not a promise that every individual residual must improve.

The installed AutoProf `Background_Mode` estimates noise from pixels below
the adopted background. If that estimate is non-finite, its existing fallback
uses half the 16th--84th percentile span of sampled intensities. Thus a
positive noiseless galaxy with zero adopted sky still receives a nonzero
internal fitting scale (0.012 ADU for the NGC1209 gate), not an injected noise
measurement. We did not add noise, change that fallback or set
`ap_set_background_noise`. This remains a noiseless-fitting caveat.

The frozen generator was recloned outside Dropbox and checked out at
`a6a90a07dc3aedd95465928ee2e93258c8ccb40a`. A one-galaxy environment check
processed 153 NGC1209 records successfully before the full new analysis.

## Analysis caveats

Recompute all selected metrics, matched primary comparisons, paired Isoster
contrasts, truth/seed checks, source hashes and figures in new directories.
Because the common aperture intersects every successful arm's finite model,
changing AutoProf can also change the metrics of unchanged Isoster/Photutils
fits. Report this explicitly in old/new comparisons. The no-harmonic renderer,
one noise draw per galaxy/scenario, auxiliary-center rounding, descriptive
harmonics and non-comparable AutoProf stop codes remain limitations.

AutoProf 1.3.4 `Pipeline.Process_Image` reseeds NumPy using a random integer,
process ID and wall-clock fraction. Its ellipse optimizer uses random
shuffles and perturbations. The wrapper does not replace this behavior or
record the resulting seed. Thus the input noise realization is exactly
reproduced, but the old and new fits do not share an optimizer realization.
Old/new differences include that variation, changed noise estimates, changed
retry triggers and changed common support; they are not a deterministic
one-factor decomposition of the sky offset. We retained one campaign rather
than selecting the best stochastic fit. Controlled repeatability would need
a separate, explicitly designed experiment.

## Full scientific selection and verification

The [original measurement contract](../../specs/2026-09-10-huang2013-scientific-analysis.md)
is unchanged. The sample remains 93 centered multi-component Sersic models,
each in nine conditions: truly noiseless z=0.05 and wide/deep Gaussian-noise
images at z=0.05, 0.20, 0.35 and 0.50. Pixel scale is 0.168 arcsec, Gaussian
PSF FWHM is 0.7 arcsec and zeropoint is 27.0. No new mock images were used.
These are exploratory smooth-model tests, not held-out survey-image validation.

The final manifest has 14,229 unique galaxy/scenario/tool/arm keys. All
10,881 non-AutoProf selections exactly match the previous manifest in key,
status and source path. Only AutoProf is replaced by the newly audited
campaign. The 837 copied reference dependencies are not added science fits.
The excluded-record table retains 3,351 documented original/recovery overlaps
and superseded original AutoProf records; the September 11 PA campaign is
the audit predecessor, not an additional candidate in that original join.

| Tool | Successes | Failures/errors | Intentional skips |
|---|---:|---:|---:|
| Isoster, nine executable arms | 7,533 | 0 | 0 |
| Isoster, `ols_noweight` | 0 | 0 | 837 |
| Photutils, three arms | 2,453 | 58 | 0 |
| AutoProf, four arms | 3,297 | 51 | 0 |
| Total | 13,283 | 109 | 837 |

Photutils' retained failures and Isoster's deliberate no-weight skips are
unchanged. AutoProf's `error` records and Photutils' `failed` records both
count as unsuccessful executions; their native labels are preserved.

All 372 regenerated PSF-convolved truth FITS were produced with the frozen
MockGal revision and the same libprofit binary (SHA-256
`dee3b2445feaba39a89ccd1babb86077ed0c8a988e87f7cb38069aeb88aa88ac`).
All 837 stored inputs reproduce exactly at float32 precision, and all 837
seed strings match the prior verified analysis, including blank noiseless
seeds. Every galaxy passed before/after source checks; an independent final
recheck also verified all 30,862 source hashes. No numeric metric is infinite.
Every successful fit has finite all-aperture and outer-zone truth RMS.
Inner-zone RMS is unavailable for 4,648 successful records and middle-zone
RMS for 253 because the relevant finite aperture is absent; these stay N/A.

The analysis uses the same no-harmonic elliptical reconstruction for all
tools. Its common aperture is the intersection of all successful arms'
finite models, outside one PSF FWHM and inside the initial maximum SMA.
Truth-relative RMS is `sqrt(mean((M-T)^2))/sqrt(mean(T^2))`; aperture flux
bias is `sum(M-T)/sum(T)`. Neither is an extrapolated total-flux measurement.
Noisy data residual RMS is divided by injected sigma, not fitted AutoProf
noise, and is not reduced chi-square. Radial reach is not a detection claim.

Common support fraction has median 0.78455, p16/p84 0.67650/0.88088 and
minimum 0.097865. The previous median was 0.71326. The support fraction
changes in 542 of 837 images; equal pixel counts do not prove identical
masks. Thus unchanged Isoster/Photutils fits can receive changed image-based
metrics. Comparisons across redshift remain conditional on different coverage.

## Primary three-tool results with prepared sky

Primary arms remain Isoster `ref_default`, Photutils `baseline_median` and
AutoProf `baseline`. Each row uses the same successful finite galaxy set
for all three tools. Values are median all-aperture truth-relative RMS in
percent, not success percentages or median per-pixel fractional errors.

| Scenario | Matched galaxies | Isoster (%) | Photutils (%) | AutoProf (%) |
|---|---:|---:|---:|---:|
| Noiseless z=0.05 | 65 | 0.8551 | 0.9027 | 0.6470 |
| Wide z=0.05 | 91 | 0.8583 | 0.8710 | 0.6222 |
| Deep z=0.05 | 91 | 0.8369 | 0.8692 | 0.5778 |
| Wide z=0.20 | 93 | 1.7254 | 1.8322 | 1.4329 |
| Deep z=0.20 | 90 | 1.4132 | 1.5050 | 1.0347 |
| Wide z=0.35 | 92 | 2.9524 | 3.3023 | 2.9116 |
| Deep z=0.35 | 92 | 1.9557 | 2.1407 | 1.7980 |
| Wide z=0.50 | 92 | 5.0787 | 5.1811 | 4.6382 |
| Deep z=0.50 | 88 | 2.6881 | 3.0063 | 2.8303 |

On these matched successful samples AutoProf has the smaller median
all-aperture RMS in eight of nine conditions; Isoster is smaller at deep
z=0.50. This reverses much of the earlier estimated-background comparison.
It is not a universal ranking: AutoProf's primary fails in 17 conditions,
all Isoster reference fits succeed, radial zones differ, and distributions
overlap. Successful-fit accuracy and execution coverage must both be shown.

For example, outer-zone RMS at wide z=0.05 is 1.878%, 1.669% and 2.873%
for Isoster, Photutils and AutoProf respectively; at wide z=0.50 it is
15.074%, 13.094% and 13.351%. AutoProf's smaller inner-zone medians do not
imply better outer recovery in every scenario. Zone-specific matched counts
are recorded in `02_primary_truth_zones.csv`, not silently assumed to be 93.

Signed aperture-flux biases at wide z=0.05 are +0.513%, +0.713% and +0.399%;
at wide z=0.50 they are +1.093%, +2.367% and +0.861%. The former large
negative AutoProf biases are no longer present in these medians. At wide
z=0.50 data/sigma RMS medians are 0.9944, 0.9982 and 1.0008: near-noise
residuals still do not establish perfect truth recovery.

AutoProf auxiliary centers remain rounded to 0.01 pixel and represent a
global center, unlike radially varying centers in the other primary tools.
Median center errors of zero are not unlimited subpixel accuracy. Its stop
code placeholders remain non-comparable and are omitted from convergence
markers. Native profile errors and harmonic diagnostics are not newly
calibrated uncertainties or cross-tool harmonic truth.

### Old/new AutoProf comparison on the same successful galaxies

The following table uses the intersection of old and new matched primary
samples. It therefore differs slightly from the published historical table,
which included galaxies that now fail. Each version still uses its own
all-arm aperture; the random optimizer realization also differs.

| Scenario | Galaxies | Estimated-sky RMS (%) | Zero-sky RMS (%) |
|---|---:|---:|---:|
| Noiseless z=0.05 | 65 | 0.8857 | 0.6470 |
| Wide z=0.05 | 91 | 1.0763 | 0.6222 |
| Deep z=0.05 | 91 | 0.8361 | 0.5778 |
| Wide z=0.20 | 93 | 2.7882 | 1.4329 |
| Deep z=0.20 | 90 | 2.0425 | 1.0347 |
| Wide z=0.35 | 92 | 4.8219 | 2.9116 |
| Deep z=0.35 | 92 | 3.3252 | 1.7980 |
| Wide z=0.50 | 92 | 6.6940 | 4.6382 |
| Deep z=0.50 | 88 | 4.6607 | 2.8303 |

For those same samples, AutoProf's wide z=0.05 median signed flux bias
changes from -3.311% to +0.399%; at wide z=0.50 from -9.292% to +0.861%.
The comparison CSV includes median paired differences as well as differences
between population medians; these are not interchangeable statistics.
Use these as descriptive policy comparisons, not an isolated causal sky
correction or evidence that every individual fit improved.

## Recomputed Isoster contrasts and structural diagnostics

Every all-aperture contrast retains 837 pairs from 93 galaxies. Differences
are arm minus reference in absolute fractional RMS units. Intervals are
95% galaxy-block bootstrap intervals from 2,000 draws, seed 20260910, with
all scenarios for a sampled galaxy kept together. They remain exploratory
and unadjusted for multiple comparisons.

| Arm minus reference | Median RMS change | 95% interval | Smaller RMS (%) |
|---|---:|---|---:|
| `geom_ea` - `ref_default` | -7.475e-5 | [-1.479e-4, -4.056e-5] | 63.9 |
| `geom_simul` - `ref_default` | -2.409e-8 | [-1.387e-6, +1.579e-6] | 50.1 |
| `geom_simul_ea` - `ref_default` | -8.164e-5 | [-1.566e-4, -4.982e-5] | 64.3 |
| `geom_simul_ea` - `geom_ea` | -7.638e-9 | [-1.008e-6, +1.089e-6] | 50.1 |
| `geom_simul_ea` - `geom_simul` | -6.775e-5 | [-1.389e-4, -3.250e-5] | 63.4 |
| `harm_simul_ea` - `geom_simul_ea` | +9.060e-7 | [-4.512e-7, +2.622e-6] | 47.3 |
| `harm_simul_ea` - `ref_default` | -6.361e-5 | [-1.207e-4, -3.275e-5] | 62.6 |
| `int_median` - `ref_default` | +1.936e-4 | [+1.532e-4, +2.532e-4] | 29.4 |
| `lsb_autolock` - `ref_default` | 0 | [0, 0] | 20.1 |
| `reg_outer_damp` - `ref_default` | -4.435e-5 | [-1.456e-4, +4.530e-7] | 56.6 |
| `stack_all` - `ref_default` | -1.308e-5 | [-8.882e-5, +1.606e-5] | 53.5 |

The earlier qualitative Isoster conclusions survive the aperture change.
EA geometry has a small favorable pooled RMS shift, not a universal gain.
Its median absolute aperture-flux error increases by 1.083e-4 (interval
+2.945e-5 to +1.985e-4). Simultaneous harmonics within EA still have no
clearly separated pooled RMS shift, although median center error decreases
by 0.001116 pixel. The common renderer excludes harmonics, so this is not
a test of the benefit of adding high-order terms to a reconstructed image.

Outer damping lowers paired center error by 0.03710 pixel, with smaller
error in 90.3% of pairs. After taking each galaxy's median over scenarios,
population median outer ellipticity roughness is 0.02470 for the reference
and 0.001041 for damping; PA roughness is 3.326 versus 0.009813 degrees.
Smoothing is not automatically more faithful recovery of a real twist.

Structural plots remain descriptive, using one wide z=0.05 image per galaxy.
They show 93 successful Isoster and Photutils primaries and 91 AutoProf
primaries; their individual-tool structural bins are not matched cohorts.
Isoster RMS medians across increasing resolution quartiles are 0.01076,
0.00940, 0.00761 and 0.00657. Across increasing initial-ellipticity quartiles
they are 0.00696, 0.00771, 0.00949 and 0.01172. Component-PA-span quartiles
are not monotonic. These parameters were not independently varied.
Centered Sersic components can produce non-elliptical combined isophotes;
harmonics are therefore descriptive, not errors relative to assumed zero.

## Products, visualization and reproduction

All large products remain outside Git, below the campaign root's `analysis/`:

```text
huang2013_scientific_analysis_zero_background_2026_09_21/
huang2013_scientific_products_zero_background_2026_09_21_v2/
```

The measurement directory includes manifests, 14,229-row metrics, coverage,
scenario summaries, paired differences/bootstrap tables, 93 per-galaxy truth
and audit folders, provenance and `independent_source_audit.json`. Analysis
ran at commit `e45f91d`; its script SHA-256 is
`4de1f677842c43117bd5460969816ea0f89e3840f3868f50d34f49a35148eeb9`.

The final products contain six summary PNG/PDF pairs, their numerical tables,
ten selected-case PNG/PDF atlas pairs, 25 successful-profile metric checks,
AutoProf background/center tables and old/new selection/support comparisons.
`previous_selection_comparison.json` records input hashes and the unchanged
non-AutoProf selection check. All 3,297 successful auxiliary backgrounds are
zero; missing auxiliary values for failed fits are not imputed.

Use `_v2`. The first export remains as a diagnostic artifact: the shared
plotter overlays `mask` on the data panel only, so its residual panels showed
native model coverage. The corrected publication caller explicitly masks
displayed models outside the numerical common aperture. No metric changed.
The six summary figures are unchanged; all atlas profiles still retain
native radial coverage. Residual maps are data minus the common no-harmonic
model in ADU, with panel-specific colorbars. Compare scales, not color alone.
The dI/I panel uses the first plotted profile, not truth; displayed centroid
offsets use the plotter's inner reference, unlike exact-center table metrics.

The ten case rules select NGC1549/wide_z005 (typical), NGC1209/noiseless_z005
(ellipticity), NGC4742/deep_z050 (resolution), NGC6673/deep_z050 (support),
IC2311/deep_z005 (AutoProf failure), ESO185-G054/noiseless_z005 (Photutils
failure), NGC6673/wide_z050 (largest primary difference and EA degradation,
separate panels), NGC4033/wide_z050 (EA benefit), and NGC4742/wide_z050
(simultaneous harmonics). Failed primaries are explicitly unavailable, not
replaced with a successful diagnostic arm. These are selected illustrations,
not ten independent validation objects or exhaustive per-fit QA.

With the recorded environment/thread settings, rerun the collector using
the same root and new, unused output directories:

```bash
uv run --with pandas==2.3.3 python -u -m benchmarks.exhausted.analysis.publication_huang \
  --root /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28 \
  --output /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/NEW_MEASUREMENTS \
  --mock-source /Users/shuang/code/isophote-test-frozen-20260921 \
  --profit-cli /Users/shuang/Dropbox/work/project/otters/isophote_test/libprofit/build/profit-cli \
  --autoprof-campaign /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/fits/publication_autoprof_zero_huang_2026_09_21 \
  --workers 8

uv run --with pandas==2.3.3 python -m benchmarks.exhausted.plotting.publication_huang \
  /path/to/NEW_MEASUREMENTS /path/to/NEW_PRODUCTS
```

For the old/new supplemental table, apply `matched_primary` separately to
the previous/current `fit_metrics.csv`, intersect galaxy/scenario indices
per metric, and report each tool's old median, new median and median paired
difference. Support comparisons use one row per galaxy/scenario. No old
metric is recomputed on an invented common old/new mask.

The source changes passed 651 unit tests and focused lint/whitespace checks.
The final post-visualization focused suite passed 47 tests; the complete
651-test suite was repeated successfully afterward. `final_product_audit.json`
verifies the script/input hashes, unchanged tables/summary PNGs, and common
support for every displayed atlas model. Driver logs are preserved in the
new fitting campaign's `correction_audit/driver_logs/`.
Existing masked-to-NaN plotting warnings remain visible; no missing values
were fabricated. All generated results, logs and analysis tables stay off Git.
The feature branch is not merged. S4G has not been run: its next campaign
can reuse the fixed-zero mode, but should explicitly account for the exposed
image-boundary and initialization failures. No new default arm or public
release decision is made, and the separate Stage 4 timing results are unchanged.
