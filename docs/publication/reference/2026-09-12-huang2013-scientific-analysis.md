# Huang2013 scientific analysis: corrected selection

## Status and interpretation

The full corrected Huang2013 selection was measured on 2026-09-12 on branch
`analysis/huang2013-corrected-science`, following the
[approved measurement contract](../../specs/2026-09-10-huang2013-scientific-analysis.md)
and [AutoProf correction](2026-09-11-autoprof-pa-correction-log.md).
No new fitting was required and no existing input or result was changed.

The principal result is robust Isoster execution with competitive
elliptical-profile reconstruction, alongside meaningful configuration
trade-offs. EA geometry has a small favorable pooled RMS shift, not a
universal improvement. Outer regularization suppresses drift and roughness
much more clearly than it changes pooled reconstruction RMS. Cross-tool
differences include background estimation and sampling, not geometry alone.

This is the detailed reference behind the
[manuscript draft](../manuscript-only/huang2013-scientific-analysis.md).
The sample is exploratory, as agreed; no held-out or global-score claim is made.

## Data, conditions and selection

The 93 galaxy models contain two, three or four centered Sersic components
(5, 70 and 18 galaxies respectively). Their parameters come from the frozen
MockGal Huang2013 recipe, not newly optimized models. The existing NGC1172
Sersic-index clamp from 11.38 to 8.0 remains part of that recipe.

There are nine scenarios per galaxy: `noiseless_z005`, then `wide` and `deep`
at z=0.05, 0.20, 0.35 and 0.50. All use 0.168 arcsec pixels, zeropoint 27.0,
a Gaussian PSF with FWHM 0.7 arcsec, and zero injected sky background. Wide
and deep images have independent deterministic Gaussian noise draws with
per-pixel sigma 0.050358 and 0.018732 ADU respectively. The recipe's 5-sigma
surface-brightness limits are 24.62395438263279 and 25.697653749397546.
The noiseless scenario contains no added noise, not a small-noise proxy.
These HSC-like settings do not reproduce all properties of real HSC images.

The exact original/recovery/afterburner join is implemented in
`benchmarks/exhausted/analysis/publication_huang.py`. It retains the original
Isoster reference over recovery-only dependencies and the documented
fixed-center recovery precedence. All 3,348 scientific AutoProf outcomes
are replaced by the audited PA-corrected campaign, irrespective of quality.
The corrected run's reference-only dependencies are not additional Isoster
science outcomes. There are 3,351 explicitly excluded records, including
superseded AutoProf outcomes and the earlier dependency overlaps.

| Tool / arms | Successes | Failures | Intentional skips |
|---|---:|---:|---:|
| Isoster: nine executable arms | 7,533 | 0 | 0 |
| Isoster: `ols_noweight` | 0 | 0 | 837 |
| Photutils: `baseline_median` | 811 | 26 | 0 |
| Photutils: `aggressive_clip` | 833 | 4 | 0 |
| Photutils: `fixed_center` | 809 | 28 | 0 |
| AutoProf: four arms | 3,348 | 0 | 0 |
| Total | 13,334 | 58 | 837 |

The nine executable Isoster arms are `ref_default`, `reg_outer_damp`,
`stack_all`, `int_median`, `geom_simul`, `lsb_autolock`, `geom_ea`,
`geom_simul_ea` and `harm_simul_ea`. Each has 837 successful outcomes.
Without variance maps, `ref_default` already uses ordinary least squares;
the 837 `ols_noweight` skips prevent counting an identical configuration twice.
The AutoProf arms are `baseline`, `deep`, `high_regularization` and `fix_center`,
each with 837 successes. Arm definitions remain in the frozen campaign files
and per-fit configuration records; the analysis does not change them.

Of the 58 Photutils failures, 56 are noiseless: 26 primary, 28 fixed-center
and two aggressive-clipping cases. The other two are aggressive-clipping
failures at wide/deep z=0.05. Recorded messages are 53 NaN-to-integer errors,
three empty IsophoteLists and two 900-second timeouts. All 744 noisy primary
Photutils fits succeeded. This supports the user's warning that noiseless
images can be numerically troublesome; it does not imply a comparable
failure rate for ordinary noisy survey data.

## Measurement definitions and audit

Let T be the regenerated PSF-convolved truth, D the retained input and M the
common no-harmonic elliptical reconstruction of a retained fitted profile.
For each galaxy/scenario, the aperture is the intersection of finite M pixels
across **all successful arms**, inside the initial maximum SMA and outside one
PSF FWHM. Initial geometry defines the fixed elliptical radius, not any
tool's fitted ellipse. The same aperture is used for all tools and contrasts.

- Truth-relative RMS: sqrt(mean((M-T)^2)) / sqrt(mean(T^2)). It is not the
  mean of per-pixel fractional errors and weights bright structure strongly.
- Aperture flux bias: sum(M-T) / sum(T). Positive means excess reconstructed
  light. The absolute-bias metric takes the absolute value per fit before
  taking population medians. Neither quantity is extrapolated total flux.
- Data residual RMS: sqrt(mean((M-D)^2)) / injected Gaussian sigma. It is
  unavailable for noiseless data, and is not a reduced chi-square statistic.
- Inner, middle and outer zones use r/R_ref <0.5, 0.5--2 and >=2 respectively,
  within that aperture. R_ref is the adapter's reference scale, not the true
  combined half-light radius. Missing zones remain N/A.
- Center error is the median fitted-center distance from the exact shared
  input center for finite isophotes outside one PSF FWHM. Radial reach is the
  maximum finite SMA / R_ref, not a validated detection radius.
- The truth-ring diagnostic evaluates 1,024 bilinear samples on each fitted
  ellipse, using its recorded polar-angle or EA basis and mean/median choice.
  It is conditional on fitted geometry and does not reproduce clipping,
  AutoProf's finite-width annuli, or the LSB arm's changing integrator.
- Existing smoothness/harmonic diagnostics remain descriptive. `native_*`
  fields retain the original campaign metrics and must not be mixed with
  the newly normalized truth/noise quantities.

Four regenerated truth images per galaxy give 372 new FITS files. All 837
stored inputs reproduce exactly after float32 conversion and their 837
exported seed entries match the frozen recipe as exact text. Small nonzero
`max_abs_difference` values compare double-precision predictions against
stored pixels; they are not failures of the float32 equality test.

The new manifest exactly matches the previously accepted corrected manifest
on galaxy/scenario/tool/arm/status/source path, with 14,229 unique keys.
All 30,913 consumed source files passed before/after SHA-256 checks and a
separate post-run recheck. No numeric metric contains infinity. All successful
fits have finite all-aperture and outer-zone truth RMS. Inner-zone values are
unavailable for 4,672 successful records and middle-zone values for 256;
these are not silently converted to zero.

Common support is an important restriction: across 837 galaxy/scenario
combinations, its fraction of the eligible aperture has median 0.7133,
p16/p84 0.5288/0.8371 and minimum 0.05249. Median support falls from 0.7942
at wide z=0.05 to 0.5078 at wide z=0.50. Thus comparisons at different
redshifts do not measure an identical physical aperture. An arm with poor
coverage can restrict every arm's metric. Profile reach and support must
accompany any accuracy comparison.

## Primary three-tool results

Primary arms are Isoster `ref_default`, Photutils `baseline_median` and
AutoProf `baseline`. The table uses the same successful, finite galaxies
for all three tools: 67 for noiseless and 93 for each noisy scenario.
Values below are median truth-relative RMS expressed as percent; these are
not success percentages. The full p16/p84 distributions are in the products.

| Scenario | Galaxies | Isoster (%) | Photutils (%) | AutoProf (%) |
|---|---:|---:|---:|---:|
| Noiseless z=0.05 | 67 | 0.8551 | 0.9027 | 0.8767 |
| Wide z=0.05 | 93 | 0.8576 | 0.8710 | 1.0763 |
| Deep z=0.05 | 93 | 0.8369 | 0.8692 | 0.8361 |
| Wide z=0.20 | 93 | 1.7131 | 1.8238 | 2.7882 |
| Deep z=0.20 | 93 | 1.4165 | 1.5119 | 2.0446 |
| Wide z=0.35 | 93 | 2.8893 | 3.0823 | 4.7249 |
| Deep z=0.35 | 93 | 1.9478 | 2.1351 | 3.3774 |
| Wide z=0.50 | 93 | 4.9708 | 5.0675 | 6.7544 |
| Deep z=0.50 | 93 | 2.5947 | 3.0225 | 4.6857 |

This is not a universal ranking. AutoProf has smaller median inner-zone RMS
in all nine scenarios, on their smaller finite inner-zone samples (65, 91,
91, 67, 67, 41, 41, 28 and 28 galaxies in scenario order). The all-aperture
and outer-zone summaries favor different aspects of profile recovery.
For wide z=0.05, outer-zone RMS medians are 1.855%, 1.659% and 10.515%; for
wide z=0.50 they are 13.573%, 12.631% and 23.062% (Isoster, Photutils, AutoProf).

Signed aperture-flux biases at wide z=0.05 are +0.524%, +0.713% and -3.311%;
at wide z=0.50 they are +1.204%, +2.041% and -9.255%. These differences
include the complete retained extraction behavior. AutoProf subtracts its
estimated background before extracting intensities, while the mock sky is
zero. Its primary background estimate is positive in 830/837 cases,
including all noiseless cases; the median is 0.008673 ADU for noiseless,
0.018471 for wide z=0.05 and 0.009760 for wide z=0.50. Galaxy wings entering
the background estimate are a plausible contributor, **not a quantified
causal decomposition**. No background was added back to the reported model.
A controlled known-sky diagnostic would be needed to isolate geometry and
extraction from sky estimation; this analysis does not launch another rerun.

AutoProf median center errors are often zero, but its auxiliary output rounds
centers to 0.01 pixel and those values populate the profile. This is a global
center estimator versus radially varying centers for the other primary tools,
not evidence of unlimited subpixel accuracy. At wide z=0.50 the medians are
0.2780, 0.2687 and 0.03162 pixel. AutoProf's zero stop-code placeholders are
explicitly marked non-comparable and removed from atlas convergence markers.

At wide z=0.50 the data/sigma RMS medians are 0.9914, 0.9933 and 1.0128,
despite appreciable truth-profile errors. A residual near the noise level
does not by itself establish unbiased recovery. Ring statistics likewise
must not be used as a common estimator ranking: their native angular bases
and extraction statistics differ.

## Paired Isoster configuration results

Each all-aperture comparison contains 837 paired values from 93 galaxies.
Differences are arm minus reference in fractional RMS units, not relative
percentage improvements. Confidence intervals use 2,000 galaxy-block
bootstrap draws with seed 20260910: resampling a galaxy keeps all its
scenarios together. Intervals are exploratory and not multiplicity-adjusted.

| Arm minus reference | Median RMS change | 95% interval | Fraction with smaller RMS |
|---|---:|---|---:|
| `geom_ea` - `ref_default` | -7.528e-5 | [-1.457e-4, -4.047e-5] | 63.9% |
| `geom_simul` - `ref_default` | -7.587e-8 | [-1.435e-6, +1.099e-6] | 50.4% |
| `geom_simul_ea` - `ref_default` | -7.428e-5 | [-1.289e-4, -3.868e-5] | 63.7% |
| `geom_simul_ea` - `geom_ea` | -9.271e-8 | [-1.162e-6, +9.158e-7] | 50.4% |
| `geom_simul_ea` - `geom_simul` | -6.400e-5 | [-1.298e-4, -3.250e-5] | 63.3% |
| `harm_simul_ea` - `geom_simul_ea` | +5.168e-7 | [-7.525e-7, +1.858e-6] | 48.7% |
| `harm_simul_ea` - `ref_default` | -6.633e-5 | [-1.426e-4, -3.902e-5] | 63.3% |
| `int_median` - `ref_default` | +1.994e-4 | [+1.540e-4, +2.581e-4] | 28.3% |
| `lsb_autolock` - `ref_default` | 0 | [0, 0] | 19.2% |
| `reg_outer_damp` - `ref_default` | -4.407e-5 | [-1.379e-4, +7.531e-7] | 56.3% |
| `stack_all` - `ref_default` | -1.308e-5 | [-8.807e-5, +1.479e-5] | 53.4% |

EA geometry's median absolute aperture-flux error increases by 1.206e-4
(95% interval +3.707e-5 to +2.155e-4); this accompanies its favorable RMS
shift and prevents an unqualified improvement claim. Its median centroid
change has an interval spanning zero. Simultaneous harmonics within EA give
a smaller median center error by 0.001116 pixel, but no clearly separated
pooled RMS shift. Harmonic reconstruction is disabled in the common renderer,
so that comparison tests the fitted base profiles, not the benefit of
including high-order terms in an image model.

Outer damping lowers median paired center error by 0.03710 pixel, with
smaller error in 90.3% of pairs. After first taking each galaxy's median
over scenarios, population median outer ellipticity roughness falls from
0.02470 (`ref_default`) to 0.001041 (`reg_outer_damp`); PA roughness falls
from 3.326 to 0.009813 degrees. This is a strong smoothing effect, not proof
that all real twists or shape changes are better recovered. The LSB arm's
all-aperture RMS is exactly unchanged in 65.4% of pairs, explaining its zero
median and zero-width median interval without implying identical behavior
for every galaxy.

`paired_summary.csv` also labels smaller harmonic amplitudes/roughness as
`improved` and greater reach as `improved` for mechanical sign bookkeeping.
For these descriptive quantities, read those labels as smaller/larger, not
scientific accuracy verdicts. Mean/median and polar/EA changes also alter the
quantity being estimated. Native AutoProf Bender conversion remains invalid;
its higher-order terms are not pooled with Isoster coefficients.

## Structural associations and selected cases

Structural plots use one wide z=0.05 realization per galaxy, avoiding nine
copies of each structural descriptor. All three primary tools have 93
successful fits here. Isoster median RMS declines across increasing
R_ref/PSF quartiles from 0.01076 to 0.00940, 0.00760 and 0.00657 (24, 23,
23, 23 galaxies). Across increasing initial-ellipticity quartiles it rises
from 0.00692 to 0.00771, 0.00949 and 0.01172. Component-PA-span quartiles
do not show a monotonic Isoster trend. These are unadjusted associations;
resolution, brightness and structural parameters were not independently varied.

Centered symmetric components do not guarantee purely elliptical combined
isophotes when shapes/PAs differ. Genuine even harmonics are possible. The
harmonic figure therefore reports amplitudes and smoothness descriptively,
with galaxy counts, not deviations from assumed zero harmonic truth.

Nine deterministic selection rules produce eight distinct atlas pages:

| Galaxy / scenario | Selection reason |
|---|---|
| NGC1549 / wide_z005 | Closest Isoster RMS to the sample median |
| NGC1209 / noiseless_z005 | Largest initial ellipticity |
| NGC4742 / deep_z050 | Smallest R_ref/PSF in that scenario |
| NGC6673 / wide_z050 | Smallest common support and largest primary AutoProf-Isoster RMS difference |
| ESO185-G054 / noiseless_z005 | First retained primary failure in sorted order; no substitute arm |
| NGC4033 / wide_z050 | Largest EA RMS benefit: 0.16267 to 0.13191 |
| NGC6673 / wide_z050 | Largest EA RMS degradation: 0.51276 to 0.57116 |
| IC2006 / wide_z050 | Largest absolute simultaneous-harmonic RMS change: 0.17798 to 0.16224 |

The NGC6673 common aperture is only 5.249% of eligible pixels; it is an
important stress case, not a representative accuracy illustration. Visual
inspection shows increasingly noisy, sparsely sampled outer profiles in
the compact/high-redshift examples. The selected ESO185-G054 panel visibly
retains the missing Photutils primary instead of choosing its successful
clipping arm. Cases were chosen by the recorded rules, not visual preference.

Atlas pages reuse the project's comparison QA layout, serif style,
fourth-root SMA axis and tool colors/markers. Residual images are **D-M in
ADU**, with panel-specific color scales; compare colorbars, not color alone.
The mask shows the quantitative common aperture. One-dimensional profiles
retain their native radial coverage, including points outside that aperture.
The dI/I panel compares with the first plotted profile, not truth. Its
center panel uses the shared plotter's inner-profile reference center,
not the exact input center used by the quantitative table. Error bars are
retained native errors, not independently calibrated uncertainties.
All 20 displayed successful profiles pass reconstructed pixel-count and
truth-RMS equality checks before atlas export.

## Products and reproducibility

All large products remain outside Git, beneath:

```text
/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/
  huang2013_scientific_analysis_2026_09_12/
  huang2013_scientific_products_2026_09_12_v3/
```

The measurement directory contains the selected/excluded manifests, coverage,
14,229-row fit metrics, scenario summaries, paired differences and bootstrap
summaries, provenance, and 93 per-galaxy folders with truth FITS, exact-seed
verification, metrics and source hashes. The products directory contains:

- `01_coverage`, `02_primary_truth_zones`, `03_primary_fidelity`,
  `04_isoster_contrasts`, `05_structural_trends`, `06_harmonics_smoothness`
  as PNG (300 dpi) and PDF.
- Numerical plot tables, `structural_trends.csv`, `harmonic_smoothness.csv`,
  `primary_additional_metrics.csv`, `autoprof_background_centers.csv`.
- `selected_cases.csv`, eight PNG/PDF pairs under `atlas/`,
  `atlas_metric_checks.json` and plotting provenance with input/script hashes.

The first plotting gate, first full exports and `_v2` remain separate
diagnostic iterations. Use `_v3`: it fixes caption spacing, summary axis
limits and the failed-case title. None of those changes altered measurements.
Masked optional profile values produced existing masked-to-NaN warnings in
the shared plotter; they were not converted to invented values or suppressed.

Measurement provenance records Python 3.12.11, macOS 15.7.3 arm64, eight
worker processes and start time 2026-09-12 16:14:16 +0800. The host is the
Mac Studio M1 Ultra (20 CPU cores, 128 GB RAM). BLAS/OpenMP/Accelerate thread
limits were one, `caffeinate -i` inhibited idle sleep, and observed thermal
checks had no warning with zero swap usage. This was not a controlled timing
experiment; no performance conclusion uses the analysis execution time.

The unchanged measurement script ran at Isoster commit
`93138136a25b783920ecc9919617944aafdbe97b`, SHA-256
`a543eefb0364220040fba89a9738700143eef9462f2406392a96aa6266749864`.
Frozen MockGal commit:
`a6a90a07dc3aedd95465928ee2e93258c8ccb40a`.
profit-cli SHA-256:
`dee3b2445feaba39a89ccd1babb86077ed0c8a988e87f7cb38069aeb88aa88ac`.
The frozen source was read from `/tmp/isoster_huang_truth.2BzppK/source`;
that temporary worktree can disappear, so reproduce it from the recorded
commit rather than depending on its continued existence.

From the Isoster checkout, use the external uv environment and pinned pandas
overlay. Set `UV_PROJECT_ENVIRONMENT=/Users/shuang/.venvs/isoster`,
`PYTHONPYCACHEPREFIX=/tmp/isoster_pycache`, `MPLBACKEND=Agg`, and
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`
as separate environment variables. Then run:

```bash
caffeinate -i uv run --with pandas==2.3.3 python -u -m benchmarks.exhausted.analysis.publication_huang \
  --root /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28 \
  --output /path/to/a/new/measurement/directory \
  --mock-source /path/to/frozen/mockgal/worktree \
  --profit-cli /Users/shuang/Dropbox/work/project/otters/isophote_test/libprofit/build/profit-cli \
  --autoprof-campaign /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/fits/publication_autoprof_pa_huang_2026_09_11 \
  --workers 8

uv run --with pandas==2.3.3 python -m benchmarks.exhausted.plotting.publication_huang \
  /path/to/measurements /path/to/a/new/products/directory
```

Both commands reject an existing output directory. For a small collector
gate add `--galaxies IC2597 NGC1209`; for a plotting gate use its measurements
and `--atlas-limit 1`. For CSV seed inspection explicitly read `seed` as text.
The independent source audit recomputes every digest in all per-galaxy
`source_hashes.json` files and compares them with the retained values.

Final verification: 44 focused tests passed in 2.65 seconds, covering the
collector, matched plotting samples, PA conversion, harmonic schema and
campaign contracts. Ruff and whitespace checks passed. The products have
six summary PNG/PDF pairs and eight atlas PNG/PDF pairs; their script/input
hashes match the final exporter and retained measurement tables. The
measurement directory contained 538,267,314 bytes and final products
17,474,810 bytes before adding the small closeout log/audit notes.

## Remaining boundaries

This completes the agreed Huang2013 descriptive analysis, not S4G analysis,
real-galaxy validation, a release decision, or selection of a new default arm.
Before a geometry-only cross-tool claim, separate known-sky behavior from
estimated-background behavior. Before claims about high-order model fidelity,
use validated native harmonic reconstruction. Repeated noise draws, realistic
contaminants and real-galaxy tests remain additional experiments, not results
implied by this smooth-model campaign. Stage 4 remains the controlled timing
reference. No existing volume data was removed or overwritten.
