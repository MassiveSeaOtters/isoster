# Publication workflow lessons

## 2026-09-23: separate residual display from metric support

- User-approved cross-tool QA now displays full finite residuals, not just the
  evaluation aperture. Inner/outer cuts still apply to every numerical metric.
  Blank model coverage remains unavailable, not zero or extrapolated.
- When expanding an alternative-model table to radial zones, replace every
  zone's metrics, not just the full-aperture columns. Otherwise baseline inner,
  middle and outer values silently remain under the alternative-model heading.
- Removing an I=0 guide also requires removing its value from the SB limit
  calculation. Keep native uncertainties, but set limits from profile values.

## 2026-09-23: native models are not interchangeable harmonic controls

- AutoProf extraction's measured intensity harmonics and its renderer's fitted
  Fourier shape modes are different outputs. Native genmodel does not include
  measured a3/b3/a4/b4 merely because they exist in the profile.
- Pair a new harmonic renderer with its own harmonic-off control. Comparing
  spline-on only against linear-off confounds the harmonic effect and renderer.
- An improvement in full-aperture RMS can coincide with worse outer-zone RMS
  and worse flux bias. Retain all radial metrics before choosing defaults.

## 2026-09-23: evaluation cut and harmonic fairness

- Use an explicit fixed 2-pixel evaluation cut, not the physical PSF metadata.
  Persist it with measurements and require it when rebuilding QA support.
  Keep the historical PSF-cut scores and native fit-record metrics labelled.
- A shared renderer does not remove sensitivity to native radial sampling
  or interpolation. Hold pixels fixed across reconstruction alternatives.
- AutoProf schema-2 `a3/b3/a4/b4` are intentionally NaN; native FFT fields and
  their angle basis are preserved separately. Setting `use_harmonics=True`
  alone is not a fair harmonic-inclusive three-tool comparison.
- For polar-basis additive reconstruction, raw amplitudes can be recovered
  without estimating an AutoProf gradient. Validate sign, factor of two, and
  PA rotation synthetically. Do not rotate EA harmonics as if they were polar.

## 2026-09-22: QA aperture and missing-data semantics

- For a good/bad metric matrix, color signed bias by its absolute magnitude,
  while printing the signed measurement. Identical values should be neutral,
  not falsely ranked. With touching radial panels, prune boundary y ticks
  and check long axis labels on both ordinary and failed-reference cases.

- A measurement aperture is not an input bad-pixel mask. The earlier atlas
  passed its complement into the data-panel mask parameter, giving ideal
  mocks a misleading red overlay. The opt-in demo uses labelled radial
  ellipses on unmasked data and retains common-aperture residual maps.
- An absent native curve of growth must be labelled unavailable. An empty
  difference panel with a zero baseline can falsely suggest agreement.
- Keep failed primary fits in statistics tables without selecting a successful
  alternative. Default-relative panels are unavailable when that default fails.


## 2026-09-21: prepared background and Dropbox isolation

- A zero injected sky does not make an automatic sky estimate zero: extended
  galaxy light can enter the estimate. For this prepared-image comparison,
  explicitly pass `ap_set_background=0.0` through the shared wrapper and
  verify both saved options and reported auxiliary values. Leave the noise
  estimator intact; a fixed sky is not a fixed noise level.
- A change in the AutoProf profile can change the all-arm common aperture.
  Recompute the entire scientific analysis, including metrics for unchanged
  Isoster/Photutils profiles. Old/new RMS differences need not describe the
  same pixels; do not label them an isolated background-estimation effect.
- Dropbox can synchronize a stale Git index while source files and HEAD are
  current. Compare HEAD, the index and disk separately before interpreting
  staged/unstaged changes as unfinished edits. Here the active index matched
  the old `fccd66c` tree, while all 430 tracked files matched current HEAD
  `02cc480`; the dated conflicted index matched current HEAD too. Work from
  a GitHub clone outside Dropbox instead of resetting either copy.
- Copy accepted reference profiles for fixed-center dependencies when those
  fits have not changed. Do not rerun Isoster merely to provide AutoProf's
  fixed center. Hash the sources before and after the campaign.
- Separate startup/source-audit time from fitting-phase time. The 12-fit
  gate's recorded phase took 55.564 seconds, but the full command also had
  source-selection startup. Parallel campaign time is not controlled timing.
- Audit the requested recipe and the verified conditional-retry policy
  separately. Fixing the background can change which inputs trigger the
  existing retry, so final saved options need not be identical. The initial
  zero-background audit incorrectly equated unchanged policy with unchanged
  triggers. Validate each recorded fallback against its image-size rule and
  first-attempt failure log before removing those keys for recipe comparison.
  Never ignore arbitrary differences or refit successful data to repair an audit.
- Check installed AutoProf semantics, not wrapper comments. In AutoProf 1.3.4,
  `ap_truncate_evaluation` stops after two non-positive intensity samples;
  it does not test the image boundary. Harmonic interpolation of an empty
  ring occurs before that test. The previous comment overstated the retry's
  protection. Zero sky exposes some failures despite the unchanged retry;
  correcting the comment does not authorize changing the fitting recipe.
- Exact input-noise seeds do not imply deterministic fitting. Installed
  AutoProf reseeds its optimizer inside `Process_Image` using process/time
  information. Preserve this distinction in old/new comparisons; do not
  claim that every change is causally isolated to the fixed background.
- The shared comparison plotter's `mask` argument overlays the data panel;
  it does not mask residual maps. For publication atlas residuals on the
  quantitative common aperture, explicitly mask the displayed models with
  NaNs outside that aperture. Keep native radial profiles unchanged.

## 2026-09-23: population reconstruction gates

- Inspect positive-radius basis flags. A central Isoster row can carry an EA flag
  even when the measured positive-radius profile is polar; it is excluded by the
  native renderer and must not change the reported harmonic basis.
- Check masked FITS coefficients with explicit NaN filling before reconstruction.
  A masked-array reduction can ignore unavailable entries; a renderer may replace
  nonfinite harmonics with zero. Neither is valid evidence for harmonic coverage.
- Large finite simultaneous-EA coefficients can yield extreme native radial-shift
  models. Retain these measured outcomes; do not clip coefficients, repair fits,
  or remove unfavorable models during a descriptive reconstruction comparison.
- A missing reconstruction is not a missing fit. The fault-injection output gate
  exposed that filtering profiles by available models removed valid native curves
  and their designated reference. Preserve profiles separately, show a labelled
  missing residual panel, and keep model metrics unavailable. The first population
  attempt was stopped and preserved; rerun into a fresh folder after this QA fix.
- The accepted noiseless analysis records injected_sigma as NaN, not numeric zero.
  Treat noise-normalized RMS as undefined for that labelled condition and record
  the reason explicitly; do not infer that missing normalization implies failure.
