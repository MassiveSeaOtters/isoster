# Publication workflow lessons

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
