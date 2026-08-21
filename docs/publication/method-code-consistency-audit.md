# Method-draft code consistency audit

Original audit: 2026-08-11 on `docs/publication-method-outline`  
Last code review: 2026-08-15 on `docs/multiband-doc-refresh`

## Scope and standard

This audit covers every `docs/publication/draft/section-*.md` file. It asks
whether each proposed scientific statement is implemented by the current code,
whether existing outputs support the quantitative claim, and whether the
material belongs in a concise ApJ Methods section. It is not a line-by-line
copy edit and it does not treat internal documentation as independent evidence.

The draft contains a strong technical basis for the paper, but it currently
mixes four kinds of material:

1. implemented and tested method descriptions;
2. measured demonstrations that can support paper claims after a current-HEAD
   reproducibility run;
3. plausible advantages that have not yet been demonstrated;
4. user-manual details that should move to documentation or an appendix.

Only the first two categories should be stated as results in the paper.

## Executive findings

### Supported central claims

- Isoster performs vectorized, path-based ellipse sampling with bilinear image
  and variance interpolation and nearest-neighbor mask sampling
  ([`isoster/sampling.py`](../../isoster/sampling.py)).
- The image-level driver fits the center and anchor ellipse, grows inward and
  outward, and can subsequently compute curve-of-growth quantities
  ([`isoster/driver.py`](../../isoster/driver.py)).
- The default geometry solve fits the first- and second-order harmonics; optional
  higher orders can be fitted after convergence or jointly in the iteration
  loop ([`isoster/fitting.py`](../../isoster/fitting.py)).
- A supplied variance map activates inverse-variance weighted harmonic fits and
  propagates sampled variances into reported intensity and geometry errors.
- Lazy gradient evaluation, first-isophote retries, update damping and clipping,
  geometry-based convergence, soft outer regularization, and automatic outer
  geometry locking are implemented safeguards for difficult images.
- Single-band forced photometry and a supported default multi-band joint
  free-fit path are implemented; selected multi-band options retain narrower
  experimental or non-default qualifications.
- Results preserve the isophote table, resolved configuration, and top-level
  metadata in a four-HDU FITS product
  ([`isoster/utils.py`](../../isoster/utils.py)).

### Resolution status of the original publication blockers

1. **Manuscript correction retained: eccentric anomaly is not physical arc
   length.** Uniform sampling in eccentric anomaly is implemented, but the
   samples are not uniform in
   Euclidean distance along the ellipse. Ciambur (2015, Figure 2) explicitly
   describes the transformation as producing equal arc lengths. Taken
   literally, that geometrical statement is inaccurate: for
   `(x, y) = (a cos(psi), b sin(psi))`,
   `ds/dpsi = sqrt(a^2 sin^2(psi) + b^2 cos^2(psi))`, which is constant only
   for a circle. Isoster nevertheless implements the eccentric-anomaly
   parametrization specified by Ciambur (2015), up to a documented reversal of
   angular sign convention. This is therefore not an Isoster-specific sampling
   bug. The manuscript should preserve the substantive ISOFIT motivation while
   politely distinguishing uniform auxiliary-circle angle from equal physical
   arc length. **Status: resolved as a wording issue, not a code change.** The
   proposed footnote below records the required distinction.
2. **Resolved: direct eccentric-anomaly geometry updates are intentional, and
   the OLS error scaling is now matched to the fitted model.** Harmonics are
   fitted against eccentric anomaly and the first four coefficients are passed
   directly to the geometry-update equations. Rechecking Jedrzejewski (1987)
   shows that these equations were derived for coefficients fitted at equal
   intervals in eccentric anomaly, so no conversion to a polar-angle Fourier
   basis is required. The earlier audit concern about the geometry update was
   based on an incorrect premise. The subsequent fix computes the OLS residual
   variance once from the complete fitted model, evaluated in the same angular
   basis as the fit, and shares that scale between geometry and simultaneous
   higher-harmonic errors. Exactly determined fits report zero rather than
   rebuilding a mismatched reduced model. Regression coverage is in
   `tests/unit/test_ols_covariance_scaling.py`; the WLS path remains unscaled as
   required for `(A^T W A)^-1`.
3. **Resolved: gradient values and errors now describe the same ring
   statistic.** `_ring_statistic_and_variance` returns a matched mean or median
   and its variance, and `compute_gradient` uses it for every sampled ring.
   The mean uses the unweighted mean's variance because the reported gradient
   uses an unweighted mean; the median includes its Gaussian-asymptotic
   `sqrt(pi/2)` penalty. This closes the implementation mismatch, although the
   resulting uncertainty is still a statistical approximation for
   interpolation-correlated galaxy images rather than an exact real-image
   covariance. Regression coverage is in `tests/unit/test_gradient_error.py`.
4. **Resolved: unusable variances are excluded rather than clamped.** In both
   single- and multi-band sampling, variance entries that are non-finite or not
   strictly positive are marked invalid, warned about, and excluded from the
   ring statistic and its uncertainty. Direct single-band sampler calls also
   validate the four source pixels used by bilinear interpolation, preventing
   an invalid neighbor from contributing a finite interpolated value. The
   retired `1e30` sentinel and `1e-30` clamp are absent. Coverage is in
   `tests/unit/test_variance_map.py` and
   `tests/multiband/test_sampling_mb.py`.
5. **Reframed: the default multi-band path is supported; selected extensions
   remain qualified.** The shared-validity, joint-intercept,
   geometry-parameterized default with independent higher harmonics is now
   tested in OLS and WLS, including planted-truth recovery, flux-unit
   invariance, one-band parity, and a known-truth color-gradient demonstration.
   The optional `simultaneous_in_loop` and `simultaneous_original`
   higher-harmonic modes remain experimental, `loose_validity=True` is repaired
   and tested but deliberately non-default, and the five-band path still lacks
   the deferred real-data validation approved by the author. The manuscript
   may present the supported default path, but must retain these per-feature and
   validation qualifications.

**Net status:** none of the original implementation-consistency bugs remains
open. The remaining publication work is to use the corrected eccentric-anomaly
wording, preserve the narrower multi-band qualifications, broaden selected
validation where noted, and rerun quoted measurements from a frozen publication
commit.

## Section-by-section audit

### `section-1.0-overview.md`

**Keep**

- The high-level scientific purpose: fast one-dimensional isophotal analysis
  with survey-pipeline use in mind.
- The four method themes: efficient sampling, variance awareness, flexible
  harmonic analysis, and safeguards for low-surface-brightness outskirts.

**Qualify or relocate**

- Separate implemented capabilities from measured advantages. “Fast,” “more
  robust,” and “survey scale” require benchmark definitions and evidence.
- Move package layout, command-line details, output-column catalogs, and lists
  of configuration flags to the user documentation or an appendix.
- Avoid positioning every optional mode as part of one recommended default.
  Most advanced modes are disabled by default and solve different scientific
  problems.

### `section-1.1-algorithmic-foundation.md`

**Consistent with code**

- A trial ellipse is sampled, the intensity variation is expanded in
  harmonics, and the first two orders drive corrections to center,
  ellipticity, and position angle.
- The default update strategy selects the largest geometry harmonic; an
  optional simultaneous strategy updates all geometry coordinates.
- The radial sequence can be geometric or linear and is grown in both
  directions from an anchor isophote.
- Higher-order coefficients are normalized by semi-major axis and the absolute
  radial intensity gradient in the single-band result path.

**Corrections**

- State the exact implemented sampling density,
  `max(64, int(2*pi*sma))`, but do not imply it is Nyquist-optimal without a
  convergence study.
- Distinguish the polar image angle `phi` from eccentric anomaly `psi`. Neither
  is arc length.
- Avoid presenting the convergence threshold as a universal statistical
  criterion. It is an operational criterion based on the largest harmonic,
  residual scatter, and a selectable radial scaling.
- The background-noise floor `sigma_bg/sqrt(N)` is an implemented heuristic,
  not proof that a smaller residual is physically impossible.

### `section-1.2-implementation-overview.md`

**Consistent with code**

- `fit_image` is the canonical single-band call and accepts image, mask,
  configuration, optional template geometry, and optional variance map.
- Configuration is validated with Pydantic; the current single-band model has
  62 fields.
- Template-based forced photometry bypasses free geometry fitting.
- The result contains a list of isophote dictionaries and a resolved
  configuration.

**Corrections**

- `IsosterConfig` is not immutable or frozen. Do not describe it as immutable.
- Multi-band fitting uses the separate `IsosterConfigMB` interface rather than
  the single-band configuration.
- FITS serialization now writes **four** HDUs: Primary, `ISOPHOTES`, `CONFIG`,
  and `META`. The draft and two older documentation passages saying three HDUs
  are stale.
- Astropy Table is an in-memory conversion, not an additional on-disk format.
- A result file cannot reproduce a fit by itself: the input image, mask,
  variance map, software version, and environment are also required.
- The complete module inventory and configuration catalog are user-manual
  material, not main Methods text.

### `section-1.3-why-fast.md`

**Consistent with code**

- Entire ellipse paths are sampled in vectorized array operations rather than
  through a Python loop over individual samples.
- Numba accelerates coordinate and design-matrix kernels when available, with
  NumPy fallbacks.
- Lazy gradient evaluation caches the radial gradient and recomputes it when
  the iteration stalls or the cached estimate is inadequate.
- The per-iteration harmonic system is small relative to the number of samples;
  image sampling is the dominant repeated operation.

**Quantitative evidence currently available**

> **Superseded (2026-08-21).** Both pilot measurements below have been
> replaced, but to *different* standards — one by a controlled timing study,
> the other by an archived single sweep. The distinction matters and is stated
> per item. They are kept for the record of what the audit found, not as
> evidence to cite.

- ~~The stored lazy-gradient experiment reports a 43.9--44.9% reduction in
  sampling calls over three galaxies, with wall-time ratios of 1.37--4.46
  relative to eager recomputation.~~ Replaced by
  `benchmarks/draft_timings/`: **60 gradient evaluations against 414**, a
  median wall-time saving of **24%** (IQR 23.0--24.1%) over eighteen sessions
  in separate interpreters, with the iteration count unchanged. Archived in
  `reference_timings.json` and checked against the prose by
  `check_draft_numbers.py` in CI. The accuracy cost is quantified too: the
  median geometry difference stays under 1% of sigma, with a tail reaching
  2.6 sigma at S/N = 10.
- ~~The stored photutils comparison contains two synthetic cases with 14.1 and
  26.0 times speedups.~~ Replaced by a synthetic Sersic sweep: **median 45x**
  (IQR 35--55x, slowest 13x) over the **237 of 243** configurations photutils
  could fit, all of which passed the script's accuracy criteria. Archived in
  `benchmarks/performance/reference_speedup.json`.

  **This is an archived single sweep, not a controlled timing rerun.** Each
  configuration is timed once per tool, isoster first and photutils second,
  with no warm-up, repetition or interleaving. The quartiles are spread across
  grid *configurations*, not across repeated timings, so they do not
  characterise timing noise and the protocol carries a known ordering bias.
  The effect is unlikely to matter at a ratio of tens of times, but it has not
  been quantified. Upgrading this to the `draft_timings` standard --
  batching, interleaved shuffled order, sessions in separate interpreters --
  is outstanding.

  The 6 excluded configurations are systematic, not random: n=1, eps=0,
  pa=pi/4 at both noisy levels, across all three sizes. A circular source has
  no defined position angle, so that corner of the grid is degenerate and
  photutils raises there. The archive lists them; the speedups describe only
  what photutils could fit.

Both records include their environment. Neither covers a morphology/size grid
wider than the synthetic Sersic family, and neither is a real-image benchmark.

**Corrections**

- Do not claim a generic three-to-five-fold Numba speedup. Existing end-to-end
  Numba measurements range from slower than fallback to modestly faster.
- Do not claim a one-second compilation cost without a measured value and a
  defined machine.
- Do not claim lazy-gradient and eager-gradient profiles are identical. Use
  explicit profile-fidelity distributions.
- Model reconstruction does not use `extract_isophote_data`; therefore the
  regular fit, forced extraction, and two-dimensional model do not literally
  share one sampling stack.
- Avoid “byte-equivalent” numerical claims across platforms. Say that the
  fallback paths are covered by numerical regression tests.
- Complexity should be described as linear in the number of sampled points per
  trial ellipse, multiplied by iterations and fitted radii. Do not claim that
  image dimensions have no effect.

### `section-1.4.1-eccentric-anomaly-isofit.md`

**Consistent with code**

- With `use_eccentric_anomaly=True`, coordinates are uniform in `psi` and the
  harmonic design matrix is evaluated in `psi`.
- `simultaneous_harmonics=True` supports an extended matrix containing orders
  1, 2, and requested higher orders.
- `isofit_mode='in_loop'` uses the extended solve during every iteration;
  `isofit_mode='original'` uses the five-parameter geometry solve in the loop
  and a joint higher-order fit after convergence.
- If the number of valid samples is insufficient for the extended matrix, the
  iteration falls back to the five-parameter solve and emits a warning.

**Scientific corrections and current validation status**

- Replace the claim that uniform `psi` gives uniform physical arc-length
  sampling. Ciambur (2015, Figure 2) makes this claim explicitly, so the
  manuscript should not imply that it originated in the Isoster draft.
  Instead, state that `psi` is uniform on the auxiliary circle and
  redistributes ellipse samples toward the major-axis ends. A concise footnote
  can note that equal increments in eccentric anomaly do not correspond to
  exactly equal Euclidean arc lengths because `ds/dpsi` varies from `b` to `a`.
- Delete the argument that a perfect elliptical constant-intensity isophote
  acquires nonzero higher harmonics solely because samples are nonuniform in
  arc length. A constant remains constant under either angular grid.
- Delete the statement that geometry coefficients are converted to `phi`
  space. That conversion is absent because the implemented correction factors
  directly consume the eccentric-anomaly coefficients, as in Jedrzejewski
  (1987). Retain a planted-geometry test to validate signs and recovery, not to
  justify a basis conversion.
- The former OLS geometry-error mismatch is fixed. In EA mode, the residual
  model is now evaluated in `psi`, the coordinate used for the fit, and the
  complete simultaneous-in-loop model is used when higher orders were fitted.
  Geometry and higher-harmonic errors from one OLS solve consume the same
  residual variance and `sigma_bg` floor. Fixed-geometry tests explicitly
  distinguish the matched-`psi` result from the retired `phi` scaling, and
  exact-model tests reject the former truncated residual model.
- Do not quote a 25--35% ISOFIT overhead until a current controlled benchmark
  compares the same fit with only the harmonic mode changed.
- Planted harmonic and geometry-recovery tests now cover coefficient recovery,
  signs, angle bookkeeping, masking, and both five-parameter and joint
  higher-order OLS error scales. A broader publication figure spanning
  ellipticity, signal-to-noise ratio, and masking is still desirable before
  making a population-level accuracy claim, but it is no longer needed to
  establish that the implemented coefficient basis and uncertainty scaling are
  internally consistent.

**Proposed manuscript footnote**

> Ciambur (2015) described eccentric-anomaly sampling as defining equal arc
> lengths on an ellipse. Strictly, equal increments in eccentric anomaly are
> equal angular increments on the auxiliary circle, while the physical
> ellipse arc element is
> `ds/dpsi = sqrt(a^2 sin^2(psi) + b^2 cos^2(psi))`. We use “uniform” here in
> the former, parameter-space sense.

This correction should remain a footnote rather than interrupting the main
method narrative. The main text should emphasize the scientifically relevant
effect: the eccentric-anomaly basis places denser samples and sharper Fourier
corrections around the major-axis ends of highly flattened isophotes.

### `section-1.4.2-variance-aware-fitting.md`

**Consistent with code**

- A variance map is sampled along each ellipse with the image.
- The harmonic solve uses `W = diag(1/variance)` and explicitly solves the
  normal equations `(A^T W A)c = A^T W y`.
- The coefficient covariance returned by the WLS path is `(A^T W A)^-1`.
- Shape validation, copied inputs, and warnings for invalid variance values are
  implemented.

**Corrections**

- The code does not obtain WLS by row scaling followed by `lstsq`; it forms and
  solves the normal equations. The paper should describe the implemented
  method and note possible conditioning implications.
- The covariance is statistically calibrated only if the supplied variances
  are calibrated and the sampled measurements can be treated as independent.
  Bilinear interpolation introduces correlations, so “exact covariance” is too
  strong for real images even though the matrix covariance and reported
  estimator are now internally matched.
- The former gradient value/covariance mismatch is fixed through the shared
  `_ring_statistic_and_variance` helper. Describe the propagated gradient error
  as matched to the selected ring estimator, not as an exact covariance for
  interpolation-correlated real-image samples.
- Non-finite and non-positive variances are now invalid samples in both
  single- and multi-band paths. They are warned about and excluded from the
  value and uncertainty together; the retired high-weight clamp must not be
  described as current behavior.
- The existing Legacy Survey example supports a demonstration that WLS changes
  outer error estimates: in this one galaxy the median WLS/OLS intensity-error
  ratio is about 1.52 and reaches 2.12 at the outermost point. It does not
  establish a universal factor for survey data.
- Do not use the existing single-run timings to estimate WLS overhead; their
  order and warm-up conditions do not isolate that cost.

### `section-1.4.3-lsb-outskirts.md`

**Consistent with code**

- The soft outer mechanism derives a flux-weighted inner reference geometry,
  uses a circular mean for position angle, and ramps the penalty with a logistic
  function of semi-major axis.
- Per-axis center, ellipticity, and position-angle weights are supported.
- `damping` shrinks the free update; `solver` can additionally pull toward the
  reference geometry.
- The selector-level penalty discourages choosing solutions far from the
  reference.
- Automatic LSB locking uses a debounced gradient-quality trigger, takes the
  pre-trigger ellipse as its anchor, fixes center/PA/ellipticity in a cloned
  configuration, and affects outward growth only.
- Forced photometry warns that automatic locking is inactive.

**Corrections and framing**

- The feature name says “center regularization,” but current default weights
  regularize center, ellipticity, and PA. Define the actual geometry vector in
  the paper.
- Solver-level regularization is not active with simultaneous harmonics; only
  the selector penalty remains and the code warns. Do not say all modes compose
  without qualification.
- Enabling outer regularization also enables geometry convergence through
  configuration validation. This coupling should be explicit.
- Existing real HSC edge-case outputs strongly demonstrate suppression of
  catastrophic centroid drift, but not a universal improvement in convergence
  rate or photometric accuracy. Call this stabilization and measure flux/profile
  bias with planted halos before recommending it generally.
- Replace the unexecuted “deep mock with bright contaminant” narrative with a
  controlled planted-halo experiment plus the existing real BCG example.

### `section-1.4.4-batch-robustness.md`

**Consistent with code**

- First-isophote retries, permissive propagation of usable nonzero stop codes,
  adaptive mean/median integration, gradient-S/N damping, per-iteration update
  clipping, geometry-convergence checks, and a background-noise floor are
  implemented.
- The retry state and automatic-lock state are preserved in top-level metadata.

**Corrections**

- Adaptive integration affects both the reported ring intensity and radial
  gradient calculation; it is not only an output-layer choice.
- The nominal center-shift clip is 5 pixels for largest-coordinate updates, but
  simultaneous updates use `max(clip_max_shift, 0.05*sma)` for the vector norm.
- Gradient-S/N damping applies only when a gradient uncertainty is available.
- A nonzero stop code does not always contain a valid “best attempt.” Stop code
  1 can return a NaN fallback, and stop code 3 can be a sparse fallback result.
- There is no named “survey batch preset” in the package. Do not present the
  proposed combination of switches as a supported preset.
- The proposed four-object SGA demonstration was not run and contains no fixed
  sample definition. A publication experiment needs preselected objects,
  failure criteria, and recovery metrics.

### `section-1.4.5-multiband.md`

**Consistent with code**

- The joint solver has per-band intercept columns and four shared geometry
  harmonic columns, giving a `(B*N) x (B+4)` design matrix in the common-valid
  case.
- It supports independent, shared, simultaneous-in-loop, and
  simultaneous-original higher-harmonic modes.
- It supports strict common validity and a loose, jagged-row validity mode,
  per-band weights, joint or reference-band harmonic combination, and optional
  decoupling of reported per-band intensities from the joint intercepts.
- The default geometry-parameterized solve scales each band's shared harmonic
  columns by its radial gradient, so the fitted shared parameters are geometry
  steps and the effective WLS information weight scales as
  `w_b*grad_b**2/variance_b`.
- The one-band case delegates to the single-band path.
- A separate `isoster-mb` command and multi-band FITS/QA paths exist.

**Scientific reframing**

- Forced photometry is not inherently biased: it measures each band on a
  reference-band aperture sequence. Independent fits estimate each band's own
  geometry, while the joint fit estimates one data-informed shared geometry.
  These are different estimands. The known-truth color-gradient demonstration
  now supplies a controlled comparison for the common-geometry case; it does
  not make joint geometry universally preferable when morphology genuinely
  changes with wavelength.
- The synthetic color-gradient sweep finds near parity at high signal-to-noise
  and roughly half the color-profile RMS error for the joint fit below
  `S/N ~ 10`. This supports an explicitly signal-to-noise-dependent motivation,
  not a universal claim that joint fitting outperforms independent fitting.
- The default path is supported rather than experimental. Retain warnings and
  cautious placement for `simultaneous_in_loop`, `simultaneous_original`, and
  the deliberately non-default `loose_validity=True` option. State that the
  five-band path has synthetic coverage but its approved real-data validation
  remains deferred.
- Higher-harmonic storage differs by mode. Independent results are already
  Bender-normalized. `shared` fits one dimensionless shared shape and stores a
  different raw amplitude for each band, which normalizes back to the same
  shape. The two experimental `simultaneous_*` modes still store one identical
  raw residual amplitude across bands. The manuscript and table schema must
  state these conventions rather than treating every mode's columns as the
  same quantity.
- A stored five-band asteris benchmark found runtime ratios of 1.30--1.52
  relative to one i-band fit for one 768-pixel cutout. This is promising but not
  a general “about two times” result.

### `section-1.4.6-diagnostics-qa.md`

**Consistent with code**

- Per-isophote stop codes use the canonical set `{-1, 0, 1, 2, 3}`.
- The standard QA summary shows the image, model, residual, intensity profile,
  centroid, ellipticity, PA, harmonics, and curve of growth; the extended
  version adds dedicated higher-order panels.
- Cross-arm and cross-tool plotting helpers and the exhausted-benchmark
  framework exist.
- FITS round trips preserve configuration and top-level metadata in addition to
  the profile table.

**Corrections and placement**

- Describe the actual figure layout rather than calling it a generic six-panel
  plot.
- Serialization supports provenance but is not sufficient for bit-for-bit
  re-execution without inputs and environment capture.
- Benchmark orchestration details belong in a Validation section or appendix,
  not the core method description.

### `section-1.5-comparison.md`

This section is not code-verifiable and should not be folded into the main
Methods section in its current form.

- “Strict generalization of photutils” and “every photutils fit is
  reproducible” are false because sampling, integration, convergence, and
  defaults differ.
- Maintenance-status claims about IRAF need current, sourced verification.
- The AutoProf summary is too reductive to publish without checking its paper
  and current implementation.
- A feature matrix reads as product marketing unless every cell has a defined,
  cited criterion.

Use a short, source-backed comparison in the Introduction or Discussion, and
reserve the Validation section for controlled output and runtime comparisons.

### `section-1.6-limitations-roadmap.md`

**Keep for Discussion**

- Isoster is not a general two-dimensional decomposition or PSF-convolution
  engine.
- Results depend on background subtraction, masks, variance calibration, and
  treatment of the extended PSF and scattered light.
- Optional safeguards, especially LSB regularization, and the breadth of
  multi-band real-data validation still need broader calibration.

**Remove or postpone**

- Do not promise roadmap items in the Methods section.
- Do not state quantitative scalability limits without a measured campaign.
- Keep installation, dependency, and compilation caveats in documentation
  unless they materially affect a published benchmark.

## Evidence status of existing demonstrations

| Topic | Existing artifact | What it supports | What it does not support |
|---|---|---|---|
| WLS | `outputs/example_variance_map/2MASXJ23065343+0031547_ols_vs_wls_qa.png` | A real-image OLS/WLS profile and uncertainty comparison | A population-wide error ratio or WLS accuracy claim |
| LSB stabilization | `outputs/example_hsc_edge_real/lsb_outer_sweep/` | Suppression of large outer centroid drift in three difficult BCG cutouts | Unbiased halo photometry or higher convergence probability |
| Lazy gradient | `outputs/benchmark_lazy_gradient/summary.json` | Fewer sampling calls and measured speed changes in three cases | Identical profiles or a universal speedup |
| Photutils speed | `outputs/benchmark_performance/bench_vs_photutils/benchmark_results.json` | Promising synthetic runtime and agreement pilot | Publication-ready survey-scale performance |
| Numba | `outputs/benchmark_performance/bench_numba_speedup/numba_benchmark_results.json` | Optional-kernel performance varies by case | A fixed three-to-five-fold acceleration |
| Multi-band | `outputs/example_asteris_denoised/`, `outputs/benchmark_multiband/`, and `outputs/example_color_gradient/color_gradient_truth_comparison.png` | End-to-end feasibility, one-object timing, planted-truth WLS recovery, flux-unit invariance, and an S/N-dependent color-profile advantage on a known common-geometry mock | Universal improvement, morphology-dependent bias, population performance, or the deferred five-band real-data validation |
| EA/ISOFIT | `outputs/example_hsc_edgecases/step1_ea_isofit/` plus `tests/unit/test_ols_covariance_scaling.py` | The mode runs on real cutouts; its fitted-angle OLS error scale, complete-model residual, and planted harmonic recovery are regression tested | A population-level recovery map over ellipticity, masking, and signal-to-noise ratio |

## Remaining decisions before drafting Methods prose

1. **Completed:** the eccentric-anomaly OLS parameter-error scale now uses the
   exact fitted model in the fitted angular basis, including
   simultaneous-in-loop higher harmonics, with planted regression coverage.
2. **Completed:** gradient values and propagated variances now use the same
   selected mean or median ring statistic.
3. **Completed:** non-finite and non-positive variance inputs are excluded from
   both ring statistics and their uncertainties in single- and multi-band
   paths.
4. **Decided:** include the supported default multi-band method as a Methods
   subsection, while labeling `simultaneous_*` as experimental, identifying
   `loose_validity` as supported but non-default, and stating that five-band
   real-data validation is deferred.
5. **Partly done** (2026-08-21). The lazy-gradient demonstration is fully
   handled: measured by `benchmarks/draft_timings/` under a controlled
   protocol, archived in `reference_timings.json`, and the technical
   chapter's transcription of it is verified in CI by
   `check_draft_numbers.py` before the docs site publishes.

   The photutils comparison is archived and, as of 2026-08-21, **is** checked
   in CI: `benchmarks/performance/check_speedup_claims.py` rebuilds the
   README, `CLAUDE.md` and `CITATION.cff` clauses from
   `reference_speedup.json` -- including the 243/237/6 coverage figures -- and
   the docs workflow runs it before publishing. It also refuses an archive
   whose own case counts do not add up. What it does **not** fix is the
   protocol: that sweep is still a single timing per configuration in a fixed
   tool order.

   **Still required:** the controlled protocol for the photutils sweep
   (batching, interleaved shuffled order, repeated sessions, as
   `draft_timings` does); the same treatment for any *other* quantitative
   demonstration reaching the manuscript; and a publication commit frozen at
   submission so the archives can be tied to a specific tree.
