# Detailed outline for the Isoster Methods section

Date: 2026-08-11  
Target: concise ApJ software/method paper  
Status: outline only; no prose has been added to `manuscript.tex`

## Editorial principle

The Methods section should explain the scientific estimator and the design
choices that distinguish Isoster. It should not reproduce the user manual.
Configuration names are useful only when they make an experiment reproducible.
Long flag lists, file-column dictionaries, command examples, installation
details, and roadmaps should remain in the online documentation or appendix.

The central narrative is:

> Isoster retains the interpretable harmonic estimator of classical
> isophotal analysis, makes its repeated image operations efficient, propagates
> survey variance information through the harmonic fit, and supplies explicit
> safeguards and diagnostics for faint outskirts and batch processing.

This claim is strong but appropriately bounded. It does not assert that every
optional feature is superior for every galaxy.

## Recommended section architecture

### 2. Method

Opening paragraph, approximately 100--150 words:

- Define the input: background-subtracted image, Boolean bad-pixel mask,
  initial ellipse geometry, and optional per-pixel variance map.
- Define the output: a radial sequence of ellipse geometries, mean/median
  intensities and uncertainties, higher-order deviations, quality states, and
  optional cumulative photometry.
- State that the implementation follows the classical harmonic correction
  framework but changes how samples are acquired, uncertainties are used, and
  unstable low-S/N updates are controlled.
- Point to Figure 1 for the full image-level and per-isophote data flow.

#### 2.1 Harmonic isophote estimator

Purpose: give enough mathematics to define what Isoster estimates, without
repeating the historical review from the Introduction.

Main ideas:

1. Parameterize a trial ellipse by center `(x_0, y_0)`, semi-major axis `a`,
   ellipticity `epsilon = 1-b/a`, and position angle `PA`.
2. Sample intensity along the trial path and write the residual angular
   structure as

   `I(theta) = I_0 + sum_n [A_n sin(n theta) + B_n cos(n theta)]`.

3. Explain that first- and second-order terms map to corrections in center,
   ellipticity, and PA through the local radial intensity gradient.
4. State the default coordinate-descent rule: update the geometry coordinate
   associated with the largest normalized low-order harmonic. Mention the
   simultaneous-update option in one sentence, not as the default.
5. Define convergence operationally: after the minimum iteration count, the
   largest geometry-driving harmonic must fall below a threshold tied to the
   measured residual scale; geometry stability can provide a secondary
   criterion in stabilized outer fits.
6. Define higher-order outputs with the normalization the code actually
   applies, and state where the two forms in the codebase differ.
   `compute_deviations` stores `a_n = A_n_raw / (a * |dI/da|)` using the
   **absolute** gradient; the plot-time helper used by the multi-band raw
   paths applies the **signed** Bender form `-A_n_raw / (a * dI/da)`. The two
   coincide wherever the gradient is negative — the normal outward-declining
   case — and differ in sign on the rare positive-gradient row, which does
   occur in the LSB regime. Use the wording already settled in
   `docs/technical/1.4.6-diagnostics-qa.md` rather than restating it here.

Equations to include:

- ellipse coordinates;
- harmonic intensity model;
- one compact equation or cited mapping for geometry corrections;
- normalized higher-order coefficients.

Avoid:

- a catalog of every stop condition;
- unsupported statistical interpretation of the convergence threshold;
- calling eccentric anomaly arc length.

#### 2.2 Image-level radial fitting sequence

Purpose: show how individual ellipse fits become a complete profile.

Main ideas:

1. Determine or accept an initial center and anchor semi-major axis.
2. Fit the central pixel and anchor ellipse.
3. If the anchor fit is unusable, retry a bounded sequence of smaller anchor
   radii with progressively constrained geometry.
4. Grow inward first. This supplies a stable central profile and, when enabled,
   the reference geometry used by outer regularization.
5. Grow outward from the anchor, using the previous accepted ellipse as the
   starting geometry.
6. Stop at the configured radius, image boundary, or terminal quality state.
7. Sort the accepted radii and optionally calculate curve-of-growth quantities.
8. A template path reuses supplied geometries for forced photometry without
   changing them.

This subsection should be paired with the left half of Figure 1 and the compact
pseudocode below. It should not list all 62 configuration fields.

#### 2.3 Efficient sampling and iterative computation

Purpose: explain the mechanisms behind the speed claim before presenting
benchmark results in the Validation section.

Main ideas:

1. At each trial geometry, sample
   `N = max(64, floor(2*pi*a))` positions along the ellipse.
2. Compute all coordinates as arrays, then acquire image and variance values in
   one vectorized interpolation call per array; sample the mask with
   nearest-neighbor interpolation.
3. Construct and solve a small harmonic linear system. The cost per trial is
   linear in the number of path samples plus the small dense solve.
4. Cache the radial gradient after its first evaluation and refresh it only
   when convergence stalls or the estimate becomes inadequate. This avoids one
   or more extra path samplings on most successful iterations.
5. Optional Numba kernels accelerate coordinate and design-matrix construction;
   NumPy/SciPy fallbacks preserve functionality. Treat this as an implementation
   detail rather than the primary explanation of speed.
6. Clarify that two-dimensional model rendering is a separate inverse
   reconstruction and not part of the fitting loop.

Quantitative statements should point forward to the controlled runtime and
fidelity experiment in Section 3. Avoid embedding old benchmark numbers here.

#### 2.4 Variance-aware harmonic fitting

Purpose: make per-pixel uncertainty support a central scientific distinction,
while stating its assumptions honestly.

Main ideas:

1. When no variance map is supplied, solve the ordinary least-squares harmonic
   system and estimate errors from residual scatter.
2. When a variance map is supplied, bilinearly sample it at the same path
   coordinates and solve

   `(A^T W A)c = A^T W I`, with `W_ii = 1/sigma_i^2`.

3. Use `(A^T W A)^-1` as the formal coefficient covariance. State that this is
   calibrated when the variance model is correct and inter-sample correlations
   are negligible; resampling correlations can make it optimistic.
4. Explain how coefficient and radial-gradient uncertainties enter the reported
   geometry and higher-harmonic errors, after the gradient-statistic mismatch is
   resolved.
5. Define accepted variance-map semantics: the input is variance, not inverse
   variance; invalid or non-positive inverse-variance pixels should be converted
   to negligible weight or included in the mask.
6. Point to the real Legacy Survey OLS/WLS comparison in Figure 3, framed as an
   example rather than population calibration.

Pre-drafting requirement: correct or explicitly document the current treatment
of non-positive variance and reconcile the gradient statistic with its
uncertainty.

#### 2.5 Higher-order structure and eccentric-anomaly mode

Purpose: describe the non-elliptical shape measurement and the
eccentric-anomaly geometry path with its original Jedrzejewski basis made
explicit.

Main ideas:

1. Default mode fits geometry with orders 1--2 and estimates requested higher
   orders after convergence.
2. `simultaneous_harmonics` enlarges the design matrix so geometry and higher
   orders can be estimated jointly. Distinguish:
   - in-loop joint fitting;
   - classical geometry iteration followed by one joint higher-order refit.
3. If the extended matrix has too few valid points, fall back to the
   five-parameter geometry fit and record a warning.
4. Eccentric-anomaly mode samples uniformly in `psi` and evaluates the harmonic
   basis in `psi`. Following Ciambur (2015), describe this as uniform angular
   sampling on the auxiliary circle, which redistributes samples toward the
   major-axis ends. Do not call it exactly uniform in physical arc length.
5. Explain why this representation is intended for strongly flattened systems,
   citing ISOFIT, but defer claims of improved recovery to the planted mock in
   the Validation section.

Add a short footnote at the first use of “uniform eccentric-anomaly sampling”:

> Ciambur (2015) described this transformation as defining equal arc lengths
> on an ellipse. More precisely, equal increments in eccentric anomaly are
> equal angular increments on the auxiliary circle; physical arc length along
> the ellipse varies as
> `ds/dpsi = sqrt(a^2 sin^2(psi) + b^2 cos^2(psi))`.

The footnote should be factual and neutral. It should immediately return to the
paper's valid physical motivation: improved representation of major-axis
structure and an ellipse-adapted harmonic basis.

Decision gate:

- The direct use of `psi`-basis low-order coefficients follows the
  Jedrzejewski (1987) correction equations and does not require conversion to a
  `phi`-basis Fourier series. Remove the earlier draft claim that such a
  conversion occurs.
- Before drafting, correct the OLS parameter-error residual scaling so it uses
  the same angular basis and fitted model as the coefficient covariance.
- Retain the focused geometry-recovery test to validate sign conventions,
  recovery accuracy, and uncertainty coverage across ellipticity.

#### 2.6 Stabilization in low-surface-brightness outskirts

Purpose: present Isoster's distinctive outer-profile controls as transparent,
optional assumptions.

Organize the safeguards from least to most restrictive:

1. **Damped and bounded iteration.** Apply a global geometry damping factor,
   reduce it further when the gradient S/N is low, and clip single-iteration
   changes in center, ellipticity, and PA.
2. **Alternative convergence information.** Allow consecutive small geometry
   changes to terminate an otherwise noisy harmonic iteration. If a measured
   background RMS is supplied, include its configured sample-count-scaled floor
   in the convergence scale; call this a heuristic.
3. **Soft outer regularization.** Build a flux-weighted inner reference geometry
   and gradually shrink or pull outer updates toward it with a logistic radial
   weight. Write the shrinkage/pull equation and define per-axis weights.
4. **Hard automatic lock.** After a debounced sequence of unreliable gradients,
   freeze center, ellipticity, and PA at the last pre-trigger geometry and
   continue measuring intensity outward.

Scientific framing:

- These mechanisms exchange freedom for stability and can suppress genuine
  outer asymmetry if used too strongly.
- They are disabled by default and must be evaluated against the science goal.
- The existing HSC BCG example demonstrates suppression of catastrophic
  centroid drift, not unbiased halo recovery.
- Solver-level outer regularization is incompatible with simultaneous-harmonic
  updates in the current code; state this rather than claiming universal
  composability.

Figure 3 should show the free and stabilized fits on the same real BCG, plus a
planted-halo bias/recovery panel once available.

#### 2.7 Multi-band joint fitting

Purpose: describe the extension in proportion to its current maturity, which
is higher than this outline originally assumed.

Do not label the interface experimental wholesale — that is now wrong. The
default configuration (shared validity, joint per-band intercepts, the
geometry-parameterised solve, `independent` higher harmonics, OLS or WLS) is
**supported**. What remains qualified is narrower and should be stated as
such: the `simultaneous_*` higher-harmonic modes are experimental and warn
when selected; `loose_validity=True` is supported but non-default and does
not warn; and the CLI arguments and Schema-1 output layout may still change.

Main ideas:

1. Define the three scientifically distinct workflows:
   - forced photometry: reference-band-defined common apertures;
   - independent fits: each band's natural fitted geometry;
   - joint fit: a single geometry informed by all included bands, with per-band
     intensities.
2. For `B` bands with a common set of `N` valid samples, show the block design
   matrix with `B` intercepts and shared `(A_1, B_1, A_2, B_2)`, having shape
   `(B*N) x (B+4)`.
3. Explain per-band weights and strict versus loose sample validity.
4. Summarize the four higher-harmonic modes in a compact table; do not explain
   every configuration switch in prose.
5. State that a one-band call delegates to the single-band implementation.
6. Disclose the current raw-versus-normalized higher-harmonic storage difference
   across modes if it has not been unified before publication.

Do not call forced photometry biased or joint fitting unbiased. Figure 4
should compare all three estimands on a planted color-gradient mock.

A real five-band example is **deferred, not required**. The asteris cutouts
are not available on the development machine, so the existing B=5 numbers are
historical and not reproducible; treat their absence as a stated evidence
limitation ("real-data validation at B=5 is outstanding") rather than a
publication blocker. The planted mock carries the estimand argument on its
own.

If the paper must be shorter, move this subsection and Figure 4 to an appendix
or defer it to a dedicated multi-band paper. The single-band method remains a
complete publication without it.

#### 2.8 Outputs, diagnostics, and reproducibility

Purpose: show how scientific users can audit a fit, in one short subsection.

Main ideas:

1. Every ellipse records geometry, intensity, uncertainty, iteration count,
   stop code, and optional diagnostic/harmonic quantities.
2. Define the five stop codes in a compact table, carefully distinguishing
   successful convergence from retained non-converged or fallback estimates.
3. The standard QA product juxtaposes image/isophotes, model, residual,
   intensity, centroid, ellipticity, PA, harmonics, and curve of growth.
4. FITS output has four HDUs: Primary, `ISOPHOTES`, `CONFIG`, and `META`.
5. Reproducibility requires those products plus the input image, mask, variance
   map, and recorded software environment.

Keep ASDF, Astropy conversion routines, command-line syntax, and the exhaustive
column schema in documentation or an appendix.

### 3. Validation and demonstrations

Although not part of the requested Methods prose, the method claims need a
separate validation section. The figures below are designed so the Methods
section can point to evidence rather than carry benchmark results itself.

#### 3.1 Geometry and harmonic recovery

- Controlled Sérsic and edge-on mocks with planted center, ellipticity, PA, and
  selected higher-order coefficients.
- Grid in ellipticity, S/N, masked fraction, and optional contaminating source.
- Compare polar-angle/post-hoc, eccentric-anomaly/post-hoc, and
  eccentric-anomaly/joint modes.
- Report median bias, robust scatter, failure fraction, radial completeness, and
  coverage of formal uncertainties.
- Use this experiment to resolve the eccentric-anomaly update question.

#### 3.2 Runtime and fidelity

- Rerun Isoster and photutils on the same simulated and real cutouts at the
  frozen publication commit.
- Include a grid of image sizes, galaxy sizes, ellipticities, masks, and profile
  depths; warm each implementation and repeat timings.
- Report wall time distributions and profile/geometry agreement, not only a
  speed ratio.
- Ablate lazy-gradient and Numba separately. For lazy gradient, report sampling
  calls and profile differences together.
- Record CPU, thread controls, dependency versions, and run order.

#### 3.3 Survey-noise and faint-outskirt demonstrations

- OLS versus WLS on the existing Legacy Survey galaxy, with the variance-map
  provenance stated.
- Free versus stabilized geometry on the existing HSC BCG example.
- Add a planted faint halo to measure how regularization/locking changes profile
  bias and radial recovery. A real-image stability plot alone cannot establish
  accuracy.

#### 3.4 Multi-band feasibility, if retained

- Planted color-gradient mock separating the forced, independent, and joint
  estimands.
- At least one real HSC five-band example using the existing asteris pipeline.
- Report geometry recovery, per-band profile bias/scatter, valid radial extent,
  failure rate, and runtime.

## Required visual material

### Figure 1 — Algorithm flow and per-isophote loop (required)

One two-part vector figure:

- **Left:** image-level flow from inputs to central/anchor fit, inward growth,
  outer-reference construction, outward growth, optional lock transition, and
  final profile/QA/FITS products.
- **Right:** per-isophote loop from coordinate generation through vectorized
  sampling, mask/variance selection, harmonic solve, gradient evaluation,
  convergence test, bounded geometry update, and retained result.
- Use color only to distinguish data operations, estimator operations, and
  optional safeguards. Dashed borders should denote optional paths.

This figure replaces several paragraphs of implementation narration.

### Figure 2 — Recovery and performance validation (required)

Recommended four panels:

1. geometry/harmonic recovery bias versus ellipticity for the angular modes;
2. recovered planted `a_4` or edge-on harmonic structure versus truth;
3. wall time versus total sampled path points for Isoster and photutils;
4. profile-fidelity distribution, with lazy-gradient ablation as an inset or
   point style.

Do not construct this from cached headline ratios alone. Regenerate from the
publication commit.

### Figure 3 — Scientific feature demonstrations (required)

Recommended three columns:

1. Legacy Survey image plus OLS/WLS outer intensity uncertainties;
2. HSC BCG free versus stabilized centroid, ellipticity, PA, and profile;
3. planted faint-halo recovery showing the stability/bias trade-off.

Reuse the existing real-image artifacts as starting points, but redraw a compact
publication figure from persisted fit tables rather than embedding full QA
screenshots.

### Figure 4 — Multi-band estimator comparison (conditional)

Recommended panels:

1. schematic of the block design matrix;
2. planted color-gradient mock with true band-dependent structure;
3. forced, independent, and joint profile/geometry recovery;
4. real five-band HSC feasibility example and residual thumbnails.

Include only if multi-band fitting remains in the main paper.

### Table 1 — Method modes and scientific purpose

A compact table with columns: capability, estimator choice, intended use,
principal assumption, and default state. Suggested rows are OLS/WLS, polar/EA
angle, post-hoc/joint harmonics, free/soft/locked outer geometry, forced/free
multi-band geometry, and mean/median/adaptive ring statistic.

This is not a competitor feature matrix.

### Table 2 — Stop codes (small; optional)

Five rows defining `-1, 0, 1, 2, 3`, whether an isophote may be retained, and
what a pipeline should inspect. It may be folded into prose if space is tight.

## Compact pseudocode for Figure 1 or an appendix

```text
function fit_image(image, mask, variance, initial_geometry, options):
    validate inputs and resolve configuration
    fit central sample
    anchor = fit_isophote(initial_radius, initial_geometry)
    if anchor is unusable:
        retry bounded smaller radii and constrained geometries

    inner_profile = grow inward from anchor
    outer_reference = summarize inner geometry if regularization is enabled

    geometry = anchor.geometry
    for radius in outward_radius_sequence:
        result = fit_isophote(radius, geometry, outer_reference)
        retain result if it satisfies the propagation rule
        if debounced low-S/N trigger fires:
            geometry = last_pretrigger_geometry
            fix center, ellipticity, and position angle for later radii
        else:
            geometry = result.geometry

    combine and sort central, inward, anchor, and outward results
    optionally compute curve of growth
    return profile, resolved configuration, and diagnostic metadata

function fit_isophote(radius, starting_geometry, outer_reference):
    geometry = starting_geometry
    for iteration in 1..maximum_iterations:
        coordinates, fit_angle = ellipse_path(radius, geometry)
        intensity, mask_state, variance = vectorized_sample(coordinates)
        select valid samples and apply configured clipping
        coefficients, covariance = solve_harmonic_system(fit_angle, intensity, variance)
        evaluate or reuse radial_gradient
        if harmonic or geometry convergence criterion is satisfied:
            return measured_isophote
        correction = map_low_order_harmonics_to_geometry(coefficients, radial_gradient)
        correction = apply_damping_clipping_and_optional_outer_regularization(correction)
        geometry = geometry + correction
    return best_retained_or_fallback_isophote_with_stop_code
```

The published pseudocode should use scientific nouns rather than literal
internal function names, and should show the forced-photometry path as a short
side branch rather than duplicating the full loop.

## Material to move out of the main Methods section

- complete module tree and public API inventory;
- all configuration defaults and validation rules;
- installation, dependency, JIT-compilation, and command-line instructions;
- detailed FITS column schema and ASDF examples;
- benchmark campaign orchestration internals;
- long competitor feature matrix;
- roadmap promises;
- full synthetic-demo scripts.

These can be cited as online documentation, placed in appendices, or archived
with the reproducibility package.

## Drafting gates

The full Methods prose should begin only after the author chooses:

1. correction of the EA+OLS parameter-error scaling and validation of geometry
   recovery and uncertainty coverage before the paper freeze;
2. whether multi-band fitting is central, appendix material, or deferred;
3. which LSB mode is the paper's demonstrated reference configuration;
4. the frozen benchmark sample and hardware environment;
5. whether the variance-gradient consistency issue is fixed in code or stated
   as a limitation.
