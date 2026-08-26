# ISOSTER Technical Specification

## Purpose

ISOSTER is a Python library for elliptical isophote fitting on 2D images, with a function-first API and vectorized sampling/fitting internals.

## Public Interfaces

- `isoster.fit_image(image, mask=None, config=None, template=None, variance_map=None)`
- `isoster.fit_isophote(...)`
- `isoster.isophote_results_to_fits(...)`
- `isoster.isophote_results_from_fits(...)`
- `isoster.isophote_results_to_astropy_tables(...)`
- `isoster.build_isoster_model(...)`
- CLI entry point: `isoster` (`isoster/cli.py`)

### Multi-Band Public Interface

Lives under a parallel module tree at `isoster/multiband/`. The single-
band interfaces above are **not** modified.

- `isoster.multiband.fit_image_multiband(images, masks=None, config=None,
  variance_maps=None, template_isophotes=None)` — joint free fit on
  aligned same-pixel-grid images (or forced photometry when a template is
  given). One shared geometry per SMA, per-band intensities and per-band
  harmonic deviations. A *complement* to forced photometry rather than a
  replacement -- the two estimate different quantities: forced photometry
  measures every band through the reference band's geometry, the joint fit
  derives one geometry to which every band contributes.
- `isoster.multiband.IsosterConfigMB` — multi-band-specific config
  (sibling of `IsosterConfig`, no inheritance, deliberately reduced field
  set). The driver validates that all inputs share one pixel grid
  (shape-only check; no WCS validation is performed).
- `isoster.multiband.load_bands_from_hdus(hdus)` — helper to extract
  `(images, masks, variance_maps, bands)` tuples from FITS HDUs.

`fit_image_multiband` with `len(bands) == 1` delegates to `fit_image` and
returns the legacy single-band schema unmodified.

Status: **supported** for the default configuration (shared validity, joint
intercepts, geometry-parameterised solve, `independent` higher harmonics), in
both OLS and WLS. Promoted from experimental (beta) on 2026-08-15 after the
maturity pass recorded in `docs/agent/plans/2026-08-14-multiband-maturity.md` (agent-internal, not published):
consistent WLS gradient weighting, a geometry-parameterised solve, full-fit
unit invariance, planted-truth WLS coverage, repaired loose validity, a
known-truth colour-gradient demonstration, and a B=1 parity harness over a
16-cell config matrix that pins agreement with single-band at machine
precision. Individual features still carry their own warnings — see
`docs/10-multiband.md`. The Schema-1 output layout may still gain columns
additively. The CLI (`isoster-mb`), FITS/ASDF I/O,
curve-of-growth attachment, LSB auto-lock, and outer-center
regularization are all implemented with multi-band semantics. The
single-band ISOFIT API (`simultaneous_harmonics`) is not carried over;
higher-order harmonics are handled via the `multiband_higher_harmonics`
enum instead. See `docs/10-multiband.md` for the user-facing reference.
The locked 24-decision design record is kept in internal planning notes
outside the public docs tree.

## Core Modules

- `isoster/driver.py`: image-level orchestration and mode routing.
- `isoster/fitting.py`: per-isophote fitting loop, gradient checks, geometry updates.
- `isoster/sampling.py`: vectorized ellipse sampling via `scipy.ndimage.map_coordinates`.
- `isoster/config.py`: `IsosterConfig` schema and validators.
- `isoster/model.py`: 2D model reconstruction by radial interpolation. `build_isoster_model()` supports `harmonic_orders=None` (auto-detect from isophote keys) and `use_eccentric_anomaly=None` (auto-detect from isophote dicts, or explicit `True`/`False` override) for correct harmonic evaluation in EA-mode fits.
- `isoster/cog.py`: curve-of-growth computation and crossing flags.
- `isoster/utils.py`: serialization to/from FITS and Astropy tables.
- `isoster/plotting.py`: QA visualization (`plot_qa_summary`, `plot_qa_summary_extended`, `plot_comparison_qa_figure`). Multi-method comparison figures support three auto-detected layout modes (solo, 1v1, 3-way), cross-method PA anchoring, errorbars, stop-code markers, mask overlays, contour-gated isophote overlays, model iso-brightness contours, and optional asinh SB-profile scaling with an intensity-zero reference line. Isophote contours dispatch by tool and harmonic convention: isoster phi-mode uses image-azimuth contours, isoster EA/psi-mode and photutils use eccentric-anomaly contours, and AutoProf deliberately falls back to pure ellipses until its coefficient scale is ported. `build_method_profile()` standardizes isophote lists or array dicts into a common profile format. `METHOD_STYLES` provides default visual styles for isoster/photutils/autoprof.
- `isoster/cli.py`: command-line interface entry point.
- `isoster/numba_kernels.py`: optional Numba-accelerated kernels with NumPy fallback.
- `isoster/output_paths.py`: output directory and file path construction helpers.
- `isoster/optimize.py`: compatibility facade re-exporting driver/fitting APIs.
- `isoster/multiband/`: parallel module tree for
  joint multi-band free fit. Sibling of the core modules, never edits
  them. Contains `sampling_mb.py`, `fitting_mb.py`, `driver_mb.py`,
  `config_mb.py`, `utils_mb.py`, `plotting_mb.py`, and a multi-band
  numba kernel. Imports shared low-level helpers from the core modules
  (`compute_ellipse_coords`, `_prepare_mask_float`, etc.) but provides
  its own joint-design-matrix solver, joint gradient combiner, and
  iteration loop. Stage-2 backports landed as new fields on
  `IsosterConfigMB` (no inheritance per design D23):
  `fit_per_band_intens_jointly` (D11, default `True`; renamed from the
  pre-cleanup `fix_per_band_background_to_zero` in Section 6) and the
  loose-validity family
  `loose_validity` / `loose_validity_min_per_band_count` /
  `loose_validity_min_per_band_frac` /
  `loose_validity_band_normalization` (D9 backport, locked
  2026-05-01).
  - `MultiIsophoteData` (sampler return type) carries two coexisting
    layouts: a rectangular `(B, N_intersect)` view (`intens` /
    `variances`) shared with the legacy shared-validity callers, and
    optional jagged per-band lists (`intens_per_band` /
    `phi_per_band` / `variances_per_band`) populated only under
    `loose_validity=True`. `n_valid_per_band` is always populated.
    The shared-validity path is byte-identical to Stage 1; the
    loose-validity path uses the numba-JIT
    `build_joint_design_matrix_jagged` builder (with a NumPy fallback).
  - **Section 6 (locked 2026-05-02):** ``multiband_higher_harmonics``
    enum (``'independent'`` | ``'shared'`` | ``'simultaneous_in_loop'`` |
    ``'simultaneous_original'``, default ``'independent'``) and
    ``harmonic_orders: List[int] = [3, 4]`` land on
    ``IsosterConfigMB``. ``'shared'`` replaces the per-band post-hoc
    ``_attach_per_band_harmonics`` step with one joint solve over a
    ``(B·N, 2·L)`` design matrix where ``L = len(harmonic_orders)``.
    ``'simultaneous_*'`` extends the per-iteration joint design matrix
    from ``(B·N, B + 4)`` to ``(B·N, B + 4 + 2·L)`` with shared
    higher-order columns; ``simultaneous_in_loop`` solves this every
    iteration, ``simultaneous_original`` runs the wider solve only
    once post-hoc (Ciambur 2015 original variant). Two new numba
    kernels (``build_joint_design_matrix_higher`` /
    ``build_joint_design_matrix_jagged_higher``) carry the wider
    matrix through both shared- and loose-validity paths. Per-band
    Schema 1 columns ``a<n>_<b>`` / ``b<n>_<b>`` (and
    ``a<n>_err_<b>`` / ``b<n>_err_<b>``) are written in every mode, but
    **not in the same units**, and ``result['harmonics_shared']`` is the
    flag that says which convention applies. Under ``'independent'`` the
    stored values are already Bender-normalized (``compute_deviations``
    normalizes at fit time) and must not be normalized again. Under
    ``'shared'`` they are raw and **band-distinct**: the solve fits one
    dimensionless shape (each band's columns scaled by ``-sma*grad_b``)
    and converts back through each band's own scale, so the invariant is
    that they all normalize to one value. Under ``'simultaneous_*'`` they
    are raw and identical across bands, which means a shared raw residual
    rather than a shared shape — one of the two reasons those modes keep
    their experimental warning. D16 per-band Bender normalization is
    applied at plotting time only when ``harmonics_shared`` is true.
    Section 6 of the internal multi-band design record (not part of the
    public docs tree).
  - **Geometry-parameterised solve (default 2026-08-15):**
    ``geometry_parameterized_solve=True`` scales each band's four
    *shared* design columns by that band's radial gradient, so the shared
    free parameters are geometry steps rather than one harmonic
    amplitude. A common step ``delta`` produces amplitude
    ``delta*grad_b`` in band ``b``, so sharing an amplitude is
    misspecified whenever the bands' gradients differ. The effective
    per-band weight becomes ``w_b*grad_b**2/var_b`` (minimum-variance)
    instead of ``w_b/var_b``. Threaded through all four solvers via
    ``per_band_gradients=``; the first iteration of each isophote falls
    back to the amplitude form for lack of a previous gradient.
    **Invariant:** the fitted block is rescaled by the pooled
    ``grad_joint`` so downstream consumers keep their units, but band
    ``b``'s fitted amplitude is ``delta*grad_b``, so every site that
    *reconstructs the model* must apply ``band_scale[b] =
    grad_b/grad_joint`` — per-band residuals, convergence RMS, OLS error
    scaling, and harmonic subtraction. Omitting it leaves the geometry
    step exact while making every residual-derived quantity wrong.
  - **Joint gradient pooling weights:** the per-band gradients are
    pooled with the weight each band carried *in the harmonic solve*,
    ``W_b = w_b * sum_i(1/var_{b,i})`` under WLS (``w_b`` under OLS), via
    ``joint_gradient_pooling_weights``, using the **post-clipping**
    variance arrays. Pooling with the bare ``w_b`` recovers the wrong
    geometry step and the error is not one-signed. Reference mode uses a
    one-hot pooling weight so the reference band's own gradient is used.
    Full derivation and the exactness caveat: ``docs/10-multiband.md``,
    "Joint gradient weighting".

## Key Constants

- `ACCEPTABLE_STOP_CODES = {0, 1, 2}` (`driver.py`): stop codes considered acceptable for continued fitting (outward/inward growth). Used by `_is_acceptable_stop_code()` to gate whether the next SMA step proceeds.

## Runtime Modes

`fit_image` selects exactly one mode in this priority order:

1. `template` provided -> template-based forced photometry (`_fit_image_template_forced`).
2. Otherwise -> regular iterative fitting (central pixel, anchor isophote, optional inward growth, outward growth).

(`template_isophotes` is supported as a deprecated alias for `template`).

### Forced Photometry Contract

`fitting.extract_forced_photometry` samples the fixed ring, sigma-clips it, and
reduces it to an intensity exactly as regular mode does; it skips only the
geometry solve. Two consequences follow from that boundary and are easy to get
wrong in opposite directions:

- **Harmonics are measured, not skipped.** When `compute_deviations` (or
  `simultaneous_harmonics`) is on, the function measures the radial gradient
  via `compute_gradient` and runs `compute_deviations` per order, so `a_n` /
  `b_n` carry the same Bender normalization as regular mode and are directly
  comparable with it. On identical rings with `use_lazy_gradient=False` the two
  paths agree bit-for-bit; under the default lazy gradient the regular path may
  reuse a cached gradient, so they agree closely instead.
- **The gradient config is built from the effective arguments.** `integrator`
  and `use_eccentric_anomaly` come from the call, not from `config`, because
  the current ring was sampled and reduced by those arguments. Taking them from
  `config` (or from defaults when `config=None`) would let the comparison ring
  follow different settings, making the Bender numerator and denominator
  describe two different ring statistics. Only `astep` / `linear_growth`, which
  have no arguments, come from `config`.
- **Unmeasurable rings report `NaN`, decided by status and never by value.**

### Measurement status, and why value tests are not enough

Two helpers return plausible numbers on their failure paths, and both numbers
are indistinguishable from legitimate results by inspection:

| Helper | Failure output | Why a value test fails |
|---|---|---|
| `compute_gradient` | `-1.0`, or `previous_gradient * 0.8` | `-1.0` is a perfectly legal gradient |
| `compute_deviations` | `0.0, 0.0, 0.0, 0.0` | `0.0` is the correct amplitude for a perfect ellipse |

`compute_gradient` also returns `gradient_error = None` on *some* measured
paths, so error-is-None does not discriminate either.

Both therefore accept an opt-in `return_status=True` that appends an explicit
status. Default `False` keeps every existing caller unchanged.

- `GRADIENT_MEASURED` / `GRADIENT_NO_CURRENT_RING` /
  `GRADIENT_NO_COMPARISON_RING` / `GRADIENT_SLOPE_GUARD`
- `DEVIATIONS_MEASURED` / `DEVIATIONS_UNDERDETERMINED` / `DEVIATIONS_SINGULAR` /
  `DEVIATIONS_NO_FACTOR`

Forced photometry writes a harmonic only when the gradient status is
`GRADIENT_MEASURED` **and** that order's deviation status is
`DEVIATIONS_MEASURED`; anything else yields `NaN` for that order. The status is
per order, so one degenerate order does not invalidate the others.

`grad` / `grad_error` / `grad_r_error` are exposed under `debug=True`, matching
the regular path.

Regression coverage: `tests/unit/test_forced_photometry_harmonics.py`.

**`simultaneous_harmonics` in forced mode.** The simultaneous solve fits the
geometry harmonics and the higher orders as one system during the iteration.
Forced mode imposes the geometry, so those semantics have nothing to attach to.
`fit_image` emits a `UserWarning` and measures the higher orders independently
per order rather than silently substituting a different strategy.

**Legacy sentinel, not a measurement:** `driver.fit_central_pixel` (the
`sma = 0` row) writes `0.0` harmonics. There is no ring at zero radius, so the
value is undefined rather than unmeasured. It is retained for backward
compatibility and is explicitly excluded from any harmonic calibration; treat
it as a sentinel, never as a measured coefficient.

## Regular Fitting Contract

For each SMA in regular mode:

1. Sample ellipse points (`sampling.extract_isophote_data`) with `n_samples = max(64, int(2*pi*sma))`.
2. Apply sigma clipping to sampled `(angle, intensity)` pairs.
3. Fit harmonics:
   - **Default path** (`simultaneous_harmonics=False`): Fit 5-param model (`I0`, `A1`, `B1`, `A2`, `B2`) via `fit_first_and_second_harmonics()`. Higher-order harmonics fitted post-hoc after convergence.
   - **ISOFIT path** (`simultaneous_harmonics=True`): Fit all harmonics simultaneously via `fit_all_harmonics()` using an extended design matrix `[1, sin(θ), cos(θ), sin(2θ), cos(2θ), sin(n₁θ), cos(n₁θ), ...]`. Falls back to 5-param when `n_points < 1 + 2*(2 + len(orders))`. Geometry updates use `A1, B1, A2, B2 = coeffs[1:5]` identically in both paths.
   - **WLS mode** (`variance_map` provided to `fit_image`): All harmonic fits use Weighted Least Squares with `w_i = 1/σ²_i`. The covariance matrix `(A^T W A)^-1` is exact — no residual-variance scaling is needed. This cleanly separates photon noise from galaxy structure scatter and automatically down-weights high-variance pixels (cosmic rays, hot pixels). When `variance_map=None`, the OLS path is byte-identical to the non-WLS code.
   - **Invalid-variance policy**: a variance-map entry that is not finite or not strictly positive is invalid; the corresponding sample is dropped during sampling rather than substituted with a placeholder value, so a ring's reported statistic and its reported uncertainty always describe the identical set of samples. Full policy, the retired sentinel/clamp it replaces, and its measured impact are below, in "Invalid-variance policy".
   - **OLS covariance scaling**: under OLS the solvers return `(A^T A)^-1`, which is only a shape — it becomes a covariance after multiplication by the residual variance of the fit. That variance is computed **once**, in `fit_isophote`, from the exact fitted model evaluated at `angles` (the coordinate the coefficients were fitted in: ψ under EA sampling, φ otherwise) with `ddof = len(coeffs)`. The single value — **including the `sigma_bg**2` floor, applied at that same point** — scales both the geometry 5×5 block and the higher-order harmonic errors, so one fit yields one error scale. It reaches `compute_parameter_errors` through its `residual_variance=` keyword; that call deliberately does **not** pass `var_residual_floor`, since the floor is already applied. Evaluating the model at `phi` under EA sampling, or using the truncated 5-term model to rescale an ISOFIT `in_loop` fit, both produce internally inconsistent uncertainties — see `docs/agent/journal/2026-08-12_ea-ols-review.md` (agent-internal, not published).
   - **Exactly determined fits**: `isofit_min_points` equals the ISOFIT parameter count exactly (`1 + 2*(2 + L)` is `5 + 2*L`), so ISOFIT switches on at precisely `N == P`, where the model passes through every sample and no residual degrees of freedom remain. The residual variance is then reported as `0.0` (subject to the `sigma_bg` floor), which propagates to zero errors meaning "not measurable". In `compute_parameter_errors`, `residual_variance=0.0` means exactly that, and is distinct from `residual_variance=None`, which means "not supplied — rebuild the five-parameter model" and is retained only for external callers of this public function.
4. Estimate radial gradient (`fitting.compute_gradient`). When `variance_map` is provided, gradient error uses exact per-sample variance (`Var(mean) = Σσ²_i / N²`) instead of scatter-based estimates. See "Gradient error and ring statistics" below for the full formulas, the deliberate mean/median asymmetry with the reported intensity, and a measured consequence for the reported gradient value itself.
5. Update geometry based on dominant harmonic coefficient.
6. Check convergence criterion: `abs(max_amp) < conver * rms` with iteration index check `i >= minit`.

### Invalid-variance policy

A per-pixel variance describes how noisy that pixel's measurement is; to carry any usable
information it must be a finite, strictly positive number. `driver.py` scans the caller's
`variance_map` once per `fit_image` call and marks every non-finite entry (`NaN` or infinite)
and every non-positive entry (zero or negative) as `NaN`, emitting one warning per category
(non-finite, non-positive). `sampling.extract_isophote_data` then drops the corresponding
sample outright, the same way it already drops masked pixels:
`valid &= np.isfinite(var_vals) & (var_vals > 0.0)`. The consequence for every downstream
ring statistic — intensity, gradient, and their uncertainties — is that the value and its
error bar are always built from the identical set of samples.

**Checking the value alone is not sufficient for a raw map.** Variances are sampled with
bilinear interpolation, which blends each sample from the four surrounding image pixels.
Non-finite entries propagate through that blend on their own, so a `NaN` or infinity always
reaches the value check. Zero and negative entries do not: an isolated zero pixel averaged
with three positive neighbours yields a positive interpolated variance that the value check
accepts, leaving an unusable sample in the ring with an understated variance. Measured on a
flat test map, one zero-variance source pixel produced an interpolated variance of 1.95
against a true 4.0, and one negative pixel produced 1.44.

`sampling._bilinear_support_is_valid` therefore also checks the four source pixels feeding
each sample, and `extract_isophote_data` applies it by default. Its cost is proportional to
the number of samples in a ring rather than to the image size, so it stays cheap on large
images.

`fit_image` has already replaced every unusable entry with `NaN` before any sampling
happens, so for it the check is redundant. It passes `variance_map_prepared=True` down
through `fit_isophote` and `compute_gradient` to skip it: instrumenting a 62-isophote fit of
a 1133×1133 image confirms the check runs **zero** times inside `fit_image`, so the normal
pipeline carries no extra cost at all. The flag defaults to `False`, so the lower-level
functions — which are public through `isoster/optimize.py` — remain correct when a caller
passes a raw, unvalidated variance map straight to them. Setting it to `True` is a promise
that unusable entries are already non-finite; it is an optimisation, not a behaviour switch.
On a direct call the check costs at most about 0.11 ms per ring (3141 samples on a 1133×1133
image).

**Multi-band follows the same rule**, through
`multiband.sampling_mb._mark_invalid_variance`. There is one structural difference worth
knowing: every multi-band entry point that accepts a raw variance map routes through
`_resolve_variance_maps`, so the stacks handed to the sampler are *always* pre-marked. That
makes the source-pixel check unnecessary on that side — there is no equivalent of
`variance_map_prepared`, because there is no unprepared path. A second difference is the
interpolation kernel: multi-band uses a numba kernel that, at an exact final row or column,
collapses onto the last cell rather than shifting to the last two the way SciPy does. The
two agree for all finite data and differ only for a `NaN` sitting in the penultimate cell of
a sample landing exactly on the final index — a corner that no real ellipse path reaches.

Two earlier mechanisms handled the same input problem differently, and both have been
retired:

- A `VARIANCE_SENTINEL = 1e30` substitution for non-finite entries, meant to give the
  affected pixel a near-zero weight in a weighted fit.
- A clamp of non-positive entries to `1e-30`, meant to give the affected pixel a bounded
  (very large but finite) weight instead of a division by zero.

Both kept the flagged pixel *in* the sample set while distorting how much it counted,
rather than removing it. The clamp was the more damaging of the two: because a ring
gradient's uncertainty combines the per-sample variances of two rings, a single `1e-30`
entry could shrink the combined gradient error by roughly fifteen orders of magnitude —
making it disappear numerically. A spuriously tiny error is not a cosmetic problem, because
three downstream checks trust it directly: the `maxgerr` gradient-quality gate, the
signal-to-noise damping applied to the geometry update, and the low-surface-brightness (LSB)
auto-lock trigger. None of the three can trip correctly when the error they compare against
is many orders of magnitude too small.

A real-data demonstration on a DECaLS cutout of `2MASXJ12504800+4231220`, with a block of
pixels deliberately set to zero variance to simulate a defect, shows the effect concretely:
the retired clamp collapsed `grad_error` by roughly 12-13 orders of magnitude on the
affected rings, while the corrected code instead drops samples from the affected rings (a
one-off count of 89 samples across 5 rings was measured during development; this specific
count is not reproduced by the committed demo script and should be read as illustrative,
not as a pinned regression value). At one semi-major axis on that image (144.21 pixels),
the retired code accepted the isophote (stop code 0); the corrected code correctly flags it
as a gradient-quality failure (stop code -1) — a rejection that the old, artificially tiny
error had been silently suppressing. Runtime effect of the fix, measured on the same
demonstration: +0.01% for pure OLS, +0.79% for a clean WLS run, and -11.76% (faster) for the
WLS run carrying the injected defect.

### Gradient error and ring statistics

The radial gradient reported for an isophote is the difference between two "ring
statistics" — a representative intensity for the current radius and for a radius one step
further out — divided by the radial step. The gradient's uncertainty is built from the
variance of those same two ring statistics, computed by one shared helper,
`_ring_statistic_and_variance` in `isoster/_shared.py`, so the reported value and its error
always come from the same estimator on the same samples:

- **Unweighted mean** (the default `integrator='mean'`, and the branch used for
  `'adaptive'` when not otherwise resolved): variance is `sum(v_i) / N**2`, where `v_i` are
  the per-sample variances and `N` the sample count — the ordinary variance of a mean of
  independent, non-identically-distributed measurements.
- **Unweighted median** (`integrator='median'`): variance is
  `pi * N / (2 * (sum(1/sqrt(v_i)))**2)`, a normal-theory asymptotic result. For uniform
  variance (all `v_i = sigma**2`) this reduces to the familiar `pi * sigma**2 / (2*N)` — the
  ordinary mean-of-`N` variance `sigma**2/N`, inflated by the `pi/2 ≈ 1.571` factor that
  makes the median a less efficient (noisier) estimator than the mean under Gaussian noise.

The median formula is an **approximation**, not an exact result: its normal-theory
derivation assumes independent, identically distributed Gaussian samples, but a real
isophote ring carries genuine azimuthal structure (the galaxy's light is not uniform around
the ring) and neighbouring samples are correlated, because bilinear interpolation
(`scipy.ndimage.map_coordinates`) blends each sample from nearby pixels. Monte Carlo tests
against both formulas showed agreement within about 0.08% for uniform per-sample variance
and 1.36% for heteroscedastic (spatially varying) variance — close enough to trust as an
error estimate, but not exact in the way a from-first-principles derivation on independent
Gaussian samples would be.

When no variance map is supplied, the same helper falls back to a scatter-based error
computed directly from the sampled intensities (`np.std(intens)**2 / N` for the mean, with
the same `pi/2` penalty for the median). This scatter reflects everything that varies
around the ring — photon noise, but also any real azimuthal structure in the galaxy's light
at that radius — so it should be read as an **upper bound** on the noise, not a pure noise
estimate.

**A deliberate asymmetry.** Under WLS (a variance map supplied), the ring **intensity**
reported for `integrator='mean'` is *not* the unweighted mean described above: it is the
inverse-variance-weighted intercept of the harmonic fit (the `y0_fit` term), which gives
more influence to lower-variance (more trustworthy) samples — see the WLS mode bullet
above. The ring **gradient**, by contrast, always uses the unweighted mean's location and
variance, through `_ring_statistic_and_variance`, and never calls the inverse-variance-
weighted helper (`_weighted_mean_variance`, which remains in `isoster/_shared.py` and is
still used for the reported intensity's own error under `integrator='mean'`, and by
`isoster/multiband/`). These are different quantities computed for different purposes: the
intensity is a per-isophote photometric measurement, where down-weighting noisy pixels is
desirable, while the gradient is a diagnostic comparison between two rings, where matching
the same, simple estimator on both sides keeps the comparison honest. The difference is
intentional and should be stated explicitly rather than left implicit.

The asymmetry applies to `integrator='mean'` only. Under `integrator='median'` the reported
intensity is an **unweighted median** even when a variance map is supplied, so `intens_err`
comes from that median's own variance via `_ring_statistic_and_variance`, not from the
weighted intercept. Rescaling the weighted intercept's error by `sqrt(pi/2)` — the factor
that converts a mean's error to a median's — is correct only when the ring's variances are
uniform. Once they vary with angle, the intercept of a five-parameter fit becomes correlated
with the sin/cos terms, and its variance is no longer the median's by any constant factor.
Measured across several spatial variance patterns, the rescaled value ranged from 0.75x to
3.21x the correct median error, so the mismatch can understate as well as overstate the
uncertainty; it is exactly 1.00x only for uniform variance.

**A consequence for the reported gradient value itself.** The per-ring location formulas
are byte-identical before and after this fix, so one might expect only the error bars to
change. That is not quite true: `gradient_error` also gates two control-flow decisions in
`fitting.compute_gradient` that select *which* ring's gradient gets reported — the
`need_second_gradient` decision (`isoster/fitting.py:908`) and the final baseline-selection
override (`isoster/fitting.py:940`), both of which compare a candidate gradient, or its
relative error, against a threshold. A corrected — typically larger — error can therefore
push a borderline case across one of these thresholds and change which baseline (one-step
or two-step) is reported, and hence the numeric gradient value itself, not only its
uncertainty. This was measured directly on the regression fixture in
`tests/unit/test_gradient_error.py::test_corrected_error_can_change_which_baseline_is_selected`:
the old code reported `(gradient, error) = (-2.56, None)`; the corrected code reports
`(-1.185, 0.2805)`. This is the fix working as intended — the gradient error is the first
thing this downstream logic consumes — but it means fitted gradients on borderline
isophotes with heteroscedastic variance maps can shift as a result of this change, not only
their reported uncertainty.

## Stop Codes (Implemented Semantics)

Stop codes currently emitted by core `isoster` fitting paths are:

- `0`: converged / successful forced extraction.
- `1`: too many flagged/clipped samples (`actual_points < total_points * (1.0 - fflag)`).
- `2`: reached `maxit` without convergence; best-so-far geometry fallback.
- `3`: too few points (`< 6`) for first/second harmonic fit.
- `-1`: gradient-related failure.

Canonical user-facing stop-code documentation lives in `docs/01-user-guide.md`.

## Output Contract

`fit_image` returns:

- `results['isophotes']`: list of dict rows, one per sampled/fitted SMA.
- `results['config']`: the resolved `IsosterConfig` object.

### FITS Output Layout

`isophote_results_to_fits` writes a 4-HDU FITS file:

| HDU | Type | Name | Contents |
|-----|------|------|----------|
| 0 | `PrimaryHDU` | — | Empty (no data, minimal header) |
| 1 | `BinTableHDU` | `ISOPHOTES` | One row per isophote; columns match the isophote dict keys |
| 2 | `BinTableHDU` | `CONFIG` | Two columns: `PARAM` (string) and `VALUE` (JSON-serialized string), one row per config field |
| 3 | `BinTableHDU` | `META` | Same `PARAM`/`VALUE` JSON layout, carrying the remaining top-level result keys (`lsb_auto_lock*`, `first_isophote_*`, outer-regularization references, …) |

This replaces the previous approach of writing config fields as FITS header keywords, which triggered `HIERARCH` warnings for long keyword names.

Backward compatibility: `isophote_results_from_fits` detects whether an `ISOPHOTES` extension is present. Legacy single-table files (config in header keywords) are still readable; the `CONFIG` HDU is simply absent and the reader returns `config=None` — no header-keyword reconstruction is performed.

Each isophote row includes geometry/intensity fields and optional blocks depending on config. See `docs/01-user-guide.md` (Output Reference) for the complete per-field reference. Summary of optional blocks:

- Harmonic deviations (`compute_deviations` or `simultaneous_harmonics`): `a{n}`, `b{n}`, `a{n}_err`, `b{n}_err` for requested `harmonic_orders`.
- Full aperture photometry (`full_photometry` or `debug`): `tflux_e`, `tflux_c`, `npix_e`, `npix_c`.
- CoG (`compute_cog` in regular mode): `cog`, `cog_annulus`, `area_annulus`, `flag_cross`, `flag_negative_area`.
- Debug diagnostics (`debug`): `ndata`, `nflag`, `grad`, `grad_error`, `grad_r_error`.
- Automatic LSB geometry lock (`lsb_auto_lock`): per-isophote `lsb_locked` (bool) and a single `lsb_auto_lock_anchor=True` marker on the first locked isophote (`False` everywhere else, including inward isophotes and the central pixel, so the schema is uniform). Top-level result dict also gains `lsb_auto_lock`, `lsb_auto_lock_sma`, `lsb_auto_lock_anchor_sma` (the true geometry anchor, which the marker does not identify), and `lsb_auto_lock_count`.
- Outer region regularization (`use_outer_center_regularization`): the top-level result dict gains `use_outer_center_regularization` (echo), `outer_reg_x0_ref`, `outer_reg_y0_ref`, `outer_reg_eps_ref`, and `outer_reg_pa_ref` carrying the frozen inner reference geometry. No per-isophote fields are added.

## Known Behavior Notes

- `template` takes precedence over regular fitting.
- `compute_cog` is only run in regular mode in current `fit_image`; forced/template branches return before CoG attachment.
- Regular mode passes `previous_geometry` during outward/inward growth, so central regularization can apply when enabled.
- Inward growth starts only when the first fitted isophote has an acceptable stop code (`0`, `1`, or `2`).
- **Inward-first loop order**: the regular-mode driver unconditionally runs the inward loop *before* the outward loop. The inward loop's outputs are unchanged — only the execution order is swapped. This is a precondition for the outer-region center regularization feature (building a stable inner reference centroid before the outward loop starts), but it is always active regardless of whether that feature is on. Consumers that iterated over `results['isophotes']` by index (rather than by `sma`) are unaffected because the result list is still assembled in sma-sorted order.
- Automatic LSB geometry lock (`lsb_auto_lock=True`): the outward growth loop maintains a one-way state machine (free → locked). The detector inspects `grad`/`grad_error`/`grad_r_error` on each new outward isophote, debounced by `lsb_auto_lock_debounce`. On commit, the driver clones the config with `fix_center=fix_pa=fix_eps=True`, `integrator=lsb_auto_lock_integrator`, geometry carried from the isophote *before* the trigger streak, and continues outward growth with the locked clone. Inward growth always uses the original free config. `debug=True` is auto-enabled internally when the caller leaves it off (with a `UserWarning`). The lock is only wired into the regular-mode driver — template-based forced photometry emits a `UserWarning` and the feature is silently inactive. It is agnostic to `use_eccentric_anomaly`, `simultaneous_harmonics`, and isofit-style modes because the detector reads only per-isophote gradient diagnostics.
- Outer region regularization (`use_outer_center_regularization=True`): after the inward loop, the driver calls `_build_outer_reference(inwards_results, anchor_iso, cfg)` to compute a flux-weighted mean over the anchor plus qualifying inward isophotes (acceptable stop codes, `sma <= sma0 * outer_reg_ref_sma_factor`). The result `(x0_ref, y0_ref, eps_ref, pa_ref)` is carried as a separate `outer_reference_geom` kwarg into `fit_isophote` for outward calls. Inside the fitting loop, the logistic ramp `lambda(sma) = outer_reg_strength / (1 + exp(-(sma - onset) / width))` drives per-axis Tikhonov damping according to `outer_reg_weights`; default `outer_reg_mode='damping'` shrinks harmonic geometry steps, while `outer_reg_mode='solver'` also pulls toward the reference. A selector-level penalty from `compute_outer_center_regularization_penalty` still contributes to `effective_amp`. The feature composes cleanly with the automatic LSB lock: after the lock, `fix_center=True`, `fix_pa=True`, and `fix_eps=True` make the corresponding weights inert. Inward growth never gets the penalty (`outer_reference_geom=None` for inward calls). The feature is only wired into the regular-mode driver — template-based forced photometry emits a `UserWarning` and the feature is silently inactive.

## Huang2013 Campaign Workflow

The external Huang2013 mock-comparison workflow under `examples/example_huang2013/` is a two-stage pipeline:

1. Profile extraction (`run_huang2013_profile_extraction.py`) with per-method status captured in `*_profiles_manifest.json` (success/failed without aborting the case).
2. QA afterburner (`run_huang2013_qa_afterburner.py`) that tolerates missing method products and emits `*_qa_manifest.json` with skip/failure metadata.

Extraction retries are method-local and shared across photutils/isoster with fixed policy:

- Maximum attempts: 5
- `sma0` increment: +2.0 pixels per retry attempt
- `astep` increment: +0.02 per retry attempt
- `maxsma`: multiplied by `0.95` per retry attempt (5% decay each attempt)
- Retry metadata (`fit_retry_log`, `attempt_count`, `max_attempts`) is persisted in per-method run JSON.

For full-sample execution, `run_huang2013_campaign.py` iterates galaxies/mock IDs, continues across case failures, and writes campaign-level summary JSON/Markdown with aggregate method failure counts plus explicit failed/timeout case labels (per method and QA stage).

Campaign controls include verbose stage telemetry (`--verbose`), per-stage logs (`--save-log`), per-stage timeout guard (`--max-runtime-seconds`, default 900), resume pointers (`--continue-from`, `--continue-from-case`), and skip-existing/rerun control (`--update`).

QA afterburner uses extraction-manifest method status as a guard and skips method QA/comparison for methods with non-success extraction status.

For QA/model reconstruction robustness, isoster 2-D model building in the Huang2013 workflow sanitizes profile rows before calling `isoster.build_isoster_model(...)`: rows with non-finite required fields are filtered, duplicate SMA rows are de-duplicated, and any residual non-finite model pixels are replaced with `0.0` with an explicit warning.

Default output layout is case-scoped:

- input FITS: `<huang-root>/<GALAXY>/<GALAXY>_mock<ID>.fits`
- generated artifacts: `<huang-root>/<GALAXY>/mock<ID>/...`

### Reorganization Boundaries (Planned, Compatibility-Preserving)

Target module boundaries for the campaign reorganization are:

1. CLI wrappers (stable user entry points):
   - `run_huang2013_campaign.py`
   - `run_huang2013_profile_extraction.py`
   - `run_huang2013_qa_afterburner.py`
2. Shared workflow contract module:
   - `huang2013_campaign_contract.py` for canonical case prefix, artifact/manifest path conventions, and manifest status parsing.
3. Future orchestration modules (next slices):
   - campaign case planner/executor and stage command builder split out of the CLI wrapper.

This boundary keeps extraction and QA behavior unchanged while reducing naming/path drift between stages.

### Manifest Compatibility Contract

Manifest compatibility is preserved with additive-only schema evolution:

- Filenames remain unchanged:
  - extraction: `*_profiles_manifest.json`
  - QA: `*_qa_manifest.json` (plus existing tag suffix variant)
- Existing stable fields remain unchanged:
  - extraction: `method_runs`, `run_summary`, `warnings`
  - QA: `method_outputs`, `method_skips`, `method_failures`, `comparison_qa`, `warnings`, `run_metadata`
- Cross-stage status contract remains:
  - QA/campaign decisions read `method_runs.<method>.status` from extraction manifest.
- New fields may be added, but existing field names/types above must remain backward compatible.

## Verification and Artifacts

- Tests: `tests/`
- Benchmarks/profiling: `benchmarks/`
- Reproducible examples: `examples/`
- Generated artifacts: `outputs/`

### Cross-tool harmonic scale (`benchmarks/harmonic_scale/`)

Settles whether isoster, photutils and AutoProf put `a_n` and `b_n` on the
same scale, which is what lets a cross-tool harmonic comparison be published.
Five modules, each with one job:

| Module | Responsibility |
|---|---|
| `conventions.py` | The one place that knows what each tool means by `a_n` and `b_n`, and the four ways AutoProf differs |
| `adapters.py` | One fixed-aperture measurement per tool, on identical imposed rings, with the returned geometry verified rather than trusted |
| `autoprof_worker.py` | Runs inside the AutoProf venv (it pins `numpy<2`), drives the forced pipeline, and instruments both sampling modes |
| `run_harmonic_scale.py` | The designed grid, the pilot/validation split, tolerance freezing, and the archive gate |
| `claims.py` | Reduces a run to the numbers Part A stands behind — shared by the freeze and the check so the two definitions cannot drift |

Two design points that are easy to undo by accident:

- **`ap_iso_interpolate_start` is a grid axis, not a setting.** It selects
  Lanczos sampling below a radius threshold and nearest-pixel rounding above
  it, and it is the largest effect in the study by an order of magnitude. The
  threshold is `ap_iso_interpolate_start * results["psf fwhm"]`, and on the
  forced pipeline that PSF is **not measured** — the `psf` step is
  `PSF_Assumed`, which hardcodes 4.0 px. The worker observes AutoProf's own
  branch anyway, by watching whether the interpolator ran, because the
  per-ring mode is the measurand and the PSF step is swappable.
- **Raw amplitudes are the primary track; Bender is secondary and currently
  unlicensed for AutoProf.** The raw reconstruction is exact from the native
  pair and `b0`, and needs no gradient. Bender normalization needs one that
  AutoProf does not report, so those columns are NaN with a stated reason
  rather than filled. See `docs/09-exhausted-benchmark.md` for the profile
  schema and its version-2 migration note.

Two campaigns, one per galaxy, each additive and independently frozen:
`sersic_n2_compact` (n=2, R_e=25, 241 px) and `sersic_n4_extended` (n=4,
R_e=40, 321 px). Each has its own archive, its own `frozen_tolerances*.json`
and its own noise-seed blocks; `FIXTURES` in `run_harmonic_scale.py` is the
registry, and `--fixture` selects one. The first campaign's `extra_cases` is
empty on purpose — that is what keeps its grid, and therefore the fingerprint
its committed archive was gated under, from moving.

The second campaign adds an `ap_set_psf` axis. Since AutoProf's `psf` step
assumes 4.0 px, that option is the only way to move the interpolation switch
independently of `ap_iso_interpolate_start`, which makes two separate knobs
onto one mechanism: at a matched threshold the two routes agree exactly.

Both archives are gated in docs CI by `check_harmonic_scale.py`; see
`docs/05-testing.md` for the rules on changing either.

### Controlled three-way timing (`benchmarks/timing/`)

The Part B timing study keeps scientific accuracy and timing eligibility as
separate records. `run_stage2_calibration.py` supplies the shared per-session
measurement path for isoster, photutils and the persistent AutoProf worker.
`frozen_stage3_parameters.json` binds the measured per-arm batch sizes, three
sessions, 25 repetitions and the independent campaign seed block with a
fingerprint checked by `stage3_parameters.py`.

`run_stage4_campaign.py` is the full-campaign entry point. It performs the idle
preflight, starts fresh monitored session interpreters, records fit-only and
fit-plus-harness time, and writes a new archive without replacing the
established two-way benchmark. A record is timing-eligible only when execution,
coverage and thermal/process conditions pass. Frozen scientific accuracy
verdicts remain descriptive and are never rewritten to make a timing eligible.
AutoProf's fixed-aperture, harmonics-off mean intensity is explicitly
unavailable because AutoProf emits `b0` only when coefficient extraction is
enabled; its median `I` column is not substituted.

## Documentation Policy

- Stable docs live in `docs/` root.
- Internal planning and review notes are kept under `docs/agent/`, which is untracked and excluded from the published site; retired dated reports are tracked under `docs/archive/` but also excluded from the site. References to either in these pages are pointers for developers working in a checkout, not links a site reader can follow.
- Use lowercase kebab-case markdown filenames.
