# Configuration Reference

## Overview

`IsosterConfig` is a Pydantic model that defines all tunable parameters for the isophote
fitting algorithm. It contains 62 parameters organized into the 14 sections below, covering
geometry initialization, fitting control, quality control, output options, and advanced
algorithm modes.

Source: `isoster/config.py` (`IsosterConfig`)

Pydantic validation enforces type constraints, value ranges, and cross-parameter
consistency (e.g., `minit <= maxit`, `integrator='adaptive'` requires `lsb_sma_threshold`,
all `harmonic_orders >= 3`).

---

## Units and Conventions

Quantities in result dicts and per-isophote rows use the following units throughout:

| Quantity | Units | Notes |
|----------|-------|-------|
| `sma`, `minsma`, `maxsma`, `sma0` | pixels | Semi-major axis lengths. |
| `x0`, `y0` (and `x0_err`, `y0_err`) | pixels | Image center coordinates. |
| `pa` (and `pa_err`) | radians | `0 <= pa < pi`; QA figures display PA in degrees. |
| `eps` (and `eps_err`) | dimensionless | `0 <= eps < 1`. |
| `intens`, `intens_err`, `rms` | image flux units | Same units as the input image pixels. |
| `a{n}`, `b{n}` (and errors) | dimensionless | Bender-style fractional-radius perturbations: raw harmonic amplitudes divided by `sma * \|gradient\|` (see the User Guide output reference for the sign convention). |
| `grad`, `grad_error` | flux per pixel | Radial intensity gradient `dI/da`; `grad_r_error` is dimensionless. |
| `tflux_e`, `tflux_c`, `cog`, `cog_annulus` | image flux units | Integrated fluxes. |
| `npix_e`, `npix_c`, `ndata`, `nflag` | counts | Pixel/point counts. |

---

## Parameter Reference

### 1. Geometry Initialization

Initial ellipse geometry. When `x0` or `y0` is `None`, the image center is used.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `x0` | `None` | `Optional[float]` | Initial center x coordinate. `None` uses `image.shape[1] / 2`. |
| `y0` | `None` | `Optional[float]` | Initial center y coordinate. `None` uses `image.shape[0] / 2`. |
| `eps` | `0.2` | `float` (0 <= eps < 1) | Initial ellipticity. |
| `pa` | `0.0` | `float` | Initial position angle in radians. |

**Interaction notes:**

- These values seed the first isophote at `sma0`. Subsequent isophotes inherit geometry
  from the previous converged fit.
- Template-based forced photometry uses geometry from the template, overriding these.
- When `fix_center=True`, `x0` and `y0` remain constant throughout the run.
  Similarly for `fix_pa` and `fix_eps`.

---

### 2. SMA Control

Semi-major axis stepping and bounds.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `sma0` | `10.0` | `float` (> 0) | Starting semi-major axis length in pixels. |
| `minsma` | `0.0` | `float` (>= 0) | Minimum SMA to fit. Set > 0 to skip central region. |
| `maxsma` | `None` | `Optional[float]` (> 0) | Maximum SMA. `None` uses `max(image_height, image_width) / 2`. |
| `astep` | `0.1` | `float` (> 0) | Step size for SMA growth. |
| `linear_growth` | `False` | `bool` | If `True`, `sma_next = sma + astep`. If `False`, `sma_next = sma * (1 + astep)`. |

**Interaction notes:**

- The driver fits at `sma0` first, then grows inward from `sma0` to
  `max(minsma, 0.5)`, then outward to `maxsma`.
- When `minsma <= 0.0`, a central pixel (SMA=0) result is prepended automatically.
- Geometric growth (default) gives denser sampling near the center and sparser sampling
  at large radii. Linear growth gives uniform spacing.
- Validated: `maxsma > minsma` when both are specified.

---

### 3. Fitting Control

Iteration limits and convergence criteria.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `maxit` | `50` | `int` (> 0) | Maximum iterations per isophote. |
| `minit` | `6` | `int` (> 0) | Minimum iterations before convergence check. |
| `conver` | `0.05` | `float` (> 0) | Convergence threshold: `max_harmonic_amplitude / rms`. |
| `convergence_scaling` | `'sector_area'` | `str` | Scale convergence threshold with SMA. Options: `'none'`, `'sector_area'`, `'sqrt_sma'`. |
| `sigma_bg` | `None` | `Optional[float]` (> 0) | Explicit background noise level (sigma). |
| `use_lazy_gradient` | `True` | `bool` | Enable Lazy Gradient Evaluation (Modified Newton Method). |

**Interaction notes:**

- Convergence is declared when the largest harmonic amplitude drops below
  `conver * rms * scale_factor`, where `scale_factor` depends on `convergence_scaling`.
- When `sigma_bg` is provided, it establishes a hard lower bound on the convergence
  threshold: `max(rms, sigma_bg / sqrt(N))`. This prevents the solver from chasing
  vanishing asymmetries in the noise floor where `rms` might naturally dip below
  the photon noise limit.
- `use_corrected_errors=True` (default) includes the radial gradient uncertainty term
  in the geometric error formulas. This correctly inflates error bars in LSB regions
  where gradient measurement is uncertain, preventing the reported "false confidence"
  of standard residual-based methods.
- `'sector_area'` (default) matches photutils behavior and eliminates most stop=2
  failures at outer isophotes. The scale factor grows with SMA.
- `'none'` uses a constant threshold (legacy behavior).
- `'sqrt_sma'` multiplies by `sqrt(sma)`.
- `use_lazy_gradient=True` (default) calculates the radial gradient only on the first
  iteration and reuses it for subsequent iterations. This cuts sampling calls by ~45%.
  The gradient is re-evaluated if convergence stalls for 3 iterations.
- `minit` iterations always run before the convergence criterion is checked.
- Validated: `minit <= maxit`.

---

### 4. Geometry Update Control

Controls how geometry corrections are applied during iteration.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `geometry_damping` | `0.7` | `float` (0 < d <= 1) | Damping factor for geometry corrections. Each correction is multiplied by this value. |
| `geometry_update_mode` | `'largest'` | `str` | `'largest'`: update only the parameter with the largest harmonic amplitude (coordinate descent). `'simultaneous'`: update all four parameters each iteration. |
| `clip_max_shift` | `5.0` | `Optional[float]` (> 0) | Maximum allowed center shift (pixels) per iteration. `None` to disable. |
| `clip_max_pa` | `0.5` | `Optional[float]` (> 0) | Maximum allowed PA change (radians) per iteration. `None` to disable. |
| `clip_max_eps` | `0.1` | `Optional[float]` (> 0) | Maximum allowed ellipticity change per iteration. `None` to disable. |
| `geometry_convergence` | `False` | `bool` | Enable secondary convergence based on geometry stability. |
| `geometry_tolerance` | `0.01` | `float` (> 0) | Threshold for geometry convergence metric. |
| `geometry_stable_iters` | `3` | `int` (>= 2) | Consecutive stable iterations required for geometry convergence. |
| `permissive_geometry` | `False` | `bool` | Enable photutils-style "best effort" geometry updates that continue even from failed fits. |

**Interaction notes:**

- `geometry_damping=0.7` is validated across 20 Huang2013 galaxies. Use `0.5` with
  `geometry_update_mode='simultaneous'` for stability. Use `1.0` for undamped legacy
  behavior.
- **Gradient SNR Damping**: `isoster` dynamically reduces `geometry_damping` in low surface brightness regions when the radial gradient is noisy (SNR < 3). This stabilizes the solver in the outer disk without affecting performance in high-S/N regions.
- **Step Clipping**: `clip_max_shift`, `clip_max_pa`, and `clip_max_eps` provide hard limits on per-iteration geometry changes. This prevents noise-induced "catastrophic jumps" that poison the geometry for all subsequent SMAs. These are safeguards and default values are relaxed enough to not interfere with real structural variations.
- `'largest'` mode (default) matches isofit/photutils coordinate-descent behavior.
  `'simultaneous'` updates x0, y0, PA, and eps every iteration, typically converging
  in fewer iterations but requiring lower damping.
- Geometry convergence declares success when
  `max(|delta_eps|, |delta_pa/pi|, |delta_x0/sma|, |delta_y0/sma|) < geometry_tolerance`
  for `geometry_stable_iters` consecutive iterations, even if the harmonic criterion is
  not met. This is a supplementary criterion useful for challenging outer isophotes.
- `permissive_geometry=True` always updates geometry from the latest fit, even when the
  fit failed. This prevents cascading failures where one bad isophote poisons all
  subsequent geometry propagation. `None` gradient errors are treated as acceptable.
- Both `geometry_update_mode` options respect `fix_center`, `fix_pa`, and `fix_eps`.

---

### 5. Quality Control

Sigma clipping and data quality thresholds.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `sclip` | `3.0` | `float` (> 0) | Symmetric sigma clipping threshold. |
| `nclip` | `1` | `int` (>= 0) | Number of sigma clipping iterations. `0` disables clipping. |
| `sclip_low` | `None` | `Optional[float]` | Lower (negative) sigma clipping threshold. Overrides `sclip` on the low side when set. |
| `sclip_high` | `None` | `Optional[float]` | Upper (positive) sigma clipping threshold. Overrides `sclip` on the high side when set. |
| `fflag` | `0.5` | `float` (0-1) | Maximum fraction of flagged data points (masked + sigma-clipped). Exceeding this triggers stop_code=1. |
| `maxgerr` | `0.5` | `float` (> 0) | Maximum relative error in the local radial gradient. Exceeding this triggers stop_code=-1. |

**Interaction notes:**

- `fflag` only counts sigma-clipped points as flagged. Mask-excluded points are removed
  during sampling before the harmonic fit.
- For high-ellipticity galaxies (eps > 0.6), relax `maxgerr` to 1.0-1.2 to prevent
  excessive gradient-error failures.
- `sclip_low` and `sclip_high` enable asymmetric clipping, useful when bright sources
  contaminate one tail of the intensity distribution.
- Sigma clipping is applied every iteration within `fit_isophote` and once in forced
  photometry extraction.

---

### 6. Constraints

Fix individual geometry parameters during iterative fitting.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `fix_center` | `False` | `bool` | Fix center coordinates (x0, y0) during fitting. |
| `fix_pa` | `False` | `bool` | Fix position angle during fitting. |
| `fix_eps` | `False` | `bool` | Fix ellipticity during fitting. |

**Interaction notes:**

- Fixed parameters retain their initial values (from `x0`/`y0`/`eps`/`pa` config or
  inherited from the previous isophote) throughout all iterations.
- When all three are `True`, fitting becomes pure photometry extraction with
  predetermined geometry (still iterates for convergence unless using `template`).
- Constraints are respected by both `'largest'` and `'simultaneous'`
  `geometry_update_mode`.
- When `compute_cog=True`, fixing all three affects crossing detection logic in the
  curve-of-growth module.

---

### 7. Eccentric Anomaly

Sampling mode for high-ellipticity isophotes.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `use_eccentric_anomaly` | `False` | `bool` | Use eccentric anomaly (psi) for uniform arc-length ellipse sampling. |

**Interaction notes:**

- When `True`, sampling is uniform in eccentric anomaly psi, which provides uniform
  arc-length coverage on the ellipse. Recommended for eps > 0.3.
- Harmonics are fitted in psi-space; geometry updates are always performed in phi-space
  (position angle) to maintain correct geometric interpretation.
- Affects both regular fitting and template-based forced photometry extraction.
- `build_isoster_model()` auto-detects whether eccentric anomaly was used from the
  isophote dicts (presence of `'use_eccentric_anomaly'` key). Can be overridden
  explicitly.

---

### 8. Higher-Order Harmonics / ISOFIT

Controls for simultaneous harmonic fitting following Ciambur (2015).

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `simultaneous_harmonics` | `False` | `bool` | Enable true ISOFIT simultaneous harmonic fitting. |
| `harmonic_orders` | `[3, 4]` | `List[int]` | Harmonic orders to fit. All values must be >= 3. |
| `isofit_mode` | `'in_loop'` | `str` | ISOFIT algorithm variant: `'in_loop'` or `'original'`. |

**Interaction notes:**

- When `simultaneous_harmonics=False` (default), higher-order harmonics (a3, b3, a4, b4,
  etc.) are fitted post-hoc after geometry convergence. Orders 1-2 are always used
  internally for geometry updates.
- When `simultaneous_harmonics=True`, higher-order harmonics are fitted jointly with
  geometry harmonics inside the iteration loop. This accounts for cross-correlations and
  yields cleaner RMS estimates.
- `isofit_mode='in_loop'` fits all orders simultaneously inside the loop (more
  aggressive). `isofit_mode='original'` matches Ciambur (2015): 5-parameter fit inside
  the loop for geometry, then simultaneous post-hoc fit for higher orders. `isofit_mode`
  is only meaningful when `simultaneous_harmonics=True`.
- Falls back to 5-parameter fit when the number of sample points is less than
  `2 * len(harmonic_orders) + 5` (the minimum for the extended design matrix).
  A `RuntimeWarning` is emitted on first fallback per isophote.
- ISOFIT coefficients are stored as `[I_0, A_1, B_1, A_2, B_2, A_n1, B_n1, ...]` where
  the k-th higher order occupies indices `5+2k` and `5+2k+1`.
- Validated: all entries in `harmonic_orders` must be >= 3.
- ISOFIT overhead is approximately 25-35% compared to the default path.

---

### 9. Central Regularization

Geometry regularization for low-S/N central regions.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `use_central_regularization` | `False` | `bool` | Enable geometry regularization at low SMA. |
| `central_reg_sma_threshold` | `5.0` | `float` (> 0) | SMA threshold in pixels. Regularization strength decays as `exp(-(sma/threshold)^2)`. |
| `central_reg_strength` | `1.0` | `float` (>= 0) | Maximum regularization strength at SMA=0. Range: 0 (none) to 10 (strong). |
| `central_reg_weights` | `{'eps': 1.0, 'pa': 1.0, 'center': 1.0}` | `dict` | Relative weights for regularization penalties on ellipticity, PA, and center. |

**Interaction notes:**

- Regularization penalizes large geometry changes relative to the previous isophote,
  stabilizing fits in noisy central regions.
- The penalty decays to effectively zero beyond approximately 3x the threshold SMA.
- Per-parameter weights allow selective regularization (e.g., fix center strongly but
  allow PA to vary by setting `{'eps': 1.0, 'pa': 0.1, 'center': 5.0}`).
- Requires a previous isophote for comparison; has no effect on the first isophote
  at `sma0`.

---

### 10. Output and Features

Toggle optional output quantities.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `compute_errors` | `True` | `bool` | Calculate parameter errors (x0_err, y0_err, eps_err, pa_err). |
| `use_corrected_errors` | `True` | `bool` | Include gradient uncertainty term in error propagation. |
| `compute_deviations` | `True` | `bool` | Calculate higher-order harmonic deviations (a3, b3, a4, b4, etc.) post-hoc. |
| `full_photometry` | `False` | `bool` | Calculate aperture flux integration metrics (tflux_e, tflux_c, npix_e, npix_c). |
| `compute_cog` | `False` | `bool` | Calculate curve-of-growth photometry (cog, cog_annulus, area_annulus, flag_cross, flag_negative_area). |
| `debug` | `False` | `bool` | Include debug fields (ndata, nflag, grad, grad_error, grad_r_error) and enable verbose output. |

**Interaction notes:**

- `compute_deviations` controls the post-hoc harmonic deviation calculation. This is
  independent of `simultaneous_harmonics`, which controls joint fitting inside the
  iteration loop.
- When `simultaneous_harmonics=True`, deviations for the specified `harmonic_orders` are
  computed as part of the joint fit. `compute_deviations=True` additionally computes
  deviations post-hoc (which may differ slightly from the simultaneous values).
- See the "Photometry Outputs" section below for details on `full_photometry` vs
  `compute_cog`.

---

### 11. Integration Mode

Controls how the representative intensity is derived from sampled points.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `integrator` | `'mean'` | `str` | Integration method: `'mean'`, `'median'`, or `'adaptive'`. |
| `lsb_sma_threshold` | `None` | `Optional[float]` (> 0) | SMA threshold in pixels for switching to median in adaptive mode. |

**Interaction notes:**

- `'mean'` reports the harmonic-fit DC term (the fitted mean `y0_fit`, not `np.mean`).
- `'median'` reports `np.median(intens)` of the sigma-clipped samples.
- `'adaptive'` uses `'mean'` for SMA <= threshold, `'median'` for SMA > threshold.
  Inner high-S/N regions benefit from harmonic fit precision; outer low-S/N regions
  benefit from median robustness.
- Validated: `lsb_sma_threshold` must be provided when `integrator='adaptive'`.
- See the "Integration Modes" section below for architectural details.

---

### 12. Automatic LSB Geometry Lock

Single-pass free-then-locked outward growth. Starts in free mode, commits a lock to the last clean isophote once the gradient diagnostics cross the LSB thresholds, then continues outward with fixed center/eps/PA and a robust integrator.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `lsb_auto_lock` | `False` | `bool` | Enable the automatic LSB geometry lock. When `False`, `fit_image` behaves exactly as before — zero behavior change for existing users. |
| `lsb_auto_lock_maxgerr` | `0.3` | `float` (> 0) | Trigger threshold on the relative gradient error `|grad_error / grad|`. More sensitive than `maxgerr=0.5`, so the lock commits before a `stop_code=-1` would have. |
| `lsb_auto_lock_debounce` | `2` | `int` (1–10) | Number of consecutive triggered outward isophotes required before the lock commits. Debounces single-isophote noise spikes. |
| `lsb_auto_lock_integrator` | `'median'` | `str` | Integrator applied to locked-region isophotes. One of `'mean'` or `'median'`. `'median'` is more robust to the contaminants that typically trigger the lock in the first place. |

**Interaction notes:**

- **Conflict with geometry locks**: `lsb_auto_lock=True` raises `ValidationError` when combined with `fix_center=True`, `fix_pa=True`, or `fix_eps=True`. The auto-lock is meaningful only when outward growth starts from free geometry.
- **`debug` requirement**: the detector reads `grad`, `grad_error`, and `grad_r_error` off each isophote dict. If the caller leaves `debug=False`, isoster emits a `UserWarning` and internally promotes `debug=True` for the duration of the fit.
- **Lock anchor**: the isophote immediately *before* the trigger streak — not the trigger isophote itself, whose geometry may already have drifted.
- **One-way state machine**: once the lock commits, the mode stays locked for the remainder of outward growth. Inward growth and the central pixel are never locked.
- **Forced photometry unaffected**: the auto-lock applies only to regular `fit_image` calls. When combined with template-based forced photometry (`template=[...]`), `fit_image` emits a `UserWarning` and the feature is silently inactive.
- **Sampling / harmonic modes**: the auto-lock is agnostic to `use_eccentric_anomaly`, `simultaneous_harmonics`, and isofit-style modes — the detector only inspects per-isophote gradient diagnostics.

Result dict additions when `lsb_auto_lock=True`:

- `result["lsb_auto_lock"]` (`bool`): echoes that the auto-lock was used.
- `result["lsb_auto_lock_sma"]` (`float` or `None`): SMA at which the lock committed, or `None` if it never triggered.
- `result["lsb_auto_lock_anchor_sma"]` (`float` or `None`): SMA of the true geometry anchor — the last clean isophote before the debounce streak whose geometry the lock froze to (the `lsb_auto_lock_anchor` per-row marker, in contrast, goes on the committing trigger isophote). `None` if the lock never committed.
- `result["lsb_auto_lock_count"]` (`int`): number of outward isophotes fit under locked geometry.

Per-isophote additions (every isophote when `lsb_auto_lock=True`; inward isophotes and the central pixel always carry `False`, keeping the schema uniform for FITS round-trips):

- `iso["lsb_locked"]` (`bool`): fit under locked geometry.
- `iso["lsb_auto_lock_anchor"]` (`bool`): `True` only on the first locked isophote.

---

### 13. Outer Region Center Regularization

Soft damping complement to the automatic LSB lock. A logistic-ramp Tikhonov term damps noisy outward geometry updates against a frozen inner reference, making it harder for contamination or background structure to drive center, ellipticity, or PA walks before the hard lock fires. The default `outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0}` damps all geometry axes; set an axis weight to `0.0` to leave that axis undamped.

This feature is disabled by default and is still being validated on a broader range of surveys. Turn it on when fitting deep, LSB structure (e.g., HSC-scale edge cases) where contamination is expected to pull the free outward fit away from the real galaxy center. On clean high-S/N galaxies the feature is essentially a no-op at the recommended strengths.

The regular-mode driver always runs the inward loop before the outward loop. The inward pass is unchanged — only the execution order is swapped — so that this feature can build its reference from inner isophote centers.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `use_outer_center_regularization` | `False` | `bool` | Enable outer-region geometry regularization. The field name is retained for backward compatibility. When `False`, behavior is unchanged except for the inward-first loop reorder (which preserves all outward semantics). |
| `outer_reg_sma_onset` | `50.0` | `float` (> 0) | Logistic midpoint in pixels. The penalty is near zero for `sma << onset` and saturates for `sma >> onset`. Should be larger than `sma0`; a `UserWarning` is emitted if not. |
| `outer_reg_sma_width` | `None` | `float` (> 0) or `None` | Logistic width (growth-step scale). `None` auto-computes `0.4 * outer_reg_sma_onset`. |
| `outer_reg_strength` | `2.0` | `float` (≥ 0) | Saturated Tikhonov amplitude. Benchmarked as a reasonable default on HSC edge cases; increase for heavier damping when outskirts are dominated by contamination. |
| `outer_reg_weights` | `{'center': 1.0, 'eps': 1.0, 'pa': 1.0}` | `dict` | Per-axis damping weights for center, ellipticity, and PA. Unknown axes are ignored by the fitter but should be avoided; zero disables damping on that axis. |
| `outer_reg_mode` | `'damping'` | `'damping'` or `'solver'` | `damping` shrinks harmonic geometry steps without a reference pull. `solver` also pulls toward the frozen inner reference and is more biasing. |
| `outer_reg_ref_sma_factor` | `2.0` | `float` (> 1) | Inner reference window: only inward isophotes with `sma <= sma0 * factor` contribute to the flux-weighted reference mean. |

**Interaction notes:**

- **Independent from `lsb_auto_lock`**: the two flags compose cleanly. Soft regularization runs pre-lock; after the lock commits, `fix_center=True` makes the penalty a no-op on the locked tail.
- **Damping plus selector**: in default `outer_reg_mode='damping'`, the Tikhonov term shrinks harmonic geometry steps in the outer region. A selector-level penalty also contributes to `effective_amp = abs(max_amp) + reg_penalty + outer_reg_penalty` so cumulative drift is bounded when several candidate iterations are available. `outer_reg_mode='solver'` additionally pulls toward the frozen reference and should be treated as more biasing.
- **Inward-first loop order**: the regular-mode driver unconditionally runs the inward pass before the outward pass. The inward pass's outputs are unchanged; consumers that iterate over `results['isophotes']` by index are unaffected because the result list is still assembled in sma-sorted order.
- **Reference construction**: `(x0_ref, y0_ref, eps_ref, pa_ref)` is a flux-weighted mean over the anchor plus inward isophotes with acceptable stop codes and `sma <= sma0 * outer_reg_ref_sma_factor`. PA uses a circular mean on `2*pa`. When no inward isophote qualifies, the reference falls back to the anchor geometry.
- **`outer_reg_sma_onset < sma0`**: the penalty would fire from the first outward step; a `UserWarning` is emitted at config time.
- **`minsma >= sma0`**: no inward isophotes exist, reference falls back to the anchor; a `UserWarning` is emitted.
- **Fixed geometry flags**: a positive weight on a fixed axis is inert because that axis never moves; a `UserWarning` is emitted at config time for `fix_center`, `fix_pa`, or `fix_eps` when the corresponding `outer_reg_weights` entry is positive.
- **Forced photometry unaffected**: the feature applies only to regular `fit_image` calls. When combined with template-based forced photometry, `fit_image` emits a `UserWarning` and the feature is silently inactive.
- **Sampling / harmonic modes**: agnostic to `use_eccentric_anomaly`, `simultaneous_harmonics`, and isofit-style modes, because the penalty rides the same `effective_amp` rail in the best-iteration selector.

Result dict additions when `use_outer_center_regularization=True`:

- `result["use_outer_center_regularization"]` (`bool`): echoes that the feature was used.
- `result["outer_reg_x0_ref"]` (`float`): frozen inner reference `x0`.
- `result["outer_reg_y0_ref"]` (`float`): frozen inner reference `y0`.
- `result["outer_reg_eps_ref"]` (`float`): frozen inner reference ellipticity.
- `result["outer_reg_pa_ref"]` (`float`): frozen inner reference PA.

No per-isophote fields are added.

---

### 14. First Isophote Robustness

Retry and failure detection for the first isophote at `sma0`. When the first isophote
fails and cannot be recovered, the fit returns only the central pixel; these options let
the driver retry with perturbed starting conditions before declaring failure.

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `max_retry_first_isophote` | `3` | `int` (0-20) | Maximum retry attempts for the first isophote when it fails (stop code not in `{0, 1, 2}`). Each attempt perturbs `sma0` and/or the initial geometry (`eps`, `pa`). `0` disables retry. |
| `first_isophote_fail_count` | `3` | `int` (1-10) | Number of consecutive initial isophotes that must all fail before declaring `FIRST_FEW_ISOPHOTE_FAILURE`. Default 3 means the first isophote at `sma0` plus the next 2 growth steps must all have unacceptable stop codes. |

**Interaction notes:**

- Retries run only when the first isophote at `sma0` fails, so the default adds near-zero
  overhead when it already succeeds. The fixed schedule tries `(0.8 * sma0, same
  geometry)`, `(1.3 * sma0, same geometry)`, `(0.6 * sma0, near-circular eps=0.05)`,
  `(1.5 * sma0, pa + pi/4)`, `(0.5 * sma0, eps=0.05, pa + pi/2)`, then an extended cycle,
  stopping at the first acceptable isophote.
- The probe loop runs only when the anchor and all retries failed: the driver tries
  `first_isophote_fail_count - 1` further growth steps, and the first acceptable probe
  becomes the anchor. `result["first_isophote_failure"]` is set to `True` (with a
  `FIRST_FEW_ISOPHOTE_FAILURE` warning) only when the anchor and every retry/probe
  attempt failed.
- Each retry attempt is logged in `result["first_isophote_retry_log"]`.

---

## Variance Map / WLS Fitting (via `variance_map`)

Weighted Least Squares fitting is enabled by passing a per-pixel variance map to
`fit_image()`. This is a function-level parameter, not an `IsosterConfig` field.

```python
results = fit_image(image, config=config, variance_map=variance_map)
```

### Input requirements

| Requirement | Behavior |
|-------------|----------|
| Shape must match `image` | `ValueError` raised on mismatch |
| `NaN` / `inf` values (non-finite) | Marked invalid; the corresponding sample is dropped during sampling, one warning emitted per call |
| Non-positive values (0, negative) | Marked invalid; the corresponding sample is dropped during sampling, one warning emitted per call |
| Caller's array | Never mutated (copy-on-write) |

A variance entry is only usable if it is finite and strictly positive. Entries that fail
this test carry no usable noise information, so the sample at that pixel is dropped from
sampling entirely — the same treatment already given to masked pixels — rather than being
kept with a substitute value. This guarantees that a ring's reported statistic (intensity,
gradient) and its reported uncertainty always describe the identical set of samples. An
earlier version of this feature instead substituted a large sentinel (`1e30`) for
non-finite entries and clamped non-positive entries to a tiny value (`1e-30`); both have
been retired because they left the flagged pixel in the sample set while distorting its
weight, which could shrink a reported gradient error by roughly fifteen orders of
magnitude and silently defeat the `maxgerr` check, the signal-to-noise damping, and the LSB
auto-lock. See `docs/04-architecture.md`, section "Invalid-variance policy", for the full
account and a measured real-data example.

### Interaction with other parameters

- `variance_map=None` (default): pure OLS, byte-identical to pre-WLS code.
- Compatible with all geometry constraints: `fix_center`, `fix_pa`, `fix_eps`.
- Compatible with `simultaneous_harmonics=True` (isofit mode).
- Compatible with `template` forced photometry mode.

---

## Forced Photometry (via `template`)

Forced photometry is performed by passing the `template` argument to `fit_image()`. This
bypasses the iterative fitting loop and extracts intensity at predetermined geometries.

### Template-Based Variable Geometry

Uses per-SMA geometry from a previous isoster run. Each isophote in the template
provides its own x0, y0, eps, and pa, enabling variable geometry along the radial
profile. This is the IRAF `ellipse` forced-mode equivalent for multiband analysis.

The `template` argument accepts:
- A results dict (e.g., from a previous `fit_image()` call)
- A file path (str/Path) to a FITS file saved by `isophote_results_to_fits()`
- A list of isophote dicts containing at least `sma`, `x0`, `y0`, `eps`, and `pa`.

```python
# Fit reference band
results_g = isoster.fit_image(image_g, mask_g, config)

# Apply g-band geometry to r-band
results_r = isoster.fit_image(image_r, mask_r, config, template=results_g)
```

### Config parameters silently dropped in forced mode

Template-based forced mode bypasses the iterative fitting loop entirely. The following
config parameters have no effect in forced mode:

- `maxit`, `minit`, `conver`, `convergence_scaling` (no iteration)
- `geometry_damping`, `geometry_update_mode`, `permissive_geometry` (no updates)
- `sclip_low`, `sclip_high` (only symmetric `sclip` is used)
- `fflag` (no flagging check)
- `full_photometry` (aperture photometry not computed)
- `compute_errors` (all errors set to 0.0)
- `compute_deviations` (all deviations set to 0.0)
- `compute_cog` (curve-of-growth not computed in forced extraction)

---

## Photometry Outputs

### `full_photometry` -- Aperture Photometry

Direct pixel summation from the 2D image within elliptical and circular apertures.
Computed per-isophote via `compute_aperture_photometry()` in `fitting.py`.

Output fields added to each isophote dict:

| Field | Description |
|-------|-------------|
| `tflux_e` | Total flux within the elliptical aperture |
| `tflux_c` | Total flux within the circular aperture (radius = SMA) |
| `npix_e` | Number of valid pixels in the elliptical aperture |
| `npix_c` | Number of valid pixels in the circular aperture |

### `compute_cog` -- Curve-of-Growth Photometry

Model-based cumulative flux profile derived from the fitted 1D intensity profile using
trapezoidal annular integration. Computed post-sweep in `driver.py` via `cog.py`. This
method never touches raw pixels; it integrates the analytic annular flux from the fitted
intensity values and ellipse geometry.

Output fields added to each isophote dict:

| Field | Description |
|-------|-------------|
| `cog` | Cumulative flux enclosed within the isophote |
| `cog_annulus` | Annular flux contribution from this isophote |
| `area_annulus` | Geometric area of the annular region |
| `flag_cross` | Boolean flag for isophote crossing detection |
| `flag_negative_area` | Boolean flag for negative annular area |

### When to Use Each

- Use `full_photometry=True` for ground-truth aperture flux measured directly from the
  image. Suitable for total magnitude estimation and aperture correction work.
- Use `compute_cog=True` for model-derived cumulative flux profiles. Suitable for smooth
  growth curves and enclosed-flux diagnostics where pixel noise should be suppressed.

---

## Integration Modes

### Sub-pixel Interpolation

All sampling modes use bilinear sub-pixel interpolation internally via
`scipy.ndimage.map_coordinates` with `order=1`. This is not configurable.

### Mean Integrator (`integrator='mean'`)

Reports the harmonic-fit DC term (`y0_fit`) as the representative intensity. This is the
fitted mean from the 5-parameter (or extended ISOFIT) least-squares harmonic model, not
a simple `np.mean` of the sampled intensities. The harmonic fit naturally downweights
outliers through the least-squares objective.

### Median Integrator (`integrator='median'`)

Reports `np.median(intens)` of the sigma-clipped intensity samples. More robust to
contamination from bright foreground objects or cosmic rays, but discards the harmonic
model information. Its error bar is the mean's `rms/√N` scaled by the
Gaussian-asymptotic median factor `√(π/2) ≈ 1.25` (the median is a noisier
estimator than the mean for Gaussian noise). This `√(π/2)` scaling now applies uniformly to
`intens_err` whether or not a `variance_map` is supplied — an earlier version applied it
only when no variance map was given, understating the reported intensity error under WLS.

The reported ring **gradient** does not use this weighted-intensity path at all: it always
uses an unweighted mean or median of the same two rings, with a matching error formula
(`sum(v_i)/N**2` for the mean, and the normal-theory median form
`pi*N/(2*(sum 1/sqrt(v_i))**2)`, which reduces to `pi*sigma**2/(2N)` for uniform variance).
This is a deliberate difference from the intensity, which under WLS is the
inverse-variance-weighted harmonic intercept — see `docs/04-architecture.md`, section
"Gradient error and ring statistics", for the full formulas and the reasoning behind the
asymmetry. Without a variance map, the same helper falls back to a scatter-based error
computed from the sampled intensities; because that scatter includes real galaxy structure
along the ring, it is an upper bound on the noise rather than a pure noise estimate.

### Adaptive Integrator (`integrator='adaptive'`)

Switches between mean and median based on SMA:

- SMA <= `lsb_sma_threshold`: uses mean (harmonic-fit DC term)
- SMA > `lsb_sma_threshold`: uses median

Rationale: inner high-S/N isophotes benefit from the precision of the harmonic fit, while
outer low-S/N isophotes benefit from the robustness of the median against noise spikes.

### Comparison with photutils

The photutils `BILINEAR` integrator uses area-weighted sector integration, which is
architecturally different from isoster's point-sampling path. Both use bilinear
interpolation at the sub-pixel level, but the integration geometry differs. Isoster
samples discrete points along the ellipse; photutils integrates over sectors spanning
the annular width.

---

## Known Limitations

The following limitations are documented based on the current implementation state.
Cross-references point to `future.md`, an internal planning document that is not part
of the public docs tree.

1. **Template-based forced mode drops most output fields.** The `template` mode
   produces isophote dicts with zero-valued errors, zero deviations, and no aperture
   photometry. Only intensity, RMS, and geometry fields are meaningful.
   (See future.md: "Typed isophote result schema".)

2. **No stop-code filtering in model builder.** `build_isoster_model()` accepts all
   rows with `sma > 0` and finite intensity/geometry values regardless of stop code.
   Rows with non-finite `intens`, `x0`, `y0`, `eps`, or `pa` are dropped automatically,
   but stop codes are not consulted; users should pre-filter isophotes by stop code
   before model reconstruction for best results.
   (See future.md: "Model builder robustness against low-quality rows".)

3. **CLI exposes only a subset of config.** Advanced options (eccentric anomaly, ISOFIT
   harmonics, CoG, permissive geometry) are not available via CLI flags. Use the Python
   API or a YAML config file.
   (See future.md: "CLI option parity with advanced config features".)

4. **Serial template-based execution.** Template-based forced mode processes
   each SMA sequentially. No parallel execution path exists.
   (See future.md: "Parallel forced/template execution path".)

5. **Central regularization requires previous isophote.** The regularization penalty
   compares against the previous isophote's geometry. It has no effect on the first
   isophote at `sma0`, which may still exhibit instability in low-S/N centers.

6. **Gradient estimator is tightly coupled.** The current gradient computation uses a
   fixed two-radius finite-difference scheme. Alternative estimators are not configurable.
   (See future.md: "Gradient estimator redesign".)

7. **Fixed sampling density formula.** Sampling density follows
   `max(64, int(2 * pi * sma))` and does not adapt to ellipticity, S/N, or local
   curvature.
   (See future.md: "Adaptive sampling density with error controls".)
