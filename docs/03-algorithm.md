# Algorithm and Implementation Notes

This document describes the behavior of the current code path in `isoster/sampling.py`, `isoster/fitting.py`, and `isoster/driver.py`.

## Pipeline Overview

Regular mode in `fit_image` (`isoster/driver.py`) runs:

1. Central pixel estimate (`fit_central_pixel`).
2. First free-geometry isophote at `sma0` (`fit_isophote`).
3. Optional inward growth until `max(minsma, 0.5)`, but only when the anchor isophote stop code is acceptable (`0`, `1`, or `2`).
4. Outward growth until `maxsma`.
5. Optional CoG attachment (`compute_cog`) in regular mode.

The inward-first execution order is unconditional. It lets the outer-region
regularization feature build an inner reference before outward growth starts,
while the returned `results["isophotes"]` list is still assembled in ascending
SMA order.

Mode selection priority is:

1. `template` provided (e.g., results dict, FITS path, or isophote list).
2. Regular mode.

(`template_isophotes` is deprecated and will be removed in a future version.)

## Sampling and Angle Spaces

`extract_isophote_data` uses `map_coordinates` to sample intensities along ellipse coordinates computed by `compute_ellipse_coords`.

Sampling density:

- `n_samples = max(64, int(2*pi*sma))`

Angle semantics:

- Regular mode (`use_eccentric_anomaly=False`): harmonics use `phi` (position angle), and `angles is phi`.
- Eccentric-anomaly mode (`use_eccentric_anomaly=True`): harmonics use `psi`. The geometry update consumes those `psi`-basis coefficients **directly** — there is no conversion back to `phi`. `phi` is carried alongside (and sigma-clipped in lockstep) so the angular bookkeeping stays aligned for diagnostics and error propagation, but it does not enter the update. The update rule is a descent direction driven to a fixed point, and the two bases agree on where the first- and second-order residual vanishes because they are related by a fixed monotonic reparametrization at fixed geometry (`tan(phi) = (1 - eps) * tan(psi)`); they disagree about the size of a mismatch mid-iteration, so the trajectory differs. Earlier versions of this line, and of the comments in `fitting.py`, claimed geometry updates used `phi`; that was never what the code did (corrected in `80618bb`).

The returned structure is `IsophoteData(angles, phi, intens, radii, variances)` where `angles` is the harmonic-fit angle basis and `variances` is `None` when no variance map is provided.

## Per-Isophote Fitting Loop

`fit_isophote` iterates up to `maxit`:

1. Sample data for current geometry.
2. Sigma-clip sampled profile (`sigma_clip`).
3. Early quality exits:
   - stop code `1` when `actual_points < total_points * (1.0 - fflag)`.
   - stop code `3` when `len(intens) < 6`.
4. Fit first/second harmonics by least squares.
5. Compute radial gradient using offset SMA sampling.
   - **Lazy Evaluation**: If `use_lazy_gradient=True`, the gradient is evaluated only
     on iteration 0 and reused for subsequent iterations. It is re-evaluated if
     convergence stalls (3 iterations without amplitude improvement). This typically
     reduces sampling overhead by ~45%.
6. Apply gradient quality checks (outward mode only):
   - uses `maxgerr`, sign checks, and a two-strike `lexceed` rule before stop code `-1`.
7. Compute dominant harmonic amplitude and geometry correction target.
8. Track the best geometry by minimal `effective_amp = abs(max_amp) + regularization_penalty`.
9. Converge when `abs(max_amp) < conver * convergence_scale * effective_rms` and `i >= minit`.
   - **Noise Floor**: If `sigma_bg` is provided, `effective_rms = max(rms, sigma_bg / sqrt(N))`. This prevents overfitting noise-induced asymmetries in LSB regions.
10. If iterations reach `maxit` without convergence, return best-so-far geometry with stop code `2`.

On convergence, optional blocks are attached:

- parameter errors (`compute_errors=True`)
- higher-order harmonics (`compute_deviations=True`, sequential or simultaneous)
- aperture metrics (`full_photometry=True`)

## Geometry Update Mapping

The dominant first/second harmonic term selects the parameter update:

- max index `0` (`A1`) -> center shift (minor-axis direction component).
- max index `1` (`B1`) -> center shift (major-axis direction component).
- max index `2` (`A2`) -> position-angle update.
- max index `3` (`B2`) -> ellipticity update with bounds/wrap handling.

Constraint flags (`fix_center`, `fix_pa`, `fix_eps`) zero the corresponding harmonic terms in the dominance selection step.

## Safety and Damping
- **Gradient SNR Damping**: If the local radial gradient is noisy ($SNR < 3$), the iteration step size (`geometry_damping`) is dynamically reduced to prevent overfitting the noise floor.
- **Step Clipping**: Hard limits (`clip_max_shift`, `clip_max_pa`, `clip_max_eps`) restrict the maximum geometry update per iteration. This prevents runaway divergence when the model temporarily evaluates across an image defect or extreme noise.

## Integrators

Supported config values:

- `mean`
- `median`
- `adaptive` (switches to `median` when `sma > lsb_sma_threshold`)

`adaptive` requires `lsb_sma_threshold` by config validation.

## Weighted Least Squares (Variance Map)

When `variance_map` is provided to `fit_image`, all harmonic fits switch to WLS:

**OLS (default)**: minimizes `||y - Ax||²`, covariance = `(A^T A)^-1 * σ²_res` where `σ²_res` is estimated from residuals.

**WLS**: minimizes `||W^{1/2}(y - Ax)||²` with `W = diag(1/σ²_i)`:
- Normal equations: `(A^T W A) x = A^T W y`
- Covariance: `(A^T W A)^-1` — exact, no residual scaling needed.

**Gradient error** (WLS): `σ_grad = sqrt(Var(mean_c) + Var(mean_g)) / Δr` where `Var(mean) = Σσ²_i / N²`.

**Intensity error** (WLS): `intens_err = sqrt(Σσ²_i / N²)` (exact variance of the mean).

The OLS path is byte-identical when `variance_map=None`. All WLS branches are gated by `if variances is not None`.

The WLS error propagation above was cross-checked against the autoprof variance-map derivation (internal review record, not part of the public docs tree).

## Forced and Template Modes

### Template-based forced (`template`)

- Accepts results dict, FITS path, or list of isophote dicts.
- Sorts template by `sma`.
- Reuses each template isophote geometry at its own SMA.
- Handles `sma == 0` via `fit_central_pixel`.

## Stop Codes

Current `isoster` fitting emits:

- `0`: success
- `1`: too many flagged points
- `2`: max-iteration fallback (best-so-far geometry)
- `3`: too few points
- `-1`: gradient-related failure


Canonical user-facing reference: `docs/01-user-guide.md`.

## Current Caveats

- Central regularization uses `previous_geometry` propagated by regular `fit_image` outward/inward growth when enabled.
- In `compute_gradient`, both linear and multiplicative growth use `/ delta_r` normalization (where `delta_r = step` for linear or `sma * step` for multiplicative).
- `build_isoster_model` filters on `sma > 0` and drops rows with non-finite intensity/geometry values, but does not filter by stop-code quality.
