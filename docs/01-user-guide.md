# ISOSTER User Guide

This guide covers installation, configuration, common workflows, and stop-code interpretation.

## Installation

Use `uv` for environment and dependency workflow in this repository.

```bash
uv sync --extra dev --extra docs
uv run isoster --help
```

## Basic Fit Workflow

```python
from astropy.io import fits
from isoster import fit_image, isophote_results_to_fits
from isoster.config import IsosterConfig

image = fits.getdata("data/m51/M51.fits")

config = IsosterConfig(
    sma0=10.0,
    minsma=0.0,
    maxsma=120.0,
    full_photometry=True,
)

results = fit_image(image, mask=None, config=config)
isophote_results_to_fits(results, "outputs/m51_isophotes.fits")
```

## Mode Selection

`fit_image` mode priority:

1. `template` provided -> template-based forced photometry.
2. Otherwise -> regular iterative fitting.

### Template-Based Forced Photometry

Forced photometry (multiband analysis) is performed by passing the `template` argument. It accepts a results dict, a FITS path, or a list of isophote dicts.

```python
from isoster import fit_image
from isoster.config import IsosterConfig

config = IsosterConfig()

# Fit reference band (geometry discovery)
results_g = fit_image(image_g, mask=mask_g, config=config)

# Apply g-band geometry to r-band (forced photometry)
results_r = fit_image(
    image_r,
    mask=mask_r,
    config=config,
    template=results_g,
)
```

To perform forced photometry with a fixed geometry at specific SMAs, provide a list of dicts to `template`:

```python
template = [
    {'sma': s, 'x0': 128.0, 'y0': 128.0, 'eps': 0.3, 'pa': 0.2}
    for s in [5, 10, 20, 40, 80]
]
results = fit_image(image, mask=mask, config=config, template=template)
```

#### Higher harmonics in forced mode

With `compute_deviations=True`, forced photometry measures `a_n` / `b_n` on
each fixed ring, exactly as regular fitting does after convergence. The
geometry is not fitted, but the harmonic solve and its Bender normalization by
`sma * |dI/da|` are the same, so forced and regular values are directly
comparable — identical, in fact, on the same rings when
`use_lazy_gradient=False`. With the default lazy gradient the regular path may
reuse a cached gradient, so the two agree closely rather than exactly.

A ring that cannot be measured — one falling outside the image, or one whose
radial gradient is zero or non-finite — reports `NaN` for its harmonics, not
`0.0`. Zero is a meaningful measured value for a perfect ellipse, so it is
never used as a placeholder for a missing one.

!!! warning "Behavior change"
    Earlier versions reported `a_n = b_n = 0.0` for every forced isophote: the
    columns were emitted but the harmonic solve was never run. Any harmonic
    measured in forced mode before this change is a fabricated zero and should
    be recomputed. See `docs/04-architecture.md` for the exact contract.

### Automatic LSB Geometry Lock

`fit_image` can be run in a mode that starts with free geometry and automatically switches to fixed geometry once the outward fit enters the low-surface-brightness regime. It combines the strengths of free fitting (true isophotal shapes in the high-S/N region) with fixed-geometry fitting (no centroid drift in the LSB outskirts) in a single pass — no post-hoc stitching.

```python
from isoster import fit_image
from isoster.config import IsosterConfig

config = IsosterConfig(
    sma0=10.0,
    maxsma=200.0,
    lsb_auto_lock=True,                   # enable automatic LSB lock
    lsb_auto_lock_maxgerr=0.3,            # lock when grad_r_error > 0.3
    lsb_auto_lock_debounce=2,             # for 2 outward isophotes in a row
    lsb_auto_lock_integrator="median",    # use median in the LSB region
    debug=True,                           # required for the gradient diagnostics
)
result = fit_image(image, mask=mask, config=config, variance_map=variance)

transition = result["lsb_auto_lock_sma"]
locked_n = result["lsb_auto_lock_count"]
print(f"Lock committed at sma = {transition}; {locked_n} locked isophotes")
```

How the detector works:

1. Free outward growth runs exactly like `lsb_auto_lock=False`.
2. After each outward isophote, the relative gradient error `|grad_error / grad|` is inspected. A trigger fires when `grad_r_error > lsb_auto_lock_maxgerr`, OR the measured gradient is non-negative (`grad >= 0`, i.e. the profile no longer declines outward), OR the isophote returned `stop_code = -1`. The threshold is more sensitive than `maxgerr` so the lock commits *before* a gradient failure.
3. Triggers are debounced: `lsb_auto_lock_debounce` consecutive triggered isophotes are required. A clean isophote in the streak resets the counter.
4. When the lock commits, the anchor is the isophote **immediately before** the streak — not the trigger isophote itself — and its geometry (`x0`, `y0`, `eps`, `pa`) is carried forward. The integrator switches to `lsb_auto_lock_integrator` for the remaining outward isophotes.
5. The transition is **one-way** per fit. Inward growth and the central pixel are unchanged.

`debug=True` is required because the detector reads `grad`, `grad_error`, and `grad_r_error` from each outward isophote dict. If the caller leaves `debug=False`, isoster emits a `UserWarning` and internally flips the flag for the duration of the fit.

The following fields are conflict-checked and raise `ValidationError`:
`lsb_auto_lock=True` cannot be combined with `fix_center=True`, `fix_pa=True`, or `fix_eps=True`.

The lock is wired only into the regular-mode driver. When combined with template-based forced photometry (`template=[...]`), `fit_image` emits a `UserWarning` and the lock is silently inactive. The same applies to all sampling and harmonic variants — the lock is agnostic to `use_eccentric_anomaly`, `simultaneous_harmonics`, and the isofit-style modes, because it only inspects the per-isophote gradient diagnostics, not the geometry update path.

### Outer Region Center Regularization

The automatic LSB lock is a hard switch — once the detector trips, the center is frozen. Outer-region regularization is a *soft* complement that runs before the lock fires: a logistic-ramp Tikhonov term that damps outward geometry updates in the LSB regime while still letting real structure dominate when the data support it. By default, the damping applies to center, ellipticity, and position angle through `outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0}`. Set an axis weight to `0.0` to leave that axis undamped.

**When to turn it on.** This feature is disabled by default because it is still being validated on a broader range of surveys. We recommend turning it on when fitting deep, low-surface-brightness structure where the outskirts are close to the sky level, and especially when contamination (nearby bright sources, scattered light, faint companions) is expected to pull the free outward fit away from the real galaxy center. On HSC-scale edge cases we see clear drift reduction in the pre-lock tail with minimal disruption in the high-S/N interior. On clean synthetic galaxies the feature is essentially a no-op at the recommended strengths.

The regular-mode driver always runs the inward loop before the outward loop (this is an unconditional reorder that preserves all outward semantics). With this feature on, the inward isophotes are used to build a stable inner reference centroid (flux-weighted mean of inner isophote centers) that the outward fit is pulled toward.

```python
from isoster import fit_image
from isoster.config import IsosterConfig

config = IsosterConfig(
    sma0=10.0,
    maxsma=200.0,
    # Automatic LSB lock (optional, composes cleanly)
    lsb_auto_lock=True,
    debug=True,
    # Outer-region soft geometry regularization
    use_outer_center_regularization=True,
    outer_reg_sma_onset=50.0,        # penalty starts ramping near sma=50
    outer_reg_sma_width=15.0,        # logistic width (growth-step scale)
    outer_reg_strength=2.0,          # penalty amplitude
    outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
)
result = fit_image(image, mask=mask, config=config, variance_map=variance)

print("inner reference:", result["outer_reg_x0_ref"], result["outer_reg_y0_ref"])
```

How it works:

1. The driver runs the inward loop first to collect high-S/N inner isophotes.
2. A flux-weighted mean of their `(x0, y0, eps, pa)` — restricted to `sma <= sma0 * outer_reg_ref_sma_factor` — becomes the frozen inner reference. If no inward isophotes qualify, the reference falls back to the anchor isophote geometry.
3. During outward growth, the logistic ramp `lambda(sma) = outer_reg_strength / (1 + exp(-(sma - onset) / width))` scales a per-axis Tikhonov damping term. In default `outer_reg_mode="damping"`, harmonic geometry steps are multiplied by `(1 - alpha)` in the outer region; in `outer_reg_mode="solver"`, the update also pulls toward the frozen reference.
4. A complementary selector-level penalty still participates in the best-iteration choice through `effective_amp`, which bounds cumulative drift when several candidate iterations are available.
5. When `lsb_auto_lock=True` fires, the locked clone sets `fix_center=True`, `fix_pa=True`, and `fix_eps=True`, so the regularization becomes a no-op for the locked tail. The two mechanisms compose: soft pre-lock, hard post-lock.

Result dict additions when `use_outer_center_regularization=True`:

```python
result["use_outer_center_regularization"]  # True
result["outer_reg_x0_ref"]                 # frozen inner reference x0
result["outer_reg_y0_ref"]                 # frozen inner reference y0
result["outer_reg_eps_ref"]                # frozen inner reference eps
result["outer_reg_pa_ref"]                 # frozen inner reference pa
```

Interaction with existing flags:

- **Independent from `lsb_auto_lock`**: either can run alone or both together.
- **`minsma >= sma0`**: the reference falls back to the anchor; a `UserWarning` is emitted at config time.
- **`outer_reg_sma_onset < sma0`**: the penalty would fire from the first outward step; a `UserWarning` is emitted.
- **Fixed geometry flags**: a positive weight on a fixed axis is inert because that axis never moves; a `UserWarning` is emitted at config time for `fix_center`, `fix_pa`, or `fix_eps` when the corresponding `outer_reg_weights` entry is positive.
- **Forced photometry (`template=[...]`)**: the feature is wired only into the regular-mode driver. When combined with a template, `fit_image` emits a `UserWarning` and the feature is silently inactive.
- **Sampling / harmonic modes**: the feature is agnostic to `use_eccentric_anomaly`, `simultaneous_harmonics`, and the isofit-style modes, because the penalty rides the same `effective_amp` rail in the best-iteration selector.

### Using Variance Maps (WLS Fitting)

When a per-pixel variance map is available (e.g., from survey pipelines like DESI Legacy Survey), pass it to `fit_image` to enable Weighted Least Squares fitting:

```python
from astropy.io import fits
from isoster import fit_image
from isoster.config import IsosterConfig
import numpy as np

image = fits.getdata("galaxy-image-r.fits.fz")
invvar = fits.getdata("galaxy-invvar-r.fits.fz")

# Convert inverse variance to variance; invvar == 0 pixels become
# NaN, which ISOSTER marks invalid and drops from the fit
variance_map = np.where(invvar > 0, 1.0 / invvar, np.nan)

config = IsosterConfig(sma0=10.0, maxsma=100.0)
results = fit_image(image, config=config, variance_map=variance_map)
```

WLS benefits:
- **Error bars traceable to a stated noise model**: they come from the supplied variance
  map rather than from fit residuals, so they can be checked against the survey's own
  noise estimate. This is *not* the same as being exact — the covariance is calibrated
  only to the extent the variance map is, and ring samples drawn by interpolation from
  a shared pixel grid are not strictly independent, which biases the reported
  uncertainties low at high sampling density.
- **Outlier down-weighting, when the map identifies the outlier**: a sample whose
  variance is large relative to its neighbours carries proportionally little weight, so
  cosmic rays and hot pixels are suppressed without the caller masking them by hand.
  A defect the variance map does not flag is *not* down-weighted.
- **Pure photon-noise gradient error**: cleanly separated from galaxy structure scatter (arms, dust, bars).
- **Byte-identical fallback**: when `variance_map=None` (default), the code path is identical to OLS.

Note on error-bar size: WLS errors are **not** systematically larger or smaller than
OLS. Under the textbook model — independent samples with a common, correctly specified
variance — the two have the same expected covariance. They diverge when those
assumptions fail, and the direction depends on how: unmodelled azimuthal structure
(arms, dust, orders n >= 3) inflates the OLS residual-scatter estimate, making OLS
errors the *larger* ones in that regime. See
[`technical/1.4.2`](technical/1.4.2-variance-aware-fitting.md).

Input requirements and safeguards:
- `variance_map` must have the same shape as `image` (raises `ValueError` otherwise).
- `NaN` and `inf` values are invalid: the corresponding sample is dropped during sampling (not kept with a substitute value) and a `RuntimeWarning` is emitted.
- Non-positive values (zeros, negatives) are also invalid and dropped the same way, with a warning. Consider masking these pixels explicitly if the exclusion is intended. (An earlier version of isoster instead substituted `1e30` for non-finite entries and clamped non-positive entries to `1e-30`; both have been retired because they kept the flagged sample in the statistic while distorting its weight — see `docs/04-architecture.md`, section "Invalid-variance policy".)
- The caller's array is never mutated (copy-on-write).
- For inverse-variance maps, convert with `variance = 1.0 / invvar` and handle zero-invvar pixels appropriately: leave them as `inf` or `nan` (ISOSTER marks these invalid and drops the sample automatically), or add them to the `mask` array explicitly. Do not substitute a large finite value such as `1e30` — it passes the validity check and stays in the ring, inflating the reported gradient error by orders of magnitude instead of removing the pixel's influence.

Compatibility:
- WLS works with all constraint modes: `fix_center`, `fix_pa`, `fix_eps`, and `simultaneous_harmonics` (isofit).
- WLS error bars are **not** systematically larger or smaller than OLS. Under independent samples with a common, correctly specified variance the two have the same expected covariance; they diverge only where those assumptions fail, and the direction depends on how. Unmodelled azimuthal structure (arms, dust, orders n >= 3) inflates the OLS residual-scatter estimate, making OLS the larger one in that regime.

## Key Configuration Options

### Sampling and Stability

- `use_eccentric_anomaly=True`: harmonics are fitted in `psi`, and the geometry update consumes those `psi`-basis coefficients **directly** — there is no conversion back to `phi`. That is correct rather than approximate, because the Jedrzejewski corrections are themselves derived from an eccentric-anomaly parametrisation (see [`03-algorithm.md`](03-algorithm.md)).
- `permissive_geometry=True`: allows geometry propagation through weaker gradient diagnostics.
- `maxgerr`: controls tolerance for gradient relative error checks.

### Harmonics

- `compute_deviations=True`: enables higher-order deviation outputs (`a{n}`, `b{n}`).
- `harmonic_orders=[3, 4, ...]`: harmonic orders to compute.
- `simultaneous_harmonics=True`: true ISOFIT (Ciambur 2015) — fits higher-order harmonics jointly with geometry harmonics inside the iteration loop via a single extended design matrix. Accounts for cross-correlations and produces cleaner RMS estimates. Falls back to 5-param fit when insufficient sample points for the full design matrix.

### First Isophote Robustness

When the first isophote at `sma0` fails, the entire fitting silently returns only the central pixel.
Two config options improve this:

- `max_retry_first_isophote` (default `3`; `0` disables): number of retry attempts with
  perturbed `sma0` and initial geometry (`eps`, `pa`). Each attempt tries a different
  combination from a fixed schedule to find an acceptable starting isophote. Because
  retries only run when the first isophote fails, the default adds near-zero overhead
  when it already succeeds. Set to `5` for robust batch processing.
- `first_isophote_fail_count` (default `3`): how many consecutive initial isophotes must
  all fail before a `FIRST_FEW_ISOPHOTE_FAILURE` warning is emitted.

When a failure is detected, the result dict contains:
- `result["first_isophote_failure"]` = `True`
- `result["first_isophote_retry_log"]` (list of attempt dicts, only when retries were attempted)

```python
config = IsosterConfig(sma0=10.0, max_retry_first_isophote=5)
result = fit_image(image, mask=None, config=config)

if result.get("first_isophote_failure"):
    print("First isophote fitting failed — check sma0 and initial geometry")
```

### Photometry Outputs

- `full_photometry=True`: adds `tflux_e`, `tflux_c`, `npix_e`, `npix_c`.
- `compute_cog=True`: adds CoG fields (`cog`, `cog_annulus`, `area_annulus`, crossing flags) in regular mode.

## Model Reconstruction

```python
from isoster import build_isoster_model

# Auto-detect harmonic orders and EA mode from isophote results
model = build_isoster_model(
    image.shape,
    results["isophotes"],
    use_harmonics=True,
)
```

`build_isoster_model` parameters for harmonic reconstruction:

- `harmonic_orders` (default `None`): When `None`, auto-detects which harmonic orders are present by scanning isophote dicts for `a{n}` keys (n >= 3). Pass an explicit list like `[3, 4]` to restrict to specific orders.
- `use_eccentric_anomaly` (default `None`): When `None`, auto-detects from the `use_eccentric_anomaly` flag stored in isophote dicts by `fit_isophote()`. When `True`, evaluates harmonics in eccentric anomaly (psi) space; when `False`, uses position angle (phi) space. Must match the angle space used during fitting for correct reconstruction.

## Serialization Helpers

### FITS

`isophote_results_to_fits` writes a 4-HDU FITS file:

- **HDU 0** — `PrimaryHDU`: empty, no data.
- **HDU 1** — `BinTableHDU` named `ISOPHOTES`: one row per fitted isophote, columns matching the isophote dict keys.
- **HDU 2** — `BinTableHDU` named `CONFIG`: two columns (`PARAM`, `VALUE`) with the full `IsosterConfig` serialized as JSON strings, one row per field.
- **HDU 3** — `BinTableHDU` named `META`: the same `PARAM`/`VALUE` JSON layout, carrying the remaining top-level keys of the results dict (`lsb_auto_lock*`, `first_isophote_*`, outer-regularization references, and so on).

This layout avoids `HIERARCH` warnings that occurred when config was written as FITS header keywords. Files written by older versions (config in header keywords, no `CONFIG` extension) are still readable, but the reader returns `config=None` for them — no header-keyword reconstruction is performed.

```python
from isoster import isophote_results_to_fits, isophote_results_from_fits

# Save
isophote_results_to_fits(results, "outputs/galaxy_isophotes.fits")

# Load
loaded = isophote_results_from_fits("outputs/galaxy_isophotes.fits")
```

### ASDF

ASDF (Advanced Scientific Data Format) is supported as an optional alternative. It preserves Python types natively and avoids FITS column-type limitations.

Install the optional dependency:

```bash
uv sync --extra asdf
```

Usage:

```python
from isoster import isophote_results_to_asdf, isophote_results_from_asdf

# Save to ASDF (requires the `asdf` extra)
isophote_results_to_asdf(results, 'galaxy.asdf')

# Load from ASDF
loaded = isophote_results_from_asdf('galaxy.asdf')
```

### Astropy table export

- `isophote_results_to_astropy_tables`: returns the isophote list as a single `astropy.table.Table` for downstream analysis without writing to disk.

## Output Reference

`fit_image()` returns a result dict with two mandatory keys and optional metadata keys. Each entry in `results["isophotes"]` is a dict with the fields described below.

### Top-Level Result Dict

| Key | Type | Always Present | Description |
|-----|------|----------------|-------------|
| `isophotes` | list[dict] | Yes | Per-isophote results, sorted by ascending SMA |
| `config` | `IsosterConfig` | Yes | The configuration object used for the fit |
| `first_isophote_failure` | bool | Only when `True` | First N isophotes all failed (see [First Isophote Robustness](#first-isophote-robustness)) |
| `first_isophote_retry_log` | list[dict] | Only when retries ran | Detailed log of retry attempts when `max_retry_first_isophote > 0` |
| `lsb_auto_lock` | bool | Only when `lsb_auto_lock=True` | Echoes that automatic LSB geometry lock was used |
| `lsb_auto_lock_sma` | float or None | Only when `lsb_auto_lock=True` | SMA at which the lock committed; `None` if the detector never triggered |
| `lsb_auto_lock_anchor_sma` | float or None | Only when `lsb_auto_lock=True` | SMA of the true geometry anchor (the last clean isophote before the debounce streak, whose geometry the lock froze to); `None` if never committed |
| `lsb_auto_lock_count` | int | Only when `lsb_auto_lock=True` | Number of outward isophotes fit under locked geometry |
| `use_outer_center_regularization` | bool | Only when `use_outer_center_regularization=True` | Echoes that outer-region regularization was used |
| `outer_reg_x0_ref` | float | Only when `use_outer_center_regularization=True` | Frozen inner reference `x0` (flux-weighted mean over qualifying inward isophotes; falls back to anchor) |
| `outer_reg_y0_ref` | float | Only when `use_outer_center_regularization=True` | Frozen inner reference `y0` |
| `outer_reg_eps_ref` | float | Only when `use_outer_center_regularization=True` | Frozen inner reference ellipticity |
| `outer_reg_pa_ref` | float | Only when `use_outer_center_regularization=True` | Frozen inner reference PA |

### Per-Isophote Fields: Always Present

These fields are included in every isophote dict regardless of configuration.

| Field | Type | Description |
|-------|------|-------------|
| `sma` | float | Semi-major axis length (pixels) |
| `x0` | float | Center x-coordinate (pixels) |
| `y0` | float | Center y-coordinate (pixels) |
| `eps` | float | Ellipticity (0 ≤ eps < 1) |
| `pa` | float | Position angle (radians, 0 ≤ pa < π) |
| `intens` | float | Mean (or median) intensity along the ellipse |
| `rms` | float | RMS scatter of intensity residuals |
| `intens_err` | float | Intensity uncertainty (rms/√N, or WLS propagated error) |
| `x0_err` | float | Center x uncertainty (0.0 when `compute_errors=False`) |
| `y0_err` | float | Center y uncertainty (0.0 when `compute_errors=False`) |
| `eps_err` | float | Ellipticity uncertainty (0.0 when `compute_errors=False`) |
| `pa_err` | float | Position angle uncertainty (0.0 when `compute_errors=False`) |
| `tflux_e` | float | Total flux within elliptical aperture (NaN unless `full_photometry=True` or `debug=True`) |
| `tflux_c` | float | Total flux within circular aperture (NaN unless `full_photometry=True` or `debug=True`) |
| `npix_e` | int | Pixel count in elliptical aperture (0 unless `full_photometry=True` or `debug=True`) |
| `npix_c` | int | Pixel count in circular aperture (0 unless `full_photometry=True` or `debug=True`) |
| `stop_code` | int | Fitting termination status (see [Stop Codes](#stop-codes-canonical-reference)) |
| `niter` | int | Number of iterations performed |
| `use_eccentric_anomaly` | bool | Whether eccentric anomaly sampling was used |

The central pixel (sma=0) additionally includes:

| Field | Type | Description |
|-------|------|-------------|
| `valid` | bool | Whether the center pixel is unmasked and within image bounds |

### Per-Isophote Fields: Higher-Order Harmonics

Present when `compute_deviations=True` or `simultaneous_harmonics=True`. For each order *n* in `harmonic_orders` (default `[3, 4]`):

| Field | Type | Description |
|-------|------|-------------|
| `a{n}` | float | Sine harmonic coefficient (normalized by sma × \|gradient\|) |
| `b{n}` | float | Cosine harmonic coefficient (normalized by sma × \|gradient\|) |
| `a{n}_err` | float | Uncertainty in `a{n}` |
| `b{n}_err` | float | Uncertainty in `b{n}` |

With the default `harmonic_orders=[3, 4]`, this produces: `a3`, `b3`, `a3_err`, `b3_err`, `a4`, `b4`, `a4_err`, `b4_err`.

**Normalization convention.** Stored `a{n}`/`b{n}` are the raw harmonic amplitudes divided by `sma × |gradient|` (absolute value of the radial intensity gradient), making them dimensionless fractional-radius perturbations. This coincides with the Bender et al. form `-A_n/(a · dI/da)` whenever the gradient is negative (the normal case of an outward-declining profile); on rare positive-gradient rows (e.g. in the LSB regime) the stored values keep the absolute-value normalization and so differ from the Bender form in sign.

### Per-Isophote Fields: Curve of Growth

Present only when `compute_cog=True` (regular fitting mode only).

| Field | Type | Description |
|-------|------|-------------|
| `cog` | float | Cumulative flux from center to this SMA |
| `cog_annulus` | float | Flux in the annulus between previous and current SMA |
| `area_annulus` | float | Area of the annulus (corrected for negative areas) |
| `flag_cross` | bool | Ellipse crossing detected at this isophote |
| `flag_negative_area` | bool | Negative annular area (geometry divergence indicator) |

### Per-Isophote Fields: Automatic LSB Geometry Lock

Present on every isophote when `lsb_auto_lock=True`. Inward isophotes and the central pixel always carry `False` (the lock only engages during outward growth); the uniform schema keeps FITS round-trips exact.

| Field | Type | Description |
|-------|------|-------------|
| `lsb_locked` | bool | `True` if this isophote was fit under locked geometry (post-transition), `False` otherwise |
| `lsb_auto_lock_anchor` | bool | `True` on the **committing trigger** row — the first locked isophote, where the lock was committed — and `False` elsewhere. Despite the name this is *not* the isophote whose geometry was frozen: that one is identified by the top-level `lsb_auto_lock_anchor_sma`, and is the last clean isophote *before* the debounce streak. Useful as a marker in QA overlays |

### Per-Isophote Fields: Debug Diagnostics

Present only when `debug=True`. Enabling debug mode also populates the `tflux_*`/`npix_*` aperture photometry fields (equivalent to `full_photometry=True`).

| Field | Type | Description |
|-------|------|-------------|
| `ndata` | int | Number of valid (unmasked, unclipped) sample points |
| `nflag` | int | Number of flagged (masked or clipped) sample points |
| `grad` | float | Radial intensity gradient dI/da |
| `grad_error` | float | Gradient uncertainty |
| `grad_r_error` | float | Relative gradient error (grad_error / |grad|) |

### Config Flags and Output Control

| Config Flag | Effect on Output |
|-------------|-----------------|
| `compute_errors` | When `False`, `*_err` fields are set to 0.0 instead of computed values |
| `compute_deviations` | When `True`, adds `a{n}`, `b{n}`, `a{n}_err`, `b{n}_err` fields |
| `simultaneous_harmonics` | When `True`, also adds harmonic fields (fitted jointly during iteration) |
| `full_photometry` | When `True`, populates `tflux_e`, `tflux_c`, `npix_e`, `npix_c` with computed values |
| `compute_cog` | When `True`, adds CoG fields (regular fitting mode only) |
| `debug` | When `True`, adds diagnostic fields and implicitly enables `full_photometry` |
| `harmonic_orders` | Controls which harmonic orders produce `a{n}`/`b{n}` fields (default `[3, 4]`) |

### Filtering by Stop Code

```python
good = [iso for iso in results["isophotes"] if iso["stop_code"] == 0]
usable = [iso for iso in results["isophotes"] if iso["stop_code"] in {0, 1, 2}]
failed = [iso for iso in results["isophotes"] if iso["stop_code"] < 0]
```

## QA Figures and Benchmark Comparisons

Use `isoster.plot_qa_summary()` or
`isoster.plot_comparison_qa_figure()` when auditing a fit visually.
The public QA standard is documented in `docs/06-qa-functions.md`:
figures should use the same persisted profile/model artifacts that feed
metrics, show image/model/residual context, keep profile-axis
conventions consistent, and display stop-code or quality-state markers
where available.

For cross-arm or cross-tool campaigns, use the exhausted benchmark
guide in `docs/09-exhausted-benchmark.md`. The benchmark separates
within-tool `composite_score` from cross-tool `cross_tool_score`:
within-tool rankings may include method-specific health terms, while
cross-tool rankings use tool-neutral residual and runtime terms.

Surface-brightness QA defaults to `log10`/magnitude behavior. Use
`sb_profile_scale="asinh"` when low-S/N outskirts, zero crossings, or
negative-intensity samples need to remain visible. With a photometric
zeropoint and pixel scale, the calibrated asinh profile matches the
log10 magnitude profile at high S/N and shows a dashed `I = 0`
reference line.

## Troubleshooting

- Too many `stop_code=1`: inspect masks and clipping settings (`fflag`, `sclip`, `nclip`).
- Many `stop_code=-1` outward: reduce `maxsma` or relax `maxgerr` after validation.
- Many `stop_code=3`: increase `minsma` and inspect masked/edge regions.
- Unstable high-ellipticity fits: enable `use_eccentric_anomaly=True`.

## Stop Codes (Canonical Reference)

This section is the canonical stop-code reference for the current `isoster` implementation.

### Code Table

| Code | Current Meaning | Primary Trigger in Core Code | Typical Action |
|---|---|---|---|
| `0` | Success | Converged fit, or successful forced extraction | Keep |
| `1` | Too many flagged samples | `actual_points < total_points * (1.0 - fflag)` | Inspect mask/clipping; treat cautiously |
| `2` | Max-iteration fallback | Reached `maxit` without convergence criterion | Keep with caution; geometry is best-so-far |
| `3` | Too few points | `< 6` valid points for harmonic fit | Discard this radius |
| `-1` | Gradient failure | Gradient checks fail (or zero gradient) | Treat as boundary/failure |

### Notes by Path

- Regular mode (`fit_isophote`) emits `0`, `1`, `2`, `3`, `-1`.
- In regular `fit_image` growth, stop codes `0`, `1`, and `2` are treated as acceptable for outward/inward propagation.
- `fit_central_pixel` emits `0` for unmasked center and `-1` for masked center.
- Forced extraction (`extract_forced_photometry`) emits `0` or `3`.

### Gradient-Failure Detail (`-1`)

In outward fitting (`going_inwards=False`), gradient quality checks are applied with a two-strike rule:

- first exceedance sets an internal flag (`lexceed=True`),
- second exceedance sets stop code `-1`.

A direct `gradient == 0` also produces `-1`.

### Minimal Filtering Patterns

```python
good = [iso for iso in results["isophotes"] if iso["stop_code"] == 0]
usable = [iso for iso in results["isophotes"] if iso["stop_code"] in {0, 1, 2}]
failed = [iso for iso in results["isophotes"] if iso["stop_code"] < 0]
```

## Related Docs

- `docs/04-architecture.md`
- `docs/03-algorithm.md`
- `docs/06-qa-functions.md`
- `docs/09-exhausted-benchmark.md`
