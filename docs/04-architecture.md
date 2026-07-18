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

### Experimental Multi-Band Public Interface

Lives under a parallel module tree at `isoster/multiband/`. The single-
band interfaces above are **not** modified.

- `isoster.multiband.fit_image_multiband(images, masks=None, config=None,
  variance_maps=None, template_isophotes=None)` — joint free fit on
  aligned same-pixel-grid images (or forced photometry when a template is
  given). One shared geometry per SMA, per-band intensities and per-band
  harmonic deviations. Replaces forced photometry as the multi-band
  workflow.
- `isoster.multiband.IsosterConfigMB` — multi-band-specific config
  (sibling of `IsosterConfig`, no inheritance, deliberately reduced field
  set). The driver validates that all inputs share one pixel grid
  (shape-only check; no WCS validation is performed).
- `isoster.multiband.load_bands_from_hdus(hdus)` — helper to extract
  `(images, masks, variance_maps, bands)` tuples from FITS HDUs.

`fit_image_multiband` with `len(bands) == 1` delegates to `fit_image` and
returns the legacy single-band schema unmodified.

Status: experimental (beta) — the API and the Schema-1 output layout may
change between releases. The CLI (`isoster-mb`), FITS/ASDF I/O,
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
- `isoster/multiband/` (Stage-1 experimental): parallel module tree for
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
    Schema 1 columns ``a<n>_<b>`` / ``b<n>_<b>`` (and ``_err``)
    continue to be written but carry the identical shared value
    across bands under non-``'independent'`` modes; D16 per-band
    Bender normalization at plotting time still produces band-distinct
    curves because per-band gradients differ. Section 6 of the internal
    multi-band design record (not part of the public docs tree).

## Key Constants

- `ACCEPTABLE_STOP_CODES = {0, 1, 2}` (`driver.py`): stop codes considered acceptable for continued fitting (outward/inward growth). Used by `_is_acceptable_stop_code()` to gate whether the next SMA step proceeds.

## Runtime Modes

`fit_image` selects exactly one mode in this priority order:

1. `template` provided -> template-based forced photometry (`_fit_image_template_forced`).
2. Otherwise -> regular iterative fitting (central pixel, anchor isophote, optional inward growth, outward growth).

(`template_isophotes` is supported as a deprecated alias for `template`).

## Regular Fitting Contract

For each SMA in regular mode:

1. Sample ellipse points (`sampling.extract_isophote_data`) with `n_samples = max(64, int(2*pi*sma))`.
2. Apply sigma clipping to sampled `(angle, intensity)` pairs.
3. Fit harmonics:
   - **Default path** (`simultaneous_harmonics=False`): Fit 5-param model (`I0`, `A1`, `B1`, `A2`, `B2`) via `fit_first_and_second_harmonics()`. Higher-order harmonics fitted post-hoc after convergence.
   - **ISOFIT path** (`simultaneous_harmonics=True`): Fit all harmonics simultaneously via `fit_all_harmonics()` using an extended design matrix `[1, sin(θ), cos(θ), sin(2θ), cos(2θ), sin(n₁θ), cos(n₁θ), ...]`. Falls back to 5-param when `n_points < 1 + 2*(2 + len(orders))`. Geometry updates use `A1, B1, A2, B2 = coeffs[1:5]` identically in both paths.
   - **WLS mode** (`variance_map` provided to `fit_image`): All harmonic fits use Weighted Least Squares with `w_i = 1/σ²_i`. The covariance matrix `(A^T W A)^-1` is exact — no residual-variance scaling is needed. This cleanly separates photon noise from galaxy structure scatter and automatically down-weights high-variance pixels (cosmic rays, hot pixels). When `variance_map=None`, the OLS path is byte-identical to the non-WLS code.
4. Estimate radial gradient (`fitting.compute_gradient`). When `variance_map` is provided, gradient error uses exact per-sample variance (`Var(mean) = Σσ²_i / N²`) instead of scatter-based estimates.
5. Update geometry based on dominant harmonic coefficient.
6. Check convergence criterion: `abs(max_amp) < conver * rms` with iteration index check `i >= minit`.

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

`isophote_results_to_fits` writes a 3-HDU FITS file:

| HDU | Type | Name | Contents |
|-----|------|------|----------|
| 0 | `PrimaryHDU` | — | Empty (no data, minimal header) |
| 1 | `BinTableHDU` | `ISOPHOTES` | One row per isophote; columns match the isophote dict keys |
| 2 | `BinTableHDU` | `CONFIG` | Two columns: `PARAM` (string) and `VALUE` (JSON-serialized string), one row per config field |

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

## Documentation Policy

- Stable docs live in `docs/` root.
- Historical records live under `docs/journal/`; internal planning and review archives are kept under `docs/agent/` (not tracked in the public repository).
- Use lowercase kebab-case markdown filenames.
