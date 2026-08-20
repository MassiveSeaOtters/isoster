"""
Multi-band isoster driver (``fit_image_multiband``).

Single public entry point for joint multi-band free fitting. Mirrors the
single-band :func:`isoster.driver.fit_image` orchestration:

- Central pixel record at SMA = 0 (when ``minsma <= 0``).
- First-isophote retry schedule.
- Inward growth from the anchor down to ``minsma``.
- Outward growth from the anchor out to ``maxsma``.
- Final result list assembled in ascending-SMA order.

LSB auto-lock and outer-center regularization are ported from the
single-band driver with multi-band semantics: the lock trigger surface
is the joint combined gradient, the lock anchor is the converged shared
geometry, and the outer-reference geometry is a flux-weighted mean over
inward isophotes (weights = reference-band intensity, circular mean on
``2*pa`` for the PA; see ``docs/10-multiband.md``).
B=1 input delegates to the existing :func:`isoster.driver.fit_image`
and returns the legacy single-band schema unmodified (decision D14).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ..config import IsosterConfig
from ..driver import fit_image
from .config_mb import IsosterConfigMB
from .fitting_mb import (
    _PER_BAND_DEBUG_KEYS,
    extract_forced_photometry_mb,
    fit_isophote_mb,
)
from .sampling_mb import prepare_inputs

ACCEPTABLE_STOP_CODES = {0, 1, 2}

# Stage-3 Stage-H: forced-photometry workflow. Mirrors single-band's
# _TEMPLATE_REQUIRED_KEYS — geometry is what we copy from the template,
# everything else (per-band intensities, harmonics) gets re-extracted
# from the multi-band image stack. Per-band ``intens_<b>`` columns may
# be present in the template (e.g. when the template is a multi-band
# result dict) but they are not required — Stage H accepts single-band
# templates too.
_TEMPLATE_REQUIRED_KEYS_MB = {"sma", "x0", "y0", "eps", "pa"}


# ---------------------------------------------------------------------------
# Lightweight helpers
# ---------------------------------------------------------------------------


def _is_acceptable(iso: Dict[str, object]) -> bool:
    return int(iso.get("stop_code", -1)) in ACCEPTABLE_STOP_CODES  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Stage-3 Stage-C: LSB auto-lock helpers (port of isoster.driver._is_lsb_isophote
# / _mark_lsb_lock_state / _build_locked_cfg, adapted to multi-band)
# ---------------------------------------------------------------------------


def _is_lsb_isophote_mb(iso: Dict[str, object], maxgerr_thresh: float) -> bool:
    """Lock-trigger predicate for the multi-band outward sweep.

    Plan section 7 S3: the trigger surface is the **joint combined
    gradient** (``grad_joint`` / ``grad_err_joint``), exposed by the
    fitter as top-level ``grad`` / ``grad_error`` row keys when
    ``debug=True``. A relative joint gradient error above
    ``maxgerr_thresh``, OR a stop_code=-1 (gradient-error stop), OR a
    non-negative joint gradient (galaxies have negative dI/da; positive
    values mean the fit has lost the profile) all count as triggering.

    None / non-finite values mean "cannot assess" and return False —
    the next isophote will get another chance to trigger.
    """
    if iso.get("stop_code") == -1:
        return True
    grad = iso.get("grad")
    grad_err = iso.get("grad_error")
    if grad is None or grad_err is None:
        return False
    try:
        grad_f = float(grad)
        grad_err_f = float(grad_err)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(grad_f) or not np.isfinite(grad_err_f) or grad_err_f == 0.0:
        return False
    if grad_f >= 0.0:
        return True
    return abs(grad_err_f / grad_f) > maxgerr_thresh


def _mark_lsb_lock_state_mb(
    iso: Dict[str, object],
    locked: bool,
    is_anchor: bool = False,
) -> None:
    """Stamp ``lsb_locked`` and ``lsb_auto_lock_anchor`` onto a multi-band
    result row. Always writes both keys so the per-isophote schema stays
    uniform and FITS round-trips cannot corrupt the flags (review B3).
    Centralized so future loop reorders cannot accidentally write the
    wrong flag."""
    iso["lsb_locked"] = bool(locked)
    iso["lsb_auto_lock_anchor"] = bool(is_anchor)


def _build_locked_cfg_mb(
    cfg: IsosterConfigMB,
    anchor_iso: Dict[str, object],
    locked_integrator: str,
) -> IsosterConfigMB:
    """Clone the multi-band config and freeze geometry to a clean anchor.

    Mirrors :func:`isoster.driver._build_locked_cfg` with two multi-
    band-specific tweaks:

    1. When ``locked_integrator='median'``, also flip
       ``fit_per_band_intens_jointly=False`` on the clone so Stage A's
       S1 validator accepts the result. The config-time validator
       already rejects ``lsb_auto_lock_integrator='median'`` ∧
       ``fit_per_band_intens_jointly=True`` (S1 × S3 composition);
       this branch makes the lock-fire path consistent regardless of
       what the user originally set.
    2. Disable outer-region damping on the clone — it makes no sense
       once geometry is hard-locked, and would consume the outer
       reference geom for nothing.
    """
    locked = cfg.model_copy(deep=True)
    locked.x0 = float(anchor_iso["x0"])  # type: ignore[arg-type]
    locked.y0 = float(anchor_iso["y0"])  # type: ignore[arg-type]
    locked.eps = float(anchor_iso["eps"])  # type: ignore[arg-type]
    locked.pa = float(anchor_iso["pa"])  # type: ignore[arg-type]
    locked.fix_center = True
    locked.fix_pa = True
    locked.fix_eps = True
    locked.integrator = locked_integrator  # type: ignore[assignment]
    if locked_integrator == "median":
        # Median requires the decoupled intercept mode (Stage A S1).
        locked.fit_per_band_intens_jointly = False
    # The auto-lock has already committed; the clone should not try to
    # re-detect on its own isophotes.
    locked.lsb_auto_lock = False
    # Outer-region damping is meaningless once geometry is frozen.
    locked.use_outer_center_regularization = False
    return locked


def _resolve_template_mb(template: object) -> List[Dict[str, object]]:
    """Normalize a template input into a validated, SMA-sorted list.

    Stage-3 Stage-H. Accepted input forms:

    - ``str`` or ``Path``: path to a Schema-1 multi-band FITS file
      OR a single-band FITS file. Single-band templates work because
      we only need ``sma`` / ``x0`` / ``y0`` / ``eps`` / ``pa`` —
      per-band ``intens_<b>`` columns get re-extracted from the
      multi-band image stack regardless.
    - ``dict`` with an ``'isophotes'`` key: a results dict (multi-band
      or single-band).
    - ``list`` of dicts: isophote dicts directly.

    Raises:
        TypeError: input not a recognized type.
        ValueError: template empty or any dict is missing required keys.
    """
    from pathlib import Path as _Path

    if isinstance(template, (str, _Path)):
        path_str = str(template)
        # Try multi-band first (most common Stage-H workflow); fall
        # back to single-band on the typical "MULTIBND" header miss.
        try:
            from .utils_mb import isophote_results_mb_from_fits

            loaded = isophote_results_mb_from_fits(path_str)
        except Exception:
            from ..utils import isophote_results_from_fits

            loaded = isophote_results_from_fits(path_str)
        iso_list = loaded["isophotes"]
    elif isinstance(template, dict):
        if "isophotes" not in template:
            raise ValueError(f"template dict must contain an 'isophotes' key; got keys: {sorted(template.keys())}")
        iso_list = template["isophotes"]
    elif isinstance(template, list):
        iso_list = template
    else:
        raise TypeError(
            f"template_isophotes must be a file path (str / Path), "
            f"results dict, or list of isophote dicts; got "
            f"{type(template).__name__}"
        )

    if not iso_list:
        raise ValueError("template_isophotes cannot be empty")

    for i, iso in enumerate(iso_list):
        missing = _TEMPLATE_REQUIRED_KEYS_MB - set(iso.keys())
        if missing:
            raise ValueError(
                f"template isophote at index {i} missing required keys: "
                f"{sorted(missing)}. Required: "
                f"{sorted(_TEMPLATE_REQUIRED_KEYS_MB)}."
            )

    return sorted(iso_list, key=lambda x: float(x["sma"]))


def _build_outer_reference_mb(
    inwards_results: Sequence[Dict[str, object]],
    anchor_iso: Dict[str, object],
    config: IsosterConfigMB,
) -> Dict[str, float]:
    """Frozen reference geometry for the multi-band outer-region damping.

    Mirrors :func:`isoster.driver._build_outer_reference`: takes a flux-
    weighted mean of inward isophotes (plus the anchor) within a sma
    window of ``sma0 * config.outer_reg_ref_sma_factor``. Multi-band
    geometry is shared, so only the flux weight differs from single-band:
    we use ``intens_<reference_band>`` since the geometry was driven
    against that band's diagnostics. The reference carries x0/y0/eps/pa;
    pa uses a flux-weighted circular mean on 2*pa (axis-like angles).
    """
    ref_band = config.reference_band
    intens_key = f"intens_{ref_band}"

    def _intens(iso: Dict[str, object]) -> Optional[float]:
        val = iso.get(intens_key)
        if val is None:
            return None
        v = float(val)
        return v if np.isfinite(v) else None

    anchor_ref = {
        "x0": float(anchor_iso["x0"]),  # type: ignore[arg-type]
        "y0": float(anchor_iso["y0"]),  # type: ignore[arg-type]
        "eps": float(anchor_iso["eps"]),  # type: ignore[arg-type]
        "pa": float(anchor_iso["pa"]),  # type: ignore[arg-type]
    }

    sma_hi = float(anchor_iso["sma"]) * config.outer_reg_ref_sma_factor  # type: ignore[arg-type]

    candidates: List[Dict[str, object]] = [anchor_iso]
    for iso in inwards_results:
        if not _is_acceptable(iso):
            continue
        sma_v = iso.get("sma")
        if sma_v is None or not np.isfinite(float(sma_v)):  # type: ignore[arg-type]
            continue
        if float(sma_v) > sma_hi:  # type: ignore[arg-type]
            continue
        intens_v = _intens(iso)
        if intens_v is None:
            continue
        for k in ("x0", "y0", "eps", "pa"):
            v = iso.get(k)
            if v is None or not np.isfinite(float(v)):  # type: ignore[arg-type]
                break
        else:
            candidates.append(iso)

    x0s = np.array([float(iso["x0"]) for iso in candidates])  # type: ignore[arg-type]
    y0s = np.array([float(iso["y0"]) for iso in candidates])  # type: ignore[arg-type]
    eps_arr = np.array([float(iso["eps"]) for iso in candidates])  # type: ignore[arg-type]
    pa_arr = np.array([float(iso["pa"]) for iso in candidates])  # type: ignore[arg-type]
    weights = np.array(
        [max(_intens(iso) or 1e-6, 1e-6) for iso in candidates],
        dtype=np.float64,
    )
    if weights.sum() <= 0.0 or not np.all(np.isfinite(weights)):
        return anchor_ref

    x0_ref = float(np.average(x0s, weights=weights))
    y0_ref = float(np.average(y0s, weights=weights))
    eps_ref = float(np.average(eps_arr, weights=weights))

    cos_sum = float(np.sum(weights * np.cos(2.0 * pa_arr)))
    sin_sum = float(np.sum(weights * np.sin(2.0 * pa_arr)))
    w_sum = float(weights.sum())
    resultant = np.hypot(cos_sum, sin_sum) / w_sum if w_sum > 0.0 else 0.0
    if resultant >= 0.1:
        pa_ref = 0.5 * float(np.arctan2(sin_sum, cos_sum))
        pa_ref = pa_ref % np.pi
    else:
        pa_ref = float(anchor_iso["pa"]) % np.pi  # type: ignore[arg-type]

    if not all(np.isfinite([x0_ref, y0_ref, eps_ref, pa_ref])):
        return anchor_ref
    return {"x0": x0_ref, "y0": y0_ref, "eps": eps_ref, "pa": pa_ref}


def _validate_inputs(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]],
    config: IsosterConfigMB,
) -> None:
    """Cross-check shapes and band counts before any sampling happens."""
    if len(images) != len(config.bands):
        raise ValueError(
            f"len(images) ({len(images)}) does not match len(config.bands) "
            f"({len(config.bands)}). Each image must correspond to a band name."
        )
    h, w = images[0].shape
    for i, im in enumerate(images):
        if im.shape != (h, w):
            raise ValueError(f"images[{i}] shape {im.shape} does not match images[0] shape {(h, w)}.")
    # Mask shape consistency is delegated to the sampler. Variance map
    # all-or-nothing semantics likewise. We validate band counts here so
    # the user gets a clear error before any expensive call.
    n_bands = len(config.bands)
    if variance_maps is not None and not isinstance(variance_maps, np.ndarray):
        try:
            n_var = len(variance_maps)
        except TypeError as exc:
            raise TypeError(
                "variance_maps must be None, a single ndarray, or a sequence "
                f"of length {n_bands}; got non-sequence {type(variance_maps).__name__}."
            ) from exc
        if n_var != n_bands:
            raise ValueError(f"len(variance_maps) ({n_var}) does not match len(config.bands) ({n_bands}).")
    if masks is not None and not isinstance(masks, np.ndarray):
        try:
            n_masks = len(masks)
        except TypeError as exc:
            raise TypeError(
                "masks must be None, a single boolean ndarray, or a sequence "
                f"of length {n_bands}; got non-sequence {type(masks).__name__}."
            ) from exc
        if n_masks != n_bands:
            raise ValueError(f"len(masks) ({n_masks}) does not match len(config.bands) ({n_bands}).")


def _delegate_single_band(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]],
    config_mb: IsosterConfigMB,
    template_isophotes: object = None,
) -> Dict[str, object]:
    """Decision D14: B=1 input delegates to single-band fit_image."""
    warnings.warn(
        "Single-band input (len(bands)==1): delegating to "
        "isoster.fit_image. The legacy single-band schema is produced "
        "(no _<band> column suffixes, no multi-band top-level keys).",
        UserWarning,
        stacklevel=3,
    )
    image = np.asarray(images[0])
    if isinstance(masks, np.ndarray):
        mask = masks
    elif masks is None:
        mask = None
    else:
        mask = masks[0]
    if isinstance(variance_maps, np.ndarray):
        var = variance_maps
    elif variance_maps is None:
        var = None
    else:
        var = variance_maps[0]

    # Build a single-band IsosterConfig that mirrors the multi-band knobs
    # we share. Multi-band-only fields are dropped; ISOFIT/LSB-lock
    # defaults stay off.
    sb_kwargs = dict(
        x0=config_mb.x0,
        y0=config_mb.y0,
        eps=config_mb.eps,
        pa=config_mb.pa,
        sma0=config_mb.sma0,
        minsma=config_mb.minsma,
        maxsma=config_mb.maxsma,
        astep=config_mb.astep,
        linear_growth=config_mb.linear_growth,
        maxit=config_mb.maxit,
        minit=config_mb.minit,
        conver=config_mb.conver,
        convergence_scaling=config_mb.convergence_scaling,
        sigma_bg=config_mb.sigma_bg,
        use_corrected_errors=config_mb.use_corrected_errors,
        use_lazy_gradient=config_mb.use_lazy_gradient,
        geometry_damping=config_mb.geometry_damping,
        clip_max_shift=config_mb.clip_max_shift,
        clip_max_pa=config_mb.clip_max_pa,
        clip_max_eps=config_mb.clip_max_eps,
        geometry_convergence=config_mb.geometry_convergence,
        geometry_tolerance=config_mb.geometry_tolerance,
        geometry_stable_iters=config_mb.geometry_stable_iters,
        geometry_update_mode=config_mb.geometry_update_mode,
        sclip=config_mb.sclip,
        nclip=config_mb.nclip,
        sclip_low=config_mb.sclip_low,
        sclip_high=config_mb.sclip_high,
        fflag=config_mb.fflag,
        maxgerr=config_mb.maxgerr,
        fix_center=config_mb.fix_center,
        fix_pa=config_mb.fix_pa,
        fix_eps=config_mb.fix_eps,
        compute_errors=config_mb.compute_errors,
        compute_deviations=config_mb.compute_deviations,
        full_photometry=config_mb.full_photometry,
        debug=config_mb.debug,
        integrator=config_mb.integrator,
        use_eccentric_anomaly=config_mb.use_eccentric_anomaly,
        permissive_geometry=config_mb.permissive_geometry,
        max_retry_first_isophote=config_mb.max_retry_first_isophote,
        first_isophote_fail_count=config_mb.first_isophote_fail_count,
        # Feature families shared by both configs (M6: these were silently
        # dropped on delegation, e.g. a B=1 run with compute_cog=True or
        # lsb_auto_lock=True produced a fit without those features).
        compute_cog=config_mb.compute_cog,
        harmonic_orders=config_mb.harmonic_orders,
        lsb_auto_lock=config_mb.lsb_auto_lock,
        lsb_auto_lock_maxgerr=config_mb.lsb_auto_lock_maxgerr,
        lsb_auto_lock_debounce=config_mb.lsb_auto_lock_debounce,
        lsb_auto_lock_integrator=config_mb.lsb_auto_lock_integrator,
        use_outer_center_regularization=config_mb.use_outer_center_regularization,
        outer_reg_strength=config_mb.outer_reg_strength,
        outer_reg_weights=config_mb.outer_reg_weights,
        outer_reg_mode=config_mb.outer_reg_mode,
        outer_reg_sma_onset=config_mb.outer_reg_sma_onset,
        outer_reg_sma_width=config_mb.outer_reg_sma_width,
        outer_reg_ref_sma_factor=config_mb.outer_reg_ref_sma_factor,
        use_central_regularization=config_mb.use_central_regularization,
        central_reg_strength=config_mb.central_reg_strength,
        central_reg_weights=config_mb.central_reg_weights,
        central_reg_sma_threshold=config_mb.central_reg_sma_threshold,
    )
    cfg_sb = IsosterConfig(**sb_kwargs)
    return fit_image(image, mask=mask, config=cfg_sb, variance_map=var, template_isophotes=template_isophotes)


def _first_isophote_perturbations(
    sma0: float,
    eps: float,
    pa: float,
    max_retries: int,
) -> List[tuple]:
    """Mirror :func:`isoster.driver._first_isophote_perturbations`."""
    schedule = [
        (0.8, eps, pa),
        (1.3, eps, pa),
        (0.6, 0.05, pa),
        (1.5, eps, pa + np.pi / 4),
        (0.5, 0.05, pa + np.pi / 2),
    ]
    extended_factors = [0.4, 0.7, 1.1, 1.6, 2.0]
    for i, factor in enumerate(extended_factors):
        ext_eps = 0.05 if i % 2 == 0 else eps
        ext_pa = pa + (i + 1) * np.pi / 6
        schedule.append((factor, ext_eps, ext_pa))
    out: List[tuple] = []
    for i in range(min(max_retries, len(schedule))):
        sma_factor, eps_new, pa_new = schedule[i]
        sma_new = max(1.0, sma0 * sma_factor)
        eps_new = min(max(0.0, eps_new), 0.99)
        out.append((sma_new, eps_new, pa_new))
    return out


def _validate_non_negative_error_fields(isophotes: Sequence[Dict[str, object]]) -> None:
    """Mirror the single-band validator: any negative error => hard error."""
    for idx, iso in enumerate(isophotes):
        for field, value in iso.items():
            if not (field.endswith("_err") or field.endswith("_error")):
                continue
            try:
                v = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v < 0.0:
                raise ValueError(
                    f"multiband isoster produced negative error value (field={field}, index={idx}, value={v})"
                )


# ---------------------------------------------------------------------------
# Central pixel (multi-band)
# ---------------------------------------------------------------------------


def _fit_central_pixel_mb(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    x0: float,
    y0: float,
    config: IsosterConfigMB,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
) -> Dict[str, object]:
    """Per-band central-pixel record at SMA=0.

    When ``variance_maps`` is provided (WLS mode), ``intens_err_<b>`` is
    set to ``sqrt(variance_maps_b[iy, ix])`` (review fix H3). In OLS
    mode the central pixel still reports ``intens_err_<b>=0.0`` because
    a single sample admits no internal error estimate. ``rms_<b>``
    remains ``0.0`` in both modes — RMS is the ring-residual scatter
    around the harmonic model and is undefined for a single pixel.
    """
    bands = list(config.bands)
    debug = bool(config.debug)
    h, w = np.asarray(images[0]).shape
    iy = int(np.round(y0))
    ix = int(np.round(x0))
    in_bounds = 0 <= iy < h and 0 <= ix < w

    row: Dict[str, object] = {
        "sma": 0.0,
        "x0": x0,
        "y0": y0,
        "eps": 0.0,
        "pa": 0.0,
        "x0_err": 0.0,
        "y0_err": 0.0,
        "eps_err": 0.0,
        "pa_err": 0.0,
        "rms": 0.0,
        "stop_code": 0 if in_bounds else -1,
        "niter": 0,
        "valid": in_bounds,
        "use_eccentric_anomaly": config.use_eccentric_anomaly,
        "tflux_e": float("nan"),
        "tflux_c": float("nan"),
        "npix_e": 0,
        "npix_c": 0,
        # Unconditional like every other row (M13: mixed presence produced
        # masked FITS cells); the central pixel is one sample, not a ring.
        "ndata": 1 if in_bounds else 0,
        "nflag": 0,
    }
    for b_idx, b in enumerate(bands):
        if in_bounds:
            mask_b = None
            if isinstance(masks, np.ndarray):
                mask_b = masks
            elif masks is not None:
                mask_b = masks[b_idx]
            val = float(np.asarray(images[b_idx])[iy, ix])
            valid_b = True
            if mask_b is not None and bool(mask_b[iy, ix]):
                valid_b = False
                val = float("nan")
        else:
            val = float("nan")
            valid_b = False
        row[f"intens_{b}"] = val if valid_b else float("nan")

        # Review fix H3: under WLS, the per-pixel variance map gives an
        # honest standard error at the central pixel. Under OLS (no
        # variance map) keep the legacy 0.0 — a single sample admits no
        # internal error estimate.
        intens_err_b = 0.0
        if valid_b and variance_maps is not None:
            if isinstance(variance_maps, np.ndarray):
                # Single (H, W) ndarray broadcast to all bands.
                var_b = variance_maps
            else:
                var_b = variance_maps[b_idx]
            try:
                v = float(np.asarray(var_b)[iy, ix])
                if np.isfinite(v) and v > 0.0:
                    intens_err_b = float(np.sqrt(v))
            except (IndexError, TypeError):
                intens_err_b = 0.0
        row[f"intens_err_{b}"] = intens_err_b
        row[f"rms_{b}"] = 0.0
        for n_order in config.harmonic_orders:
            for prefix in ("a", "b"):
                row[f"{prefix}{int(n_order)}_{b}"] = 0.0
                row[f"{prefix}{int(n_order)}_err_{b}"] = 0.0
        if debug:
            for key in _PER_BAND_DEBUG_KEYS:
                row[f"{key}_{b}"] = float("nan")
        # D9 backport: per-band surviving-sample count. The central pixel
        # is a single point, not a ring, so a value of 1 (or 0 if masked)
        # is the most honest report.
        row[f"n_valid_{b}"] = 1 if valid_b else 0
    return row


def _fit_image_template_forced_mb(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    config: IsosterConfigMB,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]],
    template: object,
) -> Dict[str, object]:
    """Stage-3 Stage-H: walk a template isophote list and run forced
    multi-band extraction at each row's geometry.

    Mirrors single-band ``isoster.driver._fit_image_template_forced``.
    Per-row dispatch:

    - ``sma == 0`` → :func:`_fit_central_pixel_mb` (per-band central
      record).
    - otherwise → :func:`isoster.multiband.fitting_mb.extract_forced_photometry_mb`
      (per-band ring extraction at the template's exact geometry).

    Validators (warn-and-ignore):

    - ``lsb_auto_lock=True`` — meaningless under forced extraction
      (no outward sweep that could trigger the lock).
    - ``use_outer_center_regularization=True`` — meaningless (no
      iteration loop that consumes the damping term).
    - ``use_central_regularization=True`` — meaningless (no
      iteration loop that consumes the selector penalty).
    - ``harmonic_combination='ref'`` — meaningless (no joint solve
      to bypass; geometry is fixed).
    - ``loose_validity=True`` — meaningless: forced extraction
      always uses shared-validity sampling (review fix B2).
    - ``multiband_higher_harmonics ∈ {shared, simultaneous_in_loop,
      simultaneous_original}`` — meaningless (review fix B2): the
      forced post-hoc harmonic step runs only the per-band
      independent path; non-independent modes need the iteration-
      loop joint solve that the forced workflow bypasses.

    The forced path otherwise composes cleanly with all other
    Stage-3 features. ``compute_cog=True`` is the primary use case
    on top of forced extraction (per-band CoG against a fixed
    geometry).
    """
    template_list = _resolve_template_mb(template)

    # Warn-and-ignore validators. The features are no-ops in forced
    # mode but their being on indicates the user may have a config
    # mistake; surface it via a clear UserWarning.
    forced_noop_features = []
    if config.lsb_auto_lock:
        forced_noop_features.append("lsb_auto_lock")
    if config.use_outer_center_regularization:
        forced_noop_features.append("use_outer_center_regularization")
    if config.use_central_regularization:
        forced_noop_features.append("use_central_regularization")
    if config.harmonic_combination == "ref":
        forced_noop_features.append("harmonic_combination='ref'")
    # ``loose_validity`` used to belong here: forced extraction always ran
    # shared-validity sampling and silently ignored the flag. It is now honoured
    # end to end — per-band sampling, per-band clipping, per-band statistics and
    # jagged harmonics — so the warning would be false.
    if config.multiband_higher_harmonics != "independent":
        forced_noop_features.append(f"multiband_higher_harmonics='{config.multiband_higher_harmonics}'")
    if forced_noop_features:
        warnings.warn(
            "template_isophotes=<provided> bypasses the iteration loop, "
            "so the following config features have no effect in forced "
            f"mode and are silently ignored: {', '.join(forced_noop_features)}.",
            UserWarning,
            stacklevel=3,
        )

    bands = list(config.bands)

    # Stage-H.1: capture per-row raw ring data so we can compute per-
    # band gradients post-hoc (via np.gradient on each band's intens
    # column) and fill harmonic deviations using
    # _attach_per_band_harmonics. Mirrors the iteration loop's per-band
    # harmonic path.
    do_harmonics = bool(config.compute_deviations and len(config.harmonic_orders) > 0)

    isophotes: List[Dict[str, object]] = []
    ring_data_per_row: List[object] = []  # MultiIsophoteData | None per row
    for template_iso in template_list:
        sma = float(template_iso["sma"])  # type: ignore[arg-type]
        if sma == 0.0:
            iso = _fit_central_pixel_mb(
                images,
                masks,
                float(template_iso["x0"]),
                float(template_iso["y0"]),  # type: ignore[arg-type]
                config,
                variance_maps=variance_maps,
            )
            ring_data_per_row.append(None)  # central pixel: no ring
        else:
            if do_harmonics:
                iso, ring = extract_forced_photometry_mb(
                    images=images,
                    masks=masks,
                    x0=float(template_iso["x0"]),  # type: ignore[arg-type]
                    y0=float(template_iso["y0"]),  # type: ignore[arg-type]
                    sma=sma,
                    eps=float(template_iso["eps"]),  # type: ignore[arg-type]
                    pa=float(template_iso["pa"]),  # type: ignore[arg-type]
                    bands=bands,
                    config=config,
                    variance_maps=variance_maps,
                    return_ring_data=True,
                )
                ring_data_per_row.append(ring)
            else:
                iso = extract_forced_photometry_mb(
                    images=images,
                    masks=masks,
                    x0=float(template_iso["x0"]),  # type: ignore[arg-type]
                    y0=float(template_iso["y0"]),  # type: ignore[arg-type]
                    sma=sma,
                    eps=float(template_iso["eps"]),  # type: ignore[arg-type]
                    pa=float(template_iso["pa"]),  # type: ignore[arg-type]
                    bands=bands,
                    config=config,
                    variance_maps=variance_maps,
                )
                ring_data_per_row.append(None)
        isophotes.append(iso)

    # Stage-H.1 second pass: compute per-band gradients via
    # np.gradient(intens_col, sma_col), then overwrite the zero-filled
    # harmonic columns with real values via _attach_per_band_harmonics.
    # Skip rows whose ring_data is None (central pixel or empty
    # extraction) — those keep the zero-filled defaults.
    if do_harmonics:
        from .fitting_mb import _attach_per_band_harmonics

        sma_col = np.array(
            [float(iso["sma"]) for iso in isophotes],  # type: ignore[arg-type]
            dtype=np.float64,
        )
        per_band_grad_col: Dict[str, NDArray[np.float64]] = {}
        for b in bands:
            intens_col = np.array(
                [float(iso.get(f"intens_{b}", float("nan"))) for iso in isophotes],
                dtype=np.float64,
            )
            # np.gradient requires at least 2 points and a strictly
            # monotonic x-axis. The template is sma-sorted by
            # _resolve_template_mb so monotonicity is fine. Non-finite
            # values fall through as NaN gradients — _attach_per_band_
            # harmonics handles that as zero-output.
            if intens_col.size >= 2 and np.all(np.isfinite(intens_col)):
                per_band_grad_col[b] = np.gradient(intens_col, sma_col)
            else:
                # Fallback: zero gradient → harmonics return zero (the
                # factor = sma * abs(grad) gate in compute_deviations
                # short-circuits cleanly).
                per_band_grad_col[b] = np.zeros_like(intens_col)

        for i, (iso, ring) in enumerate(zip(isophotes, ring_data_per_row)):
            if ring is None:
                continue  # central pixel or empty ring — keep zeros
            per_band_grad = [float(per_band_grad_col[b][i]) for b in bands]
            # Review fix B3: write per-band gradients into the row so
            # downstream plotting Bender normalization
            # (A_n_norm = -A_n / (a · dI/da_b)) has a non-NaN denominator.
            # extract_forced_photometry_mb leaves grad_<b>=NaN by design;
            # the post-hoc np.gradient over the template column is the
            # only finite-difference estimate available in forced mode.
            for b, g_b in zip(bands, per_band_grad):
                iso[f"grad_{b}"] = g_b
            # Under loose validity each band kept its own samples, so hand the
            # attacher the jagged lists; it accepts either layout. Passing the
            # rectangular intersection view instead would compute each band's
            # harmonics on samples other bands happened to share.
            if ring.intens_per_band is not None:
                harmonic_angles = ring.phi_per_band
                harmonic_intens = ring.intens_per_band
                harmonic_variances = ring.variances_per_band
            else:
                harmonic_angles = ring.angles
                harmonic_intens = ring.intens
                harmonic_variances = ring.variances
            _attach_per_band_harmonics(
                iso,
                bands,
                harmonic_angles,
                harmonic_intens,
                harmonic_variances,
                float(iso["sma"]),  # type: ignore[arg-type]
                per_band_grad,
                harmonic_orders=config.harmonic_orders,
            )

    _validate_non_negative_error_fields(isophotes)

    # Stage-3 Stage-D: per-band CoG composes with forced extraction.
    if config.compute_cog and isophotes:
        from .cog_mb import add_cog_mb_to_isophotes, compute_cog_mb

        cog_results = compute_cog_mb(
            isophotes,
            bands=list(config.bands),
            fix_center=True,  # geometry is pinned by template — no fits
            fix_geometry=True,
        )
        add_cog_mb_to_isophotes(isophotes, list(config.bands), cog_results)

    result: Dict[str, object] = {
        "isophotes": isophotes,
        "config": config,
        "bands": list(config.bands),
        "multiband": True,
        "harmonic_combination": config.harmonic_combination,
        "reference_band": config.reference_band,
        "band_weights": config.resolved_band_weights(),
        "variance_mode": config.variance_mode or ("wls" if variance_maps is not None else "ols"),
        "fit_per_band_intens_jointly": config.fit_per_band_intens_jointly,
        "loose_validity": config.loose_validity,
        "multiband_higher_harmonics": config.multiband_higher_harmonics,
        "harmonic_orders": list(config.harmonic_orders),
        "harmonics_shared": config.multiband_higher_harmonics != "independent",
        # Stage-H: surface forced-mode metadata so downstream tooling
        # can detect the workflow and show appropriate banners.
        "forced_photometry_mode": True,
        "template_n_isophotes": len(template_list),
    }
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fit_image_multiband(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]] = None,
    config: Optional[IsosterConfigMB] = None,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
    *,
    template_isophotes: object = None,
) -> Dict[str, object]:
    """
    Fit isophotes jointly across multiple aligned same-pixel-grid images.

    The default configuration is supported as of 2026-08-15; the
    ``simultaneous_*`` higher-harmonic modes and ``loose_validity=True``
    are not, and warn where they are configured (see
    :class:`~isoster.multiband.config_mb.IsosterConfigMB`). The public
    API and the Schema-1 output layout may still change between
    releases. See ``docs/10-multiband.md`` for the user-facing
    reference.

    Single shared geometry per SMA, per-band intensities and per-band
    harmonic deviations. Replaces the traditional forced-photometry
    workflow (decision D1).

    Parameters
    ----------
    images : sequence of ndarray
        ``B`` aligned images of shape ``(H, W)``. The order must match
        ``config.bands``.
    masks : None | ndarray | sequence of (ndarray|None), optional
        Bad-pixel masks. See :func:`isoster.multiband.sampling_mb.extract_isophote_data_multi`.
    config : IsosterConfigMB
        Multi-band configuration. ``bands`` and ``reference_band`` are
        required; everything else has sensible defaults.
    variance_maps : None | ndarray | sequence of ndarray, optional
        Per-pixel variance for WLS. All-or-nothing.
    template_isophotes : path / dict / list, optional
        **Stage-3 Stage-H forced-photometry mode.** When provided,
        bypass the iteration loop entirely and run forced extraction
        at the template's exact ``(sma, x0, y0, eps, pa)`` geometry on
        every band simultaneously. Accepts a Schema-1 multi-band FITS
        path, a single-band FITS path, a result dict (multi- or
        single-band), or a list of isophote dicts. Common workflow:
        run single-band ``isoster.fit_image`` on a deep band first,
        then pass the result here to extract per-band profiles at
        bit-identical geometry in one call (more efficient than
        running single-band ``B`` times). Validators warn-and-ignore
        ``lsb_auto_lock`` / ``use_outer_center_regularization`` /
        ``use_central_regularization`` / ``harmonic_combination='ref'``
        in this mode (the iteration loop, the inward / outward
        sweeps, the reference-geom build, and the lock state machine
        all become no-ops).

    Returns
    -------
    result : dict
        ``result['isophotes']`` is a list of per-isophote dicts in
        ascending SMA order. ``result['config']`` is the resolved
        :class:`IsosterConfigMB`. Multi-band top-level keys (decision D14):
        ``'bands'``, ``'multiband'``, ``'harmonic_combination'``,
        ``'reference_band'``, ``'band_weights'``, ``'variance_mode'``,
        ``'fit_per_band_intens_jointly'``, ``'loose_validity'``,
        ``'multiband_higher_harmonics'``, ``'harmonic_orders'``, and
        ``'harmonics_shared'`` (False only when
        ``multiband_higher_harmonics='independent'``).
        Conditional keys: ``'first_isophote_failure'`` when the anchor
        and all retry/probe attempts fail, ``'first_isophote_retry_log'``
        when first-isophote retries were attempted, and
        ``'lsb_auto_lock'`` / ``'lsb_auto_lock_sma'`` /
        ``'lsb_auto_lock_anchor_sma'`` / ``'lsb_auto_lock_count'`` when
        ``config.lsb_auto_lock`` is enabled.
        In forced-photometry mode the result also carries
        ``'forced_photometry_mode': True`` and
        ``'template_n_isophotes': len(template)``.

    Notes
    -----
    When ``len(config.bands) == 1`` this delegates to
    :func:`isoster.fit_image` and returns the legacy single-band schema
    unmodified (with an informational warning).
    """
    if config is None:
        raise ValueError("config is required (IsosterConfigMB with bands and reference_band)")
    _validate_inputs(images, masks, variance_maps, config)

    # B=1 fallback (decision D14).
    if len(config.bands) == 1:
        return _delegate_single_band(images, masks, variance_maps, config, template_isophotes=template_isophotes)

    # Surface the variance mode on the resolved config so it lands in
    # the FITS CONFIG HDU later. Review fix H1: build a model_copy with
    # the override rather than mutating the user's instance — reusing
    # the same config across an OLS run followed by a WLS run (or vice
    # versa) previously leaked the prior run's variance_mode into the
    # user's instance. ``config.model_copy(update={...})`` returns a
    # fresh, internally-resolved config that the driver threads through
    # the rest of the fit; the caller's original instance is untouched.
    variance_mode = "wls" if variance_maps is not None else "ols"
    config = config.model_copy(update={"variance_mode": variance_mode})

    # Stage-3 Stage-H: forced-photometry mode bypasses everything else.
    # Validators warn-and-ignore the iteration-only features that are
    # meaningless under forced extraction (lsb_auto_lock, outer-reg,
    # central-reg, harmonic_combination='ref'); the dispatcher returns
    # the standard multi-band result-dict shape.
    if template_isophotes is not None:
        return _fit_image_template_forced_mb(
            images=images,
            masks=masks,
            config=config,
            variance_maps=variance_maps,
            template=template_isophotes,
        )

    # Pre-resolve image / mask / variance arrays exactly once and reuse
    # across every per-isophote call. This is the dominant performance
    # win identified in Stage-1 benchmarking (decision D19).
    image_stack, masks_resolved, var_stack = prepare_inputs(
        images,
        masks,
        variance_maps,
    )

    h, w = np.asarray(images[0]).shape
    x0 = config.x0 if config.x0 is not None else w / 2.0
    y0 = config.y0 if config.y0 is not None else h / 2.0
    sma0 = config.sma0
    minsma = config.minsma
    maxsma = config.maxsma if config.maxsma is not None else float(np.hypot(h, w) / 2.0)
    astep = config.astep
    linear_growth = config.linear_growth

    # Central-pixel record (when minsma <= 0).
    central_result = _fit_central_pixel_mb(
        images,
        masks,
        x0,
        y0,
        config,
        variance_maps=variance_maps,
    )

    # First isophote at sma0.
    start_geometry = {"x0": x0, "y0": y0, "eps": config.eps, "pa": config.pa}
    first_iso = fit_isophote_mb(
        images,
        masks,
        sma0,
        start_geometry,
        config,
        variance_maps=variance_maps,
        image_stack=image_stack,
        masks_resolved=masks_resolved,
        var_stack=var_stack,
    )

    retry_log: List[Dict[str, object]] = []
    if not _is_acceptable(first_iso) and config.max_retry_first_isophote > 0:
        perturbations = _first_isophote_perturbations(
            sma0,
            config.eps,
            config.pa,
            config.max_retry_first_isophote,
        )
        for attempt_idx, (sma0_try, eps_try, pa_try) in enumerate(perturbations, start=1):
            retry_geom = {"x0": x0, "y0": y0, "eps": eps_try, "pa": pa_try}
            cand = fit_isophote_mb(
                images,
                masks,
                sma0_try,
                retry_geom,
                config,
                variance_maps=variance_maps,
                image_stack=image_stack,
                masks_resolved=masks_resolved,
                var_stack=var_stack,
            )
            retry_log.append(
                {
                    "attempt": attempt_idx,
                    "sma0": sma0_try,
                    "eps": eps_try,
                    "pa": pa_try,
                    "stop_code": cand["stop_code"],
                }
            )
            if _is_acceptable(cand):
                first_iso = cand
                sma0 = sma0_try
                break

    anchor_iso: Optional[Dict[str, object]] = None
    first_isophote_failure = False
    if _is_acceptable(first_iso):
        anchor_iso = first_iso
    else:
        failed_initial = [first_iso]
        probe_sma = sma0
        n_extra_probes = config.first_isophote_fail_count - 1
        for _ in range(n_extra_probes):
            probe_sma = probe_sma + astep if linear_growth else probe_sma * (1.0 + astep)
            if probe_sma > maxsma:
                break
            probe_iso = fit_isophote_mb(
                images,
                masks,
                probe_sma,
                start_geometry,
                config,
                variance_maps=variance_maps,
                image_stack=image_stack,
                masks_resolved=masks_resolved,
                var_stack=var_stack,
            )
            failed_initial.append(probe_iso)
            if _is_acceptable(probe_iso):
                anchor_iso = probe_iso
                sma0 = probe_sma
                break
        if anchor_iso is None:
            first_isophote_failure = True
            stop_codes = [iso["stop_code"] for iso in failed_initial]
            warnings.warn(
                f"FIRST_FEW_ISOPHOTE_FAILURE (multiband): the first {len(failed_initial)} "
                f"isophotes (starting sma0={config.sma0:.2f}) all failed with stop "
                f"codes {stop_codes}. Consider adjusting sma0, initial geometry, or "
                f"max_retry_first_isophote.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Inward growth.
    inwards_results: List[Dict[str, object]] = []
    if minsma < sma0 and anchor_iso is not None:
        current_iso: Dict[str, object] = anchor_iso
        current_sma = float(anchor_iso["sma"])  # type: ignore[arg-type]
        while True:
            next_sma = current_sma - astep if linear_growth else current_sma / (1.0 + astep)
            limit_sma = max(minsma, 0.5)
            if next_sma < limit_sma:
                break
            current_sma = next_sma
            current_geom = {
                "x0": float(current_iso["x0"]),  # type: ignore[arg-type]
                "y0": float(current_iso["y0"]),  # type: ignore[arg-type]
                "eps": float(current_iso["eps"]),  # type: ignore[arg-type]
                "pa": float(current_iso["pa"]),  # type: ignore[arg-type]
            }
            next_iso = fit_isophote_mb(
                images,
                masks,
                current_sma,
                current_geom,
                config,
                going_inwards=True,
                previous_geometry=current_geom,
                variance_maps=variance_maps,
                image_stack=image_stack,
                masks_resolved=masks_resolved,
                var_stack=var_stack,
            )
            inwards_results.append(next_iso)
            if _is_acceptable(next_iso) or config.permissive_geometry:
                current_iso = next_iso

    # Build the frozen outer-region reference geometry once, when the
    # outer damping feature is on and we have an anchor (Stage-3 Stage-B).
    outer_ref_geom: Optional[Dict[str, float]] = None
    if config.use_outer_center_regularization and anchor_iso is not None:
        outer_ref_geom = _build_outer_reference_mb(
            inwards_results,
            anchor_iso,
            config,
        )

    # Outward growth.
    outwards_results: List[Dict[str, object]] = []
    # Stage-3 Stage-C: LSB auto-lock state. Only mutated when
    # config.lsb_auto_lock=True; the dict is always initialized so
    # the post-loop metadata block stays uniform. Mirrors single-band.
    lsb_state: Dict[str, object] = {
        "locked": False,
        "consec": 0,
        "transition_sma": None,
        "anchor_index": None,
        "anchor_sma": None,
    }
    active_cfg: IsosterConfigMB = config
    if anchor_iso is not None:
        if config.lsb_auto_lock:
            _mark_lsb_lock_state_mb(anchor_iso, locked=False)
        outwards_results.append(anchor_iso)
        current_iso = anchor_iso
        current_sma = float(anchor_iso["sma"])  # type: ignore[arg-type]
        while True:
            next_sma = current_sma + astep if linear_growth else current_sma * (1.0 + astep)
            if next_sma > maxsma:
                break
            current_sma = next_sma
            current_geom = {
                "x0": float(current_iso["x0"]),  # type: ignore[arg-type]
                "y0": float(current_iso["y0"]),  # type: ignore[arg-type]
                "eps": float(current_iso["eps"]),  # type: ignore[arg-type]
                "pa": float(current_iso["pa"]),  # type: ignore[arg-type]
            }
            next_iso = fit_isophote_mb(
                images,
                masks,
                current_sma,
                current_geom,
                active_cfg,
                previous_geometry=current_geom,
                variance_maps=variance_maps,
                image_stack=image_stack,
                masks_resolved=masks_resolved,
                var_stack=var_stack,
                outer_reference_geom=outer_ref_geom,
            )
            if config.lsb_auto_lock:
                _mark_lsb_lock_state_mb(next_iso, locked=bool(lsb_state["locked"]))
            outwards_results.append(next_iso)
            if _is_acceptable(next_iso) or active_cfg.permissive_geometry:
                current_iso = next_iso

            # LSB auto-lock detector: free-mode only. Once locked, stays
            # locked for the rest of outward growth (one-way state
            # machine). Trigger surface is the joint combined gradient
            # (plan section 7 S3) via _is_lsb_isophote_mb.
            if config.lsb_auto_lock and not lsb_state["locked"]:
                triggered = _is_lsb_isophote_mb(
                    next_iso,
                    config.lsb_auto_lock_maxgerr,
                )
                if triggered:
                    lsb_state["consec"] = int(lsb_state["consec"]) + 1
                    if int(lsb_state["consec"]) >= config.lsb_auto_lock_debounce:
                        # Anchor = isophote immediately BEFORE the streak,
                        # NOT the trigger isophote itself. The trigger
                        # isophotes are already partially in LSB and their
                        # geometries may have begun drifting.
                        anchor_local_idx = len(outwards_results) - 1 - config.lsb_auto_lock_debounce
                        anchor_local_idx = max(0, anchor_local_idx)
                        lock_anchor = outwards_results[anchor_local_idx]
                        active_cfg = _build_locked_cfg_mb(
                            config,
                            lock_anchor,
                            config.lsb_auto_lock_integrator,
                        )
                        lsb_state["locked"] = True
                        lsb_state["transition_sma"] = float(next_iso["sma"])  # type: ignore[arg-type]
                        lsb_state["anchor_index"] = anchor_local_idx
                        # Record the true geometry anchor's identity (U1/Q2),
                        # mirroring single-band; the marker below goes on the
                        # committing trigger.
                        lsb_state["anchor_sma"] = float(lock_anchor["sma"])  # type: ignore[arg-type]
                        _mark_lsb_lock_state_mb(next_iso, locked=True, is_anchor=True)
                        # Restart the geometry carry-forward from the
                        # clean anchor so the next isophote is seeded by
                        # it, not by the (degraded) trigger isophote.
                        current_iso = lock_anchor
                else:
                    lsb_state["consec"] = 0

    # Assemble final list (ascending SMA).
    if minsma <= 0.0:
        final_list: List[Dict[str, object]] = [central_result] + inwards_results[::-1] + outwards_results
    else:
        final_list = inwards_results[::-1] + outwards_results

    if config.lsb_auto_lock:
        # Inward isophotes and the central pixel never pass through the
        # outward lock machinery; stamp them explicitly so every row carries
        # the same lock keys. A heterogeneous schema makes Table(rows=...)
        # mask the missing cells, and FITS fills masked bools with True,
        # destroying the flags on save/load (review B3).
        for iso in final_list:
            iso.setdefault("lsb_locked", False)
            iso.setdefault("lsb_auto_lock_anchor", False)

    _validate_non_negative_error_fields(final_list)

    # Stage-3 Stage-D: per-band curve-of-growth photometry. Runs once
    # over the assembled isophote list (ascending sma) and stamps
    # per-band ``cog_<b>`` / ``cog_annulus_<b>`` plus shared
    # ``area_annulus`` / ``flag_cross`` / ``flag_negative_area`` onto
    # each row dict. Schema 1 round-trip handles the new columns
    # automatically via ``Table(rows=isophotes)`` auto-inference. See
    # plan section 7 S7.
    if config.compute_cog and final_list:
        from .cog_mb import add_cog_mb_to_isophotes, compute_cog_mb

        cog_results = compute_cog_mb(
            final_list,
            bands=list(config.bands),
            fix_center=config.fix_center,
            fix_geometry=config.fix_center and config.fix_pa and config.fix_eps,
        )
        add_cog_mb_to_isophotes(final_list, list(config.bands), cog_results)

    result: Dict[str, object] = {
        "isophotes": final_list,
        "config": config,
        "bands": list(config.bands),
        "multiband": True,
        "harmonic_combination": config.harmonic_combination,
        "reference_band": config.reference_band,
        "band_weights": config.resolved_band_weights(),
        "variance_mode": variance_mode,
        "fit_per_band_intens_jointly": config.fit_per_band_intens_jointly,
        "loose_validity": config.loose_validity,
        # Section 6: higher-order harmonics mode + orders. ``harmonics_shared``
        # is a derived convenience flag (True iff a non-``independent`` mode
        # is active) so downstream readers can short-circuit without parsing
        # the enum.
        "multiband_higher_harmonics": config.multiband_higher_harmonics,
        "harmonic_orders": list(config.harmonic_orders),
        "harmonics_shared": config.multiband_higher_harmonics != "independent",
    }
    if first_isophote_failure:
        result["first_isophote_failure"] = True
    if retry_log:
        result["first_isophote_retry_log"] = retry_log
    # Stage-3 Stage-C: surface lsb_auto_lock metadata when enabled.
    if config.lsb_auto_lock:
        result["lsb_auto_lock"] = True
        result["lsb_auto_lock_sma"] = lsb_state["transition_sma"]
        result["lsb_auto_lock_anchor_sma"] = lsb_state["anchor_sma"]
        result["lsb_auto_lock_count"] = sum(1 for iso in final_list if bool(iso.get("lsb_locked", False)))
    return result
