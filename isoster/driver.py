import warnings
from pathlib import Path

import numpy as np

from .config import IsosterConfig
from .fitting import fit_isophote

ACCEPTABLE_STOP_CODES = {0, 1, 2}

_TEMPLATE_REQUIRED_KEYS = {"sma", "x0", "y0", "eps", "pa"}


def _is_acceptable_stop_code(stop_code):
    """Return True when a stop code is acceptable for geometry propagation."""
    return stop_code in ACCEPTABLE_STOP_CODES


def _first_isophote_perturbations(sma0, eps, pa, max_retries):
    """Generate perturbed (sma0, eps, pa) tuples for first-isophote retry.

    The schedule alternates between smaller/larger sma0 and progressively
    introduces geometry perturbations (near-circular eps, rotated PA).
    All sma0 values are clamped to >= 1.0 pixel.

    Args:
        sma0: Original starting semi-major axis.
        eps: Original ellipticity.
        pa: Original position angle in radians.
        max_retries: Number of perturbations to generate.

    Returns:
        List of (sma0_new, eps_new, pa_new) tuples.
    """
    # Fixed schedule for first 5 attempts
    schedule = [
        (0.8, eps, pa),  # Slightly smaller radius
        (1.3, eps, pa),  # Slightly larger radius
        (0.6, 0.05, pa),  # Smaller + near-circular
        (1.5, eps, pa + np.pi / 4),  # Larger + rotated PA
        (0.5, 0.05, pa + np.pi / 2),  # Small + circular + orthogonal
    ]

    # Extended schedule: cycle through factors with alternating geometry
    extended_factors = [0.4, 0.7, 1.1, 1.6, 2.0]
    for i, factor in enumerate(extended_factors):
        ext_eps = 0.05 if i % 2 == 0 else eps
        ext_pa = pa + (i + 1) * np.pi / 6
        schedule.append((factor, ext_eps, ext_pa))

    result = []
    for i in range(min(max_retries, len(schedule))):
        sma_factor, eps_new, pa_new = schedule[i]
        sma_new = max(1.0, sma0 * sma_factor)
        eps_new = min(max(0.0, eps_new), 0.99)
        result.append((sma_new, eps_new, pa_new))

    return result


def _is_lsb_isophote(iso, maxgerr_thresh):
    """Return True when an outward isophote indicates the LSB regime.

    Uses the gradient diagnostics populated when debug=True. A gradient of
    None / non-finite / zero-error is treated as "cannot assess" and returns
    False so the detector does not trigger on transient numerical glitches —
    the next isophote will supply another chance.
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


def _build_locked_cfg(cfg, anchor_iso, locked_integrator):
    """Clone cfg and freeze geometry to a clean anchor isophote."""
    locked = cfg.model_copy(deep=True)
    locked.x0 = float(anchor_iso["x0"])
    locked.y0 = float(anchor_iso["y0"])
    locked.eps = float(anchor_iso["eps"])
    locked.pa = float(anchor_iso["pa"])
    locked.fix_center = True
    locked.fix_pa = True
    locked.fix_eps = True
    locked.integrator = locked_integrator
    # Adaptive integrator settings would collide with the explicit lock;
    # disable them on the clone.
    locked.lsb_sma_threshold = None
    # The auto-lock has already committed on the parent fit; the clone
    # should not try to re-detect on its own isophotes.
    locked.lsb_auto_lock = False
    return locked


def _mark_lsb_lock_state(iso, locked, is_anchor=False):
    """Attach the LSB auto-lock state keys to an isophote dict.

    Centralized so future loop reorders cannot accidentally mutate an
    isophote before the correct flag is known. Always writes both
    ``lsb_locked`` and ``lsb_auto_lock_anchor`` so the per-isophote schema
    stays uniform and FITS round-trips cannot corrupt the flags (review B3).
    """
    iso["lsb_locked"] = bool(locked)
    iso["lsb_auto_lock_anchor"] = bool(is_anchor)


def _build_outer_reference(inwards_results, anchor_iso, cfg):
    """Compute a frozen reference geometry for outer-region regularization.

    Uses a flux-weighted mean of inward isophotes with acceptable stop codes
    and sma <= anchor_iso['sma'] * cfg.outer_reg_ref_sma_factor. Falls back
    to the anchor isophote's own geometry when no inward isophotes qualify,
    which also covers the case where the inward loop produced nothing
    (e.g. minsma >= sma0).

    The anchor isophote itself is always folded into the weighted mean so the
    reference is well-defined even when only a handful of inward isophotes exist.

    The reference carries x0, y0, eps, and pa. Downstream consumers (the
    outer-reg penalty) pick which fields to use based on outer_reg_weights.
    x0/y0/eps are flux-weighted arithmetic means; pa uses a flux-weighted
    circular mean on 2*pa (axis-like angles, pa is defined mod pi) so that
    isophotes with pa near 0 and near pi do not average to pi/2. If the
    circular-mean resultant length is too small for a well-defined pa
    (highly scattered inner pa), pa falls back to the anchor's pa.

    Args:
        inwards_results (list[dict]): Inward isophote results (ascending sma
            inside the list is not assumed; filtering is on `sma` directly).
        anchor_iso (dict): The successful anchor isophote at sma0.
        cfg (IsosterConfig): Full config, read for outer_reg_ref_sma_factor.

    Returns:
        dict: Reference geometry with keys 'x0', 'y0', 'eps', 'pa'.
    """
    anchor_ref = {
        "x0": float(anchor_iso["x0"]),
        "y0": float(anchor_iso["y0"]),
        "eps": float(anchor_iso["eps"]),
        "pa": float(anchor_iso["pa"]),
    }

    sma_hi = float(anchor_iso["sma"]) * cfg.outer_reg_ref_sma_factor

    candidates = [anchor_iso]
    for iso in inwards_results:
        if not _is_acceptable_stop_code(iso.get("stop_code")):
            continue
        sma = iso.get("sma")
        intens = iso.get("intens")
        eps_val = iso.get("eps")
        pa_val = iso.get("pa")
        if sma is None or not np.isfinite(sma):
            continue
        if intens is None or not np.isfinite(intens):
            continue
        if eps_val is None or not np.isfinite(eps_val):
            continue
        if pa_val is None or not np.isfinite(pa_val):
            continue
        if sma > sma_hi:
            continue
        candidates.append(iso)

    # Flux-weight (clamp to strictly positive so a negative-background iso
    # does not dominate); fall back to the anchor if total weight vanishes.
    x0s = np.array([float(iso["x0"]) for iso in candidates])
    y0s = np.array([float(iso["y0"]) for iso in candidates])
    eps_arr = np.array([float(iso["eps"]) for iso in candidates])
    pa_arr = np.array([float(iso["pa"]) for iso in candidates])
    weights = np.array([max(float(iso["intens"]), 1e-6) for iso in candidates])
    if weights.sum() <= 0.0 or not np.all(np.isfinite(weights)):
        return anchor_ref

    x0_ref = float(np.average(x0s, weights=weights))
    y0_ref = float(np.average(y0s, weights=weights))
    eps_ref = float(np.average(eps_arr, weights=weights))

    # Circular mean on 2*pa: fold axis-like angles onto the unit circle so
    # pa=0 and pa~pi average sensibly. Resultant length R < ~0.1 means the
    # inner pa values are too scattered for a useful mean; fall back to the
    # anchor's pa in that case.
    cos_sum = float(np.sum(weights * np.cos(2.0 * pa_arr)))
    sin_sum = float(np.sum(weights * np.sin(2.0 * pa_arr)))
    w_sum = float(weights.sum())
    resultant = np.hypot(cos_sum, sin_sum) / w_sum if w_sum > 0.0 else 0.0
    if resultant >= 0.1:
        pa_ref = 0.5 * float(np.arctan2(sin_sum, cos_sum))
        pa_ref = pa_ref % np.pi
    else:
        pa_ref = float(anchor_iso["pa"]) % np.pi

    if not (np.isfinite(x0_ref) and np.isfinite(y0_ref) and np.isfinite(eps_ref) and np.isfinite(pa_ref)):
        return anchor_ref

    return {"x0": x0_ref, "y0": y0_ref, "eps": eps_ref, "pa": pa_ref}


def _is_error_field(field_name):
    """Return True when the field represents an uncertainty/error quantity."""
    return field_name.endswith("_err") or field_name.endswith("_error")


def _validate_non_negative_error_fields(isophotes):
    """Ensure all finite error values in fitted isophotes are non-negative."""
    if not isophotes:
        return

    error_fields = {key for iso in isophotes if isinstance(iso, dict) for key in iso.keys() if _is_error_field(key)}
    if not error_fields:
        return

    for iso_index, iso in enumerate(isophotes):
        if not isinstance(iso, dict):
            continue
        for field_name in error_fields:
            if field_name not in iso:
                continue
            try:
                numeric_value = float(iso[field_name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value) and numeric_value < 0.0:
                raise ValueError(
                    "isoster produced negative error value "
                    f"(field={field_name}, index={iso_index}, value={numeric_value})"
                )


def _build_fit_result(isophotes, config):
    """Package fit results with mandatory post-run sanity validation."""
    _validate_non_negative_error_fields(isophotes)
    return {"isophotes": isophotes, "config": config}


def _resolve_template(template):
    """
    Normalize template input into a validated, SMA-sorted list of isophote dicts.

    Accepted input forms:
    - ``str`` or ``Path``: path to a FITS or ASDF file (dispatched by extension)
    - ``dict`` with an ``'isophotes'`` key: a results dict (e.g. from ``fit_image()``)
    - ``list`` of dicts: isophote dicts directly

    Args:
        template: Template input in any of the above forms.

    Returns:
        list[dict]: Validated list of isophote dicts sorted by SMA.

    Raises:
        TypeError: If template is not a recognized type.
        ValueError: If template is empty or any dict is missing required keys.
    """
    from .utils import isophote_results_from_asdf, isophote_results_from_fits

    # str/Path → load from file (dispatch on extension)
    if isinstance(template, (str, Path)):
        path_str = str(template)
        if path_str.endswith(".asdf"):
            loaded = isophote_results_from_asdf(path_str)
        else:
            loaded = isophote_results_from_fits(path_str)
        iso_list = loaded["isophotes"]
    # dict with 'isophotes' key → extract the list
    elif isinstance(template, dict):
        if "isophotes" not in template:
            raise ValueError(f"template dict must contain an 'isophotes' key; got keys: {sorted(template.keys())}")
        iso_list = template["isophotes"]
    # list → use directly
    elif isinstance(template, list):
        iso_list = template
    else:
        raise TypeError(
            f"template must be a file path (str/Path), results dict, or list of "
            f"isophote dicts; got {type(template).__name__}"
        )

    if not iso_list:
        raise ValueError("template cannot be empty")

    # Validate required keys in each isophote dict (R26-13)
    for i, iso in enumerate(iso_list):
        missing = _TEMPLATE_REQUIRED_KEYS - set(iso.keys())
        if missing:
            raise ValueError(f"template isophote at index {i} missing required keys: {sorted(missing)}")

    return sorted(iso_list, key=lambda x: x["sma"])


def fit_central_pixel(image, mask, x0, y0, debug=False, config=None, variance_map=None):
    """
    Fit the central pixel (SMA=0).

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray, optional): Boolean mask (True=bad).
        x0 (float): Center x coordinate.
        y0 (float): Center y coordinate.
        debug (bool): If True, include extra debug fields.
        config (IsosterConfig, optional): When provided, harmonic keys are
            only included if compute_deviations or simultaneous_harmonics is
            enabled, using config.harmonic_orders for key names — matching
            the fitted-isophote schema (review B5). When None, orders [3, 4]
            are always included for backward compatibility.
        variance_map (np.ndarray, optional): Per-pixel variance map. When
            provided, ``intens_err`` is ``sqrt(variance_map[iy, ix])`` —
            the per-pixel standard error (multiband parity, review M12).
            In OLS mode it stays 0.0: a single sample admits no internal
            error estimate.

    Returns:
        dict: A dictionary containing the fitting result for the central pixel.
    """
    # Simple estimation for center
    # Use np.round() instead of int() to avoid truncation bias
    h, w = image.shape
    iy, ix = int(np.round(y0)), int(np.round(x0))

    # Bounds check: out-of-bounds center → invalid central pixel
    if iy < 0 or iy >= h or ix < 0 or ix >= w:
        valid = False
        val = np.nan
    else:
        val = image[iy, ix]
        valid = True
        if mask is not None and mask[iy, ix]:
            valid = False

    if config is not None:
        include_harmonics = config.compute_deviations or config.simultaneous_harmonics
        harmonic_orders = config.harmonic_orders if include_harmonics else []
    else:
        harmonic_orders = [3, 4]

    # WLS: per-pixel standard error from the variance map (multiband
    # parity, review M12). OLS: 0.0 (a single sample admits no internal
    # error estimate).
    intens_err = 0.0
    if valid and variance_map is not None:
        v = float(np.asarray(variance_map)[iy, ix])
        if np.isfinite(v) and v > 0.0:
            intens_err = float(np.sqrt(v))

    result = {
        "x0": x0,
        "y0": y0,
        "eps": 0.0,
        "pa": 0.0,
        "sma": 0.0,
        "intens": val if valid else np.nan,
        "rms": 0.0,
        "intens_err": intens_err,
        "x0_err": 0.0,
        "y0_err": 0.0,
        "eps_err": 0.0,
        "pa_err": 0.0,
        "tflux_e": np.nan,
        "tflux_c": np.nan,
        "npix_e": 0,
        "npix_c": 0,
        "stop_code": 0 if valid else -1,
        "niter": 0,
        "valid": valid,
    }
    for order in harmonic_orders:
        result[f"a{order}"] = 0.0
        result[f"b{order}"] = 0.0
        result[f"a{order}_err"] = 0.0
        result[f"b{order}_err"] = 0.0
    if debug:
        result.update(
            {"ndata": 1 if valid else 0, "nflag": 0, "grad": np.nan, "grad_error": np.nan, "grad_r_error": np.nan}
        )
    return result


def fit_image(image, mask=None, config=None, template=None, template_isophotes=None, variance_map=None):
    """
    Main driver to fit isophotes to an entire image.

    This function orchestrates the fitting process, starting from a central guess,
    fitting outward to the edge, and optionally inward to the center.

    Args:
        image (np.ndarray): 2D Input image.
        mask (np.ndarray, optional): 2D Bad pixel mask (True=bad).
        config (dict or IsosterConfig, optional): Configuration parameters.
            If None, default configuration is used.
        template: Template for forced photometry. Accepts:
            - File path (str/Path) to a FITS file saved by ``isophote_results_to_fits()``
            - Results dict (e.g. from a previous ``fit_image()`` call)
            - List of isophote dicts with keys: sma, x0, y0, eps, pa
            When provided, photometry is extracted at each template SMA using the
            template's geometry. This enables multiband analysis where one band
            defines geometry and others use the same geometry.
        template_isophotes (list of dict, optional): Deprecated. Use ``template``
            instead. Will be removed in a future version.
        variance_map (np.ndarray, optional): 2D per-pixel variance map matching
            image shape. When provided, Weighted Least Squares (WLS) is used
            instead of OLS, giving exact covariance and automatic outlier
            down-weighting. Typical source: ``1.0 / invvar`` from survey pipelines.

    Returns:
        dict: A dictionary containing:
            - 'isophotes': List of dictionaries, each representing a fitted isophote.
            - 'config': The IsosterConfig object used for the fit.

    Examples
    --------
    >>> # Normal fitting
    >>> results_g = fit_image(image_g, mask_g, config)
    >>>
    >>> # With variance map (WLS)
    >>> results_g = fit_image(image_g, mask_g, config, variance_map=var_g)
    >>>
    >>> # Template-based forced photometry (multiband)
    >>> results_r = fit_image(image_r, mask_r, config, template=results_g)
    >>>
    >>> # Template from FITS file
    >>> results_r = fit_image(image_r, mask_r, config, template="galaxy_g.fits")
    """
    if config is None:
        cfg = IsosterConfig()
    elif isinstance(config, dict):
        cfg = IsosterConfig(**config)
    else:
        cfg = config

    # The LSB auto-lock detector needs grad / grad_error populated on each
    # outward isophote dict, which only happens when debug=True. The config
    # validator has already emitted a UserWarning; transparently promote a
    # working clone here so the user does not have to flip both flags.
    if cfg.lsb_auto_lock and not cfg.debug:
        cfg = cfg.model_copy(update={"debug": True})

    # Validate mutually exclusive template arguments
    if template is not None and template_isophotes is not None:
        raise ValueError(
            "Cannot specify both 'template' and 'template_isophotes'. "
            "Use 'template' (template_isophotes is deprecated)."
        )

    # Backward compatibility: template_isophotes → template with deprecation warning
    if template_isophotes is not None:
        warnings.warn(
            "template_isophotes is deprecated; use template= instead. Will be removed in a future version.",
            FutureWarning,
            stacklevel=2,
        )
        template = template_isophotes

    # Validate the variance map and mark unusable entries.
    #
    # An entry that is not finite, or not strictly positive, carries no usable
    # information about the noise. Such entries are marked NaN here and the
    # affected samples are dropped during ring sampling, so a ring statistic
    # and its uncertainty always describe the same samples. Substituting a
    # large sentinel or clamping to a tiny positive value would instead leave
    # the sample in the statistic while distorting its weight: the 1e-30 clamp
    # in particular gave a single pixel near-infinite weight and could collapse
    # the reported gradient error by fifteen orders of magnitude.
    if variance_map is not None:
        if variance_map.shape != image.shape:
            raise ValueError(f"variance_map shape {variance_map.shape} does not match image shape {image.shape}")

        # Work on a copy to avoid mutating the caller's array
        variance_map = variance_map.astype(np.float64, copy=True)

        non_finite = ~np.isfinite(variance_map)
        n_non_finite = int(np.sum(non_finite))
        if n_non_finite > 0:
            warnings.warn(
                f"variance_map contains {n_non_finite} non-finite values (NaN or infinite); "
                f"the affected samples are excluded from both the ring statistics and their "
                f"uncertainties.",
                RuntimeWarning,
                stacklevel=2,
            )

        non_positive = ~non_finite & (variance_map <= 0.0)
        n_non_positive = int(np.sum(non_positive))
        if n_non_positive > 0:
            warnings.warn(
                f"variance_map contains {n_non_positive} non-positive values; a variance of zero "
                f"or below is not physical, so the affected samples are excluded from both the "
                f"ring statistics and their uncertainties. Mask these pixels explicitly if the "
                f"exclusion is intended.",
                RuntimeWarning,
                stacklevel=2,
            )

        variance_map[non_finite | non_positive] = np.nan

    # Handle template-based forced mode
    if template is not None:
        # Neither the LSB auto-lock nor the outer center regularization is
        # wired into the forced-photometry path (extract_forced_photometry
        # does not call fit_isophote). Warn loudly when either knob is on so
        # a stray True from a copy-pasted config is surfaced instead of
        # silently ignored.
        if cfg.lsb_auto_lock:
            warnings.warn(
                "lsb_auto_lock=True is ignored in forced photometry mode "
                "(template provided): the feature requires free outward "
                "growth and is not applied by extract_forced_photometry.",
                UserWarning,
                stacklevel=2,
            )
        if cfg.use_outer_center_regularization:
            warnings.warn(
                "use_outer_center_regularization=True is ignored in forced "
                "photometry mode (template provided): the feature requires "
                "free outward growth and is not applied by "
                "extract_forced_photometry.",
                UserWarning,
                stacklevel=2,
            )
        if cfg.simultaneous_harmonics:
            warnings.warn(
                "simultaneous_harmonics=True does not apply in forced photometry "
                "mode (template provided): the simultaneous solve fits the "
                "geometry harmonics and the higher orders as one system during "
                "the iteration, and forced mode imposes the geometry instead of "
                "fitting it. The higher orders are measured independently per "
                "order, exactly as compute_deviations=True would. Set "
                "compute_deviations=True to request that explicitly.",
                UserWarning,
                stacklevel=2,
            )
        resolved = _resolve_template(template)
        return _fit_image_template_forced(image, mask, cfg, resolved, variance_map=variance_map)

    # Regular fitting mode
    h, w = image.shape

    # Initial Parameters
    x0 = cfg.x0 if cfg.x0 is not None else w / 2.0
    y0 = cfg.y0 if cfg.y0 is not None else h / 2.0
    sma0 = cfg.sma0
    minsma = cfg.minsma
    maxsma = cfg.maxsma if cfg.maxsma is not None else max(h, w) / 2.0
    astep = cfg.astep
    linear_growth = cfg.linear_growth

    # 1. Fit Central Pixel (Approximation)
    central_result = fit_central_pixel(image, mask, x0, y0, debug=cfg.debug, config=cfg, variance_map=variance_map)

    # 2. Fit First Isophote at SMA0 (with optional retry)
    start_geometry = {"x0": x0, "y0": y0, "eps": cfg.eps, "pa": cfg.pa}
    first_iso = fit_isophote(
        image, mask, sma0, start_geometry, cfg, variance_map=variance_map, variance_map_prepared=True
    )

    # Retry with perturbed parameters if first isophote failed
    retry_log = []
    if not _is_acceptable_stop_code(first_iso["stop_code"]) and cfg.max_retry_first_isophote > 0:
        perturbations = _first_isophote_perturbations(sma0, cfg.eps, cfg.pa, cfg.max_retry_first_isophote)
        for attempt_idx, (sma0_try, eps_try, pa_try) in enumerate(perturbations, start=1):
            retry_geometry = {"x0": x0, "y0": y0, "eps": eps_try, "pa": pa_try}
            candidate = fit_isophote(
                image, mask, sma0_try, retry_geometry, cfg, variance_map=variance_map, variance_map_prepared=True
            )
            retry_log.append(
                {
                    "attempt": attempt_idx,
                    "sma0": sma0_try,
                    "eps": eps_try,
                    "pa": pa_try,
                    "stop_code": candidate["stop_code"],
                }
            )
            if _is_acceptable_stop_code(candidate["stop_code"]):
                first_iso = candidate
                sma0 = sma0_try
                break

    # Determine anchor isophote for growth propagation
    anchor_iso = None
    first_isophote_failure = False

    if _is_acceptable_stop_code(first_iso["stop_code"]):
        anchor_iso = first_iso
    else:
        # First iso failed (even after retries). Probe next growth steps
        # to see if any succeed before declaring failure.
        failed_initial = [first_iso]
        probe_sma = sma0
        n_extra_probes = cfg.first_isophote_fail_count - 1

        for _ in range(n_extra_probes):
            if linear_growth:
                probe_sma = probe_sma + astep
            else:
                probe_sma = probe_sma * (1.0 + astep)
            if probe_sma > maxsma:
                break
            probe_iso = fit_isophote(
                image, mask, probe_sma, start_geometry, cfg, variance_map=variance_map, variance_map_prepared=True
            )
            failed_initial.append(probe_iso)
            if _is_acceptable_stop_code(probe_iso["stop_code"]):
                anchor_iso = probe_iso
                sma0 = probe_sma
                break

        if anchor_iso is None:
            first_isophote_failure = True
            stop_codes = [iso["stop_code"] for iso in failed_initial]
            warnings.warn(
                f"FIRST_FEW_ISOPHOTE_FAILURE: the first {len(failed_initial)} isophotes "
                f"(starting sma0={cfg.sma0:.2f}) all failed with stop codes {stop_codes}. "
                f"Consider adjusting sma0, initial geometry (eps/pa), or enabling "
                f"max_retry_first_isophote.",
                RuntimeWarning,
                stacklevel=2,
            )

    # 3. Grow Inwards FIRST. This ordering enables the outer center
    # regularization feature to build a stable inner reference from the
    # inward isophote centers before outward growth begins. It is also
    # output-equivalent under all settings when the feature is off: the
    # inward loop does not depend on outward results, and the combined
    # isophote list is reassembled in ascending-sma order below.
    inwards_results = []
    if minsma < sma0 and anchor_iso is not None:
        current_iso = anchor_iso
        current_sma = anchor_iso["sma"]

        while True:
            if linear_growth:
                next_sma = current_sma - astep
            else:
                next_sma = current_sma / (1.0 + astep)

            # Stop if smaller than minsma or effectively too small (e.g. 0.5 pixel)
            limit_sma = max(minsma, 0.5)
            if next_sma < limit_sma:
                break

            current_sma = next_sma

            # Use going_inwards=True flag
            next_iso = fit_isophote(
                image,
                mask,
                next_sma,
                current_iso,
                cfg,
                going_inwards=True,
                previous_geometry=current_iso,
                variance_map=variance_map,
                variance_map_prepared=True,
            )
            inwards_results.append(next_iso)

            # In permissive mode, always update to prevent cascading failures
            if _is_acceptable_stop_code(next_iso["stop_code"]) or cfg.permissive_geometry:
                current_iso = next_iso

    # 3b. Build the frozen outer reference geometry once, if the feature is
    # enabled and we have an anchor. The reference is a flux-weighted mean of
    # inward isophote geometry (plus the anchor itself) within a sma window.
    # All four fields (x0, y0, eps, pa) are populated; which ones feed the
    # penalty is chosen at penalty time via cfg.outer_reg_weights.
    outer_ref_geom = None
    if cfg.use_outer_center_regularization and anchor_iso is not None:
        outer_ref_geom = _build_outer_reference(inwards_results, anchor_iso, cfg)

    # 4. Grow Outwards
    outwards_results = []
    # LSB auto-lock state. Only mutated when cfg.lsb_auto_lock=True, but
    # the dict is always initialized so the post-loop metadata block below
    # stays uniform.
    lsb_state = {
        "locked": False,
        "consec": 0,
        "transition_sma": None,
        "anchor_index": None,
        "anchor_sma": None,
    }
    active_cfg = cfg

    if anchor_iso is not None:
        if cfg.lsb_auto_lock:
            _mark_lsb_lock_state(anchor_iso, locked=False)
        outwards_results.append(anchor_iso)
        current_iso = anchor_iso
        current_sma = anchor_iso["sma"]

        while True:
            if linear_growth:
                next_sma = current_sma + astep
            else:
                next_sma = current_sma * (1.0 + astep)

            if next_sma > maxsma:
                break

            # Update sma tracking
            current_sma = next_sma

            next_iso = fit_isophote(
                image,
                mask,
                next_sma,
                current_iso,
                active_cfg,
                previous_geometry=current_iso,
                variance_map=variance_map,
                variance_map_prepared=True,
                outer_reference_geom=outer_ref_geom,
            )
            if cfg.lsb_auto_lock:
                _mark_lsb_lock_state(next_iso, locked=lsb_state["locked"])
            outwards_results.append(next_iso)

            # If good fit, update geometry for next step
            # In permissive mode, always update to prevent cascading failures
            if _is_acceptable_stop_code(next_iso["stop_code"]) or active_cfg.permissive_geometry:
                current_iso = next_iso

            # LSB auto-lock detector: free-mode only. Once locked, the mode
            # stays locked for the rest of outward growth (one-way state
            # machine).
            if cfg.lsb_auto_lock and not lsb_state["locked"]:
                triggered = _is_lsb_isophote(next_iso, cfg.lsb_auto_lock_maxgerr)
                if triggered:
                    lsb_state["consec"] += 1
                    if lsb_state["consec"] >= cfg.lsb_auto_lock_debounce:
                        # Anchor = isophote immediately before the streak, NOT
                        # the trigger isophote itself. The trigger isophotes
                        # are already partially in LSB and their geometries
                        # may have begun drifting.
                        anchor_local_idx = len(outwards_results) - 1 - cfg.lsb_auto_lock_debounce
                        anchor_local_idx = max(0, anchor_local_idx)
                        lock_anchor = outwards_results[anchor_local_idx]
                        active_cfg = _build_locked_cfg(cfg, lock_anchor, cfg.lsb_auto_lock_integrator)
                        lsb_state["locked"] = True
                        lsb_state["transition_sma"] = float(next_iso["sma"])
                        lsb_state["anchor_index"] = anchor_local_idx
                        # Record the true geometry anchor's identity (U1/Q2):
                        # the marker below goes on the committing trigger, so
                        # without this the anchor isophote is not identifiable
                        # in the output at all.
                        lsb_state["anchor_sma"] = float(lock_anchor["sma"])
                        _mark_lsb_lock_state(next_iso, locked=True, is_anchor=True)
                        # Restart the geometry carry-forward from the clean
                        # anchor so the next isophote is seeded by it, not by
                        # the (degraded) trigger isophote.
                        current_iso = lock_anchor
                else:
                    lsb_state["consec"] = 0

    # Combine results
    # Inwards list needs to be reversed so SMAs are ascending
    if minsma <= 0.0:
        # Prepend central pixel
        final_list = [central_result] + inwards_results[::-1] + outwards_results
    else:
        final_list = inwards_results[::-1] + outwards_results

    if cfg.lsb_auto_lock:
        # Inward isophotes and the central pixel never pass through the
        # outward lock machinery; stamp them explicitly so every row carries
        # the same lock keys. A heterogeneous schema makes Table(rows=...)
        # mask the missing cells, and FITS fills masked bools with True,
        # destroying the flags on save/load (review B3).
        for iso in final_list:
            iso.setdefault("lsb_locked", False)
            iso.setdefault("lsb_auto_lock_anchor", False)

    # Compute Curve-of-Growth if requested
    if cfg.compute_cog:
        from .cog import add_cog_to_isophotes, compute_cog

        # Determine if geometry was fixed
        fix_geometry = cfg.fix_center and cfg.fix_pa and cfg.fix_eps

        cog_results = compute_cog(final_list, fix_center=cfg.fix_center, fix_geometry=fix_geometry)

        # Add CoG data to isophotes
        add_cog_to_isophotes(final_list, cog_results)

    # Return as dict matching legacy structure + config object
    result = _build_fit_result(final_list, cfg)
    if first_isophote_failure:
        result["first_isophote_failure"] = True
    if retry_log:
        result["first_isophote_retry_log"] = retry_log
    if cfg.lsb_auto_lock:
        result["lsb_auto_lock"] = True
        result["lsb_auto_lock_sma"] = lsb_state["transition_sma"]
        result["lsb_auto_lock_anchor_sma"] = lsb_state["anchor_sma"]
        result["lsb_auto_lock_count"] = sum(1 for iso in final_list if iso.get("lsb_locked", False))
    if cfg.use_outer_center_regularization and outer_ref_geom is not None:
        result["use_outer_center_regularization"] = True
        result["outer_reg_x0_ref"] = outer_ref_geom["x0"]
        result["outer_reg_y0_ref"] = outer_ref_geom["y0"]
        result["outer_reg_eps_ref"] = outer_ref_geom["eps"]
        result["outer_reg_pa_ref"] = outer_ref_geom["pa"]
    return result


def _fit_image_template_forced(image, mask, config, template_isophotes, variance_map=None):
    """
    Extract forced photometry using geometry from template isophotes.

    This function extracts photometry at each SMA from the template, using the
    template's geometry (x0, y0, eps, pa) at that specific SMA. This enables
    variable geometry along the radial profile for multiband analysis.

    Args:
        image (np.ndarray): 2D input image.
        mask (np.ndarray, optional): 2D bad pixel mask (True=bad).
        config (IsosterConfig): Configuration object. Uses integrator, sclip,
            nclip, and use_eccentric_anomaly settings.
        template_isophotes (list of dict): Pre-validated, SMA-sorted list of
            isophote dicts from ``_resolve_template()``.
        variance_map (np.ndarray, optional): 2D per-pixel variance map. When
            provided, forced photometry uses WLS (weighted mean intensity and
            propagated uncertainty).

    Returns:
        dict: Results dictionary with 'isophotes' (list of dicts) and 'config'
            (IsosterConfig) keys. The isophotes have intensity from the target
            image but geometry from the template.

    Notes
    -----
    The output isophotes preserve the template's geometry exactly. Only the
    intensity-related fields (intens, rms, intens_err) and derived quantities
    come from the target image.

    This is designed for multiband analysis where one band (e.g., g-band)
    defines the geometry, and the same geometry is applied to other bands
    (r, i, z) for consistent color profile measurement.
    """
    from .fitting import extract_forced_photometry

    isophotes = []
    for template_iso in template_isophotes:
        sma = template_iso["sma"]

        # Handle central pixel (sma=0) specially
        if sma == 0:
            iso = fit_central_pixel(
                image,
                mask,
                template_iso["x0"],
                template_iso["y0"],
                debug=config.debug,
                config=config,
                variance_map=variance_map,
            )
        else:
            iso = extract_forced_photometry(
                image,
                mask,
                template_iso["x0"],
                template_iso["y0"],
                sma,
                template_iso["eps"],
                template_iso["pa"],
                integrator=config.integrator,
                sclip=config.sclip,
                nclip=config.nclip,
                use_eccentric_anomaly=config.use_eccentric_anomaly,
                config=config,
                variance_map=variance_map,
                variance_map_prepared=True,
            )
        isophotes.append(iso)

    return _build_fit_result(isophotes, config)
