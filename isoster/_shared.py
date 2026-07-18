"""
Shared internal helpers used by both single-band (`isoster.fitting`,
`isoster.plotting`, `isoster.utils`) and multi-band (`isoster.multiband.*`).

This module is the **single source of truth** for the small utilities
that both code paths need. Single-band re-imports the names back into
their historical locations (`isoster.fitting._tikhonov_alpha`,
`isoster.utils._build_config_hdu`, …) for backward compatibility, so
external code that referenced the old paths continues to work without
change.

The leading underscore on the module name (and on the helper names) is
kept to signal that this is an internal API — it is not part of the
public `isoster` package surface and may evolve without notice. The
intent is *only* to give multi-band a stable, documented import path
that doesn't reach into single-band's private modules: doing so used to
make any rename of a single-band private helper silently break the
multi-band code path.

Membership criteria for adding a new helper here:

* Must be needed by both single-band and multi-band paths.
* Must not depend on either path's config / driver state. The helpers
  here are pure (numpy, math, json, FITS) — anything that needs an
  `IsosterConfig` or `IsosterConfigMB` belongs in the path-specific
  module.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from astropy.io import fits
from astropy.table import Table

logger = logging.getLogger(__name__)

# Sentinel written into variance maps for NaN/inf entries (driver.py): large
# enough to give the sample effectively zero weight in WLS solves. Error
# formulas must be built from inverse-variance weights so the sentinel (and
# values smeared into neighboring samples by interpolation) contribute ~zero
# weight; plain sums of variances would explode (review finding B2).
VARIANCE_SENTINEL = 1e30


def _weighted_mean_variance(variances):
    """Variance of an inverse-variance-weighted mean, 1/sum(1/var_i).

    Exact for the inverse-variance-weighted mean estimator; equals
    ``sum(var_i)/N^2`` (the unweighted mean's variance) for uniform
    variances. Sentinel-immune by construction: the driver's unknown-variance
    sentinel (and values smeared into neighboring samples by interpolation)
    contribute ~zero weight, so one bad variance-map pixel cannot blow up
    the error (review finding B2). Returns np.inf when no sample carries
    finite weight.
    """
    wsum = np.sum(1.0 / variances)
    if not np.isfinite(wsum) or wsum <= 0:
        return np.inf
    return float(1.0 / wsum)


# ---------------------------------------------------------------------------
# Mask / Tikhonov helpers (originally in isoster.fitting)
# ---------------------------------------------------------------------------


def _prepare_mask_float(mask):
    """Convert a boolean/integer mask to float64 once for map_coordinates.

    ``scipy.ndimage.map_coordinates`` requires a floating-point array.
    Converting a large boolean mask on every call is the dominant cost
    when a mask is supplied. Pre-converting here avoids repeated
    allocation inside the per-iteration sampling loop.

    Returns ``None`` when the input is ``None`` so callers can pass it
    through unchanged.
    """
    if mask is None:
        return None
    if mask.dtype.kind == "f":
        return mask
    return mask.astype(np.float64)


def _tikhonov_alpha(coeff, lambda_sma, weight):
    """Return the Tikhonov blend fraction in [0, 1].

    Solves the closed-form mix between the harmonic-driven update and a
    pull toward the reference under the per-iteration objective

        L = 0.5 * (harmonic residual)^2 + 0.5 * lambda * w * (param - param_ref)^2

    At the minimum, each axis' step is

        delta_param = (1 - alpha) * delta_harmonic  -  alpha * (param - param_ref)

    with ``alpha = lambda * w * coeff^2 / (1 + lambda * w * coeff^2)``,
    where ``coeff`` is the harmonic-to-parameter Jacobian already
    computed by the solver (so ``delta_harmonic = coeff * harmonic_amp``,
    or its sign-wrapped equivalent). ``alpha = 0`` recovers the
    unregularized step exactly; ``alpha → 1`` in the limit of vanishing
    gradient (``|coeff| → ∞``) or very strong regularization, fully
    pulling the parameter to its reference.

    Parameters
    ----------
    coeff : float
        Parameter's harmonic Jacobian coefficient. Larger absolute value
        means the fit has less local information about this parameter,
        and ``alpha → 1`` even at modest ``lambda·w``.
    lambda_sma : float
        Ramp value at this sma (logistic in sma).
    weight : float
        Per-axis weight from ``config.outer_reg_weights``.

    Returns
    -------
    float
        The blend fraction, clamped to ``[0, 1)``.
    """
    if weight <= 0.0 or lambda_sma <= 0.0:
        return 0.0
    coeff_sq = coeff * coeff
    if coeff_sq == 0.0 or not np.isfinite(coeff_sq):
        return 0.0
    denom = 1.0 + lambda_sma * weight * coeff_sq
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return lambda_sma * weight * coeff_sq / denom


# ---------------------------------------------------------------------------
# Plotting helper (originally in isoster.plotting)
# ---------------------------------------------------------------------------


def _normalize_harmonic_for_plot(harm, sma, grad, intens):
    """Bender-convention normalization for plotting: ``-A_n / (a · dI/da)``.

    This is the plotting counterpart to
    ``benchmarks.exhausted.analysis.scenario_summary.normalize_harmonic``.
    The two agree on the formula but the plot version is deliberately
    permissive about masking: it only drops non-finite values and zero
    denominators, so the viewer can see the noise distribution. The
    audit version uses a relative floor to suppress points near profile
    turnovers.

    Gradient fallback: when ``grad`` is non-finite on a point, use the
    local finite-difference of ``intens`` vs ``sma`` (required for the
    autoprof tool, which ships ``grad == NaN``).

    Returns a float array the same shape as ``harm``, with NaN where
    the normalization is undefined.
    """
    harm = np.asarray(harm, dtype=float)
    sma = np.asarray(sma, dtype=float)
    grad = np.asarray(grad, dtype=float)
    intens = np.asarray(intens, dtype=float)
    # Local finite-difference fallback for rows missing grad.
    if np.isfinite(sma).sum() >= 2 and np.isfinite(intens).sum() >= 2:
        # Sort by sma, unique-ify, compute np.gradient, map back.
        order = np.argsort(sma)
        s_sorted = sma[order]
        v_sorted = intens[order]
        good = np.isfinite(s_sorted) & np.isfinite(v_sorted)
        if good.sum() >= 2:
            uniq = np.concatenate(([True], np.diff(s_sorted[good]) > 0))
            s_u = s_sorted[good][uniq]
            v_u = v_sorted[good][uniq]
            if s_u.size >= 2:
                g_u = np.gradient(v_u, s_u)
                fallback = np.full_like(sma, np.nan)
                idx_back = np.where(good)[0][uniq]
                fallback[order[idx_back]] = g_u
                grad = np.where(np.isfinite(grad), grad, fallback)
    denom = sma * grad
    out = np.full_like(harm, np.nan, dtype=float)
    valid = np.isfinite(harm) & np.isfinite(denom) & (denom != 0.0)
    out[valid] = -harm[valid] / denom[valid]
    return out


# ---------------------------------------------------------------------------
# FITS / config-dict helpers (originally in isoster.utils)
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _config_to_dict(config) -> dict:
    """Convert a config object (Pydantic model, dict, or generic) to a plain dict.

    Returns an empty dict if ``config`` is ``None``.
    """
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "dict"):
        return config.dict()
    if isinstance(config, dict):
        return config
    return getattr(config, "__dict__", {})


def _build_config_hdu(results) -> fits.BinTableHDU:
    """Build a CONFIG BinTableHDU with PARAM/VALUE rows.

    Each value is JSON-serialized so that lists, dicts, bools, and
    ``None`` round-trip faithfully without FITS header length or
    HIERARCH issues. Used by both single-band
    ``isophote_results_to_fits`` and multi-band
    ``isophote_results_mb_to_fits``.
    """
    config = results.get("config", None) if isinstance(results, dict) else None
    config_dict = _config_to_dict(config)

    params: list = []
    values: list = []
    for key, value in config_dict.items():
        params.append(key)
        values.append(json.dumps(value, cls=_NumpyEncoder))

    config_tbl = Table()
    config_tbl["PARAM"] = params if params else np.array([], dtype="U1")
    config_tbl["VALUE"] = values if values else np.array([], dtype="U1")

    config_hdu = fits.table_to_hdu(config_tbl)
    config_hdu.name = "CONFIG"
    return config_hdu


def _build_meta_hdu(meta: dict) -> fits.BinTableHDU:
    """Build a META BinTableHDU with PARAM/VALUE rows from a plain dict.

    Same JSON pattern as :func:`_build_config_hdu`, for non-structural
    top-level result keys (lock-state metadata, first-isophote
    diagnostics, sky offsets, outer-reg references). Non-serializable
    values are skipped with a warning.
    """
    params: list = []
    values: list = []
    for key, value in meta.items():
        try:
            values.append(json.dumps(value, cls=_NumpyEncoder))
        except (TypeError, ValueError):
            logger.warning("Skipping non-serializable META key %r", key)
            continue
        params.append(key)
    tbl = Table()
    tbl["PARAM"] = params if params else np.array([], dtype="U1")
    tbl["VALUE"] = values if values else np.array([], dtype="U1")
    hdu = fits.table_to_hdu(tbl)
    hdu.name = "META"
    return hdu


def _parse_meta_hdu(hdu: fits.BinTableHDU) -> dict:
    """Parse a META HDU (JSON PARAM/VALUE rows) back into a plain dict."""
    meta: dict = {}
    tbl = Table.read(hdu)
    for row in tbl:
        key = str(row["PARAM"])
        try:
            meta[key] = json.loads(row["VALUE"])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Skipping unparseable META key '%s'", key)
    return meta
