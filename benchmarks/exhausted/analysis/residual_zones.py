"""Inner / mid / outer residual-zone statistics.

Ported from ``sga_isoster/scripts/analyze_photutils.py``. The three
zones are defined by the elliptical radius along the galaxy's long
axis normalized to a reference length ``R_ref`` (pixels):

- ``inner``: ``r_ell < 0.5 * R_ref``
- ``mid``:   ``0.5 * R_ref <= r_ell < 2 * R_ref``
- ``outer``: ``r_ell >= 2 * R_ref``

``R_ref`` is ``effective_Re_pix`` when available; otherwise we fall
back to ``0.25 * maxsma`` so the zones still carve the image into
three bands (the fallback is flagged as ``fallback_ref`` in the
returned dict so downstream code can note it).

The reference ellipse is anchored at the galaxy's initial geometry
(adapter-supplied ``x0, y0``, ``eps``, ``pa``). Using a fit-derived
geometry would make the zones arm-dependent and confound cross-arm
comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ZoneStats = dict[str, Any]
DEFAULT_INNER_CUT_PIX = 2.0


def evaluation_aperture(
    radius: np.ndarray, outer_radius: float, inner_cut_pix: float = DEFAULT_INNER_CUT_PIX
) -> np.ndarray:
    """Fixed evaluation cut, independent of physical PSF and fitting radii."""
    if not np.isfinite(inner_cut_pix) or inner_cut_pix < 0:
        raise ValueError("Inner radial cut must be finite and non-negative")
    return (radius >= inner_cut_pix) & (radius <= outer_radius)


def compute_elliptical_radius_grid(
    image_shape: tuple[int, int],
    x0: float,
    y0: float,
    eps: float,
    pa: float,
) -> np.ndarray:
    """Return the elliptical-radius map (same shape as the image).

    For each pixel ``(x, y)``:
        x_rot = (x - x0) cos(pa) + (y - y0) sin(pa)
        y_rot = -(x - x0) sin(pa) + (y - y0) cos(pa)
        r_ell = sqrt(x_rot^2 + (y_rot / (1 - eps))^2)
    """
    h, w = image_shape
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = x_grid - float(x0)
    dy = y_grid - float(y0)
    cos_pa = float(np.cos(pa))
    sin_pa = float(np.sin(pa))
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    axis_ratio = max(1.0 - float(eps), 1e-3)
    return np.sqrt(x_rot**2 + (y_rot / axis_ratio) ** 2)


def zone_masks(r_ell: np.ndarray, R_ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(inner, mid, outer)`` boolean masks for the given reference length."""
    inner = r_ell < 0.5 * R_ref
    outer = r_ell >= 2.0 * R_ref
    mid = (~inner) & (~outer)
    return inner, mid, outer


def _stats_on_mask(
    residual: np.ndarray,
    mask_zone: np.ndarray,
    mask_valid: np.ndarray,
    noise_sigma: float | None,
) -> dict[str, float]:
    """Return ``{rms, median, frac_above_3sigma}`` for one zone."""
    pixels = residual[mask_zone & mask_valid]
    if pixels.size == 0:
        return {
            "rms": float("nan"),
            "median": float("nan"),
            "frac_above_3sigma": float("nan"),
        }
    rms = float(np.sqrt(np.mean(pixels**2)))
    median = float(np.median(pixels))
    sigma = noise_sigma if noise_sigma is not None and noise_sigma > 0 else rms
    if sigma <= 0 or not np.isfinite(sigma):
        frac_3s = float("nan")
    else:
        frac_3s = float(np.sum(np.abs(pixels) > 3.0 * sigma) / pixels.size)
    return {"rms": rms, "median": median, "frac_above_3sigma": frac_3s}


def residual_zone_stats(
    image: np.ndarray,
    model: np.ndarray,
    x0: float,
    y0: float,
    eps: float,
    pa: float,
    *,
    R_ref: float | None,
    maxsma: float,
    mask: np.ndarray | None = None,
    noise_sigma: float | None = None,
) -> ZoneStats:
    """Compute residual statistics in three elliptical zones.

    Returns a flat dict with:

        resid_rms_inner / resid_rms_mid / resid_rms_outer
        resid_median_inner / resid_median_mid / resid_median_outer
        frac_above_3sigma_outer            (only reported for the outer zone)
        R_ref_pix                          (actual reference length used)
        zone_npix_inner / mid / outer
        zone_fallback_ref                  ("" or "Re_missing")

    All values are finite floats or ``nan``.
    """
    image = np.asarray(image, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    residual = image - model

    if R_ref is not None and np.isfinite(R_ref) and R_ref > 0:
        actual_R_ref = float(R_ref)
        fallback = ""
    else:
        actual_R_ref = float(max(1.0, 0.25 * maxsma))
        fallback = "Re_missing"

    r_ell = compute_elliptical_radius_grid(image.shape, x0, y0, eps, pa)
    inner, mid, outer = zone_masks(r_ell, actual_R_ref)

    valid = np.isfinite(residual) & (r_ell >= DEFAULT_INNER_CUT_PIX)
    if mask is not None:
        valid &= ~np.asarray(mask, dtype=bool)

    inner_stats = _stats_on_mask(residual, inner, valid, noise_sigma)
    mid_stats = _stats_on_mask(residual, mid, valid, noise_sigma)
    outer_stats = _stats_on_mask(residual, outer, valid, noise_sigma)

    return {
        "resid_rms_inner": inner_stats["rms"],
        "resid_rms_mid": mid_stats["rms"],
        "resid_rms_outer": outer_stats["rms"],
        "resid_median_inner": inner_stats["median"],
        "resid_median_mid": mid_stats["median"],
        "resid_median_outer": outer_stats["median"],
        "frac_above_3sigma_outer": outer_stats["frac_above_3sigma"],
        "R_ref_pix": actual_R_ref,
        "inner_cut_pix": DEFAULT_INNER_CUT_PIX,
        "zone_npix_inner": int(np.sum(inner & valid)),
        "zone_npix_mid": int(np.sum(mid & valid)),
        "zone_npix_outer": int(np.sum(outer & valid)),
        "zone_fallback_ref": fallback,
    }
