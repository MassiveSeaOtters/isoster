"""Per-zone residual amplitude metrics (contract v1.1).

Tool-agnostic: works for any algorithm that produces a 2-D model
image alongside the data and an object mask. Pure ``numpy`` +
``astropy.io.fits``.

Implements the *amplitude* half of the residual-metric contract:

- per-zone RMS and median of the residual map;
- per-zone robust sigma via 1.4826 * MAD (decoupled from the
  frame-wide noise so that central misfit doesn't pollute outer-zone
  noise estimates);
- per-zone integrated absolute residual normalised by a positive-
  clipped reference flux ``F_ref`` (= sum of ``max(data, 0)`` inside
  ``r < R_ref``);
- per-zone integrated absolute residual in robust-sigma units
  (mean(|r|) / sigma_zone).

For the *pattern* half (Fourier modes, quadrant imbalance) see
``azimuthal_metrics.py``. For profile-derived scalar metrics
(``max_dpa_deg``, ``min_sma_pix``, etc.) see ``profile_io.py``.

Geometry contract
-----------------

All metrics live in elliptical coordinates anchored at a *fixed*
geometry the caller provides -- typically the catalog initial
geometry. Coupling the metric to the fitted geometry would defeat
its purpose (you would be measuring how well the model captures
the geometry it is being judged against).

Zones, with ``R_ref`` a per-galaxy reference radius (e.g. ``D25/2``,
half-light radius, R_e):

    inner: r_inner_floor <= r < 0.5 * R_ref
    mid:   0.5  R_ref    <= r < 1.5 * R_ref
    outer: 1.5  R_ref    <= r < min(maxsma, 3 * R_ref)

``r_inner_floor`` defaults to a fixed 2 pixels for all tools and all zones,
independent of physical PSF FWHM and fitted radial sampling.
``maxsma`` caps the outer zone so the metric does not extend into
pure-sky territory.

Mask invariant
--------------

Every metric uses **finite, unmasked** pixels only. Masked pixels
are set to NaN BEFORE any sum, mean, MAD, or Fourier operation.
Provide an object mask whenever possible -- foreground stars inside
the galaxy footprint cause large spurious residuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .residual_zones import DEFAULT_INNER_CUT_PIX


def _elliptical_radius(
    yy: np.ndarray,
    xx: np.ndarray,
    x0: float,
    y0: float,
    eps: float,
    pa_rad: float,
) -> np.ndarray:
    """Math-convention PA (CCW from +x), eps = 1 - b/a."""
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    dx = xx - x0
    dy = yy - y0
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    b_over_a = max(1e-3, 1.0 - float(eps))
    return np.sqrt(x_rot * x_rot + (y_rot / b_over_a) ** 2)


def _zone_sigma(values: np.ndarray) -> float:
    """Robust sigma via 1.4826 * MAD on finite values; NaN if no data."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    med = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - med)))


def metrics_from_residual(
    image: np.ndarray,
    model_path: str | Path,
    *,
    mask: np.ndarray | None,
    x0: float,
    y0: float,
    eps: float,
    pa_rad: float,
    R_ref_pix: float | None,
    maxsma_pix: float,
    r_inner_floor_pix: float = DEFAULT_INNER_CUT_PIX,
) -> dict[str, Any]:
    """Compute the per-zone amplitude metrics.

    Parameters
    ----------
    image
        Data image (2-D, float).
    model_path
        Path to the model FITS (the function reads HDU 0 directly).
        Must match ``image.shape``.
    mask
        Boolean mask (True = masked). Same shape as ``image``.
        ``None`` means no mask (not recommended for real data).
    x0, y0, eps, pa_rad
        Fixed elliptical geometry (catalog initial geometry).
    R_ref_pix
        Reference radius in pixels -- mandatory in spirit. ``None``
        falls back to ``max(10, maxsma_pix / 4)`` as a safety net.
        For SGA-2020 use ``D25 / 2``; for arbitrary cutouts supply
        your own (effective radius, half-light, R26 -- whatever the
        sample uses).
    maxsma_pix
        Largest fitted sma. Caps the outer zone at
        ``min(maxsma_pix, 3 * R_ref_pix)``.
    r_inner_floor_pix
        Fixed evaluation floor in pixels, default 2. Independent of PSF
        and fitting radii. Applies to every zone.

    Returns
    -------
    dict[str, Any]
        Keys::

            F_ref, R_ref_used_pix, r_inner_floor_pix
            resid_rms_{inner,mid,outer}
            resid_median_{inner,mid,outer}
            sigma_{inner,mid,outer}
            abs_resid_over_ref_{inner,mid,outer}      -- in percent
            abs_resid_over_sigma_{inner,mid,outer}    -- dimensionless
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        return {}

    with fits.open(model_path) as hdul:
        model = np.asarray(hdul[0].data, dtype=np.float64)
    return metrics_from_model_array(
        image,
        model,
        mask=mask,
        x0=x0,
        y0=y0,
        eps=eps,
        pa_rad=pa_rad,
        R_ref_pix=R_ref_pix,
        maxsma_pix=maxsma_pix,
        r_inner_floor_pix=r_inner_floor_pix,
    )


def metrics_from_model_array(
    image: np.ndarray,
    model: np.ndarray,
    *,
    mask: np.ndarray | None,
    x0: float,
    y0: float,
    eps: float,
    pa_rad: float,
    R_ref_pix: float | None,
    maxsma_pix: float,
    r_inner_floor_pix: float = DEFAULT_INNER_CUT_PIX,
) -> dict[str, Any]:
    """Compute the per-zone amplitude metrics from an in-memory model.

    Local adapter for benchmark fitters that already have the rendered
    model array. Semantics are identical to :func:`metrics_from_residual`.
    """
    if model.shape != image.shape:
        return {"error_message_resid": (f"shape mismatch image{image.shape} vs model{model.shape}")}

    data = np.asarray(image, dtype=np.float64)
    residual = data - model
    if mask is not None:
        residual = np.where(mask, np.nan, residual)
        data_masked = np.where(mask, np.nan, data)
    else:
        data_masked = data

    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = _elliptical_radius(yy, xx, x0, y0, eps, pa_rad)

    if R_ref_pix is None or R_ref_pix <= 0:
        R_ref_pix = max(10.0, float(maxsma_pix) / 4.0)

    inner_floor = max(0.0, float(r_inner_floor_pix))
    inner_max = 0.5 * R_ref_pix
    mid_max = 1.5 * R_ref_pix
    outer_max = min(float(maxsma_pix), 3.0 * R_ref_pix)

    # Reference flux: positive-clipped sum of data inside r < R_ref,
    # excluding masked pixels. Same denominator for all three zones,
    # so the % values are commensurate across zones.
    ref_zone = (r < R_ref_pix) & np.isfinite(data_masked)
    f_ref = float(np.nansum(np.clip(data_masked[ref_zone], 0.0, None)))

    out: dict[str, Any] = {
        "F_ref": f_ref,
        "r_inner_floor_pix": inner_floor,
        "R_ref_used_pix": float(R_ref_pix),
    }

    zones = {
        "inner": (r >= inner_floor) & (r < inner_max),
        "mid": (r >= max(inner_floor, inner_max)) & (r < mid_max),
        "outer": (r >= max(inner_floor, mid_max)) & (r < outer_max),
    }

    finite_resid = np.isfinite(residual)
    for name, zone in zones.items():
        z = zone & finite_resid
        if not np.any(z):
            out[f"resid_rms_{name}"] = float("nan")
            out[f"resid_median_{name}"] = float("nan")
            out[f"sigma_{name}"] = float("nan")
            out[f"abs_resid_over_ref_{name}"] = float("nan")
            out[f"abs_resid_over_sigma_{name}"] = float("nan")
            continue
        v = residual[z]
        out[f"resid_rms_{name}"] = float(np.sqrt(np.mean(v * v)))
        out[f"resid_median_{name}"] = float(np.median(v))

        sigma_z = _zone_sigma(v)
        out[f"sigma_{name}"] = sigma_z

        sum_abs_resid = float(np.sum(np.abs(v)))
        if f_ref > 0 and np.isfinite(f_ref):
            out[f"abs_resid_over_ref_{name}"] = 100.0 * sum_abs_resid / f_ref
        else:
            out[f"abs_resid_over_ref_{name}"] = float("nan")

        if sigma_z > 0 and np.isfinite(sigma_z):
            out[f"abs_resid_over_sigma_{name}"] = float(np.mean(np.abs(v)) / sigma_z)
        else:
            out[f"abs_resid_over_sigma_{name}"] = float("nan")

    return out


__all__ = ["metrics_from_residual", "metrics_from_model_array"]
