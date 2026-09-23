"""Benchmark-local v1.1 model evaluation adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from .azimuthal_metrics import azimuthal_metrics
from .profile_io import profile_summary
from .residual_metrics import metrics_from_model_array
from .residual_zones import DEFAULT_INNER_CUT_PIX


def evaluate_model_v11(
    *,
    image: np.ndarray,
    model: np.ndarray,
    mask: np.ndarray | None,
    x0: float,
    y0: float,
    eps: float,
    pa_rad: float,
    R_ref_pix: float | None,
    maxsma_pix: float,
    r_inner_floor_pix: float = DEFAULT_INNER_CUT_PIX,
) -> dict[str, Any]:
    """Return the combined v1.1 amplitude and azimuthal metric contract."""
    residual_metrics = metrics_from_model_array(
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

    residual = np.asarray(image, dtype=np.float64) - np.asarray(model, dtype=np.float64)
    if mask is not None:
        residual = np.where(np.asarray(mask, dtype=bool), np.nan, residual)
    R_ref_used = float(residual_metrics.get("R_ref_used_pix", R_ref_pix or max(10.0, maxsma_pix / 4.0)))
    azimuthal = azimuthal_metrics(
        residual,
        x0=x0,
        y0=y0,
        eps=eps,
        pa_rad=pa_rad,
        R_ref_pix=R_ref_used,
        maxsma_pix=maxsma_pix,
        r_inner_floor_pix=r_inner_floor_pix,
    )
    return {**residual_metrics, **azimuthal}


def profile_summary_for_inventory(profile_path: str) -> dict[str, Any]:
    """Read a written profile FITS and return v1.1 profile scalar metrics."""
    return profile_summary(profile_path)


__all__ = ["evaluate_model_v11", "profile_summary_for_inventory"]
