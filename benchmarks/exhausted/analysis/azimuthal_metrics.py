"""Azimuthal pattern metrics for residual maps (contract v1.1).

Captures the *pattern* of structure in residual maps that the
amplitude-based metrics in ``per_arm_metrics.metrics_from_residual``
cannot distinguish: dipole vs quadrupole vs high-order rings.

Promoted bundle (Phase 1.6 demo, see
``tests/integration/data/azimuthal_metrics_demo/summary.md``):

  * azim_A1_{zone}            dipole significance (σ units)
  * azim_A2_{zone}            quadrupole significance (σ units)
  * azim_struct_ratio_{zone}  low-m vs high-m power ratio
  * azim_phase1_deg_{zone}    direction of m=1 mode (descriptive)
  * quadrant_imbalance_{zone} (max Q − min Q) / sum(|r|) over 4
                              azimuthal quadrants

Dropped after the demo: ``azim_A4_{zone}`` (correlated with
struct_ratio) and ``asym180_{zone}`` (no discrimination on the
outer zone, expensive).

Conventions
-----------
- Elliptical coordinates anchored at the catalog initial geometry.
- Azimuthal angle phi is measured CCW from the major axis, range
  [0, 2*pi). Computed in the rotated/scaled frame so each annulus is
  a true ring in elliptical-r space.
- All operations use only finite, unmasked residual pixels.
- LOWER = better residual (less structured) for every metric except
  ``azim_phase1_deg`` which is descriptive only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .residual_zones import DEFAULT_INNER_CUT_PIX

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _ellipse_coords(
    yy: np.ndarray,
    xx: np.ndarray,
    x0: float,
    y0: float,
    eps: float,
    pa_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_ell, phi) in elliptical coords.

    ``r_ell`` is the same elliptical radius as in ``per_arm_metrics``.
    ``phi`` is the azimuthal angle in the rotated frame, in
    [-pi, pi); take ``phi % (2 pi)`` for [0, 2 pi).
    """
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    dx = xx - x0
    dy = yy - y0
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    b_over_a = max(1e-3, 1.0 - float(eps))
    # phi computed on the un-de-projected rotated frame so the major
    # axis is at phi=0 and bins are uniform around the ellipse axes.
    phi = np.arctan2(y_rot / b_over_a, x_rot)
    r_ell = np.sqrt(x_rot * x_rot + (y_rot / b_over_a) ** 2)
    return r_ell, phi


# ---------------------------------------------------------------------------
# Azimuthal Fourier decomposition
# ---------------------------------------------------------------------------


@dataclass
class _ZoneFourier:
    """Per-zone Fourier decomposition output."""

    a0: float
    am: np.ndarray  # shape (M,), m=1..M
    bm: np.ndarray  # shape (M,)
    Am: np.ndarray  # shape (M,) = sqrt(a^2 + b^2)
    sigma_bin: float
    n_bins_used: int
    total_pixels: int


def _fourier_decompose(
    residual: np.ndarray,
    phi: np.ndarray,
    zone_mask: np.ndarray,
    n_phi: int = 36,
    m_max: int = 6,
) -> _ZoneFourier | None:
    """Bin pixels by phi, compute mean residual per bin, fit Fourier modes.

    Returns None when fewer than ``m_max + 1`` bins have any pixels
    (the linear system would be under-determined).
    """
    z = zone_mask & np.isfinite(residual)
    if not np.any(z):
        return None
    r_vals = residual[z]
    phi_vals = phi[z] % (2.0 * np.pi)

    # Bin edges in phi
    edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    bin_idx = np.minimum(np.searchsorted(edges, phi_vals, side="right") - 1, n_phi - 1)
    bin_idx = np.clip(bin_idx, 0, n_phi - 1)

    bin_sum = np.zeros(n_phi)
    bin_cnt = np.zeros(n_phi, dtype=np.int64)
    bin_sumsq = np.zeros(n_phi)
    np.add.at(bin_sum, bin_idx, r_vals)
    np.add.at(bin_cnt, bin_idx, 1)
    np.add.at(bin_sumsq, bin_idx, r_vals * r_vals)

    used = bin_cnt > 0
    n_used = int(np.sum(used))
    if n_used < m_max + 1:
        return None

    bin_mean = np.full(n_phi, np.nan)
    bin_mean[used] = bin_sum[used] / bin_cnt[used]
    bin_centres = 0.5 * (edges[:-1] + edges[1:])

    # Per-pixel sigma in the zone (robust MAD on the full pixel set).
    med = float(np.median(r_vals))
    sigma_pix = float(1.4826 * np.median(np.abs(r_vals - med)))
    # Bin-mean sigma scales as sigma_pix / sqrt(<n_per_bin>).
    mean_n = float(np.mean(bin_cnt[used]))
    sigma_bin = sigma_pix / np.sqrt(max(mean_n, 1.0))

    # Linear least squares fit of a0 + sum(am cos(m phi) + bm sin(m phi))
    # using only used bins.
    phi_used = bin_centres[used]
    y_used = bin_mean[used]
    cols = [np.ones_like(phi_used)]
    for m in range(1, m_max + 1):
        cols.append(np.cos(m * phi_used))
        cols.append(np.sin(m * phi_used))
    design = np.stack(cols, axis=1)
    # Weighted lstsq (weights = sqrt(bin_cnt)) to honour pixel counts.
    w = np.sqrt(bin_cnt[used].astype(np.float64))
    coeffs, *_ = np.linalg.lstsq(design * w[:, None], y_used * w, rcond=None)

    a0 = float(coeffs[0])
    am = coeffs[1::2].astype(np.float64)
    bm = coeffs[2::2].astype(np.float64)
    Am = np.sqrt(am * am + bm * bm)
    return _ZoneFourier(
        a0=a0,
        am=am,
        bm=bm,
        Am=Am,
        sigma_bin=sigma_bin,
        n_bins_used=n_used,
        total_pixels=int(z.sum()),
    )


# ---------------------------------------------------------------------------
# Pixel-level pattern metrics
# ---------------------------------------------------------------------------


def _quadrant_imbalance(
    residual: np.ndarray,
    phi: np.ndarray,
    zone_mask: np.ndarray,
) -> float:
    """(max Q - min Q) / sum(|r|) over 4 azimuthal quadrants.

    Quadrants defined relative to the major axis:
        Q1: [-pi/4, +pi/4)
        Q2: [+pi/4, +3pi/4)
        Q3: [+3pi/4, pi) U [-pi, -3pi/4)
        Q4: [-3pi/4, -pi/4)

    Returns the dimensionless ratio of the largest signed-flux
    difference between quadrants over the total |residual| flux in
    the zone. NaN when no pixels.
    """
    z = zone_mask & np.isfinite(residual)
    if not np.any(z):
        return float("nan")
    r_vals = residual[z]
    p_vals = phi[z]
    p_wrapped = np.mod(p_vals + np.pi / 4.0, 2.0 * np.pi)
    q_idx = np.minimum((p_wrapped // (np.pi / 2.0)).astype(np.int64), 3)
    q_sum = np.zeros(4)
    np.add.at(q_sum, q_idx, r_vals)
    denom = float(np.sum(np.abs(r_vals)))
    if denom <= 0:
        return float("nan")
    return float(q_sum.max() - q_sum.min()) / denom


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def azimuthal_metrics(
    residual: np.ndarray,
    *,
    x0: float,
    y0: float,
    eps: float,
    pa_rad: float,
    R_ref_pix: float,
    maxsma_pix: float,
    r_inner_floor_pix: float = DEFAULT_INNER_CUT_PIX,
    n_phi: int = 36,
    m_max: int = 4,
    return_timing: bool = False,
) -> dict[str, Any]:
    """Compute the contract-v1.1 azimuthal pattern metrics per zone.

    The ``residual`` array MUST already be masked (NaN at masked
    pixels) — this function does not apply masks.

    Returns a flat dict with per-zone keys:

        azim_A1_{zone}              dipole, σ units
        azim_A2_{zone}              quadrupole, σ units
        azim_struct_ratio_{zone}    low-m / high-m power
        azim_phase1_deg_{zone}      direction of m=1 mode (descriptive)
        quadrant_imbalance_{zone}   (max Q - min Q) / sum(|r|) over 4 quadrants

    Plus per-metric timing in ``_timing_s`` if ``return_timing`` is
    True.
    """
    ny, nx = residual.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_ell, phi = _ellipse_coords(yy, xx, x0, y0, eps, pa_rad)

    inner_floor = max(0.0, float(r_inner_floor_pix))
    inner_max = 0.5 * R_ref_pix
    mid_max = 1.5 * R_ref_pix
    outer_max = min(float(maxsma_pix), 3.0 * R_ref_pix)
    zones = {
        "inner": (r_ell >= inner_floor) & (r_ell < inner_max),
        "mid": (r_ell >= max(inner_floor, inner_max)) & (r_ell < mid_max),
        "outer": (r_ell >= max(inner_floor, mid_max)) & (r_ell < outer_max),
    }

    out: dict[str, Any] = {}
    timing: dict[str, float] = {}
    for name, zmask in zones.items():
        # Fourier decomposition (covers A1, A2, struct_ratio, phase1).
        t0 = time.perf_counter()
        fdec = _fourier_decompose(residual, phi, zmask, n_phi=n_phi, m_max=m_max)
        timing[f"fourier_{name}"] = time.perf_counter() - t0

        if fdec is None or fdec.sigma_bin <= 0:
            for k in (
                f"azim_A1_{name}",
                f"azim_A2_{name}",
                f"azim_struct_ratio_{name}",
                f"azim_phase1_deg_{name}",
            ):
                out[k] = float("nan")
        else:
            sig = fdec.sigma_bin
            out[f"azim_A1_{name}"] = float(fdec.Am[0]) / sig
            out[f"azim_A2_{name}"] = float(fdec.Am[1]) / sig

            low = float(fdec.am[0] ** 2 + fdec.bm[0] ** 2 + fdec.am[1] ** 2 + fdec.bm[1] ** 2)
            if m_max >= 3:
                high = float(np.sum(fdec.am[2:] ** 2 + fdec.bm[2:] ** 2))
            else:
                high = 0.0
            out[f"azim_struct_ratio_{name}"] = (low / high) if high > 0 else float("nan")
            out[f"azim_phase1_deg_{name}"] = float(np.rad2deg(np.arctan2(fdec.bm[0], fdec.am[0])))

        t0 = time.perf_counter()
        out[f"quadrant_imbalance_{name}"] = _quadrant_imbalance(residual, phi, zmask)
        timing[f"quadrant_{name}"] = time.perf_counter() - t0

    if return_timing:
        out["_timing_s"] = timing
    return out


__all__ = ["azimuthal_metrics"]
