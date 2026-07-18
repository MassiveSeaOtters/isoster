"""
Composite QA figure for multi-band isoster results.

Layout (decision D15, refined 2026-04-30):

* Top-left block — SB profile (one big panel, asinh-SB Lupton+1999).
* Top-right block — image-and-residual mosaic, 2x3 tile:
    cell (0, 0)  reference-band cutout with representative isophotes
    cells (0,1)..(1,2)  per-band residual panels (image - model) with
        the object mask overlaid translucently. Color scale computed
        from unmasked pixels only.
* Bottom-left block — Bender-normalized harmonics A3 / B3 / A4 / B4,
    stacked, physically sharing the x-axis (no inter-panel gap).
* Bottom-right block — geometry stack: eps, PA, center offsets,
    same shared-x layout.

Conventions match the single-band QA style:

* X-axis is ``SMA^0.25`` in arcsec.
* All 1-D points are scatter markers with errorbars (no connecting line).
* Y-axis ranges are computed from data points only (errorbar extents
  do not inflate the limits).
* The asinh-SB recipe is the same one used by the asteris-pair runner;
  per-band ``mu(I=0)`` reference lines are drawn as dashed horizontals.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from .._shared import _normalize_harmonic_for_plot
from ..model import build_isoster_model
from ..plotting import configure_qa_plot_style

logger = logging.getLogger(__name__)


_BAND_COLORS: tuple = (
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
)


def _band_color(band_idx: int) -> str:
    if band_idx < len(_BAND_COLORS):
        return _BAND_COLORS[band_idx]
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#444"])
    return cycle[band_idx % len(cycle)]


# ---------------------------------------------------------------------------
# asinh-SB conversion (Lupton+ 1999) — mirrors the asteris-pair recipe
# ---------------------------------------------------------------------------


def _intens_to_mu_asinh(
    intens: NDArray[np.floating],
    zeropoint: float,
    pixel_scale_arcsec: float,
    scale: float,
) -> NDArray[np.floating]:
    pixarea = pixel_scale_arcsec * pixel_scale_arcsec
    intens = np.asarray(intens, dtype=np.float64)
    i_pp = intens / pixarea
    big_b = scale / pixarea
    return -2.5 / np.log(10.0) * np.arcsinh(i_pp / (2.0 * big_b)) + zeropoint - 2.5 * np.log10(big_b)


def _mu_asinh_error(
    intens: NDArray[np.floating],
    intens_err: NDArray[np.floating],
    scale: float,
) -> NDArray[np.floating]:
    intens = np.asarray(intens, dtype=np.float64)
    intens_err = np.asarray(intens_err, dtype=np.float64)
    denom = np.maximum(np.abs(intens), scale)
    return 2.5 / np.log(10.0) * intens_err / denom


def _asinh_mu_at_zero(zeropoint: float, pixel_scale_arcsec: float, scale: float) -> float:
    """``mu_asinh(I=0) = zp - 2.5 * log10(scale / pixarea)``."""
    pixarea = pixel_scale_arcsec * pixel_scale_arcsec
    return float(zeropoint - 2.5 * np.log10(scale / pixarea))


# ---------------------------------------------------------------------------
# Post-process: subtract per-band outermost-ring sky residual
# ---------------------------------------------------------------------------


def subtract_outermost_sky_offset(
    result: dict,
    n_outer: int = 5,
) -> Tuple[dict, Dict[str, float]]:
    """
    Subtract the per-band outer-ring sky plateau from the joint fit's I0_b.

    The joint solver fits each band's intensity along an isophote as
    ``I0_b + harmonic_terms`` (decision D11). In the LSB outskirt where the
    galaxy signal drops below the per-band sky residual, ``I0_b`` saturates
    at the local sky residual rather than going to zero — a flat plateau
    visible on the SB profile's outermost rings.

    This helper estimates that plateau as the median ``I0_b`` over the
    outermost ``n_outer`` valid isophotes (per band) and returns a copy of
    the result dict with the offset subtracted from every isophote's
    ``intens_<b>``. The corresponding ``intens_err_<b>`` is unchanged.

    Returns
    -------
    corrected_result, sky_offsets
        - ``corrected_result`` is a shallow copy of ``result`` with new
          ``isophotes`` list. The original is not modified.
        - ``sky_offsets`` is a ``{band: offset}`` dict for downstream use
          (e.g., subtracting the same offset from the band image when
          building the residual mosaic).
    """
    bands = list(result.get("bands", []))
    isophotes = list(result["isophotes"])
    valid = [iso for iso in isophotes if bool(iso.get("valid", True)) and float(iso.get("sma", 0.0)) > 0.0]
    valid_sorted = sorted(valid, key=lambda iso: float(iso["sma"]))
    outer = valid_sorted[-n_outer:] if len(valid_sorted) >= n_outer else valid_sorted

    sky_offsets: Dict[str, float] = {}
    for b in bands:
        vals: List[float] = []
        for iso in outer:
            v = iso.get(f"intens_{b}")
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
        sky_offsets[b] = float(np.median(vals)) if vals else 0.0

    new_isophotes: List[dict] = []
    for iso in isophotes:
        new_iso = dict(iso)
        for b in bands:
            key = f"intens_{b}"
            v = new_iso.get(key)
            if v is not None and np.isfinite(float(v)):
                new_iso[key] = float(v) - sky_offsets[b]
        new_isophotes.append(new_iso)

    new_result = dict(result)
    new_result["isophotes"] = new_isophotes
    new_result["sky_offsets"] = sky_offsets
    return new_result, sky_offsets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xaxis_arcsec_pow(sma_pix: NDArray[np.floating], pixel_scale_arcsec: float) -> NDArray[np.floating]:
    sma_arcsec = np.asarray(sma_pix, dtype=np.float64) * float(pixel_scale_arcsec)
    out = np.full_like(sma_arcsec, np.nan, dtype=np.float64)
    good = sma_arcsec > 0
    out[good] = sma_arcsec[good] ** 0.25
    return out


def _set_ylim_from_data(
    ax: Axes,
    *value_arrays: NDArray[np.floating],
    pad_frac: float = 0.08,
) -> None:
    """Set y-limits from the union of data points only (no errorbars)."""
    finite_values: List[float] = []
    for arr in value_arrays:
        a = np.asarray(arr, dtype=np.float64)
        finite_values.extend(a[np.isfinite(a)].tolist())
    if not finite_values:
        return
    lo = float(min(finite_values))
    hi = float(max(finite_values))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    ax.set_ylim(lo - pad_frac * span, hi + pad_frac * span)


def _isophotes_to_per_band_singleband_lists(
    isophotes: Sequence[dict],
    bands: Sequence[str],
    harmonic_orders: Optional[Sequence[int]] = None,
) -> Dict[str, List[dict]]:
    # Orders were hardcoded to (3, 4), silently dropping higher orders from
    # the residual mosaic (review M11). Prefer the explicit list; otherwise
    # detect from the row keys (a{n}_<b>) of the first row that has any.
    if harmonic_orders is None:
        harmonic_orders = []
        for iso in isophotes:
            detected = sorted({int(k[1:].split("_")[0]) for k in iso if k.startswith("a") and k[1:2].isdigit()})
            if detected:
                harmonic_orders = detected
                break
        if not harmonic_orders:
            harmonic_orders = [3, 4]
    out: Dict[str, List[dict]] = {b: [] for b in bands}
    for iso in isophotes:
        if not bool(iso.get("valid", True)):
            continue
        for b in bands:
            row = {
                "sma": float(iso["sma"]),
                "x0": float(iso["x0"]),
                "y0": float(iso["y0"]),
                "eps": float(iso["eps"]),
                "pa": float(iso["pa"]),
                "intens": float(iso.get(f"intens_{b}", float("nan"))),
            }
            for n in harmonic_orders:
                row[f"a{n}"] = float(iso.get(f"a{n}_{b}", 0.0))
                row[f"b{n}"] = float(iso.get(f"b{n}_{b}", 0.0))
            if "use_eccentric_anomaly" in iso:
                row["use_eccentric_anomaly"] = bool(iso["use_eccentric_anomaly"])
            out[b].append(row)
    return out


def _resolve_softening(
    bands: Sequence[str],
    isophotes: Sequence[dict],
    softening_per_band: Optional[Dict[str, float]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for b in bands:
        if softening_per_band is not None and b in softening_per_band:
            out[b] = max(float(softening_per_band[b]), 1e-12)
            continue
        errs = np.array(
            [float(iso.get(f"intens_err_{b}", np.nan)) for iso in isophotes],
            dtype=np.float64,
        )
        finite = errs[np.isfinite(errs) & (errs > 0)]
        out[b] = max(float(np.median(finite)) if finite.size else 1.0, 1e-12)
    return out


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _annotate_sky_offsets(
    ax: Axes,
    bands: Sequence[str],
    sky_offsets: Dict[str, float],
) -> None:
    """Render the inferred per-band sky offsets as a small in-panel text box.

    Each band's offset value (in native flux per pixel) is shown on its own
    line, color-coded with the band palette. The panel reflects the value
    that ``subtract_outermost_sky_offset`` removed from ``intens_<b>`` before
    plotting, so users can read off the inferred sky residual that the
    joint-solver's ``I0_b`` saturated to in the LSB outskirt.
    """
    if not sky_offsets:
        return
    lines = ["Inferred sky"]
    colors = ["#222222"]
    for b_idx, b in enumerate(bands):
        if b not in sky_offsets:
            continue
        v = float(sky_offsets[b])
        lines.append(f"  {b}: {v:+.3e}")
        colors.append(_band_color(b_idx))
    if len(lines) == 1:
        return
    # Anchor in the upper-right of the SB panel; one text per line so each
    # color renders independently. Keep the box small and unobtrusive so it
    # does not eat into the plot area.
    line_h = 0.038
    y0 = 0.985
    for i, (text, color) in enumerate(zip(lines, colors)):
        ax.text(
            0.985,
            y0 - i * line_h,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=color,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.7", alpha=0.85) if i == 0 else None,
        )


def _plot_sb_profile(
    ax: Axes,
    isophotes: Sequence[dict],
    bands: Sequence[str],
    sb_zeropoint: Optional[float],
    pixel_scale_arcsec: Optional[float],
    softening_per_band: Dict[str, float],
    sky_offsets: Optional[Dict[str, float]] = None,
) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    pix = pixel_scale_arcsec if pixel_scale_arcsec is not None else 1.0
    x = _xaxis_arcsec_pow(sma, pix)

    has_sb = sb_zeropoint is not None and pixel_scale_arcsec is not None
    all_y_for_ylim: List[NDArray[np.float64]] = []

    for b_idx, b in enumerate(bands):
        intens = np.array([float(iso.get(f"intens_{b}", np.nan)) for iso in isophotes])
        intens_err = np.array([float(iso.get(f"intens_err_{b}", np.nan)) for iso in isophotes])
        m = valid & np.isfinite(intens) & np.isfinite(x)
        if m.sum() == 0:
            continue
        scale_b = softening_per_band.get(b, 1.0)
        if has_sb:
            mu = _intens_to_mu_asinh(
                intens[m],
                sb_zeropoint,
                pixel_scale_arcsec,
                scale_b,  # type: ignore[arg-type]
            )
            mu_err = _mu_asinh_error(intens[m], intens_err[m], scale_b)
            ax.errorbar(
                x[m],
                mu,
                yerr=mu_err,
                color=_band_color(b_idx),
                label=b,
                marker="o",
                ms=6.0,
                mfc=_band_color(b_idx),
                mec="white",
                mew=0.6,
                lw=0,
                elinewidth=1.4,
                capsize=2.5,
                capthick=1.2,
                alpha=0.95,
            )
            # Per-band I=0 reference line (dashed, same color).
            mu_zero = _asinh_mu_at_zero(
                sb_zeropoint,
                pixel_scale_arcsec,
                scale_b,  # type: ignore[arg-type]
            )
            ax.axhline(mu_zero, color=_band_color(b_idx), ls="--", lw=0.7, alpha=0.5)
            all_y_for_ylim.append(np.asarray(mu, dtype=np.float64))
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.where(intens[m] > 0, np.log10(np.maximum(intens[m], 1e-30)), np.nan)
            ax.errorbar(
                x[m],
                mu,
                yerr=None,
                color=_band_color(b_idx),
                label=b,
                marker="o",
                ms=6.0,
                lw=0,
            )
            all_y_for_ylim.append(np.asarray(mu, dtype=np.float64))

    if has_sb:
        ax.set_ylabel(r"$\mu_{\rm asinh}$ [mag/arcsec$^2$] (Lupton+1999)")
        ax.invert_yaxis()
    else:
        ax.set_ylabel(r"$\log_{10}\,I$")
    ax.legend(loc="best", fontsize=12, framealpha=0.85, markerscale=0.9)
    ax.grid(True, alpha=0.3)
    if all_y_for_ylim:
        _set_ylim_from_data(ax, *all_y_for_ylim, pad_frac=0.05)
        if has_sb:
            # invert_yaxis reverses the limits; re-invert after _set_ylim_from_data.
            lo, hi = ax.get_ylim()
            if lo < hi:
                ax.set_ylim(hi, lo)

    if sky_offsets:
        _annotate_sky_offsets(ax, bands, sky_offsets)


def _plot_harmonic(
    ax: Axes,
    harm_key: str,
    ylabel: str,
    isophotes: Sequence[dict],
    bands: Sequence[str],
    pixel_scale_arcsec: float,
    normalize: bool = False,
) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    all_y_for_ylim: List[NDArray[np.float64]] = []
    for b_idx, b in enumerate(bands):
        intens = np.array([float(iso.get(f"intens_{b}", np.nan)) for iso in isophotes])
        grad = np.array([float(iso.get(f"grad_{b}", np.nan)) for iso in isophotes])
        harm = np.array([float(iso.get(f"{harm_key}_{b}", np.nan)) for iso in isophotes])
        # normalize=True only for raw shared coefficients (D16: the joint
        # modes store one raw value per order; per-band normalization by
        # sma*|grad_b| makes the curves separate). The per-band post-hoc
        # path stores Bender-normalized values already — a second division
        # here shrank them by 1/(sma*|grad|) (review P1).
        if normalize:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yvals = _normalize_harmonic_for_plot(harm, sma, grad, intens)
        else:
            yvals = harm
        m = valid & np.isfinite(yvals) & np.isfinite(x)
        if m.sum() == 0:
            continue
        ax.plot(
            x[m],
            yvals[m],
            color=_band_color(b_idx),
            marker="o",
            ms=5.0,
            lw=0,
            alpha=0.9,
        )
        all_y_for_ylim.append(np.asarray(yvals[m], dtype=np.float64))
    ax.axhline(0.0, color="#666", lw=0.5, alpha=0.5)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    if all_y_for_ylim:
        _set_ylim_from_data(ax, *all_y_for_ylim, pad_frac=0.10)


def _plot_eps(ax: Axes, isophotes: Sequence[dict], pixel_scale_arcsec: float) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    eps = np.array([float(iso["eps"]) for iso in isophotes])
    eps_err = np.array([float(iso.get("eps_err", 0.0)) for iso in isophotes], dtype=np.float64)
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    m = valid & np.isfinite(x) & np.isfinite(eps)
    if m.any():
        yerr = np.where(eps_err > 0, eps_err, np.nan)[m]
        if not np.all(np.isnan(yerr)):
            ax.errorbar(
                x[m],
                eps[m],
                yerr=yerr,
                color="#222",
                marker="o",
                ms=5.0,
                lw=0,
                elinewidth=0.7,
                capsize=0,
            )
        else:
            ax.plot(x[m], eps[m], color="#222", marker="o", ms=5.0, lw=0)
        _set_ylim_from_data(ax, eps[m], pad_frac=0.10)
    ax.set_ylabel(r"$\epsilon$", fontsize=12)
    ax.grid(True, alpha=0.3)


def _plot_pa(ax: Axes, isophotes: Sequence[dict], pixel_scale_arcsec: float) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    pa = np.array([float(iso["pa"]) for iso in isophotes])
    pa_err = np.array([float(iso.get("pa_err", 0.0)) for iso in isophotes], dtype=np.float64)
    pa_deg = np.rad2deg(pa) % 180.0
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    m = valid & np.isfinite(x) & np.isfinite(pa_deg)
    if m.any():
        yerr = np.where(pa_err > 0, np.rad2deg(pa_err), np.nan)[m]
        if not np.all(np.isnan(yerr)):
            ax.errorbar(
                x[m],
                pa_deg[m],
                yerr=yerr,
                color="#222",
                marker="o",
                ms=5.0,
                lw=0,
                elinewidth=0.7,
                capsize=0,
            )
        else:
            ax.plot(x[m], pa_deg[m], color="#222", marker="o", ms=5.0, lw=0)
        _set_ylim_from_data(ax, pa_deg[m], pad_frac=0.10)
    ax.set_ylabel("PA [deg]", fontsize=12)
    ax.grid(True, alpha=0.3)


def _plot_n_valid_per_band(
    ax: Axes,
    isophotes: Sequence[dict],
    bands: Sequence[str],
    pixel_scale_arcsec: float,
) -> None:
    """Loose-validity QA panel: per-band ``n_valid_<b> / n_attempted`` vs SMA.

    A value of 1.0 means every sample on the ellipse path passed the
    band's validity rule + sigma clip; lower values flag radii where
    the band is contributing partial coverage. ``n_attempted`` is the
    ellipse-path sample grid ``max(64, int(2*pi*sma))`` — the same
    denominator the sampler's drop fraction uses, re-derived from sma so
    the panel needs no debug fields and fractions never exceed 1 (M8).
    """
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    n_attempted = np.maximum(64.0, np.floor(2.0 * np.pi * np.maximum(sma, 0.0)))
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    plotted_anything = False
    for b_idx, b in enumerate(bands):
        n_b = np.array(
            [float(iso.get(f"n_valid_{b}", 0)) for iso in isophotes],
            dtype=np.float64,
        )
        frac = n_b / n_attempted
        m = valid & np.isfinite(x) & np.isfinite(frac)
        if not m.any():
            continue
        plotted_anything = True
        color = _BAND_COLORS[b_idx % len(_BAND_COLORS)]
        ax.plot(x[m], frac[m], color=color, marker="o", ms=4.0, lw=0, label=b)
    if plotted_anything:
        ax.axhline(1.0, color="#666", lw=0.5, alpha=0.5)
        ax.set_ylim(-0.05, 1.10)
        ax.legend(loc="lower left", fontsize=9, ncol=len(bands), framealpha=0.85)
    ax.set_ylabel(r"$N_{\rm valid}/N_{\rm attempted}$", fontsize=11)
    ax.grid(True, alpha=0.3)


def _plot_center(ax: Axes, isophotes: Sequence[dict], pixel_scale_arcsec: float) -> None:
    sma = np.array([float(iso["sma"]) for iso in isophotes])
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    x0 = np.array([float(iso["x0"]) for iso in isophotes])
    y0 = np.array([float(iso["y0"]) for iso in isophotes])
    x = _xaxis_arcsec_pow(sma, pixel_scale_arcsec)
    m = valid & np.isfinite(x) & np.isfinite(x0) & np.isfinite(y0)
    if m.any():
        x0_med = float(np.median(x0[m]))
        y0_med = float(np.median(y0[m]))
        dx0 = x0[m] - x0_med
        dy0 = y0[m] - y0_med
        ax.plot(x[m], dx0, color="#1f77b4", marker="o", ms=5.0, lw=0, label=r"$\Delta x_0$")
        ax.plot(x[m], dy0, color="#d62728", marker="s", ms=5.0, lw=0, label=r"$\Delta y_0$")
        ax.axhline(0.0, color="#666", lw=0.5, alpha=0.5)
        ax.legend(loc="best", fontsize=10, framealpha=0.85)
        _set_ylim_from_data(ax, dx0, dy0, pad_frac=0.10)
    ax.set_ylabel(r"$\Delta_{\rm c}$ [pix]", fontsize=12)
    ax.grid(True, alpha=0.3)


def _plot_image_with_isophotes(
    ax: Axes,
    image: NDArray[np.floating],
    isophotes: Sequence[dict],
    band_label: str,
    n_rings: int = 8,
) -> None:
    """Reference-band cutout with N evenly-log-spaced isophote ellipses."""
    valid_iso = [iso for iso in isophotes if bool(iso.get("valid", True)) and iso.get("sma", 0.0) > 0]
    h, w = np.asarray(image).shape

    img = np.asarray(image, dtype=np.float64)
    finite = img[np.isfinite(img)]
    if finite.size:
        scale = float(np.nanpercentile(np.abs(finite), 70.0)) or 1e-6
        vmax = float(np.nanpercentile(np.abs(finite), 99.7)) or 1.0
    else:
        scale, vmax = 1e-3, 1.0
    stretched = np.arcsinh(img / max(scale, 1e-12)) / np.arcsinh(max(vmax, 1e-12) / max(scale, 1e-12))
    ax.imshow(stretched, origin="lower", cmap="gray_r", vmin=-0.05, vmax=1.05)

    if valid_iso:
        smas = np.array([iso["sma"] for iso in valid_iso])
        if smas.size > n_rings:
            log_targets = np.linspace(np.log(smas.min()), np.log(smas.max()), n_rings)
            chosen_idx = sorted(set(int(np.argmin(np.abs(np.log(smas) - lt))) for lt in log_targets))
        else:
            chosen_idx = list(range(len(valid_iso)))
        cmap = plt.get_cmap("plasma")
        for k, idx in enumerate(chosen_idx):
            iso = valid_iso[idx]
            color = cmap(k / max(len(chosen_idx) - 1, 1))
            ax.add_patch(
                Ellipse(
                    xy=(float(iso["x0"]), float(iso["y0"])),
                    width=2.0 * float(iso["sma"]),
                    height=2.0 * float(iso["sma"]) * (1.0 - float(iso["eps"])),
                    angle=float(np.rad2deg(float(iso["pa"]))),
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.0,
                    alpha=0.9,
                )
            )
    ax.set_title(f"{band_label} + isophotes", fontsize=10)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_residual_panel(
    ax: Axes,
    image: NDArray[np.floating],
    model: NDArray[np.floating],
    band_label: str,
    mask: Optional[NDArray[np.bool_]] = None,
) -> None:
    residual = np.asarray(image, dtype=np.float64) - np.asarray(model, dtype=np.float64)
    if mask is not None:
        unmasked = residual[~np.asarray(mask, dtype=bool) & np.isfinite(residual)]
    else:
        unmasked = residual[np.isfinite(residual)]
    if unmasked.size == 0:
        unmasked = residual[np.isfinite(residual)]
    rms = float(np.std(unmasked)) if unmasked.size else 1e-6
    scale = max(rms, 1e-9) * 5.0
    stretched = np.arcsinh(residual / scale)
    if unmasked.size:
        unmasked_stretched = np.arcsinh(unmasked / scale)
        vmax = float(np.percentile(np.abs(unmasked_stretched), 99.0))
    else:
        vmax = float(np.percentile(np.abs(stretched), 99.0))
    vmax = max(vmax, 1e-3)
    ax.imshow(stretched, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if mask is not None:
        overlay_cmap = ListedColormap([(0, 0, 0, 0), (0.2, 0.2, 0.2, 0.45)])
        ax.imshow(np.asarray(mask, dtype=bool), origin="lower", cmap=overlay_cmap, vmin=0, vmax=1)
    ax.set_title(f"residual ({band_label})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_qa_summary_mb(
    result: dict,
    images: Sequence[NDArray[np.floating]],
    *,
    bands: Optional[Sequence[str]] = None,
    sb_zeropoint: Optional[float] = None,
    pixel_scale_arcsec: Optional[float] = None,
    softening_per_band: Optional[Dict[str, float]] = None,
    object_mask: Optional[NDArray[np.bool_]] = None,
    reference_band_idx: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (20.0, 14.0),
    title: Optional[str] = None,
    sky_offsets: Optional[Dict[str, float]] = None,
) -> Figure:
    """
    Render the composite multi-band QA figure (four-block grid).

    See module docstring for layout details.

    Parameters
    ----------
    sky_offsets : dict[str, float], optional
        Externally-supplied per-band sky offsets to annotate on the SB
        profile panel. Useful when the caller computed sky offsets via
        a custom estimator (e.g. fitting a 2-D sky polynomial over masked
        sky pixels) instead of going through
        :func:`subtract_outermost_sky_offset`. When this is ``None``, the
        function automatically picks up ``result['sky_offsets']`` if it
        was stamped there by ``subtract_outermost_sky_offset``. Pass an
        empty dict to suppress the annotation entirely.
    """
    if matplotlib.get_backend().lower() == "agg":
        logger.debug("rendering multi-band QA on the Agg backend")
    plt.rcParams["text.usetex"] = False
    configure_qa_plot_style()
    plt.rcParams["text.usetex"] = False

    isophotes = list(result["isophotes"])
    if bands is None:
        bands = list(result.get("bands", []))
    if not bands:
        raise ValueError("bands is required (either via result['bands'] or explicit kwarg)")
    if len(images) != len(bands):
        raise ValueError(f"len(images) ({len(images)}) does not match len(bands) ({len(bands)})")

    pix = pixel_scale_arcsec if pixel_scale_arcsec is not None else 1.0
    softening = _resolve_softening(bands, isophotes, softening_per_band)

    if reference_band_idx is None:
        ref_band = str(result.get("reference_band", bands[0]))
        try:
            reference_band_idx = list(bands).index(ref_band)
        except ValueError:
            reference_band_idx = 0

    # No constrained_layout — it overrides our hspace=0 inner gridspec
    # requests. Explicit outer margins ensure labels still fit.
    fig = plt.figure(figsize=figsize, constrained_layout=False)

    # Outer 2x2 grid with equal-width columns. Tight outer margins so
    # the four blocks share the figure cleanly.
    outer = fig.add_gridspec(
        2,
        2,
        hspace=0.18,
        wspace=0.10,
        left=0.06,
        right=0.985,
        top=0.94,
        bottom=0.06,
    )

    # Pre-compute the shared x-range so the SB / harmonic / geometry
    # blocks line up edge-to-edge in the same column.
    sma_arr = np.array([float(iso["sma"]) for iso in isophotes], dtype=np.float64)
    valid_arr = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    x_for_range = _xaxis_arcsec_pow(sma_arr, pix)
    x_finite = x_for_range[valid_arr & np.isfinite(x_for_range)]
    if x_finite.size:
        x_lo = float(np.min(x_finite))
        x_hi = float(np.max(x_finite))
        x_pad = 0.03 * max(x_hi - x_lo, 1e-3)
        shared_xlim = (x_lo - x_pad, x_hi + x_pad)
    else:
        shared_xlim = None

    # --- Top-left: SB profile -------------------------------------------------
    ax_sb = fig.add_subplot(outer[0, 0])
    # Sky-offset annotation source: caller-provided ``sky_offsets`` kwarg
    # wins; otherwise pick up ``result['sky_offsets']`` (set by
    # ``subtract_outermost_sky_offset``); otherwise no annotation. An empty
    # dict (``{}``) explicitly suppresses the annotation.
    if sky_offsets is not None:
        sky_offsets_for_panel = sky_offsets if sky_offsets else None
    else:
        sky_offsets_for_panel = result.get("sky_offsets") if isinstance(result, dict) else None
        if not isinstance(sky_offsets_for_panel, dict):
            sky_offsets_for_panel = None
    _plot_sb_profile(
        ax_sb,
        isophotes,
        bands,
        sb_zeropoint,
        pixel_scale_arcsec,
        softening,
        sky_offsets=sky_offsets_for_panel,
    )
    if pixel_scale_arcsec is not None:
        ax_sb.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [arcsec$^{0.25}$]", fontsize=12)
    else:
        ax_sb.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [pix$^{0.25}$]", fontsize=12)
    if shared_xlim is not None:
        ax_sb.set_xlim(shared_xlim)

    # --- Top-right: image + residual mosaic (2x3) ----------------------------
    # Tight inter-cell spacing — these are images, not 1-D plots.
    gs_mosaic = outer[0, 1].subgridspec(2, 3, hspace=0.06, wspace=0.04)
    # Cell (0, 0): reference-band cutout with isophote ellipses.
    ax_img = fig.add_subplot(gs_mosaic[0, 0])
    _plot_image_with_isophotes(
        ax_img,
        images[reference_band_idx],
        isophotes,
        band_label=bands[reference_band_idx],
    )
    # Remaining cells: per-band residuals in band order. With B=5 the
    # 2x3 mosaic has exactly one image cell + 5 residual cells; for
    # B != 5 the empty cells are simply skipped.
    cell_iter = iter([(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
    per_band_lists = _isophotes_to_per_band_singleband_lists(isophotes, bands, result.get("harmonic_orders"))
    h, w = np.asarray(images[0]).shape
    for b_idx, b in enumerate(bands):
        try:
            cell = next(cell_iter)
        except StopIteration:
            break
        ax_r = fig.add_subplot(gs_mosaic[cell[0], cell[1]])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = build_isoster_model(
                    (h, w),
                    per_band_lists[b],
                    use_harmonics=True,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("model build failed for band %s: %s", b, e)
            ax_r.text(0.5, 0.5, "model error", transform=ax_r.transAxes, ha="center")
            ax_r.set_xticks([])
            ax_r.set_yticks([])
            continue
        _plot_residual_panel(ax_r, images[b_idx], model, b, mask=object_mask)

    # --- Bottom-left: harmonics stack (4 panels, share x, no gap) -----------
    # Shared/joint modes store RAW shared coefficients that need per-band
    # normalization at plot time (D16); the per-band post-hoc path stores
    # Bender-normalized values already and must not be re-normalized (P1).
    harm_normalize = bool(result.get("harmonics_shared", False))
    gs_harm = outer[1, 0].subgridspec(4, 1, hspace=0.0)
    ax_a3 = fig.add_subplot(gs_harm[0])
    ax_b3 = fig.add_subplot(gs_harm[1], sharex=ax_a3)
    ax_a4 = fig.add_subplot(gs_harm[2], sharex=ax_a3)
    ax_b4 = fig.add_subplot(gs_harm[3], sharex=ax_a3)
    _plot_harmonic(ax_a3, "a3", r"$A_3 / (a\,dI/da)$", isophotes, bands, pix, normalize=harm_normalize)
    _plot_harmonic(ax_b3, "b3", r"$B_3 / (a\,dI/da)$", isophotes, bands, pix, normalize=harm_normalize)
    _plot_harmonic(ax_a4, "a4", r"$A_4 / (a\,dI/da)$", isophotes, bands, pix, normalize=harm_normalize)
    _plot_harmonic(ax_b4, "b4", r"$B_4 / (a\,dI/da)$", isophotes, bands, pix, normalize=harm_normalize)
    for ax in (ax_a3, ax_b3, ax_a4):
        plt.setp(ax.get_xticklabels(), visible=False)
    # Prune the tick labels at panel boundaries so stacked panels look
    # truly seamless: top panel hides its lowest tick, middle panels
    # hide both extremes, bottom panel hides its uppermost tick.
    ax_a3.yaxis.set_major_locator(MaxNLocator(prune="lower", nbins=4))
    ax_b3.yaxis.set_major_locator(MaxNLocator(prune="both", nbins=4))
    ax_a4.yaxis.set_major_locator(MaxNLocator(prune="both", nbins=4))
    ax_b4.yaxis.set_major_locator(MaxNLocator(prune="upper", nbins=4))
    if pixel_scale_arcsec is not None:
        ax_b4.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [arcsec$^{0.25}$]", fontsize=12)
    else:
        ax_b4.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [pix$^{0.25}$]", fontsize=12)
    if shared_xlim is not None:
        ax_b4.set_xlim(shared_xlim)  # propagates to a3/b3/a4 via sharex

    # --- Bottom-right: geometry stack -----------
    # Default: 3 stacked panels (eps, PA, center). When the result was
    # produced under loose validity (D9 backport) we add a fourth small
    # panel showing per-band ``n_valid_<b> / n_attempted`` so the user
    # can see which radii each band actually contributed to.
    show_n_valid = bool(result.get("loose_validity", False)) and any(
        f"n_valid_{b}" in (isophotes[0] if isophotes else {}) for b in bands
    )
    n_geom_rows = 4 if show_n_valid else 3
    gs_geom = outer[1, 1].subgridspec(n_geom_rows, 1, hspace=0.0)
    ax_eps = fig.add_subplot(gs_geom[0])
    ax_pa = fig.add_subplot(gs_geom[1], sharex=ax_eps)
    ax_ctr = fig.add_subplot(gs_geom[2], sharex=ax_eps)
    _plot_eps(ax_eps, isophotes, pix)
    _plot_pa(ax_pa, isophotes, pix)
    _plot_center(ax_ctr, isophotes, pix)
    geom_axes = [ax_eps, ax_pa, ax_ctr]
    if show_n_valid:
        ax_nv = fig.add_subplot(gs_geom[3], sharex=ax_eps)
        _plot_n_valid_per_band(ax_nv, isophotes, bands, pix)
        geom_axes.append(ax_nv)
    # Hide x-tick labels on every stacked panel except the bottom-most.
    for ax in geom_axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_eps.yaxis.set_major_locator(MaxNLocator(prune="lower", nbins=4))
    ax_pa.yaxis.set_major_locator(MaxNLocator(prune="both", nbins=4))
    ax_ctr.yaxis.set_major_locator(MaxNLocator(prune="upper" if not show_n_valid else "both", nbins=4))
    if show_n_valid:
        ax_nv.yaxis.set_major_locator(MaxNLocator(prune="upper", nbins=3))
    bottom_ax = geom_axes[-1]
    if pixel_scale_arcsec is not None:
        bottom_ax.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [arcsec$^{0.25}$]", fontsize=12)
    else:
        bottom_ax.set_xlabel(r"$\mathrm{SMA}^{0.25}$  [pix$^{0.25}$]", fontsize=12)
    if shared_xlim is not None:
        bottom_ax.set_xlim(shared_xlim)  # propagates to siblings via sharex

    # Align y-axis labels in each column so the SB / harmonic / geometry
    # blocks share a single label x-coordinate per column.
    fig.align_ylabels([ax_sb, ax_a3, ax_b3, ax_a4, ax_b4])
    fig.align_ylabels(geom_axes)

    if title is not None:
        fig.suptitle(title, fontsize=13, y=0.995)

    if output_path is not None:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")

    return fig
