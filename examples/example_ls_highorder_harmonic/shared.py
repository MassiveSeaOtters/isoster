"""
Shared helpers for the LegacySurvey high-order harmonics example campaign.

Provides:
- ``GALAXY_CONFIGS``  — per-galaxy metadata (pixel scale, band names, etc.)
- ``load_legacysurvey_fits()``  — load a single band from the 3D FITS cube
- ``make_isoster_configs()``  — build the 6 fitting conditions as IsosterConfig dicts
- ``CONDITION_LABELS`` / ``CONDITION_ORDER``  — canonical ordering
- ``plot_harmonic_comparison_qa()``  — cross-condition comparison figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D

from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import (
    configure_qa_plot_style,
    normalize_pa_degrees,
    robust_limits,
    set_axis_limits_from_finite_values,
    set_x_limits_with_right_margin,
)

# ---------------------------------------------------------------------------
# Per-galaxy metadata
# ---------------------------------------------------------------------------

#: Supported galaxy names (lower-case, matching filenames under data/)
SUPPORTED_GALAXIES = ("eso243-49", "ngc3610")

#: Pixel scale in arcsec / pixel for each galaxy
PIXEL_SCALE = {
    "eso243-49": 0.25,
    "ngc3610": 1.0,
}

#: Band names corresponding to FITS data-cube indices 0, 1, 2
BAND_NAMES = {
    "eso243-49": ["g", "r", "z"],
    "ngc3610": ["g", "r", "i"],
}

#: FITS file name for each galaxy
FITS_FILENAME = {
    "eso243-49": "eso243-49.fits",
    "ngc3610": "ngc3610.fits",
}

#: Initial SMA in pixels (bright center estimate)
INITIAL_SMA = {
    "eso243-49": 6.0,
    "ngc3610": 6.0,
}

#: Masking parameters per galaxy (passed to masking.make_object_mask)
MASK_PARAMS: dict[str, dict[str, Any]] = {
    "eso243-49": dict(
        on_galaxy=True,
        on_galaxy_box=8,  # small box tracks local galaxy gradients
        on_galaxy_filter=3,
        on_galaxy_npixels=10,
        on_galaxy_dilate_fwhm=8.0,
    ),
    "ngc3610": dict(
        on_galaxy=True,
        on_galaxy_box=8,  # small box tracks local galaxy gradients
        on_galaxy_npixels=10,
        on_galaxy_dilate_fwhm=8.0,
    ),
}

# ---------------------------------------------------------------------------
# Six fitting conditions (2 × 3 grid)
# ---------------------------------------------------------------------------

#: Canonical condition order
CONDITION_ORDER = [
    "pa_baseline",
    "ea_baseline",
    "pa_harmonics_34",
    "ea_harmonics_34",
    "pa_harmonics_34567",
    "ea_harmonics_34567",
]

#: Short display label for each condition (used in figure legends)
CONDITION_DISPLAY = {
    "pa_baseline": "PA / no-sim",
    "ea_baseline": "EA / no-sim",
    "pa_harmonics_34": "PA / [3,4]",
    "ea_harmonics_34": "EA / [3,4]",
    "pa_harmonics_34567": "PA / [3–7]",
    "ea_harmonics_34567": "EA / [3–7]",
}

#: Colours for cross-condition comparison lines
CONDITION_COLORS = {
    "pa_baseline": "#1f77b4",
    "ea_baseline": "#aec7e8",
    "pa_harmonics_34": "#d62728",
    "ea_harmonics_34": "#f4a6a6",
    "pa_harmonics_34567": "#2ca02c",
    "ea_harmonics_34567": "#98df8a",
}

#: Markers for cross-condition comparison scatter
CONDITION_MARKERS = {
    "pa_baseline": "o",
    "ea_baseline": "s",
    "pa_harmonics_34": "^",
    "ea_harmonics_34": "D",
    "pa_harmonics_34567": "P",
    "ea_harmonics_34567": "v",
}


def make_isoster_configs(galaxy: str, sma0: float | None = None) -> dict[str, IsosterConfig]:
    """Build the six IsosterConfig objects for the fitting conditions.

    Parameters
    ----------
    galaxy : str
        Galaxy name (must be in :data:`SUPPORTED_GALAXIES`).
    sma0 : float or None
        Initial SMA override.  Uses :data:`INITIAL_SMA` default when None.

    Returns
    -------
    dict mapping condition label → IsosterConfig
    """
    _sma0 = sma0 if sma0 is not None else INITIAL_SMA[galaxy]

    # Shared baseline kwargs for all conditions
    base_kwargs: dict[str, Any] = dict(
        sma0=_sma0,
        convergence_scaling="sector_area",
        geometry_damping=0.7,
        permissive_geometry=True,
    )

    configs = {}
    for label in CONDITION_ORDER:
        use_ea = label.startswith("ea_")
        if "34567" in label:
            orders = [3, 4, 5, 6, 7]
            sim = True
        elif "34" in label and "harmonics" in label:
            orders = [3, 4]
            sim = True
        else:
            # baseline: no simultaneous harmonics; [3,4] computed post-hoc only
            orders = [3, 4]
            sim = False

        configs[label] = IsosterConfig(
            **base_kwargs,
            use_eccentric_anomaly=use_ea,
            simultaneous_harmonics=sim,
            harmonic_orders=orders,
        )

    return configs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_legacysurvey_fits(
    fits_path: str | Path,
    band_index: int,
) -> tuple[np.ndarray, float]:
    """Load one band from a LegacySurvey 3D FITS cube.

    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file (PRIMARY HDU expected, shape ``(3, H, W)``).
    band_index : int
        Band plane to extract (0, 1, or 2).

    Returns
    -------
    image : 2D float64 array of shape ``(H, W)``
    pixel_scale : float
        Pixel scale in arcsec/pixel derived from FITS header (``CD2_2`` or
        ``CDELT2``), or default 1.0 if header is absent / non-standard.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header

    if data.ndim == 3:
        image = data[band_index]
    elif data.ndim == 2:
        image = data
    else:
        raise ValueError(f"Unexpected data shape {data.shape} in {fits_path}")

    # Derive pixel scale from WCS header (arcsec/pixel)
    pixel_scale = 1.0
    for key in ("CD2_2", "CDELT2"):
        if key in header:
            val = abs(float(header[key]))
            if val > 0.0:
                # If WCS uses degrees, convert; if already arcsec, keep
                pixel_scale = val * 3600.0 if val < 0.01 else val
                break

    return image, pixel_scale


# ---------------------------------------------------------------------------
# Cross-condition comparison figure
# ---------------------------------------------------------------------------


def plot_harmonic_comparison_qa(
    galaxy: str,
    image: np.ndarray,
    mask: np.ndarray | None,
    condition_results: dict[str, list[dict[str, Any]]],
    *,
    condition_models: dict[str, np.ndarray] | None = None,
    harmonic_mode: str = "coefficients",
    filename: str = "comparison_qa.png",
) -> None:
    """All-condition comparison figure for a single galaxy/band.

    Layout
    ------
    *Top row* — 6 small residual panels (``data - model``), one per condition.

    *Bottom section* — 5 profile panels sharing the same SMA^0.25 x-axis,
    one colour per condition:
      1. Surface brightness log10(I)
      2. Odd harmonics (a_n per condition in coefficients mode; A_n in amplitude)
      3. Even harmonics (same style)
      4. Axis ratio b/a
      5. PA (degrees)

    Parameters
    ----------
    galaxy : str
        Galaxy name string (used in title).
    image : 2D array
        Science image (for display).
    mask : 2D bool array or None
        Bad-pixel mask overlay (True = bad).
    condition_results : dict
        Mapping condition_label → list of isophote dicts from ``fit_image()``.
    condition_models : dict or None
        Mapping condition_label → 2D model array.  When None, models are
        built on the fly from ``condition_results`` using
        ``build_isoster_model``.
    harmonic_mode : str
        ``'coefficients'`` (default) shows ``a_n`` (filled) per condition.
        ``'amplitude'`` shows ``A_n = sqrt(a_n^2 + b_n^2)`` per condition.
    filename : str
        Output path.
    """
    configure_qa_plot_style()

    conditions = [c for c in CONDITION_ORDER if c in condition_results]
    n_conditions = len(conditions)

    # Detect all harmonic orders present across conditions
    all_orders = set()
    for cond in conditions:
        for r in condition_results[cond]:
            for order in [3, 4, 5, 6, 7]:
                if np.isfinite(r.get(f"a{order}", np.nan)):
                    all_orders.add(order)
    all_orders_sorted = sorted(all_orders)
    odd_orders = [o for o in all_orders_sorted if o % 2 == 1]
    even_orders = [o for o in all_orders_sorted if o % 2 == 0]

    # Build or retrieve models for each condition
    if condition_models is None:
        condition_models = {}
    models: dict[str, np.ndarray] = {}
    for cond in conditions:
        if cond in condition_models:
            models[cond] = condition_models[cond]
        else:
            # Build model on the fly (auto-detects harmonic orders and EA mode)
            res = condition_results[cond]
            models[cond] = build_isoster_model(
                image.shape,
                res,
                use_harmonics=True,
            )

    # --------------------------------------------------------------------------
    # Figure layout: top row (residuals) + 5 profile rows
    # --------------------------------------------------------------------------
    n_profile_rows = 5
    fig_height = 3.0 * (1 + n_conditions // 3) + 1.5 * n_profile_rows
    fig = plt.figure(figsize=(14.0, fig_height))

    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.6, float(n_profile_rows)],
        hspace=0.12,
    )

    # Top: 6 residual panels in one row
    top = gridspec.GridSpecFromSubplotSpec(
        1,
        n_conditions,
        subplot_spec=outer[0],
        wspace=0.05,
    )

    # Bottom: profile panels
    profile_height_ratios = [2.5, 1.2, 1.2, 1.2, 1.2]
    bottom = gridspec.GridSpecFromSubplotSpec(
        n_profile_rows,
        1,
        subplot_spec=outer[1],
        height_ratios=profile_height_ratios,
        hspace=0.0,
    )

    fig.suptitle(
        f"{galaxy.upper()}  —  harmonic conditions comparison",
        fontsize=16,
        y=0.995,
    )

    # Compute shared residual colour scale across all conditions
    all_residuals = []
    for cond in conditions:
        res_map = np.where(np.isfinite(image), image - models[cond], np.nan)
        all_residuals.append(res_map)
    all_abs = np.abs(np.concatenate([r[np.isfinite(r)] for r in all_residuals]))
    res_limit = float(
        np.clip(
            np.nanpercentile(all_abs, 99.0) if all_abs.size else 1.0,
            0.05,
            None,
        )
    )

    # ------------------------------------------------------------------
    # Top row: residual panels (data - model)
    # ------------------------------------------------------------------
    for col, cond in enumerate(conditions):
        ax_im = fig.add_subplot(top[0, col])
        residual_map = all_residuals[col]
        ax_im.imshow(
            residual_map,
            origin="lower",
            cmap="coolwarm",
            vmin=-res_limit,
            vmax=res_limit,
            interpolation="nearest",
        )
        ax_im.set_title(CONDITION_DISPLAY[cond], fontsize=9, pad=2)
        ax_im.set_xticks([])
        ax_im.set_yticks([])

    # ------------------------------------------------------------------
    # Helper to extract arrays from isophote list
    # ------------------------------------------------------------------
    def _arr(isos, key, default=np.nan):
        return np.array([r.get(key, default) for r in isos])

    # Build profile axes
    ax_sb = fig.add_subplot(bottom[0])
    ax_odd = fig.add_subplot(bottom[1], sharex=ax_sb)
    ax_even = fig.add_subplot(bottom[2], sharex=ax_sb)
    ax_ba = fig.add_subplot(bottom[3], sharex=ax_sb)
    ax_pa_panel = fig.add_subplot(bottom[4], sharex=ax_sb)

    legend_handles = []

    for cond in conditions:
        res = condition_results[cond]
        sma = _arr(res, "sma")
        intens = _arr(res, "intens")
        eps = _arr(res, "eps")
        pa_rad = _arr(res, "pa")
        stop = _arr(res, "stop_code", 0).astype(int)

        xax = sma**0.25
        vm = np.isfinite(xax) & (sma > 1.0)

        col = CONDITION_COLORS[cond]
        mrk = CONDITION_MARKERS[cond]
        label = CONDITION_DISPLAY[cond]

        good = vm & (stop == 0)

        scatter_kw = dict(s=16, marker=mrk, alpha=0.75)

        # 1. Surface brightness
        sb_ok = vm & np.isfinite(intens) & (intens > 0)
        y_sb = np.full_like(intens, np.nan)
        y_sb[sb_ok] = np.log10(intens[sb_ok])
        ax_sb.scatter(xax[sb_ok], y_sb[sb_ok], facecolors=col, edgecolors=col, **scatter_kw, label=label)

        # 2. Odd harmonics — show a_n (filled) per condition
        for order in odd_orders:
            an = _arr(res, f"a{order}")
            bn = _arr(res, f"b{order}")
            if harmonic_mode == "amplitude":
                vals = np.sqrt(np.where(np.isfinite(an), an, 0.0) ** 2 + np.where(np.isfinite(bn), bn, 0.0) ** 2)
                vals[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
            else:
                vals = an  # show a_n only in comparison (b_n too crowded)
            ok = vm & np.isfinite(vals)
            if np.any(ok):
                ax_odd.scatter(xax[ok], vals[ok], facecolors=col, edgecolors=col, **scatter_kw)

        # 3. Even harmonics
        for order in even_orders:
            an = _arr(res, f"a{order}")
            bn = _arr(res, f"b{order}")
            if harmonic_mode == "amplitude":
                vals = np.sqrt(np.where(np.isfinite(an), an, 0.0) ** 2 + np.where(np.isfinite(bn), bn, 0.0) ** 2)
                vals[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
            else:
                vals = an
            ok = vm & np.isfinite(vals)
            if np.any(ok):
                ax_even.scatter(xax[ok], vals[ok], facecolors=col, edgecolors=col, **scatter_kw)

        # 4. Axis ratio
        ba = 1.0 - eps
        ax_ba.scatter(xax[vm & good], ba[vm & good], facecolors=col, edgecolors=col, **scatter_kw)

        # 5. PA
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        ax_pa_panel.scatter(xax[vm & good], pa_deg[vm & good], facecolors=col, edgecolors=col, **scatter_kw)

        legend_handles.append(Line2D([], [], marker=mrk, linestyle="None", color=col, markersize=6, label=label))

    # Axis labels / formatting
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_sb.set_title("Surface brightness")

    if harmonic_mode == "amplitude":
        odd_label = r"$\sqrt{a_n^2 + b_n^2}$"
        even_label = r"$\sqrt{a_n^2 + b_n^2}$"
    else:
        odd_label = r"$a_n$"
        even_label = r"$a_n$"
    ax_odd.set_ylabel(f"{odd_label} (odd)")
    ax_odd.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_even.set_ylabel(f"{even_label} (even)")
    ax_even.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)

    ax_ba.set_ylabel("axis ratio")
    ax_pa_panel.set_ylabel("PA [deg]")
    ax_pa_panel.set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")

    for ax in [ax_sb, ax_odd, ax_even, ax_ba]:
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.2)
    ax_pa_panel.grid(alpha=0.2)

    # Collect x data for x-limit
    all_xvals = []
    for cond in conditions:
        res = condition_results[cond]
        sma = _arr(res, "sma")
        vm = np.isfinite(sma) & (sma > 1.0)
        if np.any(vm):
            all_xvals.append((sma[vm] ** 0.25))

    if all_xvals:
        xcat = np.concatenate(all_xvals)
        set_x_limits_with_right_margin(ax_pa_panel, xcat)

    # Robust y-limits
    for ax, collect_fn, clip_lo, clip_hi in [
        (ax_sb, lambda c, r: np.log10(np.maximum(_arr(r, "intens"), 1e-30)), None, None),
        (ax_ba, lambda c, r: 1.0 - _arr(r, "eps"), 0.0, 1.0),
    ]:
        vals = []
        for cond in conditions:
            res = condition_results[cond]
            try:
                v = collect_fn(cond, res)
                vm = np.isfinite(v)
                if np.any(vm):
                    vals.append(v[vm])
            except Exception:  # noqa: BLE001
                pass
        if vals:
            cat = np.concatenate(vals)
            set_axis_limits_from_finite_values(
                ax,
                cat,
                margin_fraction=0.06,
                min_margin=0.05,
                lower_clip=clip_lo,
                upper_clip=clip_hi,
            )

    # PA limits from stop=0 points only
    pa_vals = []
    for cond in conditions:
        res = condition_results[cond]
        sma = _arr(res, "sma")
        stop = _arr(res, "stop_code", 0).astype(int)
        pa_rad = _arr(res, "pa")
        vm = np.isfinite(sma) & (sma > 1.0) & (stop == 0)
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        if np.any(vm & np.isfinite(pa_deg)):
            pa_vals.append(pa_deg[vm & np.isfinite(pa_deg)])
    if pa_vals:
        all_pa = np.concatenate(pa_vals)
        lo, hi = robust_limits(all_pa, 3, 97)
        margin = max(3.0, 0.08 * (hi - lo + 1e-6))
        ax_pa_panel.set_ylim(lo - margin, hi + margin)

    # Legend on SB panel
    ax_sb.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        ncol=2,
        frameon=True,
    )

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.970, hspace=0.04)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved comparison QA figure to {filename}")
