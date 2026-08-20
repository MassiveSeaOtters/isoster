"""
NGC3610 High-Order Harmonic Exploration
========================================

Two investigations:

1. **Model reconstruction**: Compare 2D residuals with and without high-order
   harmonics to confirm harmonics are having a measurable effect.

2. **Convergence tightness**: Run with very strict convergence criteria
   (conver=0.005, maxit=500) to check whether the default settings converge
   too early, leaving harmonic residuals on the table.

Output goes to ``outputs/example_ls_highorder_harmonic/ngc3610_highorder_exploration/``.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import (
    configure_qa_plot_style,
    derive_arcsinh_parameters,
    draw_isophote_overlays,
    make_arcsinh_display_from_parameters,
    normalize_pa_degrees,
    plot_qa_summary_extended,
    robust_limits,
    set_axis_limits_from_finite_values,
    set_x_limits_with_right_margin,
)

# Reuse data-loading and masking helpers from this example folder.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from masking import make_object_mask  # noqa: E402
from shared import (  # noqa: E402
    FITS_FILENAME,
    INITIAL_SMA,
    MASK_PARAMS,
    PIXEL_SCALE,
    load_legacysurvey_fits,
)

OUTPUT_DIR = Path("outputs/example_ls_highorder_harmonic/ngc3610_highorder_exploration")
GALAXY = "ngc3610"
BAND_INDEX = 1  # r-band
HARMONIC_ORDERS = [3, 4, 5, 6, 7]


# ---------------------------------------------------------------------------
# Investigation 1: Model with vs without harmonics
# ---------------------------------------------------------------------------

def plot_harmonic_impact(
    image: np.ndarray,
    mask: np.ndarray,
    isophotes: list[dict],
    mode_label: str,
    output_dir: Path,
) -> None:
    """Generate a figure comparing models with and without harmonics.

    Layout: 2 rows x 3 columns
      Row 1: Data | Model (no harmonics) | Model (with harmonics)
      Row 2: (empty) | Residual (no harmonics) | Residual (with harmonics)
    Plus a shared colorbar for residuals.
    """
    configure_qa_plot_style()

    model_no_harm = build_isoster_model(
        image.shape, isophotes, use_harmonics=False,
    )
    model_with_harm = build_isoster_model(
        image.shape, isophotes, use_harmonics=True,
    )

    res_no = image - model_no_harm
    res_with = image - model_with_harm
    diff_model = model_with_harm - model_no_harm

    # Shared arcsinh for data / model panels
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)

    # Shared residual color scale
    all_abs = np.abs(np.concatenate([
        res_no[np.isfinite(res_no)],
        res_with[np.isfinite(res_with)],
    ]))
    res_limit = float(np.clip(np.nanpercentile(all_abs, 99.0), 0.05, None))

    # Difference panel scale
    diff_abs = np.abs(diff_model[np.isfinite(diff_model)])
    diff_limit = float(np.clip(np.nanpercentile(diff_abs, 99.5), 1e-4, None))

    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.05],
        height_ratios=[1, 1],
        wspace=0.12, hspace=0.15,
    )

    fig.suptitle(
        f"NGC3610 — Harmonic impact on model ({mode_label}, orders {HARMONIC_ORDERS})",
        fontsize=14, y=0.98,
    )

    # --- Row 1: Data, Model (no harm), Model (with harm) ---
    # Data
    ax_data = fig.add_subplot(gs[0, 0])
    img_disp, iv_min, iv_max = make_arcsinh_display_from_parameters(
        image, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )
    ax_data.imshow(img_disp, origin="lower", cmap="viridis",
                   vmin=iv_min, vmax=iv_max, interpolation="none")
    if mask is not None:
        overlay = np.zeros((*image.shape, 4))
        overlay[mask] = [1, 0, 0, 0.35]
        ax_data.imshow(overlay, origin="lower")
    overlay_step = max(1, len(isophotes) // 15)
    draw_isophote_overlays(ax_data, isophotes, step=overlay_step,
                           line_width=1.0, alpha=0.7, edge_color="orangered")
    ax_data.set_title("Data + isophotes", fontsize=11)

    # Model without harmonics
    ax_m_no = fig.add_subplot(gs[0, 1])
    m_disp_no, _, _ = make_arcsinh_display_from_parameters(
        model_no_harm, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )
    ax_m_no.imshow(m_disp_no, origin="lower", cmap="viridis",
                   vmin=iv_min, vmax=iv_max, interpolation="none")
    ax_m_no.set_title("Model (no harmonics)", fontsize=11)

    # Model with harmonics
    ax_m_yes = fig.add_subplot(gs[0, 2])
    m_disp_yes, _, _ = make_arcsinh_display_from_parameters(
        model_with_harm, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )
    ax_m_yes.imshow(m_disp_yes, origin="lower", cmap="viridis",
                    vmin=iv_min, vmax=iv_max, interpolation="none")
    ax_m_yes.set_title(f"Model (harmonics {HARMONIC_ORDERS})", fontsize=11)

    # --- Row 2: Harmonic difference, Residual (no harm), Residual (with harm) ---
    # Difference: model_with - model_no (shows what harmonics add)
    ax_diff = fig.add_subplot(gs[1, 0])
    h_diff = ax_diff.imshow(
        diff_model, origin="lower", cmap="coolwarm",
        vmin=-diff_limit, vmax=diff_limit, interpolation="nearest",
    )
    ax_diff.set_title("Harmonic correction\n(model_harm - model_no_harm)", fontsize=10)

    # Residual without harmonics
    ax_r_no = fig.add_subplot(gs[1, 1])
    h_res = ax_r_no.imshow(
        res_no, origin="lower", cmap="coolwarm",
        vmin=-res_limit, vmax=res_limit, interpolation="nearest",
    )
    # Compute statistics
    valid = np.isfinite(image) & np.isfinite(model_no_harm) & (np.abs(image) > 1e-6)
    rms_no = float(np.sqrt(np.nanmean(res_no[valid] ** 2)))
    ax_r_no.set_title(f"Residual (no harm)\nRMS={rms_no:.4f}", fontsize=10)

    # Residual with harmonics
    ax_r_yes = fig.add_subplot(gs[1, 2])
    ax_r_yes.imshow(
        res_with, origin="lower", cmap="coolwarm",
        vmin=-res_limit, vmax=res_limit, interpolation="nearest",
    )
    valid_h = np.isfinite(image) & np.isfinite(model_with_harm) & (np.abs(image) > 1e-6)
    rms_yes = float(np.sqrt(np.nanmean(res_with[valid_h] ** 2)))
    ax_r_yes.set_title(f"Residual (with harm)\nRMS={rms_yes:.4f}", fontsize=10)

    # Colorbars
    cbar_ax_res = fig.add_subplot(gs[1, 3])
    fig.colorbar(h_res, cax=cbar_ax_res, label="data - model")
    cbar_ax_diff = fig.add_subplot(gs[0, 3])
    fig.colorbar(h_diff, cax=cbar_ax_diff, label="harmonic correction")

    for ax in [ax_data, ax_m_no, ax_m_yes, ax_diff, ax_r_no, ax_r_yes]:
        ax.set_xticks([])
        ax.set_yticks([])

    filename = str(output_dir / f"ngc3610_{mode_label}_harmonic_impact.png")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved harmonic impact figure -> {filename}")
    print(f"    RMS without harmonics: {rms_no:.6f}")
    print(f"    RMS with harmonics:    {rms_yes:.6f}")
    print(f"    Improvement:           {(1 - rms_yes / rms_no) * 100:.2f}%")


# ---------------------------------------------------------------------------
# Investigation 2: Convergence tightness comparison
# ---------------------------------------------------------------------------

CONVERGENCE_CONFIGS = {
    "default": {
        "conver": 0.05,
        "maxit": 50,
        "label": "Default (conver=0.05, maxit=50)",
    },
    "strict": {
        "conver": 0.01,
        "maxit": 200,
        "label": "Strict (conver=0.01, maxit=200)",
    },
    "very_strict": {
        "conver": 0.005,
        "maxit": 500,
        "label": "Very strict (conver=0.005, maxit=500)",
    },
}

CONV_COLORS = {
    "default": "#1f77b4",
    "strict": "#d62728",
    "very_strict": "#2ca02c",
}

CONV_MARKERS = {
    "default": "o",
    "strict": "^",
    "very_strict": "s",
}


def plot_convergence_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    conv_results: dict[str, list[dict]],
    conv_models: dict[str, np.ndarray],
    conv_meta: dict[str, dict],
    output_dir: Path,
) -> None:
    """Build a comparison figure for different convergence settings.

    Top row: 3 residual maps (one per setting).
    Bottom: 1D profile stack (SB, delta niter, odd harmonics, even harmonics, b/a, PA).
    """
    configure_qa_plot_style()
    configs = [k for k in CONVERGENCE_CONFIGS if k in conv_results]

    fig = plt.figure(figsize=(15, 17))
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.0, 5.0],
        hspace=0.10,
    )

    # Top: residual panels
    n_c = len(configs)
    top = gridspec.GridSpecFromSubplotSpec(
        1, n_c + 1,
        subplot_spec=outer[0],
        width_ratios=[1.0] * n_c + [0.05],
        wspace=0.08,
    )

    # Bottom: profile panels
    n_rows = 7
    ratios = [2.5, 1.2, 1.2, 1.4, 1.4, 1.2, 1.2]
    bottom = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1,
        subplot_spec=outer[1],
        height_ratios=ratios,
        hspace=0.0,
    )

    fig.suptitle(
        f"NGC3610 — Convergence criteria comparison (EA mode, orders {HARMONIC_ORDERS})",
        fontsize=14, y=0.995,
    )

    # --- Top: 2D residuals ---
    residuals = {c: image - conv_models[c] for c in configs}
    all_abs = np.abs(np.concatenate([
        residuals[c][np.isfinite(residuals[c])] for c in configs
    ]))
    res_limit = float(np.clip(np.nanpercentile(all_abs, 99.0), 0.05, None))

    im_handle = None
    for col, c in enumerate(configs):
        ax = fig.add_subplot(top[0, col])
        im_handle = ax.imshow(
            residuals[c], origin="lower", cmap="coolwarm",
            vmin=-res_limit, vmax=res_limit, interpolation="nearest",
        )
        meta = conv_meta[c]
        valid = np.isfinite(image) & np.isfinite(conv_models[c]) & (np.abs(image) > 1e-6)
        rms = float(np.sqrt(np.nanmean(residuals[c][valid] ** 2)))
        ax.set_title(
            f"{CONVERGENCE_CONFIGS[c]['label']}\n"
            f"conv={meta['n_conv']}/{meta['n_iso']}  "
            f"niter={meta['mean_niter']:.1f}  RMS={rms:.4f}",
            fontsize=8, pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_subplot(top[0, n_c])
    fig.colorbar(im_handle, cax=cbar_ax, label="data - model")

    # --- Bottom: 1D profiles ---
    def _arr(isos, key, default=np.nan):
        return np.array([r.get(key, default) for r in isos])

    ax_sb = fig.add_subplot(bottom[0])
    ax_di = fig.add_subplot(bottom[1], sharex=ax_sb)
    ax_ni = fig.add_subplot(bottom[2], sharex=ax_sb)
    ax_odd = fig.add_subplot(bottom[3], sharex=ax_sb)
    ax_even = fig.add_subplot(bottom[4], sharex=ax_sb)
    ax_ba = fig.add_subplot(bottom[5], sharex=ax_sb)
    ax_pa = fig.add_subplot(bottom[6], sharex=ax_sb)
    all_axes = [ax_sb, ax_di, ax_ni, ax_odd, ax_even, ax_ba, ax_pa]

    legend_handles = []

    # Use default as reference for delta I
    ref_sma = _arr(conv_results["default"], "sma") if "default" in conv_results else None
    ref_intens = _arr(conv_results["default"], "intens") if "default" in conv_results else None

    for c in configs:
        isos = conv_results[c]
        sma = _arr(isos, "sma")
        intens = _arr(isos, "intens")
        eps = _arr(isos, "eps")
        pa_rad = _arr(isos, "pa")
        stop = _arr(isos, "stop_code", 0).astype(int)
        niter = _arr(isos, "niter", 0)

        xax = sma ** 0.25
        vm = np.isfinite(xax) & (sma > 1.0)
        good = vm & (stop == 0)
        bad = vm & (stop != 0)

        col = CONV_COLORS[c]
        mrk = CONV_MARKERS[c]
        label = CONVERGENCE_CONFIGS[c]["label"]

        skw_good = dict(s=18, marker=mrk, facecolors=col, edgecolors=col, alpha=0.75)
        skw_bad = dict(s=12, marker=mrk, facecolors="none", edgecolors=col, alpha=0.35)

        # 1. Surface brightness
        sb_ok = good & np.isfinite(intens) & (intens > 0)
        sb_bad = bad & np.isfinite(intens) & (intens > 0)
        y_sb = np.full_like(intens, np.nan)
        pos = np.isfinite(intens) & (intens > 0)
        y_sb[pos] = np.log10(intens[pos])
        ax_sb.scatter(xax[sb_ok], y_sb[sb_ok], **skw_good, label=label)
        if np.any(sb_bad):
            ax_sb.scatter(xax[sb_bad], y_sb[sb_bad], **skw_bad)

        # 2. Relative intensity difference vs default
        if ref_sma is not None and c != "default":
            from scipy.interpolate import interp1d
            valid_bl = np.isfinite(ref_sma) & np.isfinite(ref_intens) & (ref_intens > 0)
            if np.sum(valid_bl) > 3:
                fn = interp1d(ref_sma[valid_bl], ref_intens[valid_bl],
                              kind="linear", bounds_error=False, fill_value=np.nan)
                bl_at = fn(sma)
                delta_i = np.where(
                    np.isfinite(bl_at) & (bl_at > 0) & np.isfinite(intens),
                    (intens - bl_at) / bl_at, np.nan,
                )
                di_ok = good & np.isfinite(delta_i)
                di_bad = bad & np.isfinite(delta_i)
                ax_di.scatter(xax[di_ok], delta_i[di_ok] * 100, **skw_good)
                if np.any(di_bad):
                    ax_di.scatter(xax[di_bad], delta_i[di_bad] * 100, **skw_bad)

        # 3. Number of iterations per isophote
        ni_ok = good & np.isfinite(niter)
        ni_bad = bad & np.isfinite(niter)
        ax_ni.scatter(xax[ni_ok], niter[ni_ok], **skw_good)
        if np.any(ni_bad):
            ax_ni.scatter(xax[ni_bad], niter[ni_bad], **skw_bad)

        # 4. Odd harmonic amplitudes (A3, A5, A7)
        for order in [3, 5, 7]:
            an = _arr(isos, f"a{order}")
            bn = _arr(isos, f"b{order}")
            amp = np.sqrt(
                np.where(np.isfinite(an), an, 0.0) ** 2
                + np.where(np.isfinite(bn), bn, 0.0) ** 2
            )
            amp[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
            a_ok = good & np.isfinite(amp)
            a_bad = bad & np.isfinite(amp)
            if np.any(a_ok):
                ax_odd.scatter(xax[a_ok], amp[a_ok], **skw_good)
            if np.any(a_bad):
                ax_odd.scatter(xax[a_bad], amp[a_bad], **skw_bad)

        # 5. Even harmonic amplitudes (A4, A6)
        for order in [4, 6]:
            an = _arr(isos, f"a{order}")
            bn = _arr(isos, f"b{order}")
            amp = np.sqrt(
                np.where(np.isfinite(an), an, 0.0) ** 2
                + np.where(np.isfinite(bn), bn, 0.0) ** 2
            )
            amp[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
            a_ok = good & np.isfinite(amp)
            a_bad = bad & np.isfinite(amp)
            if np.any(a_ok):
                ax_even.scatter(xax[a_ok], amp[a_ok], **skw_good)
            if np.any(a_bad):
                ax_even.scatter(xax[a_bad], amp[a_bad], **skw_bad)

        # 6. Axis ratio
        ba = 1.0 - eps
        ba_ok = good & np.isfinite(ba)
        ba_bad = bad & np.isfinite(ba)
        ax_ba.scatter(xax[ba_ok], ba[ba_ok], **skw_good)
        if np.any(ba_bad):
            ax_ba.scatter(xax[ba_bad], ba[ba_bad], **skw_bad)

        # 7. PA
        pa_deg = normalize_pa_degrees(np.degrees(pa_rad))
        pa_ok = good & np.isfinite(pa_deg)
        pa_bad = bad & np.isfinite(pa_deg)
        ax_pa.scatter(xax[pa_ok], pa_deg[pa_ok], **skw_good)
        if np.any(pa_bad):
            ax_pa.scatter(xax[pa_bad], pa_deg[pa_bad], **skw_bad)

        legend_handles.append(
            Line2D([], [], marker=mrk, linestyle="None",
                   color=col, markersize=6, label=label)
        )

    # --- Labels ---
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_di.set_ylabel(r"$\Delta I / I_{\rm def}$ [%]")
    ax_di.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax_ni.set_ylabel("niter")
    ax_odd.set_ylabel(r"$\sqrt{a_n^2 + b_n^2}$ (odd)")
    ax_odd.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax_even.set_ylabel(r"$\sqrt{a_n^2 + b_n^2}$ (even)")
    ax_even.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.6)
    ax_ba.set_ylabel("b/a")
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.set_xlabel(r"SMA$^{0.25}$ [pixel$^{0.25}$]")

    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)
    for ax in all_axes:
        ax.grid(alpha=0.2)

    # X limits
    all_xvals = []
    for c in configs:
        s = _arr(conv_results[c], "sma")
        vm = np.isfinite(s) & (s > 1.0)
        if np.any(vm):
            all_xvals.append(s[vm] ** 0.25)
    if all_xvals:
        set_x_limits_with_right_margin(ax_pa, np.concatenate(all_xvals))

    # Y limits for SB
    sb_vals = []
    for c in configs:
        intens = _arr(conv_results[c], "intens")
        ok = np.isfinite(intens) & (intens > 0)
        if np.any(ok):
            sb_vals.append(np.log10(intens[ok]))
    if sb_vals:
        set_axis_limits_from_finite_values(ax_sb, np.concatenate(sb_vals), margin_fraction=0.06)

    # Y limits for delta I
    ax_di.set_ylim(-5, 5)

    # Y limits for b/a
    ba_vals = []
    for c in configs:
        eps = _arr(conv_results[c], "eps")
        stop = _arr(conv_results[c], "stop_code", 0).astype(int)
        ok = np.isfinite(eps) & (stop == 0)
        if np.any(ok):
            ba_vals.append(1.0 - eps[ok])
    if ba_vals:
        set_axis_limits_from_finite_values(
            ax_ba, np.concatenate(ba_vals), margin_fraction=0.06,
            lower_clip=0.0, upper_clip=1.0,
        )

    # Y limits for PA
    pa_vals = []
    for c in configs:
        pa_rad = _arr(conv_results[c], "pa")
        stop = _arr(conv_results[c], "stop_code", 0).astype(int)
        ok = np.isfinite(pa_rad) & (stop == 0)
        if np.any(ok):
            pa_vals.append(normalize_pa_degrees(np.degrees(pa_rad[ok])))
    if pa_vals:
        allp = np.concatenate(pa_vals)
        lo, hi = robust_limits(allp, 3, 97)
        margin = max(3.0, 0.08 * (hi - lo + 1e-6))
        ax_pa.set_ylim(lo - margin, hi + margin)

    # Y limits for harmonic panels
    for ax, orders in [(ax_odd, [3, 5, 7]), (ax_even, [4, 6])]:
        vals = []
        for c in configs:
            isos = conv_results[c]
            stop = _arr(isos, "stop_code", 0).astype(int)
            for order in orders:
                an = _arr(isos, f"a{order}")
                bn = _arr(isos, f"b{order}")
                amp = np.sqrt(
                    np.where(np.isfinite(an), an, 0.0) ** 2
                    + np.where(np.isfinite(bn), bn, 0.0) ** 2
                )
                amp[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
                ok = np.isfinite(amp) & (stop == 0)
                if np.any(ok):
                    vals.append(amp[ok])
        if vals:
            cat = np.concatenate(vals)
            set_axis_limits_from_finite_values(ax, cat, margin_fraction=0.1)

    ax_sb.legend(handles=legend_handles, loc="upper right", fontsize=8,
                 frameon=True, ncol=1)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.03, top=0.965)
    filename = str(output_dir / "ngc3610_convergence_comparison.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved convergence comparison -> {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = _HERE.parent.parent / "data"
    fits_path = data_dir / FITS_FILENAME[GALAXY]

    print("Loading NGC3610 (r-band)...")
    image, pixel_scale = load_legacysurvey_fits(fits_path, BAND_INDEX)
    print(f"  Shape: {image.shape}, pixel scale: {PIXEL_SCALE[GALAXY]} arcsec/px")

    print("Generating mask...")
    mask_kwargs = MASK_PARAMS.get(GALAXY, {})
    center_xy = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = make_object_mask(image, center_xy=center_xy, **mask_kwargs)
    print(f"  Masked: {np.sum(mask)}/{mask.size} ({100.0 * np.sum(mask) / mask.size:.1f}%)")

    sma0 = INITIAL_SMA[GALAXY]

    # ==================================================================
    # Investigation 1: Harmonic impact on model reconstruction
    # ==================================================================
    print(f"\n{'='*60}")
    print("  Investigation 1: Harmonic impact on 2D model")
    print(f"{'='*60}\n")

    # Run baseline (in_loop) with orders [3,4,5,6,7]
    config_inv1 = IsosterConfig(
        sma0=sma0,
        convergence_scaling="sector_area",
        geometry_damping=0.7,
        permissive_geometry=True,
        use_eccentric_anomaly=True,
        harmonic_orders=HARMONIC_ORDERS,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
    )
    print("  Running ISOFIT in-loop...")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_inv1 = isoster.fit_image(image, mask, config_inv1)
    elapsed = time.perf_counter() - t0
    isos = results_inv1["isophotes"]
    n_conv = sum(1 for r in isos if r.get("stop_code") == 0)
    print(f"    {len(isos)} isophotes, {n_conv} converged, {elapsed:.1f}s")

    # Check harmonic coefficients are actually non-zero
    for order in HARMONIC_ORDERS:
        an_vals = [r.get(f"a{order}", 0.0) for r in isos]
        bn_vals = [r.get(f"b{order}", 0.0) for r in isos]
        max_an = max(abs(v) for v in an_vals if np.isfinite(v))
        max_bn = max(abs(v) for v in bn_vals if np.isfinite(v))
        n_nonzero = sum(1 for a, b in zip(an_vals, bn_vals)
                        if abs(a) > 1e-10 or abs(b) > 1e-10)
        print(f"    Order {order}: max|a{order}|={max_an:.6f}, "
              f"max|b{order}|={max_bn:.6f}, non-zero={n_nonzero}/{len(isos)}")

    plot_harmonic_impact(image, mask, isos, "isofit_in_loop", OUTPUT_DIR)

    # Also do the same for baseline (post-hoc sequential)
    config_baseline = IsosterConfig(
        sma0=sma0,
        convergence_scaling="sector_area",
        geometry_damping=0.7,
        permissive_geometry=True,
        use_eccentric_anomaly=True,
        harmonic_orders=HARMONIC_ORDERS,
        simultaneous_harmonics=False,
    )
    print("\n  Running baseline (post-hoc sequential)...")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_baseline = isoster.fit_image(image, mask, config_baseline)
    elapsed = time.perf_counter() - t0
    isos_bl = results_baseline["isophotes"]
    n_conv_bl = sum(1 for r in isos_bl if r.get("stop_code") == 0)
    print(f"    {len(isos_bl)} isophotes, {n_conv_bl} converged, {elapsed:.1f}s")

    for order in HARMONIC_ORDERS:
        an_vals = [r.get(f"a{order}", 0.0) for r in isos_bl]
        bn_vals = [r.get(f"b{order}", 0.0) for r in isos_bl]
        max_an = max(abs(v) for v in an_vals if np.isfinite(v))
        max_bn = max(abs(v) for v in bn_vals if np.isfinite(v))
        n_nonzero = sum(1 for a, b in zip(an_vals, bn_vals)
                        if abs(a) > 1e-10 or abs(b) > 1e-10)
        print(f"    Order {order}: max|a{order}|={max_an:.6f}, "
              f"max|b{order}|={max_bn:.6f}, non-zero={n_nonzero}/{len(isos_bl)}")

    plot_harmonic_impact(image, mask, isos_bl, "baseline", OUTPUT_DIR)

    # Per-run extended QA for reference
    for label, res_isos in [("isofit_in_loop", isos), ("baseline", isos_bl)]:
        model = build_isoster_model(image.shape, res_isos, use_harmonics=True)
        n_c = sum(1 for r in res_isos if r.get("stop_code") == 0)
        plot_qa_summary_extended(
            title=f"NGC3610 — {label} (orders {HARMONIC_ORDERS})\nconv={n_c}/{len(res_isos)}",
            image=image,
            isoster_model=model,
            isoster_res=res_isos,
            harmonic_mode="coefficients",
            relative_residual=False,
            mask=mask,
            filename=str(OUTPUT_DIR / f"ngc3610_{label}_extended_qa.png"),
        )

    # ==================================================================
    # Investigation 2: Convergence criteria comparison
    # ==================================================================
    print(f"\n{'='*60}")
    print("  Investigation 2: Convergence criteria comparison")
    print(f"{'='*60}\n")

    conv_results = {}
    conv_models = {}
    conv_meta = {}

    for c_name, c_params in CONVERGENCE_CONFIGS.items():
        config = IsosterConfig(
            sma0=sma0,
            convergence_scaling="sector_area",
            geometry_damping=0.7,
            permissive_geometry=True,
            use_eccentric_anomaly=True,
            harmonic_orders=HARMONIC_ORDERS,
            simultaneous_harmonics=True,
            isofit_mode="in_loop",
            conver=c_params["conver"],
            maxit=c_params["maxit"],
        )
        print(f"  [{c_name}] conver={c_params['conver']}, maxit={c_params['maxit']}")
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = isoster.fit_image(image, mask, config)
        elapsed = time.perf_counter() - t0

        isos = results["isophotes"]
        n_iso = len(isos)
        n_conv = sum(1 for r in isos if r.get("stop_code") == 0)
        niters = [r.get("niter", 0) for r in isos]
        mean_niter = float(np.mean(niters))
        max_niter = max(niters)
        print(f"    -> {n_iso} isophotes, {n_conv} converged, "
              f"mean_niter={mean_niter:.1f}, max_niter={max_niter}, {elapsed:.1f}s")

        model = build_isoster_model(image.shape, isos, use_harmonics=True)

        conv_results[c_name] = isos
        conv_models[c_name] = model
        conv_meta[c_name] = {
            "elapsed": elapsed,
            "n_iso": n_iso,
            "n_conv": n_conv,
            "mean_niter": mean_niter,
        }

        # Per-run extended QA
        valid = np.isfinite(image) & np.isfinite(model) & (np.abs(image) > 1e-6)
        rms = float(np.sqrt(np.nanmean((image[valid] - model[valid]) ** 2)))
        plot_qa_summary_extended(
            title=(
                f"NGC3610 — ISOFIT in-loop ({c_params['label']})\n"
                f"conv={n_conv}/{n_iso}  niter={mean_niter:.1f}  RMS={rms:.4f}"
            ),
            image=image,
            isoster_model=model,
            isoster_res=isos,
            harmonic_mode="coefficients",
            relative_residual=False,
            mask=mask,
            filename=str(OUTPUT_DIR / f"ngc3610_conv_{c_name}_qa.png"),
        )

    # Convergence comparison figure
    plot_convergence_comparison(
        image, mask, conv_results, conv_models, conv_meta, OUTPUT_DIR,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for c_name in CONVERGENCE_CONFIGS:
        meta = conv_meta[c_name]
        model = conv_models[c_name]
        valid = np.isfinite(image) & np.isfinite(model) & (np.abs(image) > 1e-6)
        rms = float(np.sqrt(np.nanmean((image[valid] - model[valid]) ** 2)))
        print(f"  {CONVERGENCE_CONFIGS[c_name]['label']:>45s}: "
              f"conv={meta['n_conv']}/{meta['n_iso']}  "
              f"niter={meta['mean_niter']:5.1f}  "
              f"RMS={rms:.6f}  t={meta['elapsed']:.1f}s")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
