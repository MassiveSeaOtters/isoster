"""
NGC3610 Mask Effect Exploration
================================

Compare fitting results with the object mask vs no mask to understand
how the mask affects isophote fitting and model reconstruction.

Output goes to ``outputs/example_ls_highorder_harmonic/ngc3610_mask_effect/``.
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
    make_arcsinh_display_from_parameters,
    normalize_pa_degrees,
    plot_qa_summary_extended,
    robust_limits,
    set_axis_limits_from_finite_values,
    set_x_limits_with_right_margin,
)

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

OUTPUT_DIR = Path("outputs/example_ls_highorder_harmonic/ngc3610_mask_effect")
GALAXY = "ngc3610"
BAND_INDEX = 1
HARMONIC_ORDERS = [3, 4, 5, 6, 7]

CASE_ORDER = ["with_mask", "no_mask"]
CASE_DISPLAY = {
    "with_mask": "With object mask",
    "no_mask": "No mask",
}
CASE_COLORS = {
    "with_mask": "#1f77b4",
    "no_mask": "#d62728",
}
CASE_MARKERS = {
    "with_mask": "o",
    "no_mask": "^",
}


def plot_mask_comparison(
    image: np.ndarray,
    mask: np.ndarray | None,
    case_results: dict[str, list[dict]],
    case_models: dict[str, np.ndarray],
    case_meta: dict[str, dict],
    output_dir: Path,
) -> None:
    """Side-by-side comparison of masked vs unmasked fitting.

    Top row: data+mask, residual (masked), residual (no mask), mask overlay
    Bottom: 1D profile stack comparing both cases.
    """
    configure_qa_plot_style()
    cases = [c for c in CASE_ORDER if c in case_results]

    fig = plt.figure(figsize=(16, 18))
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.2, 5.0],
        hspace=0.10,
    )

    # Top: 4 panels — data+mask, residual (masked), residual (no mask), mask display
    top = gridspec.GridSpecFromSubplotSpec(
        1, 5,
        subplot_spec=outer[0],
        width_ratios=[1, 1, 1, 1, 0.05],
        wspace=0.10,
    )

    # Bottom: 1D profiles (7 rows)
    n_rows = 7
    ratios = [2.5, 1.2, 1.2, 1.4, 1.4, 1.2, 1.2]
    bottom = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1,
        subplot_spec=outer[1],
        height_ratios=ratios,
        hspace=0.0,
    )

    fig.suptitle(
        f"NGC3610 — Mask effect on isophote fitting (EA, orders {HARMONIC_ORDERS})",
        fontsize=14, y=0.995,
    )

    # --- Top row ---
    ref_low, ref_high, ref_scale, ref_vmax = derive_arcsinh_parameters(image)

    # Panel 1: Data + mask overlay
    ax_data = fig.add_subplot(top[0, 0])
    img_disp, iv_min, iv_max = make_arcsinh_display_from_parameters(
        image, low=ref_low, high=ref_high, scale=ref_scale, vmax=ref_vmax,
    )
    ax_data.imshow(img_disp, origin="lower", cmap="viridis",
                   vmin=iv_min, vmax=iv_max, interpolation="none")
    if mask is not None:
        overlay = np.zeros((*image.shape, 4))
        overlay[mask] = [1, 0, 0, 0.4]
        ax_data.imshow(overlay, origin="lower")
    ax_data.set_title("Data + mask", fontsize=10)

    # Panel 2 & 3: Residuals
    residuals = {}
    for c in cases:
        residuals[c] = image - case_models[c]
    all_abs = np.abs(np.concatenate([
        residuals[c][np.isfinite(residuals[c])] for c in cases
    ]))
    res_limit = float(np.clip(np.nanpercentile(all_abs, 99.0), 0.05, None))

    im_handle = None
    for idx, c in enumerate(cases):
        ax = fig.add_subplot(top[0, 1 + idx])
        im_handle = ax.imshow(
            residuals[c], origin="lower", cmap="coolwarm",
            vmin=-res_limit, vmax=res_limit, interpolation="nearest",
        )
        meta = case_meta[c]
        valid = np.isfinite(image) & np.isfinite(case_models[c]) & (np.abs(image) > 1e-6)
        rms = float(np.sqrt(np.nanmean(residuals[c][valid] ** 2)))
        ax.set_title(
            f"Residual ({CASE_DISPLAY[c]})\n"
            f"conv={meta['n_conv']}/{meta['n_iso']}  RMS={rms:.5f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel 4: Difference between the two residuals
    ax_diff = fig.add_subplot(top[0, 3])
    diff = residuals["no_mask"] - residuals["with_mask"]
    diff_abs = np.abs(diff[np.isfinite(diff)])
    diff_limit = float(np.clip(np.nanpercentile(diff_abs, 99.0), 1e-4, None))
    ax_diff.imshow(
        diff, origin="lower", cmap="coolwarm",
        vmin=-diff_limit, vmax=diff_limit, interpolation="nearest",
    )
    ax_diff.set_title("Residual difference\n(no_mask - with_mask)", fontsize=9)
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])

    cbar_ax = fig.add_subplot(top[0, 4])
    fig.colorbar(im_handle, cax=cbar_ax, label="data - model")

    for ax in [ax_data]:
        ax.set_xticks([])
        ax.set_yticks([])

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

    # Use with_mask as reference for delta I
    ref_sma = _arr(case_results["with_mask"], "sma")
    ref_intens = _arr(case_results["with_mask"], "intens")

    for c in cases:
        isos = case_results[c]
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

        col = CASE_COLORS[c]
        mrk = CASE_MARKERS[c]
        label = CASE_DISPLAY[c]

        skw_good = dict(s=18, marker=mrk, facecolors=col, edgecolors=col, alpha=0.75)
        skw_bad = dict(s=12, marker=mrk, facecolors="none", edgecolors=col, alpha=0.35)

        # 1. Surface brightness
        pos = np.isfinite(intens) & (intens > 0)
        y_sb = np.full_like(intens, np.nan)
        y_sb[pos] = np.log10(intens[pos])
        sb_ok = good & pos
        sb_bad = bad & pos
        ax_sb.scatter(xax[sb_ok], y_sb[sb_ok], **skw_good, label=label)
        if np.any(sb_bad):
            ax_sb.scatter(xax[sb_bad], y_sb[sb_bad], **skw_bad)

        # 2. Relative intensity difference vs with_mask
        if c != "with_mask":
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

        # 3. Number of iterations
        ni_ok = good & np.isfinite(niter)
        ni_bad = bad & np.isfinite(niter)
        ax_ni.scatter(xax[ni_ok], niter[ni_ok], **skw_good)
        if np.any(ni_bad):
            ax_ni.scatter(xax[ni_bad], niter[ni_bad], **skw_bad)

        # 4. Odd harmonics (A3, A5, A7)
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

        # 5. Even harmonics (A4, A6)
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

    # Labels
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_di.set_ylabel(r"$\Delta I / I_{\rm mask}$ [%]")
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

    # Axis limits
    all_xvals = []
    for c in cases:
        s = _arr(case_results[c], "sma")
        vm = np.isfinite(s) & (s > 1.0)
        if np.any(vm):
            all_xvals.append(s[vm] ** 0.25)
    if all_xvals:
        set_x_limits_with_right_margin(ax_pa, np.concatenate(all_xvals))

    sb_vals = []
    for c in cases:
        intens = _arr(case_results[c], "intens")
        ok = np.isfinite(intens) & (intens > 0)
        if np.any(ok):
            sb_vals.append(np.log10(intens[ok]))
    if sb_vals:
        set_axis_limits_from_finite_values(ax_sb, np.concatenate(sb_vals), margin_fraction=0.06)

    ax_di.set_ylim(-10, 10)

    ba_vals = []
    for c in cases:
        eps = _arr(case_results[c], "eps")
        stop = _arr(case_results[c], "stop_code", 0).astype(int)
        ok = np.isfinite(eps) & (stop == 0)
        if np.any(ok):
            ba_vals.append(1.0 - eps[ok])
    if ba_vals:
        set_axis_limits_from_finite_values(
            ax_ba, np.concatenate(ba_vals), margin_fraction=0.06,
            lower_clip=0.0, upper_clip=1.0,
        )

    pa_vals = []
    for c in cases:
        pa_rad = _arr(case_results[c], "pa")
        stop = _arr(case_results[c], "stop_code", 0).astype(int)
        ok = np.isfinite(pa_rad) & (stop == 0)
        if np.any(ok):
            pa_vals.append(normalize_pa_degrees(np.degrees(pa_rad[ok])))
    if pa_vals:
        allp = np.concatenate(pa_vals)
        lo, hi = robust_limits(allp, 3, 97)
        margin = max(3.0, 0.08 * (hi - lo + 1e-6))
        ax_pa.set_ylim(lo - margin, hi + margin)

    for ax, orders in [(ax_odd, [3, 5, 7]), (ax_even, [4, 6])]:
        vals = []
        for c in cases:
            isos = case_results[c]
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

    ax_sb.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.03, top=0.965)
    filename = str(output_dir / "ngc3610_mask_effect_comparison.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved mask comparison -> {filename}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = _HERE.parent.parent / "data"
    fits_path = data_dir / FITS_FILENAME[GALAXY]

    print("Loading NGC3610 (r-band)...")
    image, pixel_scale = load_legacysurvey_fits(fits_path, BAND_INDEX)
    print(f"  Shape: {image.shape}, pixel scale: {PIXEL_SCALE[GALAXY]} arcsec/px")

    # Prepare mask
    mask_kwargs = MASK_PARAMS.get(GALAXY, {})
    center_xy = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = make_object_mask(image, center_xy=center_xy, **mask_kwargs)
    n_masked = int(np.sum(mask))
    print(f"  Mask: {n_masked}/{mask.size} pixels ({100.0 * n_masked / mask.size:.1f}%)")

    sma0 = INITIAL_SMA[GALAXY]

    base_config_kwargs = dict(
        sma0=sma0,
        convergence_scaling="sector_area",
        geometry_damping=0.7,
        permissive_geometry=True,
        use_eccentric_anomaly=True,
        harmonic_orders=HARMONIC_ORDERS,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
    )

    case_results = {}
    case_models = {}
    case_meta = {}

    for c_name, c_mask in [("with_mask", mask), ("no_mask", None)]:
        config = IsosterConfig(**base_config_kwargs)
        mask_label = f"{np.sum(c_mask)}/{c_mask.size} masked" if c_mask is not None else "no mask"
        print(f"\n  [{c_name}] {mask_label}")

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = isoster.fit_image(image, c_mask, config)
        elapsed = time.perf_counter() - t0

        isos = results["isophotes"]
        n_iso = len(isos)
        n_conv = sum(1 for r in isos if r.get("stop_code") == 0)
        niters = [r.get("niter", 0) for r in isos]
        mean_niter = float(np.mean(niters))
        print(f"    -> {n_iso} isophotes, {n_conv} converged, "
              f"mean_niter={mean_niter:.1f}, {elapsed:.1f}s")

        model = build_isoster_model(image.shape, isos, use_harmonics=True)

        valid = np.isfinite(image) & np.isfinite(model) & (np.abs(image) > 1e-6)
        rms = float(np.sqrt(np.nanmean((image[valid] - model[valid]) ** 2)))
        print(f"    RMS (with harmonics): {rms:.6f}")

        case_results[c_name] = isos
        case_models[c_name] = model
        case_meta[c_name] = {
            "elapsed": elapsed,
            "n_iso": n_iso,
            "n_conv": n_conv,
            "mean_niter": mean_niter,
        }

        # Per-run extended QA
        plot_qa_summary_extended(
            title=(
                f"NGC3610 — ISOFIT in-loop ({CASE_DISPLAY[c_name]})\n"
                f"conv={n_conv}/{n_iso}  niter={mean_niter:.1f}  RMS={rms:.5f}"
            ),
            image=image,
            isoster_model=model,
            isoster_res=isos,
            harmonic_mode="coefficients",
            relative_residual=False,
            mask=c_mask,
            filename=str(OUTPUT_DIR / f"ngc3610_{c_name}_qa.png"),
        )

    # Comparison figure
    plot_mask_comparison(
        image, mask, case_results, case_models, case_meta, OUTPUT_DIR,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
