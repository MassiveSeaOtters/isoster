"""
NGC3610 Initial SMA (sma0) Effect Exploration
===============================================

Probe whether the choice of initial semi-major axis ``sma0`` affects
convergence, geometry propagation, and final profile quality for NGC3610
with in-loop ISOFIT harmonics [3,4,5,6,7].

Seven values spanning ~2 orders of magnitude:
  [2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 80.0]

- 2.0, 4.0: Near-center (pixelisation effects, few sample points)
- 6.0: Current default (reference for delta-I)
- 10.0, 20.0: Moderate (well-constrained geometry)
- 40.0, 80.0: Large (initial geometry guess must propagate inward)

Output goes to ``outputs/example_ls_highorder_harmonic/ngc3610_sma0_effect/``.
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
from scipy.interpolate import interp1d

import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import (
    configure_qa_plot_style,
    normalize_pa_degrees,
    plot_qa_summary_extended,
    robust_limits,
    set_axis_limits_from_finite_values,
    set_x_limits_with_right_margin,
)

# Reuse data-loading helpers
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from shared import (  # noqa: E402
    FITS_FILENAME,
    PIXEL_SCALE,
    load_legacysurvey_fits,
)

OUTPUT_DIR = Path("outputs/example_ls_highorder_harmonic/ngc3610_sma0_effect")
GALAXY = "ngc3610"
BAND_INDEX = 1  # r-band
HARMONIC_ORDERS = [3, 4, 5, 6, 7]

# --- Test values and visual configuration ---

SMA0_VALUES = [2.0, 4.0, 6.0, 10.0, 20.0, 40.0, 80.0]
REFERENCE_SMA0 = 6.0  # reference for delta-I

SMA0_COLORS = {
    2.0: "#e377c2",
    4.0: "#ff7f0e",
    6.0: "#1f77b4",
    10.0: "#2ca02c",
    20.0: "#d62728",
    40.0: "#9467bd",
    80.0: "#8c564b",
}

SMA0_MARKERS = {
    2.0: "o",
    4.0: "^",
    6.0: "s",
    10.0: "D",
    20.0: "P",
    40.0: "v",
    80.0: "X",
}


def _arr(isos: list[dict], key: str, default=np.nan) -> np.ndarray:
    """Extract an array of values from isophote dicts."""
    return np.array([r.get(key, default) for r in isos])


def plot_sma0_comparison(
    image: np.ndarray,
    case_results: dict[float, list[dict]],
    case_models: dict[float, np.ndarray],
    case_meta: dict[float, dict],
    output_dir: Path,
) -> None:
    """Build the multi-panel comparison figure for all sma0 values.

    Top: 2D residual maps (2 rows of 4).
    Bottom: 7 stacked 1D panels sharing SMA^0.25 x-axis.
    """
    configure_qa_plot_style()
    cases = [s for s in SMA0_VALUES if s in case_results]
    n_cases = len(cases)

    fig = plt.figure(figsize=(18, 22))
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.6, 5.0],
        hspace=0.10,
    )

    # Top: 2D residual maps in 2 rows of 4 (last slot empty if 7 cases)
    n_cols_top = 4
    n_rows_top = 2
    top = gridspec.GridSpecFromSubplotSpec(
        n_rows_top, n_cols_top + 1,
        subplot_spec=outer[0],
        width_ratios=[1.0] * n_cols_top + [0.05],
        wspace=0.08, hspace=0.18,
    )

    # Bottom: 1D profiles (7 rows)
    n_rows_bottom = 7
    ratios = [2.5, 1.2, 1.2, 1.4, 1.4, 1.2, 1.2]
    bottom = gridspec.GridSpecFromSubplotSpec(
        n_rows_bottom, 1,
        subplot_spec=outer[1],
        height_ratios=ratios,
        hspace=0.0,
    )

    fig.suptitle(
        f"NGC3610 — sma0 effect on isophote fitting "
        f"(EA, ISOFIT in-loop, orders {HARMONIC_ORDERS})",
        fontsize=14, y=0.995,
    )

    # --- Top: 2D residual maps ---
    residuals = {}
    for s in cases:
        residuals[s] = image - case_models[s]

    all_abs = np.abs(np.concatenate([
        residuals[s][np.isfinite(residuals[s])] for s in cases
    ]))
    res_limit = float(np.clip(np.nanpercentile(all_abs, 99.0), 0.05, None))

    im_handle = None
    for idx, s in enumerate(cases):
        row = idx // n_cols_top
        col = idx % n_cols_top
        ax = fig.add_subplot(top[row, col])
        im_handle = ax.imshow(
            residuals[s], origin="lower", cmap="coolwarm",
            vmin=-res_limit, vmax=res_limit, interpolation="nearest",
        )
        meta = case_meta[s]
        valid = np.isfinite(image) & np.isfinite(case_models[s]) & (np.abs(image) > 1e-6)
        rms = float(np.sqrt(np.nanmean(residuals[s][valid] ** 2)))
        ax.set_title(
            f"sma0={s:.0f} ({meta['n_conv']}/{meta['n_iso']}, RMS={rms:.5f})",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Fill unused slots
    for idx in range(n_cases, n_rows_top * n_cols_top):
        row = idx // n_cols_top
        col = idx % n_cols_top
        ax = fig.add_subplot(top[row, col])
        ax.set_visible(False)

    # Colorbar spanning both rows
    cbar_ax = fig.add_subplot(top[:, n_cols_top])
    fig.colorbar(im_handle, cax=cbar_ax, label="data - model")

    # --- Bottom: 1D profiles ---
    ax_sb = fig.add_subplot(bottom[0])
    ax_di = fig.add_subplot(bottom[1], sharex=ax_sb)
    ax_ni = fig.add_subplot(bottom[2], sharex=ax_sb)
    ax_odd = fig.add_subplot(bottom[3], sharex=ax_sb)
    ax_even = fig.add_subplot(bottom[4], sharex=ax_sb)
    ax_ba = fig.add_subplot(bottom[5], sharex=ax_sb)
    ax_pa = fig.add_subplot(bottom[6], sharex=ax_sb)
    all_axes = [ax_sb, ax_di, ax_ni, ax_odd, ax_even, ax_ba, ax_pa]

    legend_handles = []

    # Reference for delta-I
    ref_isos = case_results.get(REFERENCE_SMA0)
    ref_sma = _arr(ref_isos, "sma") if ref_isos else None
    ref_intens = _arr(ref_isos, "intens") if ref_isos else None

    for s in cases:
        isos = case_results[s]
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

        col = SMA0_COLORS[s]
        mrk = SMA0_MARKERS[s]
        label = f"sma0={s:.0f}"

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

        # 2. Relative intensity difference vs reference sma0
        if ref_sma is not None and s != REFERENCE_SMA0:
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

    # Vertical dashed lines at each sma0^0.25 position
    for s in cases:
        xpos = s ** 0.25
        col = SMA0_COLORS[s]
        for ax in all_axes:
            ax.axvline(xpos, color=col, ls="--", lw=0.7, alpha=0.4)

    # --- Labels ---
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_di.set_ylabel(r"$\Delta I / I_{\rm ref}$ [%]")
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

    # --- Axis limits ---
    # X limits
    all_xvals = []
    for s in cases:
        sma = _arr(case_results[s], "sma")
        vm = np.isfinite(sma) & (sma > 1.0)
        if np.any(vm):
            all_xvals.append(sma[vm] ** 0.25)
    if all_xvals:
        set_x_limits_with_right_margin(ax_pa, np.concatenate(all_xvals))

    # SB limits
    sb_vals = []
    for s in cases:
        intens = _arr(case_results[s], "intens")
        ok = np.isfinite(intens) & (intens > 0)
        if np.any(ok):
            sb_vals.append(np.log10(intens[ok]))
    if sb_vals:
        set_axis_limits_from_finite_values(ax_sb, np.concatenate(sb_vals), margin_fraction=0.06)

    # Delta-I limits
    ax_di.set_ylim(-10, 10)

    # b/a limits
    ba_vals = []
    for s in cases:
        eps = _arr(case_results[s], "eps")
        stop = _arr(case_results[s], "stop_code", 0).astype(int)
        ok = np.isfinite(eps) & (stop == 0)
        if np.any(ok):
            ba_vals.append(1.0 - eps[ok])
    if ba_vals:
        set_axis_limits_from_finite_values(
            ax_ba, np.concatenate(ba_vals), margin_fraction=0.06,
            lower_clip=0.0, upper_clip=1.0,
        )

    # PA limits
    pa_vals = []
    for s in cases:
        pa_rad = _arr(case_results[s], "pa")
        stop = _arr(case_results[s], "stop_code", 0).astype(int)
        ok = np.isfinite(pa_rad) & (stop == 0)
        if np.any(ok):
            pa_vals.append(normalize_pa_degrees(np.degrees(pa_rad[ok])))
    if pa_vals:
        allp = np.concatenate(pa_vals)
        lo, hi = robust_limits(allp, 3, 97)
        margin = max(3.0, 0.08 * (hi - lo + 1e-6))
        ax_pa.set_ylim(lo - margin, hi + margin)

    # Harmonic amplitude limits
    for ax, orders in [(ax_odd, [3, 5, 7]), (ax_even, [4, 6])]:
        vals = []
        for s in cases:
            isos = case_results[s]
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

    ax_sb.legend(handles=legend_handles, loc="upper right", fontsize=7,
                 frameon=True, ncol=2)

    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.03, top=0.965)
    filename = str(output_dir / "ngc3610_sma0_effect_comparison.png")
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved sma0 comparison -> {filename}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = _HERE.parent.parent / "data"
    fits_path = data_dir / FITS_FILENAME[GALAXY]

    print("Loading NGC3610 (r-band)...")
    image, pixel_scale = load_legacysurvey_fits(fits_path, BAND_INDEX)
    print(f"  Shape: {image.shape}, pixel scale: {PIXEL_SCALE[GALAXY]} arcsec/px")

    # Fixed config kwargs (only sma0 varies)
    base_config_kwargs = dict(
        convergence_scaling="sector_area",
        geometry_damping=0.7,
        permissive_geometry=True,
        use_eccentric_anomaly=True,
        harmonic_orders=HARMONIC_ORDERS,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
    )

    case_results: dict[float, list[dict]] = {}
    case_models: dict[float, np.ndarray] = {}
    case_meta: dict[float, dict] = {}

    for sma0 in SMA0_VALUES:
        config = IsosterConfig(sma0=sma0, **base_config_kwargs)
        print(f"\n  [sma0={sma0:.0f}] Fitting...")

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                results = isoster.fit_image(image, None, config)
            except Exception as exc:
                print(f"    FAILED: {exc}")
                continue
        elapsed = time.perf_counter() - t0

        isos = results["isophotes"]
        if not isos:
            print("    No isophotes returned — skipping")
            continue

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

        case_results[sma0] = isos
        case_models[sma0] = model
        case_meta[sma0] = {
            "elapsed": elapsed,
            "n_iso": n_iso,
            "n_conv": n_conv,
            "mean_niter": mean_niter,
            "rms": rms,
        }

        # Per-run extended QA
        plot_qa_summary_extended(
            title=(
                f"NGC3610 — ISOFIT in-loop (sma0={sma0:.0f})\n"
                f"conv={n_conv}/{n_iso}  niter={mean_niter:.1f}  RMS={rms:.5f}"
            ),
            image=image,
            isoster_model=model,
            isoster_res=isos,
            harmonic_mode="coefficients",
            relative_residual=False,
            mask=None,
            filename=str(OUTPUT_DIR / f"ngc3610_sma0_{sma0:.0f}_qa.png"),
        )

    if len(case_results) < 2:
        print("\nToo few successful runs for comparison. Exiting.")
        return

    # Comparison figure
    plot_sma0_comparison(image, case_results, case_models, case_meta, OUTPUT_DIR)

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'sma0':>6s}  {'N_iso':>5s}  {'N_conv':>6s}  "
          f"{'mean_niter':>10s}  {'RMS':>10s}  {'time(s)':>7s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*7}")
    for s in SMA0_VALUES:
        if s not in case_meta:
            print(f"  {s:6.1f}  {'FAILED':>5s}")
            continue
        m = case_meta[s]
        print(f"  {s:6.1f}  {m['n_iso']:5d}  {m['n_conv']:6d}  "
              f"{m['mean_niter']:10.1f}  {m['rms']:10.6f}  {m['elapsed']:7.1f}")
    print(f"{'='*72}")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
