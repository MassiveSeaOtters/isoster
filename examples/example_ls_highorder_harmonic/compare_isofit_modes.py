"""
ISOFIT Mode Comparison: Original (Ciambur 2015) vs In-Loop
==========================================================

Compares three fitting configurations on LegacySurvey galaxies (ESO243-49,
NGC3610) to evaluate algorithmic differences:

1. **Baseline** (``simultaneous_harmonics=False``): Post-hoc sequential
   harmonic fitting (default isoster behavior).
2. **ISOFIT in_loop** (``isofit_mode='in_loop'``): Simultaneous harmonic
   fitting inside the iteration loop (current isoster ISOFIT).
3. **ISOFIT original** (``isofit_mode='original'``): 5-param geometry inside
   the loop, simultaneous post-hoc harmonics after convergence (Ciambur 2015).

Usage
-----
::

    uv run python examples/example_ls_highorder_harmonic/compare_isofit_modes.py --galaxy eso243-49
    uv run python examples/example_ls_highorder_harmonic/compare_isofit_modes.py --galaxy ngc3610
    uv run python examples/example_ls_highorder_harmonic/compare_isofit_modes.py --galaxy all

Output goes to ``outputs/example_ls_highorder_harmonic/isofit_mode_comparison/<galaxy>/``.
"""

from __future__ import annotations

import argparse
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
    normalize_pa_degrees,
    plot_qa_summary_extended,
    robust_limits,
    set_axis_limits_from_finite_values,
    set_x_limits_with_right_margin,
)

# Reuse data-loading and masking from the main script in this folder.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from masking import make_object_mask  # noqa: E402
from shared import (  # noqa: E402
    BAND_NAMES,
    FITS_FILENAME,
    INITIAL_SMA,
    MASK_PARAMS,
    PIXEL_SCALE,
    SUPPORTED_GALAXIES,
    load_legacysurvey_fits,
)

# ---------------------------------------------------------------------------
# Three ISOFIT mode configurations
# ---------------------------------------------------------------------------

MODE_ORDER = ["baseline", "isofit_in_loop", "isofit_original"]

MODE_DISPLAY = {
    "baseline": "Baseline (post-hoc sequential)",
    "isofit_in_loop": "ISOFIT in-loop (isoster)",
    "isofit_original": "ISOFIT original (Ciambur 2015)",
}

MODE_COLORS = {
    "baseline": "#1f77b4",
    "isofit_in_loop": "#d62728",
    "isofit_original": "#2ca02c",
}

MODE_MARKERS = {
    "baseline": "o",
    "isofit_in_loop": "^",
    "isofit_original": "s",
}


def make_mode_configs(galaxy: str) -> dict[str, IsosterConfig]:
    """Build IsosterConfig for each of the three modes."""
    sma0 = INITIAL_SMA[galaxy]
    base = dict(
        sma0=sma0,
        convergence_scaling="sector_area",
        geometry_damping=0.7,
        permissive_geometry=True,
        use_eccentric_anomaly=True,
        harmonic_orders=[3, 4, 5, 6, 7],
    )
    return {
        "baseline": IsosterConfig(
            **base,
            simultaneous_harmonics=False,
        ),
        "isofit_in_loop": IsosterConfig(
            **base,
            simultaneous_harmonics=True,
            isofit_mode="in_loop",
        ),
        "isofit_original": IsosterConfig(
            **base,
            simultaneous_harmonics=True,
            isofit_mode="original",
        ),
    }


# ---------------------------------------------------------------------------
# Residual statistics
# ---------------------------------------------------------------------------

def compute_residual_statistics(
    image: np.ndarray,
    model: np.ndarray,
    isophotes: list[dict],
) -> dict:
    """Compute fractional residual statistics within radial annuli."""
    cx = np.nanmedian([r["x0"] for r in isophotes if np.isfinite(r.get("x0", np.nan))])
    cy = np.nanmedian([r["y0"] for r in isophotes if np.isfinite(r.get("y0", np.nan))])
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    r_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Restrict to pixels with significant signal to avoid near-zero division artifacts
    noise_floor = np.nanpercentile(np.abs(image[np.isfinite(image)]), 10)
    valid = np.isfinite(image) & np.isfinite(model) & (np.abs(image) > max(noise_floor, 1e-10))
    frac_residual = np.where(valid, (model - image) / image, np.nan)
    abs_frac_residual = np.abs(frac_residual)

    sma_arr = np.array([r["sma"] for r in isophotes if np.isfinite(r.get("sma", np.nan))])
    max_sma = np.nanmax(sma_arr) if len(sma_arr) > 0 else image.shape[0] / 2

    stats = {}
    annuli = [
        ("inner (r<0.25*max)", 0, 0.25 * max_sma),
        ("mid (0.25-0.6*max)", 0.25 * max_sma, 0.6 * max_sma),
        ("outer (0.6-1.0*max)", 0.6 * max_sma, max_sma),
        ("all (r<max)", 0, max_sma),
    ]
    for label, r_lo, r_hi in annuli:
        ring = valid & (r_map >= r_lo) & (r_map < r_hi)
        if np.sum(ring) > 10:
            stats[label] = {
                "median_frac_pct": float(np.nanmedian(frac_residual[ring]) * 100),
                "median_abs_frac_pct": float(np.nanmedian(abs_frac_residual[ring]) * 100),
                "rms_frac_pct": float(np.nanstd(frac_residual[ring]) * 100),
                "npix": int(np.sum(ring)),
            }
        else:
            stats[label] = {"median_frac_pct": np.nan, "median_abs_frac_pct": np.nan,
                            "rms_frac_pct": np.nan, "npix": 0}
    return stats


# ---------------------------------------------------------------------------
# QA figure
# ---------------------------------------------------------------------------

def plot_isofit_mode_comparison(
    galaxy: str,
    image: np.ndarray,
    mask: np.ndarray | None,
    mode_results: dict[str, list[dict]],
    mode_models: dict[str, np.ndarray],
    mode_stats: dict[str, dict],
    mode_meta: dict[str, dict],
    filename: str,
) -> None:
    """Build the comparison QA figure.

    Layout:
    - Top row: 3 residual map panels (one per mode), shared colorbar
    - Bottom section: 6 profile rows sharing SMA^0.25 x-axis:
      log10(I), relative delta I, a4, b4, axis ratio, PA
    """
    configure_qa_plot_style()

    modes = [m for m in MODE_ORDER if m in mode_results]
    n_modes = len(modes)

    # Figure layout
    fig = plt.figure(figsize=(14, 16))
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.0, 4.5],
        hspace=0.10,
    )

    # Top: residual panels
    top = gridspec.GridSpecFromSubplotSpec(
        1, n_modes + 1,
        subplot_spec=outer[0],
        width_ratios=[1.0] * n_modes + [0.05],
        wspace=0.08,
    )

    # Bottom: profile panels (SB, delta I, odd harm, even harm, b/a, PA)
    n_profile_rows = 6
    profile_ratios = [2.5, 1.2, 1.4, 1.4, 1.2, 1.2]
    bottom = gridspec.GridSpecFromSubplotSpec(
        n_profile_rows, 1,
        subplot_spec=outer[1],
        height_ratios=profile_ratios,
        hspace=0.0,
    )

    fig.suptitle(
        f"{galaxy.upper()} — ISOFIT mode comparison (EA mode, orders [3,4,5,6,7])",
        fontsize=14, y=0.995,
    )

    # ------------------------------------------------------------------
    # Top: 2D residual maps
    # ------------------------------------------------------------------
    residuals = {}
    for m in modes:
        residuals[m] = np.where(np.isfinite(image), image - mode_models[m], np.nan)

    # Shared color scale
    all_abs = np.abs(np.concatenate([residuals[m][np.isfinite(residuals[m])] for m in modes]))
    res_limit = float(np.clip(np.nanpercentile(all_abs, 99.0), 0.05, None))

    im_handle = None
    for col, m in enumerate(modes):
        ax = fig.add_subplot(top[0, col])
        im_handle = ax.imshow(
            residuals[m], origin="lower", cmap="coolwarm",
            vmin=-res_limit, vmax=res_limit, interpolation="nearest",
        )
        # Title: mode name + stats summary
        meta = mode_meta[m]
        stat_all = mode_stats[m].get("all (r<max)", {})
        med_abs = stat_all.get("median_abs_frac_pct", np.nan)
        rms_pct = stat_all.get("rms_frac_pct", np.nan)
        ax.set_title(
            f"{MODE_DISPLAY[m]}\n"
            f"t={meta['elapsed']:.1f}s  conv={meta['n_conv']}/{meta['n_iso']}  "
            f"|res|={med_abs:.2f}%  rms={rms_pct:.2f}%",
            fontsize=8, pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar
    cbar_ax = fig.add_subplot(top[0, n_modes])
    fig.colorbar(im_handle, cax=cbar_ax, label="data - model")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _arr(isos, key, default=np.nan):
        return np.array([r.get(key, default) for r in isos])

    # ------------------------------------------------------------------
    # Bottom: 1D profile panels
    # ------------------------------------------------------------------
    ax_sb = fig.add_subplot(bottom[0])
    ax_di = fig.add_subplot(bottom[1], sharex=ax_sb)
    ax_odd = fig.add_subplot(bottom[2], sharex=ax_sb)
    ax_even = fig.add_subplot(bottom[3], sharex=ax_sb)
    ax_ba = fig.add_subplot(bottom[4], sharex=ax_sb)
    ax_pa = fig.add_subplot(bottom[5], sharex=ax_sb)

    profile_axes = [ax_sb, ax_di, ax_odd, ax_even, ax_ba, ax_pa]
    legend_handles = []

    # Use baseline as reference for relative intensity difference
    baseline_sma = None
    baseline_intens = None
    if "baseline" in mode_results:
        baseline_sma = _arr(mode_results["baseline"], "sma")
        baseline_intens = _arr(mode_results["baseline"], "intens")

    for m in modes:
        isos = mode_results[m]
        sma = _arr(isos, "sma")
        intens = _arr(isos, "intens")
        eps = _arr(isos, "eps")
        pa_rad = _arr(isos, "pa")
        stop = _arr(isos, "stop_code", 0).astype(int)
        xax = sma ** 0.25
        vm = np.isfinite(xax) & (sma > 1.0)
        good = vm & (stop == 0)
        bad = vm & (stop != 0)

        col = MODE_COLORS[m]
        mrk = MODE_MARKERS[m]
        label = MODE_DISPLAY[m]

        skw_good = dict(s=18, marker=mrk, facecolors=col, edgecolors=col, alpha=0.75)
        skw_bad = dict(s=12, marker=mrk, facecolors="none", edgecolors=col, alpha=0.35)

        # 1. Surface brightness
        sb_ok = good & np.isfinite(intens) & (intens > 0)
        sb_bad = bad & np.isfinite(intens) & (intens > 0)
        y_sb = np.full_like(intens, np.nan)
        y_sb[np.isfinite(intens) & (intens > 0)] = np.log10(intens[np.isfinite(intens) & (intens > 0)])
        ax_sb.scatter(xax[sb_ok], y_sb[sb_ok], **skw_good, label=label)
        if np.any(sb_bad):
            ax_sb.scatter(xax[sb_bad], y_sb[sb_bad], **skw_bad)

        # 2. Relative intensity difference vs baseline
        if baseline_sma is not None and m != "baseline":
            # Interpolate baseline intensity at this mode's SMA values
            from scipy.interpolate import interp1d
            valid_bl = np.isfinite(baseline_sma) & np.isfinite(baseline_intens) & (baseline_intens > 0)
            if np.sum(valid_bl) > 3:
                interp_bl = interp1d(
                    baseline_sma[valid_bl], baseline_intens[valid_bl],
                    kind="linear", bounds_error=False, fill_value=np.nan,
                )
                bl_at_sma = interp_bl(sma)
                delta_i = np.where(
                    np.isfinite(bl_at_sma) & (bl_at_sma > 0) & np.isfinite(intens),
                    (intens - bl_at_sma) / bl_at_sma,
                    np.nan,
                )
                di_ok = good & np.isfinite(delta_i)
                di_bad = bad & np.isfinite(delta_i)
                ax_di.scatter(xax[di_ok], delta_i[di_ok] * 100, **skw_good)
                if np.any(di_bad):
                    ax_di.scatter(xax[di_bad], delta_i[di_bad] * 100, **skw_bad)

        # 3. Odd harmonics amplitude (A3, A5, A7)
        for order in [3, 5, 7]:
            an = _arr(isos, f"a{order}")
            bn = _arr(isos, f"b{order}")
            amplitude = np.sqrt(
                np.where(np.isfinite(an), an, 0.0) ** 2
                + np.where(np.isfinite(bn), bn, 0.0) ** 2
            )
            amplitude[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
            a_ok = good & np.isfinite(amplitude)
            a_bad = bad & np.isfinite(amplitude)
            if np.any(a_ok):
                ax_odd.scatter(xax[a_ok], amplitude[a_ok], **skw_good)
            if np.any(a_bad):
                ax_odd.scatter(xax[a_bad], amplitude[a_bad], **skw_bad)

        # 4. Even harmonics amplitude (A4, A6)
        for order in [4, 6]:
            an = _arr(isos, f"a{order}")
            bn = _arr(isos, f"b{order}")
            amplitude = np.sqrt(
                np.where(np.isfinite(an), an, 0.0) ** 2
                + np.where(np.isfinite(bn), bn, 0.0) ** 2
            )
            amplitude[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
            a_ok = good & np.isfinite(amplitude)
            a_bad = bad & np.isfinite(amplitude)
            if np.any(a_ok):
                ax_even.scatter(xax[a_ok], amplitude[a_ok], **skw_good)
            if np.any(a_bad):
                ax_even.scatter(xax[a_bad], amplitude[a_bad], **skw_bad)

        # 5. Axis ratio
        ba = 1.0 - eps
        ba_ok = good & np.isfinite(ba)
        ba_bad = bad & np.isfinite(ba)
        ax_ba.scatter(xax[ba_ok], ba[ba_ok], **skw_good)
        if np.any(ba_bad):
            ax_ba.scatter(xax[ba_bad], ba[ba_bad], **skw_bad)

        # 6. PA
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

    # ------------------------------------------------------------------
    # Axis labels and formatting
    # ------------------------------------------------------------------
    ax_sb.set_ylabel(r"$\log_{10}(I)$")
    ax_di.set_ylabel(r"$\Delta I / I_{\rm base}$ [%]")
    ax_di.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_odd.set_ylabel(r"$\sqrt{a_n^2 + b_n^2}$ (odd)")
    ax_odd.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_even.set_ylabel(r"$\sqrt{a_n^2 + b_n^2}$ (even)")
    ax_even.axhline(0.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_ba.set_ylabel("b/a")
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.set_xlabel(r"SMA$^{0.25}$ [pixel$^{0.25}$]")

    for ax in profile_axes[:-1]:
        ax.tick_params(labelbottom=False)
    for ax in profile_axes:
        ax.grid(alpha=0.2)

    # X limits
    all_xvals = []
    for m in modes:
        sma = _arr(mode_results[m], "sma")
        vm = np.isfinite(sma) & (sma > 1.0)
        if np.any(vm):
            all_xvals.append(sma[vm] ** 0.25)
    if all_xvals:
        set_x_limits_with_right_margin(ax_pa, np.concatenate(all_xvals))

    # Y limits for SB panel
    sb_vals = []
    for m in modes:
        intens = _arr(mode_results[m], "intens")
        ok = np.isfinite(intens) & (intens > 0)
        if np.any(ok):
            sb_vals.append(np.log10(intens[ok]))
    if sb_vals:
        set_axis_limits_from_finite_values(ax_sb, np.concatenate(sb_vals), margin_fraction=0.06)

    # Y limits for axis ratio
    ba_vals = []
    for m in modes:
        eps = _arr(mode_results[m], "eps")
        stop = _arr(mode_results[m], "stop_code", 0).astype(int)
        ok = np.isfinite(eps) & (stop == 0)
        if np.any(ok):
            ba_vals.append(1.0 - eps[ok])
    if ba_vals:
        set_axis_limits_from_finite_values(
            ax_ba, np.concatenate(ba_vals),
            margin_fraction=0.06, lower_clip=0.0, upper_clip=1.0,
        )

    # Y limits for PA
    pa_vals = []
    for m in modes:
        pa_rad = _arr(mode_results[m], "pa")
        stop = _arr(mode_results[m], "stop_code", 0).astype(int)
        ok = np.isfinite(pa_rad) & (stop == 0)
        if np.any(ok):
            pa_vals.append(normalize_pa_degrees(np.degrees(pa_rad[ok])))
    if pa_vals:
        all_pa = np.concatenate(pa_vals)
        lo, hi = robust_limits(all_pa, 3, 97)
        margin = max(3.0, 0.08 * (hi - lo + 1e-6))
        ax_pa.set_ylim(lo - margin, hi + margin)

    # Y limits for harmonic panels (from stop=0 points only)
    for ax, orders in [(ax_odd, [3, 5, 7]), (ax_even, [4, 6])]:
        vals = []
        for m in modes:
            isos = mode_results[m]
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

    # Y limits for delta I (clip to reasonable range)
    ax_di.set_ylim(-5, 5)

    # Legend
    ax_sb.legend(
        handles=legend_handles, loc="upper right", fontsize=8,
        frameon=True, ncol=1,
    )

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.04, top=0.965, hspace=0.04)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison figure → {filename}")


# ---------------------------------------------------------------------------
# Print statistics table
# ---------------------------------------------------------------------------

def print_statistics_table(mode_stats: dict[str, dict], mode_meta: dict[str, dict]) -> None:
    """Print a formatted statistics table to stdout."""
    modes = [m for m in MODE_ORDER if m in mode_stats]

    print(f"\n{'Mode':<35} {'Conv':>5} {'Iter':>6} "
          f"{'|res| all%':>10} {'rms all%':>10} "
          f"{'|res| mid%':>10} {'rms mid%':>10} "
          f"{'Time':>6}")
    print("-" * 100)

    for m in modes:
        meta = mode_meta[m]
        s_all = mode_stats[m].get("all (r<max)", {})
        s_mid = mode_stats[m].get("mid (0.25-0.6*max)", {})
        print(
            f"{MODE_DISPLAY[m]:<35} "
            f"{meta['n_conv']:>3}/{meta['n_iso']:<2} "
            f"{meta['mean_niter']:>5.1f} "
            f"{s_all.get('median_abs_frac_pct', np.nan):>10.3f} "
            f"{s_all.get('rms_frac_pct', np.nan):>10.3f} "
            f"{s_mid.get('median_abs_frac_pct', np.nan):>10.3f} "
            f"{s_mid.get('rms_frac_pct', np.nan):>10.3f} "
            f"{meta['elapsed']:>5.1f}s"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_galaxy(galaxy: str, output_dir: Path, band_index: int = 1) -> None:
    """Run the three-mode comparison for one galaxy."""
    data_dir = _HERE.parent.parent / "data"
    fits_path = data_dir / FITS_FILENAME[galaxy]
    if not fits_path.exists():
        print(f"ERROR: FITS file not found: {fits_path}")
        return

    gal_dir = output_dir / galaxy
    gal_dir.mkdir(parents=True, exist_ok=True)

    band_name = BAND_NAMES[galaxy][band_index]
    print(f"\n{'='*60}")
    print(f"  Galaxy : {galaxy}")
    print(f"  Band   : {band_name} (index {band_index})")
    print(f"  Output : {gal_dir}")
    print(f"{'='*60}\n")

    # Load image
    print("Loading image...")
    image, pixel_scale = load_legacysurvey_fits(fits_path, band_index)
    print(f"  Shape: {image.shape}, pixel scale: {PIXEL_SCALE[galaxy]} arcsec/px")

    # Generate mask
    print("Generating mask...")
    mask_kwargs = MASK_PARAMS.get(galaxy, {})
    center_xy = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = make_object_mask(image, center_xy=center_xy, **mask_kwargs)
    n_masked = int(np.sum(mask))
    print(f"  Masked: {n_masked}/{mask.size} ({100.0*n_masked/mask.size:.1f}%)")

    # Build configs
    configs = make_mode_configs(galaxy)
    print(f"\nRunning {len(configs)} modes...\n")

    # Run each mode
    mode_results = {}
    mode_models = {}
    mode_stats = {}
    mode_meta = {}

    for m in MODE_ORDER:
        config = configs[m]
        print(f"  [{m}] EA={config.use_eccentric_anomaly} "
              f"sim_harm={config.simultaneous_harmonics} "
              f"isofit_mode={config.isofit_mode}")

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = isoster.fit_image(image, mask, config)
        elapsed = time.perf_counter() - t0

        isos = results["isophotes"]
        n_iso = len(isos)
        n_conv = sum(1 for r in isos if r.get("stop_code", -99) == 0)
        niters = [r.get("niter", 0) for r in isos]
        mean_niter = float(np.mean(niters)) if niters else 0.0
        print(f"    → {n_iso} isophotes, {n_conv} converged, "
              f"mean_niter={mean_niter:.1f}, {elapsed:.1f}s")

        # Build 2D model
        model = build_isoster_model(image.shape, isos, use_harmonics=True)

        # Residual statistics
        stats = compute_residual_statistics(image, model, isos)

        mode_results[m] = isos
        mode_models[m] = model
        mode_stats[m] = stats
        mode_meta[m] = {
            "elapsed": elapsed,
            "n_iso": n_iso,
            "n_conv": n_conv,
            "mean_niter": mean_niter,
        }

        # Per-run detailed QA figure
        qa_per_run = str(gal_dir / f"{galaxy}_{m}_qa.png")
        stat_all = stats.get("all (r<max)", {})
        med_abs = stat_all.get("median_abs_frac_pct", np.nan)
        plot_qa_summary_extended(
            title=(
                f"{galaxy.upper()} — {MODE_DISPLAY[m]}\n"
                f"conv={n_conv}/{n_iso}  niter={mean_niter:.1f}  "
                f"|res|={med_abs:.2f}%  t={elapsed:.1f}s"
            ),
            image=image,
            isoster_model=model,
            isoster_res=isos,
            harmonic_mode="coefficients",
            relative_residual=False,
            mask=mask,
            filename=qa_per_run,
        )

    # Print statistics table
    print_statistics_table(mode_stats, mode_meta)

    # Generate QA figure
    qa_path = str(gal_dir / f"{galaxy}_isofit_mode_comparison.png")
    plot_isofit_mode_comparison(
        galaxy=galaxy,
        image=image,
        mask=mask,
        mode_results=mode_results,
        mode_models=mode_models,
        mode_stats=mode_stats,
        mode_meta=mode_meta,
        filename=qa_path,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ISOFIT modes on LegacySurvey galaxies.",
    )
    parser.add_argument(
        "--galaxy",
        choices=list(SUPPORTED_GALAXIES) + ["all"],
        default="all",
        help="Galaxy to process (default: all).",
    )
    parser.add_argument(
        "--band-index",
        type=int,
        default=1,
        help="Band index to extract from FITS cube (default: 1 = r-band).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/example_ls_highorder_harmonic/isofit_mode_comparison"),
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    galaxies = list(SUPPORTED_GALAXIES) if args.galaxy == "all" else [args.galaxy]

    for galaxy in galaxies:
        run_galaxy(galaxy, args.output_dir, args.band_index)

    print("Done.")


if __name__ == "__main__":
    main()
