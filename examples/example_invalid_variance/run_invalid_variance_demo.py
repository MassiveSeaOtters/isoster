#!/usr/bin/env python3
"""Real-galaxy demo of the invalid-variance policy and its runtime cost.

This script exercises the behaviour introduced in Tasks 1-5 of the
"gradient-error-ring-statistics" fix: a variance-map entry that is not
finite or not strictly positive is now treated as invalid rather than being
clamped to a sentinel value. The affected pixels are dropped from ring
sampling, and each ring's reported uncertainty (``grad_error``) is derived
from the variance of the exact statistic that ring reports, computed on the
exact samples that survived. Before this fix, a non-positive variance was
clamped to ``1e-30`` (near-infinite weight), which could collapse the
reported gradient error by many orders of magnitude and make an unreliable
gradient look perfectly trustworthy.

The demo fits one real DESI Legacy Survey galaxy cutout twice:

* Run A ("clean") uses the real inverse-variance map converted to variance
  with no defect. It shows there is no regression on ordinary data.
* Run B ("defect") copies the variance map and sets a contiguous block near
  the galaxy's mid-radius to exactly ``0.0`` — the signature of a masked
  star whose inverse-variance pixels were mishandled. This is the only way
  to exercise the invalid-variance path here: the real cutouts used in this
  demo contain no ``invvar == 0`` pixels on their own (verified by hand).

It also times ``fit_image`` under three configurations (no variance map,
clean variance map, defect variance map) so that a before/after comparison
against the pre-Task-1-5 library (via ``git stash``) can quantify the
runtime cost of the new checks.

Usage
-----
    uv run python examples/example_invalid_variance/run_invalid_variance_demo.py
"""

from __future__ import annotations

import glob
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.rcParams["text.usetex"] = False

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.output_paths import resolve_output_directory
from isoster.plotting import build_method_profile, plot_comparison_qa_figure
from isoster.utils import isophote_results_to_fits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GALAXY_NAME = "2MASXJ12504800+4231220"
DATA_DIR = Path(f"/Users/shuang/Dropbox/work/project/otters/sga_isoster/data/demo/{GALAXY_NAME}")

# Legacy Survey (DECaLS) photometric convention: zeropoint and pixel scale
# are always passed as two separate values, never pre-combined (see
# CLAUDE.md "Surface brightness convention"). The pixel scale is read from
# the image WCS header with this value as a fallback.
SB_ZEROPOINT = 22.5
FALLBACK_PIXEL_SCALE_ARCSEC = 0.262

N_TIMED_RUNS = 5


def _glob_one(pattern: str) -> Path:
    """Return the single file matching *pattern* inside DATA_DIR."""
    matches = glob.glob(str(DATA_DIR / pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 match for {pattern!r} in {DATA_DIR}, got {len(matches)}: {matches}"
        )
    return Path(matches[0])


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the r-band image, variance map, mask, and pixel scale.

    Returns
    -------
    image : 2D float64 array
    variance : 2D float64 array
        Per-pixel variance derived from the real inverse-variance map.
        Pixels with ``invvar <= 0`` are set to NaN, which ``fit_image``
        then treats as invalid and drops from ring sampling (this is the
        fixed behaviour under test; it differs from the pre-Task-1-5
        convention of a ``1e30`` sentinel for those same pixels).
    mask : 2D bool array
        True = bad pixel (from the external mask file).
    pixel_scale_arcsec : float
        Pixel size in arcsec, read from the image WCS header (CD1_1) with
        a DECaLS-typical fallback if the header lacks WCS keywords.
    """
    image_path = _glob_one("*-image-r.fits.fz")
    invvar_path = _glob_one("*-invvar-r.fits.fz")
    mask_path = _glob_one("*-mask.fits")

    with fits.open(image_path) as hdul:
        image_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        image = image_hdu.data.astype(np.float64)
        header = image_hdu.header
        cd1_1 = header.get("CD1_1")
        pixel_scale_arcsec = abs(cd1_1) * 3600.0 if cd1_1 is not None else FALLBACK_PIXEL_SCALE_ARCSEC

    invvar = fits.getdata(invvar_path).astype(np.float64)
    mask = fits.getdata(mask_path).astype(bool)

    # This is the corrected conversion: invalid inverse-variance pixels
    # become NaN, which fit_image now marks invalid and excludes from ring
    # sampling. The pre-fix convention (see examples/example_variance_map)
    # instead substituted a 1e30 sentinel for these same pixels.
    variance = np.where(invvar > 0, 1.0 / invvar, np.nan)

    print(f"Image shape          : {image.shape}")
    print(f"Pixel scale (arcsec)  : {pixel_scale_arcsec:.4f}")
    print(f"Non-zero invvar frac  : {np.mean(invvar > 0):.6f}")
    print(f"Non-positive invvar   : {int(np.sum(invvar <= 0))} pixels (none expected in this cutout)")
    print(f"Mask coverage (bad px): {np.mean(mask):.4f}")

    return image, variance, mask, pixel_scale_arcsec


# ---------------------------------------------------------------------------
# Defect injection
# ---------------------------------------------------------------------------
def inject_defect(variance: np.ndarray, x0: float, y0: float) -> tuple[np.ndarray, dict]:
    """Return a copy of *variance* with a contiguous mid-radius block set to 0.

    Mimics a masked star whose inverse-variance pixels were mishandled
    (recorded as exactly 0 rather than excised). Placed at 45 degrees from
    the image center, offset along both axes by 60 pixels, which for this
    galaxy's fitted mid-radius geometry (eps~0.48, pa~54 deg) intersects
    isophote rings spanning roughly sma=67-99 pixels while leaving rings
    outside that range untouched (verified against extract_isophote_data
    directly during script development).

    Returns
    -------
    variance_defect : 2D float64 array
        Copy of *variance* with the block set to 0.0.
    info : dict
        Block bounds and pixel count, for reporting.
    """
    variance_defect = variance.copy()
    offset = 60
    half_size = 15
    cx = int(round(x0 + offset))
    cy = int(round(y0 + offset))
    row_slice = slice(max(cy - half_size, 0), min(cy + half_size, variance.shape[0]))
    col_slice = slice(max(cx - half_size, 0), min(cx + half_size, variance.shape[1]))
    variance_defect[row_slice, col_slice] = 0.0

    info = {
        "row_bounds": (row_slice.start, row_slice.stop),
        "col_bounds": (col_slice.start, col_slice.stop),
        "n_pixels": int(np.sum(variance_defect == 0.0)),
    }
    return variance_defect, info


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
def build_config(image_shape: tuple[int, int], debug: bool = False) -> IsosterConfig:
    """Build an IsosterConfig with center auto-detected from image shape.

    debug=True is required to populate the per-isophote gradient
    diagnostics (grad, grad_error, grad_r_error, ndata) that this demo
    reports; fit_image only attaches those fields when debug is set.
    """
    ny, nx = image_shape
    half_diag = 0.5 * np.sqrt(nx**2 + ny**2)
    maxsma = half_diag * 0.95

    return IsosterConfig(
        x0=nx / 2.0,
        y0=ny / 2.0,
        sma0=10.0,
        maxsma=maxsma,
        eps=0.2,
        pa=0.0,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def stop_code_histogram(isophotes: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for iso in isophotes:
        code = int(iso.get("stop_code", 0))
        counts[code] = counts.get(code, 0) + 1
    return counts


def grad_r_error_stats(isophotes: list[dict]) -> tuple[float, float, int]:
    """Median and 90th percentile of grad_r_error, and count of usable values."""
    values = np.array([iso.get("grad_r_error", np.nan) for iso in isophotes], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan, 0
    return float(np.median(finite)), float(np.percentile(finite, 90)), int(finite.size)


def print_run_summary(label: str, isophotes: list[dict]) -> None:
    n_iso = len(isophotes)
    counts = stop_code_histogram(isophotes)
    median_gre, p90_gre, n_usable = grad_r_error_stats(isophotes)
    print(f"\n--- {label} ---")
    print(f"  Isophotes           : {n_iso}")
    print(f"  Stop-code histogram : {dict(sorted(counts.items()))}")
    print(f"  grad_r_error median : {median_gre:.4f} ({n_usable} usable values)")
    print(f"  grad_r_error P90    : {p90_gre:.4f}")


def compare_runs(isophotes_a: list[dict], isophotes_b: list[dict]) -> list[dict]:
    """Match isophotes by nearest sma and report grad_error side by side.

    Returns the list of per-isophote comparison rows and prints a table
    flagging any isophote whose stop code differs between the two runs.
    """
    rows = []
    print(
        f"\n{'sma':>8s}  {'stop_A':>7s}  {'stop_B':>7s}  {'grad_err_A':>12s}  {'grad_err_B':>12s}  {'ratio':>8s}  flag"
    )
    print("-" * 78)
    # Match each isophote in A to the nearest-sma isophote in B.
    sma_b = np.array([iso["sma"] for iso in isophotes_b])
    for iso_a in isophotes_a:
        j = int(np.argmin(np.abs(sma_b - iso_a["sma"])))
        iso_b = isophotes_b[j]
        stop_a = int(iso_a.get("stop_code", 0))
        stop_b = int(iso_b.get("stop_code", 0))
        gerr_a = iso_a.get("grad_error", np.nan)
        gerr_b = iso_b.get("grad_error", np.nan)
        gerr_a = np.nan if gerr_a is None else float(gerr_a)
        gerr_b = np.nan if gerr_b is None else float(gerr_b)
        ratio = gerr_b / gerr_a if (np.isfinite(gerr_a) and gerr_a != 0 and np.isfinite(gerr_b)) else np.nan
        flag = "STOP CODE DIFFERS" if stop_a != stop_b else ""
        rows.append(
            {
                "sma": iso_a["sma"],
                "stop_a": stop_a,
                "stop_b": stop_b,
                "grad_error_a": gerr_a,
                "grad_error_b": gerr_b,
                "ratio": ratio,
                "flag": flag,
            }
        )
        print(f"{iso_a['sma']:8.2f}  {stop_a:7d}  {stop_b:7d}  {gerr_a:12.4e}  {gerr_b:12.4e}  {ratio:8.3f}  {flag}")
    return rows


def time_fit(image, mask, config, variance_map, label: str) -> dict:
    """Warm up once (excludes numba JIT), then time N_TIMED_RUNS runs.

    Returns a dict with the median wall time and derived ms/fit and
    ms/isophote figures.
    """
    fit_image(image, mask=mask, config=config, variance_map=variance_map)  # warm-up

    times = []
    n_isophotes = None
    for _ in range(N_TIMED_RUNS):
        t0 = time.perf_counter()
        result = fit_image(image, mask=mask, config=config, variance_map=variance_map)
        times.append(time.perf_counter() - t0)
        n_isophotes = len(result["isophotes"])

    median_s = float(np.median(times))
    ms_per_fit = median_s * 1000.0
    ms_per_isophote = ms_per_fit / n_isophotes if n_isophotes else np.nan
    print(
        f"  {label:<24s}: {ms_per_fit:8.2f} ms/fit   {ms_per_isophote:7.4f} ms/isophote   "
        f"({n_isophotes} isophotes, median of {N_TIMED_RUNS} runs)"
    )
    return {
        "label": label,
        "median_ms_per_fit": ms_per_fit,
        "ms_per_isophote": ms_per_isophote,
        "n_isophotes": n_isophotes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    image, variance_clean, mask, pixel_scale_arcsec = load_data()
    config_report = build_config(image.shape, debug=True)
    config_timing = build_config(image.shape, debug=False)
    output_dir = resolve_output_directory("example_invalid_variance")

    x0, y0 = config_report.x0, config_report.y0
    variance_defect, defect_info = inject_defect(variance_clean, x0, y0)
    print(
        f"\nInjected defect: rows {defect_info['row_bounds']}, "
        f"cols {defect_info['col_bounds']}, {defect_info['n_pixels']} pixels set to 0.0"
    )

    # --- Run A: clean real data ---
    with warnings.catch_warnings(record=True) as warnings_a:
        warnings.simplefilter("always")
        results_a = fit_image(image, mask=mask, config=config_report, variance_map=variance_clean)
    for w in warnings_a:
        print(f"  [Run A warning] {w.category.__name__}: {w.message}")
    isophotes_a = results_a["isophotes"]
    print_run_summary("Run A: clean variance map", isophotes_a)

    # --- Run B: injected defect ---
    with warnings.catch_warnings(record=True) as warnings_b:
        warnings.simplefilter("always")
        results_b = fit_image(image, mask=mask, config=config_report, variance_map=variance_defect)
    n_samples_dropped = 0
    for w in warnings_b:
        print(f"  [Run B warning] {w.category.__name__}: {w.message}")
        if "non-positive values" in str(w.message):
            n_samples_dropped = defect_info["n_pixels"]
    isophotes_b = results_b["isophotes"]
    print_run_summary("Run B: injected defect (0.0 block)", isophotes_b)
    print(f"  Invalid pixels reported by fit_image: {n_samples_dropped}")

    # --- Comparison ---
    print("\n--- Per-isophote comparison: Run A vs Run B ---")
    comparison_rows = compare_runs(isophotes_a, isophotes_b)
    n_stop_code_diffs = sum(1 for row in comparison_rows if row["flag"])
    print(f"\nIsophotes with a differing stop code: {n_stop_code_diffs}")

    # --- Runtime benchmark ---
    print("\n--- Runtime benchmark (config_timing, debug=False) ---")
    timing_results = [
        time_fit(image, mask, config_timing, None, "OLS (no variance map)"),
        time_fit(image, mask, config_timing, variance_clean, "WLS (clean variance)"),
        time_fit(image, mask, config_timing, variance_defect, "WLS (defect variance)"),
    ]

    # --- Save isophote results as FITS ---
    a_fits_path = output_dir / f"{GALAXY_NAME}_run_a_clean.fits"
    b_fits_path = output_dir / f"{GALAXY_NAME}_run_b_defect.fits"
    isophote_results_to_fits(results_a, str(a_fits_path))
    isophote_results_to_fits(results_b, str(b_fits_path))
    print(f"\nSaved Run A isophotes -> {a_fits_path}")
    print(f"Saved Run B isophotes -> {b_fits_path}")

    # --- QA figure ---
    model_a = build_isoster_model(image.shape, isophotes_a)
    model_b = build_isoster_model(image.shape, isophotes_b)
    profile_a = build_method_profile(isophotes_a)
    profile_b = build_method_profile(isophotes_b)
    if profile_a is None or profile_b is None:
        raise RuntimeError("One of the runs produced zero isophotes; cannot build the QA figure.")

    profiles = {"Clean": profile_a, "Defect": profile_b}
    models = {"Clean": model_a, "Defect": model_b}
    method_styles = {
        "Clean": {"color": "#1f77b4", "label": "Clean variance map"},
        "Defect": {"color": "#d62728", "label": "Injected 0.0 defect"},
    }

    qa_path = output_dir / f"{GALAXY_NAME}_invalid_variance_qa.png"
    plot_comparison_qa_figure(
        image,
        profiles,
        title=f"{GALAXY_NAME} r-band -- clean vs injected invalid-variance defect",
        output_path=qa_path,
        models=models,
        mask=mask,
        method_styles=method_styles,
        sb_zeropoint=SB_ZEROPOINT,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    print(f"\nQA figure saved to: {qa_path}")

    # --- Text summary log ---
    log_path = output_dir / f"{GALAXY_NAME}_summary.txt"
    with open(log_path, "w") as f:
        f.write(f"Galaxy: {GALAXY_NAME}\n")
        f.write(f"Image shape: {image.shape}\n")
        f.write(
            f"Injected defect: rows {defect_info['row_bounds']}, cols {defect_info['col_bounds']}, "
            f"{defect_info['n_pixels']} pixels set to 0.0\n\n"
        )
        f.write(
            f"Run A (clean): {len(isophotes_a)} isophotes, "
            f"stop codes {dict(sorted(stop_code_histogram(isophotes_a).items()))}\n"
        )
        f.write(
            f"Run B (defect): {len(isophotes_b)} isophotes, "
            f"stop codes {dict(sorted(stop_code_histogram(isophotes_b).items()))}\n"
        )
        f.write(f"Isophotes with a differing stop code: {n_stop_code_diffs}\n\n")
        f.write("Runtime benchmark:\n")
        for t in timing_results:
            f.write(
                f"  {t['label']:<24s}: {t['median_ms_per_fit']:8.2f} ms/fit   "
                f"{t['ms_per_isophote']:7.4f} ms/isophote   ({t['n_isophotes']} isophotes)\n"
            )
    print(f"Summary log saved to: {log_path}")

    print(f"\nOutputs in: {output_dir}")


if __name__ == "__main__":
    main()
