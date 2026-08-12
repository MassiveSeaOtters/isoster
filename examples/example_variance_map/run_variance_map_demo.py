#!/usr/bin/env python3
"""Variance-map demo: OLS vs WLS isophote fitting on a DESI Legacy Survey image.

Loads a real DESI Legacy Survey cutout (2MASXJ23065343+0031547), runs isoster
twice — once without a variance map (OLS) and once with (WLS) — then produces
a comparison QA figure with 2D models and prints per-isophote error-bar ratios.

Usage
-----
    uv run python examples/example_variance_map/run_variance_map_demo.py
"""

from __future__ import annotations

import glob
import time
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
GALAXY_NAME = "2MASXJ23065343+0031547"
DATA_DIR = Path(
    "/Users/shuang/Dropbox/work/project/otters/sga_isoster/data/demo/"
    f"{GALAXY_NAME}"
)


def _glob_one(pattern: str) -> Path:
    """Return the single file matching *pattern* inside DATA_DIR."""
    matches = glob.glob(str(DATA_DIR / pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 match for {pattern!r} in {DATA_DIR}, "
            f"got {len(matches)}: {matches}"
        )
    return Path(matches[0])


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load r-band image, variance map, and mask from DESI Legacy Survey cutout.

    Returns
    -------
    image : 2D float64 array
    variance : 2D float64 array
        Per-pixel variance derived from the inverse-variance map.
        Pixels with invvar == 0 carry no usable noise information and are set
        to NaN, which isoster marks invalid and drops during ring sampling.
    mask : 2D bool array
        True = bad pixel (from the external mask file).
    """
    image_path = _glob_one("*-image-r.fits.fz")
    invvar_path = _glob_one("*-invvar-r.fits.fz")
    mask_path = _glob_one("*-mask.fits")

    image = fits.getdata(image_path).astype(np.float64)
    invvar = fits.getdata(invvar_path).astype(np.float64)
    mask = fits.getdata(mask_path).astype(bool)

    # Convert inverse-variance to variance. Zero-invvar pixels become NaN
    # rather than a large finite value: a large finite variance is still
    # "valid" to isoster, so it stays in the ring and inflates the unweighted
    # gradient error, whereas NaN is marked invalid and the sample is dropped.
    variance = np.where(invvar > 0, 1.0 / invvar, np.nan)

    print(f"Image shape : {image.shape}")
    print(f"Non-zero invvar fraction : {np.mean(invvar > 0):.4f}")
    print(f"Mask coverage (bad px)   : {np.mean(mask):.4f}")

    return image, variance, mask


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------
def build_config(image_shape: tuple[int, int]) -> IsosterConfig:
    """Build an IsosterConfig with center auto-detected from image shape."""
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
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    image, variance, mask = load_data()
    config = build_config(image.shape)
    output_dir = resolve_output_directory("example_variance_map")

    # --- OLS run (no variance map) ---
    print("\n--- Running isoster WITHOUT variance map (OLS) ---")
    t0 = time.perf_counter()
    results_ols = fit_image(image, mask=mask, config=config)
    wall_ols = time.perf_counter() - t0
    n_ols = len(results_ols["isophotes"])
    print(f"OLS: {n_ols} isophotes in {wall_ols:.2f}s")

    # --- WLS run (with variance map) ---
    print("\n--- Running isoster WITH variance map (WLS) ---")
    t0 = time.perf_counter()
    results_wls = fit_image(image, mask=mask, config=config, variance_map=variance)
    wall_wls = time.perf_counter() - t0
    n_wls = len(results_wls["isophotes"])
    print(f"WLS: {n_wls} isophotes in {wall_wls:.2f}s")

    # --- Save isophote results as FITS ---
    ols_fits_path = output_dir / f"{GALAXY_NAME}_ols.fits"
    wls_fits_path = output_dir / f"{GALAXY_NAME}_wls.fits"
    isophote_results_to_fits(results_ols, str(ols_fits_path))
    isophote_results_to_fits(results_wls, str(wls_fits_path))
    print(f"\nSaved OLS isophotes → {ols_fits_path}")
    print(f"Saved WLS isophotes → {wls_fits_path}")

    # --- Build 2D models ---
    print("\nBuilding 2D reconstructed models ...")
    model_ols = build_isoster_model(image.shape, results_ols["isophotes"])
    model_wls = build_isoster_model(image.shape, results_wls["isophotes"])

    # --- Build profiles ---
    profile_ols = build_method_profile(results_ols["isophotes"])
    profile_wls = build_method_profile(results_wls["isophotes"])

    if profile_ols is None or profile_wls is None:
        raise RuntimeError(
            "One of the runs produced zero isophotes — cannot compare."
        )

    profile_ols["runtime_seconds"] = wall_ols
    profile_wls["runtime_seconds"] = wall_wls

    profiles: dict[str, dict[str, np.ndarray]] = {
        "OLS": profile_ols,
        "WLS": profile_wls,
    }
    models: dict[str, np.ndarray] = {
        "OLS": model_ols,
        "WLS": model_wls,
    }

    # --- QA figure ---
    qa_path = output_dir / f"{GALAXY_NAME}_ols_vs_wls_qa.png"
    method_styles = {
        "OLS": {"color": "#1f77b4", "label": "OLS (no variance)"},
        "WLS": {"color": "#d62728", "label": "WLS (variance map)"},
    }

    plot_comparison_qa_figure(
        image,
        profiles,
        title=f"{GALAXY_NAME} r-band — OLS vs WLS",
        output_path=qa_path,
        models=models,
        mask=mask,
        method_styles=method_styles,
    )
    print(f"\nQA figure saved to: {qa_path}")

    # --- Text log ---
    log_path = output_dir / f"{GALAXY_NAME}_summary.txt"
    with open(log_path, "w") as f:
        f.write(f"Galaxy: {GALAXY_NAME}\n")
        f.write(f"Image shape: {image.shape}\n")
        f.write(f"OLS: {n_ols} isophotes, {wall_ols:.3f}s\n")
        f.write(f"WLS: {n_wls} isophotes, {wall_wls:.3f}s\n\n")

        ols_err = profile_ols.get("intens_err")
        wls_err = profile_wls.get("intens_err")
        if ols_err is not None and wls_err is not None:
            ols_sma = profile_ols["sma"]
            wls_sma = profile_wls["sma"]
            sma_median = np.median(ols_sma)
            outer_mask_ols = ols_sma >= sma_median

            f.write(f"{'SMA':>8s}  {'OLS_err':>10s}  {'WLS_err':>10s}  {'ratio':>8s}\n")
            f.write("-" * 42 + "\n")

            for sma_val, err_val in zip(
                ols_sma[outer_mask_ols], ols_err[outer_mask_ols], strict=False
            ):
                idx_wls = np.argmin(np.abs(wls_sma - sma_val))
                if np.abs(wls_sma[idx_wls] - sma_val) / max(sma_val, 1.0) > 0.05:
                    continue
                wls_err_val = wls_err[idx_wls]
                ratio = wls_err_val / err_val if err_val > 0 else np.nan
                f.write(f"{sma_val:8.2f}  {err_val:10.4e}  {wls_err_val:10.4e}  {ratio:8.3f}\n")
    print(f"Summary log saved to: {log_path}")

    # --- Per-isophote error-bar ratios for outer isophotes ---
    print("\n--- Error-bar ratios (WLS / OLS) for outer isophotes ---")
    ols_sma = profile_ols["sma"]
    wls_sma = profile_wls["sma"]
    ols_err = profile_ols.get("intens_err")
    wls_err = profile_wls.get("intens_err")

    if ols_err is not None and wls_err is not None:
        sma_median = np.median(ols_sma)
        outer_mask_ols = ols_sma >= sma_median

        print(f"{'SMA':>8s}  {'OLS_err':>10s}  {'WLS_err':>10s}  {'ratio':>8s}")
        print("-" * 42)

        for sma_val, err_val in zip(
            ols_sma[outer_mask_ols], ols_err[outer_mask_ols], strict=False
        ):
            idx_wls = np.argmin(np.abs(wls_sma - sma_val))
            if np.abs(wls_sma[idx_wls] - sma_val) / max(sma_val, 1.0) > 0.05:
                continue
            wls_err_val = wls_err[idx_wls]
            ratio = wls_err_val / err_val if err_val > 0 else np.nan
            print(f"{sma_val:8.2f}  {err_val:10.4e}  {wls_err_val:10.4e}  {ratio:8.3f}")
    else:
        print("Error bars not available in one or both profiles.")

    # --- Summary ---
    print(f"\nOLS wall time: {wall_ols:.2f}s  ({n_ols} isophotes)")
    print(f"WLS wall time: {wall_wls:.2f}s  ({n_wls} isophotes)")
    print(f"Outputs in:    {output_dir}")


if __name__ == "__main__":
    main()
