#!/usr/bin/env python3
"""Systematic WLS testing: OLS vs WLS across fitting constraint modes.

Runs isoster OLS and WLS fits across multiple configuration presets
(free geometry, fixed center, fixed PA, fixed geometry, EA sampling,
isofit), optionally comparing against photutils and AutoProf.

Outputs per-config QA comparison figures, FITS tables, and summary CSVs.

Usage
-----
    # Smoke test — single config
    uv run python examples/example_wls_systematic/run_wls_systematic.py \
        --config default

    # Full run — all configs
    uv run python examples/example_wls_systematic/run_wls_systematic.py

    # Skip external methods
    uv run python examples/example_wls_systematic/run_wls_systematic.py \
        --no-autoprof --no-photutils
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

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
# Galaxy registry
# ---------------------------------------------------------------------------

GALAXY_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "2MASXJ23065343+0031547",
        "data_dir": Path(
            "/Users/shuang/Dropbox/work/project/otters/sga_isoster/data/demo/"
            "2MASXJ23065343+0031547"
        ),
        "image_glob": "*-image-r.fits.fz",
        "invvar_glob": "*-invvar-r.fits.fz",
        "mask_glob": "*-mask.fits",
        "eps": 0.2,
        "pa": 0.0,
        "sma0": 10.0,
        "x0": None,  # auto-center from image shape
        "y0": None,
        "maxsma_fraction": 0.95,
        # AutoProf metadata
        "pixel_scale": 0.262,  # DESI Legacy Survey r-band
        "zeropoint": 22.5,
        "background": None,  # estimate from corners
        "background_noise": None,
    },
]

# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------

# Each entry: label -> dict of IsosterConfig overrides.
# Photutils and AutoProf eligibility derived from the overrides.
CONFIG_MATRIX: dict[str, dict[str, Any]] = {
    "default": {},
    "fix_center": {"fix_center": True},
    "fix_center_pa": {"fix_center": True, "fix_pa": True},
    "fix_geometry": {"fix_center": True, "fix_pa": True, "fix_eps": True},
    "ea_sampling": {"use_eccentric_anomaly": True},
    "isofit": {"simultaneous_harmonics": True},
}

# Configs where AutoProf can run (free geometry only).
AUTOPROF_ELIGIBLE = {"default", "ea_sampling", "isofit"}

# ---------------------------------------------------------------------------
# Method styles for QA figures
# ---------------------------------------------------------------------------

METHOD_STYLES: dict[str, dict[str, str]] = {
    "isoster OLS": {"color": "#1f77b4", "marker": "o", "label": "isoster OLS"},
    "isoster WLS": {"color": "#d62728", "marker": "o", "label": "isoster WLS"},
    "photutils": {"color": "#ff7f0e", "marker": "s", "label": "photutils"},
    "AutoProf": {"color": "#2ca02c", "marker": "^", "label": "AutoProf"},
}

# ---------------------------------------------------------------------------
# Summary CSV columns
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "galaxy", "config", "method", "n_isophotes", "wall_seconds",
    "convergence_rate", "median_intens_err", "wls_ols_err_ratio",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _glob_one(data_dir: Path, pattern: str) -> Path:
    """Return the single file matching *pattern* inside data_dir."""
    matches = glob.glob(str(data_dir / pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 match for {pattern!r} in {data_dir}, "
            f"got {len(matches)}: {matches}"
        )
    return Path(matches[0])


def get_galaxy(name: str) -> dict[str, Any]:
    """Look up a galaxy entry by name (case-insensitive)."""
    key = name.lower().replace("-", "").replace("_", "").replace("+", "")
    for entry in GALAXY_REGISTRY:
        entry_key = entry["name"].lower().replace("-", "").replace("_", "").replace("+", "")
        if entry_key == key:
            return entry
    available = ", ".join(e["name"] for e in GALAXY_REGISTRY)
    raise ValueError(f"Unknown galaxy '{name}'. Available: {available}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def estimate_background_noise(
    image: np.ndarray, mask: np.ndarray | None = None, box_size: int = 50
) -> tuple[float, float]:
    """Estimate background level and noise from four corner patches.

    Parameters
    ----------
    image : 2D array
    mask : 2D bool array, optional
        True = bad pixel; excluded from statistics.
    box_size : int
        Side length of each corner box.

    Returns
    -------
    background : float
        Median pixel value in corners.
    noise : float
        Standard deviation in corners.
    """
    h, w = image.shape
    bs = min(box_size, h // 4, w // 4)
    slices = [
        (slice(None, bs), slice(None, bs)),
        (slice(None, bs), slice(-bs, None)),
        (slice(-bs, None), slice(None, bs)),
        (slice(-bs, None), slice(-bs, None)),
    ]
    pixels = []
    for sy, sx in slices:
        patch = image[sy, sx].ravel()
        if mask is not None:
            patch_mask = mask[sy, sx].ravel()
            patch = patch[~patch_mask]
        pixels.append(patch)
    corners = np.concatenate(pixels)
    corners = corners[np.isfinite(corners)]
    if corners.size == 0:
        return 0.0, 1.0
    return float(np.median(corners)), float(np.std(corners))


def load_galaxy_data(
    galaxy: dict[str, Any],
) -> dict[str, Any]:
    """Load image, variance map, mask, and geometry for a galaxy.

    Returns
    -------
    dict with keys: image, variance, mask, geometry (dict with x0, y0, eps,
    pa, sma0, maxsma), galaxy (the registry entry).
    """
    data_dir = galaxy["data_dir"]
    image_path = _glob_one(data_dir, galaxy["image_glob"])
    invvar_path = _glob_one(data_dir, galaxy["invvar_glob"])
    mask_path = _glob_one(data_dir, galaxy["mask_glob"])

    image = fits.getdata(image_path).astype(np.float64)
    invvar = fits.getdata(invvar_path).astype(np.float64)
    mask = fits.getdata(mask_path).astype(bool)

    # Convert inverse-variance to variance. Zero-invvar pixels become NaN
    # rather than a large finite value: a large finite variance is still
    # "valid" to isoster, so it stays in the ring and inflates the unweighted
    # gradient error, whereas NaN is marked invalid and the sample is dropped.
    variance = np.where(invvar > 0, 1.0 / invvar, np.nan)

    ny, nx = image.shape
    half_diag = 0.5 * np.sqrt(nx**2 + ny**2)
    maxsma = half_diag * galaxy.get("maxsma_fraction", 0.95)

    x0 = galaxy["x0"] if galaxy["x0"] is not None else nx / 2.0
    y0 = galaxy["y0"] if galaxy["y0"] is not None else ny / 2.0

    geometry = {
        "x0": x0,
        "y0": y0,
        "eps": galaxy["eps"],
        "pa": galaxy["pa"],
        "sma0": galaxy["sma0"],
        "maxsma": maxsma,
    }

    print(f"  Image shape : {image.shape}")
    print(f"  Non-zero invvar fraction : {np.mean(invvar > 0):.4f}")
    print(f"  Mask coverage (bad px)   : {np.mean(mask):.4f}")

    return {
        "image": image,
        "variance": variance,
        "mask": mask,
        "geometry": geometry,
        "galaxy": galaxy,
    }


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def run_isoster_case(
    image: np.ndarray,
    mask: np.ndarray,
    variance: np.ndarray | None,
    geometry: dict[str, Any],
    config_overrides: dict[str, Any],
    use_wls: bool,
) -> dict[str, Any]:
    """Run a single isoster fit (OLS or WLS).

    Returns dict with keys: method, isophotes, wall_seconds, n_isophotes,
    stop_code_counts.
    """
    config_kwargs = dict(geometry)
    config_kwargs.update(config_overrides)
    config = IsosterConfig(**config_kwargs)

    variance_map = variance if use_wls else None
    label = "isoster WLS" if use_wls else "isoster OLS"

    t0 = time.perf_counter()
    result = fit_image(image, mask=mask, config=config, variance_map=variance_map)
    wall = time.perf_counter() - t0

    isophotes = result["isophotes"]
    code_counts = dict(Counter(iso["stop_code"] for iso in isophotes))

    print(f"    {label}: {len(isophotes)} isophotes in {wall:.2f}s")

    return {
        "method": label,
        "isophotes": isophotes,
        "config": config,
        "wall_seconds": round(wall, 4),
        "n_isophotes": len(isophotes),
        "stop_code_counts": {str(k): v for k, v in sorted(code_counts.items())},
    }


def run_photutils_case(
    image: np.ndarray,
    geometry: dict[str, Any],
    config_overrides: dict[str, Any],
) -> dict[str, Any] | None:
    """Run photutils Ellipse.fit_image with fix_* passthrough.

    Returns dict with keys: method, isophotes, wall_seconds, n_isophotes,
    stop_code_counts. Returns None if photutils is unavailable.
    """
    try:
        from photutils.isophote import Ellipse, EllipseGeometry
    except ImportError:
        print("    photutils not available — skipping")
        return None

    geo = EllipseGeometry(
        x0=geometry["x0"],
        y0=geometry["y0"],
        sma=geometry["sma0"],
        eps=geometry["eps"],
        pa=geometry["pa"],
    )
    ellipse = Ellipse(image, geo)

    # Build fit_image kwargs — pass fix_* flags from config_overrides
    fit_kwargs: dict[str, Any] = {
        "step": 0.1,
        "minsma": 1.0,
        "maxsma": geometry["maxsma"],
        "maxgerr": 0.5,
        "nclip": 0,
        "sclip": 3.0,
        "integrmode": "bilinear",
    }
    for key in ("fix_center", "fix_pa", "fix_eps"):
        if config_overrides.get(key, False):
            fit_kwargs[key] = True

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isolist = ellipse.fit_image(**fit_kwargs)
    wall = time.perf_counter() - t0

    isophotes = _convert_photutils_isolist(isolist)
    code_counts = dict(Counter(iso["stop_code"] for iso in isophotes))

    print(f"    photutils: {len(isophotes)} isophotes in {wall:.2f}s")

    return {
        "method": "photutils",
        "isophotes": isophotes,
        "wall_seconds": round(wall, 4),
        "n_isophotes": len(isophotes),
        "stop_code_counts": {str(k): v for k, v in sorted(code_counts.items())},
    }


def _convert_photutils_isolist(isolist) -> list[dict[str, Any]]:
    """Convert photutils isolist to serializable list of dicts."""
    attribute_map = {
        "sma": "sma", "intens": "intens", "intens_err": "int_err",
        "eps": "eps", "ellip_err": "ellip_err",
        "pa": "pa", "pa_err": "pa_err",
        "x0": "x0", "x0_err": "x0_err", "y0": "y0", "y0_err": "y0_err",
        "grad": "grad", "grad_error": "grad_error",
        "rms": "rms", "pix_stddev": "pix_stddev",
        "stop_code": "stop_code", "ndata": "ndata", "nflag": "nflag",
        "niter": "niter",
        "a3": "a3", "b3": "b3", "a4": "a4", "b4": "b4",
        "a3_err": "a3_err", "b3_err": "b3_err",
        "a4_err": "a4_err", "b4_err": "b4_err",
    }
    isophotes: list[dict[str, Any]] = []
    for iso in isolist:
        row: dict[str, Any] = {}
        for output_key, source_key in attribute_map.items():
            value = getattr(iso, source_key, np.nan)
            row[output_key] = np.nan if value is None else value
        isophotes.append(row)
    return isophotes


def run_autoprof_case(
    image: np.ndarray,
    galaxy: dict[str, Any],
    geometry: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Run AutoProf via subprocess adapter.

    Returns dict with keys: method, profile (autoprof dict with arrays),
    wall_seconds, n_isophotes. Returns None on failure or unavailability.
    """
    try:
        from benchmarks.utils.autoprof_adapter import (
            check_autoprof_available,
            run_autoprof_fit,
        )
    except ImportError:
        print("    AutoProf adapter not importable — skipping")
        return None

    if not check_autoprof_available():
        print("    AutoProf not available — skipping")
        return None

    # Estimate background if not provided in registry
    bg = galaxy.get("background")
    bg_noise = galaxy.get("background_noise")
    if bg is None or bg_noise is None:
        bg_est, noise_est = estimate_background_noise(image)
        bg = bg if bg is not None else bg_est
        bg_noise = bg_noise if bg_noise is not None else noise_est

    # Write 2D FITS for AutoProf
    autoprof_workdir = output_dir / "autoprof_workdir"
    autoprof_workdir.mkdir(parents=True, exist_ok=True)
    fits_2d_path = autoprof_workdir / f"{galaxy['name']}_2d.fits"
    from astropy.io import fits as afits
    hdu = afits.PrimaryHDU(data=image.astype(np.float32))
    hdu.writeto(fits_2d_path, overwrite=True)

    result = run_autoprof_fit(
        image_path=fits_2d_path,
        output_dir=autoprof_workdir,
        galaxy_name=galaxy["name"],
        pixel_scale=galaxy["pixel_scale"],
        zeropoint=galaxy["zeropoint"],
        center=(geometry["x0"], geometry["y0"]),
        eps=geometry["eps"],
        pa_rad_math=geometry["pa"],
        background=bg,
        background_noise=bg_noise,
        run_ellipse_model=True,
    )

    if result is None:
        print("    AutoProf returned None")
        return None

    profile = result.get("profile", result)
    n_iso = profile.get("n_isophotes", 0)
    wall = profile.get("runtime_s", 0.0)

    print(f"    AutoProf: {n_iso} isophotes in {wall:.2f}s")

    return {
        "method": "AutoProf",
        "profile": profile,
        "wall_seconds": round(wall, 4),
        "n_isophotes": n_iso,
    }


# ---------------------------------------------------------------------------
# 2D model builders for external methods
# ---------------------------------------------------------------------------


def _build_photutils_model(
    image_shape: tuple[int, int],
    isophotes: list[dict[str, Any]],
) -> np.ndarray | None:
    """Build a 2D model from photutils isophote dicts using build_ellipse_model."""
    try:
        from photutils.isophote import build_ellipse_model
    except ImportError:
        return None

    required = ["sma", "intens", "eps", "pa", "x0", "y0", "grad"]

    # Filter to valid rows
    valid_rows = [
        iso for iso in isophotes
        if iso.get("sma", 0) > 0
        and all(np.isfinite(iso.get(k, np.nan)) for k in required)
    ]
    valid_rows.sort(key=lambda r: r["sma"])

    # Deduplicate by sma
    seen = set()
    unique = []
    for row in valid_rows:
        if row["sma"] not in seen:
            seen.add(row["sma"])
            unique.append(row)

    if len(unique) < 6:
        return None

    harmonic_keys = ["a3", "b3", "a4", "b4"]
    columns = {}
    for key in required + harmonic_keys:
        columns[key] = np.array([r.get(key, 0.0) for r in unique], dtype=float)

    class _SmaNode:
        def __init__(self, sma_value):
            self.sma = float(sma_value)

    class _IsolistAdapter:
        def __init__(self, cols):
            for k, v in cols.items():
                setattr(self, k, v)
            self._nodes = [_SmaNode(s) for s in cols["sma"]]

        def __len__(self):
            return len(self._nodes)

        def __getitem__(self, index):
            return self._nodes[index]

    try:
        adapter = _IsolistAdapter(columns)
        return build_ellipse_model(
            image_shape, adapter, fill=0.0,
            high_harmonics=True, sma_interval=0.1,
        )
    except Exception as exc:
        print(f"    photutils model build failed: {exc}")
        return None


def _load_autoprof_model(
    ap_result: dict[str, Any],
) -> np.ndarray | None:
    """Load AutoProf's native 2D model from its FITS output."""
    profile = ap_result.get("profile", {})
    model_path = profile.get("model_fits_path")
    if not model_path or not Path(model_path).exists():
        return None

    try:
        with fits.open(model_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    return np.asarray(hdu.data, dtype=np.float64)
    except Exception as exc:
        print(f"    AutoProf model load failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Orchestration: single config
# ---------------------------------------------------------------------------


def run_single_config(
    galaxy_data: dict[str, Any],
    galaxy: dict[str, Any],
    config_label: str,
    overrides: dict[str, Any],
    output_dir: Path,
    run_photutils: bool = True,
    run_autoprof_flag: bool = True,
) -> list[dict[str, Any]]:
    """Run one (galaxy, config) combination across all applicable methods.

    Returns a list of summary-row dicts (one per method).
    """
    image = galaxy_data["image"]
    variance = galaxy_data["variance"]
    mask = galaxy_data["mask"]
    geometry = galaxy_data["geometry"]

    config_dir = output_dir / config_label
    config_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Config: {config_label}")

    # --- Isoster OLS ---
    ols_result = run_isoster_case(
        image, mask, variance, geometry, overrides, use_wls=False
    )

    # --- Isoster WLS ---
    wls_result = run_isoster_case(
        image, mask, variance, geometry, overrides, use_wls=True
    )

    # --- Photutils ---
    phot_result = None
    if run_photutils:
        phot_result = run_photutils_case(image, geometry, overrides)

    # --- AutoProf ---
    ap_result = None
    if run_autoprof_flag and config_label in AUTOPROF_ELIGIBLE:
        ap_result = run_autoprof_case(image, galaxy, geometry, config_dir)

    # --- Build profiles and models ---
    profiles: dict[str, dict[str, np.ndarray]] = {}
    models: dict[str, np.ndarray] = {}

    for label, result in [
        ("isoster OLS", ols_result),
        ("isoster WLS", wls_result),
        ("photutils", phot_result),
    ]:
        if result is None:
            continue
        profile = build_method_profile(result["isophotes"])
        if profile is not None:
            profile["runtime_seconds"] = result["wall_seconds"]
            profiles[label] = profile

        # Save FITS table
        fits_name = label.replace(" ", "_").lower() + ".fits"
        isophote_results_to_fits(
            {"isophotes": result["isophotes"], "config": result.get("config")},
            str(config_dir / fits_name),
        )

        # Build 2D model — isoster uses its native builder
        if label.startswith("isoster") and result["isophotes"]:
            try:
                model = build_isoster_model(image.shape, result["isophotes"])
                models[label] = model
            except Exception as exc:
                print(f"    Model build failed for {label}: {exc}")

        # Build 2D model — photutils uses its native build_ellipse_model
        if label == "photutils" and result["isophotes"]:
            phot_model = _build_photutils_model(image.shape, result["isophotes"])
            if phot_model is not None:
                models["photutils"] = phot_model

    # AutoProf profile (array-based, not isophote list)
    if ap_result is not None:
        ap_profile = build_method_profile(ap_result["profile"])
        if ap_profile is not None:
            ap_profile["runtime_seconds"] = ap_result["wall_seconds"]
            profiles["AutoProf"] = ap_profile
        # AutoProf 2D model from its FITS output
        ap_model = _load_autoprof_model(ap_result)
        if ap_model is not None:
            models["AutoProf"] = ap_model

    # --- QA figures ---
    # Figure 1: isoster OLS vs WLS only
    iso_profiles = {k: v for k, v in profiles.items() if k.startswith("isoster")}
    iso_models = {k: v for k, v in models.items() if k.startswith("isoster")}
    if iso_profiles:
        qa_iso_path = config_dir / f"qa_{config_label}_ols_vs_wls.png"
        plot_comparison_qa_figure(
            image,
            iso_profiles,
            title=f"{galaxy['name']} — {config_label} — OLS vs WLS",
            output_path=qa_iso_path,
            models=iso_models if iso_models else None,
            mask=mask,
            method_styles=METHOD_STYLES,
        )
        print(f"    QA (OLS vs WLS) → {qa_iso_path}")

    # Figure 2: all methods comparison (when external methods present)
    has_external = any(
        k in profiles for k in ("photutils", "AutoProf")
    )
    if has_external and len(profiles) >= 2:
        qa_all_path = config_dir / f"qa_{config_label}_all_methods.png"
        plot_comparison_qa_figure(
            image,
            profiles,
            title=f"{galaxy['name']} — {config_label} — all methods",
            output_path=qa_all_path,
            models=models if models else None,
            mask=mask,
            method_styles=METHOD_STYLES,
        )
        print(f"    QA (all methods) → {qa_all_path}")

    # --- Compute summaries ---
    summaries = compute_case_summary(
        profiles, galaxy["name"], config_label, ols_result, wls_result
    )
    return summaries


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def compute_case_summary(
    profiles: dict[str, dict[str, np.ndarray]],
    galaxy_name: str,
    config_label: str,
    ols_result: dict[str, Any],
    wls_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute per-method summary stats and WLS/OLS error ratio.

    Returns a list of summary-row dicts (one per method present in profiles).
    """
    # Compute WLS/OLS error ratio on outer half with 5% SMA tolerance matching
    wls_ols_ratio = _compute_error_ratio(profiles)

    rows: list[dict[str, Any]] = []
    for method_label, profile in profiles.items():
        sma = profile["sma"]
        n_iso = len(sma)

        # Convergence rate from stop codes (code 0 = converged)
        stop_codes = profile.get("stop_codes")
        if stop_codes is not None:
            converged = np.sum(stop_codes == 0)
            convergence_rate = round(converged / max(n_iso, 1), 4)
        else:
            convergence_rate = np.nan

        # Median intensity error
        intens_err = profile.get("intens_err")
        if intens_err is not None:
            finite_err = intens_err[np.isfinite(intens_err)]
            median_err = float(np.median(finite_err)) if finite_err.size > 0 else np.nan
        else:
            median_err = np.nan

        # Wall time
        wall = profile.get("runtime_seconds", np.nan)

        # WLS/OLS ratio only for isoster WLS row
        ratio = wls_ols_ratio if method_label == "isoster WLS" else np.nan

        rows.append({
            "galaxy": galaxy_name,
            "config": config_label,
            "method": method_label,
            "n_isophotes": n_iso,
            "wall_seconds": wall,
            "convergence_rate": convergence_rate,
            "median_intens_err": median_err,
            "wls_ols_err_ratio": ratio,
        })

    return rows


def _compute_error_ratio(
    profiles: dict[str, dict[str, np.ndarray]],
) -> float:
    """Compute median WLS/OLS intens_err ratio for outer-half isophotes.

    Matches isophotes by SMA with 5% tolerance.
    """
    ols = profiles.get("isoster OLS")
    wls = profiles.get("isoster WLS")
    if ols is None or wls is None:
        return np.nan

    ols_err = ols.get("intens_err")
    wls_err = wls.get("intens_err")
    if ols_err is None or wls_err is None:
        return np.nan

    ols_sma = ols["sma"]
    wls_sma = wls["sma"]
    sma_median = np.median(ols_sma)
    outer_mask = ols_sma >= sma_median

    ratios = []
    for sma_val, err_val in zip(ols_sma[outer_mask], ols_err[outer_mask]):
        if err_val <= 0 or not np.isfinite(err_val):
            continue
        idx = np.argmin(np.abs(wls_sma - sma_val))
        if np.abs(wls_sma[idx] - sma_val) / max(sma_val, 1.0) > 0.05:
            continue
        wls_err_val = wls_err[idx]
        if wls_err_val <= 0 or not np.isfinite(wls_err_val):
            continue
        ratios.append(wls_err_val / err_val)

    return float(np.median(ratios)) if ratios else np.nan


# ---------------------------------------------------------------------------
# Per-galaxy loop
# ---------------------------------------------------------------------------


def run_galaxy(
    galaxy: dict[str, Any],
    config_filter: str | None = None,
    run_photutils: bool = True,
    run_autoprof_flag: bool = True,
) -> list[dict[str, Any]]:
    """Run all configs for one galaxy. Returns list of summary rows."""
    print(f"\n{'='*60}")
    print(f"Galaxy: {galaxy['name']}")
    print(f"{'='*60}")

    galaxy_data = load_galaxy_data(galaxy)
    output_dir = resolve_output_directory(
        "example_wls_systematic", galaxy["name"]
    )

    all_summaries: list[dict[str, Any]] = []

    for config_label, overrides in CONFIG_MATRIX.items():
        if config_filter and config_label != config_filter:
            continue

        summaries = run_single_config(
            galaxy_data, galaxy, config_label, overrides, output_dir,
            run_photutils=run_photutils,
            run_autoprof_flag=run_autoprof_flag,
        )
        all_summaries.extend(summaries)

    # Write per-galaxy CSV
    csv_path = output_dir / "summary_table.csv"
    _write_summary_csv(all_summaries, csv_path)
    print(f"\n  Per-galaxy summary → {csv_path}")

    return all_summaries


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write summary rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary_table(rows: list[dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout."""
    header = (
        f"{'galaxy':>30s}  {'config':>15s}  {'method':>14s}  "
        f"{'n_iso':>5s}  {'wall_s':>7s}  {'conv':>5s}  "
        f"{'med_err':>10s}  {'wls/ols':>8s}"
    )
    print(f"\n{'='*100}")
    print("GRAND SUMMARY")
    print(f"{'='*100}")
    print(header)
    print("-" * 100)
    for r in rows:
        ratio_str = f"{r['wls_ols_err_ratio']:.3f}" if np.isfinite(r["wls_ols_err_ratio"]) else ""
        conv_str = f"{r['convergence_rate']:.2f}" if np.isfinite(r["convergence_rate"]) else ""
        err_str = f"{r['median_intens_err']:.4e}" if np.isfinite(r["median_intens_err"]) else ""
        print(
            f"{r['galaxy']:>30s}  {r['config']:>15s}  {r['method']:>14s}  "
            f"{r['n_isophotes']:>5d}  {r['wall_seconds']:>7.2f}  {conv_str:>5s}  "
            f"{err_str:>10s}  {ratio_str:>8s}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Systematic WLS testing across fitting configurations."
    )
    parser.add_argument(
        "--galaxy", type=str, default=None,
        help="Run only this galaxy (default: all in registry).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Run only this config label (default: all in CONFIG_MATRIX).",
    )
    parser.add_argument(
        "--no-autoprof", action="store_true",
        help="Skip AutoProf runs.",
    )
    parser.add_argument(
        "--no-photutils", action="store_true",
        help="Skip photutils runs.",
    )
    args = parser.parse_args()

    # Validate config filter
    if args.config and args.config not in CONFIG_MATRIX:
        available = ", ".join(CONFIG_MATRIX.keys())
        print(f"Unknown config '{args.config}'. Available: {available}")
        sys.exit(1)

    # Select galaxies
    if args.galaxy:
        galaxies = [get_galaxy(args.galaxy)]
    else:
        galaxies = GALAXY_REGISTRY

    grand_summaries: list[dict[str, Any]] = []

    for galaxy in galaxies:
        summaries = run_galaxy(
            galaxy,
            config_filter=args.config,
            run_photutils=not args.no_photutils,
            run_autoprof_flag=not args.no_autoprof,
        )
        grand_summaries.extend(summaries)

    # Grand summary CSV
    grand_dir = resolve_output_directory("example_wls_systematic")
    grand_csv = grand_dir / "grand_summary.csv"
    _write_summary_csv(grand_summaries, grand_csv)
    print(f"\nGrand summary → {grand_csv}")

    # Print table
    _print_summary_table(grand_summaries)


if __name__ == "__main__":
    main()
