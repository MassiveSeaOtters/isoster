"""Shared helpers for baseline benchmark: galaxy registry, data loading, QA figures.

Supports isoster, photutils, and autoprof comparison runs with timed core
fitting, per-method 2D model reconstruction, and multi-method QA figures.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table

import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import build_method_profile, plot_comparison_qa_figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Galaxy registry
# ---------------------------------------------------------------------------

# Each entry: name, fits_path, cube_index (None for 2D), geometry overrides,
# config overrides, pixel_scale, zeropoint.

GALAXY_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "eso243-49",
        "fits_path": DATA_DIR / "eso243-49.fits",
        "cube_index": 0,
        "geometry": None,  # estimate at runtime
        "config_overrides": {
            "sma0": 10.0,
            "minsma": 1.0,
            "astep": 0.1,
            "minit": 10,
            "maxit": 50,
            "conver": 0.05,
        },
        "pixel_scale": 1.0,
        "zeropoint": 22.5,
    },
    {
        "name": "IC3370_mock2",
        "fits_path": DATA_DIR / "IC3370_mock2.fits",
        "cube_index": None,
        "geometry": {"x0": 566.0, "y0": 566.0, "eps": 0.239, "pa": -0.489},
        "config_overrides": {
            "sma0": 6.0,
            "minsma": 1.0,
            "astep": 0.1,
            "minit": 10,
            "maxit": 50,
            "conver": 0.05,
        },
        "pixel_scale": 1.0,
        "zeropoint": 22.5,
    },
    {
        "name": "ngc3610",
        "fits_path": DATA_DIR / "ngc3610.fits",
        "cube_index": 0,
        "geometry": None,  # estimate at runtime
        "config_overrides": {
            "sma0": 5.0,
            "minsma": 1.0,
            "astep": 0.1,
            "minit": 10,
            "maxit": 50,
            "conver": 0.05,
        },
        "pixel_scale": 1.0,
        "zeropoint": 22.5,
    },
]


def get_galaxy(name: str) -> dict[str, Any]:
    """Look up a galaxy entry by name (case-insensitive)."""
    key = name.lower().replace("-", "").replace("_", "")
    for entry in GALAXY_REGISTRY:
        if entry["name"].lower().replace("-", "").replace("_", "") == key:
            return entry
    available = ", ".join(e["name"] for e in GALAXY_REGISTRY)
    raise ValueError(f"Unknown galaxy '{name}'. Available: {available}")


# ---------------------------------------------------------------------------
# Data loading and geometry estimation
# ---------------------------------------------------------------------------


def load_galaxy_image(fits_path: Path, cube_index: int | None = None) -> np.ndarray:
    """Load FITS image as float64.  Handles 2D and 3D cubes."""
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float64)
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 3:
                idx = cube_index if cube_index is not None else 0
                return np.asarray(hdu.data[idx], dtype=np.float64)
    raise ValueError(f"No usable image data in {fits_path}")


def estimate_moment_geometry(image: np.ndarray, percentile: float = 80.0) -> dict:
    """Estimate center, ellipticity, and PA from image moments."""
    threshold = np.nanpercentile(image, percentile)
    weights = np.clip(image - threshold, 0, None)
    total = np.nansum(weights)
    if total <= 0:
        h, w = image.shape
        return {"x0": w / 2.0, "y0": h / 2.0, "eps": 0.2, "pa": 0.0}

    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]].astype(float)
    x0 = float(np.nansum(weights * xx) / total)
    y0 = float(np.nansum(weights * yy) / total)
    dx, dy = xx - x0, yy - y0
    mxx = float(np.nansum(weights * dx**2) / total)
    myy = float(np.nansum(weights * dy**2) / total)
    mxy = float(np.nansum(weights * dx * dy) / total)
    trace = mxx + myy
    det = mxx * myy - mxy**2
    disc = max(trace**2 / 4 - det, 0)
    lam1 = trace / 2 + np.sqrt(disc)
    lam2 = trace / 2 - np.sqrt(disc)
    axis_ratio = np.sqrt(max(lam2, 0) / max(lam1, 1e-12))
    eps = float(np.clip(1 - axis_ratio, 0.05, 0.95))
    pa = float(0.5 * np.arctan2(2 * mxy, mxx - myy))
    return {"x0": x0, "y0": y0, "eps": eps, "pa": pa}


def estimate_background(image: np.ndarray, box_size: int = 50) -> tuple[float, float]:
    """Estimate background level and noise from image corners."""
    h, w = image.shape
    bs = min(box_size, h // 4, w // 4)
    corners = np.concatenate([
        image[:bs, :bs].ravel(),
        image[:bs, -bs:].ravel(),
        image[-bs:, :bs].ravel(),
        image[-bs:, -bs:].ravel(),
    ])
    corners = corners[np.isfinite(corners)]
    if corners.size == 0:
        return 0.0, 1.0
    return float(np.median(corners)), float(np.std(corners))


def resolve_geometry(galaxy_entry: dict[str, Any], image: np.ndarray) -> dict:
    """Resolve geometry from registry entry or estimate from image."""
    if galaxy_entry["geometry"] is not None:
        return dict(galaxy_entry["geometry"])
    return estimate_moment_geometry(image)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

MAX_RETRIES = 3


def _should_retry(result: dict | None, method: str) -> bool:
    """Return True if the fit result warrants a retry."""
    if result is None:
        return True
    iso_count = result.get("isophote_count", 0)
    if iso_count < 10:
        return True
    if method == "autoprof":
        return False  # no stop codes for autoprof
    # Check stop-code quality (isoster/photutils)
    codes = result.get("stop_code_counts", {})
    good = int(codes.get("0", 0))
    return good < iso_count * 0.5


def _escalate_params(
    geometry: dict,
    config_overrides: dict,
    attempt: int,
) -> tuple[dict, dict]:
    """Return perturbed geometry and relaxed config for retry attempt.

    Returns copies; originals are not mutated.
    """
    geo = dict(geometry)
    cfg = dict(config_overrides)

    if attempt == 1:
        geo["eps"] = float(np.clip(geo["eps"] + 0.02, 0.05, 0.95))
        geo["pa"] = geo["pa"] + 0.05
        cfg["maxit"] = max(cfg.get("maxit", 50), 100)
        cfg["maxgerr"] = max(cfg.get("maxgerr", 0.5), 0.8)
    elif attempt >= 2:
        geo["eps"] = float(np.clip(geo["eps"] - 0.02, 0.05, 0.95))
        geo["pa"] = geo["pa"] - 0.05
        cfg["maxit"] = max(cfg.get("maxit", 50), 200)
        cfg["maxgerr"] = max(cfg.get("maxgerr", 0.5), 1.2)

    return geo, cfg


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def check_photutils_available() -> bool:
    """Return True if photutils.isophote is importable."""
    try:
        from photutils.isophote import Ellipse, EllipseGeometry  # noqa: F401
        return True
    except ImportError:
        return False


def check_autoprof_available() -> bool:
    """Return True if AutoProf is available via the subprocess adapter."""
    try:
        from benchmarks.utils.autoprof_adapter import check_autoprof_available as _check
        return _check()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Isoster fitting (timed core only)
# ---------------------------------------------------------------------------


def run_isoster_fit(
    image: np.ndarray,
    geometry: dict,
    config_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run isoster fit.  Times only the core fit_image() call."""
    config_kwargs = dict(geometry)
    config_kwargs.update(config_overrides)
    # Match AutoProf's eccentric anomaly sampling for fair comparison
    config_kwargs.setdefault("use_eccentric_anomaly", True)
    # LSB tuning: adaptive integrator switches to median in outskirts,
    # permissive geometry continues through weak diagnostics
    config_kwargs.setdefault("integrator", "adaptive")
    config_kwargs.setdefault("lsb_sma_threshold", 80.0)
    config_kwargs.setdefault("permissive_geometry", True)
    config = IsosterConfig(**config_kwargs)

    # Time only the core fitting
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = isoster.fit_image(image, None, config)
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start

    isophotes = result["isophotes"]
    code_counts = dict(Counter(iso["stop_code"] for iso in isophotes))

    return {
        "method": "isoster",
        "isophotes": isophotes,
        "config": config,
        "runtime": {
            "wall_time_seconds": round(wall_time, 4),
            "cpu_time_seconds": round(cpu_time, 4),
        },
        "isophote_count": len(isophotes),
        "stop_code_counts": {str(k): v for k, v in sorted(code_counts.items())},
    }


# ---------------------------------------------------------------------------
# Photutils fitting (timed core only)
# ---------------------------------------------------------------------------


def build_photutils_config(geometry: dict, config_overrides: dict) -> dict[str, Any]:
    """Build photutils-compatible config dict from geometry and overrides."""
    return {
        "x0": geometry["x0"],
        "y0": geometry["y0"],
        "eps": geometry["eps"],
        "pa": geometry["pa"],
        "sma0": config_overrides["sma0"],
        "minsma": config_overrides.get("minsma", 1.0),
        "maxsma": config_overrides.get("maxsma", None),
        "step": config_overrides.get("astep", 0.1),
        "maxgerr": config_overrides.get("maxgerr", 0.5),
        "nclip": 0,
        "sclip": 3.0,
        "integrmode": "bilinear",
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


def run_photutils_fit(
    image: np.ndarray,
    photutils_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Run photutils.isophote fit.  Times only the core fit_image() call."""
    if not check_photutils_available():
        return None

    from photutils.isophote import Ellipse, EllipseGeometry

    # Setup (excluded from timing)
    geometry = EllipseGeometry(
        x0=photutils_config["x0"],
        y0=photutils_config["y0"],
        sma=photutils_config["sma0"],
        eps=photutils_config["eps"],
        pa=photutils_config["pa"],
    )
    ellipse = Ellipse(image, geometry)

    fit_kwargs = {
        "step": photutils_config["step"],
        "minsma": photutils_config["minsma"],
        "maxgerr": photutils_config["maxgerr"],
        "nclip": photutils_config["nclip"],
        "sclip": photutils_config["sclip"],
        "integrmode": photutils_config["integrmode"],
    }
    if photutils_config["maxsma"] is not None:
        fit_kwargs["maxsma"] = photutils_config["maxsma"]

    # Time only the core fitting
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isolist = ellipse.fit_image(**fit_kwargs)
    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start

    isophotes = _convert_photutils_isolist(isolist)
    code_counts = dict(Counter(iso["stop_code"] for iso in isophotes))

    return {
        "method": "photutils",
        "isophotes": isophotes,
        "isolist": isolist,
        "runtime": {
            "wall_time_seconds": round(wall_time, 4),
            "cpu_time_seconds": round(cpu_time, 4),
        },
        "isophote_count": len(isophotes),
        "stop_code_counts": {str(k): v for k, v in sorted(code_counts.items())},
    }


# ---------------------------------------------------------------------------
# AutoProf fitting
# ---------------------------------------------------------------------------


def build_autoprof_config(
    geometry: dict,
    config_overrides: dict,
    pixel_scale: float,
    zeropoint: float,
    background: float,
    background_noise: float,
) -> dict[str, Any]:
    """Build autoprof-compatible config dict."""
    return {
        "pixel_scale": pixel_scale,
        "zeropoint": zeropoint,
        "center": [geometry["x0"], geometry["y0"]],
        "eps": geometry["eps"],
        "pa_rad_math": geometry["pa"],
        "background": background,
        "background_noise": background_noise,
    }


def prepare_2d_fits_for_autoprof(
    image: np.ndarray,
    output_dir: Path,
    galaxy_name: str,
) -> Path:
    """Write a 2D FITS file for AutoProf (handles 3D cubes transparently)."""
    out_path = output_dir / f"{galaxy_name}_2d.fits"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=image.astype(np.float32))
    hdu.writeto(out_path, overwrite=True)
    return out_path


def run_autoprof_fit(
    image: np.ndarray,
    output_dir: Path,
    galaxy_name: str,
    autoprof_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Run AutoProf via subprocess adapter.  Returns profile dict or None."""
    if not check_autoprof_available():
        return None

    from benchmarks.utils.autoprof_adapter import run_autoprof_fit as _run_autoprof

    autoprof_output_dir = output_dir / "autoprof_workdir"
    autoprof_output_dir.mkdir(parents=True, exist_ok=True)

    # AutoProf needs a 2D FITS file on disk
    fits_2d_path = prepare_2d_fits_for_autoprof(
        image, autoprof_output_dir, galaxy_name,
    )

    result = _run_autoprof(
        image_path=fits_2d_path,
        output_dir=autoprof_output_dir,
        galaxy_name=galaxy_name,
        pixel_scale=autoprof_config["pixel_scale"],
        zeropoint=autoprof_config["zeropoint"],
        center=tuple(autoprof_config["center"]),
        eps=autoprof_config["eps"],
        pa_rad_math=autoprof_config["pa_rad_math"],
        background=autoprof_config["background"],
        background_noise=autoprof_config["background_noise"],
        run_ellipse_model=True,
    )

    if result is None:
        return None

    return {
        "method": "autoprof",
        "profile": result,
        "runtime": {
            "wall_time_seconds": round(result.get("runtime_s", 0.0), 4),
            "cpu_time_seconds": 0.0,  # not available from subprocess
        },
        "isophote_count": result.get("n_isophotes", 0),
    }


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------


def build_isoster_model_image(
    image_shape: tuple[int, int],
    isophotes: list[dict],
) -> np.ndarray | None:
    """Build 2D model using isoster's native model builder."""
    if not isophotes:
        return None
    try:
        return build_isoster_model(image_shape, isophotes)
    except Exception as exc:
        print(f"    isoster model build failed: {exc}")
        return None


def build_photutils_model_image(
    image_shape: tuple[int, int],
    isophotes: list[dict],
    *,
    high_harmonics: bool = True,
    fill: float = 0.0,
) -> np.ndarray | None:
    """Build 2D model using photutils' native build_ellipse_model."""
    if not isophotes:
        return None

    try:
        from photutils.isophote import build_ellipse_model
    except ImportError:
        return None

    # Build the adapter (duck-typed isolist for photutils model builder)
    required = ["sma", "intens", "eps", "pa", "x0", "y0"]
    if high_harmonics:
        required.append("grad")
    harmonic_keys = ["a3", "b3", "a4", "b4"]

    # Filter to valid rows
    valid_rows = []
    for iso in isophotes:
        if iso.get("sma", 0) <= 0:
            continue
        if not all(np.isfinite(iso.get(k, np.nan)) for k in required):
            continue
        valid_rows.append(iso)

    if len(valid_rows) < 6:
        print(f"    photutils model: insufficient valid rows ({len(valid_rows)})")
        return None

    # Sort by sma and deduplicate
    valid_rows.sort(key=lambda r: r["sma"])
    seen_sma = set()
    unique_rows = []
    for row in valid_rows:
        if row["sma"] not in seen_sma:
            seen_sma.add(row["sma"])
            unique_rows.append(row)

    if len(unique_rows) < 6:
        return None

    # Build column arrays
    columns = {}
    for key in required + harmonic_keys:
        columns[key] = np.array(
            [r.get(key, 0.0) for r in unique_rows], dtype=float
        )
    columns.setdefault("grad", np.zeros(len(unique_rows)))
    columns["pa"] = np.unwrap(columns["pa"], period=np.pi)

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
            image_shape, adapter, fill=fill,
            high_harmonics=high_harmonics, sma_interval=0.1,
        )
    except Exception as exc:
        print(f"    photutils model build failed: {exc}")
        return None


def load_autoprof_model_image(
    autoprof_result: dict[str, Any],
) -> np.ndarray | None:
    """Load AutoProf's native 2D model from its FITS output."""
    profile = autoprof_result.get("profile", {})
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
# Artifact I/O
# ---------------------------------------------------------------------------


def save_profile_fits(isophotes: list[dict], output_path: Path) -> None:
    """Save isophote profile as a FITS table."""
    if not isophotes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Table().write(output_path, format="fits", overwrite=True)
        return

    keys = [
        k for k in isophotes[0]
        if isinstance(isophotes[0][k], (int, float, np.integer, np.floating))
    ]
    table = Table()
    for key in keys:
        table[key] = [iso.get(key, np.nan) for iso in isophotes]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_path, format="fits", overwrite=True)


def save_profile_ecsv(isophotes: list[dict], output_path: Path) -> None:
    """Save isophote profile as an ECSV table."""
    if not isophotes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Table().write(output_path, format="ascii.ecsv", overwrite=True)
        return

    keys = [
        k for k in isophotes[0]
        if isinstance(isophotes[0][k], (int, float, np.integer, np.floating))
    ]
    table = Table()
    for key in keys:
        table[key] = [iso.get(key, np.nan) for iso in isophotes]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_path, format="ascii.ecsv", overwrite=True)


def save_autoprof_profile_ecsv(profile: dict, output_path: Path) -> None:
    """Save autoprof profile arrays as an ECSV table."""
    table = Table()
    for key in ["sma", "intens", "eps", "pa", "intens_err", "eps_err", "pa_err",
                 "x0", "y0", "a3", "b3", "a4", "b4"]:
        if key in profile and isinstance(profile[key], np.ndarray):
            table[key] = profile[key]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_path, format="ascii.ecsv", overwrite=True)


def save_model_fits(model: np.ndarray, output_path: Path) -> None:
    """Save 2D model image as a FITS file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=model.astype(np.float32))
    hdu.writeto(output_path, overwrite=True)


def save_fit_configs(configs: dict[str, Any], output_path: Path) -> None:
    """Save all method fit configurations as JSON."""

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(configs, fp, indent=2, sort_keys=True, default=_serialize)


# ---------------------------------------------------------------------------
# QA figures
# ---------------------------------------------------------------------------


def make_comparison_qa_figure(
    image: np.ndarray,
    methods_data: dict[str, dict[str, Any]],
    galaxy_name: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Build multi-method comparison QA figure.

    Thin wrapper around ``isoster.plotting.plot_comparison_qa_figure``.
    Converts benchmark-specific ``methods_data`` format into the library's
    ``profiles`` + ``models`` interface.

    Parameters
    ----------
    image : np.ndarray
        Original galaxy image.
    methods_data : dict
        Keys are method names ('isoster', 'photutils', 'autoprof').
        Values are dicts with keys:
        - 'isophotes' or 'profile': list[dict] or autoprof profile dict
        - 'model': np.ndarray or None (2D reconstructed model)
        - 'runtime': dict with wall_time_seconds
    galaxy_name : str
        Galaxy identifier for the title.
    output_path : Path
        Output path for the figure.
    dpi : int
        Figure resolution.
    """
    profiles: dict[str, dict[str, np.ndarray]] = {}
    models: dict[str, np.ndarray] = {}

    for method_name, mdata in methods_data.items():
        # Build standardized profile via library helper
        if method_name == "autoprof":
            raw = mdata.get("profile")
        else:
            raw = mdata.get("isophotes", [])
        profile = build_method_profile(raw) if raw else None

        if profile is not None:
            # Inject runtime so the title can display it
            rt = mdata.get("runtime", {})
            profile["runtime_seconds"] = rt.get("wall_time_seconds", 0.0)
            retries = mdata.get("retries", 0)
            if retries > 0:
                profile["retries"] = retries
            profiles[method_name] = profile

        model = mdata.get("model")
        if model is not None:
            models[method_name] = model

    plot_comparison_qa_figure(
        image,
        profiles,
        title=galaxy_name,
        output_path=output_path,
        models=models,
        dpi=dpi,
    )
