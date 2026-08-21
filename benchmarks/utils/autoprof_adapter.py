"""AutoProf adapter for benchmark comparisons.

Provides functions to run AutoProf isophote fitting with controlled initial
parameters (bypassing automatic background/center/PSF steps) and parse
the resulting profile into a standardized format comparable with isoster output.

AutoProf requires numpy <2 and photutils <=1.5, which conflicts with isoster's
numpy 2.x environment.  This adapter runs AutoProf via subprocess using the
interpreter of an isolated virtual environment, then parses the resulting
.prof file back in the isoster environment.  See benchmarks/autoprof_env.py
for where that interpreter is looked up and benchmarks/exhausted/README.md
for the install recipe.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from math import degrees, radians
from pathlib import Path
from typing import Optional

import numpy as np

from benchmarks.autoprof_env import resolve_autoprof_python

# Interpreter of the isolated venv where AutoProf is installed (numpy 1.x).
# Override for a shell session with the AUTOPROF_PYTHON environment variable;
# the default and the resolution rules live in benchmarks/autoprof_env.py.
AUTOPROF_PYTHON = resolve_autoprof_python()


def check_autoprof_available() -> bool:
    """Return True if AutoProf is importable in the AutoProf Python env."""
    try:
        result = subprocess.run(
            [AUTOPROF_PYTHON, "-c", "import autoprof"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def isoster_pa_to_autoprof_init(pa_rad_math: float) -> float:
    """Convert isoster PA to AutoProf initial PA.

    Parameters
    ----------
    pa_rad_math : float
        isoster PA in radians, math convention (CCW from +x axis).

    Returns
    -------
    float
        AutoProf init PA in degrees, astro convention (CCW from +y axis).
        AutoProf's ``ap_isoinit_pa_set`` expects degrees and internally
        applies ``PA_shift_convention``.
    """
    return degrees(pa_rad_math) - 90.0


def autoprof_pa_to_isoster(pa_deg_astro: float) -> float:
    """Convert AutoProf output PA to isoster convention.

    Parameters
    ----------
    pa_deg_astro : float
        AutoProf PA in degrees, astro convention (CCW from +y axis).

    Returns
    -------
    float
        PA in radians, math convention (CCW from +x axis).
    """
    return radians((pa_deg_astro + 90.0) % 180.0)


def run_autoprof_fit(
    image_path: str | Path,
    output_dir: str | Path,
    galaxy_name: str,
    pixel_scale: float,
    zeropoint: float,
    center: tuple[float, float],
    eps: float,
    pa_rad_math: float,
    background: float,
    background_noise: float,
    extra_options: Optional[dict] = None,
    run_ellipse_model: bool = True,
) -> dict | None:
    """Run AutoProf isophote fitting via subprocess with fixed initial parameters.

    Spawns a system Python process (where AutoProf is installed with numpy <2)
    that runs the fit and writes the .prof file.  The profile is then parsed
    back in the current environment.

    Parameters
    ----------
    image_path : str or Path
        Path to the FITS image.
    output_dir : str or Path
        Directory for AutoProf output files and logs.
    galaxy_name : str
        Name identifier for the galaxy (used in output file naming).
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    zeropoint : float
        Photometric zeropoint (mag).
    center : tuple of float
        (x0, y0) center coordinates in pixels.
    eps : float
        Initial ellipticity.
    pa_rad_math : float
        Initial PA in radians, math convention (CCW from +x).
    background : float
        Fixed background level (counts/pixel).
    background_noise : float
        Fixed background noise sigma (counts/pixel).
    extra_options : dict, optional
        Additional AutoProf options to merge.
    run_ellipse_model : bool, optional
        If True (default), extend the AutoProf pipeline with the
        ``EllipseModel`` step to produce a native 2D model FITS file
        (``{galaxy_name}_genmodel.fits`` in output_dir). The path is
        returned as ``model_fits_path`` in the result dict.

    Returns
    -------
    dict or None
        Standardized profile dict with keys: sma, intens, eps, pa,
        intens_err, eps_err, pa_err, n_isophotes, runtime_s, and
        optionally model_fits_path (str) when run_ellipse_model=True.
        Returns None on failure.
    """
    if not check_autoprof_available():
        print("AutoProf not available — skipping.")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cx, cy = center
    pa_deg_init = isoster_pa_to_autoprof_init(pa_rad_math)

    options = {
        "ap_image_file": str(Path(image_path).resolve()),
        "ap_name": galaxy_name,
        "ap_pixscale": pixel_scale,
        "ap_zeropoint": zeropoint,
        "ap_saveto": str(output_dir.resolve()),
        "ap_fluxunits": "intensity",
        "ap_doplot": False,
        "ap_isoclip": True,
        # Measure higher-order Fourier harmonics (a3/b3, a4/b4) to match
        # isoster and photutils defaults for fair comparison.
        "ap_iso_measurecoefs": (3, 4),
        # Bypass automatic steps with fixed values
        "ap_set_center": {"x": cx, "y": cy},
        "ap_set_background": background,
        "ap_set_background_noise": background_noise,
        "ap_isoinit_pa_set": pa_deg_init,
        "ap_isoinit_ellip_set": eps,
    }

    if extra_options:
        options.update(extra_options)

    # EllipseModel reads results["prof data"]["SB_e"], which only exists when
    # AutoProf uses its default mag-based internal representation.  When
    # ap_fluxunits="intensity", the key is "I_e" instead, causing a KeyError.
    # Solution: remove ap_fluxunits when running EllipseModel so AutoProf
    # stores the profile in mag/arcsec² internally; parse_autoprof_profile
    # handles both mag and intensity formats via the 'i_col' detection.
    if run_ellipse_model:
        options.pop("ap_fluxunits", None)
        options["ap_plotpath"] = str(output_dir.resolve())

    log_path = output_dir / "AutoProf.log"
    timing_path = output_dir / "_autoprof_timing.json"
    expected_model_fits = output_dir / f"{galaxy_name}_genmodel.fits"

    # The EllipseModel step is not in the default Isophote_Pipeline steps, so
    # we call UpdatePipeline to append it when requested.  Use a single-line
    # call to avoid indentation issues when embedding in the f-string.
    ellipse_model_block = ""
    if run_ellipse_model:
        steps = (
            '["background", "psf", "center", "isophoteinit",'
            ' "isophotefit", "isophoteextract", "checkfit", "writeprof", "ellipsemodel"]'
        )
        ellipse_model_block = f"pipeline.UpdatePipeline(new_pipeline_steps={steps})"

    # Build the subprocess script that runs AutoProf
    runner_script = textwrap.dedent(f"""\
        import json
        import time
        import warnings
        warnings.filterwarnings("ignore")

        options = json.loads('''{json.dumps(options)}''')

        from autoprof.Pipeline import Isophote_Pipeline

        log_path = {str(log_path)!r}
        timing_path = {str(timing_path)!r}

        try:
            pipeline = Isophote_Pipeline(loggername=log_path)
            {ellipse_model_block.rstrip()}
            t0 = time.perf_counter()
            pipeline.Process_Image(options)
            elapsed = time.perf_counter() - t0
            with open(timing_path, "w") as fp:
                json.dump({{"runtime_s": elapsed, "status": "ok"}}, fp)
        except Exception as exc:
            with open(timing_path, "w") as fp:
                json.dump({{"runtime_s": 0, "status": "error", "error": str(exc)}}, fp)
    """)

    try:
        proc = subprocess.run(
            [AUTOPROF_PYTHON, "-c", runner_script],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("AutoProf subprocess timed out (300s)")
        return None
    except Exception as exc:
        print(f"AutoProf subprocess failed: {exc}")
        return None

    if proc.returncode != 0:
        stderr_preview = proc.stderr[:500] if proc.stderr else "(no stderr)"
        print(f"AutoProf subprocess exited with code {proc.returncode}")
        print(f"  stderr: {stderr_preview}")
        return None

    # Read timing info
    elapsed = 0.0
    if timing_path.exists():
        try:
            with open(timing_path) as fp:
                timing = json.load(fp)
            if timing.get("status") != "ok":
                print(f"AutoProf reported error: {timing.get('error', 'unknown')}")
                return None
            elapsed = timing["runtime_s"]
        except Exception:
            pass

    # Find and parse the .prof output file
    prof_candidates = list(output_dir.glob(f"{galaxy_name}*.prof"))
    if not prof_candidates:
        print(f"No .prof file found in {output_dir} for galaxy '{galaxy_name}'")
        return None

    prof_path = prof_candidates[0]
    profile = parse_autoprof_profile(prof_path, pixel_scale, zeropoint=zeropoint)
    if profile is None:
        return None

    profile["runtime_s"] = elapsed

    if run_ellipse_model and expected_model_fits.exists():
        profile["model_fits_path"] = str(expected_model_fits)

    # Extract fixed center from .aux file and inject as constant arrays
    aux_candidates = list(output_dir.glob(f"{galaxy_name}*.aux"))
    if aux_candidates:
        center_xy = parse_autoprof_center(aux_candidates[0])
        if center_xy is not None:
            n = profile["n_isophotes"]
            profile["x0"] = np.full(n, center_xy[0])
            profile["y0"] = np.full(n, center_xy[1])

    return profile


def parse_autoprof_profile(
    prof_path: str | Path,
    pixel_scale: float,
    zeropoint: float = 22.5,
) -> dict | None:
    """Parse an AutoProf .prof file into a standardized profile dict.

    Parameters
    ----------
    prof_path : str or Path
        Path to the AutoProf ``.prof`` output file.
    pixel_scale : float
        Pixel scale in arcsec/pixel (used to convert R and intensity).

    Returns
    -------
    dict or None
        Profile dict with keys: sma, intens, eps, pa, intens_err, eps_err,
        pa_err, n_isophotes.  Returns None if parsing fails.

    Notes
    -----
    AutoProf .prof column names depend on the ``ap_fluxunits`` option:

    **Intensity mode** (``ap_fluxunits="intensity"``):

    - ``R`` : semi-major axis in arcsec
    - ``I`` : intensity in flux/arcsec^2
    - ``I_e`` : intensity error in flux/arcsec^2
    - Conversion: ``intens_px = I * pixel_scale**2``

    **Default mag mode** (no ``ap_fluxunits``):

    - ``R`` : semi-major axis in arcsec
    - ``SB`` : surface brightness in mag/arcsec^2
    - ``SB_e`` : SB error in mag/arcsec^2
    - Conversion: ``intens_px = 10^((zeropoint - SB) / 2.5) * pixel_scale^2``

    The ``ellip``, ``pa`` columns are the same in both modes.

    Common conversions:

    - SMA: ``sma_px = R / pixel_scale``
    - PA: degrees astro -> radians math: ``pa_rad = radians((pa_deg + 90) % 180)``
    """
    prof_path = Path(prof_path)
    if not prof_path.exists():
        print(f"Profile file not found: {prof_path}")
        return None

    try:
        from astropy.table import Table

        # AutoProf .prof format: line 1 is "#" + units (comma-sep),
        # line 2 is column names (comma-sep), data lines follow.
        # Try csv format first (handles the two-header-line case),
        # fall back to commented_header.
        try:
            table = Table.read(str(prof_path), format="ascii.csv", comment="#")
        except Exception:
            table = Table.read(str(prof_path), format="ascii.commented_header")
    except Exception as exc:
        print(f"Failed to parse .prof file {prof_path}: {exc}")
        return None

    if len(table) == 0:
        print(f"Empty profile table: {prof_path}")
        return None

    # Extract columns — AutoProf naming varies slightly across versions
    # Common column names: R, I, I_e, ellip, ellip_e, pa, pa_e
    col_names = table.colnames
    r_col = _find_column(col_names, ["R", "r", "SMA", "sma"])
    i_col = _find_column(col_names, ["I", "SB", "sb", "intens"])
    ie_col = _find_column(col_names, ["I_e", "SB_e", "sb_e", "intens_e"])
    ellip_col = _find_column(col_names, ["ellip", "ellipticity", "eps"])
    ellip_e_col = _find_column(col_names, ["ellip_e", "ellipticity_e", "eps_e"])
    pa_col = _find_column(col_names, ["pa", "PA", "position_angle"])
    pa_e_col = _find_column(col_names, ["pa_e", "PA_e", "position_angle_e"])

    if r_col is None or i_col is None:
        print(f"Missing required columns (R, I) in {prof_path}. Found: {col_names}")
        return None

    r_arcsec = np.array(table[r_col], dtype=float)
    i_raw = np.array(table[i_col], dtype=float)

    # Convert R (arcsec) -> SMA (pixels)
    sma = r_arcsec / pixel_scale

    # Convert profile intensity to counts/pixel.
    # Detect format from the column name:
    #   "I"  → intensity mode (flux/arcsec²): multiply by pixel_scale²
    #   "SB" → mag mode (mag/arcsec²): apply standard mag→flux conversion
    if i_col in ("I", "intens"):
        intens = i_raw * pixel_scale**2
    else:
        # Mag/arcsec² → counts/pixel:
        # flux/pixel = 10^((zp - SB) / 2.5) * pixscale²
        intens = 10.0 ** ((zeropoint - i_raw) / 2.5) * pixel_scale**2

    # Ellipticity
    eps = np.array(table[ellip_col], dtype=float) if ellip_col else np.full_like(sma, np.nan)

    # PA: degrees astro -> radians math
    if pa_col:
        pa_deg = np.array(table[pa_col], dtype=float)
        pa = np.array([radians((p + 90.0) % 180.0) for p in pa_deg])
    else:
        pa = np.full_like(sma, np.nan)

    # Error columns (optional).  When ie_col is "I_e" (intensity mode), multiply
    # by pixel_scale².  When "SB_e" (mag mode), convert magnitude error to
    # intensity error: intens_err = intens * SB_e * ln(10) / 2.5
    if ie_col:
        ie_raw = np.array(table[ie_col], dtype=float)
        if ie_col == "I_e":
            intens_err = ie_raw * pixel_scale**2
        else:
            # SB_e (mag/arcsec²) → intensity error (counts/pixel)
            intens_err = intens * ie_raw * np.log(10.0) / 2.5
    else:
        intens_err = np.full_like(sma, np.nan)
    eps_err = np.array(table[ellip_e_col], dtype=float) if ellip_e_col else np.full_like(sma, np.nan)
    pa_err = np.radians(np.array(table[pa_e_col], dtype=float)) if pa_e_col else np.full_like(sma, np.nan)

    # Higher-order Fourier harmonics (optional, present when
    # ap_iso_measurecoefs is set).  AutoProf names them a3, b3, a4, b4, etc.
    harmonic_arrays = {}
    for mode in (3, 4):
        for coef in ("a", "b"):
            col_name = f"{coef}{mode}"
            if col_name in col_names:
                harmonic_arrays[col_name] = np.array(table[col_name], dtype=float)

    # Filter out non-positive SMA, non-finite intensity, or sentinel bad values.
    # AutoProf marks bad isophotes with SB = 99.999 (mag mode) or I ≤ 0 (intensity mode).
    if i_col in ("I", "intens"):
        raw_valid = i_raw > 0.0
    else:
        raw_valid = i_raw < 90.0  # exclude sentinel 99.999 in mag mode
    valid = (sma > 0) & np.isfinite(intens) & raw_valid
    sma = sma[valid]
    intens = intens[valid]
    eps = eps[valid]
    pa = pa[valid]
    intens_err = intens_err[valid]
    eps_err = eps_err[valid]
    pa_err = pa_err[valid]
    for key in harmonic_arrays:
        harmonic_arrays[key] = harmonic_arrays[key][valid]

    # Sort by SMA
    order = np.argsort(sma)
    result = {
        "sma": sma[order],
        "intens": intens[order],
        "eps": eps[order],
        "pa": pa[order],
        "intens_err": intens_err[order],
        "eps_err": eps_err[order],
        "pa_err": pa_err[order],
        "n_isophotes": len(sma),
    }
    for key, arr in harmonic_arrays.items():
        result[key] = arr[order]
    return result


def parse_autoprof_center(aux_path: str | Path) -> tuple[float, float] | None:
    """Extract the fixed center from an AutoProf .aux file.

    Parameters
    ----------
    aux_path : str or Path
        Path to the AutoProf ``.aux`` file.

    Returns
    -------
    tuple of (x0, y0) or None
        Center coordinates in pixels, or None if not found.
    """
    aux_path = Path(aux_path)
    if not aux_path.exists():
        return None
    try:
        for line in aux_path.read_text().splitlines():
            if line.startswith("option ap_set_center:"):
                # Format: option ap_set_center: {'x': 128.79, 'y': 129.54}
                import ast

                dict_str = line.split(":", 1)[1].strip()
                center = ast.literal_eval(dict_str)
                return float(center["x"]), float(center["y"])
    except Exception:
        pass
    return None


def _find_column(col_names: list[str], candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for candidate in candidates:
        if candidate in col_names:
            return candidate
    return None
