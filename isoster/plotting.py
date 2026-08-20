"""
Plotting utilities for isophote analysis visualization.

This module provides reusable plotting functions for QA figures that follow
the Huang2013 campaign baseline style (build_method_qa_figure convention).
Key features: stop-code-separated scatter, arcsinh image display with
colorbars, fractional residual maps, robust axis limits, and PA unwrap.
"""

from __future__ import annotations

import platform
import re
import shutil
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse as MPLEllipse
from matplotlib.patches import Polygon as MPLPolygon

# Shared helpers (re-imported here for backward compatibility — historical
# call sites continue to use ``isoster.plotting._normalize_harmonic_for_plot``).
# Single source of truth lives in ``isoster._shared``.
from ._shared import _normalize_harmonic_for_plot  # noqa: F401

# ---------------------------------------------------------------------------
# Stop-code styling constants
# ---------------------------------------------------------------------------

STOP_CODE_STYLES = {
    0: {"color": "#1f77b4", "marker": "o", "label": "stop=0"},
    1: {"color": "#ff7f0e", "marker": "s", "label": "stop=1"},
    2: {"color": "#2ca02c", "marker": "^", "label": "stop=2"},
    3: {"color": "#d62728", "marker": "D", "label": "stop=3"},
    -1: {"color": "#9467bd", "marker": "X", "label": "stop=-1"},
}

MONOCHROME_STOP_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
    -1: "X",
}

MONOCHROME_STOP_COLORS = {
    0: "#111111",
    1: "#1f3b73",
    2: "#1b5e20",
    3: "#7f1d1d",
    -1: "#4b2e83",
}


def _validate_sb_inputs(sb_zeropoint, pixel_scale_arcsec):
    """Repo convention: ``sb_zeropoint`` and ``pixel_scale_arcsec`` are
    always supplied together and never pre-combined. Callers pass the two
    raw survey constants and SB panels compute
    ``μ = -2.5·log10(I/pixarea) + zp`` internally.
    """
    if (sb_zeropoint is None) != (pixel_scale_arcsec is None):
        raise ValueError(
            "sb_zeropoint and pixel_scale_arcsec must be passed together "
            "(both set for mag/arcsec^2, or both None for log10(I)). "
            "Never pre-combine them into a single zeropoint."
        )
    if pixel_scale_arcsec is not None and pixel_scale_arcsec <= 0:
        raise ValueError(f"pixel_scale_arcsec must be positive, got {pixel_scale_arcsec!r}")


def transform_sb_profile(
    intens: np.ndarray,
    intens_err: np.ndarray | None = None,
    *,
    sb_zeropoint: float | None = None,
    pixel_scale_arcsec: float | None = None,
    sb_profile_scale: str = "log10",
    sb_asinh_softening: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str, bool, float | None]:
    """Transform profile intensities for the SB panel.

    Returns ``(y, yerr, ylabel, invert_yaxis, zero_intensity_y)``.
    ``sb_profile_scale='log10'`` preserves the existing behavior:
    calibrated profiles are plotted as magnitudes, uncalibrated profiles
    as ``log10(I)``. ``'asinh'``/``'arcsinh'`` uses an asinh-magnitude
    transform when calibrated so the high-S/N limit matches the log10
    magnitude profile while zero intensity remains finite.
    """
    _validate_sb_inputs(sb_zeropoint, pixel_scale_arcsec)
    scale_name = str(sb_profile_scale).lower()
    if scale_name == "arcsinh":
        scale_name = "asinh"
    if scale_name not in {"log10", "asinh"}:
        raise ValueError("sb_profile_scale must be 'log10', 'asinh', or 'arcsinh'")

    intens = np.asarray(intens, dtype=float)
    err = None if intens_err is None else np.asarray(intens_err, dtype=float)
    y = np.full_like(intens, np.nan, dtype=float)
    yerr = None if err is None else np.full_like(intens, np.nan, dtype=float)

    if scale_name == "log10":
        valid = np.isfinite(intens) & (intens > 0.0)
        if sb_zeropoint is not None:
            pixarea = float(pixel_scale_arcsec) ** 2
            flux = intens / pixarea
            y[valid] = -2.5 * np.log10(flux[valid]) + float(sb_zeropoint)
            if yerr is not None:
                yerr[valid] = 2.5 * err[valid] / (intens[valid] * np.log(10.0))
            return y, yerr, r"$\mu$ [mag/arcsec$^2$]", True, None
        y[valid] = np.log10(intens[valid])
        if yerr is not None:
            yerr[valid] = err[valid] / (intens[valid] * np.log(10.0))
        return y, yerr, r"$\log_{10}(I)$", False, None

    if sb_zeropoint is not None:
        pixarea = float(pixel_scale_arcsec) ** 2
        flux = intens / pixarea
        flux_err = None if err is None else err / pixarea
        softening = _resolve_asinh_softening(flux, sb_asinh_softening)
        valid = np.isfinite(flux)
        y[valid] = float(sb_zeropoint) - (2.5 / np.log(10.0)) * (
            np.arcsinh(flux[valid] / (2.0 * softening)) + np.log(softening)
        )
        if yerr is not None and flux_err is not None:
            yerr[valid] = (
                (2.5 / np.log(10.0)) * np.abs(flux_err[valid]) / np.sqrt(flux[valid] ** 2 + (2.0 * softening) ** 2)
            )
        zero_y = float(sb_zeropoint) - (2.5 / np.log(10.0)) * np.log(softening)
        return y, yerr, r"$\mu_{\mathrm{asinh}}$ [mag/arcsec$^2$]", True, float(zero_y)

    softening = _resolve_asinh_softening(intens, sb_asinh_softening)
    valid = np.isfinite(intens)
    y[valid] = np.arcsinh(intens[valid] / softening)
    if yerr is not None:
        yerr[valid] = np.abs(err[valid]) / (softening * np.sqrt(1.0 + (intens[valid] / softening) ** 2))
    return y, yerr, r"$\mathrm{asinh}(I / I_s)$", False, 0.0


def _resolve_asinh_softening(values: np.ndarray, softening: float | None) -> float:
    """Return a positive asinh softening scale in the same units as values."""
    if softening is not None:
        softening = float(softening)
        if not np.isfinite(softening) or softening <= 0.0:
            raise ValueError(f"sb_asinh_softening must be positive, got {softening!r}")
        return softening
    finite_positive = np.asarray(values, dtype=float)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return 1.0
    resolved = float(np.nanpercentile(finite_positive, 5.0))
    if not np.isfinite(resolved) or resolved <= 0.0:
        resolved = float(np.nanmedian(finite_positive))
    return max(resolved, 1e-30)


# ---------------------------------------------------------------------------
# Multi-method comparison constants
# ---------------------------------------------------------------------------

# Default visual styles for multi-method comparison figures.
# Each method entry provides: color, marker, marker_face ("filled" or "none"),
# overlay_color, overlay_width, label.
METHOD_STYLES: dict[str, dict[str, str | float]] = {
    "isoster": {
        "color": "#1f77b4",
        "marker": "o",
        "marker_face": "filled",
        "overlay_color": "white",
        "overlay_width": 1.0,
        "label": "isoster",
    },
    "photutils": {
        "color": "#d62728",
        "marker": "s",
        "marker_face": "none",
        "overlay_color": "orangered",
        "overlay_width": 1.1,
        "label": "photutils",
    },
    "autoprof": {
        "color": "#2ca02c",
        "marker": "^",
        "marker_face": "filled",
        "overlay_color": "lime",
        "overlay_width": 0.9,
        "label": "AutoProf",
    },
}


# ---------------------------------------------------------------------------
# Style and text helpers
# ---------------------------------------------------------------------------


def style_for_stop_code(stop_code: int, monochrome: bool = False) -> dict[str, str]:
    """Return plotting style dict (color, marker, label) for a stop code."""
    if monochrome:
        marker = MONOCHROME_STOP_MARKERS.get(stop_code, "o")
        color = MONOCHROME_STOP_COLORS.get(stop_code, "#374151")
        return {"color": color, "marker": marker, "label": f"stop={stop_code}"}
    if stop_code in STOP_CODE_STYLES:
        return STOP_CODE_STYLES[stop_code]
    return {"color": "#7f7f7f", "marker": "o", "label": f"stop={stop_code}"}


def configure_qa_plot_style() -> None:
    """Apply shared plotting style for QA figures."""
    latex_available = platform.system() != "Windows" and shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.usetex": latex_available,
            "axes.labelsize": 15,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 14,
            "axes.unicode_minus": False,
        }
    )


def latex_safe_text(label_text: str) -> str:
    """Escape percent signs when LaTeX text rendering is active."""
    if plt.rcParams.get("text.usetex", False):
        return label_text.replace("%", r"\%")
    return label_text


# ---------------------------------------------------------------------------
# PA normalization (unwrap-aware)
# ---------------------------------------------------------------------------


def normalize_pa_degrees(pa_degrees: np.ndarray, anchor: float | None = None) -> np.ndarray:
    """Normalize PA values while preserving continuity across 180-degree periodicity.

    Uses double-angle unwrap to avoid spurious jumps larger than 90 degrees.

    Args:
        pa_degrees: Input PA array in degrees.
        anchor: If provided, shift the entire unwrapped profile by multiples of
                180 to be as close as possible to this anchor value at the first
                valid point.
    """
    pa = np.asarray(pa_degrees, dtype=float)
    output = np.full(pa.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(pa)
    if not np.any(finite_mask):
        return output

    # 1. Force into [0, 180) before unwrapping to ensure a clean start
    wrapped = np.mod(pa[finite_mask], 180.0)

    # 2. Double-angle unwrap (since PA has 180-deg period, 2*PA has 360-deg period)
    doubled_rad = np.deg2rad(2.0 * wrapped)
    unwrapped_rad = np.unwrap(doubled_rad)
    normalized = np.rad2deg(0.5 * unwrapped_rad)

    # 3. Optional anchoring to a global reference (to avoid large offsets from 0-180)
    if anchor is not None:
        first_val = normalized[0]
        offset = 180.0 * np.round((anchor - first_val) / 180.0)
        normalized += offset

    output[finite_mask] = normalized
    return output


# ---------------------------------------------------------------------------
# Multi-method profile building
# ---------------------------------------------------------------------------


def build_method_profile(
    data: list[dict] | dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    """Convert isophote list or array dict into standardized profile arrays.

    Accepts either a list of isophote dicts (isoster/photutils format, each
    dict has scalar values per isophote) or a dict of arrays (autoprof format,
    each key maps to an ndarray).

    Parameters
    ----------
    data : list[dict] or dict[str, ndarray]
        Isophote data in either format.

    Returns
    -------
    dict[str, ndarray] or None
        Standardized profile dict with keys: sma, x_axis, intens, eps, pa.
        Optional keys included when present in input: stop_codes, x0, y0,
        intens_err, rms.  Returns None if input is empty.
    """
    if isinstance(data, list):
        if len(data) == 0:
            return None
        sma = np.array([iso["sma"] for iso in data])
        profile = {
            "sma": sma,
            "x_axis": sma**0.25,
            "intens": np.array([iso["intens"] for iso in data]),
            "eps": np.array([iso["eps"] for iso in data]),
            "pa": np.array([iso["pa"] for iso in data]),
        }
        if "stop_code" in data[0]:
            profile["stop_codes"] = np.array([iso.get("stop_code", 0) for iso in data])
        for key in (
            "x0",
            "y0",
            "intens_err",
            "rms",
            "eps_err",
            "pa_err",
            "x0_err",
            "y0_err",
            "grad",
            "use_eccentric_anomaly",
            "tool",
        ):
            if key in data[0]:
                profile[key] = np.array([iso.get(key, np.nan) for iso in data])
        # Preserve harmonic coefficients (a3, b3, a4, b4, ...)
        for key in data[0]:
            if len(key) >= 2 and key[0] in ("a", "b") and key[1:].isdigit():
                n = int(key[1:])
                if n >= 3:
                    profile[key] = np.array([iso.get(key, 0.0) for iso in data])
        return profile

    elif isinstance(data, dict):
        sma = np.asarray(data.get("sma", []))
        if sma.size == 0:
            return None
        profile = {
            "sma": sma,
            "x_axis": sma**0.25,
            "intens": np.asarray(data["intens"]),
            "eps": np.asarray(data["eps"]),
            "pa": np.asarray(data["pa"]),
        }
        for key in (
            "stop_codes",
            "stop_code",
            "x0",
            "y0",
            "intens_err",
            "rms",
            "eps_err",
            "pa_err",
            "x0_err",
            "y0_err",
            "grad",
            "use_eccentric_anomaly",
            "tool",
        ):
            if key in data and isinstance(data[key], np.ndarray):
                out_key = "stop_codes" if key == "stop_code" else key
                profile[out_key] = data[key]
        # Preserve harmonic coefficients (a3, b3, a4, b4, ...)
        for key in data:
            if len(key) >= 2 and key[0] in ("a", "b") and key[1:].isdigit():
                n = int(key[1:])
                if n >= 3 and isinstance(data[key], np.ndarray):
                    profile[key] = data[key]
        # Scalar runtime metadata passes through so the comparison figure's
        # runtime annotation works without post-build injection (review P4).
        for key in ("runtime_seconds", "retries"):
            if key in data and not isinstance(data[key], np.ndarray):
                profile[key] = data[key]
        return profile

    return None


# ---------------------------------------------------------------------------
# Arcsinh display pipeline
# ---------------------------------------------------------------------------


def derive_arcsinh_parameters(
    image_values: np.ndarray,
    lower_percentile: float = 0.05,
    upper_percentile: float = 99.99,
    scale_percentile: float = 70.0,
) -> tuple[float, float, float, float]:
    """Derive arcsinh display parameters (low, high, scale, vmax) from image statistics."""
    finite_values = image_values[np.isfinite(image_values)]
    if finite_values.size == 0:
        return 0.0, 1.0, 1.0, 1.0

    low = float(np.nanpercentile(finite_values, lower_percentile))
    high = float(np.nanpercentile(finite_values, upper_percentile))
    if not np.isfinite(high) or high <= low:
        high = low + 1.0

    clipped = np.clip(image_values, low, high)
    shifted = np.clip(clipped - low, 0.0, None)
    positive = shifted[np.isfinite(shifted) & (shifted > 0.0)]
    scale = float(np.nanpercentile(positive, scale_percentile)) if positive.size else 1.0
    scale = max(scale, 1e-12)

    display = np.arcsinh(shifted / scale)
    finite_display = display[np.isfinite(display)]
    vmax = float(np.nanpercentile(finite_display, 99.8)) if finite_display.size else 1.0
    vmax = max(vmax, 1e-6)

    return low, high, scale, vmax


def make_arcsinh_display_from_parameters(
    image_values: np.ndarray,
    low: float,
    high: float,
    scale: float,
    vmax: float,
) -> tuple[np.ndarray, float, float]:
    """Build arcsinh display map from explicit scaling parameters.

    Returns (display_array, vmin, vmax).
    """
    clipped = np.clip(image_values, low, high)
    shifted = np.clip(clipped - low, 0.0, None)
    display = np.arcsinh(shifted / max(scale, 1e-12))
    return display, 0.0, max(vmax, 1e-6)


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def compute_fractional_residual_percent(image: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Compute residual map as ``100 * (model - data) / data``."""
    residual = np.full(image.shape, np.nan, dtype=float)
    valid = np.isfinite(image) & (np.abs(image) > 0.0)
    residual[valid] = 100.0 * (model[valid] - image[valid]) / image[valid]
    return residual


# ---------------------------------------------------------------------------
# Robust axis-limit helpers
# ---------------------------------------------------------------------------


def robust_limits(
    values: np.ndarray, lower_percentile: float = 5.0, upper_percentile: float = 95.0
) -> tuple[float, float]:
    """Compute robust plotting limits from finite values."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return -1.0, 1.0
    low = float(np.nanpercentile(finite_values, lower_percentile))
    high = float(np.nanpercentile(finite_values, upper_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        delta = max(abs(low), 1.0)
        return low - delta, high + delta
    return low, high


def set_axis_limits_from_finite_values(
    axis,
    values: np.ndarray,
    invert: bool = False,
    margin_fraction: float = 0.08,
    min_margin: float = 0.05,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
) -> None:
    """Set y-axis limits from finite data values with a comfortable margin."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    value_low = float(np.nanmin(finite_values))
    value_high = float(np.nanmax(finite_values))
    if value_high <= value_low:
        margin = max(min_margin, 0.05 * max(abs(value_low), 1.0))
    else:
        margin = max(min_margin, margin_fraction * (value_high - value_low))

    limit_low = value_low - margin
    limit_high = value_high + margin
    if lower_clip is not None:
        limit_low = max(limit_low, lower_clip)
    if upper_clip is not None:
        limit_high = min(limit_high, upper_clip)

    if limit_high <= limit_low:
        midpoint = 0.5 * (value_low + value_high)
        half_width = max(min_margin, 0.02)
        limit_low = midpoint - half_width
        limit_high = midpoint + half_width
        if lower_clip is not None:
            limit_low = max(limit_low, lower_clip)
        if upper_clip is not None:
            limit_high = min(limit_high, upper_clip)
        if limit_high <= limit_low:
            return

    if invert:
        axis.set_ylim(limit_high, limit_low)
    else:
        axis.set_ylim(limit_low, limit_high)


def set_x_limits_with_right_margin(
    axis,
    x_values: np.ndarray,
    min_margin: float = 0.02,
    margin_fraction: float = 0.03,
    skip_innermost: bool = True,
) -> None:
    """Set x-axis limits with margins for readability.

    Parameters
    ----------
    skip_innermost : bool
        If True (default), the left edge starts from the second-smallest
        unique x value (with a small left margin) so that the innermost
        point where no isophote can be reliably defined does not dominate
        the x-range.
    """
    finite_values = np.asarray(x_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    sorted_unique = np.unique(finite_values)
    if skip_innermost and sorted_unique.size >= 2:
        x_low = float(sorted_unique[1])
    else:
        x_low = float(sorted_unique[0])

    x_high = float(sorted_unique[-1])
    width = max(x_high - x_low, 0.0)
    right_margin = max(min_margin, margin_fraction * max(width, 1.0))
    left_margin = max(min_margin, margin_fraction * max(width, 1.0))
    axis.set_xlim(x_low - left_margin, x_high + right_margin)


# ---------------------------------------------------------------------------
# Isophote overlay drawing
# ---------------------------------------------------------------------------

DELTA_ROW_GATE = 0.5
_HARMONIC_KEY = re.compile(r"^[ab](\d+)$")


def _detect_harmonic_orders(iso: dict) -> list[int]:
    """Return sorted list of harmonic orders present in an isophote dict."""
    orders: set[int] = set()
    for key in iso:
        match = _HARMONIC_KEY.match(str(key))
        if match:
            n = int(match.group(1))
            if n >= 3:
                orders.add(n)
    return sorted(orders)


def _delta_at(iso: dict, theta: np.ndarray, orders: list[int]) -> np.ndarray:
    """Evaluate delta(theta) = sum_n a_n sin(n theta) + b_n cos(n theta)."""
    out = np.zeros_like(theta, dtype=np.float64)
    for n in orders:
        an = float(iso.get(f"a{n}", 0.0))
        bn = float(iso.get(f"b{n}", 0.0))
        if np.isfinite(an) and np.isfinite(bn):
            out += an * np.sin(n * theta) + bn * np.cos(n * theta)
    return out


def _rotate_translate_contour(
    x_rot: np.ndarray,
    y_rot: np.ndarray,
    x0: float,
    y0: float,
    pa_rad: float,
) -> np.ndarray:
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    x_pix = x0 + x_rot * cos_pa - y_rot * sin_pa
    y_pix = y0 + x_rot * sin_pa + y_rot * cos_pa
    return np.column_stack([x_pix, y_pix])


def contour_pure_ellipse(iso: dict, n_points: int = 360) -> np.ndarray:
    """Plain ellipse contour with no harmonic perturbation."""
    sma = float(iso["sma"])
    eps = float(iso.get("eps", iso.get("ellipticity", 0.0)))
    pa_rad = float(iso.get("pa", 0.0))
    x0 = float(iso.get("x0", 0.0))
    y0 = float(iso.get("y0", 0.0))
    alpha = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    x_rot = sma * np.cos(alpha)
    y_rot = sma * (1.0 - eps) * np.sin(alpha)
    return _rotate_translate_contour(x_rot, y_rot, x0, y0, pa_rad)


def contour_isoster_psi(iso: dict, n_points: int = 360) -> np.ndarray:
    """Eccentric-anomaly contour matching isoster psi-mode rendering."""
    sma = float(iso["sma"])
    eps = float(iso.get("eps", iso.get("ellipticity", 0.0)))
    pa_rad = float(iso.get("pa", 0.0))
    x0 = float(iso.get("x0", 0.0))
    y0 = float(iso.get("y0", 0.0))
    psi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    delta = _delta_at(iso, psi, _detect_harmonic_orders(iso))
    if float(np.max(np.abs(delta))) > DELTA_ROW_GATE:
        return contour_pure_ellipse(iso, n_points=n_points)
    # The model's perturbed isophote sits at r = sma/(1-delta) exactly
    # (model.py evaluates the profile at r*(1-delta)); sma*(1+delta) was
    # only its first-order approximation and diverged by ~30% at the gate
    # limit |delta|=0.5 (review P4).
    one_minus_delta = 1.0 - delta
    safe = np.where(
        np.abs(one_minus_delta) > 1e-6,
        one_minus_delta,
        np.where(one_minus_delta >= 0.0, 1e-6, -1e-6),
    )
    scale = 1.0 / safe
    x_rot = sma * scale * np.cos(psi)
    y_rot = sma * (1.0 - eps) * scale * np.sin(psi)
    return _rotate_translate_contour(x_rot, y_rot, x0, y0, pa_rad)


def contour_isoster_phi(iso: dict, n_points: int = 360) -> np.ndarray:
    """Image-azimuth contour matching isoster default phi-mode rendering."""
    sma = float(iso["sma"])
    eps = float(iso.get("eps", iso.get("ellipticity", 0.0)))
    pa_rad = float(iso.get("pa", 0.0))
    x0 = float(iso.get("x0", 0.0))
    y0 = float(iso.get("y0", 0.0))
    b_over_a = max(1e-3, 1.0 - eps)
    theta_rot = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    delta = _delta_at(iso, theta_rot, _detect_harmonic_orders(iso))
    if float(np.max(np.abs(delta))) > DELTA_ROW_GATE:
        return contour_pure_ellipse(iso, n_points=n_points)
    one_minus_delta = 1.0 - delta
    safe = np.where(
        np.abs(one_minus_delta) > 1e-6,
        one_minus_delta,
        np.where(one_minus_delta >= 0.0, 1e-6, -1e-6),
    )
    r_ell = sma / safe
    scale_geom = np.sqrt(np.cos(theta_rot) ** 2 + (np.sin(theta_rot) / b_over_a) ** 2)
    r_image = r_ell / scale_geom
    x_rot = r_image * np.cos(theta_rot)
    y_rot = r_image * np.sin(theta_rot)
    return _rotate_translate_contour(x_rot, y_rot, x0, y0, pa_rad)


def contour_photutils(iso: dict, n_points: int = 360) -> np.ndarray:
    """Photutils stores harmonic coefficients normalized by ``sma * |grad|``
    in the image polar-angle (phi) basis.

    The psi-basis contour helper is reused here as a first-order
    approximation: the polar-angle and eccentric-anomaly bases coincide
    to first order in the ellipticity, so the overlay error is modest.
    """
    return contour_isoster_psi(iso, n_points=n_points)


def contour_autoprof(iso: dict, n_points: int = 360) -> np.ndarray:
    """AutoProf overlay fallback: pure ellipse until its coefficient scale is ported."""
    return contour_pure_ellipse(iso, n_points=n_points)


def select_contour_fn(tool: str, *, use_eccentric_anomaly: bool):
    """Return the contour helper for a benchmark tool and harmonic convention."""
    tool = tool.lower()
    if tool == "isoster":
        return contour_isoster_psi if use_eccentric_anomaly else contour_isoster_phi
    if tool == "photutils":
        return contour_photutils
    if tool == "autoprof":
        return contour_autoprof
    return contour_isoster_psi


def _compute_harmonic_contour(iso: dict, n_points: int = 360) -> np.ndarray:
    """Compute a gated per-row isophote contour with tool-aware conventions.

    Rows with raw harmonic |delta| above ``DELTA_ROW_GATE`` fall back
    to a pure ellipse so low-S/N coefficient spikes do not dominate QA.
    """
    tool = str(iso.get("tool", "isoster"))
    use_eccentric_anomaly = bool(iso.get("use_eccentric_anomaly", False))
    return select_contour_fn(tool, use_eccentric_anomaly=use_eccentric_anomaly)(
        iso,
        n_points=n_points,
    )


def draw_isophote_overlays(
    axis,
    isophotes,
    step: int = 10,
    line_width: float = 1.0,
    alpha: float = 0.7,
    edge_color: str | None = None,
    draw_harmonics: bool = True,
) -> None:
    """Overlay selective isophotes on an image axis.

    When ``draw_harmonics`` is True and harmonic coefficients (a3/b3,
    a4/b4, ...) are present in the isophote dicts, the overlays show
    the actual non-elliptical isophote shape.  Otherwise pure ellipses
    are drawn.

    Parameters
    ----------
    isophotes : list of dict
        Isophote results with keys: sma, x0, y0, eps, pa, stop_code.
    step : int
        Draw every *step*-th isophote.
    edge_color : str or None
        Fixed color override; if None, color by stop code.
    draw_harmonics : bool
        If True (default), incorporate harmonic deviations when
        available.  Set to False to force pure-ellipse overlays.
    """
    if len(isophotes) == 0:
        return

    step = max(step, 1)
    for index in range(0, len(isophotes), step):
        iso = isophotes[index]
        sma = float(iso.get("sma", 0.0))
        if not np.isfinite(sma) or sma <= 1.0:
            continue

        x0 = float(iso.get("x0", 0.0))
        y0 = float(iso.get("y0", 0.0))
        eps = float(iso.get("eps", 0.0))
        pa_rad = float(iso.get("pa", 0.0))
        # Skip rows with unusable geometry (P4: only sma was guarded before)
        if not all(np.isfinite(v) for v in (x0, y0, eps, pa_rad)):
            continue
        stop_code = int(np.nan_to_num(iso.get("stop_code", 0), nan=0))  # P4: NaN stop codes read as 0

        style = style_for_stop_code(stop_code)
        color = edge_color if edge_color is not None else style["color"]

        has_harmonics = draw_harmonics and len(_detect_harmonic_orders(iso)) > 0

        if has_harmonics:
            contour = _compute_harmonic_contour(iso)
            patch = MPLPolygon(
                contour,
                closed=True,
                fill=False,
                linewidth=line_width,
                alpha=alpha,
                edgecolor=color,
            )
        else:
            patch = MPLEllipse(
                (x0, y0),
                2.0 * sma,
                2.0 * sma * (1.0 - eps),
                angle=np.rad2deg(pa_rad),
                fill=False,
                linewidth=line_width,
                alpha=alpha,
                edgecolor=color,
            )
        axis.add_patch(patch)


def model_isobrightness_levels(
    model: np.ndarray,
    *,
    n_levels: int = 6,
    log_spaced: bool = True,
    inside_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Choose model intensity levels for iso-brightness contours."""
    arr = np.asarray(model, dtype=np.float64)
    valid = np.isfinite(arr)
    if inside_mask is not None:
        valid &= np.asarray(inside_mask, dtype=bool)
    values = arr[valid]
    if log_spaced:
        values = values[values > 0.0]
    if values.size < n_levels:
        return np.array([])
    lo = float(np.percentile(values, 10.0))
    hi = float(np.percentile(values, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([])
    if log_spaced:
        return np.geomspace(max(lo, 1e-12), hi, n_levels)
    return np.linspace(lo, hi, n_levels)


def overlay_model_contours(
    axis,
    model: np.ndarray,
    *,
    n_levels: int = 6,
    color: str = "tab:orange",
    linewidth: float = 0.55,
    alpha: float = 0.75,
) -> None:
    """Overlay model iso-brightness contours on an image axis."""
    levels = model_isobrightness_levels(model, n_levels=n_levels)
    if levels.size == 0:
        return
    axis.contour(
        np.asarray(model, dtype=np.float64),
        levels=levels,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Scatter-by-stop-code profile plotting
# ---------------------------------------------------------------------------


def plot_profile_by_stop_code(
    axis,
    x_values: np.ndarray,
    y_values: np.ndarray,
    stop_codes: np.ndarray,
    y_errors: np.ndarray | None = None,
    marker_face: str = "filled",
    label_prefix: str = "",
    marker_size: float = 26.0,
    monochrome: bool = False,
) -> None:
    """Scatter/errorbar profile points separated by stop codes.

    Parameters
    ----------
    marker_face : str
        ``"filled"`` for solid markers, ``"open"`` for hollow.
    monochrome : bool
        If True, use monochrome palette (non-zero stop codes are hollow).
    """
    unique_stop_codes = sorted(
        {int(code) for code in stop_codes[np.isfinite(stop_codes)]},
        key=lambda code: (0, 0) if code == 0 else (1, code) if code > 0 else (2, abs(code)),
    )
    for stop_code in unique_stop_codes:
        mask = stop_codes == stop_code
        if not np.any(mask):
            continue
        style = style_for_stop_code(stop_code, monochrome=monochrome)
        if monochrome and stop_code != 0:
            face_color = "none"
        else:
            face_color = style["color"] if marker_face == "filled" else "none"

        if y_errors is not None:
            sanitized_errors = np.asarray(y_errors[mask], dtype=float)
            sanitized_errors[sanitized_errors < 0.0] = np.nan
            axis.errorbar(
                x_values[mask],
                y_values[mask],
                yerr=sanitized_errors,
                fmt=style["marker"],
                markersize=4.5,
                color=style["color"],
                mfc=face_color,
                mec=style["color"],
                capsize=1.8,
                linewidth=0.7,
                alpha=0.85,
                label=f"{label_prefix}{style['label']}",
            )
        else:
            axis.scatter(
                x_values[mask],
                y_values[mask],
                s=marker_size,
                marker=style["marker"],
                facecolors=face_color,
                edgecolors=style["color"],
                linewidths=0.9,
                alpha=0.9,
                label=f"{label_prefix}{style['label']}",
            )


# ---------------------------------------------------------------------------
# Main QA summary figure
# ---------------------------------------------------------------------------


def plot_qa_summary(
    title,
    image,
    isoster_model,
    isoster_res,
    photutils_res=None,
    mask=None,
    filename="qa_summary.png",
    relative_residual=False,
    sb_zeropoint=None,
    pixel_scale_arcsec=None,
    sb_profile_scale="log10",
    sb_asinh_softening=None,
):
    """Generate a QA figure following the Huang2013 campaign baseline style.

    Parameters
    ----------
    title : str
        Figure title.
    image : 2D array
        Original input image.
    isoster_model : 2D array
        Reconstructed model from isoster isophotes.
    isoster_res : list of dict
        Results from isoster fitting (each dict has sma, intens, eps, pa, etc.).
    photutils_res : photutils.isophote.IsophoteList, optional
        Results from photutils fitting for comparison overlay.
    mask : 2D bool array, optional
        Bad-pixel mask (True = masked).
    filename : str
        Output filename.
    relative_residual : bool
        When True, show ``100 * (model - data) / data`` (fractional residual %).
        When False (default), show ``data - model`` (direct residual).
        Note: the sign convention differs between modes (direct uses
        data-model; fractional uses model-data).
    sb_zeropoint : float, optional
        AB magnitude zeropoint ``zp`` matching the image flux units. Must
        be supplied together with ``pixel_scale_arcsec`` — the two are
        always passed separately and never pre-combined. When both are
        given, the SB panel shows ``μ = -2.5·log10(I / pixarea) + zp`` in
        mag/arcsec². When both are None, the panel shows ``log10(I)``.
    pixel_scale_arcsec : float, optional
        Pixel size in arcsec (``pixarea = pixel_scale_arcsec**2``). Must
        accompany ``sb_zeropoint``.
    sb_profile_scale : {"log10", "asinh", "arcsinh"}
        Surface-brightness profile y-scale. ``"log10"`` preserves the
        existing log/magnitude profile. ``"asinh"`` uses an asinh
        profile and draws an intensity-zero reference line.
    sb_asinh_softening : float, optional
        Positive softening scale for ``sb_profile_scale="asinh"``. In
        calibrated mode this is in flux per arcsec².
    """
    _validate_sb_inputs(sb_zeropoint, pixel_scale_arcsec)
    configure_qa_plot_style()

    # --- Extract isoster arrays ------------------------------------------------
    def get_arr(key, default=np.nan):
        return np.array([r.get(key, default) for r in isoster_res])

    i_sma = get_arr("sma")
    i_intens = get_arr("intens")
    i_intens_err = get_arr("intens_err")
    i_eps = get_arr("eps")
    i_eps_err = get_arr("eps_err")
    i_pa = get_arr("pa")
    i_pa_err = get_arr("pa_err")
    i_x0 = get_arr("x0")
    i_x0_err = get_arr("x0_err")
    i_y0 = get_arr("y0")
    i_y0_err = get_arr("y0_err")
    i_a3 = get_arr("a3")
    i_b3 = get_arr("b3")
    i_a4 = get_arr("a4")
    i_b4 = get_arr("b4")
    i_stop = np.nan_to_num(get_arr("stop_code", 0), nan=0).astype(int)  # P4: explicit NaN stop codes read as 0

    # Stored a_n/b_n are already Bender-normalized by the fitter
    # (compute_deviations divides by sma*|grad| before storing), so plot them
    # directly. Re-normalizing here shrank the plotted values by another
    # 1/(sma*|grad|) and distorted their radial shape (review P1).
    i_a3n = i_a3
    i_b3n = i_b3
    i_a4n = i_a4
    i_b4n = i_b4

    # Derived columns
    x_axis = i_sma**0.25
    valid_mask = np.isfinite(x_axis) & (i_sma > 1.0)

    i_ba = 1.0 - i_eps
    i_pa_deg = normalize_pa_degrees(np.degrees(i_pa))
    i_pa_err_deg = np.degrees(i_pa_err)

    # Center offsets relative to median
    median_x0 = np.nanmedian(i_x0[valid_mask]) if np.any(valid_mask) else 0.0
    median_y0 = np.nanmedian(i_y0[valid_mask]) if np.any(valid_mask) else 0.0
    i_dx = i_x0 - median_x0
    i_dy = i_y0 - median_y0

    # Determine panel count: base 4 + optional harmonics + optional CoG.
    # The panel is drawn whenever any stored harmonic coefficient is finite.
    has_harmonics = np.any(np.isfinite(i_a3n)) or np.any(np.isfinite(i_a4n))
    has_cog = any("cog" in r for r in isoster_res)

    n_panels = 4  # SB, centroid, b/a, PA
    if has_harmonics:
        n_panels += 1
    if has_cog:
        n_panels += 1

    # Height ratios: SB panel 2.2x, rest 1.0x, last 1.2x
    height_ratios = [2.2] + [1.0] * (n_panels - 2) + [1.2]

    # --- Figure layout ---------------------------------------------------------
    fig = plt.figure(figsize=(13.6, 11.0))
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 2.01],
        wspace=0.27,
    )

    # Left column: 3 image rows, each with a colorbar sliver
    left = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        hspace=0.10,
        wspace=-0.20,
    )

    # Right column: 1-D profile panels
    right = gridspec.GridSpecFromSubplotSpec(
        n_panels,
        1,
        subplot_spec=outer[1],
        height_ratios=height_ratios,
        hspace=0.0,
    )

    fig.suptitle(title, fontsize=20, y=0.989)

    # --- Left column: 2-D panels ----------------------------------------------
    # Shared arcsinh parameters derived from the data image
    arc_params = derive_arcsinh_parameters(image)

    # Panel 1: Data + isophotes
    _plot_arcsinh_image_panel(
        fig,
        left[0, 0],
        left[0, 1],
        image,
        arc_params,
        title="Data",
        title_style="qa",
        title_color="w",
        title_pos=(0.15, 0.9),
        mask=mask,
        overlay_isos=isoster_res,
        overlay_step=max(1, len(isoster_res) // 15),
        overlay_width=1.2,
        cbar_label=r"arcsinh((data $-$ p0.05) / scale)",
        show_xy_labels=True,
        show_ticks=True,
    )

    # Panel 2: Model
    _plot_arcsinh_image_panel(
        fig,
        left[1, 0],
        left[1, 1],
        isoster_model,
        arc_params,
        title="Model",
        title_style="qa",
        title_color="w",
        title_pos=(0.15, 0.9),
        contours=True,
        cbar_label=r"arcsinh((model $-$ p0.05) / scale)",
        show_xy_labels=True,
        show_ticks=True,
    )

    # Panel 3: Residual (coolwarm)
    _plot_residual_panel(
        fig,
        left[2, 0],
        left[2, 1],
        image,
        isoster_model,
        "Residual",
        relative_residual=relative_residual,
        title_style="qa",
        title_color="k",
        title_pos=(0.18, 0.9),
        show_xy_labels=True,
        show_ticks=True,
        cbar_fontsize=None,
    )

    # --- Right column: 1-D profiles -------------------------------------------
    panel_idx = 0

    # 1. Surface brightness
    # P4: one shared asinh scale for isoster and the photutils overlay —
    # each used to resolve its own softening from its own intensities.
    if sb_profile_scale in ("asinh", "arcsinh"):
        sb_asinh_softening = _resolve_asinh_softening(i_intens, sb_asinh_softening)
    ax_sb = fig.add_subplot(right[panel_idx])
    panel_idx += 1

    y_sb, y_sb_err, sb_ylabel, invert_sb_axis, zero_intensity_y = transform_sb_profile(
        i_intens,
        i_intens_err,
        sb_zeropoint=sb_zeropoint,
        pixel_scale_arcsec=pixel_scale_arcsec,
        sb_profile_scale=sb_profile_scale,
        sb_asinh_softening=sb_asinh_softening,
    )
    sb_valid = valid_mask & np.isfinite(y_sb)

    plot_profile_by_stop_code(
        ax_sb,
        x_axis[sb_valid],
        y_sb[sb_valid],
        i_stop[sb_valid],
        y_errors=y_sb_err[sb_valid],
        marker_face="filled",
        monochrome=True,
    )

    if photutils_res is not None:
        _overlay_photutils_sb(
            ax_sb,
            photutils_res,
            sb_zeropoint=sb_zeropoint,
            pixel_scale_arcsec=pixel_scale_arcsec,
            sb_profile_scale=sb_profile_scale,
            sb_asinh_softening=sb_asinh_softening,
        )

    ax_sb.set_ylabel(sb_ylabel)
    ax_sb.set_title("Surface brightness profile")
    ax_sb.grid(alpha=0.25)
    if zero_intensity_y is not None:
        ax_sb.axhline(
            zero_intensity_y,
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            label="I=0",
        )
    if invert_sb_axis:
        ax_sb.invert_yaxis()
    sb_for_limits = y_sb[sb_valid & np.isfinite(y_sb)]
    if sb_for_limits.size > 0:
        if zero_intensity_y is not None and np.isfinite(zero_intensity_y):
            sb_for_limits = np.concatenate([sb_for_limits, [zero_intensity_y]])
        set_axis_limits_from_finite_values(
            ax_sb,
            sb_for_limits,
            margin_fraction=0.06,
            min_margin=0.2,
            invert=invert_sb_axis,
        )
    handles, labels = ax_sb.get_legend_handles_labels()
    if handles:
        ax_sb.legend(handles[:8], labels[:8], loc="upper right", fontsize=14, ncol=1)

    # 2. Center offset
    ax_cen = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_cen,
        x_axis[valid_mask],
        i_dx[valid_mask],
        i_stop[valid_mask],
        y_errors=i_x0_err[valid_mask],
        marker_face="filled",
        label_prefix="dx ",
        monochrome=True,
    )
    plot_profile_by_stop_code(
        ax_cen,
        x_axis[valid_mask],
        i_dy[valid_mask],
        i_stop[valid_mask],
        y_errors=i_y0_err[valid_mask],
        marker_face="open",
        label_prefix="dy ",
        monochrome=True,
    )
    ax_cen.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_cen.set_ylabel("center offset [pix]")
    ax_cen.grid(alpha=0.25)
    centroid_legend = [
        Line2D([], [], marker="o", linestyle="None", color="black", markerfacecolor="black", markersize=4.8, label="X"),
        Line2D([], [], marker="o", linestyle="None", color="black", markerfacecolor="none", markersize=4.8, label="Y"),
    ]
    ax_cen.legend(handles=centroid_legend, loc="upper right", frameon=True, fontsize=14, ncol=2)
    # Y-limits from data values only (exclude errorbars to preserve detail)
    cen_vals = np.concatenate([i_dx[valid_mask], i_dy[valid_mask]])
    set_axis_limits_from_finite_values(ax_cen, cen_vals, margin_fraction=0.08, min_margin=0.3)

    # 3. Axis ratio (b/a)
    ax_ba = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_ba,
        x_axis[valid_mask],
        i_ba[valid_mask],
        i_stop[valid_mask],
        y_errors=i_eps_err[valid_mask],
        marker_face="filled",
        monochrome=True,
    )
    ba_for_limits = i_ba[valid_mask & np.isfinite(i_ba)]
    set_axis_limits_from_finite_values(
        ax_ba,
        ba_for_limits,
        margin_fraction=0.08,
        min_margin=0.03,
        lower_clip=0.0,
        upper_clip=1.0,
    )
    ax_ba.set_ylabel("axis ratio")
    ax_ba.grid(alpha=0.25)

    # 4. Position angle
    ax_pa = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1

    plot_profile_by_stop_code(
        ax_pa,
        x_axis[valid_mask],
        i_pa_deg[valid_mask],
        i_stop[valid_mask],
        y_errors=i_pa_err_deg[valid_mask],
        marker_face="filled",
        monochrome=True,
    )
    pa_for_limits = i_pa_deg[valid_mask & np.isfinite(i_pa_deg) & (i_stop == 0)]
    if pa_for_limits.size > 1:
        pa_low, pa_high = robust_limits(pa_for_limits, 3, 97)
        pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
        ax_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.grid(alpha=0.25)

    # 5. Harmonics (A3/B3, A4/B4) — optional
    ax_harm = None
    if has_harmonics:
        ax_harm = fig.add_subplot(right[panel_idx], sharex=ax_sb)
        panel_idx += 1

        ax_harm.scatter(
            x_axis[valid_mask],
            i_a3n[valid_mask],
            s=20,
            marker="o",
            facecolors="#1f77b4",
            edgecolors="#1f77b4",
            alpha=0.7,
            label=r"$a_3$",
        )
        ax_harm.scatter(
            x_axis[valid_mask],
            i_b3n[valid_mask],
            s=20,
            marker="s",
            facecolors="none",
            edgecolors="#1f77b4",
            alpha=0.7,
            label=r"$b_3$",
        )
        ax_harm.scatter(
            x_axis[valid_mask],
            i_a4n[valid_mask],
            s=20,
            marker="^",
            facecolors="#d62728",
            edgecolors="#d62728",
            alpha=0.7,
            label=r"$a_4$",
        )
        ax_harm.scatter(
            x_axis[valid_mask],
            i_b4n[valid_mask],
            s=20,
            marker="D",
            facecolors="none",
            edgecolors="#d62728",
            alpha=0.7,
            label=r"$b_4$",
        )
        harm_values = np.concatenate(
            [
                i_a3n[valid_mask],
                i_b3n[valid_mask],
                i_a4n[valid_mask],
                i_b4n[valid_mask],
            ]
        )
        set_axis_limits_from_finite_values(
            ax_harm,
            harm_values,
            margin_fraction=0.10,
            min_margin=0.005,
        )
        ax_harm.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax_harm.set_ylabel(r"$a_n, b_n$")
        ax_harm.legend(loc="upper right", fontsize=12, ncol=4)
        ax_harm.grid(alpha=0.25)

    # 6. Curve of growth — optional
    ax_cog = None
    if has_cog:
        ax_cog = fig.add_subplot(right[panel_idx], sharex=ax_sb)
        panel_idx += 1

        cog_values = get_arr("cog")
        plot_profile_by_stop_code(
            ax_cog,
            x_axis[valid_mask],
            cog_values[valid_mask],
            i_stop[valid_mask],
            marker_face="filled",
            label_prefix="fit ",
            monochrome=True,
        )
        finite_cog = cog_values[valid_mask & np.isfinite(cog_values) & (cog_values > 0)]
        if finite_cog.size > 0 and np.all(finite_cog > 0):
            ax_cog.set_yscale("log")
        ax_cog.set_ylabel("CoG flux")
        ax_cog.grid(alpha=0.25)

    # X-axis label on bottom panel only, suppress tick labels on others
    all_right_axes = [ax_sb, ax_cen, ax_ba, ax_pa]
    if ax_harm is not None:
        all_right_axes.append(ax_harm)
    if ax_cog is not None:
        all_right_axes.append(ax_cog)

    for ax in all_right_axes[:-1]:
        ax.tick_params(labelbottom=False)
    all_right_axes[-1].set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")

    # X-limit with right margin
    if np.any(valid_mask):
        set_x_limits_with_right_margin(all_right_axes[-1], x_axis[valid_mask])

    # --- Final layout adjustment and save --------------------------------------
    fig.subplots_adjust(left=0.025, right=0.992, bottom=0.05, top=0.940, wspace=0.18)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved QA figure to {filename}")


# ---------------------------------------------------------------------------
# Extended QA figure with per-order harmonic amplitude panels
# ---------------------------------------------------------------------------


def plot_qa_summary_extended(
    title,
    image,
    isoster_model,
    isoster_res,
    *,
    harmonic_orders: list[int] | None = None,
    harmonic_mode: str = "coefficients",
    relative_residual: bool = False,
    mask=None,
    filename="qa_summary_extended.png",
    sb_zeropoint=None,
    pixel_scale_arcsec=None,
    sb_profile_scale="log10",
    sb_asinh_softening=None,
):
    """QA figure with extended harmonic visualization panels.

    Extends :func:`plot_qa_summary` with two harmonic-specific panels:

    * **Odd harmonics** — orders 3, 5, 7 (if present)
    * **Even harmonics** — orders 4, 6 (if present)

    Parameters
    ----------
    title : str
        Figure title.
    image : 2D array
        Original input image.
    isoster_model : 2D array
        Reconstructed 2D model from isoster isophotes.
    isoster_res : list of dict
        Isoster fitting results.
    harmonic_orders : list of int or None
        Which harmonic orders to show.  Auto-detected from the
        available ``a3``/``b3`` … ``a7``/``b7`` keys when None.
    harmonic_mode : str
        ``'coefficients'`` (default) shows individual ``a_n`` (filled) and
        ``b_n`` (open) per order.  ``'amplitude'`` shows
        ``A_n = sqrt(a_n^2 + b_n^2)`` per order.
    relative_residual : bool
        When False (default), the residual map shows ``data - model``
        (absolute).  When True, shows ``100 * (model - data) / data``
        (fractional percent).
    mask : 2D bool array, optional
        Bad-pixel mask (True = masked) for the data panel overlay.
    filename : str
        Output path.
    sb_zeropoint : float, optional
        AB magnitude zeropoint. Must be passed together with
        ``pixel_scale_arcsec``; see :func:`plot_qa_summary` for the
        formula and rationale.
    pixel_scale_arcsec : float, optional
        Pixel size in arcsec. Must accompany ``sb_zeropoint``.
    sb_profile_scale : {"log10", "asinh", "arcsinh"}
        Surface-brightness profile y-scale.
    sb_asinh_softening : float, optional
        Positive softening scale for ``sb_profile_scale="asinh"``.
    """
    _validate_sb_inputs(sb_zeropoint, pixel_scale_arcsec)
    configure_qa_plot_style()

    def get_arr(key, default=np.nan):
        return np.array([r.get(key, default) for r in isoster_res])

    i_sma = get_arr("sma")
    i_intens = get_arr("intens")
    i_intens_err = get_arr("intens_err")
    i_eps = get_arr("eps")
    i_eps_err = get_arr("eps_err")
    i_pa = get_arr("pa")
    i_pa_err = get_arr("pa_err")
    i_x0 = get_arr("x0")
    i_x0_err = get_arr("x0_err")
    i_y0 = get_arr("y0")
    i_y0_err = get_arr("y0_err")
    i_stop = np.nan_to_num(get_arr("stop_code", 0), nan=0).astype(int)  # P4: explicit NaN stop codes read as 0

    # Determine which harmonic orders are present
    if harmonic_orders is None:
        harmonic_orders = []
        for order in [3, 4, 5, 6, 7]:
            key = f"a{order}"
            if any(np.isfinite(r.get(key, np.nan)) for r in isoster_res):
                harmonic_orders.append(order)

    # Pre-load harmonic arrays for all orders
    harm_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for order in harmonic_orders:
        harm_data[order] = (get_arr(f"a{order}"), get_arr(f"b{order}"))

    # Split into odd and even orders
    odd_orders = [o for o in harmonic_orders if o % 2 == 1]
    even_orders = [o for o in harmonic_orders if o % 2 == 0]

    x_axis = i_sma**0.25
    valid_mask = np.isfinite(x_axis) & (i_sma > 1.0)

    i_ba = 1.0 - i_eps
    i_pa_deg = normalize_pa_degrees(np.degrees(i_pa))
    i_pa_err_deg = np.degrees(i_pa_err)

    median_x0 = np.nanmedian(i_x0[valid_mask]) if np.any(valid_mask) else 0.0
    median_y0 = np.nanmedian(i_y0[valid_mask]) if np.any(valid_mask) else 0.0
    i_dx = i_x0 - median_x0
    i_dy = i_y0 - median_y0

    # 6 panels: SB, centroid, b/a, PA, odd harmonics, even harmonics
    n_panels = 6
    height_ratios = [2.2, 1.0, 1.0, 1.0, 1.4, 1.4]

    fig = plt.figure(figsize=(13.6, 13.0))
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 2.01],
        wspace=0.27,
    )
    left = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        hspace=0.10,
        wspace=-0.20,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        n_panels,
        1,
        subplot_spec=outer[1],
        height_ratios=height_ratios,
        hspace=0.0,
    )
    fig.suptitle(title, fontsize=20, y=0.989)

    # --- Left panels: data / model / residual ---------------------------------
    arc_params = derive_arcsinh_parameters(image)

    _plot_arcsinh_image_panel(
        fig,
        left[0, 0],
        left[0, 1],
        image,
        arc_params,
        title="Data",
        title_style="qa",
        title_color="w",
        title_pos=(0.15, 0.9),
        mask=mask,
        overlay_isos=isoster_res,
        overlay_step=max(1, len(isoster_res) // 15),
        overlay_width=1.2,
        cbar_label=r"arcsinh((data $-$ p0.05) / scale)",
        show_xy_labels=True,
        show_ticks=True,
    )

    _plot_arcsinh_image_panel(
        fig,
        left[1, 0],
        left[1, 1],
        isoster_model,
        arc_params,
        title="Model",
        title_style="qa",
        title_color="w",
        title_pos=(0.15, 0.9),
        contours=True,
        cbar_label=r"arcsinh((model $-$ p0.05) / scale)",
        show_xy_labels=True,
        show_ticks=True,
    )

    _plot_residual_panel(
        fig,
        left[2, 0],
        left[2, 1],
        image,
        isoster_model,
        "Residual",
        relative_residual=relative_residual,
        title_style="qa",
        title_color="k",
        title_pos=(0.18, 0.9),
        show_xy_labels=True,
        show_ticks=True,
        cbar_fontsize=None,
    )

    # --- Right panels: 1-D profiles ------------------------------------------
    panel_idx = 0

    # 1. Surface brightness
    ax_sb = fig.add_subplot(right[panel_idx])
    panel_idx += 1
    y_sb, y_sb_err, sb_ylabel, invert_sb_axis, zero_intensity_y = transform_sb_profile(
        i_intens,
        i_intens_err,
        sb_zeropoint=sb_zeropoint,
        pixel_scale_arcsec=pixel_scale_arcsec,
        sb_profile_scale=sb_profile_scale,
        sb_asinh_softening=sb_asinh_softening,
    )
    sb_valid = valid_mask & np.isfinite(y_sb)
    plot_profile_by_stop_code(
        ax_sb,
        x_axis[sb_valid],
        y_sb[sb_valid],
        i_stop[sb_valid],
        y_errors=y_sb_err[sb_valid],
        marker_face="filled",
        monochrome=True,
    )
    ax_sb.set_ylabel(sb_ylabel)
    ax_sb.set_title("Surface brightness profile")
    ax_sb.grid(alpha=0.25)
    if zero_intensity_y is not None:
        ax_sb.axhline(
            zero_intensity_y,
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            label="I=0",
        )
    if invert_sb_axis:
        ax_sb.invert_yaxis()
    sb_for_limits = y_sb[sb_valid & np.isfinite(y_sb)]
    if sb_for_limits.size > 0:
        if zero_intensity_y is not None and np.isfinite(zero_intensity_y):
            sb_for_limits = np.concatenate([sb_for_limits, [zero_intensity_y]])
        set_axis_limits_from_finite_values(
            ax_sb,
            sb_for_limits,
            margin_fraction=0.06,
            min_margin=0.2,
            invert=invert_sb_axis,
        )
    handles, labels = ax_sb.get_legend_handles_labels()
    if handles:
        ax_sb.legend(handles[:8], labels[:8], loc="upper right", fontsize=12, ncol=1)

    # 2. Center offset
    ax_cen = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_cen,
        x_axis[valid_mask],
        i_dx[valid_mask],
        i_stop[valid_mask],
        y_errors=i_x0_err[valid_mask],
        marker_face="filled",
        label_prefix="dx ",
        monochrome=True,
    )
    plot_profile_by_stop_code(
        ax_cen,
        x_axis[valid_mask],
        i_dy[valid_mask],
        i_stop[valid_mask],
        y_errors=i_y0_err[valid_mask],
        marker_face="open",
        label_prefix="dy ",
        monochrome=True,
    )
    ax_cen.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_cen.set_ylabel("center [pix]")
    ax_cen.grid(alpha=0.25)
    # Y-limits from data values only (exclude errorbars to preserve detail)
    cen_vals = np.concatenate([i_dx[valid_mask], i_dy[valid_mask]])
    set_axis_limits_from_finite_values(ax_cen, cen_vals, margin_fraction=0.08, min_margin=0.3)

    # 3. Axis ratio
    ax_ba = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_ba,
        x_axis[valid_mask],
        i_ba[valid_mask],
        i_stop[valid_mask],
        y_errors=i_eps_err[valid_mask],
        marker_face="filled",
        monochrome=True,
    )
    ba_for_limits = i_ba[valid_mask & np.isfinite(i_ba)]
    set_axis_limits_from_finite_values(
        ax_ba,
        ba_for_limits,
        margin_fraction=0.08,
        min_margin=0.03,
        lower_clip=0.0,
        upper_clip=1.0,
    )
    ax_ba.set_ylabel("axis ratio")
    ax_ba.grid(alpha=0.25)

    # 4. PA
    ax_pa = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    plot_profile_by_stop_code(
        ax_pa,
        x_axis[valid_mask],
        i_pa_deg[valid_mask],
        i_stop[valid_mask],
        y_errors=i_pa_err_deg[valid_mask],
        marker_face="filled",
        monochrome=True,
    )
    pa_for_limits = i_pa_deg[valid_mask & np.isfinite(i_pa_deg) & (i_stop == 0)]
    if pa_for_limits.size > 1:
        pa_low, pa_high = robust_limits(pa_for_limits, 3, 97)
        pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
        ax_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)
    ax_pa.set_ylabel("PA [deg]")
    ax_pa.grid(alpha=0.25)

    # Colour cycle for harmonic orders
    order_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    order_markers = ["o", "^", "s", "D", "v"]

    def _plot_harmonic_panel(ax, orders, panel_label):
        """Plot harmonic data on a panel for the given orders."""
        all_values = []
        for idx_o, order in enumerate(orders):
            if order not in harm_data:
                continue
            an, bn = harm_data[order]
            col = order_colors[idx_o % len(order_colors)]
            mrk = order_markers[idx_o % len(order_markers)]

            if harmonic_mode == "amplitude":
                # A_n = sqrt(a_n^2 + b_n^2). The stored a_n / b_n are already
                # Bender-normalized by compute_deviations (divided by
                # sma * abs(gradient)), so A_n is dimensionless as it stands and
                # must not be normalized again -- see the note in the module
                # docstring of isoster/_shared.py.
                amplitude = np.sqrt(np.where(np.isfinite(an), an, 0.0) ** 2 + np.where(np.isfinite(bn), bn, 0.0) ** 2)
                amplitude[~(np.isfinite(an) & np.isfinite(bn))] = np.nan
                show = valid_mask & np.isfinite(amplitude)
                if np.any(show):
                    ax.scatter(
                        x_axis[show],
                        amplitude[show],
                        s=18,
                        marker=mrk,
                        facecolors=col,
                        edgecolors=col,
                        alpha=0.75,
                        label=f"$n={{{order}}}$",
                    )
                    all_values.append(amplitude[show])
            else:
                # Coefficients mode: a_n (filled) and b_n (open)
                show_a = valid_mask & np.isfinite(an)
                show_b = valid_mask & np.isfinite(bn)
                if np.any(show_a):
                    ax.scatter(
                        x_axis[show_a],
                        an[show_a],
                        s=18,
                        marker=mrk,
                        facecolors=col,
                        edgecolors=col,
                        alpha=0.75,
                        label=f"$a_{{{order}}}$",
                    )
                    all_values.append(an[show_a])
                if np.any(show_b):
                    ax.scatter(
                        x_axis[show_b],
                        bn[show_b],
                        s=18,
                        marker=mrk,
                        facecolors="none",
                        edgecolors=col,
                        linewidths=0.9,
                        alpha=0.75,
                        label=f"$b_{{{order}}}$",
                    )
                    all_values.append(bn[show_b])

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        if all_values:
            cat = np.concatenate(all_values)
            set_axis_limits_from_finite_values(
                ax,
                cat,
                margin_fraction=0.12,
                min_margin=0.001,
            )

        # Y-axis label
        if harmonic_mode == "amplitude":
            ax.set_ylabel(f"$\\sqrt{{a_n^2 + b_n^2}}$ ({panel_label})")
        else:
            ax.set_ylabel(f"$a_n, b_n$ ({panel_label})")

        ax.legend(loc="upper right", fontsize=9, ncol=min(len(orders) * 2, 6))
        ax.grid(alpha=0.25)

    # 5. Odd harmonics panel
    ax_odd = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    if odd_orders:
        _plot_harmonic_panel(ax_odd, odd_orders, "odd")
    else:
        ax_odd.set_ylabel("$a_n, b_n$ (odd)")
        ax_odd.text(
            0.5, 0.5, "no odd orders", color="gray", fontsize=10, transform=ax_odd.transAxes, ha="center", va="center"
        )
        ax_odd.grid(alpha=0.25)

    # 6. Even harmonics panel
    ax_even = fig.add_subplot(right[panel_idx], sharex=ax_sb)
    panel_idx += 1
    if even_orders:
        _plot_harmonic_panel(ax_even, even_orders, "even")
    else:
        ax_even.set_ylabel("$a_n, b_n$ (even)")
        ax_even.text(
            0.5, 0.5, "no even orders", color="gray", fontsize=10, transform=ax_even.transAxes, ha="center", va="center"
        )
        ax_even.grid(alpha=0.25)

    # X-axis housekeeping
    all_right_axes = [ax_sb, ax_cen, ax_ba, ax_pa, ax_odd, ax_even]
    for ax in all_right_axes[:-1]:
        ax.tick_params(labelbottom=False)
    all_right_axes[-1].set_xlabel(r"SMA$^{0.25}$ (pixel$^{0.25}$)")
    if np.any(valid_mask):
        set_x_limits_with_right_margin(all_right_axes[-1], x_axis[valid_mask])

    fig.subplots_adjust(left=0.025, right=0.992, bottom=0.04, top=0.940, wspace=0.18)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved extended QA figure to {filename}")


# ---------------------------------------------------------------------------
# Multi-method comparison QA figure
# ---------------------------------------------------------------------------


def _scatter_by_stop_code_in_method_color(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    stop_codes: np.ndarray,
    base_color: str,
    y_errors: np.ndarray | None = None,
    label: str = "",
    marker_size: float = 22.0,
    label_stop_codes: bool = True,
) -> None:
    """Scatter points using stop-code markers in a single method color.

    Stop=0 points are filled; non-zero stop codes use open markers with
    the same base color.  This distinguishes convergence quality while
    keeping method identity clear via color.

    Parameters
    ----------
    label_stop_codes : bool
        If True, each stop code gets its own legend entry (e.g.
        "isoster", "isoster (stop=2)").  If False, only the first
        group (stop=0) gets the method label; non-zero groups are
        plotted but unlabeled in the legend.
    """
    unique_codes = sorted(
        {int(c) for c in stop_codes[np.isfinite(stop_codes)]},
        key=lambda c: (0, 0) if c == 0 else (1, c) if c > 0 else (2, abs(c)),
    )
    for code in unique_codes:
        mask = stop_codes == code
        if not np.any(mask):
            continue
        marker = MONOCHROME_STOP_MARKERS.get(code, "o")
        face = base_color if code == 0 else "none"
        if label_stop_codes:
            code_label = f"{label} (stop={code})" if code != 0 else label
        else:
            # Only label the first group (stop=0) with the method name
            code_label = label if code == 0 else None

        if y_errors is not None:
            errs = np.asarray(y_errors[mask], dtype=float)
            errs[errs < 0] = np.nan
            axis.errorbar(
                x[mask],
                y[mask],
                yerr=errs,
                fmt=marker,
                color=base_color,
                mfc=face,
                mec=base_color,
                markersize=4.2,
                capsize=1.5,
                linewidth=0.6,
                alpha=0.8,
                label=code_label,
                zorder=3,
            )
        else:
            axis.scatter(
                x[mask],
                y[mask],
                s=marker_size,
                marker=marker,
                facecolors=face,
                edgecolors=base_color,
                linewidths=0.9,
                alpha=0.8,
                label=code_label,
                zorder=3,
            )


def _build_isos_for_overlay(prof: dict[str, np.ndarray]) -> list[dict]:
    """Reconstruct minimal isophote dicts from profile arrays for overlay.

    Includes harmonic coefficients (a3, b3, a4, b4, ...) when present
    so that ``draw_isophote_overlays`` can render the perturbed shape.
    """
    if "x0" not in prof or "y0" not in prof:
        return []
    n = len(prof["sma"])
    # Detect harmonic keys in the profile dict
    harm_keys = [k for k in prof if len(k) >= 2 and k[0] in ("a", "b") and k[1:].isdigit() and int(k[1:]) >= 3]
    isos = []
    for i in range(n):
        iso = {
            "sma": float(prof["sma"][i]),
            "x0": float(prof["x0"][i]),
            "y0": float(prof["y0"][i]),
            "eps": float(prof["eps"][i]),
            "pa": float(prof["pa"][i]),
            "stop_code": int(prof["stop_codes"][i]) if "stop_codes" in prof else 0,
        }
        if "use_eccentric_anomaly" in prof:
            ea_val = prof["use_eccentric_anomaly"][i]
            # Missing values arrive as NaN and bool(NaN) is True, which
            # silently switched rows to EA contours; isoster's default is
            # phi-mode (review P4). Skip the key so the downstream default
            # (False) applies.
            if ea_val is not None and np.isfinite(ea_val):
                iso["use_eccentric_anomaly"] = bool(ea_val)
        if "tool" in prof:
            tool_val = prof["tool"][i]
            # str(NaN) == "nan" fell through to the psi default in
            # select_contour_fn; only keep real tool names.
            if isinstance(tool_val, str):
                iso["tool"] = tool_val
        for k in harm_keys:
            iso[k] = float(prof[k][i])
        isos.append(iso)
    return isos


def _panel_title(ax, text, *, style, color, pos):
    """Draw the corner title of an image panel in one of the two house styles.

    ``style="qa"`` is the large centered title used by plot_qa_summary and
    the extended figure; ``style="small"`` is the compact top-left title
    used by the comparison figure.
    """
    if style == "qa":
        ax.text(
            pos[0],
            pos[1],
            text,
            fontsize=18,
            color=color,
            transform=ax.transAxes,
            ha="center",
            va="center",
            weight="bold",
            alpha=0.85,
        )
    else:
        ax.text(
            pos[0],
            pos[1],
            text,
            fontsize=11,
            color=color,
            fontweight="bold",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )


def _plot_arcsinh_image_panel(
    fig,
    gs_slot,
    gs_cbar_slot,
    image,
    arc_params,
    *,
    title,
    title_style="small",
    title_color="white",
    title_pos=(0.03, 0.95),
    mask=None,
    overlay_isos=None,
    overlay_step=5,
    overlay_color=None,
    overlay_width=1.0,
    overlay_alpha=0.8,
    contours=False,
    cbar_label=None,
    show_scale_bar=False,
    show_xy_labels=False,
    show_ticks=False,
):
    """Render one arcsinh-stretched image panel (data or model) with extras.

    Shared by plot_qa_summary, the extended figure, and the comparison
    figure (which used to carry three drifting copies of this block).
    ``arc_params`` is the ``(low, high, scale, vmax)`` tuple from
    :func:`derive_arcsinh_parameters`, shared across all panels of one
    figure so they use the same stretch.
    """
    low, high, scale, vmax = arc_params
    display, disp_vmin, disp_vmax = make_arcsinh_display_from_parameters(
        image,
        low=low,
        high=high,
        scale=scale,
        vmax=vmax,
    )
    ax = fig.add_subplot(gs_slot)
    handle = ax.imshow(
        display,
        origin="lower",
        cmap="viridis",
        vmin=disp_vmin,
        vmax=disp_vmax,
        interpolation="none",
    )
    if mask is not None:
        mask_overlay = np.zeros((*image.shape, 4))
        mask_overlay[mask] = [1, 0, 0, 0.4]
        ax.imshow(mask_overlay, origin="lower")
    if contours:
        overlay_model_contours(ax, image)
    if overlay_isos:
        draw_isophote_overlays(
            ax,
            overlay_isos,
            step=overlay_step,
            line_width=overlay_width,
            alpha=overlay_alpha,
            edge_color=overlay_color,
        )
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    _panel_title(ax, title, style=title_style, color=title_color, pos=title_pos)
    if show_xy_labels:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
    if show_scale_bar:
        # Scale bar: ~1/10 of the image size, rounded up to nearest 10
        img_size = max(image.shape)
        bar_length = int(np.ceil(img_size / 100.0)) * 10
        bar_x0 = image.shape[1] * 0.05
        bar_y0 = image.shape[0] * 0.05
        ax.plot(
            [bar_x0, bar_x0 + bar_length],
            [bar_y0, bar_y0],
            color="white",
            linewidth=2.5,
            solid_capstyle="butt",
        )
        ax.text(
            bar_x0 + bar_length / 2,
            bar_y0 + image.shape[0] * 0.03,
            f"{bar_length} pix",
            color="white",
            fontsize=9,
            ha="center",
            va="bottom",
        )

    ax_cbar = fig.add_subplot(gs_cbar_slot)
    if cbar_label is not None:
        fig.colorbar(handle, cax=ax_cbar).set_label(cbar_label)
    else:
        fig.colorbar(handle, cax=ax_cbar)
    return ax


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------


def _compute_residual_map(
    image: np.ndarray,
    model: np.ndarray,
    relative: bool = False,
) -> tuple[np.ndarray, str]:
    """Compute residual map and its colorbar label.

    Parameters
    ----------
    relative : bool
        If True, return ``100 * (model - data) / data`` (fractional %).
        If False (default), return ``data - model`` (absolute).
    """
    if relative:
        residual = compute_fractional_residual_percent(image, model)
        label = latex_safe_text("(model - data) / data [%]")
    else:
        residual = np.where(np.isfinite(image), image - model, np.nan)
        label = "data - model"
    return residual, label


def _plot_residual_panel(
    fig,
    gs_slot,
    gs_cbar_slot,
    image: np.ndarray,
    model: np.ndarray,
    panel_title: str,
    relative_residual: bool = False,
    overlay_isos: list[dict] | None = None,
    overlay_step: int = 5,
    overlay_color: str = "white",
    overlay_width: float = 1.0,
    title_style="small",
    title_color="black",
    title_pos=(0.03, 0.95),
    show_xy_labels=False,
    show_ticks=False,
    cbar_fontsize=8,
) -> None:
    """Draw a single residual image panel with optional isophote overlays.

    Used by all three QA figures (which used to carry three drifting
    copies of this block). ``title_style="qa"`` selects the large centered
    title of plot_qa_summary / the extended figure; ``"small"`` selects
    the comparison figure's compact top-left title.
    """
    residual, cbar_label = _compute_residual_map(
        image,
        model,
        relative=relative_residual,
    )
    abs_vals = np.abs(residual[np.isfinite(residual)])
    # Floor (0.05) and cap (8.0) the color range only in fractional
    # (percent) mode; in absolute mode an arbitrary floor flattens
    # low-flux residual maps (P4).
    res_limit = float(
        np.clip(
            np.nanpercentile(abs_vals, 99.0) if abs_vals.size else 1.0,
            0.05 if relative_residual else None,
            8.0 if relative_residual else None,
        )
    )

    ax = fig.add_subplot(gs_slot)
    handle = ax.imshow(
        residual,
        origin="lower",
        cmap="coolwarm",
        vmin=-res_limit,
        vmax=res_limit,
        interpolation="nearest",
    )
    if overlay_isos:
        draw_isophote_overlays(
            ax,
            overlay_isos,
            step=overlay_step,
            line_width=overlay_width,
            alpha=0.7,
            edge_color=overlay_color,
        )
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    _panel_title(ax, panel_title, style=title_style, color=title_color, pos=title_pos)
    if show_xy_labels:
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    ax_cbar = fig.add_subplot(gs_cbar_slot)
    if cbar_fontsize is not None:
        fig.colorbar(handle, cax=ax_cbar).set_label(cbar_label, fontsize=cbar_fontsize)
    else:
        fig.colorbar(handle, cax=ax_cbar).set_label(cbar_label)


def plot_comparison_qa_figure(
    image: np.ndarray,
    profiles: dict[str, dict[str, np.ndarray]],
    title: str = "",
    output_path: str | Path = "qa_comparison.png",
    *,
    models: dict[str, np.ndarray] | None = None,
    mask: np.ndarray | None = None,
    method_styles: dict[str, dict] | None = None,
    relative_residual: bool = False,
    sb_zeropoint: float | None = None,
    pixel_scale_arcsec: float | None = None,
    sb_profile_scale: str = "log10",
    sb_asinh_softening: float | None = None,
    dpi: int = 150,
) -> None:
    """Multi-method comparison QA figure with 2D images and 1D profiles.

    Automatically selects one of three layout modes based on the number
    of methods that provide profiles:

    **Mode 1 — Solo (isoster only):**
    Left column: original image with isophote overlays (+ mask overlay),
    reconstructed 2D model, residual.

    **Mode 2 — One-on-one (isoster + one other):**
    Left column: original image (+ mask), isoster residual with isophote
    overlays, other method's residual with its isophote overlays.

    **Mode 3 — Three-way (isoster + both):**
    Left column: original image (+ mask), isoster residual + overlays,
    photutils residual + overlays, autoprof residual + overlays.

    Right column (all modes): SB (with errorbars), relative SB diff,
    ellipticity, PA, centroid offset — sharing SMA^0.25 x-axis.

    Parameters
    ----------
    image : 2D ndarray
        Original galaxy image.
    profiles : dict[str, dict[str, ndarray]]
        Method name -> profile arrays dict (from ``build_method_profile``).
        Required keys per profile: sma, x_axis, intens, eps, pa.
        Optional: stop_codes, x0, y0, intens_err, rms, runtime_seconds.
    title : str
        Figure title.
    output_path : str or Path
        Output file path for the saved figure.
    models : dict[str, 2D ndarray] or None
        Method name -> 2D reconstructed model.
    mask : 2D bool ndarray or None
        Bad-pixel mask (True = masked).  Overlaid on the data image panel.
    method_styles : dict or None
        Override default ``METHOD_STYLES``.
    relative_residual : bool
        If True, residual panels show ``(model - data) / data`` [%].
        If False (default), show ``data - model`` (absolute).
    sb_zeropoint, pixel_scale_arcsec
        Optional calibration for the SB profile panel.
    sb_profile_scale : {"log10", "asinh", "arcsinh"}
        Surface-brightness profile y-scale.
    sb_asinh_softening : float, optional
        Positive softening scale for ``sb_profile_scale="asinh"``.
    dpi : int
        Figure resolution.
    """
    _validate_sb_inputs(sb_zeropoint, pixel_scale_arcsec)
    configure_qa_plot_style()
    plt.rcParams["text.usetex"] = False

    styles = dict(METHOD_STYLES)
    if method_styles is not None:
        styles.update(method_styles)

    if models is None:
        models = {}

    available = [m for m in profiles if profiles[m] is not None]
    n_methods = len(available)

    # P4: derive the asinh softening once from the reference method so every
    # overlaid method shares one asinh scale — per-method resolution gave
    # each method its own scale and the I=0 line came from the last one.
    if sb_profile_scale in ("asinh", "arcsinh") and available:
        sb_asinh_softening = _resolve_asinh_softening(profiles[available[0]]["intens"], sb_asinh_softening)

    # --- Determine layout mode ---
    # Mode 1 (solo): 3 left rows (image, model, residual)
    # Mode 2 (one-on-one): 3 left rows (image, isoster residual, other residual)
    # Mode 3 (three-way): 4 left rows (image, isoster res, phot res, autoprof res)
    if n_methods <= 1:
        n_left_rows = 3  # image, model, residual
    elif n_methods == 2:
        n_left_rows = 3  # image, 2x residual
    else:
        n_left_rows = 1 + n_methods  # image + one residual per method

    n_right_rows = 5
    fig_height = max(12, n_left_rows * 3.2)

    fig = plt.figure(figsize=(15, fig_height), dpi=dpi)
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.8],
        wspace=0.30,
        top=0.95,
    )
    left = gridspec.GridSpecFromSubplotSpec(
        n_left_rows,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.04],
        hspace=0.12,
        wspace=0.02,
    )
    right = gridspec.GridSpecFromSubplotSpec(
        n_right_rows,
        1,
        subplot_spec=outer[1],
        height_ratios=[2.5, 1.0, 1.0, 1.0, 1.0],
        hspace=0.0,
    )

    # --- Title (runtime shown in SB panel instead) ---
    fig.suptitle(title, fontsize=16, y=0.975)

    # Collect runtime lines for later display in SB panel
    _runtime_lines = []
    for method_name in available:
        prof = profiles[method_name]
        style = styles.get(method_name, {"label": method_name})
        label = style.get("label", method_name)
        color = style.get("color", "black")
        rt = prof.get("runtime_seconds")
        if rt is not None:
            line = f"{label}: {float(rt):.2f}s"
            retries = prof.get("retries", 0)
            if retries > 0:
                line += f"; retry: {retries}"
            _runtime_lines.append((line, color))

    # --- Left column row 0: Original image ---
    arc_params = derive_arcsinh_parameters(image)

    # Solo mode: overlays from the (single) method on the data image
    img_overlay_isos = None
    img_overlay_step = 5
    img_overlay_width = 1.0
    img_overlay_color = "white"
    if n_methods <= 1:
        for method_name in available:
            prof = profiles[method_name]
            style = {"color": "black", **styles.get(method_name, {})}  # P3: default color for unknown methods
            isos = _build_isos_for_overlay(prof)
            if isos:
                img_overlay_isos = isos
                img_overlay_step = max(1, len(isos) // 15)
                img_overlay_width = style.get("overlay_width", 1.0)
                img_overlay_color = style.get("overlay_color", "white")

    _plot_arcsinh_image_panel(
        fig,
        left[0, 0],
        left[0, 1],
        image,
        arc_params,
        title="Data",
        title_style="small",
        title_color="white",
        title_pos=(0.03, 0.95),
        mask=mask,
        overlay_isos=img_overlay_isos,
        overlay_step=img_overlay_step,
        overlay_color=img_overlay_color,
        overlay_width=img_overlay_width,
        show_scale_bar=True,
    )

    # --- Left column: mode-dependent panels ---
    if n_methods <= 1 and available:
        # Mode 1 (solo): row 1 = model, row 2 = residual
        method_name = available[0]
        model = models.get(method_name)

        if model is not None:
            # Model panel
            _plot_arcsinh_image_panel(
                fig,
                left[1, 0],
                left[1, 1],
                model,
                arc_params,
                title="Model",
                title_style="small",
                title_color="white",
                title_pos=(0.03, 0.95),
                contours=True,
            )

            # Residual panel
            _plot_residual_panel(
                fig,
                left[2, 0],
                left[2, 1],
                image,
                model,
                panel_title="Residual",
                relative_residual=relative_residual,
            )
        else:
            # No model available — leave panels blank
            for row_idx in (1, 2):
                ax_blank = fig.add_subplot(left[row_idx, 0])
                ax_blank.text(
                    0.5,
                    0.5,
                    "no model",
                    color="gray",
                    fontsize=12,
                    transform=ax_blank.transAxes,
                    ha="center",
                    va="center",
                )
                ax_blank.set_axis_off()

    elif n_methods >= 2:
        # Mode 2/3: one residual panel per method with isophote overlays
        for row_idx, method_name in enumerate(available, start=1):
            model = models.get(method_name)
            style = {"color": "black", **styles.get(method_name, {})}  # P3: default color for unknown methods
            label = style.get("label", method_name)
            isos = _build_isos_for_overlay(profiles[method_name])
            overlay_step = max(1, len(isos) // 15) if isos else 5

            if model is not None:
                _plot_residual_panel(
                    fig,
                    left[row_idx, 0],
                    left[row_idx, 1],
                    image,
                    model,
                    panel_title=f"{label} residual",
                    relative_residual=relative_residual,
                    overlay_isos=isos,
                    overlay_step=overlay_step,
                    overlay_color=style.get("overlay_color", "white"),
                    overlay_width=style.get("overlay_width", 1.0),
                )
            else:
                ax_blank = fig.add_subplot(left[row_idx, 0])
                ax_blank.text(
                    0.5,
                    0.5,
                    f"{label}: no model",
                    color="gray",
                    fontsize=12,
                    transform=ax_blank.transAxes,
                    ha="center",
                    va="center",
                )
                ax_blank.set_axis_off()

    # --- Right column: 1D profiles ---
    if not available:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    all_x = np.concatenate([profiles[m]["x_axis"] for m in available])

    # Panel 0: Surface brightness (with errorbars and stop-code markers)
    ax_sb = fig.add_subplot(right[0])
    for method_name in available:
        prof = profiles[method_name]
        style = {"color": "black", **styles.get(method_name, {})}  # P3: default color for unknown methods
        x = prof["x_axis"]
        intens = prof["intens"]
        y, y_err, sb_ylabel, invert_sb_axis, zero_intensity_y = transform_sb_profile(
            intens,
            prof.get("intens_err"),
            sb_zeropoint=sb_zeropoint,
            pixel_scale_arcsec=pixel_scale_arcsec,
            sb_profile_scale=sb_profile_scale,
            sb_asinh_softening=sb_asinh_softening,
        )
        valid = np.isfinite(x) & np.isfinite(y)

        if "stop_codes" in prof:
            _scatter_by_stop_code_in_method_color(
                ax_sb,
                x[valid],
                y[valid],
                prof["stop_codes"][valid],
                base_color=style["color"],
                y_errors=y_err[valid] if y_err is not None else None,
                label=style.get("label", method_name),
                marker_size=22,
                label_stop_codes=(n_methods <= 1),
            )
        else:
            mfc = style["color"] if style.get("marker_face") == "filled" else "none"
            if y_err is not None:
                ax_sb.errorbar(
                    x[valid],
                    y[valid],
                    yerr=y_err[valid],
                    fmt=style.get("marker", "o"),
                    color=style["color"],
                    mfc=mfc,
                    mec=style["color"],
                    markersize=4.2,
                    capsize=1.5,
                    linewidth=0.6,
                    alpha=0.7,
                    label=style.get("label", method_name),
                    zorder=3,
                )
            else:
                ax_sb.scatter(
                    x[valid],
                    y[valid],
                    color=style["color"],
                    marker=style.get("marker", "o"),
                    facecolors=mfc,
                    edgecolors=style["color"],
                    s=22,
                    alpha=0.7,
                    label=style.get("label", method_name),
                    zorder=3,
                )

    ax_sb.set_ylabel(sb_ylabel)
    ax_sb.grid(alpha=0.25)
    if zero_intensity_y is not None:
        ax_sb.axhline(
            zero_intensity_y,
            color="0.25",
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            label="I=0",
        )
    ax_sb.legend(loc="upper right", fontsize=10)
    ax_sb.tick_params(labelbottom=False)
    set_x_limits_with_right_margin(ax_sb, all_x)
    # Data-driven Y-axis: range from data values, not error bars
    all_sb_vals = []
    for m in available:
        y_vals, *_ = transform_sb_profile(
            profiles[m]["intens"],
            profiles[m].get("intens_err"),
            sb_zeropoint=sb_zeropoint,
            pixel_scale_arcsec=pixel_scale_arcsec,
            sb_profile_scale=sb_profile_scale,
            sb_asinh_softening=sb_asinh_softening,
        )
        v = np.isfinite(y_vals)
        if np.any(v):
            all_sb_vals.append(y_vals[v])
    if all_sb_vals:
        all_sb = np.concatenate(all_sb_vals)
        if zero_intensity_y is not None and np.isfinite(zero_intensity_y):
            all_sb = np.concatenate([all_sb, [zero_intensity_y]])
        set_axis_limits_from_finite_values(ax_sb, all_sb, invert=invert_sb_axis)

    # Runtime annotation in bottom-left of SB panel
    if _runtime_lines:
        y_pos = 0.04
        for line_text, line_color in _runtime_lines:
            ax_sb.text(
                0.03,
                y_pos,
                line_text,
                color=line_color,
                fontsize=9,
                fontweight="bold",
                transform=ax_sb.transAxes,
                va="bottom",
                ha="left",
            )
            y_pos += 0.06

    # Panel 1: Relative SB difference (first method as reference)
    # Outliers beyond ±100% are shown as lower-limit markers (triangles)
    # and the Y-axis is capped at ±100% when such outliers exist.
    _DIFF_CAP = 100.0  # percent
    ax_diff = fig.add_subplot(right[1], sharex=ax_sb)
    has_outliers = False
    if n_methods >= 2:
        ref_method = available[0]
        ref_prof = profiles[ref_method]
        ref_sma = ref_prof["sma"]
        ref_intens = ref_prof["intens"]
        for method_name in available:
            if method_name == ref_method:
                continue
            prof = profiles[method_name]
            style = {"color": "black", **styles.get(method_name, {})}  # P3: default color for unknown methods
            from scipy.interpolate import interp1d

            valid_ref = np.isfinite(ref_intens) & (ref_intens > 0)
            if valid_ref.sum() < 2:
                continue
            interp_func = interp1d(
                ref_sma[valid_ref],
                ref_intens[valid_ref],
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            ref_at_method = interp_func(prof["sma"])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_diff = 100.0 * (prof["intens"] - ref_at_method) / ref_at_method
            valid = np.isfinite(rel_diff)
            mfc = style["color"] if style.get("marker_face") == "filled" else "none"
            base_marker = style.get("marker", "o")

            # Split into normal points and outliers (|dI/I| > cap)
            inlier = valid & (np.abs(rel_diff) <= _DIFF_CAP)
            outlier = valid & (np.abs(rel_diff) > _DIFF_CAP)

            if np.any(outlier):
                has_outliers = True

            # Normal points
            ax_diff.scatter(
                prof["x_axis"][inlier],
                rel_diff[inlier],
                color=style["color"],
                marker=base_marker,
                facecolors=mfc,
                edgecolors=style["color"],
                s=18,
                alpha=0.7,
                label=style.get("label", method_name),
                zorder=3,
            )
            # Outlier points: clamp to ±cap, show as triangle markers
            if np.any(outlier):
                clamped = np.clip(rel_diff[outlier], -_DIFF_CAP, _DIFF_CAP)
                # Triangle pointing in the direction of the outlier
                outlier_markers = np.where(rel_diff[outlier] > 0, "^", "v")
                for om in ["^", "v"]:
                    om_mask = outlier_markers == om
                    if np.any(om_mask):
                        ax_diff.scatter(
                            prof["x_axis"][outlier][om_mask],
                            clamped[om_mask],
                            color=style["color"],
                            marker=om,
                            facecolors=style["color"],
                            edgecolors=style["color"],
                            s=30,
                            alpha=0.5,
                            zorder=4,
                        )

        ax_diff.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax_diff.set_ylabel("dI/I [%]")
        ax_diff.legend(loc="best", fontsize=9)

        # Cap Y-axis when outliers exist
        if has_outliers:
            ax_diff.set_ylim(-_DIFF_CAP * 1.08, _DIFF_CAP * 1.08)
    else:
        ax_diff.set_ylabel("dI/I [%]")
        ax_diff.text(
            0.5,
            0.5,
            "single method",
            color="gray",
            fontsize=10,
            transform=ax_diff.transAxes,
            ha="center",
            va="center",
        )
    ax_diff.grid(alpha=0.25)
    ax_diff.tick_params(labelbottom=False)

    # Panel 2: Ellipticity (with error bars when available)
    ax_eps = fig.add_subplot(right[2], sharex=ax_sb)
    for method_name in available:
        prof = profiles[method_name]
        style = {"color": "black", **styles.get(method_name, {})}  # P3: default color for unknown methods
        eps_err = prof.get("eps_err")
        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        if eps_err is not None:
            if "stop_codes" in prof:
                _scatter_by_stop_code_in_method_color(
                    ax_eps,
                    prof["x_axis"],
                    prof["eps"],
                    prof["stop_codes"],
                    base_color=style["color"],
                    y_errors=eps_err,
                    marker_size=18,
                    label_stop_codes=False,
                    label=style.get("label", method_name),
                )
            else:
                ax_eps.errorbar(
                    prof["x_axis"],
                    prof["eps"],
                    yerr=eps_err,
                    fmt=style.get("marker", "o"),
                    color=style["color"],
                    mfc=mfc,
                    mec=style["color"],
                    markersize=4.2,
                    capsize=1.5,
                    linewidth=0.6,
                    alpha=0.7,
                    zorder=3,
                )
        else:
            ax_eps.scatter(
                prof["x_axis"],
                prof["eps"],
                color=style["color"],
                marker=style.get("marker", "o"),
                facecolors=mfc,
                edgecolors=style["color"],
                s=18,
                alpha=0.7,
                zorder=3,
            )
    ax_eps.set_ylabel("Ellipticity")
    ax_eps.grid(alpha=0.25)
    ax_eps.tick_params(labelbottom=False)
    # Data-driven Y-axis: limits from data values only, not error bars
    all_eps = np.concatenate([profiles[m]["eps"] for m in available])
    set_axis_limits_from_finite_values(
        ax_eps,
        all_eps,
        margin_fraction=0.08,
        min_margin=0.03,
        lower_clip=0.0,
        upper_clip=1.0,
    )

    # Panel 3: PA (normalized with cross-method anchoring, with error bars)
    ax_pa = fig.add_subplot(right[3], sharex=ax_sb)
    all_pa_norm = []
    ref_pa_median = None
    for method_name in available:
        prof = profiles[method_name]
        style = {"color": "black", **styles.get(method_name, {})}  # P3: default color for unknown methods
        pa_deg = np.degrees(prof["pa"])

        # First method: normalize freely; subsequent methods: anchor to
        # the first method's median PA to avoid ~180 deg offsets
        pa_norm = normalize_pa_degrees(pa_deg, anchor=ref_pa_median)
        if ref_pa_median is None:
            finite_pa = pa_norm[np.isfinite(pa_norm)]
            if finite_pa.size > 0:
                ref_pa_median = float(np.nanmedian(finite_pa))

        all_pa_norm.append(pa_norm)
        pa_err_deg = None
        if "pa_err" in prof:
            pa_err_deg = np.degrees(prof["pa_err"])

        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        if pa_err_deg is not None:
            if "stop_codes" in prof:
                _scatter_by_stop_code_in_method_color(
                    ax_pa,
                    prof["x_axis"],
                    pa_norm,
                    prof["stop_codes"],
                    base_color=style["color"],
                    y_errors=pa_err_deg,
                    marker_size=18,
                    label_stop_codes=False,
                    label=style.get("label", method_name),
                )
            else:
                ax_pa.errorbar(
                    prof["x_axis"],
                    pa_norm,
                    yerr=pa_err_deg,
                    fmt=style.get("marker", "o"),
                    color=style["color"],
                    mfc=mfc,
                    mec=style["color"],
                    markersize=4.2,
                    capsize=1.5,
                    linewidth=0.6,
                    alpha=0.7,
                    zorder=3,
                )
        else:
            ax_pa.scatter(
                prof["x_axis"],
                pa_norm,
                color=style["color"],
                marker=style.get("marker", "o"),
                facecolors=mfc,
                edgecolors=style["color"],
                s=18,
                alpha=0.7,
                zorder=3,
            )
    ax_pa.set_ylabel("PA (deg)")
    ax_pa.grid(alpha=0.25)
    ax_pa.tick_params(labelbottom=False)
    # Data-driven PA limits from stop=0 data of first method, or all data
    if available and all_pa_norm:
        if "stop_codes" in profiles[available[0]]:
            sc = profiles[available[0]]["stop_codes"]
            pa_for_limits = all_pa_norm[0][sc == 0]
        else:
            pa_for_limits = all_pa_norm[0]
        pa_finite = pa_for_limits[np.isfinite(pa_for_limits)]
        if pa_finite.size > 1:
            pa_low, pa_high = robust_limits(pa_finite, 3, 97)
            pa_margin = max(3.0, 0.08 * (pa_high - pa_low + 1e-6))
            ax_pa.set_ylim(pa_low - pa_margin, pa_high + pa_margin)

    # Panel 4: Center offset (relative to isoster's robust median center)
    # Reference center: median of inner 10 stop=0 isophotes from the first
    # method (isoster), after 3-sigma clipping.
    ref_x0, ref_y0 = None, None
    first_method = available[0] if available else None
    if first_method and "x0" in profiles[first_method]:
        fprof = profiles[first_method]
        # Select stop=0 isophotes, sorted by SMA (innermost first)
        if "stop_codes" in fprof:
            good = fprof["stop_codes"] == 0
        else:
            good = np.isfinite(fprof["x0"])
        order = np.argsort(fprof["sma"])
        good_sorted = good[order]
        x0_sorted = fprof["x0"][order]
        y0_sorted = fprof["y0"][order]
        inner_x0 = x0_sorted[good_sorted][:10]
        inner_y0 = y0_sorted[good_sorted][:10]

        # 3-sigma clipping on radial distance from raw median
        if inner_x0.size > 0:
            mx, my = np.nanmedian(inner_x0), np.nanmedian(inner_y0)
            dist = np.sqrt((inner_x0 - mx) ** 2 + (inner_y0 - my) ** 2)
            sigma = np.nanstd(dist)
            if sigma > 0:
                keep = dist < 3 * sigma
                inner_x0 = inner_x0[keep]
                inner_y0 = inner_y0[keep]
            if inner_x0.size > 0:
                ref_x0 = float(np.nanmedian(inner_x0))
                ref_y0 = float(np.nanmedian(inner_y0))

    ax_cen = fig.add_subplot(right[4], sharex=ax_sb)
    all_offsets = []
    for method_name in available:
        prof = profiles[method_name]
        if "x0" not in prof:
            continue
        style = {"color": "black", **styles.get(method_name, {})}  # P3: default color for unknown methods
        # All methods use the isoster reference center if available;
        # fall back to own median if reference is not available
        if ref_x0 is not None:
            cx, cy = ref_x0, ref_y0
        else:
            cx = np.nanmedian(prof["x0"])
            cy = np.nanmedian(prof["y0"])
        offset = np.sqrt((prof["x0"] - cx) ** 2 + (prof["y0"] - cy) ** 2)
        all_offsets.append(offset)

        # Propagate center errors to offset error:
        # d(offset)/d(x0) ~ (x0-cx)/offset, etc.
        offset_err = None
        if "x0_err" in prof and "y0_err" in prof:
            safe_offset = np.where(offset > 0, offset, 1.0)
            dx = prof["x0"] - cx
            dy = prof["y0"] - cy
            offset_err = np.sqrt((dx / safe_offset * prof["x0_err"]) ** 2 + (dy / safe_offset * prof["y0_err"]) ** 2)
            # Where offset is ~0, use simple quadrature
            near_zero = offset < 1e-6
            if np.any(near_zero):
                offset_err[near_zero] = np.sqrt(prof["x0_err"][near_zero] ** 2 + prof["y0_err"][near_zero] ** 2)

        mfc = style["color"] if style.get("marker_face") == "filled" else "none"
        if offset_err is not None:
            if "stop_codes" in prof:
                _scatter_by_stop_code_in_method_color(
                    ax_cen,
                    prof["x_axis"],
                    offset,
                    prof["stop_codes"],
                    base_color=style["color"],
                    y_errors=offset_err,
                    marker_size=18,
                    label_stop_codes=False,
                    label=style.get("label", method_name),
                )
            else:
                ax_cen.errorbar(
                    prof["x_axis"],
                    offset,
                    yerr=offset_err,
                    fmt=style.get("marker", "o"),
                    color=style["color"],
                    mfc=mfc,
                    mec=style["color"],
                    markersize=4.2,
                    capsize=1.5,
                    linewidth=0.6,
                    alpha=0.7,
                    zorder=3,
                )
        else:
            ax_cen.scatter(
                prof["x_axis"],
                offset,
                color=style["color"],
                marker=style.get("marker", "o"),
                facecolors=mfc,
                edgecolors=style["color"],
                s=18,
                alpha=0.7,
                zorder=3,
            )
    ax_cen.set_ylabel(r"$\delta$ Cen (pix)")
    ax_cen.set_xlabel(r"SMA$^{0.25}$ (pix$^{0.25}$)")
    ax_cen.grid(alpha=0.25)
    # Data-driven Y-axis for center offset
    if all_offsets:
        all_off = np.concatenate(all_offsets)
        set_axis_limits_from_finite_values(
            ax_cen,
            all_off,
            margin_fraction=0.08,
            min_margin=0.05,
            lower_clip=0.0,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _overlay_photutils_sb(
    ax,
    photutils_res,
    *,
    sb_zeropoint=None,
    pixel_scale_arcsec=None,
    sb_profile_scale="log10",
    sb_asinh_softening=None,
):
    """Overlay photutils surface brightness on the SB panel (open markers)."""
    p_sma = np.array([iso.sma for iso in photutils_res])
    p_intens = np.array([iso.intens for iso in photutils_res])
    p_intens_err = np.array([iso.int_err for iso in photutils_res])

    p_y, p_yerr, *_ = transform_sb_profile(
        p_intens,
        p_intens_err,
        sb_zeropoint=sb_zeropoint,
        pixel_scale_arcsec=pixel_scale_arcsec,
        sb_profile_scale=sb_profile_scale,
        sb_asinh_softening=sb_asinh_softening,
    )
    p_mask = (p_sma > 1.5) & np.isfinite(p_y)
    p_x = p_sma[p_mask] ** 0.25

    ax.errorbar(
        p_x,
        p_y[p_mask],
        yerr=p_yerr[p_mask] if p_yerr is not None else None,
        fmt="o",
        mfc="none",
        mec="black",
        color="black",
        markersize=4,
        capsize=1.8,
        linewidth=0.7,
        alpha=0.8,
        label="photutils",
    )
