"""photutils.isophote campaign fitter.

Ported from ``/Users/shuang/Dropbox/work/project/otters/sga_isoster/scripts/
photutils_fitter.py`` and adapted to the campaign's DatasetAdapter +
shared inventory/metrics pipeline.

The fitter converts photutils's ``IsophoteList`` into the isoster-style
``results`` dict (``{"isophotes": [...]}``) so every downstream helper
— model reconstruction, QA PNG, residual zones, quality flags,
composite score — works without branching on tool.

Public API mirrors :func:`isoster_fitter.run_one_arm`: same signature,
same inventory-row schema, same per-arm output layout
(``profile.fits`` / ``model.fits`` / ``qa.png`` / ``run_record.json`` /
``config.yaml``).
"""

from __future__ import annotations

import contextlib
import json
import signal
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits

from isoster import build_isoster_model
from isoster.plotting import plot_qa_summary
from isoster.utils import isophote_results_to_fits

from ..adapters.base import GalaxyBundle
from ..analysis.inventory import INVENTORY_COLUMNS
from ..analysis.metrics import summarize_fit
from ..analysis.model_evaluation import evaluate_model_v11, profile_summary_for_inventory
from ..analysis.quality_flags import evaluate_flags


class _PhotutilsFitTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def _fit_timeout(seconds: float | None):
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise ValueError("photutils timeout must be positive")
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("photutils timeout requires SIGALRM support")

    def _raise_timeout(_signum, _frame):
        raise _PhotutilsFitTimeout

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_one_arm(
    bundle: GalaxyBundle,
    arm_id: str,
    arm_delta: dict[str, Any],
    output_dir: Path,
    *,
    write_qa: bool = True,
    write_model_fits: bool = True,
    sb_profile_scale: str = "log10",
    sb_asinh_softening: float | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one ``(galaxy, photutils-arm)`` pair. Returns an inventory row."""
    from photutils.isophote import Ellipse, EllipseGeometry

    total_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row = _empty_inventory_row(bundle.metadata.galaxy_id, arm_id)
    resolved = _resolve_arm(arm_delta, bundle)
    if resolved["skip_reason"] is not None:
        row["status"] = "skipped"
        row["error_msg"] = resolved["skip_reason"]
        _write_run_record(output_dir, {"status": "skipped", "reason": resolved["skip_reason"]})
        return row

    geom = bundle.initial_geometry
    image = np.asarray(bundle.image, dtype=np.float64)
    mask = bundle.mask
    if mask is not None:
        masked_image = np.ma.array(image, mask=np.asarray(mask, dtype=bool))
    else:
        masked_image = np.ma.array(image, mask=np.zeros_like(image, dtype=bool))

    maxsma = float(geom.get("maxsma", min(image.shape) / 2.0))
    sma0_initial = float(geom["sma0"])
    # Small retry ladder: sma0, 0.6*sma0, 1.3*sma0, 3.0 px, 10 px.
    ladder_raw = [sma0_initial, 0.6 * sma0_initial, 1.3 * sma0_initial, 3.0, 10.0]
    ladder: list[float] = []
    seen: set[float] = set()
    for v in ladder_raw:
        v = round(float(v), 2)
        if v not in seen and 0.5 < v < maxsma:
            seen.add(v)
            ladder.append(v)

    fit_start = time.perf_counter()
    isolist = None
    sma0_used = 0.0
    fit_error: str | None = None
    try:
        with _fit_timeout(timeout):
            for sma0 in ladder:
                try:
                    photutils_geom = EllipseGeometry(
                        x0=float(geom["x0"]),
                        y0=float(geom["y0"]),
                        sma=sma0,
                        eps=float(geom.get("eps", 0.2)),
                        pa=_pa_for_photutils(float(geom.get("pa", 0.0))),
                    )
                    ellipse = Ellipse(masked_image, geometry=photutils_geom)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        isolist = ellipse.fit_image(
                            integrmode=resolved["integrmode"],
                            sclip=resolved["sclip"],
                            nclip=resolved["nclip"],
                            linear=resolved["linear"],
                            step=resolved["step"],
                            fflag=resolved["fflag"],
                            conver=resolved["conver"],
                            minit=resolved["minit"],
                            maxit=resolved["maxit"],
                            maxgerr=resolved["maxgerr"],
                            minsma=resolved["minsma"],
                            maxsma=maxsma,
                            fix_center=resolved["fix_center"],
                        )
                    if len(isolist) > 0:
                        sma0_used = sma0
                        break
                except _PhotutilsFitTimeout:
                    raise
                except Exception as exc:  # noqa: BLE001 - try next sma0
                    fit_error = f"sma0={sma0}: {exc}"
                    isolist = None
                    continue
    except _PhotutilsFitTimeout:
        fit_error = f"timeout after {timeout:g}s"
        isolist = None
    wall_fit = time.perf_counter() - fit_start

    if isolist is None or len(isolist) == 0:
        row["status"] = "failed"
        row["error_msg"] = fit_error or "photutils returned empty IsophoteList"
        row["wall_time_fit_s"] = float(wall_fit)
        row["wall_time_total_s"] = float(time.perf_counter() - total_start)
        _write_run_record(
            output_dir,
            {
                "status": "failed",
                "error_msg": row["error_msg"],
                "wall_time_fit_s": float(wall_fit),
            },
        )
        return row

    # Convert photutils IsophoteList -> isoster-compatible results dict.
    results = _build_results_dict(isolist, resolved, sma0_used)

    # Per-arm outputs (profile / model / qa / config).
    profile_path = output_dir / "profile.fits"
    isophote_results_to_fits(results, str(profile_path), overwrite=True)

    config_path = output_dir / "config.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {"tool": "photutils", "arm_id": arm_id, **resolved["dumped_config"]},
            handle,
            sort_keys=False,
        )

    model_path = None
    if write_model_fits:
        model_path = output_dir / "model.fits"
        model = build_isoster_model(image.shape, results["isophotes"])
        residual = image - model
        primary = fits.PrimaryHDU(
            data=model.astype(np.float32),
            header=fits.Header({"EXTNAME": "MODEL", "ARM_ID": arm_id}),
        )
        resid_hdu = fits.ImageHDU(
            data=residual.astype(np.float32),
            header=fits.Header({"EXTNAME": "RESIDUAL", "ARM_ID": arm_id}),
        )
        fits.HDUList([primary, resid_hdu]).writeto(model_path, overwrite=True)

    qa_path = None
    if write_qa:
        qa_path = output_dir / "qa.png"
        try:
            model = build_isoster_model(image.shape, results["isophotes"])
            plot_qa_summary(
                title=f"{bundle.metadata.galaxy_id}  |  photutils::{arm_id}",
                image=image,
                isoster_model=model,
                isoster_res=results["isophotes"],
                mask=mask,
                filename=str(qa_path),
                relative_residual=False,
                sb_zeropoint=bundle.metadata.sb_zeropoint,
                pixel_scale_arcsec=bundle.metadata.pixel_scale_arcsec,
                sb_profile_scale=sb_profile_scale,
                sb_asinh_softening=sb_asinh_softening,
            )
        except Exception as exc:  # noqa: BLE001 - never abort on QA failure
            qa_path.with_suffix(".png.err.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            qa_path = None

    # Metrics + residual zones + flags (shared with isoster pipeline).
    metrics = summarize_fit(results, sma0=sma0_used, lsb_sma_threshold_pix=None)
    metrics.update(profile_summary_for_inventory(str(profile_path)))
    model_for_zones = build_isoster_model(image.shape, results["isophotes"])
    model_metrics = evaluate_model_v11(
        image=image,
        model=model_for_zones,
        mask=mask,
        x0=float(geom["x0"]),
        y0=float(geom["y0"]),
        eps=float(geom.get("eps", 0.2)),
        pa_rad=float(geom.get("pa", 0.0)),
        R_ref_pix=bundle.metadata.effective_Re_pix,
        maxsma_pix=maxsma,
        r_inner_floor_pix=float(metrics.get("min_sma_pix", 0.0) or 0.0),
    )

    row.update(
        {
            "status": "ok",
            "error_msg": "",
            "wall_time_fit_s": float(wall_fit),
            "wall_time_total_s": float(time.perf_counter() - total_start),
            **metrics,
            **model_metrics,
            # photutils does not expose a first-isophote-retry log.
            "first_isophote_failure": False,
            "first_isophote_retry_attempts": 0,
            "first_isophote_retry_stop_codes": "",
            "qa_path": str(qa_path) if qa_path else "",
            "profile_path": str(profile_path),
            "model_path": str(model_path) if model_path else "",
            "config_path": str(config_path),
        }
    )
    row.update(evaluate_flags(row))
    _write_run_record(
        output_dir,
        {
            "status": "ok",
            "wall_time_fit_s": float(wall_fit),
            "wall_time_total_s": row["wall_time_total_s"],
            "metrics": {**metrics, **model_metrics},
            "flags": row.get("flags", ""),
            "flag_severity_max": row.get("flag_severity_max", 0.0),
            "sma0_used": float(sma0_used),
            "arm_delta": arm_delta,
            "config_snapshot": resolved["dumped_config"],
        },
    )
    return row


# ---------------------------------------------------------------------------
# Arm resolution
# ---------------------------------------------------------------------------

PHOTUTILS_DEFAULTS: dict[str, Any] = {
    "integrmode": "median",
    "sclip": 3.0,
    "nclip": 1,
    "linear": False,
    "step": 0.1,
    "fflag": 0.7,
    "fix_center": False,
    "conver": 0.05,
    "minit": 10,
    "maxit": 50,
    "maxgerr": 0.5,
    "minsma": 0.0,
}

ALLOWED_ARM_KEYS = set(PHOTUTILS_DEFAULTS.keys())


def _resolve_arm(arm_delta: dict[str, Any], _bundle: GalaxyBundle) -> dict[str, Any]:
    """Merge arm delta into the photutils defaults. No sentinels today."""
    cfg = dict(PHOTUTILS_DEFAULTS)
    for key, value in (arm_delta or {}).items():
        if key in ALLOWED_ARM_KEYS:
            cfg[key] = value
        # Unknown keys silently ignored; campaign YAML may contain
        # isoster-specific keys that do not apply to photutils.
    return {
        **cfg,
        "skip_reason": None,
        "dumped_config": dict(cfg),
    }


# ---------------------------------------------------------------------------
# IsophoteList -> isoster results-dict conversion
# ---------------------------------------------------------------------------

# Mapping from isoster-style output key -> photutils Isophote attribute.
_ATTR_MAP = {
    "sma": "sma",
    "intens": "intens",
    "intens_err": "int_err",
    "eps": "eps",
    "eps_err": "ellip_err",
    "pa": "pa",
    "pa_err": "pa_err",
    "x0": "x0",
    "x0_err": "x0_err",
    "y0": "y0",
    "y0_err": "y0_err",
    "grad": "grad",
    "grad_error": "grad_error",
    "grad_r_error": "grad_rerror",
    "rms": "rms",
    "stop_code": "stop_code",
    "ndata": "ndata",
    "nflag": "nflag",
    "niter": "niter",
    "a3": "a3",
    "a3_err": "a3_err",
    "b3": "b3",
    "b3_err": "b3_err",
    "a4": "a4",
    "a4_err": "a4_err",
    "b4": "b4",
    "b4_err": "b4_err",
    "tflux_e": "tflux_e",
    "tflux_c": "tflux_c",
    "npix_e": "npix_e",
    "npix_c": "npix_c",
}


def _iso_to_dict(iso) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for out_key, attr in _ATTR_MAP.items():
        value = getattr(iso, attr, None)
        if value is None:
            out[out_key] = float("nan") if "err" in out_key or out_key in {"grad", "grad_error", "grad_r_error"} else 0
        else:
            try:
                out[out_key] = float(value)
            except (TypeError, ValueError):
                out[out_key] = 0
    # lsb_locked column: photutils has no LSB lock concept; always False.
    out["lsb_locked"] = False
    out["tool"] = "photutils"
    return out


def _build_results_dict(isolist, resolved: dict[str, Any], sma0_used: float) -> dict[str, Any]:
    """Return an isoster-compatible results dict from a photutils IsophoteList."""
    isophotes: list[dict[str, Any]] = [_iso_to_dict(iso) for iso in isolist]
    return {
        "isophotes": isophotes,
        "sma0_used": float(sma0_used),
        "tool": "photutils",
        "config": resolved["dumped_config"],
        # These keys exist in isoster's results dict; set to defaults so
        # downstream metrics see a homogeneous shape.
        "first_isophote_failure": False,
        "first_isophote_retry_log": [],
    }


def _pa_for_photutils(pa_rad: float) -> float:
    """photutils expects PA in (0, pi] radians."""
    value = float(pa_rad) % np.pi
    return max(value, 0.01)


# ---------------------------------------------------------------------------
# Inventory row skeleton + run_record helper
# ---------------------------------------------------------------------------


def _empty_inventory_row(galaxy_id: str, arm_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {col: "" for col in INVENTORY_COLUMNS}
    row["galaxy_id"] = galaxy_id
    row["tool"] = "photutils"
    row["arm_id"] = arm_id
    row["status"] = "pending"
    for int_col in (
        "n_iso",
        "n_stop_0",
        "n_stop_1",
        "n_stop_2",
        "n_stop_m1",
        "n_locked",
        "first_isophote_retry_attempts",
        "n_iso_ref_used",
    ):
        row[int_col] = 0
    row["first_isophote_failure"] = False
    for num_col in (
        "wall_time_fit_s",
        "wall_time_total_s",
        "frac_stop_nonzero",
        "combined_drift_pix",
        "spline_rms_center",
        "max_dpa_deg",
        "max_deps",
        "outer_gerr_median",
        "outward_drift_x",
        "outward_drift_y",
        "locked_drift_x",
        "locked_drift_y",
        "resid_rms_inner",
        "resid_rms_mid",
        "resid_rms_outer",
        "resid_median_inner",
        "resid_median_mid",
        "resid_median_outer",
        "frac_above_3sigma_outer",
        "image_sigma_adu",
        "flag_severity_max",
        "composite_score",
    ):
        row[num_col] = float("nan")
    return row


def _write_run_record(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_record.json").open("w") as handle:
        json.dump(record, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)
