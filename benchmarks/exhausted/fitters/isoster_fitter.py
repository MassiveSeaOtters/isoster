"""Single-galaxy × single-arm isoster fit driver.

Responsibilities:

- Resolve sentinels (``_use_Re``, ``_use_2Re``, ``_special: drop_variance``)
  against the galaxy bundle.
- Decide when an arm is inapplicable (e.g. needs Re but bundle has
  ``effective_Re_pix=None``) and return a ``status="skipped"`` inventory
  row instead of running.
- Build an effective ``IsosterConfig`` by overlaying the arm delta on
  the bundle's initial geometry and the non-negotiable campaign flags
  (``debug=True``, ``full_photometry=True``, ``compute_cog=True``).
- Run ``isoster.fit_image`` under a wall-clock timer.
- Write per-arm outputs: ``profile.fits``, ``model.fits`` (optional),
  ``qa.png``, ``run_record.json``, ``config.yaml``.
- Return an inventory row dict with status, timing, and summary metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits

from isoster import IsosterConfig, fit_image
from isoster.utils import isophote_results_to_fits

from ..adapters.base import GalaxyBundle
from ..analysis.inventory import INVENTORY_COLUMNS
from ..analysis.metrics import summarize_fit
from ..analysis.model_evaluation import evaluate_model_v11, profile_summary_for_inventory
from ..analysis.quality_flags import evaluate_flags
from ..plotting.per_galaxy_qa import build_model_cube, render_per_arm_qa

# ---------------------------------------------------------------------------
# Non-negotiable campaign flags
# ---------------------------------------------------------------------------
# These override any arm's attempt to change them; every campaign fit runs
# with debug + full photometry + COG so metrics, QA, and inventory stay
# homogeneous across arms.
CAMPAIGN_FORCED_FLAGS = {
    "debug": True,
    "full_photometry": True,
    "compute_cog": True,
}

# Sentinel tokens understood by the arm loader.
SENTINEL_RE = "_use_Re"
SENTINEL_2RE = "_use_2Re"
SPECIAL_DROP_VARIANCE = "drop_variance"


@dataclass
class ArmFitOutputs:
    profile_fits: Path
    model_fits: Path | None
    qa_png: Path | None
    run_record_json: Path
    config_yaml: Path


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
) -> dict[str, Any]:
    """Run one ``(galaxy, arm)`` pair. Returns an inventory row.

    Row schema (keys are homogeneous across all arms and tools; missing
    fields become ``nan`` / empty string):

        galaxy_id, tool, arm_id, status, error_msg,
        wall_time_fit_s, wall_time_total_s,
        n_iso, n_stop_0, n_stop_1, n_stop_2, n_stop_m1,
        frac_stop_nonzero, stop_code_hist,
        combined_drift_pix, spline_rms_center, max_dpa_deg, max_deps,
        outer_gerr_median, outward_drift_x, outward_drift_y,
        n_locked, locked_drift_x, locked_drift_y,
        qa_path, profile_path, model_path, config_path
    """
    total_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_skeleton: dict[str, Any] = _empty_inventory_row(bundle.metadata.galaxy_id, arm_id)

    # Sentinel resolution. Returns a (maybe pruned) delta or signals a skip.
    effective_delta, use_variance, skip_reason = _resolve_sentinels(arm_delta, bundle)
    if skip_reason is not None:
        row_skeleton["status"] = "skipped"
        row_skeleton["error_msg"] = skip_reason
        _write_run_record(output_dir, {"status": "skipped", "reason": skip_reason})
        return row_skeleton

    # Compose IsosterConfig.
    try:
        config = _build_config(bundle, effective_delta)
    except Exception as exc:  # noqa: BLE001 - pydantic may raise varied types
        row_skeleton["status"] = "config_error"
        row_skeleton["error_msg"] = str(exc)
        _write_run_record(output_dir, {"status": "config_error", "error": str(exc)})
        return row_skeleton

    # Decide whether to forward the variance map.
    variance_map = bundle.variance if use_variance else None

    # Run the fit.
    fit_start = time.perf_counter()
    try:
        results = fit_image(
            np.asarray(bundle.image, dtype=np.float64),
            bundle.mask,
            config,
            variance_map=variance_map,
        )
    except Exception as exc:  # noqa: BLE001 - surface any solver failure
        wall_fit = time.perf_counter() - fit_start
        row_skeleton["status"] = "failed"
        row_skeleton["error_msg"] = f"{type(exc).__name__}: {exc}"
        row_skeleton["wall_time_fit_s"] = float(wall_fit)
        row_skeleton["wall_time_total_s"] = float(time.perf_counter() - total_start)
        _write_run_record(
            output_dir,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_msg": str(exc),
                "wall_time_fit_s": float(wall_fit),
            },
        )
        return row_skeleton
    wall_fit = time.perf_counter() - fit_start

    # Write outputs.
    outputs = _write_arm_outputs(
        bundle=bundle,
        arm_id=arm_id,
        config=config,
        results=results,
        output_dir=output_dir,
        write_qa=write_qa,
        write_model_fits=write_model_fits,
        sb_profile_scale=sb_profile_scale,
        sb_asinh_softening=sb_asinh_softening,
    )

    # Summary metrics.
    lsb_threshold = getattr(config, "lsb_sma_threshold", None)
    metrics = summarize_fit(
        results,
        sma0=float(config.sma0),
        lsb_sma_threshold_pix=lsb_threshold,
    )
    metrics.update(profile_summary_for_inventory(str(outputs.profile_fits)))

    # First-isophote diagnostics (live in the results dict, not the profile FITS).
    retry_log = results.get("first_isophote_retry_log") or []
    first_isophote_diag = {
        "first_isophote_failure": bool(results.get("first_isophote_failure", False)),
        "first_isophote_retry_attempts": len(retry_log),
        "first_isophote_retry_stop_codes": ",".join(str(int(entry.get("stop_code", -99))) for entry in retry_log),
    }

    # v1.1 residual and azimuthal statistics (elliptical zones anchored at
    # the adapter's initial geometry so zones stay arm-independent).
    image = np.asarray(bundle.image, dtype=np.float64)
    model, _ = build_model_cube(bundle, results)
    geom = bundle.initial_geometry
    model_metrics = evaluate_model_v11(
        image=image,
        model=model,
        mask=bundle.mask,
        x0=float(geom["x0"]),
        y0=float(geom["y0"]),
        eps=float(geom.get("eps", 0.2)),
        pa_rad=float(geom.get("pa", 0.0)),
        R_ref_pix=bundle.metadata.effective_Re_pix,
        maxsma_pix=float(geom.get("maxsma", min(image.shape) // 2)),
    )

    row = dict(row_skeleton)
    row.update(
        {
            "status": "ok",
            "error_msg": "",
            "wall_time_fit_s": float(wall_fit),
            "wall_time_total_s": float(time.perf_counter() - total_start),
            **metrics,
            **first_isophote_diag,
            **model_metrics,
            "qa_path": str(outputs.qa_png) if outputs.qa_png else "",
            "profile_path": str(outputs.profile_fits),
            "model_path": str(outputs.model_fits) if outputs.model_fits else "",
            "config_path": str(outputs.config_yaml),
        }
    )
    # Flags depend on all metrics above, so evaluate after the row is
    # assembled.
    row.update(evaluate_flags(row))
    _write_run_record(
        output_dir,
        {
            "status": "ok",
            "wall_time_fit_s": float(wall_fit),
            "wall_time_total_s": row["wall_time_total_s"],
            "metrics": {**metrics, **first_isophote_diag, **model_metrics},
            "flags": row.get("flags", ""),
            "flag_severity_max": row.get("flag_severity_max", 0.0),
            "first_isophote_retry_log": retry_log,
            "arm_delta": effective_delta,
            "variance_used": bool(use_variance and bundle.variance is not None),
            "config_snapshot": _config_to_dict(config),
        },
    )
    return row


# ---------------------------------------------------------------------------
# Sentinel / config helpers
# ---------------------------------------------------------------------------


def _resolve_sentinels(delta: dict[str, Any], bundle: GalaxyBundle) -> tuple[dict[str, Any], bool, str | None]:
    """Return ``(resolved_delta, use_variance, skip_reason)``.

    ``use_variance`` is False when the arm explicitly asks to drop it.
    ``skip_reason`` is not None when the arm is inapplicable.
    """
    delta = dict(delta or {})
    use_variance = True

    special = delta.pop("_special", None)
    if special == SPECIAL_DROP_VARIANCE:
        if bundle.variance is None:
            return delta, use_variance, "no variance map available; arm is a no-op"
        use_variance = False
    elif special is not None:
        return delta, use_variance, f"unknown _special token {special!r}"

    if delta.get("lsb_sma_threshold") == SENTINEL_RE:
        Re = bundle.metadata.effective_Re_pix
        if Re is None:
            return delta, use_variance, f"arm requires Re; dataset '{bundle.metadata.dataset}' has none"
        delta["lsb_sma_threshold"] = float(Re)
    elif delta.get("lsb_sma_threshold") == SENTINEL_2RE:
        Re = bundle.metadata.effective_Re_pix
        if Re is None:
            return delta, use_variance, f"arm requires 2*Re; dataset '{bundle.metadata.dataset}' has none"
        delta["lsb_sma_threshold"] = float(2.0 * Re)

    return delta, use_variance, None


def _build_config(bundle: GalaxyBundle, delta: dict[str, Any]) -> IsosterConfig:
    """Overlay the arm delta onto the bundle's initial geometry and campaign flags."""
    geometry = bundle.initial_geometry
    cfg_kwargs: dict[str, Any] = {
        "x0": float(geometry["x0"]),
        "y0": float(geometry["y0"]),
        "eps": float(geometry.get("eps", 0.2)),
        "pa": float(geometry.get("pa", 0.0)),
        "sma0": float(geometry["sma0"]),
    }
    if "maxsma" in geometry:
        cfg_kwargs["maxsma"] = float(geometry["maxsma"])
    cfg_kwargs.update(delta)
    cfg_kwargs.update(CAMPAIGN_FORCED_FLAGS)
    return IsosterConfig(**cfg_kwargs)


def _config_to_dict(config: IsosterConfig) -> dict[str, Any]:
    return config.model_dump()


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def _write_arm_outputs(
    *,
    bundle: GalaxyBundle,
    arm_id: str,
    config: IsosterConfig,
    results: dict[str, Any],
    output_dir: Path,
    write_qa: bool,
    write_model_fits: bool,
    sb_profile_scale: str,
    sb_asinh_softening: float | None,
) -> ArmFitOutputs:
    profile_path = output_dir / "profile.fits"
    isophote_results_to_fits(results, str(profile_path), overwrite=True)

    config_path = output_dir / "config.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(_config_to_dict(config), handle, sort_keys=False)

    model_path: Path | None = None
    qa_path: Path | None = None
    if write_model_fits:
        model_path = output_dir / "model.fits"
        model, residual = build_model_cube(bundle, results)
        primary = fits.PrimaryHDU(
            data=model.astype(np.float32),
            header=fits.Header({"EXTNAME": "MODEL", "ARM_ID": arm_id}),
        )
        residual_hdu = fits.ImageHDU(
            data=residual.astype(np.float32),
            header=fits.Header({"EXTNAME": "RESIDUAL", "ARM_ID": arm_id}),
        )
        fits.HDUList([primary, residual_hdu]).writeto(model_path, overwrite=True)

    if write_qa:
        qa_path = output_dir / "qa.png"
        try:
            render_per_arm_qa(
                bundle,
                arm_id,
                results,
                qa_path,
                sb_profile_scale=sb_profile_scale,
                sb_asinh_softening=sb_asinh_softening,
            )
        except Exception as exc:  # noqa: BLE001 - QA must never abort the campaign
            qa_path.with_suffix(".png.err.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            qa_path = None

    return ArmFitOutputs(
        profile_fits=profile_path,
        model_fits=model_path,
        qa_png=qa_path,
        run_record_json=output_dir / "run_record.json",
        config_yaml=config_path,
    )


def _write_run_record(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_record.json"
    with path.open("w") as handle:
        json.dump(record, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Inventory row skeleton
# ---------------------------------------------------------------------------


def _empty_inventory_row(galaxy_id: str, arm_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {col: "" for col in INVENTORY_COLUMNS}
    row["galaxy_id"] = galaxy_id
    row["tool"] = "isoster"
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
    ):
        row[int_col] = 0
    row["first_isophote_failure"] = False
    for numeric_col in (
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
        "flag_severity_max",
        "composite_score",
    ):
        row[numeric_col] = float("nan")
    return row
