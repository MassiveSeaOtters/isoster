"""Campaign runner.

Phase B shipped a straight-line ``for galaxy: for arm: fit_one`` driver
with ``skip_existing`` semantics. Phase E adds optional per-galaxy
parallelism via :class:`concurrent.futures.ProcessPoolExecutor` driven
by ``execution.max_parallel_galaxies``. Tool and arm iteration remains
serial inside each galaxy.

Public entry points:
- :func:`run_campaign`   — top-level, iterates all datasets and tools
- :func:`run_dataset`    — one dataset, sequential or parallel by galaxy
- :func:`write_inventory` — per-(dataset, tool) inventory FITS
"""

from __future__ import annotations

import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..adapters.base import GalaxyBundle, safe_galaxy_id
from ..analysis.inventory import (
    INVENTORY_COLUMNS,
    column_default,
    write_inventory,
)
from ..analysis.noise import compute_image_sigma
from ..fitters.autoprof_fitter import run_one_arm as _autoprof_run_one_arm
from ..fitters.isoster_fitter import run_one_arm as _isoster_run_one_arm
from ..fitters.photutils_fitter import run_one_arm as _photutils_run_one_arm
from ..plotting.cross_arm_overlay import plot_cross_arm_overlay
from ..plotting.cross_tool_comparison import plot_cross_tool_comparison
from ..plotting.summary_grids import plot_summary_profiles, plot_summary_residuals
from .config_loader import CampaignPlan, DatasetPlan, ToolPlan, load_campaign
from .stats import (
    apply_composite_scores,
    write_cross_tool_table,
    write_per_galaxy_cross_arm_table,
    write_per_tool_cross_arm_summary,
)


@dataclass
class RunSummary:
    total_requested: int = 0
    total_ran: int = 0
    total_skipped_existing: int = 0
    total_skipped_arm: int = 0
    total_failed: int = 0
    total_ok: int = 0


def run_campaign(plan: CampaignPlan) -> RunSummary:
    """Execute the whole campaign described by ``plan``. Phase B: sequential."""
    campaign_dir = plan.output_root / plan.campaign_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    _write_campaign_snapshot(plan, campaign_dir)
    _write_environment_snapshot(campaign_dir)

    summary = RunSummary()
    for ds_plan in plan.datasets.values():
        if not ds_plan.enabled:
            print(f"[skip] dataset '{ds_plan.name}' disabled")
            continue
        ds_summary = run_dataset(plan, ds_plan, campaign_dir)
        summary.total_requested += ds_summary.total_requested
        summary.total_ran += ds_summary.total_ran
        summary.total_skipped_existing += ds_summary.total_skipped_existing
        summary.total_skipped_arm += ds_summary.total_skipped_arm
        summary.total_failed += ds_summary.total_failed
        summary.total_ok += ds_summary.total_ok

    return summary


def run_dataset(
    plan: CampaignPlan, ds_plan: DatasetPlan, campaign_dir: Path
) -> RunSummary:
    """Run every enabled tool × arm matrix on one dataset.

    Galaxies run sequentially when ``execution.max_parallel_galaxies``
    is 1 (the default) and in parallel via :class:`ProcessPoolExecutor`
    otherwise. Tools and arms always run serially inside one galaxy.
    """
    summary = RunSummary()
    dataset_dir = campaign_dir / ds_plan.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    galaxy_ids = ds_plan.adapter.list_galaxies()
    if ds_plan.select:
        allow = set(ds_plan.select)
        galaxy_ids = [g for g in galaxy_ids if g in allow]

    # Collected inventory rows per (tool, galaxy) for dataset-level summaries.
    per_tool_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}

    print(f"=== dataset '{ds_plan.name}' ({len(galaxy_ids)} galaxies) ===")

    max_par = int(plan.execution.get("max_parallel_galaxies", 1) or 1)
    max_par = max(1, min(max_par, len(galaxy_ids) or 1))
    if max_par > 1 and len(galaxy_ids) > 1:
        _run_galaxies_in_parallel(
            plan=plan,
            ds_plan=ds_plan,
            dataset_dir=dataset_dir,
            galaxy_ids=galaxy_ids,
            max_par=max_par,
            summary=summary,
            per_tool_rows=per_tool_rows,
        )
    else:
        for index, galaxy_id in enumerate(galaxy_ids, start=1):
            print(f"  [{index:>3d}/{len(galaxy_ids)}] {galaxy_id}")
            galaxy_summary, galaxy_per_tool_rows = _process_one_galaxy(
                plan=plan,
                ds_plan=ds_plan,
                dataset_dir=dataset_dir,
                galaxy_id=galaxy_id,
            )
            if galaxy_summary is None:
                continue
            _accumulate_summary(summary, galaxy_summary)
            for tool, rows in galaxy_per_tool_rows.items():
                per_tool_rows.setdefault(tool, {})[galaxy_id] = rows

    # Dataset-level per-tool cross-arm summary (Phase C).
    if plan.summary.get("per_tool_cross_arm_table", True):
        for tool_name, rows_by_galaxy in per_tool_rows.items():
            if not rows_by_galaxy:
                continue
            dataset_tool_dir = dataset_dir / tool_name
            csv_path, md_path = write_per_tool_cross_arm_summary(
                rows_by_galaxy, tool_name, dataset_tool_dir
            )
            print(f"  wrote {csv_path.relative_to(campaign_dir)}")
            print(f"  wrote {md_path.relative_to(campaign_dir)}")

    # Dataset-level multi-galaxy grids (one per (tool, arm)).
    if plan.qa.get("summary_grids", True):
        _write_summary_grids(plan, per_tool_rows, dataset_dir, campaign_dir)

    # Dataset-level cross-tool table (Phase D + Phase E tool-neutral score).
    if plan.summary.get("cross_tool_table", True) and len(per_tool_rows) >= 2:
        result = write_cross_tool_table(
            per_tool_rows,
            dataset_dir,
            cross_tool_weights=plan.summary.get("cross_tool_score_weights"),
        )
        if result is not None:
            csv_path, md_path = result
            print(f"  wrote {csv_path.relative_to(campaign_dir)}")
            print(f"  wrote {md_path.relative_to(campaign_dir)}")

    return summary


def _process_one_galaxy(
    *,
    plan: CampaignPlan,
    ds_plan: DatasetPlan,
    dataset_dir: Path,
    galaxy_id: str,
) -> tuple[RunSummary | None, dict[str, list[dict[str, Any]]]]:
    """Run every enabled tool × arm on a single galaxy.

    This is the per-galaxy work unit dispatched to parallel workers.
    It is a module-level function (picklable under the macOS ``spawn``
    start method) that performs all I/O for one galaxy: load, sigma
    estimate, manifest, per-tool fits, per-galaxy cross-tool figure.
    Dataset-level aggregation happens back in :func:`run_dataset`.

    Returns ``(None, {})`` on load failure; the caller logs and skips.
    """
    # Make sure matplotlib is headless inside every worker. The package
    # ``benchmarks.exhausted.plotting`` does this at import time, but
    # spawn workers that hit a plotting function via a different path
    # (``isoster.plotting``) also need the non-interactive backend.
    import matplotlib
    matplotlib.use("Agg")

    try:
        bundle = ds_plan.adapter.load_galaxy(galaxy_id)
    except Exception as exc:  # noqa: BLE001 - record and continue
        print(f"      ERROR loading galaxy {galaxy_id}: {exc}")
        return None, {}

    galaxy_dir = dataset_dir / safe_galaxy_id(galaxy_id)
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    sigma_info = _estimate_galaxy_sigma(bundle)
    _write_manifest(bundle, galaxy_dir, sigma_info=sigma_info)

    summary = RunSummary()
    galaxy_per_tool_rows: dict[str, list[dict[str, Any]]] = {}

    for tool_name, tool_plan in plan.tools.items():
        if not tool_plan.enabled:
            continue
        if tool_name not in _TOOL_DISPATCH:
            print(
                f"      [skip] tool '{tool_name}' — no fitter registered "
                f"(known: {sorted(_TOOL_DISPATCH)})"
            )
            continue
        tool_summary, rows = _run_tool_on_galaxy(
            plan, tool_plan, bundle, galaxy_dir, sigma_info=sigma_info
        )
        galaxy_per_tool_rows[tool_name] = rows
        _accumulate_summary(summary, tool_summary)

    if plan.qa.get("cross_tool_comparison", False):
        galaxy_inventories = {
            tool: galaxy_per_tool_rows.get(tool, [])
            for tool in plan.tools
            if plan.tools[tool].enabled
        }
        galaxy_inventories = {k: v for k, v in galaxy_inventories.items() if v}
        if len(galaxy_inventories) >= 2:
            cross_dir = galaxy_dir / "cross"
            cross_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_cross_tool_comparison(
                    galaxy_inventories,
                    cross_dir / "cross_tool_comparison.png",
                    image=np.asarray(bundle.image, dtype=np.float64),
                    mask=bundle.mask,
                    sb_zeropoint=bundle.metadata.sb_zeropoint,
                    pixel_scale_arcsec=bundle.metadata.pixel_scale_arcsec,
                    sb_profile_scale=str(plan.qa.get("sb_profile_scale", "log10")),
                    sb_asinh_softening=(
                        float(plan.qa["sb_asinh_softening"])
                        if plan.qa.get("sb_asinh_softening") is not None
                        else None
                    ),
                    title=f"{galaxy_id}  |  cross-tool",
                )
            except Exception as exc:  # noqa: BLE001
                (cross_dir / "cross_tool_comparison.err.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n"
                )

    return summary, galaxy_per_tool_rows


def _run_galaxies_in_parallel(
    *,
    plan: CampaignPlan,
    ds_plan: DatasetPlan,
    dataset_dir: Path,
    galaxy_ids: list[str],
    max_par: int,
    summary: RunSummary,
    per_tool_rows: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    """Parallel driver; mutates ``summary`` and ``per_tool_rows`` in place.

    macOS defaults to ``spawn``, so we explicitly request it on every
    platform for deterministic behavior. ``fail_fast=True`` cancels
    pending futures on the first worker exception; in-flight workers
    continue running until completion (Python cannot kill them cleanly).
    """
    fail_fast = bool(plan.execution.get("fail_fast", False))
    ctx = mp.get_context("spawn")
    print(
        f"  [parallel] max_parallel_galaxies={max_par} "
        f"({len(galaxy_ids)} galaxies; mp_context=spawn)"
    )
    with ProcessPoolExecutor(max_workers=max_par, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                _process_one_galaxy,
                plan=plan,
                ds_plan=ds_plan,
                dataset_dir=dataset_dir,
                galaxy_id=gid,
            ): gid
            for gid in galaxy_ids
        }
        done_count = 0
        fatal: BaseException | None = None
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            for fut in done:
                gid = futures[fut]
                done_count += 1
                try:
                    result = fut.result()
                except BaseException as exc:  # noqa: BLE001 - surface to caller
                    print(
                        f"  [{done_count:>3d}/{len(galaxy_ids)}] {gid} "
                        f"WORKER FAILED: {type(exc).__name__}: {exc}"
                    )
                    if fail_fast:
                        fatal = exc
                        break
                    continue
                galaxy_summary, galaxy_per_tool_rows = result
                if galaxy_summary is None:
                    print(
                        f"  [{done_count:>3d}/{len(galaxy_ids)}] {gid} "
                        f"skipped (load failure)"
                    )
                    continue
                print(
                    f"  [{done_count:>3d}/{len(galaxy_ids)}] {gid} "
                    f"ok={galaxy_summary.total_ok} "
                    f"failed={galaxy_summary.total_failed} "
                    f"skipped_existing={galaxy_summary.total_skipped_existing}"
                )
                _accumulate_summary(summary, galaxy_summary)
                for tool, rows in galaxy_per_tool_rows.items():
                    per_tool_rows.setdefault(tool, {})[gid] = rows
            if fatal is not None:
                for other in pending:
                    other.cancel()
                pending = set()
        if fatal is not None:
            raise RuntimeError(
                "fail_fast: aborting remaining galaxies after worker failure"
            ) from fatal


def _accumulate_summary(target: RunSummary, addend: RunSummary) -> None:
    target.total_requested += addend.total_requested
    target.total_ran += addend.total_ran
    target.total_skipped_existing += addend.total_skipped_existing
    target.total_skipped_arm += addend.total_skipped_arm
    target.total_failed += addend.total_failed
    target.total_ok += addend.total_ok


def _write_summary_grids(
    plan: CampaignPlan,
    per_tool_rows: dict[str, dict[str, list[dict[str, Any]]]],
    dataset_dir: Path,
    campaign_dir: Path,
) -> None:
    """One profile grid + one residual grid per (tool, arm)."""
    for tool_name, rows_by_galaxy in per_tool_rows.items():
        tool_plan = plan.tools.get(tool_name)
        if tool_plan is None:
            continue
        grid_dir = dataset_dir / tool_name / "summary_grids"
        # Infer a representative SB zeropoint and pixel scale for the profile
        # grid from the first galaxy in the map; mixed datasets degrade
        # gracefully to log10(I).
        first_galaxy_bundle_hints = _first_galaxy_surface_brightness_hints(
            rows_by_galaxy
        )
        for arm_id in tool_plan.arms.keys():
            profile_path = grid_dir / f"summary_profiles_{_safe_arm_id(arm_id)}.png"
            residual_path = grid_dir / f"summary_residuals_{_safe_arm_id(arm_id)}.png"
            try:
                plot_summary_profiles(
                    rows_by_galaxy,
                    arm_id,
                    profile_path,
                    sb_zeropoint=first_galaxy_bundle_hints.get("sb_zeropoint"),
                    pixel_scale_arcsec=first_galaxy_bundle_hints.get("pixel_scale_arcsec"),
                )
                plot_summary_residuals(rows_by_galaxy, arm_id, residual_path)
            except Exception as exc:  # noqa: BLE001 - plots must not kill the run
                grid_dir.mkdir(parents=True, exist_ok=True)
                (grid_dir / f"summary_{_safe_arm_id(arm_id)}.err.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n"
                )


def _first_galaxy_surface_brightness_hints(
    rows_by_galaxy: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Find an arm row whose config.yaml surfaces the SB zeropoint + scale.

    We look at the first galaxy's MANIFEST.json (written by
    ``_write_manifest``). Returns an empty dict when none is found.
    """
    for galaxy_id, rows in rows_by_galaxy.items():
        for row in rows:
            config_path = row.get("config_path")
            if not config_path:
                continue
            manifest_path = Path(config_path).parents[2] / "MANIFEST.json"
            if not manifest_path.is_file():
                continue
            try:
                with manifest_path.open() as handle:
                    manifest = json.load(handle)
            except json.JSONDecodeError:
                continue
            return {
                "sb_zeropoint": manifest.get("sb_zeropoint"),
                "pixel_scale_arcsec": manifest.get("pixel_scale_arcsec"),
            }
    return {}


_TOOL_DISPATCH: dict[str, Any] = {
    "isoster": _isoster_run_one_arm,
    "photutils": _photutils_run_one_arm,
    "autoprof": _autoprof_run_one_arm,
}


def _run_tool_on_galaxy(
    plan: CampaignPlan,
    tool_plan: ToolPlan,
    bundle: GalaxyBundle,
    galaxy_dir: Path,
    *,
    sigma_info: dict[str, Any] | None = None,
) -> tuple[RunSummary, list[dict[str, Any]]]:
    tool_dir = galaxy_dir / tool_plan.name
    arms_dir = tool_dir / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary = RunSummary()

    for arm_id, arm_delta in tool_plan.arms.items():
        arm_dir = arms_dir / _safe_arm_id(arm_id)
        profile_path = arm_dir / "profile.fits"
        summary.total_requested += 1

        if plan.execution["skip_existing"] and profile_path.is_file():
            print(f"        [skip] {arm_id} (profile.fits exists)")
            summary.total_skipped_existing += 1
            rows.append(_load_cached_row(arm_dir, tool_plan.name, bundle.metadata.galaxy_id, arm_id))
            continue

        print(f"        [run ] {arm_id}")
        write_qa = plan.qa.get("per_galaxy_qa", True)
        write_model_fits = plan.qa.get("residual_models", True)
        fitter_fn = _TOOL_DISPATCH[tool_plan.name]
        # AutoProf needs a venv_python path from the tool's YAML extras.
        extra_kwargs: dict[str, Any] = {
            "sb_profile_scale": str(plan.qa.get("sb_profile_scale", "log10")),
        }
        if plan.qa.get("sb_asinh_softening") is not None:
            extra_kwargs["sb_asinh_softening"] = float(plan.qa["sb_asinh_softening"])
        if tool_plan.name in {"photutils", "autoprof"}:
            extra = tool_plan.extra or {}
            if tool_plan.name == "autoprof" and "venv_python" in extra:
                extra_kwargs["venv_python"] = str(extra["venv_python"])
            if "timeout" in extra:
                extra_kwargs["timeout"] = int(extra["timeout"])
        start = time.perf_counter()
        row = fitter_fn(
            bundle,
            arm_id,
            arm_delta,
            arm_dir,
            write_qa=write_qa,
            write_model_fits=write_model_fits,
            **extra_kwargs,
        )
        elapsed = time.perf_counter() - start
        status = row.get("status", "failed")
        if status == "skipped":
            summary.total_skipped_arm += 1
            print(f"             skipped: {row.get('error_msg', '')}")
        elif status == "ok":
            summary.total_ran += 1
            summary.total_ok += 1
            drift = row.get("combined_drift_pix")
            drift_s = f"{drift:.3f}" if isinstance(drift, (int, float)) and np.isfinite(drift) else "nan"
            flags = row.get("flags", "")
            flag_s = f"  flags={flags}" if flags else ""
            print(
                f"             ok  ({elapsed:5.2f}s)  n_iso={row.get('n_iso')} "
                f"drift={drift_s}px{flag_s}"
            )
        else:
            summary.total_ran += 1
            summary.total_failed += 1
            if plan.execution.get("fail_fast"):
                raise RuntimeError(
                    f"Arm {arm_id} on {bundle.metadata.galaxy_id} returned status={status!r}: "
                    f"{row.get('error_msg')}"
                )
            print(f"             FAIL: {row.get('error_msg', '')}")
        rows.append(row)

    # Phase C: attach composite_score using campaign weights, write
    # inventory + per-galaxy cross-arm table + overlay plot.
    weights = plan.summary.get("composite_score_weights") or None
    image_sigma = float(
        (sigma_info or {}).get("image_sigma_adu", 1.0)
    )
    sigma_method = str((sigma_info or {}).get("sigma_method", "unknown"))
    rows = apply_composite_scores(rows, weights, image_sigma=image_sigma)
    # Stamp every row with the sigma provenance so the inventory is
    # self-contained for the auditor.
    for row in rows:
        row.setdefault("image_sigma_adu", image_sigma)
        row.setdefault("sigma_method", sigma_method)
    inv_path = tool_dir / "inventory.fits"
    write_inventory(rows, inv_path)
    if plan.summary.get("per_galaxy_cross_arm_table", True):
        write_per_galaxy_cross_arm_table(rows, bundle.metadata.galaxy_id, tool_dir)
    if plan.qa.get("cross_arm_overlay", True):
        overlay_path = tool_dir / "cross_arm_overlay.png"
        try:
            plot_cross_arm_overlay(
                rows,
                overlay_path,
                sb_zeropoint=bundle.metadata.sb_zeropoint,
                pixel_scale_arcsec=bundle.metadata.pixel_scale_arcsec,
                title=f"{bundle.metadata.galaxy_id}  |  {tool_plan.name}",
                tool_name=tool_plan.name,
            )
        except Exception as exc:  # noqa: BLE001 - plot failures must not abort runs
            overlay_path.with_suffix(".png.err.txt").write_text(
                f"{type(exc).__name__}: {exc}\n"
            )
    return summary, rows


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _safe_arm_id(arm_id: str) -> str:
    return arm_id.replace(":", "_").replace("/", "_")


def _load_cached_row(
    arm_dir: Path, tool: str, galaxy_id: str, arm_id: str
) -> dict[str, Any]:
    """Rebuild an inventory row from an existing run_record.json."""
    record_path = arm_dir / "run_record.json"
    row: dict[str, Any] = {col: column_default(col) for col in INVENTORY_COLUMNS}
    row.update({"galaxy_id": galaxy_id, "tool": tool, "arm_id": arm_id, "status": "cached"})
    if not record_path.is_file():
        row["status"] = "cached_missing_record"
        return row
    with record_path.open() as handle:
        record = json.load(handle)
    row["status"] = record.get("status", "cached")
    row["wall_time_fit_s"] = float(record.get("wall_time_fit_s", float("nan")))
    row["wall_time_total_s"] = float(record.get("wall_time_total_s", float("nan")))
    metrics = record.get("metrics") or {}
    for key in INVENTORY_COLUMNS:
        if key in metrics:
            row[key] = metrics[key]
    row["flags"] = record.get("flags", "")
    row["flag_severity_max"] = float(record.get("flag_severity_max", 0.0))
    # Preserve output paths
    row["profile_path"] = str(arm_dir / "profile.fits")
    model_fits = arm_dir / "model.fits"
    if model_fits.is_file():
        row["model_path"] = str(model_fits)
    qa_png = arm_dir / "qa.png"
    if qa_png.is_file():
        row["qa_path"] = str(qa_png)
    row["config_path"] = str(arm_dir / "config.yaml")
    return row


def _write_manifest(
    bundle: GalaxyBundle,
    galaxy_dir: Path,
    *,
    sigma_info: dict[str, Any] | None = None,
) -> None:
    """Write MANIFEST.json summarizing the galaxy bundle for downstream tools."""
    md = bundle.metadata
    manifest = {
        "galaxy_id": md.galaxy_id,
        "dataset": md.dataset,
        "pixel_scale_arcsec": md.pixel_scale_arcsec,
        "sb_zeropoint": md.sb_zeropoint,
        "effective_Re_pix": md.effective_Re_pix,
        "redshift": md.redshift,
        "image_shape": list(np.asarray(bundle.image).shape),
        "has_variance": bundle.variance is not None,
        "has_mask": bundle.mask is not None,
        "initial_geometry": bundle.initial_geometry,
        "image_sigma": sigma_info or {},
        "extra": md.extra,
    }
    with (galaxy_dir / "MANIFEST.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, default=_json_default)


def _estimate_galaxy_sigma(bundle: GalaxyBundle) -> dict[str, Any]:
    """Compute once-per-galaxy ``image_sigma`` for the composite score."""
    geom = bundle.initial_geometry
    return compute_image_sigma(
        np.asarray(bundle.image, dtype=np.float64),
        x0=float(geom["x0"]),
        y0=float(geom["y0"]),
        eps=float(geom.get("eps", 0.2)),
        pa=float(geom.get("pa", 0.0)),
        R_ref=bundle.metadata.effective_Re_pix,
        maxsma=float(geom.get("maxsma", min(np.asarray(bundle.image).shape) // 2)),
        mask=bundle.mask,
        variance=bundle.variance,
    )


def _write_campaign_snapshot(plan: CampaignPlan, campaign_dir: Path) -> None:
    with (campaign_dir / "campaign.yaml").open("w") as handle:
        yaml.safe_dump(plan.raw, handle, sort_keys=False)


def _write_environment_snapshot(campaign_dir: Path) -> None:
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
        "astropy": _module_version("astropy"),
        "photutils": _module_version("photutils"),
        "isoster": _module_version("isoster"),
        "git_sha": _git_sha_short(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (campaign_dir / "environment.json").open("w") as handle:
        json.dump(env, handle, indent=2)


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _git_sha_short() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001 - best-effort only
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def run_from_yaml(yaml_path: Path) -> RunSummary:
    """Convenience: load the campaign YAML and run it."""
    plan = load_campaign(yaml_path)
    return run_campaign(plan)
