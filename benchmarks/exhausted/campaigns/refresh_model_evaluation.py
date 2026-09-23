"""Refresh exhausted-campaign model metrics from persisted artifacts.

This command updates old campaign trees to the current v1.1 inventory
contract without refitting every arm. It reads each galaxy's manifest,
source image, per-arm profile, model, and run record; recomputes model
evaluation metrics and flags; then optionally rewrites run records,
inventories, and summary tables.

Photutils model FITS can be rebuilt from the stored profile with
``--refresh-photutils-harmonic-models``. This is useful for old runs
whose 2-D model image was rendered without harmonic deviations even
though the profile table contains harmonic columns.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table

from isoster import build_isoster_model

from ..analysis.inventory import INVENTORY_COLUMNS, column_default, write_inventory
from ..analysis.model_evaluation import evaluate_model_v11, profile_summary_for_inventory
from ..analysis.profile_io import read_eps, read_pa_in_radians
from ..analysis.quality_flags import evaluate_flags
from ..orchestrator.stats import apply_composite_scores, write_cross_tool_table, write_per_galaxy_cross_arm_table

TOOLS = ("isoster", "photutils", "autoprof")
SCORE_CONTEXT_FIELDS = ("composite_score", "image_sigma_adu", "n_iso_ref_used")


@dataclass(frozen=True)
class RefreshOptions:
    """Runtime options shared by refresh helpers."""

    write: bool = False
    refresh_photutils_harmonic_models: bool = False
    tools: tuple[str, ...] = TOOLS
    arms: tuple[str, ...] | None = None


@dataclass
class RefreshStats:
    """Simple counters for user-facing progress."""

    galaxies: int = 0
    arms_seen: int = 0
    arms_refreshed: int = 0
    arms_skipped: int = 0
    arms_failed: int = 0
    inventories_written: int = 0
    cross_tool_tables_written: int = 0

    def add(self, other: "RefreshStats") -> None:
        self.galaxies += other.galaxies
        self.arms_seen += other.arms_seen
        self.arms_refreshed += other.arms_refreshed
        self.arms_skipped += other.arms_skipped
        self.arms_failed += other.arms_failed
        self.inventories_written += other.inventories_written
        self.cross_tool_tables_written += other.cross_tool_tables_written


def enumerate_galaxy_dirs(
    campaign_root: Path,
    dataset: str,
    *,
    scenarios: set[str] | None = None,
    only: set[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Return galaxy directories under ``<campaign>/<dataset>``."""
    campaign_root = Path(campaign_root)
    galaxies: list[Path] = []
    for campaign_dir in sorted(campaign_root.iterdir()):
        if not campaign_dir.is_dir() or campaign_dir.name.startswith("_"):
            continue
        if scenarios and campaign_dir.name not in scenarios:
            continue
        dataset_dir = campaign_dir / dataset
        if not dataset_dir.is_dir():
            continue
        for galaxy_dir in sorted(dataset_dir.iterdir()):
            if not galaxy_dir.is_dir() or "__" not in galaxy_dir.name:
                continue
            if only and galaxy_dir.name not in only:
                continue
            galaxies.append(galaxy_dir)
            if limit is not None and len(galaxies) >= limit:
                return galaxies
    return galaxies


def refresh_galaxy(galaxy_dir: Path, options: RefreshOptions) -> tuple[RefreshStats, dict[str, list[dict[str, Any]]]]:
    """Refresh all selected tools/arms for one galaxy directory."""
    stats = RefreshStats(galaxies=1)
    rows_by_tool: dict[str, list[dict[str, Any]]] = {}
    manifest = _load_manifest(galaxy_dir)
    image = _load_image(manifest)
    image_sigma = _image_sigma_from_manifest(manifest)
    galaxy_id = str(manifest.get("galaxy_id", galaxy_dir.name))

    for tool in options.tools:
        tool_dir = galaxy_dir / tool
        arms_dir = tool_dir / "arms"
        if not arms_dir.is_dir():
            continue
        rows: list[dict[str, Any]] = []
        for arm_dir in sorted(p for p in arms_dir.iterdir() if p.is_dir()):
            arm_id = arm_dir.name
            if options.arms and arm_id not in options.arms:
                continue
            stats.arms_seen += 1
            try:
                row = refresh_arm(
                    arm_dir,
                    image=image,
                    manifest=manifest,
                    tool=tool,
                    arm_id=arm_id,
                    options=options,
                )
            except Exception as exc:  # noqa: BLE001 - keep campaign refresh moving
                row = _error_row(galaxy_id, tool, arm_id, arm_dir, exc)
                row["_refresh_traceback"] = traceback.format_exc()
            status = str(row.get("status", "")).lower()
            if row.get("_refresh_error"):
                stats.arms_failed += 1
            elif status in {"ok", "cached"}:
                stats.arms_refreshed += 1
            else:
                stats.arms_skipped += 1
            rows.append(row)

        if not rows:
            continue
        rows = apply_composite_scores(rows, image_sigma=image_sigma)
        rows_by_tool[tool] = rows
        if options.write:
            _write_score_context_to_run_records(rows)
            write_inventory(rows, tool_dir / "inventory.fits")
            write_per_galaxy_cross_arm_table(rows, galaxy_id, tool_dir)
            stats.inventories_written += 1
    return stats, rows_by_tool


def refresh_arm(
    arm_dir: Path,
    *,
    image: np.ndarray,
    manifest: dict[str, Any],
    tool: str,
    arm_id: str,
    options: RefreshOptions,
) -> dict[str, Any]:
    """Return a refreshed inventory row for one arm."""
    profile_path = arm_dir / "profile.fits"
    model_path = arm_dir / "model.fits"
    record_path = arm_dir / "run_record.json"
    record = _load_run_record(record_path)
    galaxy_id = str(manifest.get("galaxy_id", ""))
    row = _base_row(galaxy_id, tool, arm_id, arm_dir, record)

    if not profile_path.is_file():
        row["status"] = record.get("status", "missing_profile")
        row["error_msg"] = record.get("error_msg", record.get("reason", "profile.fits is missing"))
        return row

    if tool == "photutils" and options.refresh_photutils_harmonic_models:
        isophotes = _load_profile_as_isophotes(profile_path, tool=tool)
        model = build_isoster_model(image.shape, isophotes, use_harmonics=True)
        if options.write:
            _write_model_fits(model_path, model, image - model, arm_id, model_source="profile_harmonics_on")
        row["model_refresh"] = "profile_harmonics_on"
    else:
        model = _load_model(model_path)
        row["model_refresh"] = "reused_model_fits"

    if model is None:
        row["status"] = record.get("status", "missing_model")
        row["error_msg"] = record.get("error_msg", record.get("reason", "model.fits is missing or unreadable"))
        return row

    geometry = _metric_geometry(manifest, profile_path)
    profile_metrics = profile_summary_for_inventory(str(profile_path))
    model_metrics = evaluate_model_v11(
        image=image,
        model=model,
        mask=_load_mask(manifest),
        x0=geometry["x0"],
        y0=geometry["y0"],
        eps=geometry["eps"],
        pa_rad=geometry["pa_rad"],
        R_ref_pix=_safe_float(manifest.get("effective_Re_pix"), None),
        maxsma_pix=geometry["maxsma_pix"],
    )

    old_metrics = record.get("metrics") or {}
    metrics = {**old_metrics, **profile_metrics, **model_metrics}
    row.update(metrics)
    if str(row.get("status", "")).lower() in {"", "pending", "cached"}:
        row["status"] = record.get("status", "ok")
    row.update(evaluate_flags(row))

    if options.write:
        updated_record = dict(record)
        updated_record.update(
            {
                "status": row.get("status", "ok"),
                "metrics": metrics,
                "flags": row.get("flags", ""),
                "flag_severity_max": row.get("flag_severity_max", 0.0),
                "model_refresh": row.get("model_refresh", ""),
                "evaluation_refresh": {
                    "schema": "v1.1",
                    "source": "benchmarks.exhausted.campaigns.refresh_model_evaluation",
                },
            }
        )
        _write_json(record_path, updated_record)
    return row


def refresh_campaign(
    campaign_root: Path,
    dataset: str,
    *,
    scenarios: set[str] | None = None,
    only: set[str] | None = None,
    limit: int | None = None,
    options: RefreshOptions,
    max_parallel: int = 1,
    progress_every: int = 25,
) -> RefreshStats:
    """Refresh selected galaxy directories and rewrite scenario cross-tool tables."""
    galaxies = enumerate_galaxy_dirs(campaign_root, dataset, scenarios=scenarios, only=only, limit=limit)
    stats_total = RefreshStats()
    rows_by_dataset_dir: dict[Path, dict[str, dict[str, list[dict[str, Any]]]]] = {}

    def _collect(galaxy_dir: Path, stats: RefreshStats, rows_by_tool: dict[str, list[dict[str, Any]]]) -> None:
        nonlocal stats_total
        stats_total.add(stats)
        if rows_by_tool:
            dataset_dir = galaxy_dir.parent
            per_tool = rows_by_dataset_dir.setdefault(dataset_dir, {})
            manifest = _load_manifest(galaxy_dir)
            galaxy_id = str(manifest.get("galaxy_id", galaxy_dir.name))
            for tool, rows in rows_by_tool.items():
                per_tool.setdefault(tool, {})[galaxy_id] = rows

    if max_parallel <= 1:
        for index, galaxy_dir in enumerate(galaxies, start=1):
            stats, rows_by_tool = refresh_galaxy(galaxy_dir, options)
            _collect(galaxy_dir, stats, rows_by_tool)
            if progress_every > 0 and (index % progress_every == 0 or index == len(galaxies)):
                print(
                    f"  [{index}/{len(galaxies)}] arms refreshed={stats_total.arms_refreshed} "
                    f"skipped={stats_total.arms_skipped} failed={stats_total.arms_failed}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            future_to_galaxy = {executor.submit(refresh_galaxy, galaxy_dir, options): galaxy_dir for galaxy_dir in galaxies}
            for index, future in enumerate(as_completed(future_to_galaxy), start=1):
                galaxy_dir = future_to_galaxy[future]
                try:
                    stats, rows_by_tool = future.result()
                except Exception as exc:  # noqa: BLE001 - keep campaign refresh moving
                    stats = RefreshStats(galaxies=1, arms_failed=1)
                    rows_by_tool = {}
                    print(f"  [ERROR] {galaxy_dir}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                _collect(galaxy_dir, stats, rows_by_tool)
                if progress_every > 0 and (index % progress_every == 0 or index == len(galaxies)):
                    print(
                        f"  [{index}/{len(galaxies)}] arms refreshed={stats_total.arms_refreshed} "
                        f"skipped={stats_total.arms_skipped} failed={stats_total.arms_failed}",
                        flush=True,
                    )

    if options.write:
        for dataset_dir, per_tool_rows in rows_by_dataset_dir.items():
            if write_cross_tool_table(per_tool_rows, dataset_dir) is not None:
                stats_total.cross_tool_tables_written += 1
    return stats_total


def _base_row(galaxy_id: str, tool: str, arm_id: str, arm_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    row = {col: column_default(col) for col in INVENTORY_COLUMNS}
    row.update(
        {
            "galaxy_id": galaxy_id,
            "tool": tool,
            "arm_id": arm_id,
            "status": record.get("status", "cached"),
            "error_msg": record.get("error_msg", record.get("reason", "")),
            "wall_time_fit_s": _safe_float(record.get("wall_time_fit_s"), float("nan")),
            "wall_time_total_s": _safe_float(record.get("wall_time_total_s"), float("nan")),
            "qa_path": str(arm_dir / "qa.png") if (arm_dir / "qa.png").is_file() else "",
            "profile_path": str(arm_dir / "profile.fits") if (arm_dir / "profile.fits").is_file() else "",
            "model_path": str(arm_dir / "model.fits") if (arm_dir / "model.fits").is_file() else "",
            "config_path": str(arm_dir / "config.yaml") if (arm_dir / "config.yaml").is_file() else "",
            "flags": record.get("flags", ""),
            "flag_severity_max": _safe_float(record.get("flag_severity_max"), 0.0),
            "_record_path": str(arm_dir / "run_record.json"),
        }
    )
    for key, value in (record.get("metrics") or {}).items():
        if key in row:
            row[key] = value
    return row


def _error_row(galaxy_id: str, tool: str, arm_id: str, arm_dir: Path, exc: Exception) -> dict[str, Any]:
    row = _base_row(galaxy_id, tool, arm_id, arm_dir, {})
    row["status"] = "error_refresh"
    row["error_msg"] = f"{type(exc).__name__}: {exc}"
    row["_refresh_error"] = True
    row.update(evaluate_flags(row))
    return row


def _metric_geometry(manifest: dict[str, Any], profile_path: Path) -> dict[str, float]:
    geom = manifest.get("initial_geometry") or {}
    x0 = _safe_float(geom.get("x0"), None)
    y0 = _safe_float(geom.get("y0"), None)
    eps = _safe_float(geom.get("eps"), None)
    pa_rad = _safe_float(geom.get("pa"), None)
    maxsma_pix = _safe_float(geom.get("maxsma"), None)
    if None not in (x0, y0, eps, pa_rad, maxsma_pix):
        return {
            "x0": float(x0),
            "y0": float(y0),
            "eps": max(float(eps), 1e-6),
            "pa_rad": float(pa_rad),
            "maxsma_pix": float(maxsma_pix),
        }
    fallback = profile_summary_for_inventory(str(profile_path))
    return {
        "x0": float(x0 if x0 is not None else 0.0),
        "y0": float(y0 if y0 is not None else 0.0),
        "eps": max(float(eps if eps is not None else 0.0), 1e-6),
        "pa_rad": float(pa_rad if pa_rad is not None else 0.0),
        "maxsma_pix": float(maxsma_pix if maxsma_pix is not None else fallback.get("max_sma_pix", 1.0)),
    }


def _load_manifest(galaxy_dir: Path) -> dict[str, Any]:
    with (galaxy_dir / "MANIFEST.json").open() as handle:
        return json.load(handle)


def _load_run_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as handle:
        return json.load(handle)


def _load_image(manifest: dict[str, Any]) -> np.ndarray:
    extra = manifest.get("extra") or {}
    path = extra.get("fits_path")
    if not path:
        raise FileNotFoundError("manifest.extra.fits_path is missing")
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def _load_mask(manifest: dict[str, Any]) -> np.ndarray | None:
    extra = manifest.get("extra") or {}
    mask_path = extra.get("mask_path")
    if not mask_path:
        return None
    path = Path(str(mask_path))
    if not path.is_file():
        return None
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=bool)


def _load_model(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with fits.open(path) as hdul:
        for hdu in hdul:
            if getattr(hdu, "name", "").upper() == "MODEL" and hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float64)
        if hdul[0].data is not None:
            return np.asarray(hdul[0].data, dtype=np.float64)
    return None


def _load_profile_as_isophotes(path: Path, *, tool: str) -> list[dict[str, Any]]:
    table = _read_profile_table(path)
    pa_rad = read_pa_in_radians(table)
    eps = read_eps(table)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(table):
        item: dict[str, Any] = {}
        for col in table.colnames:
            value = row[col]
            if isinstance(value, np.generic):
                value = value.item()
            item[col.lower()] = value
        if pa_rad is not None:
            item["pa"] = float(pa_rad[idx])
        if eps is not None:
            item["eps"] = float(eps[idx])
        item["tool"] = tool
        rows.append(item)
    return rows


def _read_profile_table(path: Path) -> Table:
    with fits.open(path) as hdul:
        for index, hdu in enumerate(hdul):
            if getattr(hdu, "name", "").upper() == "ISOPHOTES" and getattr(hdu, "data", None) is not None:
                return Table.read(path, hdu=index)
        for index, hdu in enumerate(hdul[1:], start=1):
            if getattr(hdu, "data", None) is not None:
                return Table.read(path, hdu=index)
    raise ValueError(f"profile FITS has no readable table HDU: {path}")


def _write_model_fits(path: Path, model: np.ndarray, residual: np.ndarray, arm_id: str, *, model_source: str) -> None:
    primary = fits.PrimaryHDU(
        data=np.asarray(model, dtype=np.float32),
        header=fits.Header({"EXTNAME": "MODEL", "ARM_ID": arm_id, "MODSRC": model_source}),
    )
    residual_hdu = fits.ImageHDU(
        data=np.asarray(residual, dtype=np.float32),
        header=fits.Header({"EXTNAME": "RESIDUAL", "ARM_ID": arm_id}),
    )
    fits.HDUList([primary, residual_hdu]).writeto(path, overwrite=True)


def _image_sigma_from_manifest(manifest: dict[str, Any]) -> float:
    info = manifest.get("image_sigma") or {}
    sigma = _safe_float(info.get("image_sigma_adu"), 1.0)
    return max(float(sigma), 1e-6)


def _safe_float(value: Any, fallback: float | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(result):
        return fallback
    return result


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def _write_score_context_to_run_records(rows: list[dict[str, Any]]) -> None:
    """Store final per-tool score context in each arm's JSON record."""
    for row in rows:
        record_path = row.get("_record_path")
        if not record_path:
            continue
        path = Path(str(record_path))
        if not path.is_file():
            continue
        record = _load_run_record(path)
        metrics = dict(record.get("metrics") or {})
        for field in SCORE_CONTEXT_FIELDS:
            if field in row:
                metrics[field] = row[field]
        record["metrics"] = metrics
        _write_json(path, record)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="refresh_model_evaluation",
        description="Refresh v1.1 model metrics and inventories from existing exhausted-campaign artifacts.",
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scenario", nargs="*", default=None, help="Optional campaign directory names to refresh.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional galaxy directory names to refresh.")
    parser.add_argument("--tools", nargs="*", default=list(TOOLS), choices=list(TOOLS))
    parser.add_argument("--arms", nargs="*", default=None, help="Optional arm directory names to refresh.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of galaxy directories.")
    parser.add_argument("--write", action="store_true", help="Rewrite run records, inventories, and tables.")
    parser.add_argument("--max-parallel", type=int, default=1, help="Number of galaxy folders to refresh at once.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N galaxy folders.")
    parser.add_argument(
        "--refresh-photutils-harmonic-models",
        action="store_true",
        help="Rebuild photutils model.fits from profile.fits with harmonics enabled before metric refresh.",
    )
    args = parser.parse_args(argv)

    options = RefreshOptions(
        write=bool(args.write),
        refresh_photutils_harmonic_models=bool(args.refresh_photutils_harmonic_models),
        tools=tuple(args.tools),
        arms=tuple(args.arms) if args.arms else None,
    )
    scenarios = set(args.scenario) if args.scenario else None
    only = set(args.only) if args.only else None
    galaxies = enumerate_galaxy_dirs(args.campaign_root, args.dataset, scenarios=scenarios, only=only, limit=args.limit)
    print(
        f"Found {len(galaxies)} galaxy dirs under {args.campaign_root} / {args.dataset}"
        f" ({'write' if args.write else 'dry-run'})",
        flush=True,
    )
    if not galaxies:
        return 1
    if not args.write:
        for path in galaxies[:10]:
            print(f"  {path}", flush=True)
        if len(galaxies) > 10:
            print(f"  ... and {len(galaxies) - 10} more", flush=True)
        return 0

    stats = refresh_campaign(
        args.campaign_root,
        args.dataset,
        scenarios=scenarios,
        only=only,
        limit=args.limit,
        options=options,
        max_parallel=max(1, int(args.max_parallel)),
        progress_every=max(0, int(args.progress_every)),
    )
    print(
        "Refreshed "
        f"{stats.galaxies} galaxies, {stats.arms_refreshed}/{stats.arms_seen} arms; "
        f"skipped={stats.arms_skipped}, failed={stats.arms_failed}, "
        f"inventories={stats.inventories_written}, cross_tool_tables={stats.cross_tool_tables_written}",
        flush=True,
    )
    return 0 if stats.arms_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
