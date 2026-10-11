"""Exposure of the round-2 mock campaigns to the renderer odd-harmonic frame bug.

Before 1.1.1, ``build_isoster_model`` unwrapped the position angle by pi for
interpolation but did not negate the odd-order harmonics of the rings it rotated,
so one > pi/2 ``pa`` jump between consecutive rings rendered ``a3, b3`` with the
wrong sign over the rest of the galaxy. The benchmark's photutils model wrapper
(``baseline_shared.build_photutils_model_image``) had the same flaw. Only the
``native`` (harmonics-on) renderer rows of the campaign records are exposed; the
headline ``log_off`` renderer never used the harmonics.

Two steps, both refit-free:

1. ``census``: for every ok isoster and photutils fit, replicate the renderer's
   row filter and unwrap, and flag fits where any ring is rotated by an odd number
   of half-turns.
2. ``rescore``: for the flagged fits, render the native model with the fixed code
   and with the old behaviour (emulated exactly by pre-negating the same odd
   coefficients, which the fixed renderer then negates back), and score both
   against the saved truth image on the campaign's zones. The support is the
   evaluation aperture intersected with both models' finite pixels, not the
   campaign's common support across all arms, so the absolute numbers differ
   slightly from ``fit_metrics.csv``; the paired old-new difference is exact.

Usage (from the isoster repo root):

    uv run python -m benchmarks.exhausted.analysis.renderer_frame_fix census
    uv run python -m benchmarks.exhausted.analysis.renderer_frame_fix rescore --workers 8
    uv run python -m benchmarks.exhausted.analysis.renderer_frame_fix summary

Outputs go to ``outputs/benchmark_exhausted/renderer_frame_fix_20261012/``.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from benchmarks.benchmark_baseline.baseline_shared import build_photutils_model_image
from benchmarks.exhausted.analysis.publication_huang import load_profile, pixel_metrics, profile_dicts
from benchmarks.exhausted.analysis.residual_zones import (
    compute_elliptical_radius_grid,
    evaluation_aperture,
    zone_masks,
)
from isoster.model import build_isoster_model

ROOT = Path("/Volumes/galaxy/isophote/publication_single_band_round2_2026_10_10")
CAMPAIGNS = {
    "huang2013": ("fits/round2_huang2013_2026_10_10/huang2013", "analysis/huang2013_full_2026_10_10"),
    "s4g": ("fits/round2_s4g_2026_10_10/s4g", "analysis/s4g_full_2026_10_11"),
}
OUT = Path("outputs/benchmark_exhausted/renderer_frame_fix_20261012")
INNER_CUT_PIX = 2.0
ZONES = ("all", "inner", "mid", "outer")
TOOLS = ("isoster", "photutils")


def renderer_rows(table) -> list[dict]:
    """The rows the renderer keeps, in the order it keeps them."""
    rows = [r for r in profile_dicts(table) if r["sma"] > 0]
    rows = [r for r in rows if all(np.isfinite(r.get(k, np.nan)) for k in ("intens", "x0", "y0", "eps", "pa"))]
    return sorted(rows, key=lambda r: r["sma"])


def odd_half_turn_mask(rows: list[dict]) -> np.ndarray:
    pa_raw = np.array([r["pa"] for r in rows], dtype=float)
    half_turns = np.rint((np.unwrap(pa_raw, period=np.pi) - pa_raw) / np.pi).astype(int)
    return half_turns % 2 != 0


def odd_orders(rows: list[dict]) -> list[str]:
    keys = set()
    for r in rows:
        for k in r:
            m = re.match(r"^([ab])(\d+)$", str(k))
            if m and int(m.group(2)) % 2 == 1:
                keys.add(k)
    return sorted(keys)


def census_one(record_path: Path, dataset: str) -> dict | None:
    galaxy, scenario = record_path.parents[3].name.split("__")
    tool, arm = record_path.parents[2].name, record_path.parent.name
    if tool not in TOOLS:
        return None
    record = json.loads(record_path.read_text())
    out = dict(dataset=dataset, galaxy=galaxy, scenario=scenario, tool=tool, arm=arm, status=record["status"],
               record_path=str(record_path), affected=False, n_rotated=0, first_rotated_sma=np.nan,
               max_odd_amp_rotated=np.nan)
    if record["status"] != "ok":
        return out
    rows = renderer_rows(load_profile(record_path.parent / "profile.fits"))
    if len(rows) < 2:
        return out
    rotated = odd_half_turn_mask(rows)
    out["n_rotated"] = int(rotated.sum())
    if rotated.any():
        out["affected"] = True
        out["first_rotated_sma"] = float(rows[int(np.argmax(rotated))]["sma"])
        amps = [abs(float(r.get(k, 0.0))) for r, flag in zip(rows, rotated) if flag for k in odd_orders(rows)]
        out["max_odd_amp_rotated"] = float(np.nanmax(amps)) if amps else np.nan
    manifest = json.loads((record_path.parents[3] / "MANIFEST.json").read_text())
    out["effective_Re_pix"] = float(manifest["effective_Re_pix"])
    return out


def run_census(workers: int) -> pd.DataFrame:
    rows = []
    for dataset, (fits_rel, _) in CAMPAIGNS.items():
        paths = sorted((ROOT / fits_rel).glob("*/*/arms/*/run_record.json"))
        print(f"[census] {dataset}: {len(paths)} run records", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, r in enumerate(pool.map(census_one, paths, [dataset] * len(paths), chunksize=64)):
                if r is not None:
                    rows.append(r)
                if i % 5000 == 0 and i:
                    print(f"[census] {dataset}: {i} done", flush=True)
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "census.csv", index=False)
    ok = frame[frame.status.eq("ok")]
    print(ok.groupby(["dataset", "tool"]).affected.agg(["size", "sum", "mean"]))
    return frame


def native_model(tool: str, rows: list[dict], shape: tuple[int, int], emulate_old: bool) -> np.ndarray | None:
    if emulate_old:
        rotated = odd_half_turn_mask(rows)
        keys = odd_orders(rows) if tool == "isoster" else ["a3", "b3"]
        rows = [
            {**r, **({k: -float(r.get(k, 0.0)) for k in keys} if flag else {})} for r, flag in zip(rows, rotated)
        ]
    if tool == "isoster":
        return build_isoster_model(shape, rows, fill=np.nan, use_harmonics=True, interp_kind="linear")
    return build_photutils_model_image(shape, rows, fill=np.nan, high_harmonics=True)


def rescore_one(row: dict) -> list[dict]:
    record_path = Path(row["record_path"])
    _, analysis_rel = CAMPAIGNS[row["dataset"]]
    manifest = json.loads((record_path.parents[3] / "MANIFEST.json").read_text())
    geometry = manifest["initial_geometry"]
    image = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
    z_code = row["scenario"].split("_z")[-1]
    truth = fits.getdata(ROOT / analysis_rel / "galaxies" / row["galaxy"] / f"truth_z{z_code}.fits").astype(float)
    rows = renderer_rows(load_profile(record_path.parent / "profile.fits"))
    old = native_model(row["tool"], rows, image.shape, emulate_old=True)
    new = native_model(row["tool"], rows, image.shape, emulate_old=False)
    out = []
    if old is None or new is None:
        return out
    radius = compute_elliptical_radius_grid(image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"])
    eligible = evaluation_aperture(radius, geometry["maxsma"], INNER_CUT_PIX)
    masks = dict(zip(("inner", "mid", "outer"), zone_masks(radius, float(manifest["effective_Re_pix"]))))
    masks["all"] = np.ones(image.shape, dtype=bool)
    support = eligible & np.isfinite(old) & np.isfinite(new)
    for zone in ZONES:
        m = support & masks[zone]
        before = pixel_metrics(old, truth, image, m, np.nan)
        after = pixel_metrics(new, truth, image, m, np.nan)
        out.append(dict(
            {k: row[k] for k in ("dataset", "galaxy", "scenario", "tool", "arm")},
            zone=zone, npix=before["npix"],
            rms_old=before["truth_relative_rms"], rms_new=after["truth_relative_rms"],
            bias_old=before["flux_bias"], bias_new=after["flux_bias"],
        ))
    return out


def run_rescore(workers: int) -> pd.DataFrame:
    census = pd.read_csv(OUT / "census.csv")
    affected = census[census.status.eq("ok") & census.affected].to_dict("records")
    print(f"[rescore] {len(affected)} affected fits")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(rescore_one, affected, chunksize=8)):
            rows.extend(result)
            if i % 500 == 0 and i:
                print(f"[rescore] {i} fits done")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "rescore.csv", index=False)
    return frame


def run_summary() -> str:
    census = pd.read_csv(OUT / "census.csv")
    rescore = pd.read_csv(OUT / "rescore.csv")
    lines = ["# Renderer odd-harmonic frame fix: exposure of the round-2 mock campaigns", ""]
    ok = census[census.status.eq("ok")].copy()
    ok["first_rotated_over_Re"] = ok.first_rotated_sma / ok.effective_Re_pix
    cen = ok.groupby(["dataset", "tool", "arm"]).agg(
        fits=("affected", "size"), affected=("affected", "sum"),
        median_first_rotated_over_Re=("first_rotated_over_Re", "median"),
        median_max_odd_amp=("max_odd_amp_rotated", "median"),
    )
    cen["affected_pct"] = 100 * cen.affected / cen.fits
    lines += ["## Census: ok fits with at least one ring rotated by an odd number of half-turns", "",
              cen.round(3).to_markdown(), ""]
    rescore["rms_old_pct"] = 100 * rescore.rms_old
    rescore["rms_new_pct"] = 100 * rescore.rms_new
    rescore["delta_pct"] = rescore.rms_new_pct - rescore.rms_old_pct
    paired = rescore.groupby(["dataset", "tool", "arm", "zone"]).agg(
        n=("delta_pct", "size"), rms_old_median=("rms_old_pct", "median"), rms_new_median=("rms_new_pct", "median"),
        delta_median=("delta_pct", "median"), delta_p10=("delta_pct", lambda s: s.quantile(0.1)),
        delta_p90=("delta_pct", lambda s: s.quantile(0.9)),
    )
    lines += ["## Affected fits only: native truth RMS [%] old vs fixed renderer (paired)", "",
              paired.round(3).to_markdown(), ""]
    for dataset, (_, analysis_rel) in CAMPAIGNS.items():
        metrics = pd.read_csv(ROOT / analysis_rel / "fit_metrics.csv", usecols=[
            "galaxy", "scenario", "tool", "arm", "status", "renderer", "model_available", "zone", "truth_relative_rms"])
        native = metrics[metrics.renderer.eq("native") & metrics.status.eq("ok") & metrics.model_available
                         & metrics.tool.isin(TOOLS)].copy()
        native["rms_pct"] = 100 * native.truth_relative_rms
        patched = native.merge(
            rescore[rescore.dataset.eq(dataset)][["galaxy", "scenario", "tool", "arm", "zone", "rms_old_pct", "rms_new_pct"]],
            on=["galaxy", "scenario", "tool", "arm", "zone"], how="left")
        # Apply the paired shift to the archived value so the common support is kept.
        shift = (patched.rms_new_pct - patched.rms_old_pct).fillna(0.0)
        patched["rms_patched_pct"] = patched.rms_pct + shift
        table = patched.groupby(["tool", "arm", "zone"]).agg(
            n=("rms_pct", "size"), archived_median=("rms_pct", "median"), patched_median=("rms_patched_pct", "median"))
        table["change"] = table.patched_median - table.archived_median
        lines += [f"## {dataset}: campaign medians of the native renderer, archived vs patched by the paired shift", "",
                  table.round(3).to_markdown(), ""]
    text = "\n".join(lines)
    (OUT / "summary.md").write_text(text)
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("census", "rescore", "summary"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.step == "census":
        run_census(args.workers)
    elif args.step == "rescore":
        run_rescore(args.workers)
    else:
        run_summary()


if __name__ == "__main__":
    main()
