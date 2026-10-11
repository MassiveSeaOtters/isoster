#!/usr/bin/env python
"""Spot check of the cubic ring read on real SGA-2020 cutouts.

Fits a handful of ``benchmark_sample1`` galaxies three ways and compares them
ring by ring:

- ``bilinear``: the pre-1.1.0 read (``sampling_interpolation="bilinear"``);
- ``cubic``: the 1.1.0 default;
- ``cubic_nan_masked``: the cubic read on a copy of the image whose masked
  pixels are set to NaN, so that the prefilter sees the nearest unmasked value
  instead of the masked pixel's own value. The difference to ``cubic`` isolates
  what masked pixels (saturated stars, survey edges, zero-coverage holes) leak
  into the spline coefficients of their neighbours.

Every setting mirrors the ``sga_isoster`` campaign fitter: geometry from the SGA
ellipse table (astronomical PA converted to the math convention), ``sma0 =
max(8, 0.05 * MAJORAXIS)``, ``maxsma = min(ny, nx) // 2``, the campaign mask
(``>0`` = bad), the variance map ``1 / invvar`` (infinite where ``invvar <= 0``,
which isoster excludes), ``debug``, ``full_photometry`` and ``compute_cog`` on,
twelve first-isophote retries. Nothing is written into the data folders.

Usage::

    uv run python benchmarks/sga2020/cubic_spot_check.py --galaxies IC2535
    uv run python benchmarks/sga2020/cubic_spot_check.py            # the default dozen

Results go to ``outputs/benchmark_sga2020/cubic_spot_check/``: one CSV of
matched rings per galaxy, ``summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt, map_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isoster import fit_image  # noqa: E402
from isoster.config import IsosterConfig  # noqa: E402
from isoster.model import build_isoster_model  # noqa: E402
from isoster.numba_kernels import compute_ellipse_coords  # noqa: E402
from isoster.output_paths import resolve_output_directory  # noqa: E402

DATA_ROOT = Path("/Volumes/galaxy/isophote/sga2020/data/benchmark_sample1")

# Four galaxies per size tercile of MAJORAXIS among the first 400 of the sample;
# in each tercile the two with the most zero-inverse-variance pixels and the two
# with none (selection recorded in docs/agent/journal, 2026-10-11).
DEFAULT_GALAXIES = [
    "2MASXJ15313041+0615387",
    "2MASXJ12312909+3409146",
    "2MASXJ10510760+1951209",
    "2MASXJ16012223+4137029",
    "IC2535",
    "NGC7406",
    "ESO364-032",
    "ESO476-027",
    "NGC6289",
    "IC1256",
    "NGC3767",
    "IC5258",
]

ARMS = ("bilinear", "cubic", "cubic_nan_masked")
ZONES = (("inner", 0.0, 0.25), ("mid", 0.25, 1.0), ("outer", 1.0, np.inf))


def astro_pa_deg_to_math_rad(pa_astro_deg: float) -> float:
    """SGA astronomical PA (counter-clockwise from +y, degrees) to isoster's (from +x, radians)."""
    return float(np.deg2rad(float(pa_astro_deg) - 90.0) % np.pi)


def load_galaxy(name: str) -> dict:
    """Image, mask, variance and SGA geometry for one benchmark_sample1 galaxy."""
    folder = DATA_ROOT / name
    image = fits.getdata(glob.glob(f"{folder}/*-largegalaxy-image-r.fits.fz")[0], 1).astype(np.float64)
    invvar = fits.getdata(glob.glob(f"{folder}/*-largegalaxy-invvar-r.fits.fz")[0], 1).astype(np.float64)
    mask = fits.getdata(glob.glob(f"{folder}/*-mask.fits")[0]) > 0
    ellipse = fits.getdata(glob.glob(f"{folder}/*-ellipse.fits")[0], 1)
    row = 0
    if len(ellipse) > 1:
        sga_id = int(os.path.basename(glob.glob(f"{folder}/*-ellipse.fits")[0]).split("-")[-2])
        row = int(np.flatnonzero(ellipse["SGA_ID"] == sga_id)[0])
    variance = np.full(invvar.shape, np.inf)
    variance[invvar > 0] = 1.0 / invvar[invvar > 0]
    return dict(
        name=name,
        image=image,
        mask=mask,
        bad=mask | (invvar <= 0),
        variance=variance,
        x0=float(ellipse["X0"][row]),
        y0=float(ellipse["Y0"][row]),
        eps=float(min(0.9, max(0.0, ellipse["EPS"][row]))),
        pa=astro_pa_deg_to_math_rad(ellipse["PA"][row]),
        majoraxis=float(ellipse["MAJORAXIS"][row]),
    )


def campaign_config(galaxy: dict, **delta) -> IsosterConfig:
    ny, nx = galaxy["image"].shape
    return IsosterConfig(
        x0=galaxy["x0"],
        y0=galaxy["y0"],
        eps=galaxy["eps"],
        pa=galaxy["pa"],
        sma0=max(8.0, 0.05 * galaxy["majoraxis"]),
        maxsma=float(min(ny, nx) // 2),
        max_retry_first_isophote=12,
        debug=True,
        full_photometry=True,
        compute_cog=True,
        **delta,
    )


def fit_arm(galaxy: dict, arm: str, repeats: int) -> tuple[list[dict], float]:
    """Fit one arm; the wall time is the best of ``repeats`` runs."""
    image = galaxy["image"]
    delta = {}
    if arm == "bilinear":
        delta["sampling_interpolation"] = "bilinear"
    elif arm == "cubic_nan_masked":
        image = np.where(galaxy["bad"], np.nan, image)
    config = campaign_config(galaxy, **delta)
    best, result = np.inf, None
    for _ in range(repeats):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            result = fit_image(image, galaxy["mask"], config, variance_map=galaxy["variance"])
            best = min(best, time.perf_counter() - start)
    return result["isophotes"], best


def ring_masked_fraction(galaxy: dict, iso: dict) -> float:
    """Fraction of this ring's samples whose nearest pixel is masked or has no coverage."""
    sma = float(iso["sma"])
    n = max(64, int(2 * np.pi * sma))
    x, y, _, _ = compute_ellipse_coords(
        n, sma, float(iso["eps"]), float(iso["pa"]), float(iso["x0"]), float(iso["y0"]), False
    )
    bad = map_coordinates(galaxy["bad"].astype(np.float64), np.vstack([y, x]), order=0, mode="constant", cval=1.0)
    return float(bad.mean())


def model_residual(galaxy: dict, isophotes: list[dict]) -> dict:
    """Image minus the model rendered from the profile, in noise units.

    The campaign's data-residual metric: the native renderer (harmonics on, linear
    radial interpolation) turns each arm's profile into a 2-D model, and the RMS
    of ``(image - model) / sigma`` over unmasked, covered pixels says which profile
    describes the pixels better without needing a truth image. Zones use the
    elliptical radius of the SGA geometry in units of MAJORAXIS; the near-mask
    split (within 3 px of a masked or uncovered pixel) isolates what the masked
    pixels leak into the spline coefficients of their neighbours.
    """
    image = galaxy["image"]
    model = build_isoster_model(image.shape, isophotes)
    sigma = np.sqrt(galaxy["variance"])
    valid = np.isfinite(model) & (model != 0.0) & ~galaxy["bad"] & np.isfinite(sigma) & (sigma > 0)
    yy, xx = np.indices(image.shape)
    dx, dy = xx - galaxy["x0"], yy - galaxy["y0"]
    x_rot = dx * np.cos(galaxy["pa"]) + dy * np.sin(galaxy["pa"])
    y_rot = -dx * np.sin(galaxy["pa"]) + dy * np.cos(galaxy["pa"])
    r_ell = np.hypot(x_rot, y_rot / max(1.0 - galaxy["eps"], 0.1)) / galaxy["majoraxis"]
    near_mask = distance_transform_edt(~galaxy["bad"]) <= 3.0
    residual = (image - model) / sigma
    out = {}
    for zone, lo, hi in ZONES:
        sel = valid & (r_ell >= lo) & (r_ell < hi)
        if sel.sum() > 100:
            out[zone] = dict(
                n_pix=int(sel.sum()),
                rms_sigma=float(np.sqrt(np.mean(residual[sel] ** 2))),
                median_sigma=float(np.median(residual[sel])),
            )
    core = valid & (r_ell < 1.0)
    for label, sel in (("near_mask", core & near_mask), ("away_from_mask", core & ~near_mask)):
        if sel.sum() > 100:
            out[label] = dict(
                n_pix=int(sel.sum()),
                rms_sigma=float(np.sqrt(np.mean(residual[sel] ** 2))),
                median_sigma=float(np.median(residual[sel])),
            )
    out["flux_ratio_model_over_image"] = float(model[valid].sum() / image[valid].sum())
    return out


def matched_rings(reference: list[dict], other: list[dict]) -> list[tuple[dict, dict]]:
    """Pairs of isophotes at the same semi-major axis with a clean stop code on both sides."""
    by_sma = {round(float(iso["sma"]), 6): iso for iso in other if iso["stop_code"] == 0 and iso["sma"] > 0}
    return [
        (iso, by_sma[round(float(iso["sma"]), 6)])
        for iso in reference
        if iso["stop_code"] == 0 and iso["sma"] > 0 and round(float(iso["sma"]), 6) in by_sma
    ]


def compare(galaxy: dict, fits_by_arm: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """Per-ring rows (arm vs bilinear) and a per-galaxy summary."""
    rows, summary = [], {}
    bilinear = fits_by_arm["bilinear"]
    masked_fraction = {
        round(float(iso["sma"]), 6): ring_masked_fraction(galaxy, iso) for iso in bilinear if iso["sma"] > 0
    }
    for arm in ("cubic", "cubic_nan_masked"):
        pairs = matched_rings(bilinear, fits_by_arm[arm])
        for ref, iso in pairs:
            sma = float(ref["sma"])
            rows.append(
                dict(
                    galaxy=galaxy["name"],
                    arm=arm,
                    sma=sma,
                    sma_over_majoraxis=sma / galaxy["majoraxis"],
                    masked_fraction=masked_fraction[round(sma, 6)],
                    intens_bilinear=float(ref["intens"]),
                    intens_arm=float(iso["intens"]),
                    rel_intens=float(iso["intens"] / ref["intens"] - 1.0) if ref["intens"] else np.nan,
                    rel_intens_in_sigma=float((iso["intens"] - ref["intens"]) / ref["intens_err"])
                    if ref.get("intens_err")
                    else np.nan,
                    d_eps=float(iso["eps"] - ref["eps"]),
                    d_pa_deg=float(np.rad2deg(((iso["pa"] - ref["pa"]) + np.pi / 2) % np.pi - np.pi / 2)),
                    d_center_px=float(np.hypot(iso["x0"] - ref["x0"], iso["y0"] - ref["y0"])),
                    niter_bilinear=int(ref["niter"]),
                    niter_arm=int(iso["niter"]),
                )
            )
        arm_rows = [r for r in rows if r["arm"] == arm]
        zone_stats = {}
        for zone, lo, hi in ZONES:
            z = [r for r in arm_rows if lo <= r["sma_over_majoraxis"] < hi]
            if z:
                d = np.array([r["rel_intens"] for r in z])
                zone_stats[zone] = dict(
                    n=len(z),
                    median_rel_intens=float(np.median(d)),
                    rms_rel_intens=float(np.sqrt(np.mean(d**2))),
                    max_abs_rel_intens=float(np.max(np.abs(d))),
                    median_abs_in_sigma=float(np.median(np.abs([r["rel_intens_in_sigma"] for r in z]))),
                    median_d_eps=float(np.median([r["d_eps"] for r in z])),
                    median_d_pa_deg=float(np.median([r["d_pa_deg"] for r in z])),
                    median_d_center_px=float(np.median([r["d_center_px"] for r in z])),
                )
        touched = [r["rel_intens"] for r in arm_rows if r["masked_fraction"] > 0.05]
        clean = [r["rel_intens"] for r in arm_rows if r["masked_fraction"] <= 0.05]
        summary[arm] = dict(
            matched_rings=len(arm_rows),
            zones=zone_stats,
            rings_touching_masked_pixels=len(touched),
            rms_rel_intens_touching=float(np.sqrt(np.mean(np.square(touched)))) if touched else None,
            rms_rel_intens_clean=float(np.sqrt(np.mean(np.square(clean)))) if clean else None,
        )
    return rows, summary


def _rms(values) -> float:
    values = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def aggregate(summaries: list[dict], rows: list[dict]) -> list[str]:
    """Cross-galaxy summary lines for the markdown record."""
    lines = ["## Across the sample", ""]
    n_gal = len(summaries)
    walls = {arm: [s["wall_s"][arm] for s in summaries] for arm in ARMS}
    lines.append(
        f"- {n_gal} galaxies, {sum(len(s['n_isophotes']) and s['n_isophotes']['bilinear'] for s in summaries)} bilinear rings; "
        f"ring counts identical across arms in {sum(len(set(s['n_isophotes'].values())) == 1 for s in summaries)} of {n_gal} galaxies; "
        f"non-zero stop codes: bilinear {sum(s['n_stop_nonzero']['bilinear'] for s in summaries)}, cubic {sum(s['n_stop_nonzero']['cubic'] for s in summaries)}."
    )
    lines.append(
        f"- Wall time, best of repeats, summed over the sample: bilinear {sum(walls['bilinear']):.2f} s, cubic {sum(walls['cubic']):.2f} s "
        f"(ratio {sum(walls['cubic']) / sum(walls['bilinear']):.2f}; per-galaxy ratio median {np.median(np.array(walls['cubic']) / np.array(walls['bilinear'])):.2f})."
    )
    lines += [
        "",
        "### Ring-by-ring, cubic minus bilinear",
        "",
        "| zone (sma / MAJORAXIS) | rings | median % | RMS % | median abs in sigma | rings beyond 3 sigma | median d eps | median d PA deg | median centre shift px |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for zone, lo, hi in ZONES:
        z = [r for r in rows if r["arm"] == "cubic" and lo <= r["sma_over_majoraxis"] < hi]
        if not z:
            continue
        rel = np.array([r["rel_intens"] for r in z]) * 100
        sig = np.abs([r["rel_intens_in_sigma"] for r in z])
        lines.append(
            f"| {zone} ({lo:g}-{hi:g}) | {len(z)} | {np.median(rel):+.3f} | {_rms(rel):.2f} | {np.median(sig):.2f} | {100 * np.mean(sig > 3):.0f} % "
            f"| {np.median([r['d_eps'] for r in z]):+.4f} | {np.median([r['d_pa_deg'] for r in z]):+.2f} | {np.median([r['d_center_px'] for r in z]):.3f} |"
        )
    # Leakage of masked pixels into the coefficients: the NaN-filled cubic arm against the plain cubic arm.
    by_key = {(r["galaxy"], r["arm"], round(r["sma"], 6)): r for r in rows}
    leak = []
    for (galaxy, arm, sma), r in by_key.items():
        other = by_key.get((galaxy, "cubic_nan_masked", sma))
        if arm == "cubic" and other is not None:
            leak.append(
                (r["masked_fraction"], (1 + other["rel_intens"]) / (1 + r["rel_intens"]) - 1, r["sma_over_majoraxis"])
            )
    leak = np.array(leak) if leak else np.zeros((0, 3))
    lines += [
        "",
        "### Masked pixels filled before the prefilter (cubic with NaN-masked pixels minus plain cubic), rings inside MAJORAXIS",
        "",
    ]
    for label, sel in (
        ("rings touching masked pixels (> 5 % of samples)", leak[:, 0] > 0.05),
        ("clean rings", leak[:, 0] <= 0.05),
    ):
        sel = sel & (leak[:, 2] < 1.0)
        if sel.any():
            d = leak[sel, 1] * 100
            lines.append(
                f"- {label}: {int(sel.sum())} rings, median {np.median(d):+.3f} %, RMS {_rms(d):.2f} %, max |d| {np.max(np.abs(d)):.2f} %."
            )
    lines += [
        "",
        "### Data residual, RMS of (image - model) / sigma over unmasked covered pixels",
        "",
        "| region | bilinear | cubic | cubic, masked pixels filled | galaxies where cubic beats bilinear |",
        "|---|---|---|---|---|",
    ]
    for region in ("inner", "mid", "outer", "near_mask", "away_from_mask"):
        vals = {
            arm: [s["data_residual"][arm][region]["rms_sigma"] for s in summaries if region in s["data_residual"][arm]]
            for arm in ARMS
        }
        if not vals["bilinear"]:
            continue
        wins = sum(c < b for b, c in zip(vals["bilinear"], vals["cubic"]))
        lines.append(
            f"| {region} | {np.median(vals['bilinear']):.3f} | {np.median(vals['cubic']):.3f} | {np.median(vals['cubic_nan_masked']):.3f} | {wins} of {len(vals['bilinear'])} |"
        )
    lines.append("")
    lines.append(
        "Medians over galaxies of each galaxy's RMS. The `near_mask` and `away_from_mask` rows cover pixels inside MAJORAXIS within and beyond 3 px of a masked or uncovered pixel."
    )
    return lines


def write_markdown(summaries: list[dict], rows: list[dict], path: Path) -> None:
    lines = [
        "# Cubic ring read on SGA-2020 cutouts: spot check",
        "",
        "Relative intensity differences are (arm - bilinear) / bilinear over rings matched at the same",
        "semi-major axis with stop code 0 on both sides; zones are sma / MAJORAXIS of the SGA ellipse;",
        "`sigma` is the bilinear ring's own intensity error. Wall times are the best of the timing repeats.",
        "",
        "| galaxy | side px | MAJORAXIS px | rings bil / cubic | stop!=0 bil / cubic | wall bil / cubic s | cubic-bil median % inner / mid | inner median abs sigma | rings touching mask | residual RMS sigma inner bil / cubic | residual RMS sigma mid bil / cubic |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        c = s["arms"]["cubic"]

        def zone(key, zone_name, scale=100.0, fmt=".2f"):
            z = c["zones"].get(zone_name)
            return format(scale * z[key], fmt) if z else "-"

        def resid(arm, region):
            r = s["data_residual"][arm].get(region)
            return f"{r['rms_sigma']:.3f}" if r else "-"

        lines.append(
            f"| {s['galaxy']} | {s['side']} | {s['majoraxis']:.0f} | {s['n_isophotes']['bilinear']} / {s['n_isophotes']['cubic']} "
            f"| {s['n_stop_nonzero']['bilinear']} / {s['n_stop_nonzero']['cubic']} "
            f"| {s['wall_s']['bilinear']:.2f} / {s['wall_s']['cubic']:.2f} "
            f"| {zone('median_rel_intens', 'inner')} / {zone('median_rel_intens', 'mid')} "
            f"| {zone('median_abs_in_sigma', 'inner', 1.0)} | {c['rings_touching_masked_pixels']} "
            f"| {resid('bilinear', 'inner')} / {resid('cubic', 'inner')} | {resid('bilinear', 'mid')} / {resid('cubic', 'mid')} |"
        )
    lines += [""] + aggregate(summaries, rows)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--galaxies", nargs="*", default=DEFAULT_GALAXIES)
    parser.add_argument("--repeats", type=int, default=2, help="timing repeats per arm; the best is kept")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    out_dir = Path(
        resolve_output_directory("benchmark_sga2020", "cubic_spot_check", explicit_output_directory=args.output)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, all_rows = [], []
    for name in args.galaxies:
        galaxy = load_galaxy(name)
        fits_by_arm, wall = {}, {}
        for arm in ARMS:
            fits_by_arm[arm], wall[arm] = fit_arm(galaxy, arm, args.repeats)
        rows, arm_summary = compare(galaxy, fits_by_arm)
        residuals = {arm: model_residual(galaxy, fits_by_arm[arm]) for arm in ARMS}
        summary = dict(
            galaxy=name,
            side=int(galaxy["image"].shape[0]),
            majoraxis=galaxy["majoraxis"],
            eps_sga=galaxy["eps"],
            bad_pixels=int(galaxy["bad"].sum()),
            masked_fraction=float(galaxy["mask"].mean()),
            n_isophotes={arm: len(fits_by_arm[arm]) for arm in ARMS},
            n_stop_nonzero={arm: int(sum(iso["stop_code"] != 0 for iso in fits_by_arm[arm])) for arm in ARMS},
            wall_s=wall,
            arms=arm_summary,
            data_residual=residuals,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        with open(out_dir / f"{name}_rings.csv", "w") as handle:
            if rows:
                handle.write(",".join(rows[0].keys()) + "\n")
                for r in rows:
                    handle.write(",".join(str(v) for v in r.values()) + "\n")
        c = arm_summary["cubic"]
        print(
            f"{name}: {summary['side']} px, MAJORAXIS {galaxy['majoraxis']:.0f}; rings {summary['n_isophotes']}; "
            f"wall {wall['bilinear']:.2f}/{wall['cubic']:.2f}/{wall['cubic_nan_masked']:.2f} s; cubic-bilinear outer RMS "
            f"{100 * c['zones'].get('outer', {}).get('rms_rel_intens', np.nan):.2f} %; rings touching mask {c['rings_touching_masked_pixels']}",
            flush=True,
        )
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, default=float))
    write_markdown(summaries, all_rows, out_dir / "summary.md")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
