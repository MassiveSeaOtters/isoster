#!/usr/bin/env python3
"""Cross-mode sweep for the two LSB-regime features on HSC edge cases.

This is the final validation sweep for the renamed ``lsb_auto_lock`` and
``use_outer_center_regularization`` features. It crosses three feature arms
with three sampling/harmonic modes and records both the drift/lock behaviour
and the wall-clock runtime per galaxy per arm.

Feature arms
------------
- ``baseline``: free outward fit, no lock, no outer regularization.
- ``A`` (``lock``): ``lsb_auto_lock=True`` only.
- ``B`` (``lock_reg``): ``lsb_auto_lock=True`` + ``use_outer_center_regularization=True``
  at the new default ``outer_reg_strength=2.0`` (with onset=50, width=15).

Sampling / harmonic modes
-------------------------
- ``std``: default sampling (true anomaly) and post-hoc harmonics.
- ``ea``: ``use_eccentric_anomaly=True`` (steps psi uniformly, distributing
  samples more evenly around an elongated ellipse than stepping the position
  angle -- though not exactly uniformly in arc length).
- ``isofit``: ``simultaneous_harmonics=True`` (Ciambur 2015 joint fit),
  ``isofit_mode="in_loop"`` (the isoster default).

Cross
-----
The full sweep is 1 (baseline std) + 2 (A, B) × 3 (std, ea, isofit) = 7 arms.
``baseline_ea`` and ``baseline_isofit`` are skipped — the user's cross-check is
whether the two LSB features remain mode-agnostic, not a re-benchmark of the
baseline modes.

Metrics per galaxy × arm
------------------------
- ``elapsed``          : wall-clock seconds for ``fit_image``.
- ``n_iso``            : number of converged/attempted isophotes.
- ``lock_sma``         : sma where the auto-lock committed (``--`` if never).
- ``locked``           : count of isophotes in the locked tail.
- ``pre_lock_combined``: max sqrt(dx² + dy²) of *pre-lock* outward centroids
  relative to the outer-reg inner reference (or the anchor when no reg arm).
- ``pre_lock_rms``     : smoothing-spline RMS of x0(sma), y0(sma) on the
  pre-lock outward range (guards against small-sample cases with NaN).
- ``locked_drift``     : max |Δx| + max |Δy| on the *locked* tail (should be
  ~0 when the auto-lock hard lock is engaged).
- ``outward_drift``    : (max |dx|, max |dy|) over the whole outward range
  relative to the anchor isophote at ``sma = sma0``.
- ``stop_codes``       : compact histogram string.

Outputs
-------
- Per-galaxy QA figures and FITS:
  ``outputs/example_hsc_edgecases/lsb_mode_sweep/{arm}/{obj_id}/...``
- Summary tables are printed to stdout and saved as a Markdown/CSV pair at
  ``outputs/example_hsc_edgecases/lsb_mode_sweep/_summary.{md,csv}``.

Usage
-----
    uv run python examples/example_hsc_edgecases/run_lsb_mode_sweep.py
    uv run python examples/example_hsc_edgecases/run_lsb_mode_sweep.py --smoke
    uv run python examples/example_hsc_edgecases/run_lsb_mode_sweep.py \
        --galaxies 10140088 10140006
    uv run python examples/example_hsc_edgecases/run_lsb_mode_sweep.py \
        --arms baseline_std A_std A_ea A_isofit
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.rcParams["text.usetex"] = False

import numpy as np
from astropy.io import fits
from scipy.interpolate import UnivariateSpline

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model
from isoster.plotting import plot_qa_summary
from isoster.utils import isophote_results_to_fits

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "example_hsc_edgecases"
SWEEP_DIR = OUTPUT_ROOT / "lsb_mode_sweep"

GALAXIES = [
    ("10140088", "clear case"),
    ("10140002", "nearby bright star"),
    ("10140006", "nearby large galaxy"),
    ("10140009", "blending bright star"),
    ("10140056", "artifact"),
    ("10140093", "small blending source"),
]

BAND = "HSC_I"
SB_ZEROPOINT = 27.0
PIXEL_SCALE_ARCSEC = 0.168  # HSC coadd

# Shared free-fit base — identical to run_lsb_auto_lock.py and
# run_outer_center_regularization.py, so the sweep rows are directly
# comparable to the single-feature benchmarks.
BASE_CONFIG = dict(
    sma0=10.0,
    minsma=0.0,
    astep=0.1,
    linear_growth=False,
    eps=0.2,
    pa=0.0,
    fix_center=False,
    fix_pa=False,
    fix_eps=False,
    compute_deviations=True,
    full_photometry=True,
    compute_cog=True,
    debug=False,
    max_retry_first_isophote=5,
    # Asymmetric sigma clipping: clip bright outliers tighter (inner-region
    # improvement on HSC edge cases; see run_step1_sclip asymmetric variant).
    sclip_low=3.0,
    sclip_high=2.0,
    nclip=3,
)

# Feature arms (extra kwargs applied on top of BASE_CONFIG).
FEATURE_ARMS = {
    "baseline": dict(),
    "A": dict(
        lsb_auto_lock=True,
        lsb_auto_lock_maxgerr=0.3,
        lsb_auto_lock_debounce=2,
        lsb_auto_lock_integrator="median",
    ),
    "B": dict(
        lsb_auto_lock=True,
        lsb_auto_lock_maxgerr=0.3,
        lsb_auto_lock_debounce=2,
        lsb_auto_lock_integrator="median",
        use_outer_center_regularization=True,
        outer_reg_strength=2.0,
        outer_reg_sma_onset=50.0,
        outer_reg_sma_width=15.0,
    ),
}

# Sampling / harmonic modes.
MODE_KWARGS = {
    "std": dict(),
    "ea": dict(use_eccentric_anomaly=True),
    "isofit": dict(simultaneous_harmonics=True, isofit_mode="in_loop"),
}

# Default arm list: baseline gets one row (std only); A and B are crossed
# with all three modes. 7 arms total.
DEFAULT_ARMS = [
    "baseline_std",
    "A_std",
    "A_ea",
    "A_isofit",
    "B_std",
    "B_ea",
    "B_isofit",
]


def build_config(arm_name: str) -> dict:
    """Return a full kwargs dict for ``IsosterConfig`` for one arm."""
    feature, mode = arm_name.split("_", 1)
    if feature not in FEATURE_ARMS:
        raise ValueError(f"unknown feature '{feature}' in arm {arm_name!r}")
    if mode not in MODE_KWARGS:
        raise ValueError(f"unknown mode '{mode}' in arm {arm_name!r}")
    cfg = dict(BASE_CONFIG)
    cfg.update(FEATURE_ARMS[feature])
    cfg.update(MODE_KWARGS[mode])
    return cfg


def load_galaxy_data(obj_id: str):
    galaxy_dir = DATA_DIR / obj_id
    image = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_image.fits").astype(np.float64)
    variance = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_variance.fits").astype(np.float64)
    mask = fits.getdata(galaxy_dir / f"{obj_id}_{BAND}_mask.fits").astype(bool)
    return image, variance, mask


def stop_code_summary(isophotes):
    counts: dict = {}
    for iso in isophotes:
        code = iso.get("stop_code", -99)
        counts[code] = counts.get(code, 0) + 1
    return ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def pre_lock_outward(isophotes, sma0):
    """Outward isophotes with sma >= sma0, excluding any locked tail."""
    out = []
    for iso in isophotes:
        if iso.get("sma", 0.0) < sma0:
            continue
        if iso.get("lsb_locked", False):
            continue
        out.append(iso)
    return out


def locked_tail(isophotes):
    return [iso for iso in isophotes if iso.get("lsb_locked", False)]


def combined_drift(isos, x0_ref, y0_ref):
    if not isos:
        return 0.0, 0.0, 0.0
    dx = np.array([abs(iso["x0"] - x0_ref) for iso in isos])
    dy = np.array([abs(iso["y0"] - y0_ref) for iso in isos])
    return (
        float(np.max(dx)),
        float(np.max(dy)),
        float(np.sqrt(np.max(dx) ** 2 + np.max(dy) ** 2)),
    )


def spline_rms(isos):
    if len(isos) < 6:
        return float("nan")
    sma = np.array([iso["sma"] for iso in isos])
    x0s = np.array([iso["x0"] for iso in isos])
    y0s = np.array([iso["y0"] for iso in isos])
    order = np.argsort(sma)
    sma, x0s, y0s = sma[order], x0s[order], y0s[order]
    try:
        spl_x = UnivariateSpline(sma, x0s, k=3, s=len(sma))
        spl_y = UnivariateSpline(sma, y0s, k=3, s=len(sma))
    except Exception:
        return float("nan")
    rx = x0s - spl_x(sma)
    ry = y0s - spl_y(sma)
    return float(np.sqrt(np.mean(rx**2) + np.mean(ry**2)))


def outward_drift_from_anchor(isophotes, sma0):
    outward = [iso for iso in isophotes if iso["sma"] >= sma0]
    if not outward:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    anchor = outward[0]
    dx = np.array([abs(iso["x0"] - anchor["x0"]) for iso in outward])
    dy = np.array([abs(iso["y0"] - anchor["y0"]) for iso in outward])
    return (
        float(np.nanmax(dx)),
        float(np.nanmax(dy)),
        float(anchor["x0"]),
        float(anchor["y0"]),
    )


def locked_tail_drift(isophotes):
    tail = locked_tail(isophotes)
    if not tail:
        return float("nan"), float("nan")
    x0s = np.array([iso["x0"] for iso in tail])
    y0s = np.array([iso["y0"] for iso in tail])
    return float(x0s.max() - x0s.min()), float(y0s.max() - y0s.min())


def run_one(obj_id: str, desc: str, arm_name: str, arm_dir: Path, save_qa: bool):
    image, variance, mask = load_galaxy_data(obj_id)
    h, w = image.shape
    x0, y0 = w / 2.0, h / 2.0
    max_sma = min(h, w) / 2.0 - 10

    cfg_kwargs = build_config(arm_name)
    config = IsosterConfig(x0=x0, y0=y0, maxsma=max_sma, **cfg_kwargs)

    t0 = time.perf_counter()
    results = fit_image(image, mask=mask, config=config, variance_map=variance)
    elapsed = time.perf_counter() - t0

    isophotes = results["isophotes"]
    n_iso = len(isophotes)

    transition = results.get("lsb_auto_lock_sma")
    locked_count = results.get("lsb_auto_lock_count", 0)
    x0_ref = results.get("outer_reg_x0_ref")
    y0_ref = results.get("outer_reg_y0_ref")

    # Pick reference centroid for pre-lock drift: outer-reg inner ref when
    # available, else the anchor isophote at sma=sma0.
    if x0_ref is None or y0_ref is None:
        anchor_iso = next(
            (iso for iso in isophotes if iso.get("sma") == BASE_CONFIG["sma0"]),
            None,
        )
        if anchor_iso is None:
            anchor_iso = next(
                (iso for iso in isophotes if iso.get("sma", 0) >= BASE_CONFIG["sma0"]),
                isophotes[0],
            )
        x0_ref_metric = float(anchor_iso["x0"])
        y0_ref_metric = float(anchor_iso["y0"])
    else:
        x0_ref_metric = float(x0_ref)
        y0_ref_metric = float(y0_ref)

    pre_lock = pre_lock_outward(isophotes, BASE_CONFIG["sma0"])
    pl_mdx, pl_mdy, pl_combined = combined_drift(pre_lock, x0_ref_metric, y0_ref_metric)
    pl_rms = spline_rms(pre_lock)
    lock_dx, lock_dy = locked_tail_drift(isophotes)
    out_dx, out_dy, anchor_x0, anchor_y0 = outward_drift_from_anchor(isophotes, BASE_CONFIG["sma0"])

    if save_qa:
        galaxy_out = arm_dir / obj_id
        galaxy_out.mkdir(parents=True, exist_ok=True)
        tag = f"sweep_{arm_name}"
        isophote_results_to_fits(results, str(galaxy_out / f"{obj_id}_{tag}_results.fits"))
        model = build_isoster_model(image.shape, isophotes, use_harmonics=True)
        plot_qa_summary(
            title=f"{obj_id} — {desc} ({arm_name})",
            image=image,
            isoster_model=model,
            isoster_res=isophotes,
            mask=mask,
            filename=str(galaxy_out / f"{obj_id}_{tag}_qa.png"),
            relative_residual=False,
            sb_zeropoint=SB_ZEROPOINT,
            pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        )

    return {
        "obj_id": obj_id,
        "desc": desc,
        "arm": arm_name,
        "n_iso": n_iso,
        "elapsed": elapsed,
        "transition_sma": transition,
        "locked_count": locked_count,
        "x0_ref": x0_ref_metric,
        "y0_ref": y0_ref_metric,
        "pre_lock_max_dx": pl_mdx,
        "pre_lock_max_dy": pl_mdy,
        "pre_lock_combined": pl_combined,
        "pre_lock_rms": pl_rms,
        "locked_drift_x": lock_dx,
        "locked_drift_y": lock_dy,
        "outward_drift_x": out_dx,
        "outward_drift_y": out_dy,
        "anchor_x0": anchor_x0,
        "anchor_y0": anchor_y0,
        "stop_codes": stop_code_summary(isophotes),
    }


def print_runtime_table(rows, arms, galaxy_order):
    """Wide runtime table: one row per galaxy, one column per arm."""
    arm_w = max(7, max(len(a) for a in arms))
    header = f"  {'ID':>10s}  " + "  ".join(f"{a:>{arm_w}s}" for a in arms)
    print()
    print("  Runtime (s) per galaxy × arm")
    print(header)
    print("  " + "-" * (len(header) - 2))
    by_key = {(r["obj_id"], r["arm"]): r for r in rows}
    for obj_id in galaxy_order:
        cells = []
        for arm in arms:
            row = by_key.get((obj_id, arm))
            cells.append(f"{row['elapsed']:>{arm_w}.2f}" if row else f"{'--':>{arm_w}s}")
        print(f"  {obj_id:>10s}  " + "  ".join(cells))
    # Total row
    totals = []
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        tot = sum(r["elapsed"] for r in arm_rows) if arm_rows else float("nan")
        totals.append(f"{tot:>{arm_w}.1f}" if arm_rows else f"{'--':>{arm_w}s}")
    print(f"  {'TOTAL':>10s}  " + "  ".join(totals))


def print_drift_table(rows, arms, galaxy_order):
    """Per-galaxy drift/lock summary, grouped by galaxy then arm."""
    arm_w = max(7, max(len(a) for a in arms))
    print()
    print("  Drift + lock summary (pre_lock metrics relative to inner ref / anchor)")
    print(
        f"  {'ID':>10s}  {'arm':<{arm_w}s}  "
        f"{'n_iso':>5s}  {'T(s)':>5s}  {'lock_sma':>8s}  {'locked':>6s}  "
        f"{'pl_comb':>7s}  {'pl_dx':>6s}  {'pl_dy':>6s}  {'pl_rms':>6s}  "
        f"{'out_dx':>6s}  {'out_dy':>6s}  stop_codes"
    )
    print("  " + "-" * 122)
    by_galaxy = {}
    for r in rows:
        by_galaxy.setdefault(r["obj_id"], []).append(r)
    for obj_id in galaxy_order:
        gal_rows = sorted(by_galaxy.get(obj_id, []), key=lambda r: arms.index(r["arm"]))
        for r in gal_rows:
            ts = f"{r['transition_sma']:8.2f}" if r["transition_sma"] is not None else "      --"
            rms_str = f"{r['pre_lock_rms']:6.2f}" if np.isfinite(r["pre_lock_rms"]) else "   nan"
            print(
                f"  {r['obj_id']:>10s}  {r['arm']:<{arm_w}s}  "
                f"{r['n_iso']:>5d}  {r['elapsed']:5.1f}  {ts}  {r['locked_count']:>6d}  "
                f"{r['pre_lock_combined']:>7.2f}  {r['pre_lock_max_dx']:>6.2f}  "
                f"{r['pre_lock_max_dy']:>6.2f}  {rms_str}  "
                f"{r['outward_drift_x']:>6.2f}  {r['outward_drift_y']:>6.2f}  "
                f"{r['stop_codes']}"
            )
        print()


def write_summary_files(rows, arms, galaxy_order, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "_summary.csv"
    fieldnames = [
        "obj_id",
        "desc",
        "arm",
        "n_iso",
        "elapsed",
        "transition_sma",
        "locked_count",
        "x0_ref",
        "y0_ref",
        "pre_lock_max_dx",
        "pre_lock_max_dy",
        "pre_lock_combined",
        "pre_lock_rms",
        "locked_drift_x",
        "locked_drift_y",
        "outward_drift_x",
        "outward_drift_y",
        "anchor_x0",
        "anchor_y0",
        "stop_codes",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    md_path = out_dir / "_summary.md"
    lines = []
    lines.append(f"# LSB feature × mode sweep — HSC edge cases\n")
    lines.append(
        "Feature arms: `baseline` (free fit), `A` (`lsb_auto_lock=True`), "
        "`B` (A + `use_outer_center_regularization=True`, strength=2.0).\n"
    )
    lines.append(
        "Modes: `std` (default), `ea` (`use_eccentric_anomaly=True`), "
        "`isofit` (`simultaneous_harmonics=True`, `isofit_mode=in_loop`).\n"
    )
    lines.append("## Runtime table (s)\n")
    header = "| ID | " + " | ".join(arms) + " |"
    sep = "|" + "---|" * (len(arms) + 1)
    lines.append(header)
    lines.append(sep)
    by_key = {(r["obj_id"], r["arm"]): r for r in rows}
    for obj_id in galaxy_order:
        cells = []
        for arm in arms:
            row = by_key.get((obj_id, arm))
            cells.append(f"{row['elapsed']:.2f}" if row else "--")
        lines.append(f"| {obj_id} | " + " | ".join(cells) + " |")
    totals = []
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        totals.append(f"{sum(r['elapsed'] for r in arm_rows):.1f}" if arm_rows else "--")
    lines.append(f"| **TOTAL** | " + " | ".join(totals) + " |")
    lines.append("")
    lines.append("## Drift + lock summary\n")
    lines.append(
        "| ID | arm | n_iso | T(s) | lock_sma | locked | pl_comb | pl_dx | pl_dy | pl_rms | out_dx | out_dy | stop_codes |"
    )
    lines.append("|" + "---|" * 13)
    by_galaxy = {}
    for r in rows:
        by_galaxy.setdefault(r["obj_id"], []).append(r)
    for obj_id in galaxy_order:
        gal_rows = sorted(by_galaxy.get(obj_id, []), key=lambda r: arms.index(r["arm"]))
        for r in gal_rows:
            ts = f"{r['transition_sma']:.2f}" if r["transition_sma"] is not None else "--"
            rms = f"{r['pre_lock_rms']:.2f}" if np.isfinite(r["pre_lock_rms"]) else "nan"
            lines.append(
                f"| {r['obj_id']} | {r['arm']} | {r['n_iso']} | "
                f"{r['elapsed']:.1f} | {ts} | {r['locked_count']} | "
                f"{r['pre_lock_combined']:.2f} | {r['pre_lock_max_dx']:.2f} | "
                f"{r['pre_lock_max_dy']:.2f} | {rms} | "
                f"{r['outward_drift_x']:.2f} | {r['outward_drift_y']:.2f} | "
                f"{r['stop_codes']} |"
            )
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first galaxy (fast sanity check, still all arms).",
    )
    ap.add_argument(
        "--galaxies",
        nargs="+",
        default=None,
        help="Restrict to these galaxy IDs (default: all 6 HSC edge cases).",
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=f"Restrict to these arms. Default: {DEFAULT_ARMS}",
    )
    ap.add_argument(
        "--no-qa",
        action="store_true",
        help="Skip per-galaxy QA figure and FITS output (faster, less disk).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    galaxies = GALAXIES
    if args.smoke:
        galaxies = GALAXIES[:1]
    if args.galaxies:
        wanted = set(args.galaxies)
        galaxies = [g for g in GALAXIES if g[0] in wanted]
        if not galaxies:
            raise SystemExit(f"No matching galaxies for {args.galaxies}")
    galaxy_order = [g[0] for g in galaxies]

    arms = args.arms or DEFAULT_ARMS
    for arm in arms:
        build_config(arm)  # fail fast on unknown arm names

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("  LSB feature × mode sweep on HSC edge cases")
    print(f"  Arms: {arms}")
    print(f"  Galaxies: {galaxy_order}")
    print(f"  Output: {SWEEP_DIR}")
    print("=" * 76)

    rows = []
    for arm in arms:
        arm_dir = SWEEP_DIR / arm
        print(f"\n-- arm: {arm} --")
        for obj_id, desc in galaxies:
            print(f"  fitting {obj_id} ({desc}) ...", flush=True)
            row = run_one(obj_id, desc, arm, arm_dir, save_qa=not args.no_qa)
            ts = f"{row['transition_sma']:.2f}" if row["transition_sma"] is not None else "--"
            print(
                f"    {row['n_iso']} iso, {row['elapsed']:.1f}s, "
                f"lock_sma={ts}, locked={row['locked_count']}, "
                f"pl_comb={row['pre_lock_combined']:.2f} px"
            )
            rows.append(row)

    print_runtime_table(rows, arms, galaxy_order)
    print_drift_table(rows, arms, galaxy_order)

    csv_path, md_path = write_summary_files(rows, arms, galaxy_order, SWEEP_DIR)
    print(f"\nSummary CSV: {csv_path}")
    print(f"Summary MD : {md_path}")


if __name__ == "__main__":
    main()
