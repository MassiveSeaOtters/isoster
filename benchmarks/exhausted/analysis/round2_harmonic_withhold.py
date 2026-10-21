"""Should isoster withhold harmonics on converged rings where the gradient or the harmonic is noise?

Experiment on the round-2 mock campaigns (Huang2013, S4G) against the PSF-convolved truth
cached by ``round2_small_sample`` (``analysis/<run>/galaxies/<name>/truth_zNNN.fits``,
verified there to reproduce each campaign image exactly). Every stored isoster profile of
the chosen arms is re-drawn under each rule with two radial interpolators (``linear``, the
native renderer of the round-2 records, and ``log_linear``, the planned headline) and scored
by the truth-relative RMS per zone, on one pixel support per model shared by all its renders.
No refits: the rules act on the stored a_n, b_n, their errors, the relative gradient error
and the stop code, and the fitter never reads stored harmonics back.

Rules, applied to a3, b3, a4, b4 before ``build_isoster_model`` (whose >= 1 guard is on):

    stored    profile as written (in-loop arms keep failed-ring harmonics, isoster <= 1.1.1)
    current   isoster 1.1.2: zero on rings whose stop code is not 0 or 2
    off       no harmonics anywhere
    gradX     current, plus zero on rings whose relative gradient error exceeds X (or is not finite)
    snrK      current, plus zero order n where |(a_n, b_n)| / sqrt((err_a^2 + err_b^2) / 2) < K
    A+B       both cuts (``--rules current grad0.5+snr2 ...``)

Run with ``python -m benchmarks.exhausted.analysis.round2_harmonic_withhold --help``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits

from benchmarks.exhausted.analysis.publication_huang import load_profile, pixel_metrics, profile_dicts
from benchmarks.exhausted.analysis.residual_zones import (
    DEFAULT_INNER_CUT_PIX,
    compute_elliptical_radius_grid,
    evaluation_aperture,
    zone_masks,
)
from isoster.model import build_isoster_model

ROOT = Path("/Volumes/galaxy/isophote/publication_single_band_round2_2026_10_10")
SAMPLES = {
    "huang2013": ("fits/round2_huang2013_2026_10_10/huang2013", "analysis/huang2013_full_2026_10_10/galaxies"),
    "s4g": ("fits/round2_s4g_2026_10_10/s4g", "analysis/s4g_full_2026_10_11/galaxies"),
}
ARMS = ("ref_default", "samp_cubic", "samp_cubic_ea", "lsb_autolock", "stack_all", "harm_simul_ea")
RULES = ("stored", "current", "off", "grad0.5", "grad0.3", "grad0.1", "snr1", "snr2", "snr3")
COMBINED = ("grad0.5+snr2", "grad0.5+snr3")
INTERPS = ("linear", "log_linear")
ZONES = ("all", "inner", "mid", "outer")
ORDERS = (3, 4)


def withhold(table, rule: str):
    """Copy of ``table`` with the rule's harmonics zeroed, and the fraction of converged ring-orders it zeroed."""
    out = table.copy()
    if rule == "stored":
        return out, 0.0
    parts = rule.split("+")  # "grad0.5+snr2" applies both cuts
    converged = np.isin(np.asarray(out["stop_code"]), (0, 2))
    drop_ring = ~converged | (rule == "off")
    for part in (p for p in parts if p.startswith("grad")):
        drop_ring = drop_ring | ~(np.abs(np.asarray(out["grad_r_error"], float)) <= float(part[4:]))
    dropped = 0
    for n in ORDERS:
        drop = drop_ring.copy()
        for part in (p for p in parts if p.startswith("snr")):
            a, b = np.asarray(out[f"a{n}"], float), np.asarray(out[f"b{n}"], float)
            noise = np.sqrt(0.5 * (np.asarray(out[f"a{n}_err"], float) ** 2 + np.asarray(out[f"b{n}_err"], float) ** 2))
            with np.errstate(divide="ignore", invalid="ignore"):
                drop |= ~(np.hypot(a, b) / noise >= float(part[3:]))  # 0/0 (no harmonic) counts as dropped
        dropped += int(np.sum(drop & converged))
        for key in (f"a{n}", f"b{n}"):
            out[key] = np.where(drop, 0.0, np.asarray(out[key], float))
    return out, dropped / max(1, len(ORDERS) * int(converged.sum()))


def measure_galaxy(sample: str, name: str, arms: tuple[str, ...], inner_cut_pix: float, rules=RULES) -> list[dict]:
    fits_dir, truth_dir = (ROOT / p for p in SAMPLES[sample])
    rows = []
    for case in sorted(fits_dir.glob(f"{name}__*")):
        scenario = case.name.split("__")[1]
        manifest = json.loads((case / "MANIFEST.json").read_text())
        geometry = manifest["initial_geometry"]
        truth = fits.getdata(truth_dir / name / f"truth_{scenario.split('_')[1]}.fits").astype(float)
        radius = compute_elliptical_radius_grid(truth.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"])
        eligible = evaluation_aperture(radius, geometry["maxsma"], inner_cut_pix)
        masks = dict(zip(("inner", "mid", "outer"), zone_masks(radius, float(manifest["effective_Re_pix"]))))
        masks["all"] = np.ones(truth.shape, dtype=bool)
        for arm in arms:
            path = case / "isoster/arms" / arm / "profile.fits"
            if not path.is_file():
                continue
            table = load_profile(path)
            models, withheld = {}, {}
            try:
                for rule in rules:
                    ruled, withheld[rule] = withhold(table, rule)
                    dicts = profile_dicts(ruled)
                    for interp in INTERPS:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            models[rule, interp] = build_isoster_model(
                                truth.shape, dicts, fill=np.nan, use_harmonics=True, interp_kind=interp,
                                harmonic_min_snr=0,  # the rules above are the only cuts
                            )
            except (ValueError, TypeError, KeyError):  # same exclusions as round2_small_sample.render
                continue
            support = eligible & np.logical_and.reduce([np.isfinite(m) for m in models.values()])
            for (rule, interp), model in models.items():
                for zone in ZONES:
                    metrics = pixel_metrics(model, truth, truth, support & masks[zone], np.nan)
                    rows.append(
                        dict(
                            sample=sample, galaxy=name, scenario=scenario, arm=arm, rule=rule, interp=interp,
                            zone=zone, withheld_fraction=withheld[rule], npix=metrics["npix"],
                            truth_relative_rms=metrics["truth_relative_rms"], flux_bias=metrics["flux_bias"],
                        )
                    )
    return rows


def summarize(frame, rules=RULES) -> str:
    """Per sample, interpolator and zone: rule / current RMS ratio (median) and paired better / worse fractions."""
    lines = ["# Withholding noise harmonics: round-2 mocks", "",
             "Cell: median of (rule RMS / current RMS); percent of models better (ratio < 0.98) / worse (> 1.02).",
             "Bottom row of each table: median truth RMS of `current` in percent.", ""]
    for (sample, interp, zone), z in frame.groupby(["sample", "interp", "zone"], sort=False):
        rms = z.pivot_table(index=["galaxy", "scenario", "arm"], columns="rule", values="truth_relative_rms")
        ratio = rms.div(rms["current"], axis=0)
        lines += [f"## {sample}, `{interp}`, zone `{zone}`", "", "| rule | " + " | ".join(ARMS) + " |",
                  "|---|" + "---:|" * len(ARMS)]
        for rule in rules:
            cells = []
            for arm in ARMS:
                r = ratio.xs(arm, level="arm")[rule].dropna() if arm in ratio.index.get_level_values("arm") else []
                cells.append(f"{np.median(r):.3f}; {100 * np.mean(r < 0.98):.0f}/{100 * np.mean(r > 1.02):.0f}"
                             if len(r) else "")
            lines.append(f"| {rule} | " + " | ".join(cells) + " |")
        cur = rms["current"].groupby(level="arm").median() * 100
        lines += ["| current RMS % | " + " | ".join(f"{cur.get(a, np.nan):.3f}" for a in ARMS) + " |", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=sorted(SAMPLES), choices=sorted(SAMPLES))
    parser.add_argument("--limit", type=int, default=0, help="first N galaxies per sample (0: all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rules", nargs="+", default=list(RULES), help="must include current")
    parser.add_argument("--inner-cut-pix", type=float, default=DEFAULT_INNER_CUT_PIX)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for sample in args.samples:
        names = sorted({p.name.split("__")[0] for p in (ROOT / SAMPLES[sample][0]).glob("*__*")})
        jobs += [(sample, n) for n in names[: args.limit or None]]
    started, rows = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(measure_galaxy, s, n, ARMS, args.inner_cut_pix, args.rules) for s, n in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            rows += future.result()
            if index % 10 == 0 or index == len(jobs):
                print(f"[withhold] {index}/{len(jobs)} {time.monotonic() - started:.0f} s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "withhold_metrics.csv.gz", index=False)
    text = summarize(frame, args.rules)
    (args.output / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
