"""Refit the round-2 small-sample campaign with a radius-dependent sampling kernel.

Cubic ring sampling removes the bilinear read bias but reads about 20 % more noise per
ring (``ring_noise_response.py``), which costs a few percent of truth RMS in the
noise-dominated outskirts. Every campaign input is refitted from the saved ``samp_cubic``
configuration with rings beyond a switch radius read bilinearly: either an oracle radius
set per input from the known galaxy size or PSF (``sampling_cubic_max_sma``), or a
self-contained switch that fires on the rings' own statistics
(``sampling_cubic_min_snr``, ``sampling_cubic_max_relative_rms``). Each variant is
rendered with the linear harmonic-off renderer and scored against the truth by zone on
one common support, and the radius at which each switch fired is tabulated against the
reference radius. The ``bilinear`` and ``cubic`` variants must reproduce the
saved ``ref_default`` and ``samp_cubic`` profiles exactly, which ties the refits to the
campaign.

Run with ``uv run --with pandas --with tabulate python -m
benchmarks.exhausted.analysis.round2_kernel_switch_refit --help``.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from benchmarks.exhausted.analysis.publication_huang import load_profile, pixel_metrics, profile_dicts
from benchmarks.exhausted.analysis.residual_zones import (
    compute_elliptical_radius_grid,
    evaluation_aperture,
    zone_masks,
)
from benchmarks.exhausted.analysis.round2_outer_zone_decomposition import GEOMETRY, ZONES, render, zone_of
from isoster import IsosterConfig, fit_image

VARIANTS = {
    "bilinear": lambda row: dict(sampling_interpolation="bilinear"),
    "cubic": lambda row: {},
    # Oracle switch radii (need the galaxy size): kept as references.
    "cubic_max_1rref": lambda row: dict(sampling_cubic_max_sma=1 * row["reference_pix"]),
    "cubic_max_2rref": lambda row: dict(sampling_cubic_max_sma=2 * row["reference_pix"]),
    "cubic_max_5psf": lambda row: dict(sampling_cubic_max_sma=5 * row["psf_pix"]),
    # Self-contained switches driven by the ring's own statistics.
    "cubic_snr_30": lambda row: dict(sampling_cubic_min_snr=30.0),
    "cubic_snr_50": lambda row: dict(sampling_cubic_min_snr=50.0),
    "cubic_snr_100": lambda row: dict(sampling_cubic_min_snr=100.0),
    "cubic_snr_200": lambda row: dict(sampling_cubic_min_snr=200.0),
    "cubic_rms_0.1": lambda row: dict(sampling_cubic_max_relative_rms=0.1),
    "cubic_rms_0.2": lambda row: dict(sampling_cubic_max_relative_rms=0.2),
    "cubic_rms_0.4": lambda row: dict(sampling_cubic_max_relative_rms=0.4),
}
REPLAY = {"bilinear": "ref_default", "cubic": "samp_cubic"}


def same_profile(isophotes, saved) -> bool:
    if len(isophotes) != len(saved):
        return False
    for key in GEOMETRY + ("sma", "intens"):
        a = np.array([iso[key] for iso in isophotes], float)
        b = np.array([iso[key] for iso in saved], float)
        if not np.allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True):
            return False
    return True


def analyze_input(row: dict, campaign: str, analysis: str, repeats: int) -> list[dict]:
    galaxy, scenario = row["galaxy"], row["scenario"]
    input_dir = Path(campaign) / "huang2013" / f"{galaxy}__{scenario}"
    manifest = json.loads((input_dir / "MANIFEST.json").read_text())
    image = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
    truth = fits.getdata(Path(analysis) / "galaxies" / galaxy / f"truth_z{scenario[-3:]}.fits").astype(float)
    geometry = manifest["initial_geometry"]
    reference, inner_cut, sigma = float(row["reference_pix"]), float(row["inner_cut_pix"]), float(row["injected_sigma"])
    radius = compute_elliptical_radius_grid(
        image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
    )
    eligible = evaluation_aperture(radius, geometry["maxsma"], inner_cut)
    masks = dict(zip(("inner", "mid", "outer"), zone_masks(radius, reference)))
    masks["all"] = np.ones(image.shape, dtype=bool)
    arms = input_dir / "isoster" / "arms"
    saved_config = yaml.safe_load((arms / "samp_cubic" / "config.yaml").read_text())

    fits_by_variant, models, timings, replay, switches = {}, {}, {}, {}, {}
    for name, overrides in VARIANTS.items():
        config = IsosterConfig(**{**saved_config, **overrides(row)})
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = fit_image(image, None, config)
            elapsed.append(time.perf_counter() - started)
        fits_by_variant[name] = result["isophotes"]
        timings[name] = min(elapsed)
        # Radius beyond which rings were read bilinearly: the oracle's configured radius,
        # or the ring at which a self-contained switch fired (None when it never did).
        switches[name] = overrides(row).get("sampling_cubic_max_sma", result.get("sampling_switch_sma"))
        models[name] = render(image.shape, result["isophotes"])
        if name in REPLAY:
            saved = profile_dicts(load_profile(arms / REPLAY[name] / "profile.fits"))
            replay[name] = same_profile(result["isophotes"], saved)
    common = eligible.copy()
    for model in models.values():
        if model is not None:
            common &= np.isfinite(model)
    rows = []
    for name, isophotes in fits_by_variant.items():
        outer = [iso for iso in isophotes if iso["sma"] >= inner_cut and zone_of(iso["sma"], reference) == "outer"]
        base = dict(
            galaxy=galaxy,
            scenario=scenario,
            depth=scenario.split("_")[0],
            redshift=round(float(scenario[-3:]) / 100, 2),
            variant=name,
            switch_sma=np.nan if switches[name] is None else float(switches[name]),
            switch_over_reference=np.nan if switches[name] is None else float(switches[name]) / reference,
            fit_time_s=timings[name],
            fit_time_ratio=timings[name] / timings["bilinear"],
            replay_exact=replay.get(name),
            n_isophotes=len(isophotes),
            outer_isophotes=len(outer),
            outer_gradient_flagged=sum(iso["stop_code"] == -1 for iso in outer),
        )
        model = models[name]
        for zone in ZONES:
            mask = common & masks[zone] if model is not None else np.zeros(image.shape, dtype=bool)
            rows.append(
                dict(
                    base,
                    zone=zone,
                    model_available=model is not None,
                    **pixel_metrics(model if model is not None else image, truth, image, mask, sigma),
                )
            )
    return rows


def summarize(frame, output: Path) -> str:
    ok = frame[frame.model_available].copy()
    ok["rms"] = 100 * ok.truth_relative_rms
    variants = list(VARIANTS)
    lines = [
        "# Radius-dependent sampling kernel: refits of the round-2 small-sample campaign",
        "",
        f"{frame.groupby(['galaxy', 'scenario']).ngroups} inputs; linear harmonic-off renderer; "
        "truth RMS in percent, medians over inputs. Rings beyond `switch` return to bilinear reads.",
        "",
    ]
    checks = frame.drop_duplicates(["galaxy", "scenario", "variant"])
    for name in REPLAY:
        subset = checks[checks.variant.eq(name)]
        lines.append(f"Replay of `{REPLAY[name]}` by `{name}`: {int(subset.replay_exact.sum())}/{len(subset)} exact.")
    lines.append("")
    for zone in ZONES:
        z = ok[ok.zone.eq(zone)]
        wide = z.pivot_table(index=["galaxy", "scenario"], columns="variant", values="rms")
        table = z.groupby("variant").rms.median().to_frame("median_rms").reindex(variants)
        table["ratio_to_cubic"] = (wide.div(wide["cubic"], axis=0)).median()
        table["beats_cubic"] = (wide.lt(wide["cubic"], axis=0)).mean()
        table["beats_bilinear"] = (wide.lt(wide["bilinear"], axis=0)).mean()
        table["worst_ratio_to_cubic"] = (wide.div(wide["cubic"], axis=0)).max()
        lines += [f"## Zone `{zone}`", "", table.round(3).to_markdown(), ""]
    outer = ok[ok.zone.eq("outer")]
    for key in ("depth", "redshift"):
        table = outer.pivot_table(index=key, columns="variant", values="rms", aggfunc="median")[variants]
        lines += [f"## Outer zone by {key}", "", table.round(3).to_markdown(), ""]
    timing = checks.groupby("variant").fit_time_ratio.quantile([0.5, 0.9]).unstack().reindex(variants)
    timing.columns = ["median", "p90"]
    flagged = checks.groupby("variant")[["outer_gradient_flagged", "outer_isophotes"]].sum().reindex(variants)
    timing["outer_gradient_flagged_fraction"] = flagged.outer_gradient_flagged / flagged.outer_isophotes
    lines += [
        "## Fit time relative to bilinear, and outer rings held by the gradient check",
        "",
        timing.round(3).to_markdown(),
        "",
    ]
    switched = checks.groupby("variant").switch_sma.apply(lambda s: s.notna().mean()).to_frame("switched_fraction")
    switched["switch_over_reference_p10"] = checks.groupby("variant").switch_over_reference.quantile(0.1)
    switched["switch_over_reference_median"] = checks.groupby("variant").switch_over_reference.median()
    switched["switch_over_reference_p90"] = checks.groupby("variant").switch_over_reference.quantile(0.9)
    by_depth = checks.pivot_table(index="variant", columns="depth", values="switch_over_reference", aggfunc="median")
    switched = switched.join(by_depth.add_prefix("median_")).reindex(variants)
    lines += [
        "## Where the switch fired, in units of the reference radius (inputs where it fired)",
        "",
        switched.round(2).to_markdown(),
        "",
    ]
    text = "\n".join(lines)
    (output / "summary.md").write_text(text)
    return text


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument(
        "--analysis", required=True, type=Path, help="round2_small_sample.py output (truth images, metrics)"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.analysis / "fit_metrics.csv")
    inputs = frame.drop_duplicates(["galaxy", "scenario"])[
        ["galaxy", "scenario", "reference_pix", "inner_cut_pix", "injected_sigma", "psf_pix"]
    ].to_dict("records")
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(analyze_input, row, str(args.campaign), str(args.analysis), args.repeats) for row in inputs
        ]
        for done, future in enumerate(as_completed(futures), 1):
            rows += future.result()
            print(f"[{done}/{len(inputs)}] {time.monotonic() - started:.0f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "metrics.csv", index=False)
    print(summarize(frame, args.output))


if __name__ == "__main__":
    main()
