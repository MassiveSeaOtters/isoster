"""Where does the truth RMS of each zone come from: intensity noise or geometry error?

For every Isoster arm of a round-2 small-sample campaign, each input is rendered four
ways with the linear harmonic-off renderer and scored against the PSF-convolved truth
on one common pixel support:

* ``full``: the fit as saved in the campaign;
* ``geometry_only``: the saved geometry with the noiseless truth averaged along each
  saved ellipse, which removes the intensity noise and keeps the geometry error;
* ``intensity_only``: the geometry of a noise-free fit of the truth image with the saved
  (noisy) intensities at the same semi-major axes;
* ``noise_free``: the noise-free fit of the truth image with the arm's saved
  configuration, the method's floor without noise.

The saved isophotes are also compared one by one with the noise-free fit at the same
semi-major axis (ellipticity, position angle, centre, intensity) and with the truth
averaged along the saved ellipse (intensity noise given the geometry).

Run with ``uv run --with pandas --with tabulate python -m
benchmarks.exhausted.analysis.round2_outer_zone_decomposition --help``.
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
from isoster import IsosterConfig, fit_image
from isoster.model import build_isoster_model
from isoster.sampling import extract_isophote_data, prepare_sampling_image

ARMS = ("ref_default", "geom_ea", "int_median", "samp_cubic", "samp_cubic_ea", "samp_cubic_ea_median")
KINDS = ("full", "geometry_only", "intensity_only", "noise_free")
ZONES = ("all", "inner", "mid", "outer")
GEOMETRY = ("x0", "y0", "eps", "pa")


def truth_ring_means(truth, isophotes):
    """Mean of the noiseless truth along each saved ellipse, read with cubic sampling."""
    prepared, order = prepare_sampling_image(truth, "cubic")
    means = []
    for iso in isophotes:
        if not iso["sma"] > 0 or not np.all(np.isfinite([iso[k] for k in GEOMETRY])):
            means.append(iso["intens"])
            continue
        data = extract_isophote_data(
            prepared, None, iso["x0"], iso["y0"], iso["sma"], iso["eps"], iso["pa"], interpolation_order=order
        )
        means.append(float(np.mean(data.intens)) if len(data.intens) else np.nan)
    return means


def wrap_position_angle(delta):
    return (delta + np.pi / 2) % np.pi - np.pi / 2


def zone_of(sma, reference):
    return "inner" if sma < 0.5 * reference else ("outer" if sma >= 2 * reference else "mid")


def render(shape, isophotes):
    try:
        model = build_isoster_model(shape, isophotes, fill=np.nan, use_harmonics=False, interp_kind="linear")
    except (ValueError, TypeError, KeyError):
        return None
    return model if np.isfinite(model).any() else None


def analyze_input(row: dict, campaign: str, analysis: str) -> tuple[list[dict], list[dict]]:
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
    base = dict(galaxy=galaxy, scenario=scenario, depth=scenario.split("_")[0], reference_pix=reference)

    models, per_isophote = {}, []
    for arm in ARMS:
        arm_dir = input_dir / "isoster" / "arms" / arm
        if not (arm_dir / "profile.fits").exists():
            continue
        saved = profile_dicts(load_profile(arm_dir / "profile.fits"))
        config = IsosterConfig(**yaml.safe_load((arm_dir / "config.yaml").read_text()))
        noise_free = fit_image(truth, None, config)["isophotes"]
        truth_means = truth_ring_means(truth, saved)
        reference_sma = np.array([iso["sma"] for iso in noise_free], float)
        intensity_only = []
        for iso, truth_mean in zip(saved, truth_means):
            hit = np.flatnonzero(np.isclose(reference_sma, iso["sma"], rtol=1e-9, atol=1e-9))
            matched = noise_free[hit[0]] if hit.size else None
            if matched is not None:
                intensity_only.append(dict(matched, intens=iso["intens"]))
            if not iso["sma"] >= inner_cut:
                continue
            per_isophote.append(
                dict(
                    base,
                    arm=arm,
                    sma=iso["sma"],
                    zone=zone_of(iso["sma"], reference),
                    stop_code=iso["stop_code"],
                    intens=iso["intens"],
                    truth_ring_mean=truth_mean,
                    intens_noise_rel=(iso["intens"] - truth_mean) / truth_mean if truth_mean else np.nan,
                    d_intens_rel=(iso["intens"] - matched["intens"]) / matched["intens"] if matched else np.nan,
                    d_eps=iso["eps"] - matched["eps"] if matched else np.nan,
                    d_pa_deg=np.degrees(wrap_position_angle(iso["pa"] - matched["pa"])) if matched else np.nan,
                    d_center_pix=np.hypot(iso["x0"] - matched["x0"], iso["y0"] - matched["y0"]) if matched else np.nan,
                )
            )
        variants = dict(
            full=saved,
            geometry_only=[dict(iso, intens=value) for iso, value in zip(saved, truth_means)],
            intensity_only=intensity_only,
            noise_free=noise_free,
        )
        for kind, isophotes in variants.items():
            models[arm, kind] = render(image.shape, isophotes)

    common = eligible.copy()
    for model in models.values():
        if model is not None:
            common &= np.isfinite(model)
    metrics = []
    for (arm, kind), model in models.items():
        for zone in ZONES:
            mask = common & masks[zone] if model is not None else np.zeros(image.shape, dtype=bool)
            metrics.append(
                dict(
                    base,
                    arm=arm,
                    kind=kind,
                    zone=zone,
                    model_available=model is not None,
                    **pixel_metrics(model if model is not None else image, truth, image, mask, sigma),
                )
            )
    return metrics, per_isophote


def summarize(metrics, isophotes, output: Path) -> str:
    noisy = metrics[~metrics.scenario.eq("noiseless_z005") & metrics.model_available].copy()
    noisy["rms"] = 100 * noisy.truth_relative_rms
    lines = [
        "# Outer-zone decomposition of the round-2 small-sample campaign",
        "",
        f"{metrics.groupby(['galaxy', 'scenario']).ngroups} inputs; tables use the "
        f"{noisy.groupby(['galaxy', 'scenario']).ngroups} noisy inputs, linear harmonic-off renderer, "
        "median truth RMS in percent.",
        "",
    ]
    for zone in ZONES:
        table = noisy[noisy.zone.eq(zone)].pivot_table(index="arm", columns="kind", values="rms", aggfunc="median")
        lines += [
            f"## Zone `{zone}`: median truth RMS by model kind",
            "",
            table[list(KINDS)].round(3).to_markdown(),
            "",
        ]
    outer = noisy[noisy.zone.eq("outer")]
    by_depth = outer.pivot_table(index=["depth", "arm"], columns="kind", values="rms", aggfunc="median")
    lines += ["## Outer zone by depth", "", by_depth[list(KINDS)].round(3).to_markdown(), ""]
    wide = outer.pivot_table(index=["galaxy", "scenario"], columns=["arm", "kind"], values="rms")
    for kind in ("full", "geometry_only"):
        ratio = wide[("samp_cubic", kind)] / wide[("ref_default", kind)]
        lines.append(
            f"Outer zone, `{kind}`: cubic / bilinear RMS ratio median {ratio.median():.3f}, "
            f"cubic better in {100 * (ratio < 1).mean():.0f} % of noisy inputs."
        )
    lines.append("")

    iso = isophotes[~isophotes.scenario.eq("noiseless_z005")].copy()
    for column in ("intens_noise_rel", "d_intens_rel"):
        iso[column] = 100 * iso[column].abs()
    for column in ("d_eps", "d_pa_deg", "d_center_pix"):
        iso[column] = iso[column].abs()
    columns = ["intens_noise_rel", "d_intens_rel", "d_eps", "d_pa_deg", "d_center_pix"]
    scatter = iso.groupby(["zone", "arm"])[columns].median().reindex(ZONES[1:], level=0)
    scatter["n_isophotes"] = iso.groupby(["zone", "arm"]).size()
    scatter["flagged_fraction"] = iso.assign(flagged=iso.stop_code != 0).groupby(["zone", "arm"]).flagged.mean()
    lines += [
        "## Per-isophote deviations from the noise-free fit (medians of absolute values, noisy inputs)",
        "",
        "`intens_noise_rel`: |saved intensity - truth along the saved ellipse| / truth, percent. "
        "`d_intens_rel`: |saved - noise-free fit| / noise-free fit, percent. "
        "`d_eps`, `d_pa_deg` (degrees), `d_center_pix`: geometry versus the noise-free fit.",
        "",
        scatter.round(4).to_markdown(),
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.analysis / "fit_metrics.csv")
    inputs = frame.drop_duplicates(["galaxy", "scenario"])[
        ["galaxy", "scenario", "reference_pix", "inner_cut_pix", "injected_sigma"]
    ].to_dict("records")
    started = time.monotonic()
    metrics, isophotes = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_input, row, str(args.campaign), str(args.analysis)) for row in inputs]
        for done, future in enumerate(as_completed(futures), 1):
            m, i = future.result()
            metrics += m
            isophotes += i
            print(f"[{done}/{len(inputs)}] {time.monotonic() - started:.0f}s", flush=True)
    metrics, isophotes = pd.DataFrame(metrics), pd.DataFrame(isophotes)
    metrics.to_csv(args.output / "model_metrics.csv", index=False)
    isophotes.to_csv(args.output / "isophote_deviations.csv", index=False)
    print(summarize(metrics, isophotes, args.output))


if __name__ == "__main__":
    main()
