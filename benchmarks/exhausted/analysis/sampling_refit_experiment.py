"""Refit selected Huang2013 inputs with alternative Isoster sampling and rendering choices.

Each variant changes only the named IsosterConfig fields on top of the exact saved
campaign configuration, so the ``default`` variant must reproduce the saved
profile. Every fit is rendered with several harmonic-off radial interpolators and
scored on the accepted common support with the fixed 2-pixel cut, next to the saved
AutoProf primary rendered the same way. Fit wall time is the minimum of repeats.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from benchmarks.exhausted.analysis.publication_huang import digest, load_profile, pixel_metrics, profile_dicts
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid, zone_masks
from isoster import IsosterConfig, fit_image
from isoster.model import build_isoster_model

ROOT = Path("/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis")
BASE = ROOT / "huang2013_scientific_analysis_two_pixel_2026_09_23"
FINAL = ROOT / "huang2013_final_reconstruction_2026_09_23_v2"
VARIANTS = {
    "default": {},
    "cubic": {"sampling_interpolation": "cubic"},
    "ea": {"use_eccentric_anomaly": True},
    "median": {"integrator": "median"},
    "cubic_ea": {"sampling_interpolation": "cubic", "use_eccentric_anomaly": True},
    "cubic_median": {"sampling_interpolation": "cubic", "integrator": "median"},
    "cubic_ea_median": {"sampling_interpolation": "cubic", "use_eccentric_anomaly": True, "integrator": "median"},
    "cubic3_ea": {"sampling_interpolation": "cubic", "use_eccentric_anomaly": True, "sampling_cubic_min_sma": 3.0},
    "cubic4_ea": {"sampling_interpolation": "cubic", "use_eccentric_anomaly": True, "sampling_cubic_min_sma": 4.0},
}
RENDERERS = ("linear", "cubic", "log_linear")
ZONES = ("all", "inner", "mid", "outer")
GEOMETRY = ("sma", "x0", "y0", "eps", "pa", "intens", "stop_code")


def render(shape, isophotes, renderer, harmonics):
    model = build_isoster_model(shape, isophotes, fill=np.nan, use_harmonics=harmonics, interp_kind=renderer)
    return model if np.isfinite(model).any() else None


def run_case(galaxy, scenario, frame, output, repeats, variants):
    rows = frame[frame.galaxy.eq(galaxy) & frame.scenario.eq(scenario) & frame.primary]
    source = rows[rows.tool.eq("isoster")].iloc[0]
    record = Path(source.record_path).parent
    manifest = json.loads((record.parents[2] / "MANIFEST.json").read_text())
    image = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
    truth = fits.getdata(BASE / "galaxies" / galaxy / f"truth_z{scenario[-3:]}.fits").astype(float)
    support = fits.getdata(FINAL / f"{galaxy}__{scenario}" / "matched_support.fits").astype(bool)
    saved_config = yaml.safe_load((record / "config.yaml").read_text())
    saved = profile_dicts(load_profile(record / "profile.fits"))
    hashes = {str(p): digest(Path(p)) for p in (manifest["extra"]["fits_path"], record / "profile.fits")}

    profiles = {"isoster_saved": saved}
    timings = {"isoster_saved": float(source.wall_time_fit_s)}
    autoprof = rows[rows.tool.eq("autoprof")].iloc[0]
    if autoprof.status == "ok":
        profiles["autoprof_saved"] = profile_dicts(load_profile(Path(autoprof.record_path).parent / "profile.fits"))
        timings["autoprof_saved"] = float(autoprof.wall_time_fit_s)
    for name in variants:
        config = IsosterConfig(**{**saved_config, **VARIANTS[name]})
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = fit_image(image, None, config)
            elapsed.append(time.perf_counter() - started)
        profiles[name] = result["isophotes"]
        timings[name] = min(elapsed)

    # Exact replay check: the default variant must reproduce the saved campaign profile.
    replay = {}
    if "default" in variants:
        for key in GEOMETRY:
            a = np.array([iso[key] for iso in profiles["default"]], float)
            b = np.array([iso[key] for iso in saved], float)
            replay[key] = bool(len(a) == len(b) and np.allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True))

    models = {}
    for name, isophotes in profiles.items():
        for renderer in RENDERERS:
            models[(name, renderer, False)] = render(image.shape, isophotes, renderer, False)
        if name != "autoprof_saved":
            models[(name, "linear", True)] = render(image.shape, isophotes, "linear", True)
    common = support.copy()
    for model in models.values():
        if model is not None:
            common &= np.isfinite(model)
    geometry = manifest["initial_geometry"]
    radius = compute_elliptical_radius_grid(image.shape, **{k: geometry[k] for k in ("x0", "y0", "eps", "pa")})
    masks = dict(zip(ZONES[1:], zone_masks(radius, source.reference_pix)))
    masks["all"] = np.ones(image.shape, bool)
    metrics = []
    for (name, renderer, harmonics), model in models.items():
        isophotes = profiles[name]
        for zone in ZONES:
            mask = common & masks[zone] if model is not None else np.zeros(image.shape, bool)
            metrics.append(
                dict(
                    galaxy=galaxy,
                    scenario=scenario,
                    variant=name,
                    renderer=renderer,
                    harmonics=harmonics,
                    zone=zone,
                    fit_time_s=timings[name],
                    n_iso=len(isophotes),
                    n_stop0=int(sum(iso["stop_code"] == 0 for iso in isophotes)),
                    **pixel_metrics(model if model is not None else image, truth, image, mask, source.injected_sigma),
                )
            )
    destination = output / f"{galaxy}__{scenario}"
    destination.mkdir()
    pd.DataFrame(metrics).to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(
        [
            dict(variant=name, **{k: iso.get(k, np.nan) for k in GEOMETRY})
            for name, isos in profiles.items()
            for iso in isos
        ]
    ).to_csv(destination / "profiles.csv", index=False)
    audit = dict(
        galaxy=galaxy,
        scenario=scenario,
        common_pixels=int(common.sum()),
        support_pixels=int(support.sum()),
        replay_exact=replay,
        fit_time_s=timings,
        repeats=repeats,
        source_hashes=hashes,
    )
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return pd.DataFrame(metrics), replay


def summarize(frame):
    """ALL/INNER/MID/OUTER truth RMS and signed flux bias, in percent, per variant and renderer."""
    off = frame[~frame.harmonics].copy()
    off["rms"] = 100 * off.truth_relative_rms
    off["bias"] = 100 * off.flux_bias
    lines = []
    for (galaxy, scenario), group in off.groupby(["galaxy", "scenario"]):
        lines.append(f"\n{galaxy}/{scenario}  (fit time: min of repeats)")
        lines.append(
            f"{'variant':18s}{'render':11s}{'time[s]':>8s}  {'RMS all/in/mid/out [%]':>34s}  {'bias all/in/mid/out [%]':>36s}"
        )
        for (variant, renderer), g in group.groupby(["variant", "renderer"], sort=False):
            z = g.set_index("zone")
            rms = "/".join(f"{z.loc[k, 'rms']:.3f}" for k in ZONES)
            bias = "/".join(f"{z.loc[k, 'bias']:+.3f}" for k in ZONES)
            lines.append(f"{variant:18s}{renderer:11s}{z.fit_time_s.iloc[0]:8.3f}  {rms:>34s}  {bias:>36s}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="galaxy/scenario (repeatable)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    variants = [v for v in args.variants.split(",") if v]
    frame = pd.read_csv(BASE / "fit_metrics.csv")
    results = []
    for case in args.case:
        galaxy, scenario = case.split("/")
        metrics, replay = run_case(galaxy, scenario, frame, args.output, args.repeats, variants)
        results.append(metrics)
        print(f"{case}: replay exact = {replay}", flush=True)
    frame = pd.concat(results)
    frame.to_csv(args.output / "metrics_all.csv", index=False)
    text = summarize(frame)
    (args.output / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
