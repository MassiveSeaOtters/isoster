"""Authorized special-case refits: change only Isoster's fix_center flag."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from matplotlib.patches import Ellipse

from benchmarks.benchmark_baseline.baseline_shared import build_photutils_model_image
from benchmarks.exhausted.analysis.huang_final import native_model
from benchmarks.exhausted.analysis.publication_huang import digest, load_profile, pixel_metrics, profile_dicts
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid, zone_masks
from benchmarks.exhausted.plotting.individual_qa_demo import statistics_table
from isoster import IsosterConfig, fit_image, isophote_results_to_fits
from isoster.model import build_isoster_model
from isoster.plotting import build_method_profile, configure_qa_plot_style, plot_comparison_qa_figure

BASE = Path("/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis")
MEASUREMENTS = BASE / "huang2013_scientific_analysis_two_pixel_2026_09_23"
FINAL = BASE / "huang2013_final_reconstruction_2026_09_23_v2"
ZONES = ("all", "inner", "mid", "outer")
COLORS = {"free": "#0077BB", "fixed": "#CC6677", "autoprof": "#228833"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    hashes = {}

    def track(path):
        path = Path(path)
        hashes[str(path)] = digest(path)
        return path

    selection = pd.read_csv(track(args.output / "selection.csv")).iloc[args.case]
    galaxy, scenario = selection.galaxy, selection.scenario
    destination = args.output / f"{galaxy}__{scenario}"
    destination.mkdir(exist_ok=False)
    frame = pd.read_csv(track(MEASUREMENTS / "fit_metrics.csv"))
    rows = frame[frame.galaxy.eq(galaxy) & frame.scenario.eq(scenario) & frame.primary]
    source = rows[rows.tool.eq("isoster")].iloc[0]
    reference = rows[rows.tool.eq("autoprof")].iloc[0]
    parent = Path(source.record_path).parent
    manifest = json.loads(track(parent.parents[2] / "MANIFEST.json").read_text())
    geometry = manifest["initial_geometry"]
    assert not manifest["has_mask"] and not manifest["has_variance"]
    image = fits.getdata(track(manifest["extra"]["fits_path"])).astype(float)
    truth = fits.getdata(track(MEASUREMENTS / "galaxies" / galaxy / f"truth_z{scenario[-3:]}.fits"))
    config = yaml.safe_load(track(parent / "config.yaml").read_text())
    assert config["x0"] == geometry["x0"] and config["y0"] == geometry["y0"]
    assert config["fix_center"] is False
    free = load_profile(track(parent / "profile.fits"))
    autoprof = load_profile(track(Path(reference.record_path).parent / "profile.fits"))
    for row in (source, reference):
        track(row.record_path)
    replay_check = None
    if args.replay:
        replay = fit_image(image, None, IsosterConfig(**config))
        isophote_results_to_fits(replay, destination / "free_replay.fits", overwrite=False)
        replay_table = load_profile(destination / "free_replay.fits")
        replay_check = {}
        for name in ("sma", "x0", "y0", "eps", "pa", "intens", "grad", "stop_code"):
            np.testing.assert_allclose(replay_table[name], free[name], rtol=1e-10, atol=1e-10, equal_nan=True)
            replay_check[name] = float(np.nanmax(np.abs(replay_table[name] - free[name])))
    fixed_config = {**config, "fix_center": True}
    assert [k for k in config if config[k] != fixed_config[k]] == ["fix_center"]
    (destination / "config.yaml").write_text(yaml.safe_dump(fixed_config, sort_keys=False))
    fit_start = perf_counter()
    try:
        fitted = fit_image(image, None, IsosterConfig(**fixed_config))
        fit_seconds = perf_counter() - fit_start
        isophote_results_to_fits(fitted, destination / "fixed_profile.fits", overwrite=False)
        fixed = load_profile(destination / "fixed_profile.fits")
        assert np.all(fixed["x0"] == geometry["x0"]) and np.all(fixed["y0"] == geometry["y0"])
    except Exception as error:
        (destination / "failure.json").write_text(json.dumps(dict(error=repr(error), source_hashes=hashes), indent=2))
        raise
    tables = dict(free=free, fixed=fixed, autoprof=autoprof)
    models, outcomes = {}, []
    for mode in ("linear_off", "spline_off", "native"):
        models[mode] = {}
        for name, table in tables.items():
            records = profile_dicts(table)
            try:
                if mode == "linear_off":
                    model = build_isoster_model(image.shape, records, fill=np.nan, use_harmonics=False)
                elif mode == "spline_off":
                    model = build_photutils_model_image(image.shape, records, fill=np.nan, high_harmonics=False)
                else:
                    model = native_model(table, reference if name == "autoprof" else source, image.shape, track)
                assert np.isfinite(model).any()
                models[mode][name] = model
                outcomes.append(dict(mode=mode, name=name, status="ok", reason=""))
            except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                outcomes.append(dict(mode=mode, name=name, status="unavailable", reason=repr(error)))
    original = fits.getdata(track(FINAL / f"{galaxy}__{scenario}" / "matched_support.fits")).astype(bool)
    common = original.copy()
    for group in models.values():
        for model in group.values():
            common &= np.isfinite(model)
    radius = compute_elliptical_radius_grid(
        image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
    )
    assert np.all(radius[common] >= 2) and np.all(radius[common] <= geometry["maxsma"])
    zones = dict(zip(ZONES[1:], zone_masks(radius, source.reference_pix)))
    zones["all"] = np.ones(image.shape, bool)
    metrics = []
    for outcome in outcomes:
        model = models[outcome["mode"]].get(outcome["name"])
        for zone in ZONES:
            values = pixel_metrics(
                model if model is not None else image,
                truth,
                image,
                common & zones[zone] if model is not None else np.zeros(image.shape, bool),
                source.injected_sigma,
            )
            metrics.append(dict(galaxy=galaxy, scenario=scenario, **outcome, zone=zone, **values))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(destination / "metrics.csv", index=False)
    fits.writeto(destination / "common_support.fits", common.astype(np.uint8))
    radial = []
    for name, table in tables.items():
        for row in table:
            item = {
                key: float(row[key])
                for key in ("sma", "x0", "y0", "intens", "grad", "grad_r_error", "stop_code", "x0_err", "y0_err")
            }
            item.update(name=name, dx=item["x0"] - geometry["x0"], dy=item["y0"] - geometry["y0"])
            item["offset"] = float(np.hypot(item["dx"], item["dy"]))
            item["radius_over_reference"] = item["sma"] / source.reference_pix
            radial.append(item)
    radial = pd.DataFrame(radial)
    radial.to_csv(destination / "center_profiles.csv", index=False)
    configure_qa_plot_style()
    plt.rcParams["text.usetex"] = False
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for name, color in COLORS.items():
        values = radial[(radial.name == name) & (radial.sma >= 2)]
        for axis, key in zip(axes, ("dx", "dy", "grad_r_error")):
            axis.plot(values.sma, values[key], ".-", ms=3, color=color, label=name)
        bad = values[values.stop_code.ne(0)] if name != "autoprof" else values.iloc[:0]
        axes[0].scatter(bad.sma, bad.dx, marker="x", color=color, s=25)
    for axis, label in zip(axes, ("x − truth [pixel]", "y − truth [pixel]", "Gradient relative error")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.axvline(2 * source.reference_pix, color=".5", ls=":", label="Outer-zone boundary")
    axes[-1].axhline(config["maxgerr"], color=".5", ls="--")
    axes[-1].set(xlabel="Semi-major axis [pixel]", xscale="log")
    axes[0].legend()
    fig.suptitle(f"{galaxy} / {scenario}: centers relative to injected truth")
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"centers.{suffix}", dpi=150)
    plt.close(fig)
    profiles = {name: build_method_profile(profile_dicts(table)) for name, table in tables.items()}
    profiles["autoprof"].pop("stop_codes", None)
    for mode, group in models.items():
        fig = plot_comparison_qa_figure(
            image,
            profiles,
            models=group,
            mask=None,
            method_styles={name: dict(color=color, label=name) for name, color in COLORS.items()},
            sb_zeropoint=manifest["sb_zeropoint"],
            pixel_scale_arcsec=manifest["pixel_scale_arcsec"],
            sb_profile_scale="asinh",
            sb_asinh_softening=max(manifest["image_sigma"]["image_sigma_adu"], 1e-10),
            return_figure=True,
        )
        for axis in fig.axes:
            pos = axis.get_position()
            axis.set_position([pos.x0, 0.1 + (pos.y0 - 0.11) * 0.87, pos.width, pos.height * 0.87])
        fig.suptitle(f"{galaxy} / {scenario} — {mode}; center-only Isoster test", y=0.985, fontsize=16)
        display = []
        for name in tables:
            values = metrics[(metrics["mode"] == mode) & metrics.name.eq(name)].set_index("zone")
            item = dict(
                tool=name,
                arm="center test",
                status=values.iloc[0].status,
                flags="",
                wall_time_fit_s=np.nan,
                native_max_sma_pix=float(tables[name]["sma"].max()),
            )
            for zone in ZONES:
                for key in ("truth_relative_rms", "flux_bias"):
                    item[f"{key}_{zone}"] = values.loc[zone, key]
            display.append(item)
        fig.canvas.draw()
        data_axis = next(axis for axis in fig.axes if axis.images)
        left = data_axis.get_position().x0
        right = max(axis.get_position().x1 for axis in fig.axes)
        bottom = max(axis.get_position().y1 for axis in fig.axes) + 0.012
        statistics_table(
            fig.add_axes([left, bottom, right - left, 0.94 - bottom]), pd.DataFrame(display), cross_tool=True
        )
        for cut, color, label in ((2, "white", "Inner"), (geometry["maxsma"], "#ffcf40", "Outer")):
            data_axis.add_patch(
                Ellipse(
                    (geometry["x0"], geometry["y0"]),
                    2 * cut,
                    2 * cut * (1 - geometry["eps"]),
                    angle=np.degrees(geometry["pa"]),
                    fill=False,
                    color=color,
                    lw=1,
                    label=label,
                )
            )
        data_axis.legend(fontsize=6)
        limit = max(
            float(np.percentile(np.concatenate([np.abs((image - model)[common]) for model in group.values()]), 99)),
            1e-10,
        )
        for axis in fig.axes:
            if axis.images and axis is not data_axis:
                axis.images[0].set_clim(-limit, limit)
        for suffix in ("png", "pdf"):
            fig.savefig(destination / f"comparison_{mode}.{suffix}", dpi=150)
        plt.close(fig)
    (destination / "caption.txt").write_text(
        "Only fix_center changed. Fixed center is injected truth, not an estimated global center. "
        "Metrics use identical pixels across available methods/modes, within historical matched support and the 2-pixel cut. "
        "Residual maps show full finite coverage. Native Isoster includes n=3,4; native AutoProf is ellipse-only. "
        "linear_off and spline_off are independent harmonic-off controls. Times omitted from QA because inherited times are not controlled. "
        "QA centroid panel uses the shared plotter inner-reference center; centers.png and center_profiles.csv use injected truth. "
        "Cross marks in centers.png identify nonzero Isoster stop codes. AutoProf gradient errors are unavailable.\n"
    )
    for path, value in hashes.items():
        assert digest(Path(path)) == value, f"Source changed: {path}"
    audit = dict(
        galaxy=galaxy,
        scenario=scenario,
        fit_seconds=fit_seconds,
        elapsed_seconds=perf_counter() - started,
        changed_options=["fix_center"],
        replay_max_abs_difference=replay_check,
        original_pixels=int(original.sum()),
        common_pixels=int(common.sum()),
        source_hashes=hashes,
        outcomes=outcomes,
    )
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("source_hashes", "outcomes")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
