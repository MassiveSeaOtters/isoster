"""Demonstrate reconstruction alternatives on measured primary-tool discrepancies.

No refits or source writes. The native AutoProf model is NOT harmonic-enabled;
the common raw-polar reconstruction includes measured n=3,4 for all three tools.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from benchmarks.benchmark_baseline.baseline_shared import build_photutils_model_image
from benchmarks.exhausted.analysis.publication_huang import PRIMARY, digest, load_profile, pixel_metrics, profile_dicts
from benchmarks.exhausted.analysis.reconstruction_diagnostic import raw_polar_carrier
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid, evaluation_aperture, zone_masks
from benchmarks.exhausted.plotting.individual_qa_demo import cross_tool
from isoster.model import build_isoster_model
from isoster.plotting import build_method_profile, configure_qa_plot_style

MODE_LABELS = {
    "shared_baseline": "Shared linear / harmonics off",
    "shared_spline_off": "Shared spline / harmonics off",
    "shared_spline_on": "Shared spline / raw harmonics 3,4",
    "native": "Native: Isoster & Photutils harmonics on; AutoProf ellipse model",
}


def select_cases(frame, count):
    """Rank before new rendering, excluding small absolute differences and repeats."""
    selected = (
        frame[frame.primary & frame.status.eq("ok")]
        .pivot(index=["galaxy", "scenario"], columns="tool", values="truth_relative_rms_all")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=list(PRIMARY))
    )
    selected["ratio"] = selected.isoster / selected.autoprof
    selected["gap_pp"] = 100 * (selected.isoster - selected.autoprof)
    selected = selected[(selected.gap_pp > 1) & (selected.autoprof > 0)]
    selected = selected.sort_values(["ratio", "gap_pp"], ascending=False).reset_index().drop_duplicates("galaxy")
    if count < 1 or len(selected) < count:
        raise ValueError("Insufficient distinct eligible galaxies, or invalid case count")
    return selected.head(count)


def metric_figure(scores, title):
    """Individual-case measurements; no population uncertainty is implied."""
    configure_qa_plot_style()
    plt.rcParams["text.usetex"] = False
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), layout="constrained")
    for column, zone in enumerate(("all", "inner", "mid", "outer")):
        for row, metric in enumerate(("truth_relative_rms", "flux_bias_abs")):
            axis = axes[row, column]
            for tool, color, marker in zip(PRIMARY, ("#332288", "#008877", "#CC6677"), ("o", "s", "^")):
                values = scores[scores.tool.eq(tool) & scores.zone.eq(zone)].set_index("mode")
                axis.plot(range(4), 100 * values.loc[list(MODE_LABELS), metric], color=color, marker=marker, label=tool)
            axis.set_xticks(range(4), ["Linear\noff", "Spline\noff", "Spline\non", "Native*"], fontsize=10)
            axis.set_ylim(bottom=0)
            axis.set_ylabel("Truth RMS [%]" if row == 0 else "Absolute flux bias [%]")
            if row == 0:
                axis.set_title(zone.capitalize())
    axes[0, 0].legend(fontsize=10)
    fig.suptitle(title + " — matched pixels; *native AutoProf has no intensity harmonics", fontsize=14)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def track(path):
        path = Path(path)
        hashes[str(path)] = digest(path)
        return path

    frame = pd.read_csv(track(args.measurements / "fit_metrics.csv"))
    if not frame.inner_cut_pix.eq(2).all():
        raise ValueError("This demo requires the adopted two-pixel analysis")
    selection = select_cases(frame, args.count)
    selection.to_csv(args.output / "selection.csv", index=False)
    scores, baseline_checks, rendering = [], [], []
    for case in selection.itertuples():
        started = perf_counter()
        rows = frame[frame.galaxy.eq(case.galaxy) & frame.scenario.eq(case.scenario)]
        primary = rows[rows.primary].copy()
        if len(primary) != 3 or not primary.status.eq("ok").all():
            raise ValueError("Demonstration requires three successful designated primaries")
        tag = f"{case.galaxy}__{case.scenario}"
        destination = args.output / tag
        destination.mkdir()
        manifest = json.loads(track(Path(rows.iloc[0].record_path).parents[3] / "MANIFEST.json").read_text())
        geometry = manifest["initial_geometry"]
        image = fits.getdata(track(manifest["extra"]["fits_path"])).astype(float)
        truth = fits.getdata(track(args.measurements / "galaxies" / case.galaxy / f"truth_z{case.scenario[-3:]}.fits"))
        radius = compute_elliptical_radius_grid(
            image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
        )
        support = evaluation_aperture(radius, geometry["maxsma"], 2)
        profiles, tables, baselines = {}, {}, {}
        for row in rows.itertuples():
            track(row.record_path)
            if row.status != "ok":
                continue
            table = load_profile(track(Path(row.record_path).parent / "profile.fits"))
            records = profile_dicts(table)
            model = build_isoster_model(image.shape, records, fill=np.nan, use_harmonics=False)
            support &= np.isfinite(model)
            if row.primary:
                tables[row.tool] = table
                baselines[row.tool] = model
                profile = build_method_profile(records)
                if "cog" in table.colnames:
                    profile["cog"] = np.asarray(table["cog"], float)
                if row.tool == "autoprof":
                    profile.pop("stop_codes", None)
                profiles[row.tool, row.arm] = profile
        zones = dict(zip(("inner", "mid", "outer"), zone_masks(radius, primary.iloc[0].reference_pix)))
        zones["all"] = np.ones(image.shape, bool)
        for row in primary.itertuples():
            for zone, mask in zones.items():
                values = pixel_metrics(baselines[row.tool], truth, image, support & mask, row.injected_sigma)
                for key in ("npix", "truth_relative_rms", "flux_bias"):
                    np.testing.assert_allclose(values[key], getattr(row, f"{key}_{zone}"), rtol=1e-9, atol=1e-12)
                baseline_checks.append(
                    dict(galaxy=case.galaxy, scenario=case.scenario, tool=row.tool, zone=zone, **values)
                )
        original_support = support.copy()
        models = {"shared_baseline": baselines, "shared_spline_off": {}, "shared_spline_on": {}, "native": {}}
        for row in primary.itertuples():
            table = tables[row.tool]
            records = profile_dicts(table)
            raw = raw_polar_carrier(table, row.tool)
            for mode in ("shared_spline_off", "shared_spline_on", "native"):
                render_start = perf_counter()
                if mode.startswith("shared_spline"):
                    model = build_photutils_model_image(
                        image.shape, raw, high_harmonics=mode.endswith("on"), fill=np.nan
                    )
                elif row.tool == "isoster":
                    model = build_isoster_model(
                        image.shape, records, fill=np.nan, use_harmonics=True, harmonic_orders=[3, 4]
                    )
                elif row.tool == "photutils":
                    model = build_photutils_model_image(image.shape, records, high_harmonics=True, fill=np.nan)
                else:
                    options = json.loads(
                        track(Path(row.record_path).parent / "tmp" / f"{tag}_options.json").read_text()
                    )
                    if options.get("ap_set_background") != 0 or options.get("ap_isofit_fitcoefs") is not None:
                        raise ValueError("Native AutoProf model does not match the zero-background ellipse-only label")
                    native = Path(row.record_path).parent / "raw" / f"{tag}_genmodel.fits"
                    model = fits.getdata(track(native), ext=1).astype(float)
                    # Native AutoProf writes zero outside its supported positive-SB model.
                    model[model == 0] = np.nan
                if model is None or model.shape != image.shape:
                    raise ValueError(f"Missing/invalid reconstruction: {row.tool}/{mode}")
                models[mode][row.tool] = model
                support &= np.isfinite(model)
                rendering.append(
                    dict(
                        galaxy=case.galaxy,
                        scenario=case.scenario,
                        tool=row.tool,
                        mode=mode,
                        reconstruction_wall_s=perf_counter() - render_start,
                    )
                )
        if not support.any():
            raise ValueError("No common reconstruction support")
        fits.writeto(destination / "matched_support.fits", support.astype(np.uint8))
        residual_limit = max(
            float(
                np.percentile(
                    np.concatenate(
                        [np.abs((image - model)[support]) for group in models.values() for model in group.values()]
                    ),
                    99,
                )
            ),
            1e-10,
        )
        case_scores = []
        for mode, group in models.items():
            display = primary.copy()
            for row in primary.itertuples():
                model = group[row.tool]
                fits.writeto(destination / f"{row.tool}__{mode}.fits", model)
                for zone, mask in zones.items():
                    values = pixel_metrics(model, truth, image, support & mask, row.injected_sigma)
                    case_scores.append(
                        dict(
                            galaxy=case.galaxy,
                            scenario=case.scenario,
                            tool=row.tool,
                            mode=mode,
                            zone=zone,
                            inner_cut_pix=2,
                            original_npix=int(original_support.sum()),
                            support_fraction=float(support.sum() / original_support.sum()),
                            **values,
                        )
                    )
                    for key in ("truth_relative_rms", "flux_bias"):
                        display.loc[display.tool.eq(row.tool), f"{key}_{zone}"] = values[key]
            fig = cross_tool(
                display,
                profiles,
                {(tool, PRIMARY[tool]): model for tool, model in group.items()},
                image,
                support,
                manifest,
            )
            fig.suptitle(tag + " — " + MODE_LABELS[mode], y=0.985, fontsize=15)
            image_axes = [axis for axis in fig.axes if axis.images]
            for axis in image_axes[1:]:
                axis.images[0].set_clim(-residual_limit, residual_limit)
            for extension in ("png", "pdf"):
                fig.savefig(destination / f"{mode}.{extension}", dpi=180, bbox_inches="tight")
            plt.close(fig)
        scores.extend(case_scores)
        fig = metric_figure(pd.DataFrame(case_scores), tag)
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"zone_metrics.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        (destination / "caption.txt").write_text(
            f"{tag}: deliberately selected high Isoster/AutoProf baseline RMS ratio, not a population ranking.\n"
            f"Matched aperture: {support.sum()} of {original_support.sum()} original pixels; inner cut 2 pixels.\n"
            "All reconstruction alternatives use these same pixels, in all and three reference-radius zones.\n"
            "Truth RMS = 100 sqrt(sum((model-truth)^2)/sum(truth^2)); signed flux bias = 100 sum(model-truth)/sum(truth).\n"
            "Absolute flux bias is the absolute value of that signed sum, not a sum of absolute residuals.\n"
            "QA maps show full finite data-model in ADU without evaluation-cut masking; genuinely missing coverage stays blank.\n"
            "One pooled symmetric 99th-percentile scale, measured inside the common aperture, is used across all modes.\n"
            "Tables give ALL / INNER / MIDDLE / OUTER metrics on the common aperture, not the full displayed maps.\n"
            "Ellipses mark aperture limits, not bad-pixel masks. Native 1-D profiles do not change between pages.\n"
            "Shared baseline: linear Isoster renderer without harmonics. Shared spline off/on: Photutils renderer,\n"
            "same geometry and intensity profiles, calibrated raw polar n=3,4 amplitudes on. Synthetic gradient -1\n"
            "is only an exact coefficient transport, not an estimated physical AutoProf gradient.\n"
            "Native: Isoster radial-shift n=3,4; Photutils native additive n=3,4; saved AutoProf ellipse model\n"
            "WITHOUT measured intensity harmonics. AutoProf native output is not the harmonic-on alternative.\n"
            "Baseline reproduction checks retain the original all-arm support; displayed metrics use matched support.\n"
            "Fit times are inherited parallel-campaign measurements, unchanged by reconstruction; no controlled timing claim.\n"
        )
        print(
            f"[harmonic-demo] {tag}: 12 models, {support.sum()} common pixels, {perf_counter() - started:.3f} s",
            flush=True,
        )
    pd.DataFrame(scores).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(baseline_checks).to_csv(args.output / "baseline_reproduction.csv", index=False)
    pd.DataFrame(rendering).to_csv(args.output / "reconstruction_times.csv", index=False)
    assert all(digest(Path(path)) == value for path, value in hashes.items())
    source_files = [
        Path(__file__),
        Path("isoster/model.py"),
        Path("isoster/plotting.py"),
        Path("benchmarks/benchmark_baseline/baseline_shared.py"),
        Path("benchmarks/exhausted/analysis/reconstruction_diagnostic.py"),
        Path("benchmarks/harmonic_scale/conventions.py"),
        Path("benchmarks/exhausted/plotting/individual_qa_demo.py"),
    ]
    (args.output / "audit.json").write_text(
        json.dumps(
            dict(
                source_hashes=hashes,
                code_hashes={str(path): digest(path) for path in source_files},
                package_versions={
                    name: version(name) for name in ("numpy", "scipy", "photutils", "astropy", "matplotlib")
                },
                cases=len(selection),
                baseline_checks=len(baseline_checks),
                models=12 * len(selection),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
