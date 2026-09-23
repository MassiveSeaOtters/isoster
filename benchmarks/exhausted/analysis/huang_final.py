"""Retained-population reconstruction/QA from accepted saved fits; never refits."""

from __future__ import annotations

import argparse
import html
import json
import resource
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from benchmarks.benchmark_baseline.baseline_shared import build_photutils_model_image
from benchmarks.exhausted.analysis.harmonic_demo import MODE_LABELS
from benchmarks.exhausted.analysis.publication_huang import PRIMARY, digest, load_profile, pixel_metrics, profile_dicts
from benchmarks.exhausted.analysis.reconstruction_diagnostic import raw_polar_carrier
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid, evaluation_aperture, zone_masks
from benchmarks.exhausted.plotting.individual_qa_demo import cross_arm, cross_tool
from isoster.model import build_isoster_model
from isoster.plotting import build_method_profile

ZONES = ("all", "inner", "mid", "outer")
METRICS = ("npix", "truth_relative_rms", "flux_bias", "flux_bias_abs", "data_sigma_rms")
CAPTION = """Metrics use the fixed 2-pixel elliptical inner cut, recorded outer radius and common finite support.
Zones: inner <0.5 R_ref, middle 0.5–2 R_ref, outer >=2 R_ref. CSV values are fractions; tables show percent.
Truth RMS = sqrt(sum((model-truth)^2)/sum(truth^2)); signed flux bias = sum(model-truth)/sum(truth).
Absolute flux bias is abs(signed flux bias). Noise RMS compares data-model divided by injected sigma.
Cross-tool maps show full finite Data-model coverage; cross-arm thumbnails show the common aperture.
Cross-tool scales are pooled over all available modes/arms inside that aperture; missing coverage is blank.
Ellipses mark evaluation limits, not bad pixels. Profiles and native uncertainties are unchanged by reconstruction.
Shared spline on uses calibrated raw polar n=3,4, with its own spline-off control; EA on is unsupported there.
Native Isoster uses basis-aware radial shifts; native Photutils uses additive n=3,4.
Native AutoProf is a verified saved zero-background ellipse model WITHOUT measured intensity harmonics.
Failures and unsupported coefficients/bases remain unavailable; no designated primary is replaced.
Fit times are inherited parallel-campaign measurements, not controlled timing; reconstruction times are separate.
Original-support baseline reproduction is separate from the matched-support metrics shown here.
"""


def harmonic_metadata(table, tool):
    """Audit stored positive-radius basis and orders without repairing coefficients."""
    positive = table[table["sma"] > 0]
    orders = sorted(int(k[1:]) for k in table.colnames if k.startswith("a") and k[1:].isdigit() and int(k[1:]) >= 3)
    if tool == "autoprof":
        bases = {str(v.decode() if isinstance(v, bytes) else v) for v in positive["harmonic_basis"]}
    else:
        bases = (
            {"ea" if bool(v) else "polar" for v in positive["use_eccentric_anomaly"]}
            if "use_eccentric_anomaly" in positive.colnames
            else {"polar"}
        )
    return ",".join(sorted(bases)), ",".join(map(str, orders))


def native_model(table, row, shape, track):
    """Require native provenance or finite coefficients; never use an AutoProf fallback."""
    records = profile_dicts(table)
    if row.tool == "autoprof":
        tag = f"{row.galaxy}__{row.scenario}"
        parent = Path(row.record_path).parent
        options = json.loads(track(parent / "tmp" / f"{tag}_options.json").read_text())
        if options.get("ap_set_background") != 0 or options.get("ap_isofit_fitcoefs") is not None:
            raise ValueError("native_options_not_zero_background_ellipse")
        model = fits.getdata(track(parent / "raw" / f"{tag}_genmodel.fits"), ext=1).astype(float)
        model[model == 0] = np.nan
        return model
    positive = table[table["sma"] > 0]
    for key in ("a3", "b3", "a4", "b4") + (("grad",) if row.tool == "photutils" else ()):
        if key not in positive.colnames or not np.isfinite(np.ma.filled(positive[key], np.nan)).all():
            raise ValueError("missing_native_coefficients:" + key)
    basis, _ = harmonic_metadata(table, row.tool)
    if basis not in ("ea", "polar"):
        raise ValueError("mixed_or_unknown_native_basis:" + basis)
    if row.tool == "isoster":
        return build_isoster_model(shape, records, fill=np.nan, use_harmonics=True, harmonic_orders=[3, 4])
    if basis != "polar":
        raise ValueError("unsupported_photutils_native_basis:" + basis)
    return build_photutils_model_image(shape, records, fill=np.nan, high_harmonics=True)


def process_input(task):
    """Write one complete input, with source and product checksums for safe restart."""
    measurements, output, galaxy, scenario = task
    started = perf_counter()
    destination = Path(output) / f"{galaxy}__{scenario}"
    done = destination / "complete.json"
    if done.exists():
        audit = json.loads(done.read_text())
        for path, value in audit["source_hashes"].items():
            if digest(Path(path)) != value:
                raise ValueError("Changed source: " + path)
        for name, value in audit["product_hashes"].items():
            if digest(destination / name) != value:
                raise ValueError("Changed product: " + name)
        return audit
    # An interrupted directory is preserved; a fresh attempt gets a distinct name.
    if destination.exists():
        attempt = 1
        while destination.with_name(destination.name + f".incomplete_{attempt}").exists():
            attempt += 1
        destination.rename(destination.with_name(destination.name + f".incomplete_{attempt}"))
    destination.mkdir(parents=True)
    hashes = {}
    historical = json.loads((Path(measurements) / "galaxies" / galaxy / "source_hashes.json").read_text())

    def track(path):
        path = Path(path)
        hashes[str(path)] = digest(path)
        if str(path) in historical and hashes[str(path)] != historical[str(path)]:
            raise ValueError("Source differs from accepted baseline: " + str(path))
        return path

    frame = pd.read_csv(track(Path(measurements) / "fit_metrics.csv"))
    rows = frame[frame.galaxy.eq(galaxy) & frame.scenario.eq(scenario)].copy()
    assert len(rows) == 17 and rows.inner_cut_pix.eq(2).all()
    manifest = json.loads(track(Path(rows.iloc[0].record_path).parents[3] / "MANIFEST.json").read_text())
    image = fits.getdata(track(manifest["extra"]["fits_path"])).astype(float)
    truth = fits.getdata(track(Path(measurements) / "galaxies" / galaxy / f"truth_z{scenario[-3:]}.fits"))
    geometry = manifest["initial_geometry"]
    radius = compute_elliptical_radius_grid(
        image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
    )
    eligible = evaluation_aperture(radius, geometry["maxsma"], 2)
    zones = dict(zip(("inner", "mid", "outer"), zone_masks(radius, rows.iloc[0].reference_pix)))
    zones["all"] = np.ones(image.shape, bool)
    models = {mode: {} for mode in MODE_LABELS}
    profiles, outcomes = {}, []
    for row in rows.itertuples():
        track(row.record_path)
        key = row.tool, row.arm
        table = None
        basis, orders = "unavailable", ""
        if row.status == "ok":
            table = load_profile(track(Path(row.record_path).parent / "profile.fits"))
            basis, orders = harmonic_metadata(table, row.tool)
            records = profile_dicts(table)
            profiles[key] = build_method_profile(records)
            if "cog" in table.colnames:
                profiles[key]["cog"] = np.asarray(table["cog"], float)
            if row.tool == "autoprof":
                profiles[key].pop("stop_codes", None)
        for mode in MODE_LABELS:
            render_start = perf_counter()
            status, reason = row.status, "" if row.status == "ok" else "fit_" + row.status
            if table is not None:
                try:
                    if mode == "shared_baseline":
                        model = build_isoster_model(image.shape, records, fill=np.nan, use_harmonics=False)
                    elif mode == "shared_spline_off":
                        model = build_photutils_model_image(image.shape, records, fill=np.nan, high_harmonics=False)
                    elif mode == "shared_spline_on":
                        model = build_photutils_model_image(
                            image.shape, raw_polar_carrier(table, row.tool), fill=np.nan, high_harmonics=True
                        )
                    else:
                        model = native_model(table, row, image.shape, track)
                    if model is None or model.shape != image.shape or not np.isfinite(model).any():
                        raise ValueError("invalid_or_empty_model")
                    models[mode][key] = model
                except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                    if mode == "shared_baseline":
                        raise
                    status, reason = "unavailable", f"{type(error).__name__}:{error}"
            outcomes.append(
                dict(
                    galaxy=galaxy,
                    scenario=scenario,
                    tool=row.tool,
                    arm=row.arm,
                    primary=row.primary,
                    fit_status=row.status,
                    status=status,
                    reason=reason,
                    mode=mode,
                    basis=basis,
                    available_orders=orders,
                    rendered_orders="3,4"
                    if status == "ok" and (mode == "shared_spline_on" or mode == "native" and row.tool != "autoprof")
                    else "",
                    renderer="isoster_linear"
                    if mode == "shared_baseline" or mode == "native" and row.tool == "isoster"
                    else (
                        "autoprof_saved_genmodel" if mode == "native" and row.tool == "autoprof" else "photutils_spline"
                    ),
                    rendered_basis=("polar_major_axis" if mode == "shared_spline_on" else basis)
                    if status == "ok" and (mode == "shared_spline_on" or mode == "native" and row.tool != "autoprof")
                    else "none",
                    reconstruction_wall_s=perf_counter() - render_start,
                )
            )
    original = eligible.copy()
    for model in models["shared_baseline"].values():
        original &= np.isfinite(model)
    checks = []
    for row in rows[rows.status.eq("ok")].itertuples():
        for zone in ZONES:
            values = pixel_metrics(
                models["shared_baseline"][row.tool, row.arm], truth, image, original & zones[zone], row.injected_sigma
            )
            for metric in METRICS:
                np.testing.assert_allclose(
                    values[metric], getattr(row, f"{metric}_{zone}"), rtol=1e-9, atol=1e-12, equal_nan=True
                )
            checks.append(dict(galaxy=galaxy, scenario=scenario, tool=row.tool, arm=row.arm, zone=zone, **values))
    common = original.copy()
    for group in models.values():
        for model in group.values():
            common &= np.isfinite(model)
    scores = []
    for outcome in outcomes:
        model = models[outcome["mode"]].get((outcome["tool"], outcome["arm"]))
        for zone in ZONES:
            support = common & zones[zone]
            values = pixel_metrics(
                model if model is not None else image,
                truth,
                image,
                support if model is not None else np.zeros(image.shape, bool),
                rows.iloc[0].injected_sigma,
            )
            reason = outcome["reason"] or ("empty_common_zone" if not values["npix"] else "")
            scores.append(
                dict(
                    **{**outcome, "reason": reason},
                    zone=zone,
                    inner_cut_pix=2,
                    original_npix=int((original & zones[zone]).sum()),
                    eligible_npix=int((eligible & zones[zone]).sum()),
                    support_fraction=float(support.sum() / max(1, (eligible & zones[zone]).sum())),
                    **values,
                )
            )
    metrics = pd.DataFrame(scores)
    metrics.to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(destination / "baseline_reproduction.csv", index=False)
    pd.DataFrame(outcomes).to_csv(destination / "outcomes.csv", index=False)
    rows.to_csv(destination / "accepted_rows.csv", index=False)
    fits.writeto(destination / "matched_support.fits", common.astype(np.uint8))
    residual_limit = (
        max(
            float(
                np.percentile(
                    np.concatenate(
                        [np.abs((image - model)[common]) for group in models.values() for model in group.values()]
                    ),
                    99,
                )
            ),
            1e-10,
        )
        if common.any()
        else None
    )
    pages = []
    for mode, group in models.items():
        display = rows.copy()
        for outcome in [v for v in outcomes if v["mode"] == mode]:
            chosen = display.tool.eq(outcome["tool"]) & display.arm.eq(outcome["arm"])
            display.loc[chosen, "status"] = outcome["status"]
            for zone in ZONES:
                value = metrics[
                    metrics["mode"].eq(mode)
                    & metrics.tool.eq(outcome["tool"])
                    & metrics.arm.eq(outcome["arm"])
                    & metrics.zone.eq(zone)
                ].iloc[0]
                for metric in METRICS:
                    display.loc[chosen, f"{metric}_{zone}"] = value[metric]
        display.to_csv(destination / f"{mode}_display.csv", index=False)
        available_profiles = profiles
        kinds = ["cross_tool"] + (list(PRIMARY) if mode in ("shared_baseline", "native") else [])
        for kind in kinds:
            name = f"{mode}__{kind}"
            selected = display[display.primary] if kind == "cross_tool" else display[display.tool.eq(kind)]
            if not common.any() or not any((r.tool, r.arm) in profiles for r in selected.itertuples()):
                fig, axis = plt.subplots(figsize=(14, 7))
                axis.set_axis_off()
                axis.text(
                    0.05,
                    0.9,
                    "\n".join(f"{r.tool}/{r.arm}: {r.status}" for r in selected.itertuples())
                    + "\nNo common metric support or available model.",
                    va="top",
                    transform=axis.transAxes,
                )
            elif kind == "cross_tool":
                fig = cross_tool(display, available_profiles, group, image, common, manifest)
                for axis in [a for a in fig.axes if a.images][1:]:
                    axis.images[0].set_clim(-residual_limit, residual_limit)
            else:
                fig = cross_arm(selected, available_profiles, manifest, kind, group, image, common)
            fig.suptitle(f"{galaxy}/{scenario} — {kind}\n{MODE_LABELS[mode]}", y=0.99, fontsize=14)
            for extension in ("png", "pdf"):
                fig.savefig(destination / f"{name}.{extension}", dpi=150, bbox_inches="tight")
            plt.close(fig)
            reasons = "\n".join(
                f"{v['tool']}/{v['arm']}: {v['status']} {v['reason']}"
                for v in outcomes
                if v["mode"] == mode and (kind == "cross_tool" and v["primary"] or kind == v["tool"])
            )
            (destination / f"{name}.caption.txt").write_text(
                MODE_LABELS[mode]
                + "\n"
                + CAPTION
                + f"Common pixels: {common.sum()}; original: {original.sum()}.\n"
                + reasons
                + "\n"
            )
            pages.append(name)
    (destination / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>'
        + galaxy
        + "/"
        + scenario
        + "</title>"
        + "\n".join(
            f'<p><a href="{name}.pdf">{name}</a> · <a href="{name}.caption.txt">caption</a></p><img width="100%" loading="lazy" src="{name}.png">'
            for name in pages
        )
    )
    for path, value in hashes.items():
        assert digest(Path(path)) == value, path
    products = {p.name: digest(p) for p in destination.iterdir() if p.is_file()}
    audit = dict(
        galaxy=galaxy,
        scenario=scenario,
        source_hashes=hashes,
        product_hashes=products,
        wall_s=perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        product_bytes=sum(p.stat().st_size for p in destination.iterdir() if p.is_file()),
        pages=len(pages),
        baseline_checks=len(checks),
        outcomes=len(outcomes),
        matched_pixels=int(common.sum()),
    )
    done.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="*", help="Explicit galaxy/scenario gate roster; omit for full population")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.measurements / "fit_metrics.csv")
    assert len(frame) == 14229 and frame.inner_cut_pix.eq(2).all()
    roster = list(frame[["galaxy", "scenario"]].drop_duplicates().itertuples(index=False, name=None))
    if args.cases:
        requested = [tuple(value.split("/")) for value in args.cases]
        assert len(set(requested)) == len(requested) and set(requested) <= set(roster)
        roster = requested
    code_paths = sorted(set(Path("benchmarks").rglob("*.py")) | set(Path("isoster").rglob("*.py")))
    policy = dict(
        measurements=str(args.measurements.resolve()),
        measurement_sha256=digest(args.measurements / "fit_metrics.csv"),
        roster=roster,
        modes=MODE_LABELS,
        pages_per_input=10,
        inner_cut_pix=2,
        support="original all-arm baseline; then intersection across every available arm/mode",
        model_storage="memory only; hashed source profiles reproduce models",
        code_hashes={str(p): digest(p) for p in code_paths},
    )
    manifest_path = args.output / "run_manifest.json"
    if args.resume:
        saved = json.loads(manifest_path.read_text())
        assert saved == json.loads(json.dumps(policy)), "Run policy or source code changed; use a fresh destination"
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(policy, indent=2) + "\n")
    free = shutil.disk_usage(args.output).free
    print(f"Free bytes {free}; workers {args.workers}; inputs {len(roster)}", flush=True)
    tasks = [(str(args.measurements), str(args.output), galaxy, scenario) for galaxy, scenario in roster]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, audit in enumerate(pool.map(process_input, tasks), 1):
            line = dict(
                completed=index,
                total=len(tasks),
                **{k: v for k, v in audit.items() if k not in ("source_hashes", "product_hashes")},
            )
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Huang2013 reconstruction QA</title><h1>Huang2013 reconstruction QA</h1><pre>'
        + html.escape(CAPTION)
        + "</pre>"
        + "\n".join(f'<p><a href="{g}__{s}/index.html">{g}/{s}</a></p>' for g, s in roster)
    )


if __name__ == "__main__":
    main()
