"""Figures and selected-case QA from the frozen Huang2013 science measurements.

Usage: python -m benchmarks.exhausted.plotting.publication_huang MEASUREMENTS OUTPUT
OUTPUT must be new. No fitting or source-campaign writes are performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from benchmarks.exhausted.analysis.publication_huang import (
    PRIMARY,
    SCENARIOS,
    digest,
    load_profile,
    pixel_metrics,
    profile_dicts,
)
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid
from isoster.model import build_isoster_model
from isoster.plotting import METHOD_STYLES, build_method_profile, configure_qa_plot_style, plot_comparison_qa_figure

TOOLS = list(PRIMARY)
MARKERS = dict(zip(TOOLS, ["o", "s", "^"]))
LABELS = [f"{s.split('_z')[0]}\nz={int(s[-3:]) / 100:.2f}" for s in SCENARIOS]


def finite_summary(values):
    """Report descriptive population spread, never uncertainty of the mean."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return (len(values), *np.percentile(values, [16, 50, 84])) if len(values) else (0, np.nan, np.nan, np.nan)


def matched_primary(frame, metric):
    """Use the same successful, finite galaxy/scenario set for all three tools."""
    primary = frame[frame.primary & frame.status.eq("ok")]
    wide = primary.pivot(index=["galaxy", "scenario"], columns="tool", values=metric).reindex(columns=TOOLS)
    return wide.replace([np.inf, -np.inf], np.nan).dropna()


def save_figure(fig, output, name, caption):
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.955))
    fig.text(0.01, 0.005, caption, fontsize=8, va="bottom")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def scenario_panels(frame, metrics, labels, output, name):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), layout="constrained")
    exported = []
    for ax, metric, label in zip(np.atleast_1d(axes), metrics, labels):
        paired = matched_primary(frame, metric)
        for offset, tool in enumerate(TOOLS):
            statistics = []
            for scenario in SCENARIOS:
                values = paired.loc[paired.index.get_level_values("scenario") == scenario, tool]
                count, low, med, high = finite_summary(values)
                statistics.append((count, low, med, high))
                exported.append(
                    dict(metric=metric, scenario=scenario, tool=tool, n=count, p16=low, median=med, p84=high)
                )
            values = np.array(statistics)
            ax.errorbar(
                np.arange(9) + (offset - 1) * 0.16,
                values[:, 2],
                yerr=[values[:, 2] - values[:, 1], values[:, 3] - values[:, 2]],
                fmt=MARKERS[tool],
                color=METHOD_STYLES[tool]["color"],
                capsize=3,
                label=tool,
            )
        ax.set_xticks(range(9), LABELS)
        ax.set_ylabel(label)
        limits = [r for r in exported if r["metric"] == metric and r["n"]]
        low, high = min(r["p16"] for r in limits), max(r["p84"] for r in limits)
        if metric == "reach_ref":
            ax.set_ylim(low * 0.9, high * 1.1)
        elif low > 0:
            ax.set_yscale("log")
            ax.set_ylim(low / 1.5, high * 1.5)
        else:
            ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_ylim(-1e-5, max(high * 1.5, 1e-4))
        counts = [int(np.sum(paired.index.get_level_values("scenario") == s)) for s in SCENARIOS]
        ax.set_title("Matched galaxies per scenario: " + ", ".join(map(str, counts)), fontsize=10)
        ax.grid(alpha=0.2)
    np.atleast_1d(axes)[0].legend(ncol=3)
    save_figure(
        fig,
        output,
        name,
        "Median and p16–p84 population spread; matched primary samples. Log error axes (symmetric log if zero); radial reach is linear.",
    )
    pd.DataFrame(exported).to_csv(output / f"{name}.csv", index=False)


def coverage_figure(frame, output):
    roster = frame[["tool", "arm"]].drop_duplicates().sort_values(["tool", "arm"])
    matrix, labels = [], []
    for row in roster.itertuples():
        group = frame[frame.tool.eq(row.tool) & frame.arm.eq(row.arm)]
        matrix.append([int((group.scenario.eq(s) & group.status.eq("ok")).sum()) for s in SCENARIOS])
        labels.append(f"{row.tool}: {row.arm}")
    total = frame.galaxy.nunique()
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    shown = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=total, aspect="auto")
    for y, values in enumerate(matrix):
        for x, count in enumerate(values):
            ax.text(
                x, y, str(count), ha="center", va="center", color="black" if count > total / 2 else "white", fontsize=9
            )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(9), LABELS)
    ax.set_title(f"Execution coverage: successful fits / {total} galaxies per cell")
    fig.colorbar(shown, ax=ax, label="Successful fits")
    save_figure(
        fig,
        output,
        "01_coverage",
        "Execution success is not an accuracy verdict. OLS no-weight is an intentional no-op; retained failures are not replaced.",
    )


def paired_figure(summary, output):
    metrics = ["truth_relative_rms_all", "flux_bias_abs_all", "center_median_pix"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), layout="constrained")
    for ax, metric in zip(axes, metrics):
        subset = summary[summary.metric.eq(metric)].sort_values(["reference", "arm"])
        labels = [f"{r.arm} − {r.reference}\n(n={r.n}, G={r.galaxies})" for r in subset.itertuples()]
        for i, row in enumerate(subset.itertuples()):
            ax.plot([row.median_ci_low, row.median_ci_high], [i, i], color=METHOD_STYLES["isoster"]["color"])
            ax.scatter(row.median, i, color=METHOD_STYLES["isoster"]["color"], s=18)
        ax.axvline(0, ls="--", color="0.5")
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [])
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.invert_yaxis()
    save_figure(
        fig,
        output,
        "04_isoster_contrasts",
        "Paired arm minus reference: negative means smaller error. Bars: 95% galaxy-block bootstrap CI of pooled median; n=pairs, G=galaxies.",
    )


def structural_figure(frame, output):
    descriptors = ["reference_psf", "initial_eps", "component_pa_span", "n_components"]
    labels = ["Reference radius / PSF FWHM", "Initial ellipticity", "Component PA span (deg)", "Number of components"]
    subset = frame[frame.primary & frame.status.eq("ok") & frame.scenario.eq("wide_z005")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    records = []
    for ax, descriptor, label in zip(axes.flat, descriptors, labels):
        for tool in TOOLS:
            group = subset[subset.tool.eq(tool)]
            ax.scatter(
                group[descriptor],
                group.truth_relative_rms_all,
                s=18,
                alpha=0.65,
                color=METHOD_STYLES[tool]["color"],
                marker=MARKERS[tool],
                label=f"{tool} (n={len(group)})",
            )
            bins = (
                group[descriptor]
                if group[descriptor].nunique() <= 4
                else pd.qcut(group[descriptor], 4, duplicates="drop")
            )
            for category, part in group.groupby(bins, observed=True):
                n, lo, med, hi = finite_summary(part.truth_relative_rms_all)
                records.append(
                    dict(tool=tool, descriptor=descriptor, interval=str(category), n=n, p16=lo, median=med, p84=hi)
                )
        ax.set_xlabel(label)
        ax.set_ylabel("Truth-relative RMS")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    save_figure(
        fig,
        output,
        "05_structural_trends",
        "One wide z=0.05 realization per galaxy; successful primary fits shown individually. Descriptive associations, not controlled causal effects.",
    )
    pd.DataFrame(records).to_csv(output / "structural_trends.csv", index=False)


def harmonic_figure(frame, output):
    metrics = [
        "abs_a3n_median_outer",
        "abs_a4n_median_outer",
        "max_local_resid_eps_outer",
        "max_local_resid_pa_outer_deg",
    ]
    labels = [
        "Outer median |a3| (Bender)",
        "Outer median |a4| (Bender)",
        "Outer ellipticity roughness",
        "Outer PA roughness (deg)",
    ]
    subset = frame[frame.tool.eq("isoster") & frame.status.eq("ok")]
    arms = sorted(subset.arm.unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    records = []
    for ax, metric, label in zip(axes.flat, metrics, labels):
        for i, arm in enumerate(arms):
            values = subset[subset.arm.eq(arm)].groupby("galaxy")[metric].median()
            n, low, med, high = finite_summary(values)
            ax.errorbar(
                i, med, yerr=[[med - low], [high - med]], fmt="o", capsize=3, color=METHOD_STYLES["isoster"]["color"]
            )
            records.append(dict(arm=arm, metric=metric, galaxies=n, p16=low, median=med, p84=high))
        ax.set_xticks(range(len(arms)), arms, rotation=65, ha="right")
        ax.set_ylabel(label)
        rows = [r for r in records if r["metric"] == metric]
        low, high = min(r["p16"] for r in rows), max(r["p84"] for r in rows)
        if low > 0:
            ax.set_yscale("log")
            ax.set_ylim(low / 1.5, high * 1.5)
        else:
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.set_ylim(-1e-6, high * 1.5)
        ax.set_title("Galaxies per arm: " + ", ".join(str(r["galaxies"]) for r in rows), fontsize=10)
        ax.grid(alpha=0.2)
    save_figure(
        fig,
        output,
        "06_harmonics_smoothness",
        "Median over scenarios within each galaxy, then median and p16–p84 across galaxies. Smaller harmonics/smoother curves need not be more accurate.",
    )
    pd.DataFrame(records).to_csv(output / "harmonic_smoothness.csv", index=False)


def select_cases(frame, pairs):
    """Fixed descriptive rules; retain both EA benefit and degradation cases."""
    selected = []
    reference = frame[frame.tool.eq("isoster") & frame.arm.eq("ref_default") & frame.status.eq("ok")]

    def add(group, column, largest, reason, contrast=""):
        group = group[np.isfinite(group[column])].sort_values(["galaxy", "scenario"])
        if len(group):
            row = group.loc[group[column].idxmax() if largest else group[column].idxmin()]
            selected.append(dict(galaxy=row.galaxy, scenario=row.scenario, reason=reason, contrast=contrast))

    typical = reference[reference.scenario.eq("wide_z005")].copy()
    typical["distance"] = abs(typical.truth_relative_rms_all - typical.truth_relative_rms_all.median())
    add(typical, "distance", False, "closest to median Isoster RMS in wide_z005")
    add(reference[reference.scenario.eq("noiseless_z005")], "initial_eps", True, "largest initial ellipticity")
    add(
        reference[reference.scenario.eq("deep_z050")],
        "reference_psf",
        False,
        "smallest resolved reference radius at deep_z050",
    )
    add(reference, "common_support_fraction", False, "smallest common finite support")
    failed = frame[frame.primary & frame.status.eq("failed")].sort_values(["galaxy", "scenario", "tool"])
    if len(failed):
        row = failed.iloc[0]
        selected.append(
            dict(galaxy=row.galaxy, scenario=row.scenario, reason="first retained primary failure", contrast="")
        )
    primary = matched_primary(frame, "truth_relative_rms_all").reset_index()
    primary["difference"] = abs(primary.autoprof - primary.isoster)
    add(primary, "difference", True, "largest absolute primary AutoProf–Isoster RMS difference")
    ea = pairs[pairs.arm.eq("geom_ea") & pairs.reference.eq("ref_default") & pairs.metric.eq("truth_relative_rms_all")]
    add(ea, "delta", False, "largest EA RMS benefit", "geom_ea")
    add(ea, "delta", True, "largest EA RMS degradation", "geom_ea")
    harm = pairs[
        pairs.arm.eq("harm_simul_ea") & pairs.reference.eq("geom_simul_ea") & pairs.metric.eq("truth_relative_rms_all")
    ].copy()
    harm["absolute_delta"] = abs(harm.delta)
    add(harm, "absolute_delta", True, "largest simultaneous-harmonic RMS change", "harm_simul_ea")
    return pd.DataFrame(selected)


def render_case(frame, measurements, case, output):
    """Reuse comparison QA; verify the rebuilt common aperture against the table."""
    subset = frame[frame.galaxy.eq(case.galaxy) & frame.scenario.eq(case.scenario)]
    source = Path(subset.iloc[0].record_path).parents[3]
    manifest = json.loads((source / "MANIFEST.json").read_text())
    geometry = manifest["initial_geometry"]
    image = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
    truth = fits.getdata(measurements / "galaxies" / case.galaxy / f"truth_z{case.scenario[-3:]}.fits")
    radius = compute_elliptical_radius_grid(
        image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
    )
    support = (radius >= subset.iloc[0].psf_pix) & (radius <= geometry["maxsma"])
    models, profiles = {}, {}
    for row in subset[subset.status.eq("ok")].itertuples():
        table = load_profile(Path(row.record_path).parent / "profile.fits")
        key = row.tool, row.arm
        profiles[key] = build_method_profile(profile_dicts(table))
        if row.tool == "autoprof":
            profiles[key].pop("stop_codes", None)
        models[key] = build_isoster_model(image.shape, profile_dicts(table), fill=np.nan, use_harmonics=False)
        support &= np.isfinite(models[key])
    if case.contrast == "geom_ea":
        chosen = [("isoster", "ref_default"), ("isoster", "geom_ea")]
    elif case.contrast == "harm_simul_ea":
        chosen = [("isoster", "geom_simul_ea"), ("isoster", "harm_simul_ea")]
    else:
        chosen = list(PRIMARY.items())
    use_profiles, use_models, styles, checks = {}, {}, {}, []
    for index, key in enumerate(chosen):
        if key not in models:
            continue
        row = subset[subset.tool.eq(key[0]) & subset.arm.eq(key[1])].iloc[0]
        values = pixel_metrics(models[key], truth, image, support, row.injected_sigma)
        assert values["npix"] == row.npix_all
        assert np.isclose(values["truth_relative_rms"], row.truth_relative_rms_all, rtol=1e-9, atol=1e-12)
        name = key[0] if not case.contrast else key[1]
        use_profiles[name], use_models[name] = profiles[key], models[key]
        styles[name] = dict(METHOD_STYLES[TOOLS[index]], label=key[0] if not case.contrast else key[1])
        checks.append(dict(tool=key[0], arm=key[1], **values))
    stem = f"{case.galaxy}__{case.scenario}__{case.contrast or 'primary'}"
    failures = ", ".join(f"{r.tool}/{r.arm}" for r in subset[subset.primary & subset.status.ne("ok")].itertuples())
    title = f"{case.galaxy} / {case.scenario} — common no-harmonic renderer"
    if failures:
        title = f"{case.galaxy} / {case.scenario} — unavailable: {failures}"
    for extension in ("png", "pdf"):
        plot_comparison_qa_figure(
            image,
            use_profiles,
            title=title,
            output_path=output / f"{stem}.{extension}",
            models=use_models,
            mask=~support,
            method_styles=styles,
            sb_zeropoint=manifest["sb_zeropoint"],
            pixel_scale_arcsec=manifest["pixel_scale_arcsec"],
            sb_profile_scale="asinh",
            sb_asinh_softening=max(float(manifest["image_sigma"]["image_sigma_adu"]), 1e-10),
            dpi=300,
        )
    return dict(stem=stem, reason=case.reason, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--atlas-limit", type=int, help="Small export gate only")
    args = parser.parse_args()
    frame = pd.read_csv(args.measurements / "fit_metrics.csv")
    pairs = pd.read_csv(args.measurements / "paired_deltas.csv")
    summary = pd.read_csv(args.measurements / "paired_summary.csv")
    args.output.mkdir(parents=True, exist_ok=False)
    configure_qa_plot_style()
    plt.rcParams.update(
        {"text.usetex": False, "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9}
    )
    coverage_figure(frame, args.output)
    scenario_panels(
        frame,
        ["truth_relative_rms_inner", "truth_relative_rms_mid", "truth_relative_rms_outer"],
        ["Inner truth-relative RMS", "Middle truth-relative RMS", "Outer truth-relative RMS"],
        args.output,
        "02_primary_truth_zones",
    )
    scenario_panels(
        frame,
        ["flux_bias_abs_all", "center_median_pix", "reach_ref"],
        ["Absolute aperture flux bias", "Median center error (pixel)", "Maximum finite SMA / reference radius"],
        args.output,
        "03_primary_fidelity",
    )
    paired_figure(summary, args.output)
    structural_figure(frame, args.output)
    harmonic_figure(frame, args.output)
    primary_rows = []
    for metric in ("truth_relative_rms_all", "flux_bias_all", "data_sigma_rms_all", "ring_relative_rms"):
        paired = matched_primary(frame, metric)
        for scenario, group in paired.groupby(level="scenario"):
            for tool in TOOLS:
                n, low, med, high = finite_summary(group[tool])
                primary_rows.append(
                    dict(metric=metric, scenario=scenario, tool=tool, n=n, p16=low, median=med, p84=high)
                )
    pd.DataFrame(primary_rows).to_csv(args.output / "primary_additional_metrics.csv", index=False)
    background_rows = []
    for row in frame[frame.tool.eq("autoprof") & frame.status.eq("ok")].itertuples():
        aux = json.loads(Path(row.record_path).read_text())["autoprof_aux"]
        background_rows.append(
            dict(
                galaxy=row.galaxy,
                scenario=row.scenario,
                arm=row.arm,
                background=aux["background"],
                background_noise=aux["background_noise"],
                center_x=aux["center_x"],
                center_y=aux["center_y"],
            )
        )
    pd.DataFrame(background_rows).to_csv(args.output / "autoprof_background_centers.csv", index=False)
    cases = select_cases(frame, pairs)
    cases.to_csv(args.output / "selected_cases.csv", index=False)
    atlas = args.output / "atlas"
    atlas.mkdir()
    checks = []
    unique_cases = cases.drop_duplicates(["galaxy", "scenario", "contrast"])
    if args.atlas_limit is not None:
        unique_cases = unique_cases.head(args.atlas_limit)
    for case in unique_cases.itertuples():
        checks.append(render_case(frame, args.measurements, case, atlas))
        print("[figures]", checks[-1]["stem"], flush=True)
    (args.output / "atlas_metric_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    input_hashes = {
        name: digest(args.measurements / name)
        for name in ("fit_metrics.csv", "paired_deltas.csv", "paired_summary.csv")
    }
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(
                measurements=str(args.measurements),
                script_sha256=digest(Path(__file__)),
                input_sha256=input_hashes,
                atlas_limit=args.atlas_limit,
            ),
            indent=2,
        )
        + "\n"
    )
    print("[figures] complete", args.output, flush=True)


if __name__ == "__main__":
    main()
