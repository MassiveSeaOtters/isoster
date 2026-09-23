"""Approved Huang2013 individual QA; reads science products and writes a NEW directory.

Usage: uv run --with pandas python -m benchmarks.exhausted.plotting.individual_qa_demo MEASUREMENTS OUTPUT
Historical module name retained for command compatibility. No fitting or campaign edits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator

from benchmarks.exhausted.analysis.publication_huang import (
    PRIMARY,
    digest,
    load_profile,
    pixel_metrics,
    profile_dicts,
)
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid, evaluation_aperture
from isoster.model import build_isoster_model
from isoster.plotting import (
    _scatter_by_stop_code_in_method_color,
    build_method_profile,
    configure_qa_plot_style,
    normalize_pa_degrees,
    plot_comparison_qa_figure,
    set_axis_limits_from_finite_values,
    transform_sb_profile,
)

COLORS = ["#332288", "#0077BB", "#008877", "#117733", "#887711", "#BB6600", "#CC6677", "#882255", "#AA4499", "#555555"]
CASES = [("IC1459", "wide_z050"), ("NGC4742", "wide_z050")]


def number(value, scale=1):
    """Missing measurements are not zero, including failed-fit durations."""
    return f"{value * scale:.3f}" if pd.notna(value) and np.isfinite(value) else "—"


def statistics_table(axis, rows, *, cross_tool=False, colors=None):
    """Display saved measurements, retaining unsuccessful outcomes."""
    cells = []
    for row in rows.itertuples():
        name = f"{row.tool} /\n{row.arm}" if cross_tool else row.arm
        status = "N/A" if row.status == "unavailable" else row.status.upper()
        if pd.notna(row.flags) and str(row.flags).strip():
            status += "*"
        cells.append(
            [
                name,
                status,
                number(row.wall_time_fit_s),
                " / ".join(
                    number(getattr(row, f"truth_relative_rms_{zone}"), 100)
                    for zone in (("all", "inner", "mid", "outer") if cross_tool else ("all",))
                )
                if row.status == "ok"
                else "—",
                " / ".join(
                    number(getattr(row, f"flux_bias_{zone}"), 100)
                    for zone in (("all", "inner", "mid", "outer") if cross_tool else ("all",))
                )
                if row.status == "ok"
                else "—",
                number(row.native_max_sma_pix) if row.status == "ok" else "—",
            ]
        )
    axis.set_axis_off()
    table = axis.table(
        cellText=cells,
        colLabels=[
            "Tool / arm" if cross_tool else "Arm",
            "Status",
            "Fit time [s]",
            "Truth RMS [%]\nALL / INNER / MIDDLE / OUTER" if cross_tool else "Truth RMS [%]",
            "Flux bias [%]\nALL / INNER / MIDDLE / OUTER" if cross_tool else "Flux bias [%]",
            "Reach [pix]",
        ],
        colWidths=[0.18, 0.06, 0.08, 0.30, 0.30, 0.08] if cross_tool else [0.36, 0.13, 0.12, 0.14, 0.14, 0.11],
        bbox=[0, 0, 1, 1],
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9 if cross_tool else 10)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if i == 0:
            cell.set_facecolor("#e8edf2")
            cell.get_text().set_weight("bold")
        elif j == 0 and colors:
            cell.get_text().set_color(colors[i - 1])
        if j == 0:
            cell.get_text().set_ha("left")
    return table


def differences(profile, reference, field):
    """Relative native-profile differences, never extrapolated beyond reference."""
    if reference is None or field not in profile or field not in reference:
        return np.full(len(profile["sma"]), np.nan)
    valid = np.isfinite(reference["sma"]) & np.isfinite(reference[field])
    x, indices = np.unique(reference["sma"][valid], return_index=True)
    if len(x) < 2:
        return np.full(len(profile["sma"]), np.nan)
    values = reference[field][valid][indices]
    target = np.interp(profile["sma"], x, values, left=np.nan, right=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(target != 0, 100 * (profile[field] - target) / target, np.nan)


def metric_matrix(axis, rows):
    """Independent physical scales per column; grey never means a numerical zero."""
    fields = ["wall_time_fit_s", "truth_relative_rms_all", "flux_bias_all", "native_max_sma_pix"]
    labels = ["Fit time\n[s]", "Truth RMS\n[%]", "Flux bias\n[%]", "Reach\n[pix]"]
    values = rows[fields].to_numpy(dtype=float).copy()
    values[:, 1:3] *= 100
    values[rows.status.ne("ok").to_numpy()] = np.nan
    for column in range(4):
        color_values = np.abs(values[:, column]) if column == 2 else values[:, column]
        finite = color_values[np.isfinite(color_values)]
        low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        if low == high:
            low, high = low - max(abs(low) * 0.05, 0.01), high + max(abs(high) * 0.05, 0.01)
        palette = ListedColormap(plt.get_cmap("RdBu_r" if column == 3 else "RdBu")(np.linspace(0.18, 0.82, 256)))
        palette.set_bad("#cccccc")
        artist = axis.imshow(
            color_values[:, None],
            origin="upper",
            aspect="auto",
            extent=[column - 0.5, column + 0.5, len(rows) - 0.5, -0.5],
            cmap=palette,
            norm=Normalize(low, high),
            interpolation="nearest",
        )
        color_axis = axis.inset_axes([column / 4 + 0.025, -0.035, 0.20, 0.02])
        colorbar = axis.figure.colorbar(artist, cax=color_axis, orientation="horizontal", ticks=[low, high])
        colorbar.ax.set_xticklabels([f"{low:.3g}", f"{high:.3g}"], fontsize=8)
        colorbar.ax.get_xticklabels()[0].set_ha("left")
        colorbar.ax.get_xticklabels()[-1].set_ha("right")
        for index, value in enumerate(values[:, column]):
            axis.text(column, index, number(value), ha="center", va="center", color="black", fontsize=11)
    axis.set(xlim=(-0.5, 3.5), ylim=(len(rows) - 0.5, -0.5))
    axis.set_xticks(range(4), labels, fontsize=11)
    axis.xaxis.tick_top()
    axis.tick_params(axis="both", length=0, pad=8)
    axis.minorticks_off()
    axis.grid(False)
    names = rows.arm.tolist()
    axis.set_yticks(range(len(rows)), names, fontsize=10)
    for label, color in zip(axis.get_yticklabels(), COLORS):
        label.set_color(color)
    for boundary in np.arange(-0.5, len(rows), 1):
        axis.axhline(boundary, color="white", lw=1.5)
    for boundary in np.arange(-0.5, 4, 1):
        axis.axvline(boundary, color="white", lw=1.5)
    return values


def residual_thumbnails(parent, rows, models, image, support):
    """Same field of view and absolute residual scale for every arm and tool."""
    residuals = [np.abs((image - model)[support]) for model in models.values()]
    limit = max(float(np.percentile(np.concatenate(residuals), 99)), 1e-10) if residuals and support.any() else 1e-10
    parent.text(0.905, 1.012, "Residual", ha="center", fontsize=12)
    for index, row in enumerate(rows.itertuples()):
        axis = parent.inset_axes([0.82, (len(rows) - 1 - index) / len(rows), 0.17, 1 / len(rows) * 0.98])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_box_aspect(image.shape[0] / image.shape[1])
        axis.set_anchor("N")
        model = models.get((row.tool, row.arm))
        if row.status == "ok" and model is not None:
            residual = np.where(support, image - model, np.nan)
            axis.imshow(residual, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            assert np.array_equal(np.isfinite(residual), support)
        else:
            axis.set_facecolor("#cccccc")
            if row.status != "skipped":
                axis.text(0.5, 0.5, row.status.upper(), transform=axis.transAxes, ha="center", va="center", fontsize=8)
        for spine in axis.spines.values():
            spine.set_color(COLORS[index])


def cross_arm(rows, profiles, manifest, tool, models, image, support):
    """Six-panel native profile comparison with a fixed default reference."""
    configure_qa_plot_style()
    plt.rcParams["text.usetex"] = False
    rows = rows.assign(default=rows.arm.eq(PRIMARY[tool])).sort_values(["default", "arm"], ascending=[False, True])
    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(
        6,
        2,
        width_ratios=[1.12, 1],
        height_ratios=[2.5, 1, 1, 1, 1, 1],
        left=0.09,
        right=0.98,
        bottom=0.10,
        top=0.90,
        wspace=0.22,
        hspace=0,
    )
    left = fig.add_subplot(grid[:, 0])
    left.set_axis_off()
    matrix_axis = left.inset_axes([0.12, 0, 0.69, 1])
    metric_matrix(matrix_axis, rows)
    residual_thumbnails(left, rows, models, image, support)
    axes = []
    for index in range(6):
        axes.append(fig.add_subplot(grid[index, 1], sharex=axes[0] if axes else None))
    reference = profiles.get((tool, PRIMARY[tool]))
    geometry = manifest["initial_geometry"]
    labels = ["", "Center drift [pix]", "Axis ratio b/a", "PA [deg]", "ΔI / I ref [%]", "ΔCoG / CoG ref [%]"]
    collected = [[] for _ in axes]
    for index, row in enumerate(rows.itertuples()):
        profile = profiles.get((tool, row.arm))
        if profile is None:
            continue
        keep = profile["sma"] > 1.5
        x = profile["sma"][keep] ** 0.25
        sb, error, labels[0], _, _ = transform_sb_profile(
            profile["intens"],
            profile.get("intens_err"),
            sb_zeropoint=manifest["sb_zeropoint"],
            pixel_scale_arcsec=manifest["pixel_scale_arcsec"],
            sb_profile_scale="asinh",
            sb_asinh_softening=max(manifest["image_sigma"]["image_sigma_adu"], 1e-10),
        )
        # Keep the historical within-profile median-center drift definition.
        drift = np.hypot(profile["x0"] - np.nanmedian(profile["x0"]), profile["y0"] - np.nanmedian(profile["y0"]))
        values = [
            sb,
            drift,
            1 - profile["eps"],
            normalize_pa_degrees(np.degrees(profile["pa"]), anchor=np.degrees(geometry["pa"])),
            differences(profile, reference, "intens"),
            differences(profile, reference, "cog"),
        ]
        for panel, (axis, value) in enumerate(zip(axes, values)):
            errors = error[keep] if panel == 0 and error is not None else None
            if panel == 2 and "eps_err" in profile:
                errors = profile["eps_err"][keep]
            elif panel == 3 and "pa_err" in profile:
                errors = np.degrees(profile["pa_err"][keep])
            if "stop_codes" in profile:
                _scatter_by_stop_code_in_method_color(
                    axis, x, value[keep], profile["stop_codes"][keep], COLORS[index], label=row.arm, y_errors=errors
                )
            else:
                axis.errorbar(
                    x, value[keep], yerr=errors, fmt="o", ms=3, color=COLORS[index], elinewidth=0.5, alpha=0.8
                )
            collected[panel].extend(value[keep].tolist())
    for index, (axis, label, values) in enumerate(zip(axes, labels, collected)):
        axis.set_ylabel(label, fontsize=10)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4 if index else 6, prune="both"))
        axis.yaxis.set_label_coords(-0.12, 0.5)
        axis.tick_params(labelsize=10, labelbottom=index == 5)
        axis.grid(alpha=0.2)
        set_axis_limits_from_finite_values(
            axis, np.asarray(values), margin_fraction=0.08, min_margin=0.05, invert=index == 0
        )
    for panel, axis in enumerate(axes[4:], start=4):
        if reference is None:
            axis.text(
                0.5, 0.65, f"Reference {PRIMARY[tool]} unavailable", transform=axis.transAxes, ha="center", fontsize=11
            )
        elif not np.isfinite(collected[panel]).any():
            axis.text(0.5, 0.65, "Native CoG unavailable", transform=axis.transAxes, ha="center", fontsize=11)
        else:
            axis.axhline(0, color=".4", ls="--", lw=0.6)
    axes[-1].set_xlabel(r"SMA$^{0.25}$ [pixel$^{0.25}$]", fontsize=12)
    axes[0].set_title("Native radial profiles", fontsize=15, pad=12)
    axes[0].legend(
        handles=[
            Line2D([], [], color=COLORS[index], marker="o", linestyle="none", markersize=4, label=row.arm)
            for index, row in enumerate(rows.itertuples())
            if (tool, row.arm) in profiles
        ],
        loc="upper right",
        fontsize=8,
        ncol=2,
        framealpha=0.85,
    )
    assert np.isclose(matrix_axis.get_position().y1, axes[0].get_position().y1)
    assert np.isclose(matrix_axis.get_position().y0, axes[-1].get_position().y0)
    assert all(np.isclose(upper.get_position().y0, lower.get_position().y1) for upper, lower in zip(axes, axes[1:]))
    fig.suptitle(f"{manifest['galaxy_id']} — {tool}: all arms", fontsize=20, y=0.97)
    return fig


def cross_tool(rows, profiles, models, image, support, manifest):
    """Show full residuals; the common evaluation aperture only defines metrics."""
    chosen = rows[rows.primary].set_index("tool").loc[list(PRIMARY)].reset_index()
    selected = {tool: profiles[tool, arm] for tool, arm in PRIMARY.items() if (tool, arm) in profiles}
    shown_models = {tool: models[tool, PRIMARY[tool]] for tool in selected if (tool, PRIMARY[tool]) in models}
    fig = plot_comparison_qa_figure(
        image,
        selected,
        models=shown_models,
        mask=None,
        sb_zeropoint=manifest["sb_zeropoint"],
        pixel_scale_arcsec=manifest["pixel_scale_arcsec"],
        sb_profile_scale="asinh",
        sb_asinh_softening=max(manifest["image_sigma"]["image_sigma_adu"], 1e-10),
        return_figure=True,
    )
    # Reserve a title/table band without reimplementing the shared plotting code.
    for axis in fig.axes:
        position = axis.get_position()
        axis.set_position([position.x0, 0.10 + (position.y0 - 0.11) * 0.87, position.width, position.height * 0.87])
    fig.suptitle(f"{manifest['galaxy_id']} — primary cross-tool comparison", y=0.985, fontsize=17)
    fig.canvas.draw()
    data_axis = next(axis for axis in fig.axes if axis.images)
    panel_left = data_axis.get_position().x0
    panel_right = max(axis.get_position().x1 for axis in fig.axes)
    table_bottom = max(axis.get_position().y1 for axis in fig.axes) + 0.012
    table_axis = fig.add_axes([panel_left, table_bottom, panel_right - panel_left, 0.94 - table_bottom])
    statistics_table(table_axis, chosen, cross_tool=True)
    assert np.allclose([table_axis.get_position().x0, table_axis.get_position().x1], [panel_left, panel_right])
    data_axis = next(axis for axis in fig.axes if axis.images)
    geometry = manifest["initial_geometry"]
    for radius, color, style, label in [
        (rows.iloc[0].inner_cut_pix, "white", "-", "Inner"),
        (geometry["maxsma"], "#ffcf40", "--", "Outer"),
    ]:
        data_axis.add_patch(
            Ellipse(
                (geometry["x0"], geometry["y0"]),
                2 * radius,
                2 * radius * (1 - geometry["eps"]),
                angle=np.degrees(geometry["pa"]),
                fill=False,
                color=color,
                ls=style,
                lw=1.2,
                label=label,
            )
        )
    data_axis.legend(
        loc="upper right",
        fontsize=6,
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.2,
        facecolor="#333333",
        labelcolor="white",
        framealpha=0.85,
    )
    residual_values = [np.abs((image - model)[support]) for model in shown_models.values()]
    limit = (
        max(float(np.percentile(np.concatenate(residual_values), 99)), 1e-10)
        if residual_values and support.any()
        else 1e-10
    )
    residual_axes = [axis for axis in fig.axes if axis.images and axis is not data_axis]
    for axis in residual_axes:
        axis.images[0].set_clim(-limit, limit)
        if axis.images[0].colorbar:
            axis.images[0].colorbar.set_label("Data − model [ADU]", fontsize=9)
    assert len(data_axis.images) == 1, "Measurement aperture must not become a data mask"
    assert len(data_axis.patches) == 2
    assert all(axis.images[0].get_clim() == (-limit, limit) for axis in residual_axes)
    failures = [
        f"{r.tool}/{r.arm}: {r.status.upper()} — "
        + (
            "profile retained; model unavailable."
            if (r.tool, r.arm) in profiles
            else "no profile or residual; no replacement."
        )
        for r in chosen.itertuples()
        if r.status != "ok"
    ]
    if failures:
        fig.text(panel_left, 0.025, "\n".join(failures), fontsize=10, color="#9a3412")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.measurements / "fit_metrics.csv")
    if "inner_cut_pix" not in frame or not np.isfinite(frame.inner_cut_pix).all():
        raise ValueError("Measurements must record inner_cut_pix; rerun historical analysis before plotting")
    hashes = {}
    checks = []

    def track(path):
        hashes[str(path)] = digest(path)
        return path

    track(args.measurements / "fit_metrics.csv")
    for galaxy, scenario in CASES:
        rows = frame[frame.galaxy.eq(galaxy) & frame.scenario.eq(scenario)]
        assert len(rows) == 17
        source = Path(rows.iloc[0].record_path).parents[3]
        manifest = json.loads(track(source / "MANIFEST.json").read_text())
        image = fits.getdata(track(Path(manifest["extra"]["fits_path"]))).astype(float)
        truth = fits.getdata(track(args.measurements / "galaxies" / galaxy / f"truth_z{scenario[-3:]}.fits"))
        geometry = manifest["initial_geometry"]
        radius = compute_elliptical_radius_grid(
            image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
        )
        assert rows.inner_cut_pix.nunique() == 1
        support = evaluation_aperture(radius, geometry["maxsma"], rows.iloc[0].inner_cut_pix)
        profiles, models = {}, {}
        for row in rows.itertuples():
            track(Path(row.record_path))
            if row.status != "ok":
                continue
            table = load_profile(track(Path(row.record_path).parent / "profile.fits"))
            key = row.tool, row.arm
            profiles[key] = build_method_profile(profile_dicts(table))
            if "cog" in table.colnames:
                profiles[key]["cog"] = np.asarray(table["cog"], float)
            if row.tool == "autoprof":
                profiles[key].pop("stop_codes", None)
            models[key] = build_isoster_model(image.shape, profile_dicts(table), fill=np.nan, use_harmonics=False)
            support &= np.isfinite(models[key])
        for row in rows[rows.status.eq("ok")].itertuples():
            values = pixel_metrics(models[row.tool, row.arm], truth, image, support, row.injected_sigma)
            assert values["npix"] == row.npix_all
            for metric in ("truth_relative_rms", "flux_bias"):
                assert np.isclose(values[metric], getattr(row, metric + "_all"), rtol=1e-9, atol=1e-12)
            checks.append(dict(galaxy=galaxy, scenario=scenario, tool=row.tool, arm=row.arm, **values))
        rows.to_csv(args.output / f"{galaxy}__{scenario}__statistics.csv", index=False)
        for kind in ("cross_tool", *PRIMARY):
            fig = (
                cross_tool(rows, profiles, models, image, support, manifest)
                if kind == "cross_tool"
                else cross_arm(rows[rows.tool.eq(kind)], profiles, manifest, kind, models, image, support)
            )
            for extension in ("png", "pdf"):
                fig.savefig(args.output / f"{galaxy}__{scenario}__{kind}.{extension}", dpi=180, bbox_inches="tight")
            plt.close(fig)
            caption_rows = rows[rows.primary] if kind == "cross_tool" else rows[rows.tool.eq(kind)]
            caption = (
                f"{galaxy}/{scenario} — {kind}. Approved individual QA layout.\n\n"
                "Residuals are data minus the common no-harmonic model in ADU. Cross-tool maps show all finite "
                "model coverage without evaluation-cut masking; cross-arm thumbnails retain the shared science aperture. "
                "All cross-arm thumbnails use the same field of view and "
                "symmetric color scale across all tools for this input (pooled 99th percentile of absolute residuals). "
                "Cross-tool residual scales are shared across the selected primaries, calibrated on the common aperture. "
                "Cross-tool tables report ALL / INNER / MIDDLE / OUTER metrics on that aperture. Ellipses indicate inner "
                "and outer radial limits, not bad pixels; finite model coverage can reduce the aperture.\n\n"
                "Truth RMS = 100 × RMS(model − truth) / RMS(truth); flux bias = 100 × Σ(model − truth) / Σtruth. "
                "Reach is the maximum fitted semi-major axis in pixels. Runtime is parallel-campaign fit time, "
                "not controlled timing. Heatmap columns have independent linear scales: red indicates lower "
                "runtime, lower RMS, smaller absolute flux bias, or larger reach; blue indicates the opposite. "
                "Reach measures coverage, not necessarily fit quality. Flux-bias colors use absolute values, "
                "but black numbers retain the sign. Grey cells indicate failed, skipped or missing measurements, "
                "not zero. Constant columns use a symmetric padded range and therefore a neutral color.\n\n"
                "Cross-arm differences use the designated default without fallback; center drift is relative "
                "to each profile’s median. Cross-tool differences and centroid offsets use the Isoster primary. "
                "Filled profile points indicate stop=0; other stop codes use open markers. AutoProf has no "
                "equivalent stop codes. Error bars are native uncertainties. SB limits follow finite profile values; "
                "no I=0 reference is shown.\n\n"
            )
            caption += f"Common aperture: {support.sum()} pixels; inner radius {rows.iloc[0].inner_cut_pix:.3f} pix; outer radius {geometry['maxsma']:.3f} pix.\n\n"
            caption += "\n".join(
                f"{r.tool}/{r.arm}: {r.status}; flags: {r.flags if pd.notna(r.flags) else 'none'}"
                for r in caption_rows.itertuples()
            )
            (args.output / f"{galaxy}__{scenario}__{kind}.caption.txt").write_text(caption + "\n")
        print(f"[qa] {galaxy}/{scenario}: four figure pairs, {len(models)} metric checks", flush=True)
    assert all(digest(Path(path)) == value for path, value in hashes.items())
    (args.output / "audit.json").write_text(
        json.dumps(dict(source_hashes=hashes, metric_checks=checks, script_sha256=digest(Path(__file__))), indent=2)
        + "\n"
    )
    print(f"[qa] {len(hashes)} source hashes unchanged; {args.output}")


if __name__ == "__main__":
    main()
