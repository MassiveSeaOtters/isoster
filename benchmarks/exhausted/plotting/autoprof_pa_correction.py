"""Before/after diagnostics in the existing QA style, without hidden PA outliers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Ellipse

from benchmarks.exhausted.analysis.publication_huang import load_profile, profile_dicts
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid
from benchmarks.exhausted.campaigns.run_autoprof_pa_correction import archived_sources
from isoster.model import build_isoster_model
from isoster.plotting import METHOD_STYLES, configure_qa_plot_style, normalize_pa_degrees, transform_sb_profile


def plot_case(galaxy_dir: Path, old_record: Path, output: Path):
    """Use shared-domain no-harmonic models; this is not the final science score."""
    manifest = json.loads((galaxy_dir / "MANIFEST.json").read_text())
    image = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
    paths = {
        "Isoster reference": galaxy_dir / "isoster/arms/ref_default/profile.fits",
        "AutoProf old PA": old_record.parent / "profile.fits",
        "AutoProf corrected PA": galaxy_dir / "autoprof/arms/baseline/profile.fits",
    }
    profiles = {name: load_profile(path) for name, path in paths.items() if path.is_file()}
    models = {
        name: build_isoster_model(image.shape, profile_dicts(profile), fill=np.nan, use_harmonics=False)
        for name, profile in profiles.items()
    }
    geometry = manifest["initial_geometry"]
    radius = compute_elliptical_radius_grid(
        image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
    )
    floor = manifest["extra"]["psf_fwhm_arcsec"] / manifest["pixel_scale_arcsec"]
    support = (radius >= floor) & (radius <= geometry["maxsma"])
    for model in models.values():
        support &= np.isfinite(model)
    residuals = {name: np.where(support, image - model, np.nan) for name, model in models.items()}
    finite = np.concatenate([abs(value[np.isfinite(value)]) for value in residuals.values()])
    limit = max(float(np.percentile(finite, 99)), 1e-10) if finite.size else 1
    styles = [
        (METHOD_STYLES["isoster"]["color"], "o", "-"),
        ("0.4", "s", "--"),
        (METHOD_STYLES["autoprof"]["color"], "^", "-."),
    ]
    configure_qa_plot_style()
    plt.rcParams["text.usetex"] = False
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(4, 2, width_ratios=[1, 1.7], height_ratios=[2, 1, 1, 1])
    axes = [fig.add_subplot(grid[i, 1]) for i in range(4)]
    panels = [fig.add_subplot(grid[i, 0]) for i in range(4)]
    sigma = manifest["image_sigma"]["image_sigma_adu"]
    panels[0].imshow(np.arcsinh(image / max(sigma, 1e-10)), origin="lower", cmap="viridis")
    panels[0].set_title("Data and fitted ellipses", fontsize=11)
    values_for_limits = [[] for _ in axes]
    rms = {}
    for index, (name, profile) in enumerate(profiles.items()):
        color, marker, linestyle = styles[index]
        data = profile[np.asarray(profile["sma"]) > 0]
        sma = np.asarray(data["sma"], float)
        sb, error, label, invert, _ = transform_sb_profile(
            np.asarray(data["intens"]),
            np.asarray(data["intens_err"]),
            sb_zeropoint=manifest["sb_zeropoint"],
            pixel_scale_arcsec=manifest["pixel_scale_arcsec"],
            sb_profile_scale="asinh",
            sb_asinh_softening=max(sigma, 1e-10),
        )
        pa = normalize_pa_degrees(np.degrees(data["pa"]))
        # Align the wrapped branch to the intended initial orientation.
        center_pa = np.degrees(geometry["pa"])
        pa = (pa - center_pa + 90) % 180 - 90 + center_pa
        values = [sb, np.asarray(data["eps"]), pa, np.hypot(data["x0"] - geometry["x0"], data["y0"] - geometry["y0"])]
        for axis, y, collected in zip(axes, values, values_for_limits):
            axis.scatter(sma**0.25, y, color=color, marker=marker, s=12, alpha=0.8, label=name)
            collected.extend(np.asarray(y)[np.isfinite(y)].tolist())
            axis.axvline(floor**0.25, color="0.6", ls=":", lw=0.6)
        axes[0].errorbar(sma**0.25, sb, yerr=error, color=color, fmt="none", lw=0.4, alpha=0.35)
        for a in [0.5 * manifest["effective_Re_pix"], 2 * manifest["effective_Re_pix"]]:
            row = data[np.argmin(abs(sma - a))]
            panels[0].add_patch(
                Ellipse(
                    (row["x0"], row["y0"]),
                    2 * row["sma"],
                    2 * row["sma"] * (1 - row["eps"]),
                    angle=np.degrees(row["pa"]),
                    fill=False,
                    color=color,
                    ls=linestyle,
                    lw=0.8,
                )
            )
        shown = panels[index + 1].imshow(residuals[name], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        rms[name] = float(np.sqrt(np.mean(residuals[name][support] ** 2))) if support.any() else None
        panels[index + 1].set_title(name + ": data minus common renderer", fontsize=10)
        fig.colorbar(shown, ax=panels[index + 1], label="Intensity / pixel", fraction=0.035)
    for axis, vals in zip(axes, values_for_limits):
        if vals:
            low, high = min(vals), max(vals)
            margin = max(0.05 * (high - low), 0.01)
            axis.set_ylim(low - margin, high + margin)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(label)
    if invert:
        axes[0].invert_yaxis()
    axes[0].legend(fontsize=10)
    for axis, label in zip(axes[1:], ["Ellipticity", "PA [deg]", "Center offset [pixel]"]):
        axis.set_ylabel(label)
    axes[-1].set_xlabel(r"SMA$^{0.25}$ [pixel$^{0.25}$]")
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    for axis in axes:
        axis.set_xlim(0, max(float(np.max(p["sma"])) ** 0.25 for p in profiles.values()) * 1.03)
    for panel in panels:
        panel.set_xticks([])
        panel.set_yticks([])
    fig.suptitle(manifest["galaxy_id"] + " — PA correction diagnostic (no-harmonic renderer)", fontsize=15)
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + extension), dpi=180)
    plt.close(fig)
    return dict(galaxy=manifest["galaxy_id"], common_pixels=int(support.sum()), data_residual_rms=rms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    galaxies = sorted(args.campaign.glob("*/*/MANIFEST.json"))
    sources = archived_sources(args.campaign.parent.parent, {p.parents[1].name for p in galaxies})
    results = []
    for manifest in galaxies:
        galaxy = json.loads(manifest.read_text())["galaxy_id"]
        dataset = manifest.parents[1].name
        old = sources[dataset, galaxy, "autoprof", "baseline"]
        results.append(plot_case(manifest.parent, old, args.output / (dataset + "__" + manifest.parent.name)))
    (args.output / "diagnostic_metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
