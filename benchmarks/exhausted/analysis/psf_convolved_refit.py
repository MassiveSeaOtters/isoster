"""Refit Huang2013 noiseless mocks after applying the PSF the campaign intended.

The publication mocks were rendered by profit-cli with ``-P psf.fits`` but without
``convolve=1`` in the profile spec, so the stated 0.7 arcsec Gaussian PSF was never
applied and every core is a bare Sersic cusp. This script convolves the noiseless
image with that Gaussian (FWHM 0.7"/0.168" = 4.17 px), optionally adds the
campaign's wide-depth noise, refits with the saved Isoster configuration under each
sampling variant, and scores the harmonic-off reconstructions against the convolved
noiseless image on the accepted common support with the 2-pixel cut. No AutoProf
fits are repeated here; the question is only how Isoster's variants behave once the
cores are PSF-limited.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from benchmarks.exhausted.analysis.publication_huang import pixel_metrics
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid, zone_masks
from benchmarks.exhausted.analysis.sampling_refit_experiment import BASE, FINAL, VARIANTS, ZONES, render
from isoster import IsosterConfig, fit_image

PSF_FWHM_PIX = 0.7 / 0.168
SEED = 20261009


def run_case(galaxy, frame, output, variants, noise):
    rows = frame[frame.galaxy.eq(galaxy) & frame.primary & frame.tool.eq("isoster")]
    source = rows[rows.scenario.eq("noiseless_z005")].iloc[0]
    record = Path(source.record_path).parent
    manifest = json.loads((record.parents[2] / "MANIFEST.json").read_text())
    raw = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
    truth = gaussian_filter(raw, PSF_FWHM_PIX / 2.3548, mode="constant")
    support = fits.getdata(FINAL / f"{galaxy}__noiseless_z005" / "matched_support.fits").astype(bool)
    config = yaml.safe_load((record / "config.yaml").read_text())
    geometry = manifest["initial_geometry"]
    radius = compute_elliptical_radius_grid(raw.shape, **{k: geometry[k] for k in ("x0", "y0", "eps", "pa")})
    masks = dict(zip(ZONES[1:], zone_masks(radius, source.reference_pix)))
    masks["all"] = np.ones(raw.shape, bool)
    images = {"noiseless": (truth, np.nan)}
    if noise:
        wide = rows[rows.scenario.eq("wide_z005")]
        if len(wide):
            sigma = float(wide.iloc[0].injected_sigma)
            noisy = truth + np.random.default_rng(SEED).normal(0.0, sigma, truth.shape)
            images["wide_noise"] = (noisy, sigma)
    metrics, core = (
        [],
        float(truth[raw.shape[0] // 2, raw.shape[1] // 2 + 1] / truth[raw.shape[0] // 2, raw.shape[1] // 2]),
    )
    for condition, (image, sigma) in images.items():
        models = {}
        for name in variants:
            result = fit_image(image, None, IsosterConfig(**{**config, **VARIANTS[name]}))
            for renderer in ("linear", "cubic"):
                models[(name, renderer)] = render(image.shape, result["isophotes"], renderer, False)
        common = support.copy()
        for model in models.values():
            if model is not None:
                common &= np.isfinite(model)
        for (name, renderer), model in models.items():
            for zone in ZONES:
                mask = common & masks[zone] if model is not None else np.zeros(raw.shape, bool)
                metrics.append(
                    dict(
                        galaxy=galaxy,
                        condition=condition,
                        variant=name,
                        renderer=renderer,
                        zone=zone,
                        core_ratio=core,
                        **pixel_metrics(model if model is not None else image, truth, image, mask, sigma),
                    )
                )
    frame = pd.DataFrame(metrics)
    frame.to_csv(output / f"{galaxy}.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--galaxy", action="append", required=True)
    parser.add_argument("--variants", default="default,cubic_ea,cubic3_ea")
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(BASE / "fit_metrics.csv")
    variants = args.variants.split(",")
    results = [run_case(g, frame, args.output, variants, not args.no_noise) for g in args.galaxy]
    table = pd.concat(results)
    table.to_csv(args.output / "metrics_all.csv", index=False)
    table["rms"] = 100 * table.truth_relative_rms
    table["bias"] = 100 * table.flux_bias
    lines = []
    for (condition, renderer, zone), group in table.groupby(["condition", "renderer", "zone"]):
        rms = group.pivot_table(index="galaxy", columns="variant", values="rms")
        bias = group.pivot_table(index="galaxy", columns="variant", values="bias")
        lines.append(
            f"{condition:10s} {renderer:7s} {zone:6s} medRMS "
            + " ".join(f"{v}={rms[v].median():.3f}" for v in variants)
            + " | win vs default "
            + " ".join(f"{v}={np.mean(rms[v] < rms[variants[0]]):.2f}" for v in variants[1:])
            + " | med|bias| "
            + " ".join(f"{v}={bias[v].abs().median():.3f}" for v in variants)
        )
    text = "\n".join(lines)
    (args.output / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
