"""Bounded reconstruction sensitivity check from saved profiles; never refits.

Compare both tools with each shared renderer, on one common finite aperture.
The harmonic diagnostic uses raw polar intensity amplitudes (orders 3, 4),
not an invented AutoProf gradient or a conversion between polar and EA bases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from benchmarks.benchmark_baseline.baseline_shared import build_photutils_model_image
from benchmarks.exhausted.analysis.publication_huang import (
    digest,
    load_profile,
    pixel_metrics,
    profile_dicts,
    ring_truth,
)
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid, evaluation_aperture, zone_masks
from benchmarks.harmonic_scale.conventions import raw_from_autoprof, raw_from_bender, rotate_raw_to_major_axis
from isoster.model import build_isoster_model


def raw_polar_carrier(table, tool):
    """Feed exact raw amplitudes to Photutils' additive harmonic renderer.

    Its renderer multiplies stored harmonics by -gradient*sma. A synthetic
    gradient of -1 and coefficients raw/sma cancel exactly. This is solely
    an interface carrier, NOT an estimate of the galaxy's physical gradient.
    Reject EA/unknown bases rather than silently rotating incompatible modes.
    """
    rows = profile_dicts(table)
    result = []
    for row in rows:
        if row["sma"] <= 0:
            continue
        if not all(np.isfinite(row[key]) for key in ("sma", "intens", "eps", "pa", "x0", "y0")):
            raise ValueError("Nonfinite geometry/intensity in harmonic input")
        carrier = dict(row, grad=-1.0)
        if tool != "autoprof" and row.get("use_eccentric_anomaly", False):
            raise ValueError("EA harmonics require a renderer with an explicit EA basis")
        for order in (3, 4):
            if tool == "autoprof":
                basis = row["harmonic_basis"]
                if isinstance(basis, bytes):
                    basis = basis.decode()
                if basis != "polar_from_image_x_axis":
                    raise ValueError(f"Unsupported AutoProf basis: {basis}")
                sine, cosine = raw_from_autoprof(
                    row[f"autoprof_a{order}_native"], row[f"autoprof_b{order}_native"], row["autoprof_b0"]
                )
                sine, cosine = rotate_raw_to_major_axis(sine, cosine, order, row["pa"])
            else:
                sine, cosine = raw_from_bender(row[f"a{order}"], row[f"b{order}"], row["sma"], row["grad"])
            if not np.isfinite([sine, cosine]).all():
                raise ValueError("Missing raw harmonic amplitude; cannot treat it as zero")
            carrier[f"a{order}"], carrier[f"b{order}"] = sine / row["sma"], cosine / row["sma"]
        result.append(carrier)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--galaxies", nargs="+", default=["IC1459", "NGC4697"])
    parser.add_argument("--scenarios", nargs="+", default=["wide_z005", "wide_z050"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def track(path):
        path = Path(path)
        hashes[str(path)] = digest(path)
        return path

    frame = pd.read_csv(track(args.measurements / "fit_metrics.csv"))
    selected = frame[frame.primary & frame.galaxy.isin(args.galaxies) & frame.scenario.isin(args.scenarios)]
    expected = {
        (galaxy, scenario, tool)
        for galaxy in args.galaxies
        for scenario in args.scenarios
        for tool in ("isoster", "photutils", "autoprof")
    }
    actual = set(selected[["galaxy", "scenario", "tool"]].itertuples(index=False, name=None))
    if actual != expected or len(selected) != len(expected):
        raise ValueError(f"Requested primary roster is missing or duplicated: {expected - actual}")
    scores, failures, rings, coverage = [], [], [], []
    for (galaxy, scenario), rows in selected.groupby(["galaxy", "scenario"]):
        manifest = json.loads(track(Path(rows.iloc[0].record_path).parents[3] / "MANIFEST.json").read_text())
        geometry = manifest["initial_geometry"]
        image = fits.getdata(track(manifest["extra"]["fits_path"])).astype(float)
        truth = fits.getdata(track(args.measurements / "galaxies" / galaxy / f"truth_z{scenario[-3:]}.fits"))
        radius = compute_elliptical_radius_grid(
            image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
        )
        eligible = evaluation_aperture(radius, geometry["maxsma"])
        models = {}
        for row in rows.itertuples():
            track(row.record_path)
            if row.status != "ok":
                failures.append(dict(galaxy=galaxy, scenario=scenario, tool=row.tool, mode="fit", reason=row.status))
                continue
            table = load_profile(track(Path(row.record_path).parent / "profile.fits"))
            records = profile_dicts(table)
            coverage.append(
                dict(
                    galaxy=galaxy,
                    scenario=scenario,
                    tool=row.tool,
                    minimum_positive_sma=float(np.min(table["sma"][table["sma"] > 0])),
                    n_positive=int(np.sum(table["sma"] > 0)),
                )
            )
            # Compare a shared polar mean diagnostic as well as the native-statistic proxy.
            for statistic, ea, median in [
                ("shared_polar_mean", False, False),
                ("native_proxy", row.tool == "autoprof", row.tool != "isoster"),
            ]:
                rings.append(
                    dict(
                        galaxy=galaxy,
                        scenario=scenario,
                        tool=row.tool,
                        statistic=statistic,
                        **ring_truth(table, truth, ea, median, 2.0, samples=2048),
                    )
                )
            for mode in ("isoster_linear", "isoster_cubic", "photutils_spline", "photutils_raw_harmonics"):
                try:
                    if mode.startswith("isoster"):
                        model = build_isoster_model(
                            image.shape, records, fill=np.nan, use_harmonics=False, interp_kind=mode.split("_")[1]
                        )
                    else:
                        harmonic = mode == "photutils_raw_harmonics"
                        model = build_photutils_model_image(
                            image.shape,
                            raw_polar_carrier(table, row.tool) if harmonic else records,
                            high_harmonics=harmonic,
                            fill=np.nan,
                        )
                    if model is None:
                        raise ValueError("Renderer did not produce a model")
                    models[row.tool, mode] = model
                except (ValueError, TypeError, KeyError) as error:
                    failures.append(dict(galaxy=galaxy, scenario=scenario, tool=row.tool, mode=mode, reason=str(error)))
        common = eligible.copy()
        for model in models.values():
            common &= np.isfinite(model)
        masks = dict(zip(("inner", "mid", "outer"), zone_masks(radius, rows.iloc[0].reference_pix)))
        masks["all"] = np.ones(image.shape, bool)
        for (tool, mode), model in models.items():
            for zone, mask in masks.items():
                scores.append(
                    dict(
                        galaxy=galaxy,
                        scenario=scenario,
                        tool=tool,
                        mode=mode,
                        zone=zone,
                        inner_cut_pix=2.0,
                        common_support_fraction=float(common.sum() / eligible.sum()),
                        **pixel_metrics(model, truth, image, common & mask, rows.iloc[0].injected_sigma),
                    )
                )
        print(f"[reconstruction] {galaxy}/{scenario}: {len(models)} models, {common.sum()} common pixels", flush=True)
    pd.DataFrame(scores).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(rings).to_csv(args.output / "ring_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output / "profile_coverage.csv", index=False)
    (args.output / "unavailable.json").write_text(json.dumps(failures, indent=2) + "\n")
    assert all(digest(Path(path)) == value for path, value in hashes.items())
    (args.output / "audit.json").write_text(
        json.dumps(
            dict(
                source_hashes=hashes,
                script_sha256=digest(Path(__file__)),
                renderer_sha256=digest(Path("isoster/model.py")),
                photutils_adapter_sha256=digest(Path("benchmarks/benchmark_baseline/baseline_shared.py")),
                scope="Bounded sensitivity diagnostic, not a population ranking; common support across all available tools/modes",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
