"""Truth-based measurement of a round-2 (PSF-convolved) Huang2013 campaign.

Every arm of every tool is rendered with several harmonic-off radial interpolators
(Isoster linear, cubic and log-linear; the Photutils spline painter) and with each
tool's native model, then scored against the regenerated PSF-convolved truth on one
matched pixel support with the fixed 2-pixel inner cut. The truth comes from the
frozen round-2 generator and must reproduce the campaign image exactly (noise seed
included), which ties the measurement to the data provenance.

Run with ``python -m benchmarks.exhausted.analysis.round2_small_sample --help``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits

from benchmarks.benchmark_baseline.baseline_shared import build_photutils_model_image
from benchmarks.exhausted.analysis.publication_huang import digest, load_profile, pixel_metrics, profile_dicts
from benchmarks.exhausted.analysis.residual_zones import (
    DEFAULT_INNER_CUT_PIX,
    compute_elliptical_radius_grid,
    evaluation_aperture,
    zone_masks,
)
from isoster.model import build_isoster_model

FROZEN_MOCK_COMMIT = "8632e1fd1807d43c384d3a4a5355a39b95b49e26"
MANIFEST_NAME = "inputs/huang2013/runs/huang2013_publication_single_band_psf.yaml"
SCENARIOS = ["noiseless_z005"] + [f"{depth}_z{z}" for z in ("005", "020", "035", "050") for depth in ("wide", "deep")]
PRIMARY = {"isoster": "ref_default", "photutils": "baseline_median", "autoprof": "baseline"}
RENDERERS = ("linear_off", "cubic_off", "log_off", "spline_off", "native")
ZONES = ("all", "inner", "mid", "outer")

mock = mock_manifest = mock_galaxies = profit_executable = None


def initialize_mock(source: str, executable: str) -> None:
    """Load the frozen round-2 generator in each worker without editing that repository."""
    global mock, mock_manifest, mock_galaxies, profit_executable
    source = Path(source)
    os.environ["DYLD_LIBRARY_PATH"] = str(Path(executable).parent)
    spec = importlib.util.spec_from_file_location("round2_frozen_generator", source / "scripts/generate_mocks.py")
    mock = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mock
    spec.loader.exec_module(mock)
    mock_manifest = mock.resolve_run_manifest(source / MANIFEST_NAME)
    mock_galaxies = {
        mock.sanitize_filename(g.name): g for g in mock.load_galaxies(source / mock_manifest.model_file, None, False)
    }
    profit_executable = executable


def render(tool, table, record, arm_dir, shape, renderer):
    """One harmonic-off or native model per (arm, renderer); None when unsupported."""
    rows = profile_dicts(table)
    if renderer == "linear_off":
        return build_isoster_model(shape, rows, fill=np.nan, use_harmonics=False, interp_kind="linear")
    if renderer == "cubic_off":
        return build_isoster_model(shape, rows, fill=np.nan, use_harmonics=False, interp_kind="cubic")
    if renderer == "log_off":
        return build_isoster_model(shape, rows, fill=np.nan, use_harmonics=False, interp_kind="log_linear")
    if renderer == "spline_off":
        return build_photutils_model_image(shape, rows, fill=np.nan, high_harmonics=False)
    if tool == "isoster":
        return build_isoster_model(shape, rows, fill=np.nan, use_harmonics=True, interp_kind="linear")
    if tool == "photutils":
        return build_photutils_model_image(shape, rows, fill=np.nan, high_harmonics=True)
    candidates = sorted((arm_dir / "raw").glob("*_genmodel.fits"))
    if not candidates:
        return None
    model = fits.getdata(candidates[0]).astype(float)
    if model.shape != shape:
        return None
    return np.where(model == 0, np.nan, model)  # AutoProf's zero exterior is absent support, not a model value


def analyze_galaxy(name: str, records: list[dict], output: str, inner_cut_pix: float) -> dict:
    import pandas as pd

    started = time.monotonic()
    target = Path(output) / "galaxies" / name
    target.mkdir(parents=True, exist_ok=False)
    galaxy = mock_galaxies[name]
    truth_cache, verification, measured = {}, [], []
    for scenario in SCENARIOS:
        subset = [r for r in records if r["scenario"] == scenario]
        if not subset:
            continue
        manifest_dir = Path(subset[0]["record_path"]).parents[3]
        manifest = json.loads((manifest_dir / "MANIFEST.json").read_text())
        geometry = manifest["initial_geometry"]
        image = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
        recipe = next(r for r in mock_manifest.rows if r.name == scenario)
        z = recipe.redshift
        values = dict(recipe.config_values, profit_cli_path=profit_executable)
        if z not in truth_cache:
            generator = mock.MockImageGenerator(
                mock.build_image_config(f"truth_z{z}", dict(values, noise_enabled=False))
            )
            at_z = mock.MockGalaxy(
                name=galaxy.name, redshift=z, components=galaxy.components, re_overall=galaxy.re_overall
            )
            truth, metadata = generator.generate(at_z)
            if metadata["engine"] != "libprofit" or not metadata.get("psf_convolved"):
                raise RuntimeError("Truth renderer fell back or did not convolve; rejecting truth")
            fits.writeto(target / f"truth_z{round(z * 100):03d}.fits", truth, overwrite=False)
            truth_cache[z] = truth
        truth = truth_cache[z]
        values["noise_seed"] = mock.resolve_noise_seed(galaxy.name, recipe)
        generator = mock.MockImageGenerator(mock.build_image_config(scenario, values))
        predicted = generator._add_noise(truth, {}) if values["noise_enabled"] else truth
        equal = np.array_equal(predicted.astype(np.float32), image.astype(np.float32))
        verification.append(
            dict(galaxy=name, scenario=scenario, exact_float32_match=equal, seed=str(values["noise_seed"]))
        )
        if not equal:
            raise RuntimeError(f"Truth/noise reproduction failed: {name}/{scenario}")
        sigma = (
            10 ** (-0.4 * (values["sky_sb_limit"] - values["zeropoint"])) * values["pixel_scale"] ** 2 / 5
            if values["noise_enabled"]
            else np.nan
        )
        radius = compute_elliptical_radius_grid(
            image.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
        )
        reference = float(manifest["effective_Re_pix"])
        eligible = evaluation_aperture(radius, geometry["maxsma"], inner_cut_pix)
        masks = dict(zip(("inner", "mid", "outer"), zone_masks(radius, reference)))
        masks["all"] = np.ones(image.shape, dtype=bool)
        models, profiles = {}, {}
        for row in subset:
            if row["status"] != "ok":
                continue
            arm_dir = Path(row["record_path"]).parent
            table = load_profile(arm_dir / "profile.fits")
            profiles[row["tool"], row["arm"]] = table
            for renderer in RENDERERS:
                try:
                    model = render(row["tool"], table, row, arm_dir, image.shape, renderer)
                except (ValueError, TypeError, KeyError):
                    model = None
                if model is not None and not np.isfinite(model).any():
                    model = None
                models[row["tool"], row["arm"], renderer] = model
        common = eligible.copy()
        for model in models.values():
            if model is not None:
                common &= np.isfinite(model)
        for row in subset:
            record = json.loads(Path(row["record_path"]).read_text())
            base = dict(
                row,
                primary=row["arm"] == PRIMARY[row["tool"]],
                redshift=z,
                depth=scenario.split("_")[0],
                reference_pix=reference,
                psf_pix=float(manifest["extra"]["psf_fwhm_arcsec"]) / float(manifest["pixel_scale_arcsec"]),
                inner_cut_pix=inner_cut_pix,
                injected_sigma=sigma,
                image_pixels=image.size,
                common_support_fraction=float(common.sum() / eligible.sum()),
                wall_time_fit_s=record.get("wall_time_fit_s", np.nan),
                error=str(record.get("error_msg") or record.get("error") or "")[:200],
            )
            if row["status"] == "ok":
                table = profiles[row["tool"], row["arm"]]
                valid = np.asarray(table["sma"]) > 0
                for column in ("sma", "x0", "y0", "eps", "pa", "intens"):
                    valid &= np.isfinite(table[column])
                finite = table[valid]
                base["n_finite"] = len(finite)
                base["n_stop0"] = int(np.sum(finite["stop_code"] == 0))
                outside = finite[finite["sma"] >= inner_cut_pix]
                drift = np.hypot(outside["x0"] - geometry["x0"], outside["y0"] - geometry["y0"])
                base["center_median_pix"] = float(np.median(drift)) if len(drift) else np.nan
                base["center_max_pix"] = float(np.max(drift)) if len(drift) else np.nan
            for renderer in RENDERERS:
                model = models.get((row["tool"], row["arm"], renderer))
                for zone in ZONES:
                    mask = common & masks[zone] if model is not None else np.zeros(image.shape, dtype=bool)
                    measured.append(
                        dict(
                            base,
                            renderer=renderer,
                            model_available=model is not None,
                            zone=zone,
                            **pixel_metrics(model if model is not None else image, truth, image, mask, sigma),
                        )
                    )
    pd.DataFrame(measured).to_csv(target / "fit_metrics.csv", index=False)
    pd.DataFrame(verification).to_csv(target / "truth_verification.csv", index=False)
    return dict(galaxy=name, records=len(measured), elapsed_seconds=round(time.monotonic() - started, 1))


def collect_records(campaign: Path) -> list[dict]:
    rows = []
    for path in sorted((campaign / "huang2013").glob("*/*/arms/*/run_record.json")):
        galaxy, scenario = path.parents[3].name.split("__")
        record = json.loads(path.read_text())
        rows.append(
            dict(
                galaxy=galaxy,
                scenario=scenario,
                tool=path.parents[2].name,
                arm=path.parent.name,
                status=record["status"],
                record_path=str(path),
            )
        )
    return rows


def summarize(frame, output: Path) -> str:
    """Medians, paired win fractions against both references, failures and regressions."""
    import pandas as pd

    frame = frame.copy()
    frame["rms"] = 100 * frame.truth_relative_rms
    frame["bias"] = 100 * frame.flux_bias
    lines = ["# Round-2 small-sample summary", ""]
    ok = frame[frame.status.eq("ok") & frame.model_available]
    status = (
        frame[frame.zone.eq("all") & frame.renderer.eq("linear_off")]
        .groupby(["tool", "arm"])
        .status.value_counts()
        .unstack(fill_value=0)
    )
    lines += ["## Fit outcomes per arm", "", status.to_markdown(), ""]
    times = (
        frame[frame.zone.eq("all") & frame.renderer.eq("linear_off") & frame.status.eq("ok")]
        .groupby(["tool", "arm"])
        .wall_time_fit_s.quantile([0.5, 0.9])
        .unstack()
    )
    times.columns = ["median_s", "p90_s"]
    lines += ["## Fit wall time [s]", "", times.round(3).to_markdown(), ""]
    for renderer in RENDERERS:
        for zone in ZONES:
            z = ok[ok.renderer.eq(renderer) & ok.zone.eq(zone)]
            if z.empty:
                continue
            rms = z.pivot_table(index=["galaxy", "scenario"], columns=["tool", "arm"], values="rms")
            bias = z.pivot_table(index=["galaxy", "scenario"], columns=["tool", "arm"], values="bias")

            def wins(reference):
                if reference not in rms:
                    return pd.Series(np.nan, index=rms.columns)
                paired = rms.notna() & rms[reference].notna().to_numpy()[:, None]
                return (rms.lt(rms[reference], axis=0) & paired).sum() / paired.sum()

            table = pd.DataFrame(
                {
                    "n": rms.count(),
                    "median_rms": rms.median(),
                    "median_abs_bias": bias.abs().median(),
                    "median_signed_bias": bias.median(),
                    "beats_isoster_default": wins(("isoster", "ref_default")),
                    "beats_autoprof_baseline": wins(("autoprof", "baseline")),
                }
            )
            lines += [
                f"## Renderer `{renderer}`, zone `{zone}` (truth RMS and flux bias in percent)",
                "",
                table.round(3).to_markdown(),
                "",
            ]
    z = ok[ok.renderer.eq("linear_off") & ok.zone.eq("all")]
    rms = z.pivot_table(index=["galaxy", "scenario"], columns=["tool", "arm"], values="rms")
    key = [
        c
        for c in rms.columns
        if c[1] in ("ref_default", "samp_cubic_ea", "samp_cubic_ea_min4", "baseline", "baseline_median")
    ]
    by_scenario = rms[key].reset_index().groupby("scenario").median(numeric_only=True)
    by_scenario.columns = [f"{t}:{a}" for t, a in by_scenario.columns]
    lines += [
        "## Median full-aperture truth RMS [%] by scenario (linear renderer)",
        "",
        by_scenario.round(3).to_markdown(),
        "",
    ]
    for arm in ("samp_cubic_ea", "samp_cubic_ea_min4"):
        if ("isoster", arm) in rms:
            ratio = (rms["isoster", arm] / rms["isoster", "ref_default"]).dropna().sort_values(ascending=False)
            lines += [
                f"## Regressions: `{arm}` / `ref_default` full RMS ratio (linear renderer)",
                "",
                f"quantiles 10/50/90: {np.quantile(ratio, [0.1, 0.5, 0.9]).round(3).tolist()}; worse by >20 % in {int((ratio > 1.2).sum())} of {len(ratio)} inputs",
                "",
                ratio.head(8).round(3).to_frame("ratio").to_markdown(),
                "",
            ]
    text = "\n".join(lines)
    (output / "summary.md").write_text(text + "\n")
    return text


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="<output_root>/<campaign_name>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mock-source", type=Path, required=True)
    parser.add_argument("--profit-cli", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inner-cut-pix", type=float, default=DEFAULT_INNER_CUT_PIX)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.mock_source), "rev-parse", "HEAD"], text=True).strip()
    if head != FROZEN_MOCK_COMMIT:
        raise ValueError(f"Wrong mock generator revision: {head}")
    rows = collect_records(args.campaign)
    if not rows:
        raise ValueError("No run records found")
    names = sorted({r["galaxy"] for r in rows})
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output / "manifest.csv", index=False)
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(
                arguments={k: str(v) for k, v in vars(args).items()},
                mock_commit=FROZEN_MOCK_COMMIT,
                profit_sha256=digest(args.profit_cli),
                analysis_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                script_sha256=digest(Path(__file__)),
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            ),
            indent=2,
        )
        + "\n"
    )
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_mock, initargs=(str(args.mock_source), str(args.profit_cli))
    ) as executor:
        futures = [
            executor.submit(
                analyze_galaxy, name, [r for r in rows if r["galaxy"] == name], str(args.output), args.inner_cut_pix
            )
            for name in names
        ]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[analysis] {index}/{len(names)} {future.result()}", flush=True)
    frame = pd.concat(
        [pd.read_csv(args.output / "galaxies" / name / "fit_metrics.csv") for name in names], ignore_index=True
    )
    frame.to_csv(args.output / "fit_metrics.csv", index=False)
    print(summarize(frame, args.output))


if __name__ == "__main__":
    main()
