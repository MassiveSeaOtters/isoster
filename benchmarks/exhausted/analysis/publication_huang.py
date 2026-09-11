"""Read-only campaign join and truth-based Huang2013 publication measurements.

Run with ``python -m benchmarks.exhausted.analysis.publication_huang --help``.
Only a new output directory is writable; fitting products are never repaired here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import map_coordinates

from benchmarks.exhausted.analysis.profile_io import read_eps, read_pa_in_radians
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid, zone_masks
from benchmarks.exhausted.analysis.scenario_summary import compute_prior_metrics, load_galaxy_manifest
from isoster.model import build_isoster_model

FROZEN_MOCK_COMMIT = "a6a90a07dc3aedd95465928ee2e93258c8ccb40a"
CAMPAIGNS = (
    "publication_single_band_round1",
    "publication_single_band_huang_recovery_ngc1209_v2_2026_09_08",
    "publication_single_band_huang_recovery_ngc1209_fix_center_2026_09_08",
    "publication_single_band_huang_recovery_ngc3585_2026_09_08",
    "publication_single_band_isoster_afterburner_huang_2026_09_09",
)
PRIMARY = {"isoster": "ref_default", "photutils": "baseline_median", "autoprof": "baseline"}
ARMS = {
    "isoster": {
        "ref_default",
        "reg_outer_damp",
        "stack_all",
        "int_median",
        "geom_simul",
        "lsb_autolock",
        "geom_ea",
        "geom_simul_ea",
        "harm_simul_ea",
        "ols_noweight",
    },
    "photutils": {"baseline_median", "aggressive_clip", "fixed_center"},
    "autoprof": {"baseline", "deep", "high_regularization", "fix_center"},
}
SCENARIOS = ["noiseless_z005"] + [f"{depth}_z{z}" for z in ("005", "020", "035", "050") for depth in ("wide", "deep")]
CONTRASTS = [(arm, "ref_default") for arm in sorted(ARMS["isoster"] - {"ref_default", "ols_noweight"})]
CONTRASTS += [("geom_simul_ea", "geom_ea"), ("geom_simul_ea", "geom_simul"), ("harm_simul_ea", "geom_simul_ea")]
METRICS = [
    "truth_relative_rms_all",
    "flux_bias_abs_all",
    "center_median_pix",
    "reach_ref",
    "truth_relative_rms_inner",
    "truth_relative_rms_mid",
    "truth_relative_rms_outer",
    "data_sigma_rms_outer",
    "ring_relative_rms",
    "max_local_resid_eps_outer",
    "max_local_resid_pa_outer_deg",
    "abs_a3n_median_outer",
    "abs_b3n_median_outer",
    "abs_a4n_median_outer",
    "abs_b4n_median_outer",
]


def digest(path: Path) -> str:
    """Hash input bytes rather than relying on timestamps."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def join_records(
    root: Path, galaxies: list[str] | None = None, autoprof_campaign: Path | None = None
) -> tuple[list[dict], list[dict]]:
    """Resolve only the documented recovery overlaps; reject unexpected duplicates."""
    selected, excluded = {}, []
    for campaign in CAMPAIGNS:
        for path in sorted((root / "fits" / campaign / "huang2013").glob("*/*/arms/*/run_record.json")):
            galaxy, scenario = path.parents[3].name.split("__")
            if galaxies and galaxy not in galaxies:
                continue
            tool, arm = path.parents[2].name, path.parent.name
            if tool not in ARMS or arm not in ARMS[tool] or scenario not in SCENARIOS:
                raise ValueError(f"Unexpected logical key: {path}")
            record = json.loads(path.read_text())
            row = dict(
                galaxy=galaxy,
                scenario=scenario,
                tool=tool,
                arm=arm,
                status=record["status"],
                record_path=str(path),
                campaign=campaign,
            )
            key = galaxy, scenario, tool, arm
            if key in selected:
                if tool == "isoster" and arm == "ref_default" and "recovery" in campaign:
                    excluded.append(dict(row, reason="recovery dependency; original retained"))
                    continue
                if (galaxy, scenario, tool, arm) == ("NGC1209", "noiseless_z005", "autoprof", "fix_center"):
                    if selected[key]["status"] != "skipped" or record["status"] != "ok":
                        raise ValueError("Unexpected fixed-center recovery statuses")
                    excluded.append(dict(selected[key], reason="dependency skip replaced by documented followup"))
                else:
                    raise ValueError(f"Undocumented duplicate: {key}")
            selected[key] = row
    names = sorted({key[0] for key in selected})
    if galaxies and set(names) != set(galaxies):
        raise ValueError("Requested galaxies missing")
    expected = {(g, s, t, a) for g in names for s in SCENARIOS for t, arms in ARMS.items() for a in arms}
    if set(selected) != expected:
        raise ValueError(f"Incomplete join: {sorted(expected - set(selected))}")
    if autoprof_campaign is not None:
        audit = autoprof_campaign / "correction_audit"
        completion = json.loads((audit / "completion.json").read_text())
        if not completion.get("all_source_hashes_unchanged") or not completion.get("all_saved_pa_correct"):
            raise ValueError("Corrected AutoProf campaign has not passed its audit")
        corrected = []
        for entry in json.loads((audit / "accepted_records.json").read_text()):
            if entry["dataset"] != "huang2013":
                continue
            galaxy, scenario = entry["galaxy"].split("/")
            if galaxy not in names:
                continue
            path = Path(entry["record_path"])
            expected_path = (
                autoprof_campaign
                / "huang2013"
                / f"{galaxy}__{scenario}"
                / "autoprof"
                / "arms"
                / entry["arm"]
                / "run_record.json"
            )
            if path.resolve() != expected_path.resolve():
                raise ValueError("Corrected record has an unexpected source path")
            if json.loads(path.read_text())["status"] != entry["status"]:
                raise ValueError("Corrected status differs from the accepted audit")
            corrected.append(
                dict(
                    galaxy=galaxy,
                    scenario=scenario,
                    tool="autoprof",
                    arm=entry["arm"],
                    status=entry["status"],
                    record_path=str(path),
                    campaign=autoprof_campaign.name,
                )
            )
        selected, superseded = replace_autoprof_records(selected, corrected)
        excluded.extend(superseded)
    return list(selected.values()), excluded


def replace_autoprof_records(selected: dict, corrected: list[dict]) -> tuple[dict, list[dict]]:
    """Replace the complete requested AutoProf roster, including failures, never best-of."""
    replacements = {}
    for row in corrected:
        key = row["galaxy"], row["scenario"], row["tool"], row["arm"]
        if key in replacements or row["tool"] != "autoprof":
            raise ValueError("Duplicate or non-AutoProf correction record")
        replacements[key] = row
    expected = {key for key in selected if key[2] == "autoprof"}
    if set(replacements) != expected:
        raise ValueError("Corrected AutoProf roster is incomplete or has unexpected records")
    superseded = [dict(selected[key], reason="superseded by audited PA-corrected campaign") for key in sorted(expected)]
    return selected | replacements, superseded


def load_profile(path: Path) -> Table:
    """Preserve FITS units and normalize geometry aliases before using shared helpers."""
    table = Table.read(path, hdu=1)
    table["pa"] = read_pa_in_radians(table)
    table["eps"] = read_eps(table)
    return table


def profile_dicts(table: Table) -> list[dict]:
    return [{name: row[name] for name in table.colnames} for row in table]


def pixel_metrics(model: np.ndarray, truth: np.ndarray, image: np.ndarray, support: np.ndarray, sigma: float) -> dict:
    """Same pixel aperture for every arm; missing support is not a zero residual."""
    count = int(support.sum())
    out = dict(npix=count, truth_relative_rms=np.nan, data_sigma_rms=np.nan, flux_bias=np.nan, flux_bias_abs=np.nan)
    if not count:
        return out
    actual, target, data = model[support], truth[support], image[support]
    norm = np.sqrt(np.mean(target**2))
    if norm > 0:
        out["truth_relative_rms"] = float(np.sqrt(np.mean((actual - target) ** 2)) / norm)
    if sigma > 0:
        out["data_sigma_rms"] = float(np.sqrt(np.mean((actual - data) ** 2)) / sigma)
    if target.sum() != 0:
        out["flux_bias"] = float((actual.sum() - target.sum()) / target.sum())
        out["flux_bias_abs"] = abs(out["flux_bias"])
    return out


def ring_truth(
    table: Table, truth: np.ndarray, use_ea: bool, median: bool, inner_floor: float, samples: int = 1024
) -> dict:
    """Dense bilinear truth extraction; a conditional diagnostic, not ellipse truth.

    Full rings only. No sigma clipping or finite-width annuli are reproduced.
    AutoProf uses the EA median diagnostic, not an exact reimplementation of its
    radius-dependent extraction. Compare 1024/2048 samples in the small gate.
    """
    angle = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    measured, targets = [], []
    for row in table:
        a, q, pa = float(row["sma"]), 1 - float(row["eps"]), float(row["pa"])
        if a < inner_floor or not 0 < q <= 1 or not np.isfinite([a, q, pa, row["intens"]]).all():
            continue
        if use_ea:
            x, y = a * np.cos(angle), a * q * np.sin(angle)
        else:
            radius = a * q / np.sqrt((q * np.cos(angle)) ** 2 + np.sin(angle) ** 2)
            x, y = radius * np.cos(angle), radius * np.sin(angle)
        xx = row["x0"] + x * np.cos(pa) - y * np.sin(pa)
        yy = row["y0"] + x * np.sin(pa) + y * np.cos(pa)
        values = map_coordinates(truth, [yy, xx], order=1, mode="constant", cval=np.nan)
        if np.isfinite(values).all():
            targets.append(float(np.median(values) if median else np.mean(values)))
            measured.append(float(row["intens"]))
    target, actual = np.array(targets), np.array(measured)
    norm = np.sqrt(np.mean(target**2)) if target.size else np.nan
    return {
        "ring_n": len(target),
        "ring_relative_rms": float(np.sqrt(np.mean((actual - target) ** 2)) / norm) if norm > 0 else np.nan,
    }


def initialize_mock(source: str, executable: str) -> None:
    """Load the frozen external renderer without editing that repository."""
    global mock, mock_manifest, mock_galaxies, profit_executable
    source = Path(source)
    os.environ["DYLD_LIBRARY_PATH"] = str(Path(executable).parent)
    spec = importlib.util.spec_from_file_location("huang_frozen_generator", source / "scripts/generate_mocks.py")
    mock = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mock
    spec.loader.exec_module(mock)
    mock_manifest = mock.resolve_run_manifest(source / "inputs/huang2013/runs/huang2013_publication_single_band.yaml")
    mock_galaxies = {
        mock.sanitize_filename(g.name): g for g in mock.load_galaxies(source / mock_manifest.model_file, None, False)
    }
    profit_executable = executable


def analyze_galaxy(rows: list[dict], output: str) -> dict:
    """Verify truth and measure all scenarios of one galaxy in a worker."""
    import pandas as pd

    started = time.monotonic()
    name = rows[0]["galaxy"]
    target = Path(output) / "galaxies" / name
    target.mkdir(parents=True, exist_ok=False)
    paths = set()
    for row in rows:
        path = Path(row["record_path"])
        paths.update([path, path.parents[3] / "MANIFEST.json"])
        if row["status"] == "ok":
            paths.add(path.parent / "profile.fits")
        manifest = json.loads((path.parents[3] / "MANIFEST.json").read_text())
        paths.add(Path(manifest["extra"]["fits_path"]))
    before = {str(p): digest(p) for p in sorted(paths)}
    galaxy = mock_galaxies[name]
    truth_cache, verification, measured = {}, [], []
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario]
        manifest_dir = Path(subset[0]["record_path"]).parents[3]
        manifest = json.loads((manifest_dir / "MANIFEST.json").read_text())
        prior_manifest = load_galaxy_manifest(manifest_dir)
        geometry = manifest["initial_geometry"]
        image = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
        recipe = next(r for r in mock_manifest.rows if r.name == scenario)
        z = recipe.redshift
        values = dict(recipe.config_values, profit_cli_path=profit_executable)
        if z not in truth_cache:
            truth_values = dict(values, noise_enabled=False)
            generator = mock.MockImageGenerator(mock.build_image_config(f"truth_z{z}", truth_values))
            at_z = mock.MockGalaxy(
                name=galaxy.name, redshift=z, components=galaxy.components, re_overall=galaxy.re_overall
            )
            truth, metadata = generator.generate(at_z)
            if metadata["engine"] != "libprofit":
                raise RuntimeError("Renderer silently fell back; rejecting truth")
            fits.writeto(target / f"truth_z{round(z * 100):03d}.fits", truth, overwrite=False)
            truth_cache[z] = truth
        truth = truth_cache[z]
        values["noise_seed"] = mock.resolve_noise_seed(galaxy.name, recipe)
        generator = mock.MockImageGenerator(mock.build_image_config(scenario, values))
        predicted = generator._add_noise(truth, {}) if values["noise_enabled"] else truth
        equal = np.array_equal(predicted.astype(np.float32), image.astype(np.float32))
        verification.append(
            dict(
                galaxy=name,
                scenario=scenario,
                exact_float32_match=equal,
                seed=str(values["noise_seed"]) if values["noise_seed"] is not None else "",
                max_abs_difference=float(np.max(abs(predicted - image))),
            )
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
        psf = prior_manifest.psf_fwhm_pix
        eligible = (radius >= psf) & (radius <= geometry["maxsma"])
        common = eligible.copy()
        profiles, models = {}, {}
        for row in subset:
            if row["status"] != "ok":
                continue
            key = row["tool"], row["arm"]
            table = load_profile(Path(row["record_path"]).parent / "profile.fits")
            profiles[key] = table
            model = build_isoster_model(image.shape, profile_dicts(table), fill=np.nan, use_harmonics=False)
            models[key] = model
            common &= np.isfinite(model)
        masks = dict(zip(("inner", "mid", "outer"), zone_masks(radius, reference)))
        masks["all"] = np.ones(image.shape, dtype=bool)
        components = manifest["extra"]["truth_components"]
        component_pa = np.array([c["pa"] for c in components])
        pa_differences = (component_pa[:, None] - component_pa[None, :] + 90) % 180 - 90
        for row in subset:
            record = json.loads(Path(row["record_path"]).read_text())
            result = dict(
                row,
                primary=row["arm"] == PRIMARY[row["tool"]],
                redshift=z,
                depth=scenario.split("_")[0],
                reference_pix=reference,
                psf_pix=psf,
                reference_psf=reference / psf,
                initial_eps=geometry["eps"],
                n_components=len(components),
                component_pa_span=float(np.max(abs(pa_differences))),
                component_size_ratio=max(c["re_px"] for c in components) / min(c["re_px"] for c in components),
                image_pixels=image.size,
                injected_sigma=sigma,
                common_support_fraction=float(common.sum() / eligible.sum()),
                successful_arms=len(models),
                flags=record.get("flags", ""),
                wall_time_fit_s=record.get("wall_time_fit_s", np.nan),
                failure_detail=json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k
                        in ("error", "error_msg", "error_message", "reason", "failure_reason", "exception", "traceback")
                    }
                ),
            )
            for metric, value in record.get("metrics", {}).items():
                if isinstance(value, (int, float, str, bool)):
                    result[f"native_{metric}"] = value
            if row["status"] != "ok":
                measured.append(result)
                continue
            key = row["tool"], row["arm"]
            table, model = profiles[key], models[key]
            valid = np.asarray(table["sma"]) > 0
            for column in ("sma", "x0", "y0", "eps", "pa", "intens"):
                valid &= np.isfinite(table[column])
            finite = table[valid]
            result["n_finite"] = len(finite)
            result["n_stop0"] = int(np.sum(finite["stop_code"] == 0))
            result["stop0_comparable"] = row["tool"] != "autoprof"
            if len(finite):
                result["reach_ref"] = float(np.max(finite["sma"]) / reference)
                outside_psf = finite[finite["sma"] >= psf]
                drift = np.hypot(outside_psf["x0"] - geometry["x0"], outside_psf["y0"] - geometry["y0"])
                result["center_median_pix"] = float(np.median(drift)) if len(drift) else np.nan
            result.update(compute_prior_metrics(finite, prior_manifest))
            use_ea = row["tool"] == "autoprof" or record.get("config_snapshot", {}).get("use_eccentric_anomaly", False)
            median = row["tool"] != "isoster" or record.get("config_snapshot", {}).get("integrator") == "median"
            result.update(ring_truth(finite, truth, use_ea, median, psf))
            result.update(angular_basis="psi" if use_ea else "phi", ring_statistic="median" if median else "mean")
            for zone, mask in masks.items():
                result.update(
                    {f"{k}_{zone}": v for k, v in pixel_metrics(model, truth, image, common & mask, sigma).items()}
                )
                result[f"eligible_npix_{zone}"] = int(np.sum(eligible & mask))
            measured.append(result)
    after = {p: digest(Path(p)) for p in before}
    if before != after:
        raise RuntimeError(f"Source bytes changed during analysis: {name}")
    pd.DataFrame(measured).to_csv(target / "fit_metrics.csv", index=False)
    pd.DataFrame(verification).to_csv(target / "truth_verification.csv", index=False)
    (target / "source_hashes.json").write_text(json.dumps(before, indent=2) + "\n")
    return dict(galaxy=name, records=len(measured), elapsed_seconds=time.monotonic() - started)


def summarize(frame, output: Path) -> None:
    """Finite-value summaries and paired contrasts; bootstrap independent galaxies."""
    import pandas as pd

    frame.groupby(["scenario", "tool", "arm", "status"]).size().rename("count").reset_index().to_csv(
        output / "coverage.csv", index=False
    )
    summary = []
    for key, group in frame.groupby(["scenario", "tool", "arm"]):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            quantiles = np.percentile(values, [16, 50, 84]) if len(values) else [np.nan] * 3
            summary.append(
                dict(zip(("scenario", "tool", "arm"), key))
                | dict(metric=metric, n=len(values), p16=quantiles[0], median=quantiles[1], p84=quantiles[2])
            )
    pd.DataFrame(summary).to_csv(output / "scenario_summary.csv", index=False)
    paired = []
    isoster = frame[(frame.tool == "isoster") & (frame.status == "ok")]
    for arm, reference in CONTRASTS:
        left = isoster[isoster.arm == arm]
        right = isoster[isoster.arm == reference]
        match = left.merge(right, on=["galaxy", "scenario"], suffixes=("_arm", "_ref"), validate="one_to_one")
        for metric in METRICS:
            for _, row in match.iterrows():
                a, b = row[metric + "_arm"], row[metric + "_ref"]
                if np.isfinite(a) and np.isfinite(b):
                    paired.append(
                        dict(
                            galaxy=row.galaxy,
                            scenario=row.scenario,
                            arm=arm,
                            reference=reference,
                            metric=metric,
                            value_arm=a,
                            value_ref=b,
                            delta=a - b,
                        )
                    )
    pairs = pd.DataFrame(paired)
    pairs.to_csv(output / "paired_deltas.csv", index=False)
    pooled = []
    random = np.random.default_rng(20260910)
    for key, group in pairs.groupby(["arm", "reference", "metric"]):
        values = group.delta.to_numpy()
        blocks = [g.delta.to_numpy() for _, g in group.groupby("galaxy")]
        bootstrap = [
            np.median(np.concatenate([blocks[i] for i in random.integers(0, len(blocks), len(blocks))]))
            for _ in range(2000)
        ]
        p16, med, p84 = np.percentile(values, [16, 50, 84])
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        sign = -1 if key[2] == "reach_ref" else 1
        pooled.append(
            dict(zip(("arm", "reference", "metric"), key))
            | dict(
                n=len(values),
                galaxies=len(blocks),
                p16=p16,
                median=med,
                p84=p84,
                median_ci_low=low,
                median_ci_high=high,
                improved=float(np.mean(sign * values < -1e-12)),
                equal=float(np.mean(abs(values) <= 1e-12)),
                degraded=float(np.mean(sign * values > 1e-12)),
            )
        )
    pd.DataFrame(pooled).to_csv(output / "paired_summary.csv", index=False)


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mock-source", type=Path, required=True)
    parser.add_argument("--profit-cli", type=Path, required=True)
    parser.add_argument("--galaxies", nargs="+", help="Optional diagnostic subset")
    parser.add_argument(
        "--autoprof-campaign", type=Path, help="Audited PA-corrected campaign; required for full analysis"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if (
        subprocess.check_output(["git", "-C", str(args.mock_source), "rev-parse", "HEAD"], text=True).strip()
        != FROZEN_MOCK_COMMIT
    ):
        raise ValueError("Wrong mock generator revision")
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1--8 workers")
    if args.output.resolve().parent != (args.root / "analysis").resolve():
        raise ValueError("Output must be a new child of campaign/analysis")
    if not args.galaxies and args.autoprof_campaign is None:
        parser.error("Full analysis requires an audited --autoprof-campaign")
    rows, excluded = join_records(args.root, args.galaxies, args.autoprof_campaign)
    names = sorted({r["galaxy"] for r in rows})
    if not args.galaxies and len(names) != 93:
        raise ValueError("Full Huang2013 analysis requires all 93 galaxies")
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output / "manifest.csv", index=False)
    pd.DataFrame(excluded).to_csv(args.output / "excluded_records.csv", index=False)
    metadata = dict(
        arguments={k: str(v) for k, v in vars(args).items()},
        python=sys.version,
        platform=platform.platform(),
        mock_commit=FROZEN_MOCK_COMMIT,
        profit_sha256=digest(args.profit_cli),
        analysis_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=digest(Path(__file__)),
        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize_mock, initargs=(str(args.mock_source), str(args.profit_cli))
    ) as executor:
        futures = [
            executor.submit(analyze_galaxy, [r for r in rows if r["galaxy"] == name], str(args.output))
            for name in names
        ]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[analysis] {index}/{len(names)} {result}", flush=True)
    frame = pd.concat(
        [pd.read_csv(args.output / "galaxies" / name / "fit_metrics.csv") for name in names], ignore_index=True
    )
    frame.to_csv(args.output / "fit_metrics.csv", index=False)
    summarize(frame, args.output)
    print(f"[analysis] complete: {len(frame)} records in {args.output}", flush=True)


if __name__ == "__main__":
    main()
