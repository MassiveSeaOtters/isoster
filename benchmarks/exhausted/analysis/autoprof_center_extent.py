"""Read accepted primary profiles/options without fitting or changing source data."""

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from benchmarks.exhausted.analysis.publication_huang import digest, load_profile, profile_dicts
from benchmarks.exhausted.analysis.residual_zones import compute_elliptical_radius_grid
from isoster.model import build_isoster_model


def profile_measurements(table, geometry):
    """Keep all rows; separately label converged and propagation-accepted radii."""
    radius = np.asarray(table["sma"], float)
    offset = np.hypot(table["x0"] - geometry["x0"], table["y0"] - geometry["y0"])
    selected = (radius >= 2) & np.isfinite(offset)
    codes = np.asarray(table["stop_code"], int)
    positive = radius > 0
    return {
        "profile_rows": len(table),
        "last_profile_radius": float(radius.max()),
        "last_converged_radius": float(radius[codes == 0].max()) if np.any(codes == 0) else np.nan,
        "last_propagation_accepted_radius": float(radius[np.isin(codes, [0, 1, 2])].max())
        if np.any(np.isin(codes, [0, 1, 2]))
        else np.nan,
        "center_median_pix": float(np.median(offset[selected])) if selected.any() else np.nan,
        "center_max_pix": float(np.max(offset[selected])) if selected.any() else np.nan,
        "center_last_pix": float(offset[np.argmax(radius)]),
        "radius_step_median": float(np.median(radius[positive][1:] / radius[positive][:-1])),
        "nonconverged_fraction": float(np.mean(codes != 0)),
        "last_stop_code": int(codes[np.argmax(radius)]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", action="store_true", help="Measure shared harmonic-off finite support")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    hashes = {}

    def track(path):
        path = Path(path)
        hashes[str(path)] = digest(path)
        return path

    frame = pd.read_csv(track(args.measurements / "fit_metrics.csv"))
    rows = frame[frame.primary & frame.tool.isin(["isoster", "autoprof"])]
    if args.limit:
        keys = rows[["galaxy", "scenario"]].drop_duplicates().head(args.limit)
        rows = rows.merge(keys, on=["galaxy", "scenario"], validate="many_to_one")
    results = []
    for row in rows.itertuples():
        parent = Path(row.record_path).parent
        record = json.loads(track(row.record_path).read_text())
        manifest = json.loads(track(parent.parents[2] / "MANIFEST.json").read_text())
        geometry = manifest["initial_geometry"]
        result = dict(
            galaxy=row.galaxy,
            scenario=row.scenario,
            tool=row.tool,
            status=row.status,
            campaign=row.campaign,
            record_path=row.record_path,
            requested_maxsma=geometry["maxsma"],
            has_variance=manifest["has_variance"],
            has_mask=manifest["has_mask"],
            inner_cut_pix=2,
            truth_relative_rms_all=row.truth_relative_rms_all,
            truth_relative_rms_outer=row.truth_relative_rms_outer,
            flux_bias_all=row.flux_bias_all,
        )
        config = yaml.safe_load(track(parent / "config.yaml").read_text()) if (parent / "config.yaml").exists() else {}
        if row.tool == "autoprof":
            tag = f"{row.galaxy}__{row.scenario}"
            options = json.loads(track(parent / "tmp" / f"{tag}_options.json").read_text())
            result.update(
                retry=bool(record.get("small_image_fallback")),
                truncate=options.get("ap_truncate_evaluation", False),
                extractfull=options.get("ap_extractfull", False),
                set_center="ap_set_center" in options,
                guess_matches_truth=options.get("ap_guess_center") == {"x": geometry["x0"], "y": geometry["y0"]},
                background=options.get("ap_set_background"),
                fit_limit=options.get("ap_fit_limit"),
                centeringring=options.get("ap_centeringring", 10),
            )
            assert result["background"] == 0 and not result["set_center"]
            assert result["truncate"] == result["retry"] and not result["extractfull"]
            if row.status == "ok":
                aux = track(parent / "raw" / f"{tag}.aux").read_text()
                result["geometry_fit_radius"] = float(re.search(r"fit limit semi-major axis: ([\d.]+)", aux)[1])
                raw = np.atleast_1d(
                    np.genfromtxt(track(parent / "raw" / f"{tag}.prof"), delimiter=",", names=True, skip_header=1)
                )
                result["raw_extraction_radius"] = float(raw["R"].max() / options["ap_pixscale"])
                result["filtered_rows"] = int(np.sum(~(raw["SB"] < 90)))
                assert result["filtered_rows"] == record["autoprof_n_filtered"]
        else:
            for name in [
                "fix_center",
                "lsb_auto_lock",
                "use_outer_center_regularization",
                "permissive_geometry",
                "sma0",
                "maxsma",
                "astep",
                "integrator",
                "geometry_update_mode",
                "clip_max_shift",
                "geometry_damping",
            ]:
                result[name] = config.get(name)
        if row.status == "ok":
            table = load_profile(track(parent / "profile.fits"))
            result.update(profile_measurements(table, geometry))
            if row.tool == "autoprof":
                # The wrapper synthesizes these codes; they are not optimizer diagnostics.
                for name in [
                    "last_converged_radius",
                    "last_propagation_accepted_radius",
                    "nonconverged_fraction",
                    "last_stop_code",
                ]:
                    result[name] = np.nan
                assert np.ptp(table["x0"]) == 0 and np.ptp(table["y0"]) == 0
            else:
                result["geometry_fit_radius"] = result["last_profile_radius"]
                result["raw_extraction_radius"] = result["last_profile_radius"]
            if args.models:
                model = build_isoster_model(
                    tuple(manifest["image_shape"]), profile_dicts(table), fill=np.nan, use_harmonics=False
                )
                radius = compute_elliptical_radius_grid(
                    model.shape, geometry["x0"], geometry["y0"], geometry["eps"], geometry["pa"]
                )
                finite = np.isfinite(model)
                result["shared_off_finite_radius_truth_frame"] = float(radius[finite].max()) if finite.any() else np.nan
                result["shared_off_finite_pixels"] = int(finite.sum())
        results.append(result)
    output = pd.DataFrame(results)
    output.to_csv(args.output / "profiles.csv", index=False)
    numeric = [
        "center_median_pix",
        "center_max_pix",
        "center_last_pix",
        "last_profile_radius",
        "geometry_fit_radius",
        "raw_extraction_radius",
        "nonconverged_fraction",
    ]
    output.groupby(["tool", "scenario"])[numeric].median().to_csv(args.output / "scenario_medians.csv")
    output.groupby(["tool", "galaxy"])[numeric].median().to_csv(args.output / "galaxy_medians.csv")
    paired = output.pivot(index=["galaxy", "scenario"], columns="tool", values=numeric)
    paired.columns = ["_".join(column) for column in paired.columns]
    paired["profile_extent_ratio"] = paired.last_profile_radius_autoprof / paired.last_profile_radius_isoster
    paired["geometry_extent_ratio"] = paired.geometry_fit_radius_autoprof / paired.geometry_fit_radius_isoster
    paired.to_csv(args.output / "paired.csv")
    assert len(output) == len(rows) and not output.duplicated(["galaxy", "scenario", "tool"]).any()
    for path, value in hashes.items():
        assert digest(Path(path)) == value, f"Source changed: {path}"
    audit = dict(
        rows=len(output),
        inputs=len(paired),
        elapsed_seconds=perf_counter() - started,
        status_counts=output.groupby(["tool", "status"]).size().to_dict(),
        source_hashes=hashes,
        models=args.models,
    )
    audit["status_counts"] = {str(k): int(v) for k, v in audit["status_counts"].items()}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
