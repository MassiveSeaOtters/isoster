"""Audit and summarize all completed retained-population reconstruction products."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.exhausted.analysis.publication_huang import PRIMARY, digest

METRICS = ("truth_relative_rms", "flux_bias", "flux_bias_abs", "data_sigma_rms")


def matched_wins(frame):
    """Three-way wins require all designated primary tools on identical pixels."""
    rows = []
    for (mode, zone), group in frame[frame.primary].groupby(["mode", "zone"]):
        for metric in ("truth_relative_rms", "flux_bias_abs", "data_sigma_rms"):
            for scenario in ["ALL", *sorted(group.scenario.unique())]:
                subset = group if scenario == "ALL" else group[group.scenario.eq(scenario)]
                pivot = (
                    subset.pivot(index=["galaxy", "scenario"], columns="tool", values=metric)
                    .reindex(columns=list(PRIMARY))
                    .dropna()
                )
                tied = np.isclose(pivot.to_numpy(), pivot.min(axis=1).to_numpy()[:, None], rtol=1e-9, atol=1e-12)
                for index, tool in enumerate(PRIMARY):
                    rows.append(
                        dict(
                            mode=mode,
                            zone=zone,
                            metric=metric,
                            scenario=scenario,
                            tool=tool,
                            denominator=len(pivot),
                            sole_wins=int((tied[:, index] & (tied.sum(axis=1) == 1)).sum()),
                            tied_best=int((tied[:, index] & (tied.sum(axis=1) > 1)).sum()),
                            eligible_inputs=subset[["galaxy", "scenario"]].drop_duplicates().shape[0],
                        )
                    )
    return pd.DataFrame(rows)


def contrasts(frame):
    """Paired arm/default and mode/control changes, keeping galaxy consistency."""
    pairs = []
    for (mode, zone, tool), group in frame.groupby(["mode", "zone", "tool"]):
        reference = group[group.arm.eq(PRIMARY[tool])].set_index(["galaxy", "scenario"])
        for arm, actual in group.groupby("arm"):
            if arm == PRIMARY[tool]:
                continue
            matched = actual.set_index(["galaxy", "scenario"]).join(reference[list(METRICS)], rsuffix="_reference")
            for metric in METRICS:
                for key, row in matched.iterrows():
                    pairs.append(
                        dict(
                            kind="arm_vs_primary",
                            mode=mode,
                            zone=zone,
                            tool=tool,
                            arm=arm,
                            galaxy=key[0],
                            scenario=key[1],
                            metric=metric,
                            value=row[metric],
                            reference=row[metric + "_reference"],
                        )
                    )
    for (tool, arm, zone), group in frame.groupby(["tool", "arm", "zone"]):
        for mode, control in [
            ("shared_spline_on", "shared_spline_off"),
            ("shared_spline_off", "shared_baseline"),
            ("native", "shared_baseline"),
        ]:
            reference = group[group["mode"].eq(control)].set_index(["galaxy", "scenario"])
            matched = (
                group[group["mode"].eq(mode)]
                .set_index(["galaxy", "scenario"])
                .join(reference[list(METRICS)], rsuffix="_reference")
            )
            for metric in METRICS:
                for key, row in matched.iterrows():
                    pairs.append(
                        dict(
                            kind="mode_vs_" + control,
                            mode=mode,
                            zone=zone,
                            tool=tool,
                            arm=arm,
                            galaxy=key[0],
                            scenario=key[1],
                            metric=metric,
                            value=row[metric],
                            reference=row[metric + "_reference"],
                        )
                    )
    result = pd.DataFrame(pairs)
    result["delta"] = result.value - result.reference
    result["valid"] = np.isfinite(result.value) & np.isfinite(result.reference)
    result["tie"] = result.valid & np.isclose(result.value, result.reference, rtol=1e-9, atol=1e-12)
    result["lower"] = result.valid & ~result.tie & (result.value < result.reference)
    keys = ["kind", "mode", "zone", "tool", "arm", "metric"]
    summary = (
        result.groupby(keys)
        .agg(
            roster=("valid", "size"),
            denominator=("valid", "sum"),
            lower=("lower", "sum"),
            ties=("tie", "sum"),
            median_delta=("delta", "median"),
        )
        .reset_index()
    )
    galaxy = (
        result[result.valid]
        .groupby(keys + ["galaxy"])
        .agg(
            conditions=("delta", "size"),
            median_delta=("delta", "median"),
            lower_conditions=("lower", "sum"),
            ties=("tie", "sum"),
        )
        .reset_index()
    )
    return result, summary, galaxy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    sources, products, audits, frames, checks, outcomes = {}, [], [], [], [], []
    for galaxy, scenario in manifest["roster"]:
        folder = args.run / f"{galaxy}__{scenario}"
        audit = json.loads((folder / "complete.json").read_text())
        assert audit["pages"] == 10 and audit["outcomes"] == 68
        for path, value in audit["source_hashes"].items():
            if path in sources:
                assert sources[path] == value
            sources[path] = value
        for name, value in audit["product_hashes"].items():
            path = folder / name
            assert digest(path) == value, str(path)
            products.append(
                dict(galaxy=galaxy, scenario=scenario, path=str(path), sha256=value, bytes=path.stat().st_size)
            )
        measured = pd.read_csv(folder / "metrics.csv")
        assert len(measured) == 272
        accepted = pd.read_csv(folder / "accepted_rows.csv")
        assert len(accepted) == 17
        local_outcomes = pd.read_csv(folder / "outcomes.csv")
        expected = accepted[["tool", "arm", "status"]].rename(columns={"status": "fit_status"})
        for mode in manifest["modes"]:
            selected = local_outcomes[local_outcomes["mode"].eq(mode)]
            pd.testing.assert_frame_equal(
                selected[["tool", "arm", "fit_status"]].reset_index(drop=True), expected.reset_index(drop=True)
            )
            display = pd.read_csv(folder / f"{mode}_display.csv").set_index(["tool", "arm"])
            for row in measured[measured["mode"].eq(mode)].itertuples():
                for metric in METRICS:
                    np.testing.assert_allclose(
                        display.loc[(row.tool, row.arm), f"{metric}_{row.zone}"],
                        getattr(row, metric),
                        rtol=1e-12,
                        atol=1e-14,
                        equal_nan=True,
                    )
        audits.append({k: v for k, v in audit.items() if k not in ("source_hashes", "product_hashes")})
        frames.append(measured)
        checks.append(pd.read_csv(folder / "baseline_reproduction.csv"))
        outcomes.append(local_outcomes)
    for path, value in sources.items():
        assert digest(Path(path)) == value, path
    historical = {}
    for galaxy in sorted({g for g, s in manifest["roster"]}):
        historical.update(
            json.loads((Path(manifest["measurements"]) / "galaxies" / galaxy / "source_hashes.json").read_text())
        )
    for path, value in historical.items():
        assert digest(Path(path)) == value, path
    frame = pd.concat(frames, ignore_index=True)
    retained = pd.concat(outcomes, ignore_index=True)
    frame.to_csv(args.output / "metrics.csv", index=False)
    retained.to_csv(args.output / "outcomes.csv", index=False)
    pd.concat(checks).to_csv(args.output / "baseline_reproduction.csv", index=False)
    pd.DataFrame(products).to_csv(args.output / "product_inventory.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output / "resources.csv", index=False)
    matched_wins(frame).to_csv(args.output / "primary_wins.csv", index=False)
    pair, summary, galaxy = contrasts(frame)
    pair.to_csv(args.output / "paired_changes.csv", index=False)
    summary.to_csv(args.output / "paired_summary.csv", index=False)
    galaxy.to_csv(args.output / "galaxy_consistency.csv", index=False)
    retained.groupby(["mode", "tool", "arm", "fit_status", "status", "reason"], dropna=False).size().rename(
        "count"
    ).reset_index().to_csv(args.output / "coverage.csv", index=False)
    frame.groupby(["mode", "tool", "arm", "zone", "reason"], dropna=False).size().rename("count").reset_index().to_csv(
        args.output / "zone_coverage.csv", index=False
    )
    for name, keys in [
        ("descriptive_summary", ["mode", "tool", "arm", "zone"]),
        ("scenario_summary", ["mode", "tool", "arm", "zone", "scenario"]),
    ]:
        frame.groupby(keys)[list(METRICS)].agg(["count", "median", "mean", "min", "max"]).to_csv(
            args.output / f"{name}.csv"
        )
    accepted = pd.read_csv(Path(manifest["measurements"]) / "fit_metrics.csv")
    accepted.groupby(["tool", "arm", "status"]).wall_time_fit_s.agg(["count", "median", "min", "max"]).to_csv(
        args.output / "inherited_fit_times.csv"
    )
    (args.output / "audit.json").write_text(
        json.dumps(
            dict(
                inputs=len(audits),
                outcomes=len(retained),
                metric_rows=len(frame),
                pages=sum(a["pages"] for a in audits),
                source_hashes=sources,
                historical_source_hashes_checked=len(historical),
                product_count=len(products),
                product_bytes=sum(p["bytes"] for p in products),
                baseline_zone_checks=sum(a["baseline_checks"] for a in audits),
                policy="Descriptive summaries; no independent-condition uncertainty claims. Negative delta means lower, not necessarily better for signed bias.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(
                inputs=len(audits),
                outcomes=len(retained),
                metric_rows=len(frame),
                historical_hashes=len(historical),
                sources=len(sources),
                products=len(products),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
