"""Combined Huang2013 + S4G summary of the round-2 truth-based campaigns, every metric.

Reads the ``fit_metrics.csv`` written by ``round2_small_sample.py`` and the
``switch_radius.csv`` written by ``round2_switch_radius.py`` for each dataset and writes
one markdown file with the two datasets side by side: fit outcomes, fit wall time, profile
metrics (isophote count, converged fraction, centre offset), truth RMS and flux bias per
zone, the data residual in noise units, paired win fractions, regression ratios, depth and
redshift dependence, renderer dependence, kernel-switch radii and the integrator check.
This is the source of every number in the combined publication record.

Run with ``uv run --with pandas==2.3.3 --with tabulate python -m
benchmarks.exhausted.analysis.round2_combined_summary --help``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.exhausted.analysis.round2_full_report import KEY_ARMS, RENDERERS, ZONES, paired_wins, pivot

PRIMARY_ARMS = [
    ("isoster", "ref_default"),
    ("isoster", "samp_cubic"),
    ("isoster", "samp_cubic_rms010"),
    ("isoster", "samp_cubic_snr200"),
    ("isoster", "samp_cubic_outer2re"),
    ("photutils", "baseline_median"),
    ("autoprof", "baseline"),
]
REGRESSION_ARMS = [
    ("isoster", "samp_cubic"),
    ("isoster", "samp_cubic_ea"),
    ("isoster", "samp_cubic_rms010"),
    ("isoster", "samp_cubic_snr200"),
    ("isoster", "samp_cubic_outer2re"),
]
METRIC_COLUMNS = [
    "galaxy",
    "scenario",
    "tool",
    "arm",
    "status",
    "depth",
    "redshift",
    "renderer",
    "zone",
    "model_available",
    "wall_time_fit_s",
    "n_finite",
    "n_stop0",
    "center_median_pix",
    "center_max_pix",
    "truth_relative_rms",
    "data_sigma_rms",
    "flux_bias",
]


def label(key) -> str:
    return f"{key[0]}:{key[1]}"


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "fit_metrics.csv", usecols=METRIC_COLUMNS)
    frame["rms"] = 100 * frame.truth_relative_rms
    frame["bias"] = 100 * frame.flux_bias
    return frame


def one_row_per_fit(frame: pd.DataFrame) -> pd.DataFrame:
    """Renderer- and zone-independent columns, one row per (input, tool, arm)."""
    return frame.drop_duplicates(["galaxy", "scenario", "tool", "arm"])


def side_by_side(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-dataset tables column-wise; column names carry the dataset."""
    parts = []
    for dataset, table in tables.items():
        part = table.copy()
        part.columns = [f"{c} [{dataset}]" for c in part.columns]
        parts.append(part)
    return pd.concat(parts, axis=1)


def outcomes(fits: pd.DataFrame) -> pd.DataFrame:
    counts = fits.groupby(["tool", "arm"]).status.value_counts().unstack(fill_value=0)
    counts["not_ok"] = counts.drop(columns="ok", errors="ignore").sum(axis=1)
    out = counts[["ok", "not_ok"]]
    out.index = [label(k) for k in out.index]
    return out


def wall_time(fits: pd.DataFrame) -> pd.DataFrame:
    ok = fits[fits.status.eq("ok")]
    grouped = ok.groupby(["tool", "arm"]).wall_time_fit_s
    out = pd.DataFrame({"median_s": grouped.median(), "p90_s": grouped.quantile(0.9)})
    out["over_isoster_default"] = out.median_s / out.loc[("isoster", "ref_default"), "median_s"]
    out.index = [label(k) for k in out.index]
    return out


def profile_metrics(fits: pd.DataFrame) -> pd.DataFrame:
    """Medians over successful fits of the profile-level metrics."""
    ok = fits[fits.status.eq("ok")].copy()
    ok["converged_fraction"] = ok.n_stop0 / ok.n_finite
    grouped = ok.groupby(["tool", "arm"])
    out = pd.DataFrame(
        {
            "n_isophotes": grouped.n_finite.median(),
            "converged_fraction": grouped.converged_fraction.median(),
            "centre_offset_median_pix": grouped.center_median_pix.median(),
            "centre_offset_max_pix": grouped.center_max_pix.median(),
            "centre_offset_max_p90_pix": grouped.center_max_pix.quantile(0.9),
        }
    )
    out.index = [label(k) for k in out.index]
    return out


def select(table: pd.DataFrame, arms) -> pd.DataFrame:
    keep = [label(a) for a in arms if label(a) in table.index]
    return table.loc[keep]


def rms_by_zone(frame: pd.DataFrame, renderer: str, statistic: str) -> pd.DataFrame:
    out = {}
    for zone in ZONES:
        rms = pivot(frame, renderer, zone)
        cols = [c for c in KEY_ARMS if c in rms]
        out[zone] = rms[cols].median() if statistic == "median" else rms[cols].quantile(0.9)
    table = pd.DataFrame(out)
    table.index = [label(k) for k in table.index]
    return table


def bias_by_zone(frame: pd.DataFrame, renderer: str) -> pd.DataFrame:
    out = {}
    for zone in ZONES:
        bias = pivot(frame, renderer, zone, "bias")
        cols = [c for c in KEY_ARMS if c in bias]
        out[f"{zone} signed"] = bias[cols].median()
        out[f"{zone} abs"] = bias[cols].abs().median()
    table = pd.DataFrame(out)
    table.index = [label(k) for k in table.index]
    return table


def data_residual(frame: pd.DataFrame, renderer: str) -> pd.DataFrame:
    """Median RMS of (model - noisy data) in units of the injected pixel noise, noisy inputs only."""
    noisy = frame[frame.depth.ne("noiseless")]
    out = {}
    for zone in ZONES:
        z = noisy[noisy.renderer.eq(renderer) & noisy.zone.eq(zone)]
        table = z.pivot_table(index=["galaxy", "scenario"], columns=["tool", "arm"], values="data_sigma_rms")
        cols = [c for c in KEY_ARMS if c in table]
        out[zone] = table[cols].median()
    table = pd.DataFrame(out)
    table.index = [label(k) for k in table.index]
    return table


def wins(frame: pd.DataFrame, renderer: str, zone: str) -> pd.DataFrame:
    rms = pivot(frame, renderer, zone)
    cols = [c for c in KEY_ARMS if c in rms]
    table = pd.DataFrame(
        {
            "beats_isoster_default": paired_wins(rms, ("isoster", "ref_default"))[cols],
            "beats_samp_cubic": paired_wins(rms, ("isoster", "samp_cubic"))[cols],
            "beats_autoprof_baseline": paired_wins(rms, ("autoprof", "baseline"))[cols],
        }
    )
    table.index = [label(k) for k in table.index]
    return table


def regressions(frame: pd.DataFrame, renderer: str, zone: str = "all") -> pd.DataFrame:
    rms = pivot(frame, renderer, zone)
    rows = {}
    for arm in REGRESSION_ARMS:
        ratio = (rms[arm] / rms[("isoster", "ref_default")]).dropna()
        rows[label(arm)] = dict(
            n=len(ratio),
            q10=ratio.quantile(0.1),
            q50=ratio.median(),
            q90=ratio.quantile(0.9),
            worse_than_default_by_20pct=int((ratio > 1.2).sum()),
            worst=ratio.max(),
        )
    return pd.DataFrame(rows).T


def by_group(frame: pd.DataFrame, renderer: str, zone: str, group: str) -> pd.DataFrame:
    rms = pivot(frame, renderer, zone)
    arms = [c for c in PRIMARY_ARMS if c in rms]
    keys = frame.drop_duplicates(["galaxy", "scenario"]).set_index(["galaxy", "scenario"])[group]
    out = rms[arms].copy()
    out[group] = keys.loc[out.index].to_numpy()
    med = out.groupby(group)[arms].median()
    med.columns = [label(c) for c in med.columns]
    return med


def renderer_table(frame: pd.DataFrame, zone: str) -> pd.DataFrame:
    rows = {}
    for renderer in RENDERERS:
        rms = pivot(frame, renderer, zone)
        arms = [c for c in PRIMARY_ARMS if c in rms]
        rows[renderer] = {label(c): rms[c].median() for c in arms}
    return pd.DataFrame(rows).T


def switch_table(switch: pd.DataFrame) -> pd.DataFrame:
    switch = switch.copy()
    switch["switch_over_re"] = switch.switch_sma / switch.reference_pix
    switch["fired"] = switch.switch_sma.notna()
    grouped = switch.groupby(["arm", "depth"])
    out = pd.DataFrame(
        {
            "fired_fraction": grouped.fired.mean(),
            "p10": grouped.switch_over_re.quantile(0.1),
            "p50": grouped.switch_over_re.median(),
            "p90": grouped.switch_over_re.quantile(0.9),
        }
    )
    out.index = [f"{a} / {d}" for a, d in out.index]
    return out


def ring_snr(switch: pd.DataFrame) -> pd.DataFrame:
    one = switch.drop_duplicates(["galaxy", "scenario"])
    cols = [c for c in one.columns if c.startswith("snr_at_")]
    return one.groupby(["depth", "redshift"])[cols].median()


def integrator(frame: pd.DataFrame, renderer: str) -> pd.DataFrame:
    rms = pivot(frame, renderer, "all")
    pairs = [
        (("isoster", "int_median"), ("isoster", "ref_default")),
        (("isoster", "samp_cubic_ea_median"), ("isoster", "samp_cubic_ea")),
    ]
    rows = {}
    for median_arm, mean_arm in pairs:
        both = rms[[median_arm, mean_arm]].dropna()
        rows[f"{label(median_arm)} vs {label(mean_arm)}"] = dict(
            median_rms_median_integrator=both[median_arm].median(),
            median_rms_mean_integrator=both[mean_arm].median(),
            median_beats_mean=(both[median_arm] < both[mean_arm]).mean(),
        )
    return pd.DataFrame(rows).T


def campaign_facts(frame: pd.DataFrame) -> pd.Series:
    fits = one_row_per_fit(frame)
    inputs = fits.drop_duplicates(["galaxy", "scenario"])
    return pd.Series(
        {
            "galaxies": inputs.galaxy.nunique(),
            "scenarios": inputs.scenario.nunique(),
            "inputs": len(inputs),
            "arms": fits.drop_duplicates(["tool", "arm"]).shape[0],
            "fits": len(fits),
            "fits_ok": int(fits.status.eq("ok").sum()),
            "isoster_failures": int(fits[fits.tool.eq("isoster") & fits.status.ne("ok")].shape[0]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--analysis",
        nargs="+",
        required=True,
        metavar="DATASET=DIR",
        help="round2_small_sample.py output folder per dataset, e.g. huang2013=/path s4g=/path",
    )
    parser.add_argument(
        "--switch",
        nargs="*",
        default=[],
        metavar="DATASET=DIR",
        help="round2_switch_radius.py output folder per dataset",
    )
    parser.add_argument("--renderer", default="log_off", choices=RENDERERS, help="headline renderer")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    analysis = dict(item.split("=", 1) for item in args.analysis)
    switches = dict(item.split("=", 1) for item in args.switch)
    frames, fits = {}, {}
    for dataset, folder in analysis.items():
        frame = load(Path(folder))
        frames[dataset] = frame[frame.status.eq("ok") & frame.model_available]
        fits[dataset] = one_row_per_fit(frame)

    r = args.renderer
    sections: list[tuple[str, pd.DataFrame]] = []
    sections.append(("Campaign facts", pd.DataFrame({d: campaign_facts(f) for d, f in fits.items()})))
    sections.append(
        ("Fit outcomes (successful fits and failures per arm)", side_by_side({d: outcomes(f) for d, f in fits.items()}))
    )
    sections.append(
        (
            "Fit wall time [s] and ratio to the Isoster default",
            side_by_side({d: select(wall_time(f), KEY_ARMS) for d, f in fits.items()}),
        )
    )
    sections.append(
        (
            "Profile metrics: isophote count, converged fraction (stop code 0), centre offset from the true centre [px]",
            side_by_side({d: select(profile_metrics(f), KEY_ARMS) for d, f in fits.items()}),
        )
    )
    sections.append(
        (
            f"Median truth RMS [%] per zone, renderer `{r}`",
            side_by_side({d: rms_by_zone(f, r, "median") for d, f in frames.items()}),
        )
    )
    sections.append(
        (
            f"90th-percentile truth RMS [%] per zone, renderer `{r}`",
            side_by_side({d: rms_by_zone(f, r, "p90") for d, f in frames.items()}),
        )
    )
    sections.append(
        (
            f"Flux bias [%] per zone (median signed, median absolute), renderer `{r}`",
            side_by_side({d: bias_by_zone(f, r) for d, f in frames.items()}),
        )
    )
    sections.append(
        (
            f"Data residual in units of the pixel noise (median over noisy inputs), renderer `{r}`",
            side_by_side({d: data_residual(f, r) for d, f in frames.items()}),
        )
    )
    for zone in ("all", "inner", "mid", "outer"):
        sections.append(
            (
                f"Paired win fractions, zone `{zone}`, renderer `{r}`",
                side_by_side({d: wins(f, r, zone) for d, f in frames.items()}),
            )
        )
    sections.append(
        (
            f"Regression ratios to the Isoster default, zone `all`, renderer `{r}`",
            side_by_side({d: regressions(f, r) for d, f in frames.items()}),
        )
    )
    for zone in ("all", "outer"):
        for group in ("depth", "redshift"):
            sections.append(
                (
                    f"Median truth RMS [%] by {group}, zone `{zone}`, renderer `{r}`",
                    pd.concat({d: by_group(f, r, zone, group) for d, f in frames.items()}, names=["dataset", group]),
                )
            )
    for zone in ("all", "inner", "mid", "outer"):
        sections.append(
            (
                f"Renderer dependence: median truth RMS [%], zone `{zone}`",
                pd.concat({d: renderer_table(f, zone) for d, f in frames.items()}, names=["dataset", "renderer"]),
            )
        )
    sections.append(
        (
            f"Integrator: median against clipped mean, zone `all`, renderer `{r}`",
            pd.concat({d: integrator(f, r) for d, f in frames.items()}, names=["dataset", "pair"]),
        )
    )
    for dataset, folder in switches.items():
        switch = pd.read_csv(Path(folder) / "switch_radius.csv")
        exact = switch.replay_exact.mean() if "replay_exact" in switch else np.nan
        sections.append(
            (
                f"Kernel switch radius in units of R_ref, {dataset} (refits replaying the saved profile exactly: {exact:.4f})",
                switch_table(switch),
            )
        )
        sections.append((f"Median ring S/N at multiples of R_ref, {dataset}", ring_snr(switch)))

    lines = [f"# Round-2 combined summary ({', '.join(frames)}), headline renderer `{r}`", ""]
    for title, table in sections:
        lines += [f"## {title}", "", table.round(4).to_markdown(), ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output} ({len(sections)} sections)")


if __name__ == "__main__":
    main()
