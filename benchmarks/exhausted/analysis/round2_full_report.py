"""Curated report of a round-2 truth-based analysis (``round2_small_sample.py`` output).

The measurement script writes one table per renderer and zone for every arm, which is
exhaustive but hard to read for 25 arms. This report keeps the arms that matter for the
sampling-kernel decision, uses one renderer as the headline (the log-linear harmonic-off
renderer by default, the least biased one in the small-sample study) and shows the
renderer dependence separately. All win fractions are paired: an arm is scored only on
inputs where both it and the reference produced a model.

Run with ``uv run --with pandas --with tabulate python -m
benchmarks.exhausted.analysis.round2_full_report --help``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY_ARMS = [
    ("isoster", "ref_default"),
    ("isoster", "samp_cubic"),
    ("isoster", "samp_cubic_ea"),
    ("isoster", "samp_cubic_ea_min4"),
    ("isoster", "samp_cubic_rms010"),
    ("isoster", "samp_cubic_snr200"),
    ("isoster", "samp_cubic_outer2re"),
    ("isoster", "int_median"),
    ("isoster", "samp_cubic_ea_median"),
    ("photutils", "baseline_median"),
    ("photutils", "default_bilinear"),
    ("autoprof", "baseline"),
    ("autoprof", "deep"),
]
ZONES = ("all", "inner", "mid", "outer")
RENDERERS = ("linear_off", "cubic_off", "log_off", "spline_off", "native")


def paired_wins(rms: pd.DataFrame, reference) -> pd.Series:
    """Fraction of inputs where each column beats the reference column, over inputs where both exist."""
    if reference not in rms:
        return pd.Series(np.nan, index=rms.columns)
    paired = rms.notna() & rms[reference].notna().to_numpy()[:, None]
    return (rms.lt(rms[reference], axis=0) & paired).sum() / paired.sum()


def pivot(frame: pd.DataFrame, renderer: str, zone: str, value: str = "rms") -> pd.DataFrame:
    z = frame[frame.renderer.eq(renderer) & frame.zone.eq(zone)]
    return z.pivot_table(index=["galaxy", "scenario"], columns=["tool", "arm"], values=value)


def zone_table(frame: pd.DataFrame, renderer: str, zone: str) -> pd.DataFrame:
    rms = pivot(frame, renderer, zone)
    bias = pivot(frame, renderer, zone, "bias")
    cols = [c for c in KEY_ARMS if c in rms]
    table = pd.DataFrame(
        {
            "n": rms[cols].count(),
            "median_rms": rms[cols].median(),
            "p90_rms": rms[cols].quantile(0.9),
            "median_abs_bias": bias[cols].abs().median(),
            "beats_isoster_default": paired_wins(rms, ("isoster", "ref_default"))[cols],
            "beats_samp_cubic": paired_wins(rms, ("isoster", "samp_cubic"))[cols],
            "beats_autoprof_baseline": paired_wins(rms, ("autoprof", "baseline"))[cols],
        }
    )
    table.index = [f"{t}:{a}" for t, a in table.index]
    return table


def by_group(frame: pd.DataFrame, renderer: str, zone: str, group: str, arms) -> pd.DataFrame:
    """Median RMS of selected arms per depth or redshift, with the ratio of each arm to samp_cubic."""
    rms = pivot(frame, renderer, zone)
    arms = [c for c in arms if c in rms]
    out = rms[arms].reset_index()
    out[group] = (
        frame.drop_duplicates(["galaxy", "scenario"])
        .set_index(["galaxy", "scenario"])
        .loc[list(zip(out.galaxy, out.scenario)), group]
        .to_numpy()
    )
    med = out.groupby(group)[arms].median()
    med.columns = [f"{t}:{a}" for t, a in med.columns]
    return med


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="fit_metrics.csv of a round-2 analysis")
    parser.add_argument("--switch", type=Path, default=None, help="switch_radius.csv of round2_switch_radius.py")
    parser.add_argument("--renderer", default="log_off", choices=RENDERERS)
    parser.add_argument("--output", type=Path, required=True, help="report markdown path")
    args = parser.parse_args()

    frame = pd.read_csv(args.metrics)
    frame["rms"] = 100 * frame.truth_relative_rms
    frame["bias"] = 100 * frame.flux_bias
    n_inputs = frame.drop_duplicates(["galaxy", "scenario"]).shape[0]
    ok = frame[frame.status.eq("ok") & frame.model_available]
    lines = [f"# Round-2 full-sample report ({n_inputs} inputs, headline renderer `{args.renderer}`)", ""]

    outcomes = (
        frame[frame.zone.eq("all") & frame.renderer.eq("linear_off")]
        .groupby(["tool", "arm"])
        .status.value_counts()
        .unstack(fill_value=0)
    )
    lines += ["## Fit outcomes per arm", "", outcomes.to_markdown(), ""]

    times = (
        frame[frame.zone.eq("all") & frame.renderer.eq("linear_off") & frame.status.eq("ok")]
        .groupby(["tool", "arm"])
        .wall_time_fit_s.quantile([0.5, 0.9])
        .unstack()
    )
    times.columns = ["median_s", "p90_s"]
    ref = times.loc[("isoster", "ref_default"), "median_s"]
    times["median_over_isoster_default"] = times.median_s / ref
    lines += ["## Fit wall time [s]", "", times.round(3).to_markdown(), ""]

    for zone in ZONES:
        lines += [
            f"## Zone `{zone}`, renderer `{args.renderer}` (truth RMS and flux bias in percent, paired win fractions)",
            "",
            zone_table(ok, args.renderer, zone).round(3).to_markdown(),
            "",
        ]

    primary = [
        ("isoster", "ref_default"),
        ("isoster", "samp_cubic"),
        ("autoprof", "baseline"),
        ("photutils", "baseline_median"),
    ]
    rows = {}
    for renderer in RENDERERS:
        rms = pivot(ok, renderer, "all")
        rows[renderer] = {f"{t}:{a}": rms[(t, a)].median() for t, a in primary if (t, a) in rms}
    lines += [
        "## Renderer dependence: median full-aperture RMS [%] of the primary arms",
        "",
        pd.DataFrame(rows).T.round(3).to_markdown(),
        "",
    ]

    kernel = [
        ("isoster", "ref_default"),
        ("isoster", "samp_cubic"),
        ("isoster", "samp_cubic_rms010"),
        ("isoster", "samp_cubic_snr200"),
        ("isoster", "samp_cubic_outer2re"),
        ("autoprof", "baseline"),
    ]
    for zone in ("all", "outer"):
        for group in ("depth", "redshift"):
            lines += [
                f"## Zone `{zone}` median RMS [%] by {group}, renderer `{args.renderer}`",
                "",
                by_group(ok, args.renderer, zone, group, kernel).round(3).to_markdown(),
                "",
            ]

    rms_all = pivot(ok, args.renderer, "all")
    for arm in ("samp_cubic", "samp_cubic_rms010"):
        if ("isoster", arm) not in rms_all:
            continue
        ratio = (rms_all["isoster", arm] / rms_all["isoster", "ref_default"]).dropna().sort_values(ascending=False)
        lines += [
            f"## Regressions: `{arm}` / `ref_default` full RMS ratio, renderer `{args.renderer}`",
            "",
            f"quantiles 10/50/90: {np.quantile(ratio, [0.1, 0.5, 0.9]).round(3).tolist()}; "
            f"worse by >20 % in {int((ratio > 1.2).sum())} of {len(ratio)} inputs",
            "",
            ratio.head(8).round(3).to_frame("ratio").to_markdown(),
            "",
        ]

    if args.switch is not None:
        sw = pd.read_csv(args.switch)
        sw["fired"] = sw.switch_sma.notna()
        sw["switch_over_Re"] = sw.switch_sma / sw.reference_pix
        g = sw.groupby(["arm", "depth"])
        table = (
            g.fired.mean()
            .to_frame("fired_fraction")
            .join(g.switch_over_Re.quantile([0.1, 0.5, 0.9]).unstack().set_axis(["p10", "p50", "p90"], axis=1))
        )
        lines += ["## Where the switch fired (multiples of the reference radius)", "", table.round(3).to_markdown(), ""]

    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
