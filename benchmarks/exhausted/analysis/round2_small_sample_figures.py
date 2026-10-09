"""Summary figures for a round-2 small-sample analysis (``round2_small_sample.py`` output).

Figure 1: per-input full-aperture truth RMS relative to the Isoster default and to the
AutoProf baseline, one box per arm, for the linear and cubic radial renderers.
Figure 2: median truth RMS per scenario and zone for the key arms.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isoster.plotting import configure_qa_plot_style

KEY_ARMS = [
    ("isoster", "ref_default"),
    ("isoster", "geom_ea"),
    ("isoster", "samp_cubic"),
    ("isoster", "samp_cubic_ea"),
    ("isoster", "samp_cubic_ea_outer_damp"),
    ("photutils", "baseline_median"),
    ("autoprof", "baseline"),
    ("autoprof", "deep"),
]
SCENARIOS = ["noiseless_z005"] + [f"{depth}_z{z}" for z in ("005", "020", "035", "050") for depth in ("wide", "deep")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    configure_qa_plot_style()
    plt.rcParams.update({"text.usetex": False, "font.size": 9})
    frame = pd.read_csv(args.analysis / "fit_metrics.csv")
    ok = frame[frame.status.eq("ok") & frame.model_available].copy()
    ok["rms"] = 100 * ok.truth_relative_rms

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for col, renderer in enumerate(("linear_off", "cubic_off")):
        z = ok[ok.renderer.eq(renderer) & ok.zone.eq("all")]
        rms = z.pivot_table(index=["galaxy", "scenario"], columns=["tool", "arm"], values="rms")
        for row, reference in enumerate((("isoster", "ref_default"), ("autoprof", "baseline"))):
            ax = axes[row, col]
            arms = [c for c in rms.columns if c != reference]
            data = [np.log10((rms[c] / rms[reference]).dropna()) for c in arms]
            ax.boxplot(data, vert=False, tick_labels=[f"{t}:{a}" for t, a in arms], showfliers=True, flierprops=dict(ms=2))
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel(f"log10( full RMS / {reference[0]}:{reference[1]} )")
            ax.set_title(f"renderer {renderer}: relative to {reference[1]}")
            ax.grid(alpha=0.2)
    fig.suptitle(
        "Round-2 small sample: full-aperture truth RMS per input, relative to two references (left of 0 = better)"
    )
    fig.savefig(args.analysis / "figure_rms_ratios.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained", sharex=True)
    for col, zone in enumerate(("all", "inner", "mid", "outer")):
        for row, renderer in enumerate(("linear_off", "cubic_off")):
            ax = axes[row, col]
            z = ok[ok.renderer.eq(renderer) & ok.zone.eq(zone)]
            med = z.pivot_table(index="scenario", columns=["tool", "arm"], values="rms", aggfunc="median").reindex(
                SCENARIOS
            )
            for key in KEY_ARMS:
                if key in med:
                    ax.plot(range(len(SCENARIOS)), med[key], marker="o", ms=3, label=f"{key[0]}:{key[1]}")
            ax.set_yscale("log")
            ax.set_title(f"{zone} zone, renderer {renderer}")
            ax.set_xticks(range(len(SCENARIOS)))
            ax.set_xticklabels(SCENARIOS, rotation=60, ha="right")
            ax.grid(alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("median truth RMS [%]")
    axes[1, 0].set_ylabel("median truth RMS [%]")
    fig.suptitle("Round-2 small sample: median truth RMS by scenario")
    fig.savefig(args.analysis / "figure_rms_by_scenario.png", dpi=130)
    plt.close(fig)
    print("figures written to", args.analysis)


if __name__ == "__main__":
    main()
