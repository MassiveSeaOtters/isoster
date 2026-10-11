"""Where the self-contained sampling switch fired across a round-2 campaign.

Campaign run records keep the profile but not ``sampling_switch_sma``, so every
Isoster switch arm (``samp_cubic_rms010``, ``samp_cubic_snr200``) is refitted from its
saved ``config.yaml`` on the campaign image. The refit must reproduce the saved profile
exactly, which ties the recovered switch radius to the campaign. The oracle arm
``samp_cubic_outer2re`` needs no refit: its radius is the configured
``sampling_cubic_max_sma``. The summary tabulates how often each switch fired and at
what multiple of the reference radius, by image depth and redshift.

Run with ``uv run --with pandas --with tabulate python -m
benchmarks.exhausted.analysis.round2_switch_radius --help``.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from benchmarks.exhausted.analysis.publication_huang import load_profile, profile_dicts
from benchmarks.exhausted.analysis.round2_kernel_switch_refit import same_profile, snr_at_reference_multiples
from isoster import IsosterConfig, fit_image

SWITCH_ARMS = ("samp_cubic_rms010", "samp_cubic_snr200")
ORACLE_ARM = "samp_cubic_outer2re"


def analyze_input(input_path: str) -> list[dict]:
    input_dir = Path(input_path)
    galaxy, scenario = input_dir.name.split("__")
    manifest = json.loads((input_dir / "MANIFEST.json").read_text())
    reference = float(manifest["effective_Re_pix"])
    base = dict(
        galaxy=galaxy,
        scenario=scenario,
        depth=scenario.split("_")[0],
        redshift=round(float(scenario[-3:]) / 100, 2),
        reference_pix=reference,
    )
    image = None
    rows = []
    for arm in SWITCH_ARMS + (ORACLE_ARM,):
        arm_dir = input_dir / "isoster" / "arms" / arm
        record = json.loads((arm_dir / "run_record.json").read_text())
        if record.get("status") != "ok":
            continue
        config = yaml.safe_load((arm_dir / "config.yaml").read_text())
        saved = profile_dicts(load_profile(arm_dir / "profile.fits"))
        if arm == ORACLE_ARM:
            switch = float(config["sampling_cubic_max_sma"])
            rows.append(
                dict(
                    base,
                    arm=arm,
                    switch_sma=switch,
                    replay_exact=True,
                    n_isophotes=len(saved),
                    **snr_at_reference_multiples(saved, reference),
                )
            )
            continue
        if image is None:
            image = fits.getdata(manifest["extra"]["fits_path"]).astype(float)
        result = fit_image(image, None, IsosterConfig(**config))
        switch = result.get("sampling_switch_sma")
        rows.append(
            dict(
                base,
                arm=arm,
                switch_sma=np.nan if switch is None else float(switch),
                replay_exact=same_profile(result["isophotes"], saved),
                n_isophotes=len(saved),
                **snr_at_reference_multiples(saved, reference),
            )
        )
    return rows


def summarize(frame, output: Path) -> str:
    frame = frame.copy()
    frame["fired"] = frame.switch_sma.notna()
    frame["switch_over_reference"] = frame.switch_sma / frame.reference_pix
    lines = ["# Sampling switch radius across the campaign", ""]
    lines += [
        f"inputs: {frame.groupby('arm').size().to_dict()}; refits reproducing the saved profile exactly: {frame.replay_exact.mean():.4f}",
        "",
    ]

    def table(groups):
        g = frame.groupby(groups)
        out = g.fired.mean().to_frame("fired_fraction")
        q = g.switch_over_reference.quantile([0.1, 0.5, 0.9]).unstack()
        q.columns = ["switch_over_Re_p10", "switch_over_Re_p50", "switch_over_Re_p90"]
        return out.join(q)

    lines += ["## Per arm", "", table("arm").round(3).to_markdown(), ""]
    lines += ["## Per arm and depth", "", table(["arm", "depth"]).round(3).to_markdown(), ""]
    lines += ["## Per arm and redshift", "", table(["arm", "redshift"]).round(3).to_markdown(), ""]
    snr = (
        frame[frame.arm.eq(SWITCH_ARMS[0])]
        .groupby(["depth", "redshift"])[[c for c in frame.columns if c.startswith("snr_at_")]]
        .median()
    )
    lines += [
        "## Median ring signal-to-noise at multiples of the reference radius (from the saved profiles)",
        "",
        snr.round(0).to_markdown(),
        "",
    ]
    text = "\n".join(lines)
    (output / "summary.md").write_text(text + "\n")
    return text


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="<output_root>/<campaign_name>")
    parser.add_argument("--dataset", default="huang2013")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    inputs = sorted(str(p) for p in (args.campaign / args.dataset).glob("*__*") if (p / "MANIFEST.json").exists())
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(analyze_input, p) for p in inputs]
        for index, future in enumerate(as_completed(futures), 1):
            rows += future.result()
            if index % 100 == 0 or index == len(inputs):
                print(f"[switch] {index}/{len(inputs)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "switch_radius.csv", index=False)
    print(summarize(frame, args.output))


if __name__ == "__main__":
    main()
