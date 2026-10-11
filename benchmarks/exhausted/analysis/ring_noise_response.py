"""Noise response of the ring statistics for bilinear versus cubic ring sampling.

Pure Gaussian white noise (sigma = 1) is sampled along ellipses exactly as the fitter
does, and the scatter over noise realizations of every ring statistic the fit relies
on is measured: the per-sample noise, the ring intensity under each integrator (mean,
one-pass 3-sigma clipped mean, median, 10 % trimmed mean, biweight location) and the
first- and second-harmonic coefficients that drive the centre and shape updates. The
``independent`` column is the ``1/sqrt(n_samples)`` scatter that ``n_samples``
independent pixels would give; the ratio to it is the effective noise penalty of
the sampling path.

Run with ``uv run --with pandas python -m benchmarks.exhausted.analysis.ring_noise_response``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import biweight_location
from scipy.stats import trim_mean

from isoster.fitting import fit_first_and_second_harmonics, sigma_clip
from isoster.sampling import extract_isophote_data, prepare_sampling_image

SMAS = (5, 10, 20, 40, 80, 150)
SIZE = 400
ELLIPTICITY, POSITION_ANGLE = 0.3, 0.6
STATISTICS = ("mean", "clipped_mean", "median", "trimmed_mean", "biweight", "A1", "B1", "A2", "B2", "A2_clipped")


def ring_statistics(prepared, order, x0, y0, sma):
    data = extract_isophote_data(prepared, None, x0, y0, sma, ELLIPTICITY, POSITION_ANGLE, interpolation_order=order)
    phi, intens = data.phi, data.intens
    phi_clipped, intens_clipped, _ = sigma_clip(phi, intens, 3.0, 1)
    coeffs = fit_first_and_second_harmonics(phi, intens)[0]
    coeffs_clipped = fit_first_and_second_harmonics(phi_clipped, intens_clipped)[0]
    return dict(
        n_samples=len(intens),
        sample=float(np.std(intens)),
        mean=float(np.mean(intens)),
        clipped_mean=float(np.mean(intens_clipped)),
        median=float(np.median(intens)),
        trimmed_mean=float(trim_mean(intens, 0.1)),
        biweight=float(biweight_location(intens)),
        A1=coeffs[1],
        B1=coeffs[2],
        A2=coeffs[3],
        B2=coeffs[4],
        A2_clipped=coeffs_clipped[3],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_exhausted/ring_noise_response"))
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for trial in range(args.trials):
        noise = rng.standard_normal((SIZE, SIZE))
        x0, y0 = SIZE / 2 + rng.uniform(-0.5, 0.5, 2)  # random sub-pixel phase of the ring
        for interpolation in ("bilinear", "cubic"):
            prepared, order = prepare_sampling_image(noise, interpolation)
            for sma in SMAS:
                rows.append(
                    dict(
                        trial=trial,
                        interpolation=interpolation,
                        sma=sma,
                        **ring_statistics(prepared, order, x0, y0, sma),
                    )
                )
    frame = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "ring_statistics.csv", index=False)

    scatter = frame.groupby(["interpolation", "sma"])[list(STATISTICS)].std()
    scatter["sample"] = frame.groupby(["interpolation", "sma"])["sample"].mean()
    scatter["independent"] = 1 / np.sqrt(frame.groupby(["interpolation", "sma"])["n_samples"].first())
    ratio = (scatter.loc["cubic"] / scatter.loc["bilinear"]).drop(columns="independent")
    penalty = scatter[list(STATISTICS)].div(scatter["independent"], axis=0)
    lines = [
        "# Ring statistic scatter on unit white noise (sigma = 1)",
        "",
        f"{args.trials} noise realizations, ellipticity {ELLIPTICITY}, random sub-pixel ring phase.",
        "",
        "## Scatter over realizations (sigma units)",
        "",
        scatter.round(4).to_markdown(),
        "",
        "## Cubic / bilinear scatter ratio",
        "",
        ratio.round(3).to_markdown(),
        "",
        "## Scatter relative to n_samples independent pixels (1 = no penalty)",
        "",
        penalty.round(3).to_markdown(),
        "",
    ]
    (args.output / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
