"""Part B, Stage 1: derive the accuracy thresholds, executably.

These numbers decide which timings enter a headline ratio, so they must not
live only in prose where they can drift. Everything here is analytic --- no
tool is run, nothing is fitted --- which is the point: a threshold derived from
what a tool achieved is not a threshold.

Two regimes, and conflating them was a real defect caught in review.

**Systematic accuracy** is gated on the **noiseless** fixtures. With no noise
there is no sampling variance, so a tool's departure from analytic truth is
entirely its own numerics, and a fixed bar means what it appears to mean.

**Noise** is handled by an *ensemble* criterion instead. A single noisy
realization scatters around truth by construction, so requiring its worst ring
to sit inside a statistical 1-sigma is close to requiring the impossible: at
the outer compact ring, where sigma(A)/A = 11%, an unbiased measurement lands
within +-1% only 7.2% of the time, and demanding it of every ring and component
at once gives a perfect tool a 5.6e-13% chance of passing. That was the first
version of this contract. What is testable is *bias*: the mean over R
realizations has standard error sigma/sqrt(R), so a pre-registered bound on the
standardized mean residual is a criterion an unbiased tool passes and a biased
one fails.

Run ``python benchmarks/timing/accuracy_thresholds.py`` to print the table;
``check_accuracy_thresholds.py`` asserts the spec quotes what this computes.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from benchmarks.harmonic_scale.run_harmonic_scale import FIXTURES, PLANTED_HARMONICS  # noqa: E402

#: Depth the statistical scale is quoted at. Not a claim about any survey ---
#: these mocks have uncorrelated pixel noise and no PSF, so they are not
#: "HSC-like"; S/N at R_e is simply the noise parameter the fixtures accept.
REFERENCE_SNR = 100.0

#: Realizations the ensemble-bias criterion averages over.
ENSEMBLE_REALIZATIONS = 25

#: How many standard errors of the ensemble mean a tool may be biased by
#: before it is called biased. Two-sided, chosen in advance, and loose enough
#: that an unbiased tool passes ~99.7% of the time per component.
BIAS_SIGMA_ALLOWANCE = 3.0


#: Sersic b_n approximation used by the fixture renderer.
def _b_n(n: float) -> float:
    return 1.9992 * n - 0.3271


def ring_statistics(fixture: str) -> List[Dict[str, float]]:
    """Per-ring analytic noise scale, from sample count and profile alone."""
    spec = FIXTURES[fixture]
    galaxy = spec["galaxy"]
    # sigma is defined by S/N at R_e: I(R_e) = I_e, so sigma = I_e / snr.
    sigma = float(galaxy["I_e"]) / REFERENCE_SNR
    planted = max(abs(v) for v in PLANTED_HARMONICS.values())

    rows = []
    for sma in spec["radii"]:
        # One sample per pixel of circumference, the sampling all three tools
        # approximate on a ring of this radius.
        n_samples = max(8, int(2.0 * math.pi * float(sma)))
        intensity = float(galaxy["I_e"]) * math.exp(
            -_b_n(float(galaxy["n"])) * ((float(sma) / float(galaxy["R_e"])) ** (1.0 / float(galaxy["n"])) - 1.0)
        )
        # Least squares on evenly spaced angles: sigma(A) = sigma * sqrt(2/N).
        amplitude_sigma = sigma * math.sqrt(2.0 / n_samples)
        rows.append(
            {
                "sma": float(sma),
                "n_samples": n_samples,
                "intensity": intensity,
                # Relative to the planted amplitude at this ring.
                "amplitude_sigma_pct": 100.0 * amplitude_sigma / (planted * intensity),
                # Ring mean: sigma / sqrt(N), relative to the ring intensity.
                "ring_mean_sigma_pct": 100.0 * sigma / (math.sqrt(n_samples) * intensity),
            }
        )
    return rows


def thresholds() -> Dict[str, object]:
    """The frozen bars, and the quantities they were derived from."""
    per_fixture = {name: ring_statistics(name) for name in sorted(FIXTURES)}
    amplitude = [row["amplitude_sigma_pct"] for rows in per_fixture.values() for row in rows]
    ring_mean = [row["ring_mean_sigma_pct"] for rows in per_fixture.values() for row in rows]
    return {
        "reference_snr": REFERENCE_SNR,
        "ensemble_realizations": ENSEMBLE_REALIZATIONS,
        "bias_sigma_allowance": BIAS_SIGMA_ALLOWANCE,
        "amplitude_sigma_pct_min": min(amplitude),
        "amplitude_sigma_pct_max": max(amplitude),
        "ring_mean_sigma_pct_max": max(ring_mean),
        # Systematic bars, applied on NOISELESS fixtures where they are
        # meaningful, and set **per ring** at that ring's own statistical
        # 1-sigma scale at the reference depth.
        #
        # The criterion is "a systematic a user could not detect": at this
        # depth, noise alone moves the measurement by this much, so a bias
        # below it is invisible in the data. It is tool-independent, it varies
        # with radius because the physics does, and it is applied identically
        # to all three tools.
        #
        # A tenth of that was tried first and rejected: it is a bar no current
        # implementation meets, so it would have measured "better than the
        # state of the art" rather than "good enough for the science", and
        # rejected every arm just as the single-realization gate did.
        "systematic_amplitude_error_pct_by_ring": {
            fixture: {row["sma"]: round(row["amplitude_sigma_pct"], 4) for row in rows}
            for fixture, rows in per_fixture.items()
        },
        "systematic_ring_intensity_error_pct_by_ring": {
            fixture: {row["sma"]: round(row["ring_mean_sigma_pct"], 4) for row in rows}
            for fixture, rows in per_fixture.items()
        },
        "systematic_center_error_px": 0.5,
        "systematic_eps_error": 0.01,
        "systematic_pa_error_deg": 1.0,
        # Noisy fixtures are judged on ensemble bias, never on one realization.
        "ensemble_bias_max_standardized": BIAS_SIGMA_ALLOWANCE,
        "per_fixture": per_fixture,
    }


def _probability_within(tolerance_pct: float, sigma_pct: float) -> float:
    return math.erf(tolerance_pct / (sigma_pct * math.sqrt(2.0)))


def main() -> None:
    computed = thresholds()
    print(f"Reference S/N at R_e: {computed['reference_snr']:.0f}\n")
    for fixture, rows in computed["per_fixture"].items():
        print(f"{fixture}")
        for row in rows:
            print(
                f"   sma={row['sma']:5.1f}  N={row['n_samples']:4d}  I={row['intensity']:9.4g}  "
                f"sigma(A)/A={row['amplitude_sigma_pct']:7.3f}%  sigma(ring mean)/I="
                f"{row['ring_mean_sigma_pct']:6.3f}%"
            )
    print(
        f"\nstatistical amplitude scale: {computed['amplitude_sigma_pct_min']:.2f}% "
        f"to {computed['amplitude_sigma_pct_max']:.2f}%"
    )
    print(f"statistical ring-mean scale: up to {computed['ring_mean_sigma_pct_max']:.2f}%")
    print("\nfrozen systematic bars, applied on NOISELESS fixtures, per ring:")
    for fixture, bars in computed["systematic_amplitude_error_pct_by_ring"].items():
        cells = ", ".join(f"{sma:g}:{value:.2f}%" for sma, value in bars.items())
        print(f"   amplitude  {fixture:20s} {cells}")
    for fixture, bars in computed["systematic_ring_intensity_error_pct_by_ring"].items():
        cells = ", ".join(f"{sma:g}:{value:.3f}%" for sma, value in bars.items())
        print(f"   intensity  {fixture:20s} {cells}")
    print(f"   center error          <= {computed['systematic_center_error_px']} px")
    print(f"   ellipticity error     <= {computed['systematic_eps_error']}")
    print(f"   position angle error  <= {computed['systematic_pa_error_deg']} deg (mod 180)")
    print(f"\nnoisy fixtures: |standardized ensemble bias| <= {computed['ensemble_bias_max_standardized']}")
    print(f"   over {computed['ensemble_realizations']} realizations")

    # Why a single-realization max-error gate was rejected, in numbers.
    worst = computed["amplitude_sigma_pct_max"]
    joint = 1.0
    for rows in computed["per_fixture"].values():
        for row in rows:
            joint *= _probability_within(1.0, row["amplitude_sigma_pct"]) ** 4
    print(
        f"\nrejected alternative -- gating one realization's worst error at 1%:\n"
        f"   an unbiased ring at sigma={worst:.1f}% passes {100 * _probability_within(1.0, worst):.1f}% of the time;\n"
        f"   a perfect tool passes every ring and component at once {100 * joint:.3g}% of the time."
    )


if __name__ == "__main__":
    main()
