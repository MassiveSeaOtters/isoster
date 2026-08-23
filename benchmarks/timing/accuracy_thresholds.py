"""Part B, Stage 1: derive the accuracy thresholds, executably.

These numbers decide which timings enter a headline ratio, so they must not
live only in prose where they can drift. Everything here is analytic --- no
tool is run and nothing is fitted --- which is the point: a threshold derived
from what a tool achieved is not a threshold.

Three defects in earlier versions are worth stating, because each would have
produced a gate that looked principled and was not.

**The statistical scale must be computed against the quantity actually
measured.** These arms report *raw intensity* harmonics, whose truth is not
``planted_fraction * intensity``: it depends on the radial gradient, the Sersic
curvature, products between modes, and each component's own planted amplitude.
An earlier version used the largest planted fraction times the ring intensity
for all four components, giving one bar that was simultaneously too strict for
the large modes and too loose for the small ones --- and since eligibility
takes the worst component, systematically penalising the smallest. The bars are
now **per component**, from :func:`integrated_harmonic_truth`, the same dense
Fourier integration Part A measures against.

**Systematic and statistical accuracy are different regimes.** Gating a single
noisy realization's worst error at a statistical 1-sigma is close to requiring
the impossible: an unbiased ring at sigma = 9.8% lands inside +-1% only 8% of
the time. Systematic accuracy is therefore gated on **noiseless** fixtures, and
noise is judged by an ensemble-bias statistic instead.

**The tolerance may not be justified by what current tools achieve.** An
earlier version rejected a tighter bar on the grounds that "no current
implementation meets it", which is exactly the reasoning the contract forbids.
The bar is now stated for what it is: **a declared fraction of an ideal
estimator's uncertainty**, with the fraction fixed in advance at 1.0 and
defended on its own terms --- a systematic at or below the noise a user faces
at the reference depth cannot be detected in their data, whoever wrote the
tool. Whether existing tools clear it is a *result*, reported separately, never
an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import norm  # noqa: E402

from benchmarks.harmonic_scale.run_harmonic_scale import ORDERS, PLANTED_HARMONICS  # noqa: E402
from benchmarks.timing.stage1_fixtures import stage1_fixtures  # noqa: E402
from benchmarks.utils.sersic_model import (  # noqa: E402
    compute_bn,
    create_sersic_image_with_harmonics,
    integrated_harmonic_truth,
)

#: Depth the statistical scale is quoted at. Not a survey claim: these mocks
#: have uncorrelated pixel noise and no PSF, so they are not "HSC-like".
#: S/N at R_e is simply the noise parameter the fixtures accept.
REFERENCE_SNR = 100.0

#: The declared fraction of the ideal estimator's uncertainty that a
#: systematic may occupy. Fixed in advance, independent of every tool.
IDEAL_SIGMA_FRACTION = 1.0

#: Realizations the ensemble-bias criterion averages over.
ENSEMBLE_REALIZATIONS = 25

#: Family-wise false-alarm rate for the arm-level bias test.
#:
#: A per-test 3-sigma screen is not enough on its own: across the 20 harmonic
#: tests in one arm an unbiased tool trips at least one 5.3% of the time, and
#: an earlier draft that pooled all six fixtures into 120 tests would have
#: tripped 27.7% of the time --- a figure that draft misreported as "about
#: 10%", which is the number for 40 tests.
#:
#: The decisive test is therefore **Holm-Bonferroni** over the family, not a
#: pooled chi-squared. A sum of squared standardized residuals is chi-squared
#: only if the residual vector is independent standard normal or has been
#: whitened by its covariance, and these residuals are emphatically not
#: independent: components and radii share the same noise image, the same
#: interpolation, the same fitted geometry and overlapping ring samples. With
#: 25 realizations a 20x20 covariance cannot be estimated, let alone inverted.
#: Holm controls the family-wise error rate under *arbitrary* dependence, which
#: is exactly the assumption available here.
ENSEMBLE_FAMILY_ALPHA = 0.01

#: Geometry bars, in the units of the quantity. Half a pixel is the scale below
#: which a centre difference cannot change which pixels a ring samples.
GEOMETRY_BARS = {"center_error_px": 0.5, "eps_error": 0.01, "pa_error_deg": 1.0}

#: Free-fit coverage target, as fractions of R_e, and the fraction of it an arm
#: must reach to count as complete.
TARGET_INTERVAL_R_E = (0.3, 3.0)
MIN_COVERAGE_FRACTION = 0.60

#: Contamination limits. The ceiling is derived from the hardware --- 10 cores,
#: so 2.0 is 20% of capacity committed before the benchmark starts --- because
#: the only load sample available was taken on a busy machine and cannot set
#: the bar it was meant to validate.
CONTAMINATION = {
    "baseline_samples": 30,
    "baseline_interval_s": 10,
    "baseline_median_max": 2.0,
    "in_session_excess_max": 2.0,
    "thermal_signal": "pmset -g therm",
    "max_campaign_retries": 3,
}

#: Seed blocks, disjoint from all four Part A blocks.
SEED_BLOCKS = {"calibration": 100_000, "campaign": 60_260_822}

_COMPONENTS = tuple(f"{prefix}{order}_raw_major" for order in ORDERS for prefix in ("s", "c"))


@lru_cache(maxsize=None)
def _fixture_meta(fixture: str):
    """Render once per fixture. The 1921 px ladder rung is not cheap."""
    spec = stage1_fixtures()[fixture]
    galaxy = spec["galaxy"]
    _, meta = create_sersic_image_with_harmonics(
        n=galaxy["n"],
        R_e=galaxy["R_e"],
        I_e=galaxy["I_e"],
        eps=spec["reference_eps"],
        pa=0.0,
        shape=galaxy["shape"],
        center=galaxy["center"],
        harmonics=PLANTED_HARMONICS,
    )
    return meta


def _component_truth(fixture: str, sma: float) -> Dict[str, float]:
    """Exact raw harmonic amplitudes on this ring, by dense integration.

    ``integrated_harmonic_truth`` keys its result by *order*, with ``s_raw``
    and ``c_raw`` inside; reading it as if it were keyed by component name
    yields nothing and silently produced an infinite bar for every component.
    """
    truth = integrated_harmonic_truth(_fixture_meta(fixture), float(sma), ORDERS)
    out = {}
    for order in ORDERS:
        entry = truth.get(order) or {}
        out[f"s{order}_raw_major"] = float(entry.get("s_raw", float("nan")))
        out[f"c{order}_raw_major"] = float(entry.get("c_raw", float("nan")))
    return out


def _ring_mean(fixture: str, sma: float) -> float:
    """Azimuthally integrated ring mean of the analytic model."""
    truth = integrated_harmonic_truth(_fixture_meta(fixture), float(sma), ORDERS)
    value = float(next(iter(truth.values()))["mean_intensity"])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{fixture} sma={sma:g}: ring mean is {value!r}")
    return value


def ring_statistics(fixture: str) -> List[Dict[str, object]]:
    """Per-ring, per-component analytic noise scale.

    Nothing here depends on a tool: the sample count follows from the ring's
    circumference, sigma from the fixture's own S/N definition, and the truth
    from dense Fourier integration of the analytic model.
    """
    spec = stage1_fixtures()[fixture]
    galaxy = spec["galaxy"]
    sigma = float(galaxy["I_e"]) / REFERENCE_SNR
    b_n = compute_bn(float(galaxy["n"]))

    rows = []
    for sma in spec["radii"]:
        n_samples = max(8, int(2.0 * math.pi * float(sma)))
        intensity = float(galaxy["I_e"]) * math.exp(
            -b_n * ((float(sma) / float(galaxy["R_e"])) ** (1.0 / float(galaxy["n"])) - 1.0)
        )
        amplitude_sigma = sigma * math.sqrt(2.0 / n_samples)
        truth = _component_truth(fixture, float(sma))
        per_component = {}
        for component in _COMPONENTS:
            magnitude = abs(truth.get(component, float("nan")))
            # Fail closed. An earlier version turned a missing, zero or NaN
            # truth into an infinite bar, which is a tolerance every
            # measurement passes -- and it fired for real when a truth lookup
            # was reading the wrong keys, producing an authoritative-looking
            # table of `inf`. A derivation that cannot compute its own bar must
            # stop, not wave everything through.
            if not math.isfinite(magnitude) or magnitude <= 0.0:
                raise ValueError(
                    f"{fixture} sma={sma:g}: truth for {component} is {truth.get(component)!r}; "
                    "cannot derive a bar from it"
                )
            per_component[component] = 100.0 * amplitude_sigma / magnitude
        rows.append(
            {
                "sma": float(sma),
                "n_samples": n_samples,
                "intensity": intensity,
                "amplitude_sigma_pct": per_component,
                # From the integrated ring mean, not the undistorted Sersic
                # value at sma: the planted distortion moves the azimuthal mean
                # by ~0.1% on the outer rings.
                "ring_mean_sigma_pct": 100.0 * sigma / (math.sqrt(n_samples) * _ring_mean(fixture, float(sma))),
            }
        )
    return rows


def thresholds() -> Dict[str, object]:
    """The frozen bars, per fixture, ring and component."""
    per_fixture = {name: ring_statistics(name) for name in sorted(stage1_fixtures())}
    amplitude_bars, intensity_bars = {}, {}
    for fixture, rows in per_fixture.items():
        amplitude_bars[fixture] = {
            row["sma"]: {
                component: round(IDEAL_SIGMA_FRACTION * value, 4)
                for component, value in row["amplitude_sigma_pct"].items()
            }
            for row in rows
        }
        intensity_bars[fixture] = {
            row["sma"]: round(IDEAL_SIGMA_FRACTION * row["ring_mean_sigma_pct"], 4) for row in rows
        }

    # A family is ONE arm: one tool, one fixture, one harmonic setting. An
    # earlier draft summed every fixture together and called 120 "tests per
    # arm", which is six arms' worth.
    rings_per_fixture = {name: len(rows) for name, rows in per_fixture.items()}
    harmonic_tests_per_arm = max(rings_per_fixture.values()) * len(_COMPONENTS)
    intensity_tests_per_arm = max(rings_per_fixture.values())
    return {
        "reference_snr": REFERENCE_SNR,
        "ideal_sigma_fraction": IDEAL_SIGMA_FRACTION,
        "ensemble_realizations": ENSEMBLE_REALIZATIONS,
        "ensemble_family_alpha": ENSEMBLE_FAMILY_ALPHA,
        # Arm-level statistic: the sum of squared standardized mean residuals
        # over every component and ring is chi-squared with that many degrees
        # of freedom under the no-bias hypothesis. One decisive test per arm,
        # so multiplicity is handled by construction rather than by a screen
        # that fires 10% of the time on an unbiased tool.
        "ensemble_family_unit": "one tool x fixture x harmonic-setting arm",
        "ensemble_harmonic_tests_per_arm": harmonic_tests_per_arm,
        "ensemble_intensity_tests_per_arm": intensity_tests_per_arm,
        "ensemble_correction": "holm_bonferroni",
        # Holm's smallest critical level: the most significant of k tests is
        # compared against alpha/k. Quoted so the spec has something concrete
        # to state and the gate something concrete to guard.
        "ensemble_holm_smallest_alpha": round(ENSEMBLE_FAMILY_ALPHA / harmonic_tests_per_arm, 6),
        "ensemble_holm_largest_z": round(float(norm.isf(ENSEMBLE_FAMILY_ALPHA / (2.0 * harmonic_tests_per_arm))), 4),
        "systematic_amplitude_error_pct_by_component": amplitude_bars,
        "systematic_ring_intensity_error_pct_by_ring": intensity_bars,
        "geometry_bars": dict(GEOMETRY_BARS),
        "target_interval_r_e": list(TARGET_INTERVAL_R_E),
        "outer_limit_min_sigma": OUTER_LIMIT_MIN_SIGMA,
        "outer_limit_significance": outer_limit_significance(),
        "min_coverage_fraction": MIN_COVERAGE_FRACTION,
        "contamination": dict(CONTAMINATION),
        "seed_blocks": dict(SEED_BLOCKS),
    }


def amplitude_bar_at(fixture: str, sma: float, component: str) -> float:
    """The systematic bar for a component at an **arbitrary** radius.

    Free-fit arms return radii the frozen table does not contain, and there was
    no rule for judging a ring at, say, 2.37 R_e. The bar is therefore computed
    from the same analytic expressions the table is built from, at whatever
    radius is asked for --- the table is a printed sample of this function, not
    a lookup the contract depends on.

    Raises rather than extrapolating outside the frozen target interval: a bar
    for a radius nobody agreed to score is not a bar.
    """
    spec = stage1_fixtures()[fixture]
    r_e = float(spec["galaxy"]["R_e"])
    low, high = (f * r_e for f in TARGET_INTERVAL_R_E)
    if not (low <= float(sma) <= high):
        raise ValueError(f"{fixture}: sma={sma:g} lies outside the target interval [{low:g}, {high:g}]")

    galaxy = spec["galaxy"]
    sigma = float(galaxy["I_e"]) / REFERENCE_SNR
    n_samples = max(8, int(2.0 * math.pi * float(sma)))
    amplitude_sigma = sigma * math.sqrt(2.0 / n_samples)
    magnitude = abs(_component_truth(fixture, float(sma)).get(component, float("nan")))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError(f"{fixture} sma={sma:g}: no finite truth for {component}")
    return IDEAL_SIGMA_FRACTION * 100.0 * amplitude_sigma / magnitude


def ring_intensity_bar_at(fixture: str, sma: float) -> float:
    """Ring-mean bar at an arbitrary radius, from the **integrated** ring mean.

    The mean is taken from the same dense integration as the harmonics, not
    from the undistorted Sersic value at ``sma``: the planted distortion moves
    the azimuthal mean by ~0.1% on the outer rings, which is small but is a
    difference between the analytic model and a convenient stand-in for it.
    """
    spec = stage1_fixtures()[fixture]
    r_e = float(spec["galaxy"]["R_e"])
    low, high = (f * r_e for f in TARGET_INTERVAL_R_E)
    if not (low <= float(sma) <= high):
        raise ValueError(f"{fixture}: sma={sma:g} lies outside the target interval [{low:g}, {high:g}]")
    sigma = float(spec["galaxy"]["I_e"]) / REFERENCE_SNR
    n_samples = max(8, int(2.0 * math.pi * float(sma)))
    truth = integrated_harmonic_truth(_fixture_meta(fixture), float(sma), ORDERS)
    mean_intensity = float(next(iter(truth.values()))["mean_intensity"])
    if not math.isfinite(mean_intensity) or mean_intensity <= 0.0:
        raise ValueError(f"{fixture} sma={sma:g}: no finite ring mean")
    return IDEAL_SIGMA_FRACTION * 100.0 * sigma / (math.sqrt(n_samples) * mean_intensity)


#: Metric-specific accuracy outcomes. An earlier contract used one
#: ``accuracy_status`` with a ``not_evaluated`` value for harmonics-off arms,
#: and an eligibility rule of ``!= 'fail'`` --- so those arms passed without
#: their *intensity* or *geometry* ever being judged, which they can and must
#: be. Each metric now carries its own status and ``not_applicable`` means only
#: that this metric does not apply to this arm.
ACCURACY_METRICS = ("harmonic_accuracy_status", "intensity_accuracy_status", "geometry_accuracy_status")


def headline_eligible(outcome: Dict[str, object]) -> bool:
    """The single executable eligibility rule.

    The contract used to freeze an eligibility *string*, which the runner would
    then have had to reimplement --- two definitions of one rule, and no way to
    tell when they diverged. The contract now stores this function's name and
    the runner calls it.
    """
    if outcome.get("execution_status") != "ok":
        return False
    if outcome.get("coverage_status") != "complete":
        return False
    if outcome.get("contamination_status") != "clean":
        return False
    for metric in ACCURACY_METRICS:
        status = outcome.get(metric)
        if status not in ("pass", "not_applicable"):
            # Includes None: an unevaluated applicable metric is not a pass.
            return False
    return True


def outer_limit_significance() -> Dict[str, float]:
    """Weakest planted raw component's significance at the outer target radius.

    The upper end of the target interval has a rule --- the faintest component
    must still be measurable there --- and a rule asserted about two fixtures is
    not a rule. This evaluates it for every fixture, so adding one cannot
    quietly move the outer limit past where its harmonics vanish.
    """
    out = {}
    for fixture, spec in stage1_fixtures().items():
        galaxy = spec["galaxy"]
        sma = TARGET_INTERVAL_R_E[1] * float(galaxy["R_e"])
        sigma = float(galaxy["I_e"]) / REFERENCE_SNR
        n_samples = max(8, int(2.0 * math.pi * sma))
        amplitude_sigma = sigma * math.sqrt(2.0 / n_samples)
        weakest = min(abs(v) for v in _component_truth(fixture, sma).values())
        out[fixture] = round(weakest / amplitude_sigma, 3)
    return out


#: Significance the weakest planted component must retain at the outer limit.
OUTER_LIMIT_MIN_SIGMA = 3.0


def stage_1_contract() -> Dict[str, object]:
    """Every decisive Stage 1 field in one object, so all of it can be gated.

    The previous checker guarded four transcribed table rows. Everything else
    --- the realization count, the family-wise alpha, all three geometry bars,
    the target interval, the coverage fraction and the contamination limits ---
    could be edited in the spec without any check firing. A contract is not
    partly frozen.
    """
    contract = {
        "fixtures": {
            name: {
                "n": spec["galaxy"]["n"],
                "R_e": spec["galaxy"]["R_e"],
                "shape": list(spec["galaxy"]["shape"]),
                "eps": spec["reference_eps"],
                "radii": [float(r) for r in spec["radii"]],
                "scope": spec["scope"],
            }
            for name, spec in sorted(stage1_fixtures().items())
        },
        "components": list(_COMPONENTS),
        "reduction_order": ["component", "radius", "seed", "session"],
        # The rule is a function, not a string the runner must reimplement.
        "eligibility_function": "benchmarks.timing.accuracy_thresholds.headline_eligible",
        "accuracy_metrics": list(ACCURACY_METRICS),
        "outcome_fields": [
            "execution_status",
            "coverage_status",
            *ACCURACY_METRICS,
            "contamination_status",
            "headline_eligible",
        ],
        **thresholds(),
    }
    contract["fingerprint"] = hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the whole contract as JSON.")
    args = parser.parse_args()
    contract = stage_1_contract()
    if args.json:
        print(json.dumps(contract, indent=2, default=str))
        return

    print(f"Reference S/N at R_e: {contract['reference_snr']:.0f}")
    print(f"Systematic bar: {contract['ideal_sigma_fraction']}x the ideal estimator's 1-sigma\n")
    for fixture, bars in contract["systematic_amplitude_error_pct_by_component"].items():
        print(f"{fixture}")
        for sma, components in bars.items():
            cells = "  ".join(f"{c.split('_')[0]}={v:.3f}%" for c, v in components.items())
            print(f"   sma={sma:7.1f}  {cells}")
    print("\nring-intensity bars:")
    for fixture, bars in contract["systematic_ring_intensity_error_pct_by_ring"].items():
        cells = ", ".join(f"{sma:g}:{v:.3f}%" for sma, v in bars.items())
        print(f"   {fixture:20s} {cells}")
    print(f"\ngeometry: {contract['geometry_bars']}")
    print(
        f"ensemble bias: chi-squared over {contract['ensemble_tests_per_arm']} tests, "
        f"critical {contract['ensemble_chi2_critical']} at alpha={contract['ensemble_family_alpha']}"
    )
    print(f"fingerprint: {contract['fingerprint'][:16]}")


if __name__ == "__main__":
    main()
