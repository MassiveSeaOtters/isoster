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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import t as student_t  # noqa: E402

from benchmarks.harmonic_scale.run_harmonic_scale import ORDERS  # noqa: E402
from benchmarks.timing.stage1_fixtures import stage1_fixtures  # noqa: E402
from benchmarks.utils.sersic_model import (  # noqa: E402
    analytic_truth_on_aperture,
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

#: Maximum allowed displacement of a returned aperture boundary from the true
#: boundary. This replaces fixed eps/PA tolerances, whose physical displacement
#: grows with radius (eps=0.01 moves a 600-pixel ring by six pixels).
MAX_APERTURE_DISPLACEMENT_PX = 1.0

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
        pa=spec["reference_pa"],
        shape=galaxy["shape"],
        center=galaxy["center"],
        harmonics=spec["reference_harmonics"],
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


def _component_from_truth(truth: Dict[int, Dict[str, float]], component: str) -> float:
    """Read one schema-named raw component from an integrated truth result."""
    prefix = component.split("_", 1)[0]
    if len(prefix) < 2 or prefix[0] not in ("s", "c") or not prefix[1:].isdigit():
        raise ValueError(f"unknown raw harmonic component {component!r}")
    order = int(prefix[1:])
    key = "s_raw" if prefix[0] == "s" else "c_raw"
    try:
        return float(truth[order][key])
    except KeyError as error:
        raise ValueError(f"truth does not contain {component!r}") from error


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


def accuracy_family_members(
    radii: List[float],
    *,
    harmonics_enabled: bool,
    geometry_free: bool,
) -> Dict[str, tuple[str, ...]]:
    """Build the exact Holm families for one tool/fixture/setting arm.

    Harmonic and intensity members are signed residuals at each returned ring.
    Free-fit geometry uses four signed primitive residuals per ring (x, y,
    ellipticity and axis-periodic PA). The separate noiseless geometry gate is
    the maximum boundary displacement in pixels; a non-negative displacement
    is not suitable for a one-sample zero-bias test.
    """
    radius_labels = tuple(f"{float(radius):g}" for radius in radii)
    families: Dict[str, tuple[str, ...]] = {
        "intensity": tuple(f"ring_mean@sma={radius}" for radius in radius_labels),
    }
    if harmonics_enabled:
        families["harmonic"] = tuple(
            f"{component}@sma={radius}" for radius in radius_labels for component in _COMPONENTS
        )
    if geometry_free:
        geometry_components = ("x0", "y0", "eps", "pa_rad")
        families["geometry"] = tuple(
            f"{component}@sma={radius}" for radius in radius_labels for component in geometry_components
        )
    return families


def geometry_bias_residuals(reference: Dict[str, float], measured: Dict[str, float]) -> Dict[str, float]:
    """Signed geometry residuals used by the noisy-arm bias family.

    Position angle describes an axis, so its signed difference is wrapped into
    ``[-pi/2, pi/2)``. The returned keys match the geometry family members.
    """
    pa_difference = (float(measured["pa"]) - float(reference["pa"]) + math.pi / 2.0) % math.pi - math.pi / 2.0
    residuals = {
        "x0": float(measured["x0"]) - float(reference["x0"]),
        "y0": float(measured["y0"]) - float(reference["y0"]),
        "eps": float(measured["eps"]) - float(reference["eps"]),
        "pa_rad": pa_difference,
    }
    if not all(math.isfinite(value) for value in residuals.values()):
        raise ValueError(f"geometry residuals must be finite, got {residuals!r}")
    return residuals


def one_sample_bias_p_value(residuals: List[float]) -> float:
    """Two-sided one-sample t-test of a realization ensemble against zero.

    The scatter is measured from the realizations themselves, so this is a
    Student-t test with ``R - 1`` degrees of freedom, not a normal z-test with
    an assumed known variance. Missing or non-finite samples fail closed.
    """
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("bias test requires at least two finite residuals")
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    statistic = mean / (standard_deviation / math.sqrt(values.size))
    return float(2.0 * student_t.sf(abs(statistic), df=values.size - 1))


def holm_bonferroni(p_values: Dict[str, float], alpha: float = ENSEMBLE_FAMILY_ALPHA) -> Dict[str, object]:
    """Apply Holm's step-down family-wise error correction.

    A rejected null means statistically detectable ensemble bias, so an
    accuracy family passes only when no member is rejected. The ordered audit
    table is returned so Stage 2 can archive every comparison and threshold.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if not p_values:
        raise ValueError("Holm family must contain at least one p-value")
    checked = {}
    for name, value in p_values.items():
        p_value = float(value)
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"p-value for {name!r} must lie in [0, 1], got {value!r}")
        checked[str(name)] = p_value

    ordered = sorted(checked.items(), key=lambda item: (item[1], item[0]))
    rows = []
    continue_rejecting = True
    family_size = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        threshold = alpha / (family_size - index)
        rejected = bool(continue_rejecting and p_value <= threshold)
        if not rejected:
            continue_rejecting = False
        rows.append(
            {
                "name": name,
                "p_value": p_value,
                "threshold": threshold,
                "rejected": rejected,
            }
        )
    return {"family_passed": not any(row["rejected"] for row in rows), "ordered_tests": rows}


def evaluate_bias_family(samples_by_test: Dict[str, List[float]]) -> Dict[str, object]:
    """Compute one p-value per named residual series, then apply Holm."""
    p_values = {name: one_sample_bias_p_value(values) for name, values in samples_by_test.items()}
    return {"p_values": p_values, **holm_bonferroni(p_values)}


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

    # A family is ONE arm: one tool, one fixture, one harmonic setting. Build
    # the partition through the same function Stage 2 must call rather than
    # freezing a prose description of it.
    rings_per_fixture = {name: len(rows) for name, rows in per_fixture.items()}
    representative_radii = [float(index) for index in range(max(rings_per_fixture.values()))]
    all_families = accuracy_family_members(representative_radii, harmonics_enabled=True, geometry_free=True)
    harmonic_tests_per_arm = len(all_families["harmonic"])
    intensity_tests_per_arm = len(all_families["intensity"])
    geometry_tests_per_arm = len(all_families["geometry"])
    family_members_by_fixture = {
        fixture: {
            family: list(members)
            for family, members in accuracy_family_members(
                [float(row["sma"]) for row in rows],
                harmonics_enabled=True,
                geometry_free=True,
            ).items()
        }
        for fixture, rows in per_fixture.items()
    }
    return {
        "reference_snr": REFERENCE_SNR,
        "ideal_sigma_fraction": IDEAL_SIGMA_FRACTION,
        "ensemble_realizations": ENSEMBLE_REALIZATIONS,
        "ensemble_family_alpha": ENSEMBLE_FAMILY_ALPHA,
        "ensemble_family_unit": "one tool x fixture x harmonic-setting arm",
        "ensemble_test": "two_sided_one_sample_t",
        "ensemble_degrees_of_freedom": ENSEMBLE_REALIZATIONS - 1,
        "ensemble_harmonic_tests_per_arm": harmonic_tests_per_arm,
        "ensemble_intensity_tests_per_arm": intensity_tests_per_arm,
        "ensemble_geometry_tests_per_arm": geometry_tests_per_arm,
        "ensemble_family_builder": "benchmarks.timing.accuracy_thresholds.accuracy_family_members",
        "ensemble_family_evaluator": "benchmarks.timing.accuracy_thresholds.evaluate_bias_family",
        "geometry_bias_residual_function": "benchmarks.timing.accuracy_thresholds.geometry_bias_residuals",
        "ensemble_family_applicability": {
            "harmonic": "harmonics_enabled",
            "intensity": "always",
            "geometry": "geometry_free",
        },
        "ensemble_family_members_by_fixture": family_members_by_fixture,
        "ensemble_correction": "holm_bonferroni",
        # Holm's smallest critical level: the most significant of k tests is
        # compared against alpha/k. Quoted so the spec has something concrete
        # to state and the gate something concrete to guard.
        "ensemble_holm_smallest_alpha": round(ENSEMBLE_FAMILY_ALPHA / harmonic_tests_per_arm, 6),
        "systematic_amplitude_error_pct_by_component": amplitude_bars,
        "systematic_ring_intensity_error_pct_by_ring": intensity_bars,
        "systematic_aperture_displacement_error_px": MAX_APERTURE_DISPLACEMENT_PX,
        "descriptive_geometry_metrics": ["center_error_px", "eps_error", "pa_error_deg"],
        "target_interval_r_e": list(TARGET_INTERVAL_R_E),
        "outer_limit_min_sigma": OUTER_LIMIT_MIN_SIGMA,
        "outer_limit_significance": outer_limit_significance(),
        "min_coverage_fraction": MIN_COVERAGE_FRACTION,
        "contamination": dict(CONTAMINATION),
        "seed_blocks": dict(SEED_BLOCKS),
    }


def _validate_scored_radius(fixture: str, sma: float) -> Dict[str, object]:
    spec = stage1_fixtures()[fixture]
    r_e = float(spec["galaxy"]["R_e"])
    low, high = (fraction * r_e for fraction in TARGET_INTERVAL_R_E)
    if not (low <= float(sma) <= high):
        raise ValueError(f"{fixture}: sma={sma:g} lies outside the target interval [{low:g}, {high:g}]")
    return spec


def amplitude_bar_on_aperture(
    fixture: str,
    x0: float,
    y0: float,
    sma: float,
    eps: float,
    pa: float,
    component: str,
) -> float:
    """Systematic harmonic bar on the aperture a free fit returned."""
    spec = _validate_scored_radius(fixture, sma)
    sigma = float(spec["galaxy"]["I_e"]) / REFERENCE_SNR
    n_samples = max(8, int(2.0 * math.pi * float(sma)))
    amplitude_sigma = sigma * math.sqrt(2.0 / n_samples)
    truth = analytic_truth_on_aperture(_fixture_meta(fixture), x0, y0, sma, eps, pa, ORDERS)
    magnitude = abs(_component_from_truth(truth, component))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError(f"{fixture} sma={sma:g}: no finite truth for {component} on returned aperture")
    return IDEAL_SIGMA_FRACTION * 100.0 * amplitude_sigma / magnitude


def amplitude_bar_at(fixture: str, sma: float, component: str) -> float:
    """The systematic bar on the planted aperture at an arbitrary radius.

    Free-fit arms return radii the frozen table does not contain, and there was
    no rule for judging a ring at, say, 2.37 R_e. The bar is therefore computed
    from the same analytic expressions the table is built from, at whatever
    radius is asked for --- the table is a printed sample of this function, not
    a lookup the contract depends on.

    Raises rather than extrapolating outside the frozen target interval: a bar
    for a radius nobody agreed to score is not a bar.
    """
    spec = _validate_scored_radius(fixture, sma)
    return amplitude_bar_on_aperture(
        fixture,
        *spec["galaxy"]["center"],
        sma,
        spec["reference_eps"],
        spec["reference_pa"],
        component,
    )


def ring_intensity_bar_on_aperture(
    fixture: str,
    x0: float,
    y0: float,
    sma: float,
    eps: float,
    pa: float,
) -> float:
    """Ring-mean bar on the aperture a free fit returned."""
    spec = _validate_scored_radius(fixture, sma)
    sigma = float(spec["galaxy"]["I_e"]) / REFERENCE_SNR
    n_samples = max(8, int(2.0 * math.pi * float(sma)))
    truth = analytic_truth_on_aperture(_fixture_meta(fixture), x0, y0, sma, eps, pa, ORDERS)
    mean_intensity = float(next(iter(truth.values()))["mean_intensity"])
    if not math.isfinite(mean_intensity) or mean_intensity <= 0.0:
        raise ValueError(f"{fixture} sma={sma:g}: no finite ring mean on returned aperture")
    return IDEAL_SIGMA_FRACTION * 100.0 * sigma / (math.sqrt(n_samples) * mean_intensity)


def ring_intensity_bar_at(fixture: str, sma: float) -> float:
    """Ring-mean bar at an arbitrary radius, from the **integrated** ring mean.

    The mean is taken from the same dense integration as the harmonics, not
    from the undistorted Sersic value at ``sma``: the planted distortion moves
    the azimuthal mean by ~0.1% on the outer rings, which is small but is a
    difference between the analytic model and a convenient stand-in for it.
    """
    spec = _validate_scored_radius(fixture, sma)
    return ring_intensity_bar_on_aperture(
        fixture,
        *spec["galaxy"]["center"],
        sma,
        spec["reference_eps"],
        spec["reference_pa"],
    )


#: Metric-specific accuracy outcomes. An earlier contract used one
#: ``accuracy_status`` with a ``not_evaluated`` value for harmonics-off arms,
#: and an eligibility rule of ``!= 'fail'`` --- so those arms passed without
#: their *intensity* or *geometry* ever being judged, which they can and must
#: be. Each metric now carries its own status and ``not_applicable`` means only
#: that this metric does not apply to this arm.
ACCURACY_METRICS = ("harmonic_accuracy_status", "intensity_accuracy_status", "geometry_accuracy_status")


def metric_applicability(*, harmonics_enabled: bool, geometry_free: bool) -> Dict[str, bool]:
    """Return which accuracy metrics must be evaluated for this arm."""
    return {
        "harmonic_accuracy_status": bool(harmonics_enabled),
        "intensity_accuracy_status": True,
        "geometry_accuracy_status": bool(geometry_free),
    }


def headline_eligible(
    outcome: Dict[str, object],
    *,
    harmonics_enabled: bool,
    geometry_free: bool,
) -> bool:
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
    for metric, applicable in metric_applicability(
        harmonics_enabled=harmonics_enabled, geometry_free=geometry_free
    ).items():
        status = outcome.get(metric)
        expected = "pass" if applicable else "not_applicable"
        if status != expected:
            # An arm cannot assign itself out of a required comparison, and a
            # non-applicable metric cannot masquerade as additional evidence.
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
        significance = weakest / amplitude_sigma
        if not math.isfinite(significance) or significance < OUTER_LIMIT_MIN_SIGMA:
            raise ValueError(
                f"{fixture}: weakest outer component is {significance:.3g} sigma, "
                f"below the frozen {OUTER_LIMIT_MIN_SIGMA:g}-sigma requirement"
            )
        out[fixture] = round(significance, 3)
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
                "I_e": spec["galaxy"]["I_e"],
                "shape": list(spec["galaxy"]["shape"]),
                "center": list(spec["galaxy"]["center"]),
                "eps": spec["reference_eps"],
                "pa": spec["reference_pa"],
                "harmonics": [
                    {"order": order, "kind": kind, "amplitude": amplitude}
                    for (order, kind), amplitude in sorted(spec["reference_harmonics"].items())
                ],
                "radii": [float(r) for r in spec["radii"]],
                "scope": spec["scope"],
                "scientific_identity": spec["scientific_identity"],
                "independent_scientific_fixture": spec["independent_scientific_fixture"],
            }
            for name, spec in sorted(stage1_fixtures().items())
        },
        "components": list(_COMPONENTS),
        "reduction_order": ["component", "radius", "seed", "session"],
        "aperture_truth_function": "benchmarks.utils.sersic_model.analytic_truth_on_aperture",
        "amplitude_bar_function": "benchmarks.timing.accuracy_thresholds.amplitude_bar_on_aperture",
        "intensity_bar_function": "benchmarks.timing.accuracy_thresholds.ring_intensity_bar_on_aperture",
        "aperture_displacement_function": "benchmarks.utils.sersic_model.aperture_displacement_error_px",
        # The rule is a function, not a string the runner must reimplement.
        "eligibility_function": "benchmarks.timing.accuracy_thresholds.headline_eligible",
        "metric_applicability_function": "benchmarks.timing.accuracy_thresholds.metric_applicability",
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
    print(
        "\ngeometry: maximum aperture-boundary displacement "
        f"{contract['systematic_aperture_displacement_error_px']:.1f} px"
    )
    print(
        "ensemble bias: two-sided one-sample t tests with Holm-Bonferroni at "
        f"alpha={contract['ensemble_family_alpha']}; family sizes "
        f"harmonic={contract['ensemble_harmonic_tests_per_arm']}, "
        f"intensity={contract['ensemble_intensity_tests_per_arm']}, "
        f"geometry={contract['ensemble_geometry_tests_per_arm']}"
    )
    print(f"fingerprint: {contract['fingerprint'][:16]}")


if __name__ == "__main__":
    main()
