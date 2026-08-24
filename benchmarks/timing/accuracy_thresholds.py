"""Part B, Stage 1: derive the accuracy thresholds, executably.

These numbers decide which timings enter a headline ratio, so they must not
live only in prose where they can drift. Everything here is analytic --- no
tool is run and nothing is fitted --- which is the point: a threshold derived
from what a tool achieved is not a threshold.

Several defects in earlier versions are worth stating, because each would have
produced a gate that looked principled and was not.

**A relative metric may be singular even when the measurement is valid.**
These arms report *raw intensity* harmonics, whose truth is not
``planted_fraction * intensity``: it depends on the radial gradient, the Sersic
curvature, products between modes, and each component's own planted amplitude.
An earlier version used the largest planted fraction times the ring intensity
for all four components, giving one bar that was simultaneously too strict for
the large modes and too loose for the small ones --- and since eligibility
takes the worst component, systematically penalising the smallest. Dividing by
truth also becomes undefined when a returned aperture rotates one component
through zero. The decisive gate is now the absolute raw-amplitude residual in
intensity units against the absolute ideal Fourier uncertainty. Dense analytic
integration still supplies the truth; it no longer appears in the denominator
of the tolerance.

**Systematic and statistical accuracy are different regimes.** Gating a single
noisy realization's worst error at a statistical 1-sigma is close to requiring
the impossible: an unbiased ring at sigma = 9.8% lands inside +-1% only 8% of
the time. Systematic accuracy is therefore gated on **noiseless** fixtures, and
noise is judged by ensemble root-mean-square error instead.

**Failure to detect bias is not evidence of accuracy.** A former version used
two-sided one-sample t-tests and passed a family when zero bias was not
rejected. That made high scatter protective: residuals with mean 10 and
standard deviation 100 passed, while a constant residual of 1e-12 failed. The
current statistic measures bias and scatter together against the finite-sample
envelope of the independently defined ideal estimator.

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

from scipy.stats import chi2  # noqa: E402

from benchmarks.harmonic_scale.run_harmonic_scale import ORDERS  # noqa: E402
from benchmarks.timing.stage1_fixtures import (  # noqa: E402
    ENSEMBLE_REALIZATIONS,
    NOISE_ARMS,
    REFERENCE_SNR,
    SEED_BLOCKS,
    stage1_fixtures,
)
from benchmarks.utils.sersic_model import (  # noqa: E402
    compute_bn,
    create_sersic_image_with_harmonics,
    integrated_harmonic_truth,
)

#: Depth the statistical scale is quoted at. Not a survey claim: these mocks
#: have uncorrelated pixel noise and no PSF, so they are not "HSC-like".
#: S/N at R_e is simply the noise parameter the fixtures accept.
#: The declared fraction of the ideal estimator's uncertainty that a
#: systematic may occupy. Fixed in advance, independent of every tool.
IDEAL_SIGMA_FRACTION = 1.0

#: Family-wise false-rejection rate for the noisy-arm accuracy envelope.
#:
#: Each member is judged by its root-mean-square error (RMSE), normalized by
#: the ideal estimator's known noise scale. This measures bias and scatter
#: together. Under the frozen independent-Gaussian mock, R times that squared
#: RMSE follows chi-squared(R) for an ideal estimator. Bonferroni assigns
#: alpha/k to each of k members, so an ideal arm is rejected with probability
#: at most alpha without assuming that radii or components are independent.
ENSEMBLE_FAMILY_ALPHA = 0.01

#: Maximum allowed displacement of a returned aperture boundary from the true
#: boundary. This replaces fixed eps/PA tolerances, whose physical displacement
#: grows with radius (eps=0.01 moves a 600-pixel ring by six pixels).
MAX_APERTURE_DISPLACEMENT_PX = 1.0

#: Free-fit coverage target, as fractions of R_e, and the fraction of it an arm
#: must reach to count as complete.
TARGET_INTERVAL_R_E = (0.3, 3.0)
MIN_COVERAGE_FRACTION = 0.60

#: Natural end-to-end fits use different native grids. Accuracy is nevertheless
#: judged at these five common fractions of R_e after bracketed interpolation.
#: This prevents a tool returning fewer rings from receiving fewer tests.
END_TO_END_EVALUATION_RADIUS_FRACTIONS = (0.3, 0.5, 1.0, 1.8, 3.0)

#: Contamination limits. The ceiling is derived from the hardware at 0.2 times
#: the 20 logical CPUs. Load average is only a contamination proxy, not literal
#: CPU utilization; the idle baseline still has to pass before timing starts.
CONTAMINATION = {
    "baseline_samples": 30,
    "baseline_interval_s": 10,
    "baseline_median_max": 4.0,
    "in_session_excess_max": 2.0,
    "in_session_consecutive_load_samples": 2,
    "thermal_signal": "pmset -g therm",
    "max_campaign_retries": 3,
}

#: The contamination rule above is intentionally host-specific. A run on a
#: different machine needs a committed Stage 1 amendment, not a silent reuse of
#: a ten-core load ceiling or a macOS-only thermal command.
BENCHMARK_HOST = {
    "system": "Darwin",
    "machine": "arm64",
    "machine_model": "Mac13,2",
    "logical_cpu_count": 20,
    "thermal_command": "/usr/bin/pmset -g therm",
}

SCIENTIFIC_INPUT = {
    "renderer_function": "benchmarks.utils.sersic_model.create_sersic_image_with_harmonics",
    "pixel_sampling": "pixel_centres_without_subpixel_integration",
    "psf": "none",
    "background": 0.0,
    "mask": "none",
    "variance_map": "constant_noise_variance_for_gaussian_reference; none_for_noiseless",
    "noise_arms": NOISE_ARMS,
    "seed_derivation": "seed_blocks[stage] + realization_index",
    "harmonic_basis": "physical_polar_angle_from_major_axis",
    "harmonic_conversion_requirement": "every_source_ring_observed_valid",
    "tool_harmonic_settings": {
        "isoster": {"use_eccentric_anomaly": False},
        "photutils": {"basis": "native_physical_polar_angle"},
        "autoprof": {
            "ap_isoclip": True,
            "ap_iso_interpolate_start": 1000.0,
            "ap_isoband_fixed": True,
            "ap_isoband_width": 0.1,
            "required_observed_sampling_mode": "line_interpolated",
        },
    },
}

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
    """Per-ring ideal noise scales plus descriptive relative forms.

    Nothing here depends on a tool: the sample count follows from the ring's
    circumference and sigma from the fixture's own S/N definition. Dense truth
    enters only the percentage diagnostics retained for continuity; the
    decisive limits are absolute.
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
                "harmonic_absolute_sigma": amplitude_sigma,
                "amplitude_sigma_pct": per_component,
                # From the integrated ring mean, not the undistorted Sersic
                # value at sma: the planted distortion moves the azimuthal mean
                # by ~0.1% on the outer rings.
                "ring_mean_absolute_sigma": sigma / math.sqrt(n_samples),
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
    """Build the exact accuracy families for one tool/fixture/setting arm.

    Harmonic and intensity members are signed residuals at each fixed evaluation
    ring. Free-fit geometry is the boundary-displacement RMSE at each ring. The
    latter is deliberately one physical metric rather than four parameter
    residuals whose meaning changes with radius.
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
        families["geometry"] = tuple(f"aperture_displacement@sma={radius}" for radius in radius_labels)
    return families


def familywise_standardized_rmse_limit(
    family_size: int,
    *,
    realizations: int = ENSEMBLE_REALIZATIONS,
    alpha: float = ENSEMBLE_FAMILY_ALPHA,
) -> float:
    """Finite-sample RMSE envelope for an ideal Gaussian estimator.

    For one member with independent N(0, 1) residuals, ``R * RMSE**2`` is
    chi-squared with R degrees of freedom. Bonferroni uses ``alpha / k`` for
    each of k members, controlling the whole arm under arbitrary dependence
    between members.
    """
    if not isinstance(family_size, int) or family_size <= 0:
        raise ValueError(f"family_size must be a positive integer, got {family_size!r}")
    if not isinstance(realizations, int) or realizations <= 0:
        raise ValueError(f"realizations must be a positive integer, got {realizations!r}")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    quantile = float(chi2.ppf(1.0 - alpha / family_size, df=realizations))
    return math.sqrt(quantile / realizations)


def evaluate_accuracy_family(
    residuals_by_test: Dict[str, List[float]],
    ideal_sigma_by_test: Dict[str, float],
) -> Dict[str, object]:
    """Judge noisy harmonic or intensity accuracy by normalized RMSE.

    This is intentionally not a test of whether bias is distinguishable from
    zero. Failure to detect bias rewards imprecision. RMSE combines bias and
    scatter, and the ideal-Gaussian envelope provides the finite-sample
    allowance fixed before any tool is run.
    """
    if not residuals_by_test:
        raise ValueError("accuracy family must contain at least one residual series")
    if set(residuals_by_test) != set(ideal_sigma_by_test):
        raise ValueError("residual and ideal-scale families must contain the same members")
    limit = familywise_standardized_rmse_limit(len(residuals_by_test))
    rows = []
    for name in sorted(residuals_by_test):
        values = np.asarray(residuals_by_test[name], dtype=np.float64)
        if values.ndim != 1 or values.size != ENSEMBLE_REALIZATIONS or not np.all(np.isfinite(values)):
            raise ValueError(f"{name}: accuracy family requires exactly {ENSEMBLE_REALIZATIONS} finite residuals")
        scale = float(ideal_sigma_by_test[name])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name}: ideal scale must be finite and positive, got {scale!r}")
        normalized_rmse = float(np.sqrt(np.mean(np.square(values / scale))))
        rows.append(
            {
                "name": name,
                "normalized_rmse": normalized_rmse,
                "limit": limit,
                "passed": normalized_rmse <= limit,
            }
        )
    return {"family_passed": all(row["passed"] for row in rows), "member_results": rows}


def evaluate_geometry_accuracy_family(displacements_by_radius: Dict[str, List[float]]) -> Dict[str, object]:
    """Require the 25-realization RMS boundary displacement to stay within 1 px."""
    if not displacements_by_radius:
        raise ValueError("geometry accuracy family must contain at least one radius")
    rows = []
    for name in sorted(displacements_by_radius):
        values = np.asarray(displacements_by_radius[name], dtype=np.float64)
        if (
            values.ndim != 1
            or values.size != ENSEMBLE_REALIZATIONS
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError(
                f"{name}: geometry family requires exactly {ENSEMBLE_REALIZATIONS} finite non-negative displacements"
            )
        rms_displacement = float(np.sqrt(np.mean(np.square(values))))
        rows.append(
            {
                "name": name,
                "rms_displacement_px": rms_displacement,
                "limit_px": MAX_APERTURE_DISPLACEMENT_PX,
                "passed": rms_displacement <= MAX_APERTURE_DISPLACEMENT_PX,
            }
        )
    return {"family_passed": all(row["passed"] for row in rows), "member_results": rows}


def evaluate_systematic_accuracy_family(
    residual_by_test: Dict[str, float],
    absolute_limit_by_test: Dict[str, float],
) -> Dict[str, object]:
    """Apply the noiseless absolute-error limits without reimplementing them."""
    if not residual_by_test or set(residual_by_test) != set(absolute_limit_by_test):
        raise ValueError("systematic residual and limit families must contain the same non-empty members")
    rows = []
    for name in sorted(residual_by_test):
        residual = float(residual_by_test[name])
        limit = float(absolute_limit_by_test[name])
        if not math.isfinite(residual):
            raise ValueError(f"{name}: systematic residual must be finite, got {residual!r}")
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError(f"{name}: systematic limit must be finite and positive, got {limit!r}")
        absolute_error = abs(residual)
        rows.append({"name": name, "absolute_error": absolute_error, "limit": limit, "passed": absolute_error <= limit})
    return {"family_passed": all(row["passed"] for row in rows), "member_results": rows}


def thresholds() -> Dict[str, object]:
    """The frozen bars, per fixture, ring and component."""
    per_fixture = {name: ring_statistics(name) for name in sorted(stage1_fixtures())}
    harmonic_absolute_limits, intensity_absolute_limits = {}, {}
    descriptive_amplitude_percent, descriptive_intensity_percent = {}, {}
    for fixture, rows in per_fixture.items():
        harmonic_absolute_limits[fixture] = {
            row["sma"]: round(IDEAL_SIGMA_FRACTION * row["harmonic_absolute_sigma"], 8) for row in rows
        }
        intensity_absolute_limits[fixture] = {
            row["sma"]: round(IDEAL_SIGMA_FRACTION * row["ring_mean_absolute_sigma"], 8) for row in rows
        }
        descriptive_amplitude_percent[fixture] = {
            row["sma"]: {
                component: round(IDEAL_SIGMA_FRACTION * value, 4)
                for component, value in row["amplitude_sigma_pct"].items()
            }
            for row in rows
        }
        descriptive_intensity_percent[fixture] = {
            row["sma"]: round(IDEAL_SIGMA_FRACTION * row["ring_mean_sigma_pct"], 4) for row in rows
        }

    # A family is ONE arm: one tool, one fixture, one harmonic setting. Build
    # the partition through the same function Stage 2 must call rather than
    # freezing a prose description of it.
    rings_per_fixture = {name: len(rows) for name, rows in per_fixture.items()}
    representative_radii = [float(index + 1) for index in range(max(rings_per_fixture.values()))]
    all_families = accuracy_family_members(representative_radii, harmonics_enabled=True, geometry_free=True)
    harmonic_tests_per_arm = len(all_families["harmonic"])
    intensity_tests_per_arm = len(all_families["intensity"])
    geometry_tests_per_arm = len(all_families["geometry"])
    family_members_by_fixture_and_scope = {
        fixture: {
            "fixed_aperture": {
                family: list(members)
                for family, members in accuracy_family_members(
                    [float(row["sma"]) for row in rows], harmonics_enabled=True, geometry_free=False
                ).items()
            },
            "end_to_end": {
                family: list(members)
                for family, members in accuracy_family_members(
                    [
                        float(fraction) * float(stage1_fixtures()[fixture]["galaxy"]["R_e"])
                        for fraction in END_TO_END_EVALUATION_RADIUS_FRACTIONS
                    ],
                    harmonics_enabled=True,
                    geometry_free=True,
                ).items()
            },
        }
        for fixture, rows in per_fixture.items()
    }
    family_limits = {
        "harmonic": round(familywise_standardized_rmse_limit(harmonic_tests_per_arm), 8),
        "intensity": round(familywise_standardized_rmse_limit(intensity_tests_per_arm), 8),
    }
    return {
        "reference_snr": REFERENCE_SNR,
        "ideal_sigma_fraction": IDEAL_SIGMA_FRACTION,
        "ensemble_realizations": ENSEMBLE_REALIZATIONS,
        "ensemble_family_alpha": ENSEMBLE_FAMILY_ALPHA,
        "ensemble_family_unit": "one tool x fixture x harmonic-setting arm",
        "ensemble_accuracy_statistic": "standardized_root_mean_square_error",
        "ensemble_reference_distribution": "independent_standard_normal_per_realization",
        "ensemble_harmonic_tests_per_arm": harmonic_tests_per_arm,
        "ensemble_intensity_tests_per_arm": intensity_tests_per_arm,
        "ensemble_geometry_tests_per_arm": geometry_tests_per_arm,
        "ensemble_family_builder": "benchmarks.timing.accuracy_thresholds.accuracy_family_members",
        "ensemble_family_evaluator": "benchmarks.timing.accuracy_thresholds.evaluate_accuracy_family",
        "geometry_family_evaluator": "benchmarks.timing.accuracy_thresholds.evaluate_geometry_accuracy_family",
        "systematic_family_evaluator": ("benchmarks.timing.accuracy_thresholds.evaluate_systematic_accuracy_family"),
        "ensemble_family_applicability": {
            "harmonic": "harmonics_enabled",
            "intensity": "always",
            "geometry": "geometry_free",
        },
        "ensemble_family_members_by_fixture_and_scope": family_members_by_fixture_and_scope,
        "ensemble_correction": "bonferroni_familywise_rmse_envelope",
        "ensemble_member_alpha_by_family": {
            "harmonic": round(ENSEMBLE_FAMILY_ALPHA / harmonic_tests_per_arm, 6),
            "intensity": round(ENSEMBLE_FAMILY_ALPHA / intensity_tests_per_arm, 6),
        },
        "ensemble_standardized_rmse_limit_by_family": family_limits,
        "systematic_harmonic_absolute_error_by_ring": harmonic_absolute_limits,
        "systematic_ring_intensity_absolute_error_by_ring": intensity_absolute_limits,
        "descriptive_harmonic_error_pct_by_component": descriptive_amplitude_percent,
        "descriptive_ring_intensity_error_pct_by_ring": descriptive_intensity_percent,
        "systematic_aperture_displacement_error_px": MAX_APERTURE_DISPLACEMENT_PX,
        "descriptive_geometry_metrics": ["center_error_px", "eps_error", "pa_error_deg"],
        "target_interval_r_e": list(TARGET_INTERVAL_R_E),
        "end_to_end_evaluation_radius_fractions": list(END_TO_END_EVALUATION_RADIUS_FRACTIONS),
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


def harmonic_absolute_error_limit_on_aperture(fixture: str, sma: float) -> float:
    """Absolute raw-amplitude limit at a returned semi-major axis.

    The ideal Fourier uncertainty depends on noise and sample count, not on the
    component's true amplitude. Keeping this in intensity units avoids the
    singular percentage metric when a valid returned aperture rotates one
    component through zero.
    """
    spec = _validate_scored_radius(fixture, sma)
    sigma = float(spec["galaxy"]["I_e"]) / REFERENCE_SNR
    n_samples = max(8, int(2.0 * math.pi * float(sma)))
    return IDEAL_SIGMA_FRACTION * sigma * math.sqrt(2.0 / n_samples)


def harmonic_absolute_error_limit_at(fixture: str, sma: float) -> float:
    """Absolute raw-amplitude limit at an arbitrary scored radius."""
    return harmonic_absolute_error_limit_on_aperture(fixture, sma)


def ring_intensity_absolute_error_limit_on_aperture(fixture: str, sma: float) -> float:
    """Absolute ring-mean intensity limit at a returned semi-major axis."""
    spec = _validate_scored_radius(fixture, sma)
    sigma = float(spec["galaxy"]["I_e"]) / REFERENCE_SNR
    n_samples = max(8, int(2.0 * math.pi * float(sma)))
    return IDEAL_SIGMA_FRACTION * sigma / math.sqrt(n_samples)


def ring_intensity_absolute_error_limit_at(fixture: str, sma: float) -> float:
    """Absolute ring-mean intensity limit at an arbitrary scored radius."""
    return ring_intensity_absolute_error_limit_on_aperture(fixture, sma)


def ideal_sigma_by_family_member(fixture: str, members: List[str]) -> Dict[str, float]:
    """Build the noisy-family scales from the same names the family builder emits."""
    scales = {}
    for member in members:
        try:
            metric, radius_text = member.rsplit("@sma=", 1)
            radius = float(radius_text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid accuracy-family member {member!r}") from error
        if metric == "ring_mean":
            scales[member] = ring_intensity_absolute_error_limit_at(fixture, radius) / IDEAL_SIGMA_FRACTION
        elif metric in _COMPONENTS:
            scales[member] = harmonic_absolute_error_limit_at(fixture, radius) / IDEAL_SIGMA_FRACTION
        else:
            raise ValueError(f"{member!r} has no ideal harmonic/intensity scale")
    return scales


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
                "fixed_aperture_radii": [float(r) for r in spec["radii"]],
                "end_to_end_evaluation_radii": [
                    float(fraction) * float(spec["galaxy"]["R_e"])
                    for fraction in END_TO_END_EVALUATION_RADIUS_FRACTIONS
                ],
                "scope": spec["scope"],
                "scientific_identity": spec["scientific_identity"],
                "independent_scientific_fixture": spec["independent_scientific_fixture"],
            }
            for name, spec in sorted(stage1_fixtures().items())
        },
        "components": list(_COMPONENTS),
        "reduction_order": ["component", "radius", "seed", "session"],
        "scientific_input": SCIENTIFIC_INPUT,
        "benchmark_host": BENCHMARK_HOST,
        "benchmark_host_validation_function": "benchmarks.timing.accuracy_thresholds.benchmark_host_mismatches",
        "aperture_truth_function": "benchmarks.utils.sersic_model.analytic_truth_on_aperture",
        "harmonic_absolute_error_limit_function": (
            "benchmarks.timing.accuracy_thresholds.harmonic_absolute_error_limit_on_aperture"
        ),
        "intensity_absolute_error_limit_function": (
            "benchmarks.timing.accuracy_thresholds.ring_intensity_absolute_error_limit_on_aperture"
        ),
        "ensemble_ideal_scale_function": "benchmarks.timing.accuracy_thresholds.ideal_sigma_by_family_member",
        "end_to_end_profile_evaluation_function": (
            "benchmarks.timing.profile_evaluation.interpolate_profile_to_evaluation_radii"
        ),
        "pa_harmonic_canonicalization_function": ("benchmarks.timing.profile_evaluation.canonicalize_pa_and_harmonics"),
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


def benchmark_host_mismatches(observed: Dict[str, object]) -> List[str]:
    """Return every host field that differs from the frozen benchmark host."""
    mismatches = []
    for name, expected in BENCHMARK_HOST.items():
        actual = observed.get(name)
        if actual != expected:
            mismatches.append(f"{name}: expected {expected!r}, observed {actual!r}")
    for name in sorted(set(observed) - set(BENCHMARK_HOST)):
        mismatches.append(f"{name}: unexpected host field {observed[name]!r}")
    return mismatches


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
    for fixture, bars in contract["systematic_harmonic_absolute_error_by_ring"].items():
        print(f"{fixture}")
        for sma, value in bars.items():
            print(f"   sma={sma:7.1f}  |raw residual| <= {value:.6g}")
    print("\nring-intensity absolute limits:")
    for fixture, bars in contract["systematic_ring_intensity_absolute_error_by_ring"].items():
        cells = ", ".join(f"{sma:g}:{v:.6g}" for sma, v in bars.items())
        print(f"   {fixture:20s} {cells}")
    print(
        "\ngeometry: maximum aperture-boundary displacement "
        f"{contract['systematic_aperture_displacement_error_px']:.1f} px"
    )
    print(
        "ensemble accuracy: family-wise normalized RMSE against the ideal-Gaussian envelope at "
        f"alpha={contract['ensemble_family_alpha']}; family sizes "
        f"harmonic={contract['ensemble_harmonic_tests_per_arm']}, "
        f"intensity={contract['ensemble_intensity_tests_per_arm']}, "
        f"geometry={contract['ensemble_geometry_tests_per_arm']}"
    )
    print(f"fingerprint: {contract['fingerprint'][:16]}")


if __name__ == "__main__":
    main()
