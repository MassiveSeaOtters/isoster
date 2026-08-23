"""A4 Track 2: can AutoProf's Bender coefficients be reconstructed at all?

Track 1 recovers raw sine and cosine amplitudes exactly for all three tools,
and needs no gradient. Track 2 is the dimensionless quantity a reader actually
plots --- ``a_n = S_n / (sma * |dI/da|)`` --- and AutoProf reports no radial
gradient, so one has to be built from a profile it does report.

The design decision, and why it is not the obvious one
------------------------------------------------------
Every bit of Track 2's error lives in the denominator, since Track 1 already
gives the numerator exactly. The obvious move is to finite-difference
AutoProf's ``b0`` into the most *accurate* gradient available. That is wrong.

isoster and photutils do not divide by ``dI/da``. They divide by a **forward
secant** over ``sma -> sma*(1 + astep)`` with ``astep = 0.1``, which on the
Part A fixture sits 11-14% below the point derivative at every radius. A
point-derivative denominator would therefore have put AutoProf's ``a_n`` 12%
away from isoster's *by construction*, and this campaign would have reported a
definition mismatch as a disagreement between tools.

So the denominator is a matched secant: AutoProf's ``b0`` at ``sma`` and at
``sma*(1 + astep)``, differenced over the same interval. The **interval** is
shared, because it must be or the comparison is meaningless. The **value** is
measured from AutoProf's own ``b0`` and nothing else --- no isoster quantity
enters the reconstruction.

Cost: every measurement radius needs its comparison radius measured too, so
the ring count doubles. The realized pairing is archived.

Acceptance, fixed before this ran
---------------------------------
As pre-registered, Track 2 was to be "licensed" on two criteria. **Criterion 2
was withdrawn on 2026-08-23 after review**: it asked that the Bender agreement
be no worse than the raw agreement plus the gradient error, but Bender *is* raw
divided by gradient, so it compared a quantity against the two it is built
from. That is an arithmetic identity check, not evidence.

The word "licensed" went with it. It merged three separate things into one
boolean, and the three are now reported separately:

1. ``conversion_method_validated`` --- criterion 1, unchanged. Does the matched
   secant beat a point derivative against the same target, decisively? This is
   an accuracy comparison and is campaign-level empirical support for the
   *method*, not a property of any ring.
2. ``harmonic_conversion_valid`` --- per **ring pair**, from **realized**
   provenance: the polar angular basis, both rings sampled by interpolation,
   both measured. Structural, and observed rather than inferred from what was
   requested.
3. per-regime accuracy, reported and gating nothing.

Usage::

    uv run python benchmarks/harmonic_scale/run_gradient_reconstruction.py --fixture sersic_n2_compact --mode pilot
    uv run python benchmarks/harmonic_scale/run_gradient_reconstruction.py --fixture sersic_n2_compact --freeze-tolerances <pilot.json>
    uv run python benchmarks/harmonic_scale/run_gradient_reconstruction.py --fixture sersic_n2_compact --mode validation --archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harmonic_scale.adapters import (  # noqa: E402
    measure_autoprof_fixed,
    measure_isoster_fixed,
    measure_photutils_fixed,
)
from benchmarks.harmonic_scale.conventions import (  # noqa: E402
    DEFAULT_GRADIENT_STEP,
    bender_from_raw,
    comparison_radius,
    matched_secant_gradient,
)
from benchmarks.harmonic_scale.run_harmonic_scale import (  # noqa: E402
    DETERMINISTIC_FLOOR_PCT,
    FIXTURES,
    NOISE_REALIZATIONS,
    ORDERS,
    PLANTED_HARMONICS,
    TOLERANCE_SAFETY_FACTOR,
    _environment,
    _git_state,
)
from benchmarks.utils.sersic_model import (  # noqa: E402
    add_noise,
    create_sersic_image_with_harmonics,
    integrated_harmonic_truth,
)
from isoster.output_paths import resolve_output_directory  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Seed blocks disjoint from all four the A3 campaigns already use
#: (900000 / 20260822 and 700000 / 30260822), and from each other.
SEED_BLOCKS = {
    "sersic_n2_compact": {"pilot": 500_000, "validation": 40_260_822},
    "sersic_n4_extended": {"pilot": 300_000, "validation": 50_260_822},
}

ACTIVE_FIXTURE = "sersic_n2_compact"

#: The interval, shared with isoster by necessity. Recorded per run, because a
#: campaign that does not say which interval it used has not said what it
#: measured.
GRADIENT_STEP = DEFAULT_GRADIENT_STEP


def _spec() -> Dict[str, object]:
    return FIXTURES[ACTIVE_FIXTURE]


def archive_path() -> Path:
    return HERE / f"reference_gradient_reconstruction_{ACTIVE_FIXTURE}.json"


def tolerances_path() -> Path:
    return HERE / f"frozen_tolerances_gradient_{ACTIVE_FIXTURE}.json"


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def _case(name, kind, why, **overrides) -> Dict[str, object]:
    spec = {
        "name": name,
        "kind": kind,
        "why": why,
        "eps": _spec()["reference_eps"],
        "pa_deg": 0.0,
        "isoclip": True,
        "interpolate_start": 100.0,
        "background_offset": 0.0,
        "snr": None,
        "n_realizations": 1,
    }
    unknown = set(overrides) - set(spec)
    if unknown:
        raise ValueError(f"case {name!r} overrides unknown axes: {sorted(unknown)}")
    spec.update(overrides)
    if spec["snr"] is not None and "n_realizations" not in overrides:
        spec["n_realizations"] = NOISE_REALIZATIONS
    return spec


def build_grid() -> List[Dict[str, object]]:
    """Smaller than A3's, because the measurand is narrower.

    Only axes that can plausibly move a *gradient* are here. The angle basis
    is absent because the reconstruction is scoped to the polar-resampled path
    and nothing else; running the eccentric-anomaly arm would produce numbers
    for a conversion the design already refuses.
    """
    return [
        _case(
            "reference",
            "reference",
            "Noiseless, moderate ellipticity, interpolation everywhere. The configuration "
            "the acceptance criterion is evaluated on.",
        ),
        _case(
            "eps_circular",
            "one_factor",
            "Near-circular rings, where sampling differences between the tools are smallest.",
            eps=0.02,
        ),
        _case(
            "eps_high",
            "one_factor",
            "Strong ellipticity. The ring statistic each tool forms differs most here, and "
            "the secant is a difference of two such statistics.",
            eps=0.6,
        ),
        _case(
            "pa_30",
            "one_factor",
            "A gradient is a scalar and should not care about orientation. A null result "
            "that would expose a sampling asymmetry if it were not one.",
            pa_deg=30.0,
        ),
        _case(
            "interpolate_default",
            "one_factor",
            "AutoProf's default sampling. Nearest-pixel rounding perturbs b0 itself, so it "
            "perturbs a difference of two b0 values -- possibly more than it perturbs either.",
            interpolate_start=5.0,
        ),
        _case(
            "background_offset",
            "one_factor",
            "A constant cancels exactly in a difference, so every reconstructed gradient must "
            "be unchanged. The sharpest null test on this grid.",
            background_offset=50.0,
        ),
        _case(
            "noise_snr100",
            "one_factor",
            "A secant differences two noisy ring means over a short baseline, which amplifies "
            "noise relative to either mean. This is where the reconstruction is expected to hurt.",
            snr=100.0,
        ),
        _case(
            "noise_snr30",
            "one_factor",
            "Harder noise, and the case that decides whether the reconstruction survives "
            "realistic data or only clean fixtures.",
            snr=30.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def _paired_rings(spec: Dict[str, object]) -> tuple[List[dict], List[float]]:
    """Measurement radii plus the comparison radius each one needs."""
    x0, y0 = _spec()["galaxy"]["center"]
    base = [float(r) for r in _spec()["radii"]]
    radii = base + [comparison_radius(r, GRADIENT_STEP) for r in base]
    request = [
        {
            "sma": float(r),
            "x0": float(x0),
            "y0": float(y0),
            "eps": float(spec["eps"]),
            "pa": float(np.radians(float(spec["pa_deg"]))),
        }
        for r in radii
    ]
    return request, base


def _by_sma(rows: Sequence[dict]) -> Dict[float, dict]:
    """Key rows by radius. Never by position --- the tools do not agree on order.

    isoster returns its isophotes sorted by ``sma`` while the request may not
    be, so positional zipping silently pairs a ring with a different ring's
    comparison. That mistake produced a plausible-looking 60% disagreement
    before it was caught.
    """
    keyed = {round(float(r["sma"]), 6): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError("two rows share a semi-major axis; cannot key by radius")
    return keyed


def _build_image(spec: Dict[str, object], seed: int | None):
    galaxy = _spec()["galaxy"]
    image, meta = create_sersic_image_with_harmonics(
        n=galaxy["n"],
        R_e=galaxy["R_e"],
        I_e=galaxy["I_e"],
        eps=float(spec["eps"]),
        pa=np.radians(float(spec["pa_deg"])),
        shape=galaxy["shape"],
        center=galaxy["center"],
        harmonics=PLANTED_HARMONICS,
    )
    noise_sigma = 0.0
    if spec["snr"] is not None:
        image, noise_sigma = add_noise(
            image,
            snr_at_Re=float(spec["snr"]),
            R_e=galaxy["R_e"],
            I_e=galaxy["I_e"],
            seed=seed,
        )
    return image + float(spec["background_offset"]), meta, float(noise_sigma)


def _relative(measured: float, reference: float) -> float:
    """``|measured/reference - 1|`` in percent, or NaN where undefined."""
    if not np.isfinite(measured) or not np.isfinite(reference) or reference == 0:
        return float("nan")
    return abs(measured / reference - 1.0) * 100.0


def measure_case(spec: Dict[str, object], seed_block: int, workspace_root: Path) -> Dict[str, object]:
    request, base_radii = _paired_rings(spec)
    realizations = []
    provenance = None

    for index in range(int(spec["n_realizations"])):
        seed = None if spec["snr"] is None else seed_block + index
        image, meta, noise_sigma = _build_image(spec, seed)

        isoster = _by_sma(measure_isoster_fixed(image, request, ORDERS))
        photutils = _by_sma(measure_photutils_fixed(image, request, ORDERS))
        with tempfile.TemporaryDirectory(dir=str(workspace_root)) as work:
            autoprof_rows, provenance = measure_autoprof_fixed(
                image,
                request,
                orders=ORDERS,
                workspace=work,
                isoclip=bool(spec["isoclip"]),
                interpolate_start=float(spec["interpolate_start"]),
            )
        autoprof = _by_sma(autoprof_rows)

        rings = []
        for sma in base_radii:
            near, far = round(sma, 6), round(comparison_radius(sma, GRADIENT_STEP), 6)
            truth = integrated_harmonic_truth(meta, sma, ORDERS)

            # Reconstructed from AutoProf's own profile, two ways: the mean of
            # the vector that entered the FFT, and the median column that is
            # the wrong estimator for this purpose.
            b0_secant = matched_secant_gradient(
                autoprof[near]["autoprof_b0"], autoprof[far]["autoprof_b0"], sma, GRADIENT_STEP
            )
            median_secant = matched_secant_gradient(
                autoprof[near]["autoprof_median_flux"],
                autoprof[far]["autoprof_median_flux"],
                sma,
                GRADIENT_STEP,
            )
            isoster_gradient = float(isoster[near]["gradient"])
            point_derivative = float(truth[ORDERS[0]]["gradient"])

            entry = {
                "sma": sma,
                "comparison_sma": comparison_radius(sma, GRADIENT_STEP),
                "autoprof_b0_secant": b0_secant,
                "autoprof_median_secant": median_secant,
                "isoster_gradient": isoster_gradient,
                "photutils_gradient": float(photutils[near]["gradient"]),
                "analytic_point_derivative": point_derivative,
                # The three comparisons the pre-registration promised.
                "b0_secant_vs_isoster_pct": _relative(b0_secant, isoster_gradient),
                "median_secant_vs_isoster_pct": _relative(median_secant, isoster_gradient),
                "isoster_vs_point_derivative_pct": _relative(isoster_gradient, point_derivative),
                "b0_secant_vs_point_derivative_pct": _relative(b0_secant, point_derivative),
                "status": autoprof[near]["status"],
                "sampling_mode": autoprof[near].get("harmonic_sampling_mode"),
            }

            for order in ORDERS:
                # Track 1: raw amplitudes, which need no gradient.
                for component in ("s", "c"):
                    key = f"{component}{order}_raw"
                    truth_key = f"{component}_raw"
                    entry[f"{key}_autoprof_vs_truth_pct"] = _relative(autoprof[near][key], truth[order][truth_key])
                    entry[f"{key}_isoster_vs_truth_pct"] = _relative(isoster[near][key], truth[order][truth_key])
                    # The baseline criterion 2 actually needs. A raw-versus-
                    # *truth* number and a Bender-versus-*isoster* number are
                    # not comparable: under noise both tools see the same
                    # realization, so their errors correlate and the
                    # tool-to-tool gap is far smaller than either gap to
                    # truth. The pilot made that concrete -- at S/N = 30 it
                    # read raw 34.5% against Bender 9.9%, which would have
                    # looked like normalizing *improved* matters. Same pair of
                    # tools, same realization, only the normalization
                    # differing, is the only well-posed form.
                    entry[f"{key}_autoprof_vs_isoster_pct"] = _relative(autoprof[near][key], isoster[near][key])
                # Track 2: the same raw amplitudes, normalized by the
                # reconstructed gradient, against isoster's own Bender values.
                a_ap, b_ap = bender_from_raw(
                    autoprof[near][f"s{order}_raw"],
                    autoprof[near][f"c{order}_raw"],
                    sma,
                    b0_secant,
                )
                entry[f"a{order}_bender_autoprof"] = a_ap
                entry[f"b{order}_bender_autoprof"] = b_ap
                entry[f"a{order}_bender_isoster"] = float(isoster[near][f"a{order}_bender"])
                entry[f"b{order}_bender_isoster"] = float(isoster[near][f"b{order}_bender"])
                entry[f"a{order}_bender_vs_isoster_pct"] = _relative(a_ap, float(isoster[near][f"a{order}_bender"]))
                entry[f"b{order}_bender_vs_isoster_pct"] = _relative(b_ap, float(isoster[near][f"b{order}_bender"]))
            rings.append(entry)

        realizations.append({"seed": seed, "noise_sigma": noise_sigma, "rings": rings})

    return {
        "spec": spec,
        "autoprof_provenance": provenance,
        "realizations": realizations,
        "summary": _summarize(realizations, base_radii),
    }


_SUMMARY_KEYS = (
    "b0_secant_vs_isoster_pct",
    "median_secant_vs_isoster_pct",
    "isoster_vs_point_derivative_pct",
    "b0_secant_vs_point_derivative_pct",
)


def _spread(values: Sequence[float]) -> Dict[str, object] | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    ordered = sorted(finite)
    entry: Dict[str, object] = {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
    }
    if len(ordered) >= 2:
        entry["stdev"] = round(statistics.stdev(ordered), 6)
    return entry


def _summarize(realizations: Sequence[dict], base_radii: Sequence[float]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for index, sma in enumerate(base_radii):
        ring: Dict[str, object] = {}
        keys = list(_SUMMARY_KEYS)
        for order in ORDERS:
            keys += [
                f"a{order}_bender_vs_isoster_pct",
                f"b{order}_bender_vs_isoster_pct",
                f"s{order}_raw_autoprof_vs_truth_pct",
                f"c{order}_raw_autoprof_vs_truth_pct",
                f"s{order}_raw_autoprof_vs_isoster_pct",
                f"c{order}_raw_autoprof_vs_isoster_pct",
            ]
        for key in keys:
            ring[key] = _spread([r["rings"][index][key] for r in realizations])
        ring["statuses"] = sorted({r["rings"][index]["status"] for r in realizations})
        summary[f"sma={sma:g}"] = ring
    return summary


# ---------------------------------------------------------------------------
# Provenance, claims, freezing
# ---------------------------------------------------------------------------


def fixture_fingerprint() -> str:
    payload = json.dumps(
        {
            "fixture": ACTIVE_FIXTURE,
            "galaxy": _spec()["galaxy"],
            "radii": list(_spec()["radii"]),
            "orders": list(ORDERS),
            "planted": {f"{o}_{k}": v for (o, k), v in PLANTED_HARMONICS.items()},
            "gradient_step": GRADIENT_STEP,
            "grid": [{k: v for k, v in case.items() if k != "why"} for case in build_grid()],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


#: How a claim collapses its per-ring (and, for the harmonic families, its
#: per-order) values into one number. Both are archived for every claim,
#: because they answer different questions and neither substitutes for the
#: other.
#:
#: ``worst_ring`` is the pre-registered statistic and the one the licensing
#: criteria are evaluated on: a licence is a statement about the worst ring a
#: reader might write into a table, not the typical one. It is, however, an
#: unstable summary under noise --- it reports whichever of five rings happened
#: to be worst in this seed block --- so it reproduces loosely and its
#: tolerance is correspondingly loose.
#:
#: ``typical_ring`` is the median over the same values. It carries no verdict,
#: but it reproduces roughly forty times more tightly between seed blocks and
#: is therefore what actually constrains the gate.
CLAIM_REDUCTIONS = {
    "typical_ring": "median over rings (and over orders, where a claim spans several)",
    "worst_ring": "max over rings (and over orders, where a claim spans several)",
}

#: Every gated quantity, before the reduction prefix is applied.
#:
#: A ``key`` selector names one exact per-ring column. A ``suffix`` selector
#: gathers every per-ring column ending in it, which is how one harmonic claim
#: covers a3, b3, a4 and b4 at once.
CLAIM_DEFINITIONS: tuple[Dict[str, object], ...] = (
    {"stem": "gradient_agreement_pct_clean", "case": "reference", "key": "b0_secant_vs_isoster_pct"},
    *(
        {"stem": f"gradient_agreement_pct_{name}", "case": name, "key": "b0_secant_vs_isoster_pct"}
        for name in (
            "eps_circular",
            "eps_high",
            "pa_30",
            "background_offset",
            "interpolate_default",
            "noise_snr100",
            "noise_snr30",
        )
    ),
    # The wrong estimator, quantified rather than dismissed.
    {"stem": "median_estimator_penalty_pct", "case": "reference", "key": "median_secant_vs_isoster_pct"},
    # The convention offset, which is the finding that forced this design.
    {"stem": "secant_vs_point_derivative_pct", "case": "reference", "key": "isoster_vs_point_derivative_pct"},
    # Criterion 2: does normalizing introduce a new systematic?
    *(
        {"stem": f"{family}_agreement_pct_{name}", "case": name, "suffix": suffix}
        for name in ("reference", "eps_high", "noise_snr30")
        for family, suffix in (
            ("bender", "_bender_vs_isoster_pct"),
            ("raw", "_raw_autoprof_vs_isoster_pct"),
        )
    ),
)


def claims_fingerprint() -> str:
    """Hash of the claim *definitions*, so a redefinition cannot pass unnoticed.

    The fixture fingerprint covers what was measured. This covers what is
    claimed about it. Without it, changing a reduction while leaving the
    frozen tolerances in place would compare a validation value computed one
    way against a pilot value computed another --- a silent comparison of two
    different quantities, which is the failure this whole procedure exists to
    prevent.
    """
    payload = json.dumps(
        {"reductions": CLAIM_REDUCTIONS, "definitions": [dict(sorted(d.items())) for d in CLAIM_DEFINITIONS]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _selected_keys(case: Dict[str, object], definition: Dict[str, object]) -> List[str]:
    """The per-ring columns one claim reduces over, in a stable order."""
    if "key" in definition:
        return [str(definition["key"])]
    suffix = str(definition["suffix"])
    rings = list(case["summary"].values())
    if not rings:
        return []
    return sorted(key for key, entry in rings[0].items() if key.endswith(suffix) and isinstance(entry, dict))


def _reduce(values: Sequence[float], reduction: str) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return max(finite) if reduction == "worst_ring" else statistics.median(finite)


def extract_claims(results: Dict[str, object]) -> Dict[str, float]:
    """The numbers Track 2's acceptance turns on, under both reductions."""
    cases = {case["spec"]["name"]: case for case in results["cases"]}
    claims: Dict[str, float] = {}

    for definition in CLAIM_DEFINITIONS:
        case = cases.get(str(definition["case"]))
        if case is None:
            for reduction in CLAIM_REDUCTIONS:
                claims[f"{reduction}_{definition['stem']}"] = float("nan")
            continue
        keys = _selected_keys(case, definition)
        values = [
            entry[key]["median"]
            for entry in case["summary"].values()
            for key in keys
            if isinstance(entry.get(key), dict)
        ]
        for reduction in CLAIM_REDUCTIONS:
            claims[f"{reduction}_{definition['stem']}"] = _reduce(values, reduction)
    return claims


#: Criterion 1 asks that the matched secant beat the point derivative by a
#: decisive margin, not by an arbitrary one. Ten times is decisive; the
#: measured margin is nearer a hundred.
DECISIVE_MARGIN = 10.0


#: The realized basis Track 2's reconstruction is defined in. Anything else --
#: notably AutoProf's eccentric-anomaly basis -- mixes harmonic orders.
VALID_HARMONIC_BASIS = "polar_from_image_x_axis"

#: The realized sampling mode a ring must have been measured with.
VALID_SAMPLING_MODE = "line_interpolated"


def _comparison_ring_sampling_mode(provenance: Dict[str, object], index: int, n_rings: int) -> tuple[str | None, str]:
    """Realized sampling mode of the *comparison* ring, observed where possible.

    The worker records ``sampling_mode.per_ring_interpolated`` with one entry
    per requested ring: ``n_rings`` base rings followed by their ``n_rings``
    comparison partners, in request order. So the partner's mode is *observed*,
    and an earlier version of this function deriving it from ``rad_interp_pix``
    put inference back into the one field this design says is measured.

    The derivation survives only as a labelled fallback for archives written
    before the observation was read, and it reports itself as such. It is exact
    on those archives --- no pair in either straddles the threshold --- but
    "exact on the data in hand" is not the same as "measured", and a new run
    must never reach it.
    """
    modes = (provenance.get("sampling_mode") or {}).get("per_ring_interpolated")
    if isinstance(modes, list) and len(modes) == 2 * n_rings:
        observed = modes[n_rings + index]
        # Three states, not two. The worker records ``None`` when it cannot
        # attribute a sampling call to a ring, and ``bool(None)`` is ``False``
        # --- so an *unattributed* ring was archived as observed nearest-pixel.
        # That is a fabricated observation. The row is invalid either way, but
        # for opposite reasons, and only one of them is true.
        if observed is True:
            return VALID_SAMPLING_MODE, "observed"
        if observed is False:
            return "line_nearest_pixel", "observed"
        return None, "observed_unattributed"
    threshold = provenance.get("rad_interp_pix")
    if threshold is None:
        return None, "unavailable"
    return None, "legacy_derivation_unavailable"


def _comparison_ring_legacy_mode(ring: Dict[str, object], provenance: Dict[str, object]) -> str | None:
    threshold = provenance.get("rad_interp_pix")
    comparison = ring.get("comparison_sma")
    if threshold is None or comparison is None:
        return None
    return VALID_SAMPLING_MODE if float(comparison) < float(threshold) else "line_nearest_pixel"


def ring_pair_validity(
    ring: Dict[str, object],
    provenance: Dict[str, object],
    index: int,
    n_rings: int,
    completion: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Is the conversion defined for *this* ring and its comparison partner?

    Row-level, and read from **realized** provenance rather than from what was
    requested --- a review caught the previous version inferring both the basis
    and the sampling mode from the requested configuration, which inverts this
    project's rule to instrument the sampling mode and never predict it.

    Two separable things are reported, because they fail for different reasons
    and on different schedules:

    * **Structural applicability** is a property of the aperture and the
      configuration --- basis and both rings' sampling modes. It does not vary
      between noise realizations.
    * **Measurement completion** does vary: a ring can be structurally fine and
      still fail to measure in some realizations. ``completion`` carries the
      per-ring finite count from the summary, which covers *every* realization
      rather than the single one archives retain.
    """
    base_mode = ring.get("sampling_mode")
    comparison_mode, source = _comparison_ring_sampling_mode(provenance, index, n_rings)
    # An unattributed observation must not fall through to the derivation:
    # "we could not tell" is a different statement from "we worked it out".
    if comparison_mode is None and source != "observed_unattributed":
        comparison_mode, source = _comparison_ring_legacy_mode(ring, provenance), "derived_legacy_archive"

    basis = provenance.get("harmonic_basis")
    structural_reasons = []
    if basis != VALID_HARMONIC_BASIS:
        structural_reasons.append(f"realized harmonic basis is {basis!r}, which mixes orders")
    if base_mode != VALID_SAMPLING_MODE:
        structural_reasons.append(f"base ring realized sampling is {base_mode!r}, not interpolated")
    if source == "observed_unattributed":
        structural_reasons.append(
            "comparison ring sampling mode is unknown: the worker could not attribute a sampling call to it"
        )
    elif comparison_mode != VALID_SAMPLING_MODE:
        structural_reasons.append(f"comparison ring realized sampling is {comparison_mode!r}, not interpolated")

    measured, expected = (completion or {}).get("measured"), (completion or {}).get("expected")
    completion_reasons = []
    if ring.get("status") != "measured":
        completion_reasons.append(f"ring status is {ring.get('status')!r} in the archived realization")
    if measured is not None and expected is not None and measured < expected:
        completion_reasons.append(f"only {measured} of {expected} realizations produced a finite secant")

    return {
        "sma": ring.get("sma"),
        "comparison_sma": ring.get("comparison_sma"),
        "harmonic_basis": basis,
        "base_sampling_mode": base_mode,
        "comparison_sampling_mode": comparison_mode,
        "comparison_sampling_mode_source": source,
        "structurally_applicable": not structural_reasons,
        "measurement_complete": not completion_reasons,
        "realizations_measured": measured,
        "realizations_expected": expected,
        "realizations_measured_source": (completion or {}).get("source"),
        "harmonic_conversion_valid": not (structural_reasons or completion_reasons),
        "harmonic_conversion_reasons": structural_reasons + completion_reasons,
    }


def structural_validity(case: Dict[str, object]) -> Dict[str, object]:
    """Row-level validity for a case, plus aggregates for summaries only.

    The rows are the result. The aggregates exist so a table can say "3 of 5",
    and must never be what a decision reads --- that was the original defect.
    """
    provenance = case.get("autoprof_provenance") or {}
    realizations = case.get("realizations") or []
    rings = realizations[0]["rings"] if realizations else []
    expected = int(case.get("spec", {}).get("n_realizations") or 0) or None

    # Keyed by radius, never by position. Iterating ``summary.values()`` and
    # assigning positionally silently pairs a ring with another ring's count
    # whenever the summary is ordered differently from the ring list --- the
    # same defect ``_by_sma`` exists to prevent, reintroduced one level up.
    #
    # And the count comes from ``b0_secant_vs_point_derivative_pct``, not from
    # the isoster comparison: the latter is undefined when *isoster's* gradient
    # is zero or invalid, which says nothing about whether AutoProf measured
    # both of its rings. The analytic point derivative is deterministic and
    # non-zero, so that column is finite exactly when AutoProf's secant is.
    summary = case.get("summary") or {}
    by_radius: Dict[float, Dict[str, object]] = {}
    for key, entry in summary.items():
        if not isinstance(entry, dict):
            continue
        try:
            radius = round(float(str(key).split("=", 1)[1]), 6)
        except (IndexError, ValueError):
            continue
        explicit = entry.get("autoprof_secant_measured")
        proxy = entry.get("b0_secant_vs_point_derivative_pct")
        if isinstance(explicit, dict):
            by_radius[radius] = {"measured": explicit.get("n"), "source": "autoprof_secant_measured"}
        elif isinstance(proxy, dict):
            by_radius[radius] = {"measured": proxy.get("n"), "source": "b0_secant_vs_point_derivative_pct"}

    rows = []
    for index, ring in enumerate(rings):
        completion = dict(by_radius.get(round(float(ring.get("sma", float("nan"))), 6), {"measured": None}))
        completion["expected"] = expected
        rows.append(ring_pair_validity(ring, provenance, index, len(rings), completion))
    valid = [row for row in rows if row["harmonic_conversion_valid"]]
    return {
        "rows": rows,
        "valid_rows": len(valid),
        "total_rows": len(rows),
        "all_rows_valid": bool(rows) and len(valid) == len(rows),
        "any_row_valid": bool(valid),
        "structurally_applicable_rows": sum(1 for row in rows if row["structurally_applicable"]),
        "measurement_complete_rows": sum(1 for row in rows if row["measurement_complete"]),
    }


def algebraic_consistency(case: Dict[str, object]) -> Dict[str, object]:
    """Does Bender error obey the exact bound implied by ``B = R / G``?

    This is **not** evidence that the conversion is right, and it is recorded
    here as a diagnostic so that it cannot be mistaken for evidence again. The
    Bender amplitude is *constructed from* the same raw amplitude and the same
    gradient, so a comparison between them is an arithmetic identity check.

    It replaces a criterion that added the two percentage errors linearly.
    That is only a first-order approximation; the exact bound carries the
    denominator,

        |B - 1| <= (|R - 1| + |G - 1|) / (1 - |G - 1|),

    and the difference is what produced a scatter of tiny paired "failures" on
    configurations that are in fact consistent --- excesses of 3e-4 and 7e-5,
    which were briefly read as evidence that the paired form was unusable.
    Under the exact bound the noiseless cases satisfy it to rounding.
    """
    worst, checked, violations = -float("inf"), 0, 0
    for entry in case["summary"].values():
        gradient = entry.get("b0_secant_vs_isoster_pct")
        if not isinstance(gradient, dict):
            continue
        g = float(gradient["median"])
        for order in ORDERS:
            for bender_prefix, raw_prefix in (("a", "s"), ("b", "c")):
                bender = entry.get(f"{bender_prefix}{order}_bender_vs_isoster_pct")
                raw = entry.get(f"{raw_prefix}{order}_raw_autoprof_vs_isoster_pct")
                if not (isinstance(bender, dict) and isinstance(raw, dict)):
                    continue
                denominator = 1.0 - g / 100.0
                if denominator <= 0:
                    continue
                bound = (float(raw["median"]) + g) / denominator
                excess = float(bender["median"]) - bound
                worst = max(worst, excess)
                checked += 1
                violations += excess > 1e-6
    if not checked:
        return {"checked": 0, "consistent": None}
    return {
        "checked": checked,
        "violations": violations,
        "worst_excess_pct": round(worst, 9),
        "consistent": bool(violations == 0),
        "note": "diagnostic only; an identity check, never an input to the verdict",
    }


def evaluate_licensing(results: Dict[str, object]) -> Dict[str, object]:
    """Decide whether the conversion may be used, and report how well it does.

    **Revised 2026-08-23 after review.** The original design licensed Track 2
    on two criteria. Criterion 2 asked that the Bender agreement be no worse
    than the raw agreement plus the gradient error --- but Bender *is* raw
    divided by gradient, so that compares a quantity with the two quantities it
    is built from. It is an arithmetic consistency check, not independent
    evidence, and it cannot license anything. It also compared maxima drawn
    from different rings and different components, and added percentage errors
    linearly where the exact relation carries a denominator.

    So validity is now **structural** --- correct angular basis, interpolated
    sampling, a measured comparison ring --- and accuracy is reported as
    *performance*, per regime, without gating on it. The consistency check
    survives as an explicitly labelled diagnostic.

    This revises a pre-registered criterion after seeing data, which needs
    saying plainly. It is defensible only because the objection is structural:
    criterion 2 could not have been evidence whatever numbers it returned, and
    the revision neither loosens nor tightens a bar to reach a conclusion.
    Criterion 1, which does compare two independent candidate reconstructions
    against one target, is unchanged and still carries the verdict.
    """
    claims = extract_claims(results)
    cases = {case["spec"]["name"]: case for case in results["cases"]}

    secant = claims.get("worst_ring_gradient_agreement_pct_clean", float("nan"))
    point = claims.get("worst_ring_secant_vs_point_derivative_pct", float("nan"))
    criterion_1 = bool(np.isfinite(secant) and np.isfinite(point) and point > DECISIVE_MARGIN * secant)

    regimes = {}
    for name in sorted(cases):
        stem = "gradient_agreement_pct_clean" if name == "reference" else f"gradient_agreement_pct_{name}"
        regimes[name] = {
            # Performance, reported and never gated on.
            "gradient_agreement_pct": claims.get(f"worst_ring_{stem}", float("nan")),
            "raw_agreement_pct": claims.get(f"worst_ring_raw_agreement_pct_{name}"),
            "bender_agreement_pct": claims.get(f"worst_ring_bender_agreement_pct_{name}"),
            "typical_ring_gradient_agreement_pct": claims.get(f"typical_ring_{stem}"),
            "structural_validity": structural_validity(cases[name]),
            "algebraic_consistency": algebraic_consistency(cases[name]),
        }

    reference = regimes.get("reference", {}).get("structural_validity", {})
    reference_valid = bool(reference.get("all_rows_valid"))
    return {
        # 1. Did the *method* get empirical support? This is criterion 1, and it
        #    is an accuracy comparison -- two candidate reconstructions weighed
        #    against one target. Campaign-level, and named so that nobody reads
        #    it as a property of a ring.
        "conversion_method_validated": criterion_1,
        "criterion_1_beats_point_derivative": criterion_1,
        "criterion_1_margin": float(point / secant) if secant else float("nan"),
        "criterion_1_decisive_margin_required": DECISIVE_MARGIN,
        # 2. Does a given *row* structurally support conversion? Per ring pair,
        #    from realized provenance. See regimes[*].structural_validity.rows.
        "all_reference_rows_structurally_valid": reference_valid,
        # 3. How accurate was it, per regime? Reported, gating nothing.
        #    Deliberately no "licensed_on_..." field: it merged all three of
        #    these into one boolean and read as a property of the conversion
        #    when it was partly a statement about accuracy in one regime.
        "regimes": regimes,
        "withdrawn_criterion_2": (
            "The original criterion 2 (bender <= raw + gradient) was withdrawn on 2026-08-23. "
            "Bender is constructed from raw and gradient, so the comparison is an arithmetic "
            "identity check rather than independent evidence; it also compared maxima from "
            "different rings and added percentage errors linearly where the exact bound carries "
            "a denominator. It survives as the 'algebraic_consistency' diagnostic."
        ),
        "conditions": [
            "polar-resampled path only (ap_isoclip=True); never the eccentric-anomaly basis",
            "rings sampled with interpolation, not nearest-pixel rounding",
            "the comparison ring at sma*(1+astep) must be measured, not interpolated",
        ],
    }


#: Bootstrap resamples used to measure how far a claim moves between seed
#: blocks. Two thousand is far more than a standard deviation needs and costs
#: under a second per claim, so there is no reason to economize.
BOOTSTRAP_DRAWS = 2000


def _claim_table(case: Dict[str, object], definition: Dict[str, object]) -> np.ndarray | None:
    """Per-realization values for every column one claim reduces over.

    Shape ``(n_realizations, n_rings * n_keys)``. Returns ``None`` for a
    deterministic case, which has a single realization and therefore no
    run-to-run scatter to measure.
    """
    realizations = case.get("realizations") or []
    if len(realizations) < 2:
        return None
    keys = _selected_keys(case, definition)
    n_rings = len(case["summary"])
    rows = []
    for realization in realizations:
        rings = realization["rings"]
        rows.append([rings[index][key] for index in range(n_rings) for key in keys])
    return np.asarray(rows, dtype=float)


def _bootstrap_scatter(
    case: Dict[str, object], definition: Dict[str, object], reduction: str, seed: int
) -> float | None:
    """Standard deviation of the claim itself, over resampled realizations.

    This replaces an earlier derivation that took the largest *single-ring*
    standard error and used it for a claim that was a max over five rings.
    That proxy is only correct when the claim is one ring's median: a max
    additionally varies through *which* ring wins, so its true spread is
    larger, and the gate duly failed a validation run whose measurement was
    fine. Resampling whole realizations keeps the rings correlated the way the
    data has them and reduces exactly the statistic being claimed, so the
    tolerance measures the claim rather than a stand-in for it.

    The seed is derived from the claim's name, so every claim gets its own
    stream and the frozen file is reproducible.
    """
    table = _claim_table(case, definition)
    if table is None or table.shape[1] == 0:
        return None
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, table.shape[0], size=(BOOTSTRAP_DRAWS, table.shape[0]))
    with warnings.catch_warnings():
        # An all-NaN column is a ring nothing measured; it must not become a
        # loud failure here, because the claim itself already drops it.
        warnings.simplefilter("ignore", RuntimeWarning)
        per_column = np.nanmedian(table[picks], axis=1)
        statistic = np.nanmax(per_column, axis=1) if reduction == "worst_ring" else np.nanmedian(per_column, axis=1)
    finite = statistic[np.isfinite(statistic)]
    if finite.size < 2:
        return None
    return float(np.std(finite, ddof=1))


def _claim_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def freeze_tolerances(pilot: Dict[str, object]) -> Dict[str, object]:
    claims = extract_claims(pilot)
    cases = {case["spec"]["name"]: case for case in pilot["cases"]}
    frozen: Dict[str, object] = {}

    for definition in CLAIM_DEFINITIONS:
        case = cases.get(str(definition["case"]))
        for reduction in CLAIM_REDUCTIONS:
            name = f"{reduction}_{definition['stem']}"
            value = float(claims[name])
            scatter = None if case is None else _bootstrap_scatter(case, definition, reduction, _claim_seed(name))
            if scatter is None:
                tolerance = max(DETERMINISTIC_FLOOR_PCT, 0.02 * abs(value))
                basis = "deterministic_floor"
            else:
                tolerance = max(TOLERANCE_SAFETY_FACTOR * scatter, DETERMINISTIC_FLOOR_PCT)
                basis = "bootstrap"
            frozen[name] = {
                "pilot_value": round(value, 6),
                "tolerance": round(float(tolerance), 6),
                "basis": basis,
                "reduction": reduction,
            }
            if scatter is not None:
                frozen[name]["bootstrap_stdev"] = round(scatter, 6)

    return {
        "frozen_from": {
            "mode": pilot["mode"],
            "fixture": pilot["fixture"],
            "seed_block": pilot["seed_block"],
            "commit": pilot["environment"]["git"]["commit"],
            # Recorded, not enforced. A pilot only sets the expected value; the
            # archive it is judged against is what must come from a clean tree.
            "dirty": pilot["environment"]["git"].get("dirty"),
            "fixture_fingerprint": pilot["environment"]["fixture_fingerprint"],
        },
        "policy": {
            "safety_factor": TOLERANCE_SAFETY_FACTOR,
            "deterministic_floor_pct": DETERMINISTIC_FLOOR_PCT,
            "gradient_step": GRADIENT_STEP,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "claims_fingerprint": claims_fingerprint(),
            "claim_reductions": dict(CLAIM_REDUCTIONS),
            "note": (
                "Each claim's tolerance is measured by bootstrapping that claim's own "
                "definition -- the same reduction over the same columns -- across resampled "
                "realizations of the pilot. Never from another claim, and never from a "
                "single-ring standard error standing in for a reduction over many rings. "
                "The validation run draws from a disjoint seed block and is judged against "
                "these values."
            ),
        },
        "claims": frozen,
    }


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------


def run_grid(mode: str, only: str | None = None) -> Dict[str, object]:
    seed_block = SEED_BLOCKS[ACTIVE_FIXTURE][mode]
    grid = build_grid()
    if only:
        grid = [case for case in grid if case["name"] == only]
        if not grid:
            raise SystemExit(f"no such case: {only}")

    cases = []
    with tempfile.TemporaryDirectory(prefix="gradient-recon-") as workspace_root:
        for index, spec in enumerate(grid, start=1):
            print(f"[gradient] {index}/{len(grid)} {spec['name']} ...", flush=True)
            cases.append(measure_case(spec, seed_block, Path(workspace_root)))

    environment = _environment()
    environment["fixture_fingerprint"] = fixture_fingerprint()
    results = {
        "measurement": "a4_track2_gradient_reconstruction",
        "mode": mode,
        "fixture": ACTIVE_FIXTURE,
        "gradient_step": GRADIENT_STEP,
        "seed_block": seed_block,
        "seed_blocks": SEED_BLOCKS[ACTIVE_FIXTURE],
        "environment": environment,
        "cases": cases,
    }
    results["licensing"] = evaluate_licensing(results)
    return results


def _refuse_archive(results: Dict[str, object], only: str | None) -> List[str]:
    problems = []
    if results["mode"] != "validation":
        problems.append(f"mode is {results['mode']!r}; only a validation run may be archived")
    if only:
        problems.append(f"--only {only} runs one case; the archive must contain the whole grid")
    for when, git in (("when the run was made", results["environment"]["git"]), ("now", _git_state())):
        if git.get("dirty") is None:
            problems.append(f"could not determine whether the working tree was clean {when}")
        elif git["dirty"]:
            problems.append(f"working tree was dirty {when}")
    if not tolerances_path().exists():
        problems.append(
            f"{tolerances_path().name} does not exist; tolerances must be frozen from the "
            "pilot before the validation run"
        )
    return problems


def _write(results: Dict[str, object], archive: bool) -> None:
    output_root = Path(resolve_output_directory("benchmark_harmonic_scale"))
    out_path = output_root / f"gradient_{ACTIVE_FIXTURE}_{results['mode']}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[gradient] wrote {out_path}")
    if archive:
        trimmed = json.loads(json.dumps(results, default=str))
        dropped = 0
        for case in trimmed["cases"]:
            realizations = case["realizations"]
            case["realizations_run"] = len(realizations)
            case["realizations_stored"] = min(1, len(realizations))
            if len(realizations) > 1:
                dropped += len(realizations) - 1
                case["realizations"] = realizations[:1]
        trimmed["archive_note"] = (
            f"{dropped} per-realization records summarized rather than stored; every claim "
            "comes from the complete 'summary' block, and the dropped records are "
            "regenerable exactly from the recorded seed block."
        )
        archive_path().write_text(json.dumps(trimmed, indent=2, default=str))
        print(f"[gradient] archived {archive_path()} ({archive_path().stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    global ACTIVE_FIXTURE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=sorted(SEED_BLOCKS), default="sersic_n2_compact")
    parser.add_argument("--mode", choices=("pilot", "validation"), default="pilot")
    parser.add_argument("--only", help="Run a single named case (never archivable).")
    parser.add_argument("--freeze-tolerances", type=Path, metavar="PILOT_JSON")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument(
        "--regenerate-licensing",
        action="store_true",
        help=(
            "Recompute only the archive's derived licensing block from its own stored "
            "measurements. Touches no measured value; used when a verdict's definition "
            "changes and re-measuring would be pointless."
        ),
    )
    args = parser.parse_args()
    ACTIVE_FIXTURE = args.fixture

    if args.freeze_tolerances is not None:
        pilot = json.loads(args.freeze_tolerances.read_text())
        if pilot.get("mode") != "pilot":
            raise SystemExit(f"{args.freeze_tolerances} is a {pilot.get('mode')!r} run, not a pilot")
        if pilot.get("fixture") != ACTIVE_FIXTURE:
            raise SystemExit(
                f"{args.freeze_tolerances} is a {pilot.get('fixture')!r} pilot but --fixture is {ACTIVE_FIXTURE!r}"
            )
        frozen = freeze_tolerances(pilot)
        tolerances_path().write_text(json.dumps(frozen, indent=2))
        print(f"[gradient] froze {len(frozen['claims'])} tolerances to {tolerances_path()}")
        for name, entry in sorted(frozen["claims"].items()):
            print(f"   {name:44s} {entry['pilot_value']:>10.5g} +- {entry['tolerance']:<8.4g}")
        return

    if args.regenerate_licensing:
        archive = json.loads(archive_path().read_text())
        before = json.dumps(archive["cases"], sort_keys=True)
        archive["licensing"] = evaluate_licensing(archive)
        if json.dumps(archive["cases"], sort_keys=True) != before:
            raise SystemExit("refusing to write: regeneration altered a measured value")
        archive["licensing_regenerated"] = (
            "The licensing block was recomputed from this archive's own stored measurements "
            "after criterion 2 was withdrawn on 2026-08-23. No measured value was touched; "
            "the summaries and their fingerprint are unchanged."
        )
        archive_path().write_text(json.dumps(archive, indent=2, default=str))
        print(f"[gradient] regenerated licensing in {archive_path().name}")
        verdict = archive["licensing"]
        print(f"   conversion_method_validated={verdict['conversion_method_validated']}")
        print(f"   all_reference_rows_structurally_valid={verdict['all_reference_rows_structurally_valid']}")
        return

    results = run_grid(args.mode, only=args.only)
    if args.archive:
        problems = _refuse_archive(results, args.only)
        if problems:
            _write(results, archive=False)
            raise SystemExit("refusing to archive:\n  - " + "\n  - ".join(problems))
    _write(results, archive=args.archive)


if __name__ == "__main__":
    main()
