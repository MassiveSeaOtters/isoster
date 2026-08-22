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
Track 2 is licensed --- meaning A5's schema may write ``a_n``/``b_n`` for an
AutoProf arm instead of NaN --- only if **both** hold on the clean
configuration:

1. the ``b0`` matched secant reproduces isoster's gradient within the frozen
   tolerance, and
2. normalizing introduces no *new* systematic: the Bender agreement is no
   worse than Track 1's raw agreement plus that gradient error.

If either fails, Track 2 stays unlicensed. A criterion that can only be met is
not a criterion.

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


def extract_claims(results: Dict[str, object]) -> Dict[str, float]:
    """The numbers Track 2's acceptance turns on."""
    cases = {case["spec"]["name"]: case for case in results["cases"]}
    claims: Dict[str, float] = {}

    def worst(case_name: str, key: str) -> float:
        case = cases.get(case_name)
        if case is None:
            return float("nan")
        values = [entry[key]["median"] for entry in case["summary"].values() if isinstance(entry.get(key), dict)]
        return max(values) if values else float("nan")

    # Criterion 1: does the reconstruction reproduce isoster's gradient?
    claims["gradient_agreement_pct_clean"] = worst("reference", "b0_secant_vs_isoster_pct")
    for name in ("eps_circular", "eps_high", "pa_30", "background_offset", "interpolate_default"):
        claims[f"gradient_agreement_pct_{name}"] = worst(name, "b0_secant_vs_isoster_pct")
    for name in ("noise_snr100", "noise_snr30"):
        claims[f"gradient_agreement_pct_{name}"] = worst(name, "b0_secant_vs_isoster_pct")

    # The wrong estimator, quantified rather than dismissed.
    claims["median_estimator_penalty_pct"] = worst("reference", "median_secant_vs_isoster_pct")
    # The convention offset, which is the finding that forced this design.
    claims["secant_vs_point_derivative_pct"] = worst("reference", "isoster_vs_point_derivative_pct")

    # Criterion 2: does normalizing introduce a new systematic?
    for name in ("reference", "eps_high", "noise_snr30"):
        bender = [
            entry[key]["median"]
            for entry in (cases[name]["summary"].values() if name in cases else [])
            for key in entry
            if key.endswith("_bender_vs_isoster_pct") and isinstance(entry.get(key), dict)
        ]
        raw = [
            entry[key]["median"]
            for entry in (cases[name]["summary"].values() if name in cases else [])
            for key in entry
            if key.endswith("_raw_autoprof_vs_isoster_pct") and isinstance(entry.get(key), dict)
        ]
        claims[f"bender_agreement_pct_{name}"] = max(bender) if bender else float("nan")
        claims[f"raw_agreement_pct_{name}"] = max(raw) if raw else float("nan")
    return claims


#: Criterion 1 asks that the matched secant beat the point derivative by a
#: decisive margin, not by an arbitrary one. Ten times is decisive; the
#: measured margin is nearer a hundred.
DECISIVE_MARGIN = 10.0


def evaluate_licensing(results: Dict[str, object]) -> Dict[str, object]:
    """Apply the two pre-registered criteria and state the verdict.

    Both are comparisons, not thresholds: criterion 1 weighs two candidate
    reconstructions against the same target, criterion 2 weighs a normalized
    quantity against the unnormalized one it is built from. Neither needs a
    number chosen in advance, which is what keeps them able to fail.

    The verdict is per regime rather than global. The reconstruction is not
    uniformly good, and a single yes/no would either overstate it where it
    degrades or discard it where it works.
    """
    claims = extract_claims(results)
    cases = {case["spec"]["name"]: case for case in results["cases"]}

    secant = claims.get("gradient_agreement_pct_clean", float("nan"))
    point = claims.get("secant_vs_point_derivative_pct", float("nan"))
    criterion_1 = bool(np.isfinite(secant) and np.isfinite(point) and point > DECISIVE_MARGIN * secant)

    regimes = {}
    for name in sorted(cases):
        gradient_key = "gradient_agreement_pct_clean" if name == "reference" else f"gradient_agreement_pct_{name}"
        gradient = claims.get(gradient_key, float("nan"))
        bender = claims.get(f"bender_agreement_pct_{name}")
        raw = claims.get(f"raw_agreement_pct_{name}")
        if bender is None or raw is None:
            regimes[name] = {
                "gradient_agreement_pct": gradient,
                "criterion_2": None,
                "note": "criterion 2 not evaluated for this case",
            }
            continue
        budget = raw + gradient
        regimes[name] = {
            "gradient_agreement_pct": gradient,
            "raw_agreement_pct": raw,
            "bender_agreement_pct": bender,
            "budget_pct": budget,
            "criterion_2": bool(bender <= budget),
        }

    licensed = criterion_1 and bool(regimes.get("reference", {}).get("criterion_2"))
    return {
        "criterion_1_beats_point_derivative": criterion_1,
        "criterion_1_margin": float(point / secant) if secant else float("nan"),
        "criterion_1_decisive_margin_required": DECISIVE_MARGIN,
        "licensed_on_reference_configuration": licensed,
        "regimes": regimes,
        "conditions": [
            "polar-resampled path only (ap_isoclip=True); never the eccentric-anomaly basis",
            "rings sampled with interpolation, not nearest-pixel rounding",
            "the comparison ring at sma*(1+astep) must be measured, not interpolated",
        ],
    }


def freeze_tolerances(pilot: Dict[str, object]) -> Dict[str, object]:
    claims = extract_claims(pilot)
    noisy = {"noise_snr100", "noise_snr30"}
    frozen: Dict[str, object] = {}
    for name, value in claims.items():
        if any(tag in name for tag in noisy):
            scatter = _noise_scatter(pilot)
            tolerance = max(TOLERANCE_SAFETY_FACTOR * scatter, DETERMINISTIC_FLOOR_PCT)
            basis = "scatter"
        else:
            tolerance = max(DETERMINISTIC_FLOOR_PCT, 0.02 * abs(float(value)))
            basis = "deterministic_floor"
        frozen[name] = {
            "pilot_value": round(float(value), 6),
            "tolerance": round(float(tolerance), 6),
            "basis": basis,
        }
    return {
        "frozen_from": {
            "mode": pilot["mode"],
            "fixture": pilot["fixture"],
            "seed_block": pilot["seed_block"],
            "commit": pilot["environment"]["git"]["commit"],
            "fixture_fingerprint": pilot["environment"]["fixture_fingerprint"],
        },
        "policy": {
            "safety_factor": TOLERANCE_SAFETY_FACTOR,
            "deterministic_floor_pct": DETERMINISTIC_FLOOR_PCT,
            "gradient_step": GRADIENT_STEP,
            "note": (
                "Derived from the pilot's measured scatter by one uniform rule. The "
                "validation run draws from a disjoint seed block and is judged against "
                "these values."
            ),
        },
        "claims": frozen,
    }


def _noise_scatter(pilot: Dict[str, object]) -> float:
    """Largest across-realization scatter of the gradient agreement."""
    worst = 0.0
    for case in pilot["cases"]:
        if case["spec"]["snr"] is None:
            continue
        count = max(1, int(case["spec"]["n_realizations"]))
        for ring in case["summary"].values():
            entry = ring.get("b0_secant_vs_isoster_pct")
            if isinstance(entry, dict) and "stdev" in entry:
                worst = max(worst, float(entry["stdev"]) / np.sqrt(count))
    return worst


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

    results = run_grid(args.mode, only=args.only)
    if args.archive:
        problems = _refuse_archive(results, args.only)
        if problems:
            _write(results, archive=False)
            raise SystemExit("refusing to archive:\n  - " + "\n  - ".join(problems))
    _write(results, archive=args.archive)


if __name__ == "__main__":
    main()
