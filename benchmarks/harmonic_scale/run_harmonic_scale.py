"""Measure whether the three tools put ``a_n`` and ``b_n`` on the same scale.

This is A3 of ``docs/specs/2026-08-22-three-way-benchmark-comparison-design.md``.
A1 pinned what each tool's coefficients *mean* on exactly known rings; A2 built
a planted-harmonic image fixture with an integrated analytic truth and one
fixed-aperture adapter per tool. This script runs those adapters over a
designed grid and reports, per case and per ring, the ratio of each tool's
measurement to that truth.

What the grid is, and what it deliberately is not
-------------------------------------------------
Seven axes were identified; six are active (the mask axis is deferred, because
AutoProf's forced pipeline composes no mask-loading step and a masked arm would
silently mask nothing). Their full Cartesian product is not run. A saturated
design is both large and uninformative: when a case comes out wrong it cannot
say which factor did it. The structure is instead

1. one **reference** configuration, which everything else perturbs;
2. **one factor at a time** off that reference;
3. the four **interactions** with a reason to exist --- radius x interpolation
   start, ellipticity x basis, PA x polar resampling, noise x clipping;
4. one **combined stress** case, as a check rather than a measurement.

The first axis, ``ap_iso_interpolate_start``, is the largest effect in the
study by an order of magnitude: AutoProf reads 13-25% high at its default and
agrees with analytic truth to 0.1% with interpolation used everywhere. Both
readings are true at their own setting, which is why the setting is crossed
rather than chosen. It selects Lanczos sampling where ``Rlim < rad_interp``
and rounding to the nearest pixel otherwise, with
``rad_interp = ap_iso_interpolate_start * results["psf fwhm"]``. On this
pipeline that PSF is not measured: the ``psf`` step is ``PSF_Assumed``, which
hardcodes 4.0 px unless told otherwise, so the threshold sits at 20 px at the
default setting. Every ring nonetheless carries the mode it actually got,
observed inside the run rather than predicted -- the per-ring mode is the
thing being studied, and the PSF step is swappable.

Pilot and validation are separate runs on disjoint seed blocks
--------------------------------------------------------------
``--mode pilot`` measures the numerical scatter. Tolerances are then frozen
from that scatter and written to ``frozen_tolerances.json`` *before*
``--mode validation`` runs. Tuning a tolerance on the realizations it will
later judge is selection on the evaluated data at one remove, so the two modes
draw from seed blocks that do not overlap and each run records which block it
used.

Usage::

    uv run python benchmarks/harmonic_scale/run_harmonic_scale.py --mode pilot
    uv run python benchmarks/harmonic_scale/run_harmonic_scale.py --freeze-tolerances
    uv run python benchmarks/harmonic_scale/run_harmonic_scale.py --mode validation --archive

Results go to ``outputs/benchmark_harmonic_scale/``; ``--archive`` promotes a
validation run to the committed ``reference_harmonic_scale.json``, and refuses
to do so from a dirty working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harmonic_scale.adapters import (  # noqa: E402
    assert_rings_match_request,
    measure_autoprof_fixed,
    measure_isoster_fixed,
    measure_photutils_fixed,
)
from benchmarks.utils.sersic_model import (  # noqa: E402
    add_noise,
    create_sersic_image_with_harmonics,
    integrated_harmonic_truth,
    linearized_harmonic_truth,
)
from isoster.output_paths import resolve_output_directory  # noqa: E402

HERE = Path(__file__).resolve().parent
ARCHIVE_PATH = HERE / "reference_harmonic_scale.json"
TOLERANCES_PATH = HERE / "frozen_tolerances.json"

# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

#: Two galaxies, each its own campaign with its own archive, its own frozen
#: tolerances and its own seed blocks. They are *additive*: the first is
#: frozen and must stay byte-identical, because its archive is already gated,
#: so the second is a separate measurement rather than an extension of it.
#:
#: Why a second at all: the first campaign has six axes and one galaxy, so
#: every number it reports is conditional on that galaxy. The second differs
#: in Sersic index, size and image dimensions --- a steeper core and a wider
#: profile, which changes the radial gradient the Bender normalization
#: divides by and the number of pixels a ring spans.
#:
#: ``extra_cases`` is what keeps the first frozen: it is empty there, so the
#: grid, and therefore the fixture fingerprint, is exactly what was archived.
FIXTURES = {
    "sersic_n2_compact": {
        # Names the campaign in prose. Claims are qualified by it, because
        # "the three tools agree to X%" is a different sentence for each
        # galaxy and an unqualified one cannot be checked against either.
        "label": "compact n=2",
        "galaxy": {
            "n": 2.0,
            "R_e": 25.0,
            "I_e": 100.0,
            "shape": (241, 241),
            "center": (120.0, 120.0),
        },
        "radii": (12.0, 18.0, 25.0, 35.0, 45.0),
        "reference_eps": 0.3,
        "extra_cases": (),
        "pilot_seed_block": 900_000,
        "validation_seed_block": 20_260_822,
        "archive": "reference_harmonic_scale.json",
        "tolerances": "frozen_tolerances.json",
    },
    "sersic_n4_extended": {
        "label": "extended n=4",
        "galaxy": {
            "n": 4.0,
            "R_e": 40.0,
            "I_e": 100.0,
            "shape": (321, 321),
            "center": (160.0, 160.0),
        },
        # Wider galaxy, so wider rings; still straddling the switch at both
        # the default threshold (20 px) and the ap_set_psf=8 one (40 px).
        "radii": (18.0, 28.0, 40.0, 55.0, 70.0),
        "reference_eps": 0.3,
        "extra_cases": ("psf_set_8", "psf_x_interpolate", "threshold_matched_control"),
        # Blocks disjoint from both of the first campaign's, and separated by
        # far more than any realization count. An earlier draft used
        # 20_260_823 -- one away from the first campaign's validation block,
        # so with 25 realizations the two campaigns would have shared 24 of
        # their 25 noise draws and their "independent" results would not have
        # been independent at all.
        "pilot_seed_block": 700_000,
        "validation_seed_block": 30_260_822,
        "archive": "reference_harmonic_scale_n4.json",
        "tolerances": "frozen_tolerances_n4.json",
    },
}

#: The campaign being run. Module-level because the fixture is a property of
#: the whole run rather than of any one call, and because threading it through
#: every function would be a large diff across code whose output is already
#: frozen and gated. Same pattern, and same reasoning, as ``ACTIVE_SEED`` in
#: ``benchmarks/draft_timings/run_draft_timings.py``.
ACTIVE_FIXTURE = "sersic_n2_compact"

FIXTURE = FIXTURES[ACTIVE_FIXTURE]["galaxy"]

#: Four planted modes, all four amplitudes distinct. Both components are
#: populated at each order because the AutoProf conversion ends in a rotation
#: by ``n * PA``, which mixes sine into cosine: a fixture carrying only cosine
#: would leave half of that rotation untested. Distinct amplitudes mean a
#: transposed index shows up as a wrong number rather than as agreement.
#:
#: Planting several modes at once makes the first-order result
#: ``a_n = eps_n^sin`` false --- Sersic curvature and products between modes
#: generate additional harmonics --- which is why the reference value comes
#: from :func:`integrated_harmonic_truth` and the linearized value is carried
#: only as a diagnostic of how far that approximation is off.
PLANTED_HARMONICS = {
    (3, "sin"): 0.015,
    (3, "cos"): 0.020,
    (4, "sin"): 0.010,
    (4, "cos"): 0.030,
}

#: A single mode, used by one control case. Added after the pilot, which
#: found photutils reading up to 1.8% high on ``c3`` in the *cleanest*
#: configuration while isoster and AutoProf sat near 0.03%. Isolating the
#: cause needed a one-mode ring: with only ``m=3 cos`` planted, photutils
#: recovers it to 0.24% and the excess is gone.
#:
#: The proposed mechanism is angular sampling, and it is a hypothesis. What is
#: certain is arithmetic: photutils' ring samples are not evenly spaced in
#: polar angle -- measured spacing varies by a factor of 1.4 around one ring --
#: so the harmonic basis is not orthogonal on them (``<sin 3phi, cos 4phi>`` =
#: 6e-4, against 1e-18 on an even grid). That *would* let a fit modelling one
#: order at a time absorb part of the others. isoster and AutoProf's
#: polar-resampled path both work on evenly spaced angles, where the orders are
#: orthogonal and no such leakage is possible.
#:
#: What is **not established** is that this non-orthogonality is the cause of
#: the residual's size. The metric below compares differences between worst
#: errors, possibly at different radii, and is not a cross-order response
#: coefficient. Settling it needs a joint four-component fit on photutils' own
#: sampled angles, or the 4x4 response matrix built from those angles.
#:
#: Either way the effect is estimator-dependent rather than a scale error, so
#: it must not be folded into the scale tolerance; the control case is what
#: keeps the two separable in the archive.
SINGLE_PLANTED_HARMONIC = {(3, "cos"): 0.020}

ORDERS = (3, 4)

#: Five radii, chosen to straddle the interpolation switch at more than one
#: setting. AutoProf's ``psf`` step assumes 4.0 px on this pipeline, so the
#: switch sits near 20 px at the default setting of 5 and near 32 px at 8.
#: Both splits are therefore visible within one radius set, which is what
#: makes radius x interpolation start an interaction rather than two separate
#: one-factor cases. Where the switch actually fell is recorded per run.
RADII = FIXTURES[ACTIVE_FIXTURE]["radii"]

#: Reference ellipticity: non-circular enough that the angle-basis question
#: bites, modest enough to be the case everything else perturbs.
REFERENCE_EPS = FIXTURES[ACTIVE_FIXTURE]["reference_eps"]

#: A constant added to the whole image. Raw amplitudes should be *invariant*
#: to this --- a constant contributes only to ``m = 0`` --- and so should
#: Bender coefficients, since a constant does not change a radial gradient.
#:
#: What this axis tests for AutoProf is **not** what the spec's revision note
#: predicted, and the pilot is what corrected it. That note measured the FFT
#: expression directly and found a +100 offset on a ring of mean 50 moving
#: ``a4`` by the ring-mean ratio. Through the *pipeline*, it does not: the
#: forced sequence begins with a ``background`` step, which estimates and
#: subtracts the sky before extraction, so an additive offset never reaches
#: the harmonic normalization at all. Measured in the pilot: a +50 offset
#: moved AutoProf's ``b0`` from 99.35842 to 99.35843 and left its native
#: ``b4`` identical to six decimals.
#:
#: Both statements are true at their own level, and the distinction is worth
#: keeping: A1's unit tests cover the *formula's* sensitivity to a background
#: error, and this axis covers the *pipeline's* response to one. A tool whose
#: normalization is offset-sensitive but which subtracts the offset first is
#: not exposed to that sensitivity in normal use.
BACKGROUND_OFFSET = 50.0

#: Raised from 5 after the pilot. At S/N = 30 the ratio scatters with a
#: standard deviation near 0.35, so five realizations give a standard error
#: near 0.16 -- too coarse to pin anything, and a five-sample standard
#: deviation is itself uncertain at the tens-of-percent level. The noisy arms
#: exist to characterize that distribution rather than to calibrate the
#: scale, which the noiseless arms do; 25 makes the scatter estimate itself
#: stable enough to quote.
NOISE_REALIZATIONS = 25

#: Disjoint by construction, and both recorded in every output. The pilot
#: block sets the tolerances; the validation block is judged by them.
PILOT_SEED_BLOCK = 900_000
VALIDATION_SEED_BLOCK = 20_260_822

#: AutoProf's own default, reproduced so that "the default arm" means the
#: default rather than something close to it.
AUTOPROF_DEFAULT_INTERPOLATE_START = 5.0

#: High enough to put every ring on this fixture below the switch. Whether it
#: did is measured, not presumed.
INTERPOLATE_EVERYWHERE = 100.0


def use_fixture(name: str) -> None:
    """Point every module-level campaign constant at one fixture.

    Called once, before anything measures. Rebinding rather than threading
    keeps the diff off the code paths that produced the already-archived
    campaign, which is the point: the first fixture's grid, and therefore its
    fingerprint, must not move.
    """
    global ACTIVE_FIXTURE, FIXTURE, RADII, REFERENCE_EPS, ARCHIVE_PATH, TOLERANCES_PATH

    if name not in FIXTURES:
        raise SystemExit(f"unknown fixture {name!r}; choose from {sorted(FIXTURES)}")
    ACTIVE_FIXTURE = name
    spec = FIXTURES[name]
    FIXTURE = spec["galaxy"]
    RADII = spec["radii"]
    REFERENCE_EPS = spec["reference_eps"]
    ARCHIVE_PATH = HERE / spec["archive"]
    TOLERANCES_PATH = HERE / spec["tolerances"]


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def _case(name, kind, why, **overrides) -> Dict[str, object]:
    """One grid case: the reference settings with named overrides applied.

    Writing every case as a delta from the reference is the point of the
    design --- it is what lets a failure name a factor.
    """
    spec = {
        "name": name,
        "kind": kind,
        "why": why,
        "eps": REFERENCE_EPS,
        "pa_deg": 0.0,
        "isoclip": True,
        "interpolate_start": INTERPOLATE_EVERYWHERE,
        "background_offset": 0.0,
        "snr": None,
        "n_realizations": 1,
        "planted": "four_modes",
        # None leaves AutoProf's assumed 4.0 px in place. Only the second
        # campaign varies it; the first never sets it, so its grid -- and its
        # fingerprint -- are unchanged by this axis existing.
        "set_psf": None,
    }
    unknown = set(overrides) - set(spec)
    if unknown:
        raise ValueError(f"case {name!r} overrides unknown axes: {sorted(unknown)}")
    spec.update(overrides)
    if spec["snr"] is not None and "n_realizations" not in overrides:
        spec["n_realizations"] = NOISE_REALIZATIONS
    return spec


def _extra_cases() -> Dict[str, Dict[str, object]]:
    """Cases that exist only on the second fixture.

    All three are about the *other* half of the interpolation threshold.
    ``rad_interp = ap_iso_interpolate_start * results["psf fwhm"]``, and since
    AutoProf's ``psf`` step assumes 4.0 px rather than measuring it,
    ``ap_set_psf`` is the only way to move the switch radius without touching
    the interpolation setting. Two independent knobs onto one mechanism is a
    considerably stronger test of "sampling mode is the cause" than one knob
    could ever be: if the ratio depends only on the mode, then a ring must not
    care *which* option put it there.
    """
    return {
        "psf_set_8": _case(
            "psf_set_8",
            "one_factor",
            "Double the assumed PSF with the interpolation setting still at 100. The "
            "threshold doubles to 800 px, every ring stays interpolated, and every ratio "
            "must be unchanged. A null result by construction -- which is the point, since "
            "it shows ap_set_psf acts through the threshold and nowhere else.",
            set_psf=8.0,
        ),
        "psf_x_interpolate": _case(
            "psf_x_interpolate",
            "interaction",
            "Default interpolation setting with a doubled PSF, putting the switch at 40 px "
            "instead of 20. Rings change sampling mode without the interpolation setting "
            "changing at all.",
            interpolate_start=AUTOPROF_DEFAULT_INTERPOLATE_START,
            set_psf=8.0,
        ),
        "threshold_matched_control": _case(
            "threshold_matched_control",
            "interaction",
            "Interpolation setting 10 with a halved PSF: different option values from the "
            "default arm in both factors, and the identical 20 px threshold. Every ring must "
            "return the identical ratio. This is the sharpest form of the claim -- only the "
            "product matters, not either factor.",
            interpolate_start=10.0,
            set_psf=2.0,
        ),
    }


def build_grid() -> List[Dict[str, object]]:
    """The designed grid: reference, one-factor-at-a-time, interactions, stress.

    The active fixture's ``extra_cases`` are appended. That list is empty for
    the first fixture, so its grid is exactly the archived one.
    """
    grid = [
        _case(
            "reference",
            "reference",
            "Noiseless, moderate ellipticity, no PA offset, interpolation everywhere. "
            "The cleanest measurement of the scale, and the case every other one perturbs.",
        ),
        _case(
            "single_mode_control",
            "control",
            "The reference ring with only one harmonic planted. Separates a tool's ability to "
            "recover an isolated mode from its ability to recover one mode out of several, which "
            "the pilot showed are different questions for photutils and not for the other two.",
            planted="single_mode",
        ),
        # --- one factor at a time -------------------------------------------------
        _case(
            "interpolate_default",
            "one_factor",
            "AutoProf's own default sampling. The single largest effect on the grid; "
            "with the reference case this is the whole 0.1%-versus-13-25% story.",
            interpolate_start=AUTOPROF_DEFAULT_INTERPOLATE_START,
        ),
        _case(
            "eps_circular",
            "one_factor",
            "At eps ~ 0 polar angle and eccentric anomaly coincide, so the angle-basis "
            "question vanishes and every other difference is isolated.",
            eps=0.02,
        ),
        _case(
            "eps_high",
            "one_factor",
            "Strong ellipticity, where the basis difference bites hardest.",
            eps=0.6,
        ),
        _case(
            "pa_30",
            "one_factor",
            "A non-zero position angle is what makes AutoProf's sky-frame polar basis "
            "differ from the major-axis frame, so it is what tests the n*PA rotation.",
            pa_deg=30.0,
        ),
        _case(
            "isoclip_off",
            "one_factor",
            "Puts AutoProf on the eccentric-anomaly basis, where the same-order rotation "
            "is not a valid conversion. Measured to show the size of the error it would make.",
            isoclip=False,
        ),
        _case(
            "background_offset",
            "one_factor",
            "A constant added to the image. Raw amplitudes must be unchanged; AutoProf's "
            "native coefficients must move by the ring-mean ratio and the reconstruction "
            "must cancel it exactly.",
            background_offset=BACKGROUND_OFFSET,
        ),
        _case(
            "noise_snr100",
            "one_factor",
            "Mild noise, several realizations. One realization measures a realization.",
            snr=100.0,
        ),
        _case(
            "noise_snr30",
            "one_factor",
            "Harder noise, several realizations; sets the scatter the tolerances answer to.",
            snr=30.0,
        ),
        # --- interactions with a reason to exist -----------------------------------
        _case(
            "radius_x_interpolate",
            "interaction",
            "Interpolation start is itself a radius threshold, so a ring's sampling mode is "
            "decided by both factors jointly and neither alone predicts it. This setting puts "
            "the switch between two different pairs of radii than the default does, so one "
            "radius set spans three switch positions across the grid.",
            interpolate_start=8.0,
        ),
        _case(
            "eps_x_basis",
            "interaction",
            "High ellipticity on the eccentric-anomaly basis: the cell where order mixing is "
            "predicted to be worst, and the one that settles whether that path can be converted.",
            eps=0.6,
            isoclip=False,
        ),
        _case(
            "pa_x_resampling",
            "interaction",
            "A non-zero PA without polar resampling. The rotation the conversion applies is "
            "derived for the resampled path only; this is the cell that shows what it costs "
            "to apply it anywhere else.",
            pa_deg=30.0,
            isoclip=False,
        ),
        _case(
            "noise_x_clipping",
            "interaction",
            "Sigma clipping removes samples asymmetrically around a ring, so its effect on a "
            "harmonic is not the effect of noise alone. Clipping off, at the noisier level.",
            snr=30.0,
            isoclip=False,
        ),
        # --- combined stress -------------------------------------------------------
        _case(
            "combined_stress",
            "stress",
            "Every axis at its hardest setting at once. A check that nothing falls over, not "
            "a measurement: with several factors moving together it cannot attribute a "
            "discrepancy, which is exactly why the rest of the grid is one factor at a time.",
            eps=0.6,
            pa_deg=30.0,
            isoclip=False,
            interpolate_start=AUTOPROF_DEFAULT_INTERPOLATE_START,
            background_offset=BACKGROUND_OFFSET,
            snr=30.0,
        ),
    ]
    extra = _extra_cases()
    for name in FIXTURES[ACTIVE_FIXTURE]["extra_cases"]:
        if name not in extra:
            raise SystemExit(f"fixture {ACTIVE_FIXTURE!r} asks for unknown case {name!r}")
        grid.append(extra[name])
    return grid


# ---------------------------------------------------------------------------
# Measuring one case
# ---------------------------------------------------------------------------


def _build_image(spec: Dict[str, object], seed: int | None):
    """Render the fixture for one case, and return it with its analytic truth.

    The truth comes from the *undistorted* analytic profile, so it is
    unaffected by the background offset and by the noise realization --- both
    of which is the point. A constant contributes only to ``m = 0``, and noise
    is zero-mean, so neither changes what the ring signal truly is. Any
    movement in a measured ratio under those two axes is the tool responding,
    not the target moving.
    """
    harmonics = SINGLE_PLANTED_HARMONIC if spec["planted"] == "single_mode" else PLANTED_HARMONICS
    image, meta = create_sersic_image_with_harmonics(
        n=FIXTURE["n"],
        R_e=FIXTURE["R_e"],
        I_e=FIXTURE["I_e"],
        eps=float(spec["eps"]),
        pa=np.radians(float(spec["pa_deg"])),
        shape=FIXTURE["shape"],
        center=FIXTURE["center"],
        harmonics=harmonics,
    )

    noise_sigma = 0.0
    if spec["snr"] is not None:
        image, noise_sigma = add_noise(
            image,
            snr_at_Re=float(spec["snr"]),
            R_e=FIXTURE["R_e"],
            I_e=FIXTURE["I_e"],
            seed=seed,
        )
    image = image + float(spec["background_offset"])
    return image, meta, float(noise_sigma)


def _ring_request(spec: Dict[str, object]) -> List[Dict[str, float]]:
    x0, y0 = FIXTURE["center"]
    return [
        {
            "sma": float(sma),
            "x0": float(x0),
            "y0": float(y0),
            "eps": float(spec["eps"]),
            "pa": float(np.radians(float(spec["pa_deg"]))),
        }
        for sma in RADII
    ]


#: A component is calibratable only if its true amplitude is a meaningful
#: fraction of the largest one on the same ring. The control case plants a
#: single mode, so its other three components are truth-zero up to integration
#: residual; dividing by those would manufacture enormous ratios that look
#: like tool failures but are failures of the question.
RATIO_FLOOR_FRACTION = 1e-3


def _ratio(measured: float, truth: float, scale: float | None = None) -> float:
    """Measured over true, or NaN where the truth is too small to divide by.

    The guard is on the *truth*, not on the measurement, and it is relative to
    the ring's own largest planted amplitude rather than absolute: what makes
    a component uncalibratable is being negligible *next to the signal that is
    there*, which no fixed threshold in intensity units can express.
    """
    if not np.isfinite(measured) or not np.isfinite(truth):
        return float("nan")
    floor = 1e-12 if scale is None else max(1e-12, RATIO_FLOOR_FRACTION * abs(scale))
    if abs(truth) < floor:
        return float("nan")
    return float(measured / truth)


def _conversion_validity(spec: Dict[str, object]) -> tuple[bool, str]:
    """Whether AutoProf's coefficients can be converted for this case.

    The supported conversion requires the polar-resampled path. Changing to
    the eccentric-anomaly basis mixes harmonic *orders*, so no same-order
    two-component rotation can express it --- the arms that run with clipping
    off are measured and reported, but their converted values are not
    calibration.
    """
    if not spec["isoclip"]:
        return False, "eccentric_anomaly_basis_mixes_orders"
    return True, ""


def _compare_rows(rows: Sequence[dict], truths: Dict[float, dict], tool: str) -> List[dict]:
    """Turn one tool's rows into per-ring, per-order ratios against truth."""
    out = []
    for row in rows:
        truth_for_ring = truths[round(float(row["sma"]), 6)]
        entry = {
            "tool": tool,
            "sma": float(row["sma"]),
            "status": row["status"],
            "gradient": float(row["gradient"]),
            "gradient_truth": float(truth_for_ring[ORDERS[0]]["gradient"]),
            "mean_intensity": float(row["mean_intensity"]),
        }
        for key in ("harmonic_sampling_mode", "harmonic_basis", "rad_interp_pix", "autoprof_b0"):
            if key in row:
                entry[key] = row[key]
        # The ring's largest planted amplitude, which sets the floor below
        # which a component is not calibratable.
        ring_scale = max(abs(float(truth_for_ring[order][key])) for order in ORDERS for key in ("s_raw", "c_raw"))
        for order in ORDERS:
            truth = truth_for_ring[order]
            for component, raw_key, truth_key in (
                ("s", f"s{order}_raw", "s_raw"),
                ("c", f"c{order}_raw", "c_raw"),
            ):
                measured = float(row.get(raw_key, float("nan")))
                entry[f"{component}{order}_raw"] = measured
                entry[f"{component}{order}_raw_truth"] = float(truth[truth_key])
                entry[f"{component}{order}_raw_ratio"] = _ratio(measured, float(truth[truth_key]), ring_scale)
            # Rotation-invariant, so it survives a sign or basis error in a
            # single component. Reported *alongside* the components rather
            # than instead of them, because catching that error is a purpose
            # of this grid and the magnitude alone would hide it.
            measured_amp = float(np.hypot(row.get(f"s{order}_raw", np.nan), row.get(f"c{order}_raw", np.nan)))
            truth_amp = float(np.hypot(truth["s_raw"], truth["c_raw"]))
            entry[f"amp{order}_raw"] = measured_amp
            entry[f"amp{order}_raw_truth"] = truth_amp
            entry[f"amp{order}_raw_ratio"] = _ratio(measured_amp, truth_amp, ring_scale)
            bender_scale = max(abs(float(truth_for_ring[o][key])) for o in ORDERS for key in ("a_bender", "b_bender"))
            for component, bender_key, truth_key in (
                ("a", f"a{order}_bender", "a_bender"),
                ("b", f"b{order}_bender", "b_bender"),
            ):
                measured = float(row.get(bender_key, float("nan")))
                entry[f"{component}{order}_bender"] = measured
                entry[f"{component}{order}_bender_truth"] = float(truth[truth_key])
                entry[f"{component}{order}_bender_ratio"] = _ratio(measured, float(truth[truth_key]), bender_scale)
        out.append(entry)
    return out


def measure_case(
    spec: Dict[str, object],
    seed_block: int,
    workspace_root: Path,
    autoprof_python: str | None = None,
) -> Dict[str, object]:
    """Run every tool over one case, across its noise realizations."""
    request = _ring_request(spec)
    realizations = []
    autoprof_provenance = None

    for index in range(int(spec["n_realizations"])):
        seed = None if spec["snr"] is None else seed_block + index
        image, meta, noise_sigma = _build_image(spec, seed)
        truths = {round(float(sma), 6): integrated_harmonic_truth(meta, float(sma), ORDERS) for sma in RADII}

        isoster_rows = measure_isoster_fixed(image, request, ORDERS)
        photutils_rows = measure_photutils_fixed(image, request, ORDERS)
        # Geometry is checked, not trusted: a silently re-fitted ring
        # invalidates every comparison built on it and is invisible in output.
        assert_rings_match_request(isoster_rows, request)
        assert_rings_match_request(photutils_rows, request)

        with tempfile.TemporaryDirectory(dir=str(workspace_root)) as work:
            autoprof_rows, autoprof_provenance = measure_autoprof_fixed(
                image,
                request,
                orders=ORDERS,
                workspace=work,
                isoclip=bool(spec["isoclip"]),
                interpolate_start=float(spec["interpolate_start"]),
                set_psf=None if spec["set_psf"] is None else float(spec["set_psf"]),
                venv_python=autoprof_python,
            )
        assert_rings_match_request(autoprof_rows, request, tolerances={"sma": 1e-6, "eps": 1e-6})

        realizations.append(
            {
                "seed": seed,
                "noise_sigma": noise_sigma,
                "isoster": _compare_rows(isoster_rows, truths, "isoster"),
                "photutils": _compare_rows(photutils_rows, truths, "photutils"),
                "autoprof": _compare_rows(autoprof_rows, truths, "autoprof"),
                "linearized_truth_gap": _linearized_gap(meta, truths),
            }
        )

    valid, reason = _conversion_validity(spec)
    return {
        "spec": spec,
        "harmonic_conversion_valid": valid,
        "harmonic_conversion_reason": reason,
        "autoprof_provenance": autoprof_provenance,
        "realizations": realizations,
        "summary": _summarize_case(realizations),
    }


def _linearized_gap(meta: dict, truths: Dict[float, dict]) -> Dict[str, float]:
    """How far the first-order shortcut is from the integrated truth.

    Reported rather than assumed negligible. With four modes planted at once,
    products between them and the Sersic curvature both feed back into the
    same orders, and this says by how much.
    """
    linear = linearized_harmonic_truth(meta, ORDERS)
    gaps = {}
    for sma, truth_for_ring in truths.items():
        exact_scale = max(abs(float(truth_for_ring[o][key])) for o in ORDERS for key in ("a_bender", "b_bender"))
        for order in ORDERS:
            for component, truth_key in (("a", "a_bender"), ("b", "b_bender")):
                exact = float(truth_for_ring[order][truth_key])
                approx = float(linear[order][truth_key])
                gaps[f"sma{sma:g}_{component}{order}"] = _ratio(approx, exact, exact_scale)
    return gaps


def _spread(values: Sequence[float]) -> Dict[str, object] | None:
    """Median with min/max and, where there are enough points, quartiles."""
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
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4)
        entry["q1"] = round(quartiles[0], 6)
        entry["q3"] = round(quartiles[2], 6)
    if len(ordered) >= 2:
        entry["stdev"] = round(statistics.stdev(ordered), 6)
    return entry


def _summarize_case(realizations: Sequence[dict]) -> Dict[str, object]:
    """Per tool, per ring, per order: the ratio spread across realizations.

    Kept per ring rather than pooled over radii. Raw amplitudes are
    radius-dependent and the interpolation switch splits a radius set into two
    populations with genuinely different behaviour, so a median over all radii
    would average across the very thing the grid is measuring.
    """
    summary: Dict[str, object] = {}
    for tool in ("isoster", "photutils", "autoprof"):
        per_tool: Dict[str, object] = {}
        for ring_index, sma in enumerate(RADII):
            keys = [f"{component}{order}_raw_ratio" for order in ORDERS for component in ("s", "c", "amp")]
            ring_entry: Dict[str, object] = {}
            for key in keys:
                ring_entry[key] = _spread([r[tool][ring_index].get(key, float("nan")) for r in realizations])
            modes = {
                r[tool][ring_index].get("harmonic_sampling_mode")
                for r in realizations
                if "harmonic_sampling_mode" in r[tool][ring_index]
            }
            if modes:
                ring_entry["sampling_modes"] = sorted(m for m in modes if m is not None)
            statuses = {r[tool][ring_index]["status"] for r in realizations}
            ring_entry["statuses"] = sorted(statuses)
            per_tool[f"sma={sma:g}"] = ring_entry
        summary[tool] = per_tool
    return summary


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_state() -> Dict[str, object]:
    """Commit and cleanliness of the working tree.

    ``--archive`` refuses on a dirty tree: an archive is evidence for a claim
    in a paper, and one produced from uncommitted code cannot be reproduced
    from the repository it is committed to.
    """

    def run(*args):
        try:
            return subprocess.run(args, cwd=str(REPO_ROOT), capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_paths": [] if not status else status.splitlines(),
    }


def _fixture_fingerprint() -> str:
    """A digest of everything that defines the measurand.

    If the fixture, the planted modes, the radii or the grid change, the
    archived numbers describe a different experiment. The digest is what lets
    the checker say so instead of comparing numbers across two experiments.
    """
    payload = json.dumps(
        {
            "fixture": {k: v for k, v in FIXTURE.items()},
            "planted": {f"{order}_{kind}": amp for (order, kind), amp in PLANTED_HARMONICS.items()},
            "planted_single": {f"{order}_{kind}": amp for (order, kind), amp in SINGLE_PLANTED_HARMONIC.items()},
            "orders": list(ORDERS),
            "radii": list(RADII),
            "background_offset": BACKGROUND_OFFSET,
            "noise_realizations": NOISE_REALIZATIONS,
            # ``why`` is prose and does not define the measurement. ``set_psf``
            # is dropped *when None* because None means the option is never
            # passed to AutoProf at all: a campaign that does not set the PSF
            # is the same experiment as one with no such axis, and the first
            # campaign -- already archived and gated -- must keep the exact
            # fingerprint it was archived under. Any non-None value is kept,
            # because then it genuinely is part of the experiment.
            "grid": [
                {k: v for k, v in case.items() if k != "why" and not (k == "set_psf" and v is None)}
                for case in build_grid()
            ],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _environment() -> Dict[str, object]:
    import photutils

    import isoster

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "photutils": photutils.__version__,
        "isoster": getattr(isoster, "__version__", "unknown"),
        "git": _git_state(),
        "fixture_fingerprint": _fixture_fingerprint(),
    }


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------


def run_grid(
    mode: str,
    only: str | None = None,
    autoprof_python: str | None = None,
) -> Dict[str, object]:
    fixture_spec = FIXTURES[ACTIVE_FIXTURE]
    seed_block = fixture_spec["pilot_seed_block"] if mode == "pilot" else fixture_spec["validation_seed_block"]
    grid = build_grid()
    if only:
        grid = [case for case in grid if case["name"] == only]
        if not grid:
            raise SystemExit(f"no such case: {only}")

    output_root = Path(resolve_output_directory("benchmark_harmonic_scale"))
    cases = []
    with tempfile.TemporaryDirectory(prefix="harmonic-scale-") as workspace_root:
        for index, spec in enumerate(grid, start=1):
            print(f"[harmonic-scale] {index}/{len(grid)} {spec['name']} ...", flush=True)
            cases.append(measure_case(spec, seed_block, Path(workspace_root), autoprof_python))

    return {
        "mode": mode,
        "fixture": ACTIVE_FIXTURE,
        "seed_block": seed_block,
        "seed_blocks": {
            "pilot": fixture_spec["pilot_seed_block"],
            "validation": fixture_spec["validation_seed_block"],
        },
        "environment": _environment(),
        "cases": cases,
        "output_directory": str(output_root),
    }


#: Multiplier applied to the pilot's own run-to-run scatter to get a
#: tolerance. Three is wide enough that ordinary variation between two seed
#: blocks does not fail the gate, and narrow enough that a real change in the
#: answer does. It is applied uniformly rather than tuned per claim, because a
#: per-claim margin chosen to make each number pass is the failure this whole
#: freeze-before-validate procedure exists to prevent.
TOLERANCE_SAFETY_FACTOR = 3.0

#: Floor for a claim the pilot measured as deterministic. Noiseless arms
#: repeat exactly within a machine, but a tolerance of zero would gate on bit
#: reproducibility across platforms and BLAS builds, which is not the claim.
DETERMINISTIC_FLOOR_PCT = 0.05

#: Floor for the invariance claims, which the pilot measured at the level of
#: floating-point round-off (1e-15 for isoster and AutoProf, 2e-7 for
#: photutils, whose leastsq floor A1 already characterised). Set above
#: photutils' floor so the gate tests invariance rather than optimizer noise.
INVARIANCE_FLOOR = 1e-5


def _claim_scatter(pilot: Dict[str, object]) -> Dict[str, float]:
    """The run-to-run scatter available for each claim, from the pilot itself.

    Noiseless claims have no realizations to scatter over, so their tolerance
    comes from the floors above. Noise-bearing claims do: the quantity that
    will move between seed blocks is a median over realizations, whose
    standard error is the observed standard deviation over the square root of
    the realization count.
    """
    scatter: Dict[str, float] = {}
    for case in pilot["cases"]:
        if case["spec"]["snr"] is None:
            continue
        count = max(1, int(case["spec"]["n_realizations"]))
        for tool, per_tool in case["summary"].items():
            deviations = [
                float(entry["stdev"])
                for ring in per_tool.values()
                for entry in ring.values()
                if isinstance(entry, dict) and "stdev" in entry
            ]
            if not deviations:
                continue
            label = "100" if float(case["spec"]["snr"]) >= 100 else "30"
            key = f"noise_scatter_snr{label}_{tool}"
            # The claim *is* a standard deviation, and a standard deviation
            # estimated from n samples is itself uncertain by roughly
            # sigma/sqrt(2n). That, not sigma/sqrt(n), is what moves between
            # two seed blocks here.
            scatter[key] = max(scatter.get(key, 0.0), max(deviations) / np.sqrt(2.0 * count))
    return scatter


def freeze_tolerances(pilot: Dict[str, object]) -> Dict[str, object]:
    """Turn a pilot run into the tolerance file the validation run is judged by.

    Written *before* the validation run and committed, so that the numbers
    which decide pass or fail existed before the numbers they decide about.
    """
    from benchmarks.harmonic_scale.claims import extract_claims

    claims = extract_claims(pilot)
    scatter = _claim_scatter(pilot)

    frozen: Dict[str, object] = {}
    for name, value in claims.items():
        if name in scatter:
            tolerance = TOLERANCE_SAFETY_FACTOR * scatter[name]
            basis = "scatter"
        elif name.startswith("background_invariance") or name == "mode_matched_spread":
            tolerance = INVARIANCE_FLOOR
            basis = "invariance_floor"
        else:
            # A percentage claim off the deterministic arms. The tolerance is
            # a floor plus a small relative allowance, because a 60% claim and
            # a 0.5% claim cannot share an absolute margin.
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
            "fixture": pilot.get("fixture", "sersic_n2_compact"),
            "seed_block": pilot["seed_block"],
            "commit": pilot["environment"]["git"]["commit"],
            "fixture_fingerprint": pilot["environment"]["fixture_fingerprint"],
        },
        "policy": {
            "safety_factor": TOLERANCE_SAFETY_FACTOR,
            "deterministic_floor_pct": DETERMINISTIC_FLOOR_PCT,
            "invariance_floor": INVARIANCE_FLOOR,
            "note": (
                "Tolerances are derived from the pilot's measured scatter by one uniform "
                "rule, not chosen per claim. The validation run draws from a disjoint seed "
                "block and is judged against these values."
            ),
        },
        "claims": frozen,
    }


#: Noise cases whose per-realization rows are summarized rather than stored in
#: the committed archive. Above this count the raw rows dominate the file --- 25
#: realizations is about 0.5 MB per case against 13 kB for its summary --- and
#: nothing reads them: every claim is computed from the summary, which already
#: carries n, median, min, max, quartiles and standard deviation.
#:
#: They are dropped rather than kept because they are *regenerable*, not
#: irreplaceable: the run is deterministic given the recorded seed block, so
#: any realization can be reproduced exactly by re-running the same mode. The
#: full untrimmed run is always written to ``outputs/`` regardless.
ARCHIVE_REALIZATION_LIMIT = 1


def _archive_payload(results: Dict[str, object]) -> Dict[str, object]:
    """The committed archive: everything except regenerable bulk.

    What is dropped is stated in the file rather than silently omitted --- a
    reader who finds one realization where the run made twenty-five should be
    able to tell that from the archive itself, not from this source file.
    """
    import copy

    trimmed = copy.deepcopy(results)
    dropped = 0
    for case in trimmed["cases"]:
        realizations = case["realizations"]
        if len(realizations) <= ARCHIVE_REALIZATION_LIMIT:
            case["realizations_stored"] = len(realizations)
            case["realizations_run"] = len(realizations)
            continue
        dropped += len(realizations) - ARCHIVE_REALIZATION_LIMIT
        case["realizations_run"] = len(realizations)
        case["realizations_stored"] = ARCHIVE_REALIZATION_LIMIT
        case["realizations"] = realizations[:ARCHIVE_REALIZATION_LIMIT]

    trimmed["archive_note"] = (
        f"Per-realization rows beyond the first {ARCHIVE_REALIZATION_LIMIT} are summarized "
        f"rather than stored; {dropped} were dropped. Every claim is computed from the "
        "'summary' block, which is complete. The dropped rows are regenerable exactly: "
        "the run is deterministic given the recorded seed block. The full untrimmed run "
        "is in outputs/benchmark_harmonic_scale/."
    )
    return trimmed


def _write(results: Dict[str, object], archive: bool) -> Path:
    output_root = Path(resolve_output_directory("benchmark_harmonic_scale"))
    out_path = output_root / f"{results.get('fixture', ACTIVE_FIXTURE)}_{results['mode']}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[harmonic-scale] wrote {out_path}")
    if archive:
        ARCHIVE_PATH.write_text(json.dumps(_archive_payload(results), indent=2, default=str))
        size_mb = ARCHIVE_PATH.stat().st_size / 1e6
        print(f"[harmonic-scale] archived {ARCHIVE_PATH} ({size_mb:.2f} MB)")
    return out_path


def _describe_dirty(git: Dict[str, object], when: str) -> List[str]:
    """State whether a tree was clean, naming what was not."""
    if git.get("dirty") is None:
        return [f"could not determine whether the working tree was clean {when}"]
    if not git["dirty"]:
        return []
    paths = list(git.get("dirty_paths") or [])
    return [
        f"working tree was dirty {when} ("
        + ", ".join(paths[:5])
        + ("..." if len(paths) > 5 else "")
        + "); an archive that cannot be reproduced from a commit is not evidence"
    ]


def _refuse_archive(results: Dict[str, object], only: str | None) -> List[str]:
    """Every reason this run must not become the committed archive.

    The tree is checked at **two** times, and both must be clean. The state
    recorded in the run says whether the numbers were produced from committed
    code. The state *now* says whether the file about to be written can be
    committed alongside it. For a run that archives immediately these are the
    same check, but ``--rearchive`` can promote a months-old run from a tree
    that has since changed --- and reading only the recorded state there would
    let a dirty tree through, which is exactly the hole this once had.
    """
    problems = []
    if results["mode"] != "validation":
        problems.append(f"mode is {results['mode']!r}; only a validation run may be archived")
    if only:
        problems.append(f"--only {only} runs one case; the archive must contain the whole grid")
    problems.extend(_describe_dirty(results["environment"]["git"], "when the run was made"))
    problems.extend(_describe_dirty(_git_state(), "now"))
    if not TOLERANCES_PATH.exists():
        problems.append(
            f"{TOLERANCES_PATH.name} does not exist; tolerances must be frozen from the "
            "pilot before the validation run, not chosen after seeing its result"
        )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("pilot", "validation"),
        default="pilot",
        help=(
            "pilot draws from the pilot seed block and is what tolerances are set from; "
            "validation draws from a disjoint block and is judged by them."
        ),
    )
    parser.add_argument(
        "--fixture",
        choices=sorted(FIXTURES),
        default="sersic_n2_compact",
        help=(
            "Which galaxy campaign to run. Each has its own archive, its own frozen "
            "tolerances and its own seed blocks; they are separate measurements, not "
            "one grid with an extra axis."
        ),
    )
    parser.add_argument("--only", help="Run a single named case (never archivable).")
    parser.add_argument(
        "--autoprof-python",
        help="Interpreter for the AutoProf venv; defaults to benchmarks/autoprof_env.py's resolution.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help=(
            "Promote a validation run to reference_harmonic_scale.json. Requires the whole "
            "grid, a clean working tree, and tolerances already frozen."
        ),
    )
    parser.add_argument(
        "--rearchive",
        type=Path,
        metavar="VALIDATION_JSON",
        help=(
            "Re-write the committed archive from a completed validation run without "
            "re-measuring anything. For changes to what the archive *stores*, never to "
            "what it says: the numbers come from the run, and the same refusals apply."
        ),
    )
    parser.add_argument(
        "--freeze-tolerances",
        type=Path,
        metavar="PILOT_JSON",
        help=(
            "Derive tolerances from a completed pilot run and write "
            "frozen_tolerances.json. Runs no measurements. Do this, and commit the "
            "result, before the validation run it will judge."
        ),
    )
    args = parser.parse_args()
    use_fixture(args.fixture)

    if args.rearchive is not None:
        results = json.loads(args.rearchive.read_text())
        # The run must still describe the experiment this code defines, or
        # re-archiving would quietly promote numbers from a different grid.
        archived_fingerprint = results.get("environment", {}).get("fixture_fingerprint")
        if archived_fingerprint != _fixture_fingerprint():
            raise SystemExit(
                f"{args.rearchive} was produced by a different grid "
                f"(fingerprint {archived_fingerprint}); re-run --mode validation --archive"
            )
        problems = _refuse_archive(results, None)
        if problems:
            raise SystemExit("refusing to overwrite the committed archive:\n  - " + "\n  - ".join(problems))
        _write(results, archive=True)
        return

    if args.freeze_tolerances is not None:
        pilot = json.loads(args.freeze_tolerances.read_text())
        if pilot.get("mode") != "pilot":
            raise SystemExit(
                f"{args.freeze_tolerances} is a {pilot.get('mode')!r} run; tolerances must be "
                "frozen from a pilot, or they are fitted to the data they will judge"
            )
        # Each campaign is its own measurement, so its tolerances must come
        # from its own pilot. Freezing one fixture's tolerances from another's
        # scatter would judge a galaxy by a different galaxy's noise.
        if pilot.get("fixture", "sersic_n2_compact") != ACTIVE_FIXTURE:
            raise SystemExit(
                f"{args.freeze_tolerances} is a {pilot.get('fixture')!r} pilot but --fixture "
                f"is {ACTIVE_FIXTURE!r}; tolerances must be frozen from the same campaign"
            )
        frozen = freeze_tolerances(pilot)
        TOLERANCES_PATH.write_text(json.dumps(frozen, indent=2))
        print(f"[harmonic-scale] froze {len(frozen['claims'])} tolerances to {TOLERANCES_PATH}")
        for name, entry in sorted(frozen["claims"].items()):
            print(f"   {name:38s} {entry['pilot_value']:>12.6g} +- {entry['tolerance']:<10.4g} ({entry['basis']})")
        return

    results = run_grid(args.mode, only=args.only, autoprof_python=args.autoprof_python)

    if args.archive:
        problems = _refuse_archive(results, args.only)
        if problems:
            _write(results, archive=False)
            raise SystemExit("refusing to overwrite the committed archive:\n  - " + "\n  - ".join(problems))
    _write(results, archive=args.archive)


if __name__ == "__main__":
    main()
