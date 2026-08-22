"""The quantities Part A actually claims, extracted from a grid run.

One module, two consumers. ``run_harmonic_scale.py --freeze-tolerances`` reads
a *pilot* run through here to set tolerances; ``check_harmonic_scale.py`` reads
the *validation* archive through here to test them. Sharing the extraction is
the point: if the freeze and the check each computed "the clean-configuration
agreement" their own way, the two definitions could drift apart and the
tolerance would stop describing the thing being judged.

Every claim is a single number with a stated definition. A claim that cannot
be reduced to one number is not on this list --- it belongs in the archive as
a diagnostic, where a reader can look at it, rather than in a gate that will
be read as pass or fail.

The claims
----------
``clean_agreement_pct_<tool>`` and ``clean_agreement_pct_<tool>_sma<r>``
    Worst ``|ratio - 1|``, in percent, over every noiseless case that runs on
    the supported conversion path (clipping on) at moderate ellipticity, and
    over only those rings AutoProf actually interpolated. This is the headline
    "the three tools measure the same signal to X%" number. Reported per
    radius as well as pooled, because the pooled worst case is set by the
    smallest ring, where the fixture itself is marginally resolved --- quoting
    it alone would present pixelation as the tools' agreement floor.

``nearest_pixel_excess_pct_sma<r>``
    AutoProf's ``|ratio - 1|`` at radius ``r`` on rings it sampled by rounding
    to the nearest pixel. Radius-resolved because the excess varies strongly
    and *non-monotonically* with radius: how badly a ring aliases depends on
    where its sample positions happen to fall on the integer grid, so a nearly
    clean ring can sit between two badly aliased ones. Pooling over radius
    would average a structured effect into a meaningless single number.

``mode_matched_spread``
    The largest spread, across grid cases, of AutoProf's ratio at a fixed
    radius among cases whose sampling mode at that radius agreed. If sampling
    mode is what drives the excess, this is ~0 while the cases themselves
    differ by tens of percent. It is the sharpest statement the grid makes.

``background_invariance_<tool>``
    Worst ``|ratio(offset) - ratio(no offset)|``. Raw amplitudes must not
    respond to a constant added to the image.

``eccentric_anomaly_error_pct_eps<e>``
    AutoProf's worst ``|ratio - 1|`` with clipping off, where the same-order
    rotation is not a valid conversion. Quantifies what applying it anyway
    would cost.

``leakage_pct_<tool>``
    Worst ``|ratio - 1|`` on ``c3`` in the single-mode control minus the same
    quantity in the four-mode reference, both noiseless and otherwise
    identical. Isolates a tool's response to *other* modes being present from
    its response to the one being measured.

``noise_scatter_snr<s>_<tool>``
    Worst standard deviation of the ratio across realizations. Characterizes a
    distribution rather than calibrating a scale, which is what the noisy arms
    are for.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

TOOLS = ("isoster", "photutils", "autoprof")

#: The four ratio keys a claim is allowed to look at. Amplitude magnitude is
#: deliberately excluded: it is rotation-invariant, so it would hide exactly
#: the sign and basis errors several of these claims exist to catch.
RATIO_KEYS = ("s3_raw_ratio", "c3_raw_ratio", "s4_raw_ratio", "c4_raw_ratio")

#: Cases that define "clean": noiseless, on the supported conversion path, and
#: not deliberately stressed. High ellipticity is excluded because the pilot
#: showed it is an estimator-sampling regime of its own, not a scale question.
CLEAN_CASES = ("reference", "eps_circular", "pa_30", "background_offset")


def _case_map(results: Dict[str, object]) -> Dict[str, dict]:
    return {case["spec"]["name"]: case for case in results["cases"]}


def _radii(results: Dict[str, object]) -> List[float]:
    reference = results["cases"][0]
    return [float(key.split("=")[1]) for key in reference["summary"]["isoster"]]


def _ring_key(sma: float) -> str:
    return f"sma={sma:g}"


def _medians(case: dict, tool: str, sma: float) -> List[float]:
    """The median ratio, across realizations, for each component at one ring."""
    entry = case["summary"][tool][_ring_key(sma)]
    out = []
    for key in RATIO_KEYS:
        summary = entry.get(key)
        if summary is not None:
            out.append(float(summary["median"]))
    return out


def _interpolated(case: dict, ring_index: int) -> bool | None:
    """Whether AutoProf interpolated this ring, as observed during the run."""
    modes = case["autoprof_provenance"]["sampling_mode"]["per_ring_interpolated"]
    if not case["autoprof_provenance"]["sampling_mode"]["attribution_ok"]:
        return None
    return bool(modes[ring_index])


def _worst_deviation(values: Sequence[float]) -> float:
    """Largest departure from unity, in percent."""
    finite = [abs(v - 1.0) for v in values if np.isfinite(v)]
    return 100.0 * max(finite) if finite else float("nan")


def extract_claims(results: Dict[str, object]) -> Dict[str, float]:
    """Reduce a whole grid run to the numbers Part A stands behind."""
    cases = _case_map(results)
    radii = _radii(results)
    claims: Dict[str, float] = {}

    # --- clean-configuration agreement ------------------------------------
    # Reported per radius as well as worst-case. The worst case alone would
    # hide the structure: agreement degrades at small radii because the ring
    # is only a few pixels across a quarter of an m=4 cycle, which is
    # pixelation of the fixture rather than disagreement between the tools.
    # Quoting one number would make that look like the tools' error floor.
    for tool in TOOLS:
        values: List[float] = []
        for index, sma in enumerate(radii):
            per_radius: List[float] = []
            for name in CLEAN_CASES:
                case = cases.get(name)
                if case is None or _interpolated(case, index) is not True:
                    continue
                per_radius.extend(_medians(case, tool, sma))
            if per_radius:
                claims[f"clean_agreement_pct_{tool}_sma{sma:g}"] = _worst_deviation(per_radius)
                values.extend(per_radius)
        claims[f"clean_agreement_pct_{tool}"] = _worst_deviation(values)

    # --- the nearest-pixel excess, radius by radius ------------------------
    for index, sma in enumerate(radii):
        values = []
        for case in cases.values():
            if case["spec"]["snr"] is not None or _interpolated(case, index) is not False:
                continue
            values.extend(_medians(case, "autoprof", sma))
        if values:
            claims[f"nearest_pixel_excess_pct_sma{sma:g}"] = _worst_deviation(values)

    # --- sampling mode, not setting or radius, is what moves the ratio -----
    # Among noiseless cases that agree on everything except the interpolation
    # setting, group by the mode a ring actually got and ask how much the
    # ratio varies inside a group. Near zero means the mode is the cause.
    spreads = []
    interpolation_only = [
        name
        for name, case in cases.items()
        if case["spec"]["snr"] is None
        and case["spec"]["eps"] == cases["reference"]["spec"]["eps"]
        and case["spec"]["pa_deg"] == cases["reference"]["spec"]["pa_deg"]
        and case["spec"]["isoclip"] == cases["reference"]["spec"]["isoclip"]
        and case["spec"]["background_offset"] == cases["reference"]["spec"]["background_offset"]
        and case["spec"]["planted"] == cases["reference"]["spec"]["planted"]
    ]
    for index, sma in enumerate(radii):
        for wanted_mode in (True, False):
            grouped = [
                _medians(cases[name], "autoprof", sma)
                for name in interpolation_only
                if _interpolated(cases[name], index) is wanted_mode
            ]
            if len(grouped) < 2:
                continue
            for component in range(len(grouped[0])):
                column = [group[component] for group in grouped]
                spreads.append(max(column) - min(column))
    claims["mode_matched_spread"] = max(spreads) if spreads else float("nan")

    # --- invariance to a constant added to the image -----------------------
    reference, offset = cases.get("reference"), cases.get("background_offset")
    if reference and offset:
        for tool in TOOLS:
            differences = []
            for sma in radii:
                for before, after in zip(_medians(reference, tool, sma), _medians(offset, tool, sma)):
                    if np.isfinite(before) and np.isfinite(after):
                        differences.append(abs(after - before))
            claims[f"background_invariance_{tool}"] = max(differences) if differences else float("nan")

    # --- the unconvertible basis -------------------------------------------
    for name, label in (("isoclip_off", "0.3"), ("eps_x_basis", "0.6")):
        case = cases.get(name)
        if case is None:
            continue
        values = [value for sma in radii for value in _medians(case, "autoprof", sma)]
        claims[f"eccentric_anomaly_error_pct_eps{label}"] = _worst_deviation(values)

    # --- one mode alone versus one mode among four -------------------------
    control, multi = cases.get("single_mode_control"), cases.get("reference")
    if control and multi:
        for tool in TOOLS:
            alone, among = [], []
            for sma in radii:
                # c3 is the mode the control plants; index 1 of RATIO_KEYS.
                entry_alone = control["summary"][tool][_ring_key(sma)].get("c3_raw_ratio")
                entry_among = multi["summary"][tool][_ring_key(sma)].get("c3_raw_ratio")
                if entry_alone:
                    alone.append(float(entry_alone["median"]))
                if entry_among:
                    among.append(float(entry_among["median"]))
            claims[f"leakage_pct_{tool}"] = _worst_deviation(among) - _worst_deviation(alone)

    # --- how far the noisy arms scatter ------------------------------------
    for name, label in (("noise_snr100", "100"), ("noise_snr30", "30")):
        case = cases.get(name)
        if case is None:
            continue
        for tool in TOOLS:
            deviations = []
            for sma in radii:
                entry = case["summary"][tool][_ring_key(sma)]
                for key in RATIO_KEYS:
                    summary = entry.get(key)
                    if summary and "stdev" in summary:
                        deviations.append(float(summary["stdev"]))
            if deviations:
                claims[f"noise_scatter_snr{label}_{tool}"] = max(deviations)

    return claims


def structural_problems(results: Dict[str, object]) -> List[str]:
    """Ways an archive can be wrong that no tolerance would catch.

    These are not measurements; they are the preconditions under which the
    measurements mean anything. A grid whose sampling modes could not be
    attributed, or whose rings silently fell into isophotal-band sampling,
    produces numbers that look fine and describe something else.
    """
    problems: List[str] = []
    cases = _case_map(results)
    radii = _radii(results)

    for name, case in cases.items():
        sampling = case["autoprof_provenance"]["sampling_mode"]
        if not sampling["attribution_ok"]:
            problems.append(f"{name}: sampling mode not attributable ({sampling['attribution_note']})")
        if not sampling["all_rings_line_sampled"]:
            problems.append(
                f"{name}: {sampling['band_sampling_calls']} ring(s) fell into isophotal-band "
                "sampling, which measures a different quantity than line sampling"
            )
        # The conversion is valid exactly where the polar-resampled path ran.
        expected_valid = bool(case["spec"]["isoclip"])
        if bool(case["harmonic_conversion_valid"]) != expected_valid:
            problems.append(
                f"{name}: harmonic_conversion_valid is {case['harmonic_conversion_valid']} "
                f"but isoclip is {case['spec']['isoclip']}"
            )
        if not expected_valid and not case["harmonic_conversion_reason"]:
            problems.append(f"{name}: conversion marked invalid with no reason recorded")
        for tool in TOOLS:
            for sma in radii:
                statuses = case["summary"][tool][_ring_key(sma)]["statuses"]
                bad = [s for s in statuses if s != "measured"]
                if bad:
                    problems.append(f"{name}: {tool} at sma={sma:g} reported {sorted(set(bad))}")

    if "reference" not in cases:
        problems.append("no reference case in the archive; every other case is a delta from it")
    return problems
