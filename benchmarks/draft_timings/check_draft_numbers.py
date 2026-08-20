"""Check that every timing quoted in the publication draft matches the archive.

The draft's timing tables and summary sentences are transcribed by hand from
``reference_timings.json``, and a transcription that drifts is worse than no
number at all — it still looks measured. This script rebuilds each quoted
*table row* and each quoted *sentence* from the archive and asserts the draft
contains it verbatim.

Whole strings, not fragments. An earlier version of this checker looked for the
bare substrings ``23`` and ``25`` to validate an interquartile range; those two
digits occur throughout the prose, so the check passed against a draft whose
numbers had been deliberately corrupted. Anything short of a complete row or a
complete clause can pass by accident, so nothing here matches less than one.

Run ``--self-test`` to confirm the checks actually fail on mutated text; a
checker nobody has tried to fool is not evidence.

Usage::

    uv run python benchmarks/draft_timings/check_draft_numbers.py
    uv run python benchmarks/draft_timings/check_draft_numbers.py --self-test
    uv run python benchmarks/draft_timings/check_draft_numbers.py --results outputs/benchmark_draft_timings/timings.json

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = REPO_ROOT / "docs" / "publication" / "draft"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "reference_timings.json"

SPEED = "section-1.3-why-fast.md"
EA = "section-1.4.1-eccentric-anomaly-isofit.md"
ROADMAP = "section-1.6-limitations-roadmap.md"

Check = Tuple[str, str, str]  # (name, filename, expected verbatim text)


def _squash(text: str) -> str:
    """Collapse runs of whitespace so markdown line-wrapping cannot hide a match.

    The claims being checked are whole clauses and whole table rows, and a
    clause may be wrapped across lines in the source. Normalising both sides
    keeps the check on the *content* without weakening it to a fragment match.
    """
    return re.sub(r"\s+", " ", text)


def _load_draft() -> Dict[str, str]:
    if not DRAFT_DIR.is_dir():
        raise SystemExit(f"draft directory not found: {DRAFT_DIR} (it is gitignored; nothing to check)")
    return {path.name: path.read_text() for path in DRAFT_DIR.glob("*.md")}


def _thousands(value: int) -> str:
    """Render an integer the way the draft's tables do (space-grouped)."""
    return f"{value:,}".replace(",", " ")


def _pct_saving(runtime_ratio: float) -> int:
    """A runtime ratio expressed as the whole-percent saving the draft quotes."""
    return round(100 * (1 - runtime_ratio))


def _pct_saving_precise(runtime_ratio: float) -> str:
    """The same saving to one decimal place.

    The headline median is quoted to the whole percent, which is the precision
    its between-run reproducibility supports. The interquartile range needs one
    more digit: this archive's quartiles are close enough together that rounding
    both to the whole percent would print the width of the interval as zero.
    """
    return f"{100 * (1 - runtime_ratio):.1f}"


def _pct_surcharge(runtime_ratio: float) -> int:
    """What the *uncached* path costs extra, as a whole percent.

    Not the saving with its sign flipped. The saving is measured against the
    uncached path and the surcharge against the cached one, so they have
    different denominators: a ratio of 0.762 is a 24% saving but a 31%
    surcharge. Quoting the saving where the sentence says "surcharge" is an
    easy error to make and a hard one to see, which is why it is derived here.
    """
    return round(100 * (1.0 / runtime_ratio - 1.0))


def build_checks(results: Dict[str, object]) -> List[Check]:
    """Rebuild every quoted row and sentence from the archive."""
    checks: List[Check] = []
    across = results.get("across_sessions", {})

    # --- Section 1.4.1: the eccentric-anomaly / ISOFIT table ----------------
    ea = results.get("ea_isofit")
    if ea:
        labels = [
            ("default ($\\phi$, post-hoc)", "default_phi_posthoc"),
            ("eccentric anomaly only", "eccentric_anomaly_only"),
            ("ISOFIT `in_loop` only", "isofit_in_loop_only"),
            ("eccentric anomaly + ISOFIT", "eccentric_anomaly_plus_isofit"),
        ]
        for label, key in labels:
            entry = ea[key]
            row = f"| {label} | {entry['median_ms']:.1f} ms | {entry['ratio_to_default']['median']:.2f} |"
            checks.append((f"EA table row: {label}", EA, row))

        combined = across.get("ea_isofit_ratio")
        if combined:
            checks.append(
                (
                    "EA combined ratio, median and IQR",
                    EA,
                    f"median of ${combined['median']:.2f}$ and an interquartile range of "
                    f"${combined['q1']:.2f}$–${combined['q3']:.2f}$",
                )
            )
            checks.append(("EA session count", EA, f"Across {_words(combined['n_sessions'])} sessions"))

    # --- Section 1.6.5: the joint-solve band table --------------------------
    bands = results.get("joint_solve_bands")
    if isinstance(bands, dict) and "ols" in bands and "wls" in bands:
        ols, wls = bands["ols"], bands["wls"]
        for n_bands in (3, 6, 12, 24):
            ols_entry, wls_entry = ols[f"geometry_B={n_bands}"], wls[f"geometry_B={n_bands}"]
            ols_ms = _ms(ols_entry["median_ms"])
            wls_ms = _ms(wls_entry["median_ms"])
            if n_bands == 3:
                row = f"| 3 | {ols_ms} | — | {wls_ms} | — |"
            else:
                ols_ratio = ols_entry["ratio_to_previous"]["median"]
                wls_ratio = wls_entry["ratio_to_previous"]["median"]
                row = f"| {n_bands} | {ols_ms} | {ols_ratio:.1f}× | {wls_ms} | {wls_ratio:.1f}× |"
            checks.append((f"band table row: B={n_bands}", ROADMAP, row))

        amplitude = ols["amplitude_B=24"]["ratio_to_previous"]["median"]
        checks.append(("legacy amplitude-form ratio", ROADMAP, f"about ${amplitude:.1f}\\times$ from"))

        b_ratio = across.get("joint_geometry_B24_over_B12")
        if b_ratio:
            checks.append(("band ratio session count", ROADMAP, f"over {_words(b_ratio['n_sessions'])} sessions its"))
            checks.append(
                (
                    "band ratio, median and IQR",
                    ROADMAP,
                    f"median is ${b_ratio['median']:.1f}$ with an interquartile range of "
                    f"${b_ratio['q1']:.1f}$–${b_ratio['q3']:.1f}$",
                )
            )

    # --- Section 1.6.5: the maxsma table and its ratio summary --------------
    maxsma = results.get("maxsma")
    if maxsma:
        for limit, key in ((100, "maxsma=100"), (200, "maxsma=200"), (400, "maxsma=400")):
            entry = maxsma[key]
            row = (
                f"| {limit} px | {entry['n_isophotes']} | {entry['sampling_calls']} | "
                f"{_thousands(entry['sampled_points'])} | {entry['median_ms']:.1f} ms |"
            )
            checks.append((f"maxsma table row: {limit} px", ROADMAP, row))

        combined = across.get("maxsma_ratio_100_to_200"), across.get("maxsma_ratio_200_to_400")
        if all(combined):
            typical = (combined[0]["median"] + combined[1]["median"]) / 2
            checks.append(
                ("maxsma per-doubling summary", ROADMAP, f"rising by about ${typical:.1f}\\times$ per doubling")
            )

        paired = across.get("maxsma_paired_difference")
        if paired:
            checks.append(
                (
                    "maxsma paired difference, sign count and median",
                    ROADMAP,
                    f"positive in {paired['n_positive']} of {paired['n_sessions']} sessions with a median of "
                    f"${paired['median']:+.2f}$",
                )
            )
            # The count of negative sessions is prose ("the two negative
            # sessions"), so it has to be derived too — otherwise a re-archive
            # that changes the sign split leaves a stale word behind a fresh
            # number. A clean sweep needs different wording entirely: there is
            # no "worst negative session" to describe when there are none, and
            # an earlier version of this check happily generated "the 0
            # negative sessions are small, the worse of them by +0.04".
            n_negative = int(paired["n_sessions"]) - int(paired["n_positive"])
            if n_negative:
                clause = (
                    f"the {_words(n_negative)} negative sessions are small, the worse of them by ${paired['min']:+.2f}$"
                )
            else:
                clause = f"the smallest of them ${paired['min']:+.2f}$"
            checks.append(("maxsma paired difference, negative sessions", ROADMAP, clause))

        r1, r2 = across.get("maxsma_ratio_100_to_200"), across.get("maxsma_ratio_200_to_400")
        if r1 and r2:
            # The retracted arithmetic is quoted in the draft as a correction and
            # must stay correct, so it is derived here rather than transcribed.
            checks.append(
                (
                    "retracted median-difference arithmetic",
                    ROADMAP,
                    f"${r2['median'] - r1['median']:.3f}$, exceeds both IQR widths, "
                    f"${r1['q3'] - r1['q1']:.3f}$ and ${r2['q3'] - r2['q1']:.3f}$",
                )
            )

        for label, key in (("100→200", "maxsma_ratio_100_to_200"), ("200→400", "maxsma_ratio_200_to_400")):
            summary = across.get(key)
            if summary:
                checks.append(
                    (
                        f"maxsma step ratio {label}, median and IQR",
                        ROADMAP,
                        f"${summary['median']:.2f}$ (IQR ${summary['q1']:.2f}$–${summary['q3']:.2f}$)",
                    )
                )

    # --- Section 1.3.2: the lazy-gradient result ----------------------------
    lazy = results.get("lazy_gradient")
    if lazy:
        checks.append(
            (
                "lazy-gradient evaluation counts",
                SPEED,
                f"**{lazy['lazy']['gradient_evaluations']} gradient evaluations against "
                f"{lazy['classical']['gradient_evaluations']}**",
            )
        )
        ratio = across.get("lazy_gradient_runtime_ratio")
        if ratio:
            checks.append(
                (
                    "lazy-gradient saving, median",
                    SPEED,
                    f"**median of ${_pct_saving(ratio['median'])}\\%$** across",
                )
            )
            checks.append(
                (
                    "lazy-gradient saving, IQR",
                    SPEED,
                    f"interquartile range of ${_pct_saving_precise(ratio['q3'])}\\%$ to "
                    f"${_pct_saving_precise(ratio['q1'])}\\%$",
                )
            )
            checks.append(("lazy-gradient session count", SPEED, f"{_words(ratio['n_sessions'])} sessions"))
            # Both sections advise setting use_lazy_gradient=False in the LSB
            # regime and name the price. It is the reciprocal of the saving,
            # and both had been quoting the saving.
            surcharge = _pct_surcharge(ratio["median"])
            checks.append(
                (
                    "exact-path surcharge (speed section)",
                    SPEED,
                    f"pay the roughly {surcharge}% surcharge",
                )
            )
            checks.append(
                (
                    "exact-path surcharge (limitations list)",
                    ROADMAP,
                    f"paying the roughly {surcharge}% surcharge",
                )
            )
            # The draft argues for median-and-IQR by contrasting this archive's
            # extremes with the previous archive's. The claim about *this* one
            # has to come from it rather than from memory of an earlier run.
            checks.append(
                (
                    "lazy-gradient extremes in this archive",
                    SPEED,
                    f"between ${_pct_saving_precise(ratio['max'])}\\%$ and ${_pct_saving_precise(ratio['min'])}\\%$",
                )
            )
            # The reproduction command names a session count; if it drifts from
            # the archive the reader cannot regenerate what is quoted.
            checks.append(
                (
                    "lazy-gradient reproduction command",
                    SPEED,
                    f"run_draft_timings.py --sessions {ratio['n_sessions']}`",
                )
            )

    # --- Section 1.3.2: the lazy-gradient accuracy table --------------------
    # Each cell is the across-session median, the sessions differing in their
    # noise realisation. The table's own spread claim is checked too: without it
    # a reader would take the digits at face value, and the tail moves.
    sweep = across.get("snr_sweep")
    if isinstance(sweep, dict):
        for snr in (200, 50, 10):
            row_key = f"snr={snr}"
            row_data = sweep.get(row_key)
            if not isinstance(row_data, dict):
                continue
            row = (
                f"| {snr} | {_sci1(row_data['median_abs_delta_sigma']['median'])} "
                f"| {_sig2(row_data['max_abs_delta_sigma']['median'])} "
                f"| {_sig2(row_data['max_abs_dx0_px']['median'])} "
                f"| {_sig2(row_data['max_abs_dpa_rad']['median'])} |"
            )
            checks.append((f"accuracy table row: S/N = {snr}", SPEED, row))

        tail = sweep.get("snr=50", {}).get("max_abs_delta_sigma")
        if tail:
            checks.append(
                (
                    "accuracy table, tail spread at S/N = 50",
                    SPEED,
                    # The S/N label is inside the claim on purpose: a spread
                    # attached to the wrong row is as wrong as a bad digit.
                    f"$S/N = 50$ maximum has an interquartile range of ${_sig2(tail['q1'])}$–${_sig2(tail['q3'])}$",
                )
            )
            checks.append(
                (
                    "accuracy table, noise-realisation count",
                    SPEED,
                    f"median across {_words(tail['n_sessions'])} noise realisations",
                )
            )
            checks.append(
                (
                    "accuracy table, tail extremes at S/N = 50",
                    SPEED,
                    f"range from ${_sig2(tail['min'])}$ to ${_sig2(tail['max'])}$",
                )
            )

        # §1.6.2 restates the worst-case figure when it lists the lazy gradient
        # as an accuracy limitation. Restated numbers drift the most, because
        # nothing about editing one section prompts a look at the other.
        worst = sweep.get("snr=10", {}).get("max_abs_delta_sigma")
        if worst:
            checks.append(
                (
                    "lazy-gradient worst case restated in the limitations list",
                    ROADMAP,
                    f"reaching ${_sig2(worst['median'])}\\sigma$ on the worst isophote",
                )
            )

    # --- Section 1.6.5: the NumPy fallback trade-off ------------------------
    # The draft advises when to prefer the fallback, so the break-even fit
    # count and the slowdown behind that advice are both derived here. An
    # earlier version asserted the fallback "can be net faster" for a single
    # fit while stating in the next sentence that the penalty was unmeasured.
    slowdown = across.get("numba_fallback_slowdown")
    break_even = across.get("numba_fallback_break_even_fits")
    if slowdown and break_even:
        checks.append(
            (
                "NumPy fallback slowdown",
                ROADMAP,
                f"about ${slowdown['median']:.2f}\\times$ slower per fit",
            )
        )
        checks.append(
            (
                "NumPy fallback break-even fit count",
                ROADMAP,
                f"break-even at about {round(break_even['median'])} fits per process",
            )
        )
        checks.append(
            (
                "NumPy fallback break-even spread",
                ROADMAP,
                f"IQR {round(break_even['q1'])}–{round(break_even['q3'])}",
            )
        )
        # The extremes are quoted on purpose, so a reader can see how far the
        # tail runs past the quartiles -- so they are derived like everything
        # else rather than characterised in a word.
        checks.append(
            (
                "NumPy fallback break-even extremes",
                ROADMAP,
                f"sessions ranged from {round(break_even['min'])} to {round(break_even['max'])} fits",
            )
        )

    # --- Sections 1.3.2 / 1.6.5: cold start ---------------------------------
    # Quoted as an across-session median with an IQR, each to the precision its
    # own spread supports, and every figure taken from the archive. An earlier
    # version built the import bracket as "median minus 0.1", which is not a
    # measurement, and left the lower compilation bound unguarded entirely.
    cold_components = (
        ("import", "cold_start_import_s", 2),
        ("compilation", "cold_start_compilation_s", 2),
        ("cache load", "cold_start_cache_load_s", 3),
    )
    for label, key, places in cold_components:
        summary = across.get(key)
        if not summary:
            continue
        clause = (
            f"${summary['median']:.{places}f}\\,\\mathrm{{s}}$ "
            f"(IQR ${summary['q1']:.{places}f}$–${summary['q3']:.{places}f}$)"
        )
        checks.append((f"cold start, {label} (roadmap)", ROADMAP, clause))
        checks.append((f"cold start, {label} (speed section)", SPEED, clause))

    return checks


def _ms(value: float) -> str:
    """Match the draft's mixed precision: 3 decimals below 1 ms, else 2."""
    return f"{value:.3f} ms" if value < 1.0 else f"{value:.2f} ms"


def _sci1(value: float) -> str:
    """One significant figure in the LaTeX scientific notation the table uses.

    The accuracy table's median column spans three decades, so a fixed number of
    decimal places would either truncate the smallest entry to zero or pad the
    largest with digits the measurement does not support. One significant figure
    is all the run-to-run spread justifies.
    """
    if value <= 0:
        raise ValueError(f"cannot render {value} in scientific notation")
    exponent = math.floor(math.log10(value))
    mantissa = round(value / 10.0**exponent)
    if mantissa == 10:  # e.g. 9.7e-4 rounds up into the next decade
        mantissa, exponent = 1, exponent + 1
    return f"${mantissa} \\times 10^{{{exponent}}}$"


def _sig2(value: float) -> str:
    """Two significant figures as a plain decimal, as the table's later columns."""
    if value <= 0:
        raise ValueError(f"cannot render {value} to two significant figures")
    places = max(0, 1 - math.floor(math.log10(value)))
    return f"{value:.{places}f}"


_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 9: "nine", 18: "eighteen"}


def _words(count: int) -> str:
    return _NUMBER_WORDS.get(count, str(count))


def _run(checks: List[Check], draft: Dict[str, str], verbose: bool = True) -> List[str]:
    failures: List[str] = []
    for name, filename, expected in checks:
        ok = _squash(expected) in _squash(draft.get(filename, ""))
        if verbose:
            print(f"{'OK  ' if ok else 'FAIL'} {name}  [{filename}]")
            if not ok:
                print(f"       expected verbatim: {expected!r}")
        if not ok:
            failures.append(name)
    return failures


def _self_test(checks: List[Check], draft: Dict[str, str]) -> int:
    """Perturb one numeric *occurrence* at a time and require a check to notice.

    Mutating a token globally is not enough: a guarded ``18`` can fail a check
    while an unguarded ``18`` two paragraphs away goes unnoticed, because the
    global edit changes both and the failure is credited to the guarded one.
    This walks every individual number in the guarded sections, changes just
    that occurrence, and asks whether any check objects.

    Occurrences nobody objects to are reported. Most are legitimately outside
    the harness's scope — configuration values, cross-references, results from
    other experiments — so this is a coverage map rather than a pass/fail on
    the draft. What it must not contain is a harness-derived timing.
    """
    guarded_files = sorted({filename for _, filename, _ in checks})
    unguarded: List[Tuple[str, str, str]] = []
    total = 0

    for filename in guarded_files:
        text = draft.get(filename, "")
        for match in re.finditer(r"\d+(?:\.\d+)?", text):
            total += 1
            token = match.group()
            bumped = f"{float(token) + 1.0:.{len(token.split('.')[1])}f}" if "." in token else str(int(token) + 1)
            mutated = dict(draft)
            mutated[filename] = text[: match.start()] + bumped + text[match.end() :]
            if not _run(checks, mutated, verbose=False):
                context = re.sub(r"\s+", " ", text[max(0, match.start() - 45) : match.end() + 25]).strip()
                unguarded.append((filename, token, context))

    print(f"self-test: perturbed {total} numeric occurrences, one at a time")
    caught = total - len(unguarded)
    print(f"  {caught} were caught by at least one check; {len(unguarded)} were not")

    if unguarded:
        print("\ncoverage: occurrences no check speaks for (expected for anything")
        print("outside the harness's scope; a harness-derived timing here is a bug)")
        by_file: Dict[str, int] = {}
        for filename, _, _ in unguarded:
            by_file[filename] = by_file.get(filename, 0) + 1
        for filename, count in sorted(by_file.items()):
            print(f"  {filename}: {count}")
            for name, token, context in unguarded:
                if name == filename:
                    print(f"      {token:>9}  …{context}…")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Result JSON to check against.")
    parser.add_argument("--self-test", action="store_true", help="Verify the checks fail on corrupted text.")
    args = parser.parse_args()

    if not args.results.is_file():
        raise SystemExit(f"results file not found: {args.results}\nRun run_draft_timings.py --archive first.")

    results = json.loads(args.results.read_text())
    draft = _load_draft()
    checks = build_checks(results)
    if not checks:
        raise SystemExit("no checks were derivable from this results file")

    if args.self_test:
        sys.exit(_self_test(checks, draft))

    failures = _run(checks, draft)
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
    if failures:
        print("\nThe draft no longer matches the archived measurements:")
        for name in failures:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
