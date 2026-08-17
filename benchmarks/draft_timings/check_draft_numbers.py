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
                    f"median of ${combined['median']:.2f}$ and an\ninterquartile range of "
                    f"${combined['q1']:.2f}$–${combined['q3']:.2f}$",
                )
            )
            checks.append(("EA session count", EA, f"Across {_words(combined['n_sessions'])}\nsessions"))

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
                    f"median is ${b_ratio['median']:.1f}$ with an interquartile range of\n"
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
                    f"interquartile range of ${_pct_saving(ratio['q3'])}\\%$ to ${_pct_saving(ratio['q1'])}\\%$",
                )
            )
            checks.append(("lazy-gradient session count", SPEED, f"{_words(ratio['n_sessions'])} sessions"))

    # --- Sections 1.3.2 / 1.6.5: cold start ---------------------------------
    cold = results.get("cold_start")
    if cold:
        seconds = {name: cold[name]["median_ms"] / 1000.0 for name in ("import", "compilation", "cache_load")}
        # The import figure is quoted as a bracket, so the whole clause is the
        # claim; matching only the upper bound would leave the lower unguarded.
        lo_import = _sig1(seconds["import"] - 0.1)
        checks.append(
            (
                "cold-start import bracket (roadmap)",
                ROADMAP,
                f"${lo_import}\\,\\mathrm{{s}}$ to ${_sig1(seconds['import'])}\\,\\mathrm{{s}}$",
            )
        )
        # §1.3.2 repeats all three figures; they were previously unguarded.
        checks.append(
            (
                "cold-start import bracket (speed section)",
                SPEED,
                f"(${lo_import}$–${_sig1(seconds['import'])}\\,\\mathrm{{s}}$ on the reference machine)",
            )
        )
        checks.append(
            (
                "cold-start cache load (speed section)",
                SPEED,
                f"(roughly ${_sig1(seconds['cache_load'])}\\,\\mathrm{{s}}$)",
            )
        )
        checks.append(
            (
                "cold-start compilation (speed section)",
                SPEED,
                f"(a further ${_sig1(seconds['compilation'])}\\,\\mathrm{{s}}$)",
            )
        )
        checks.append(
            (
                "cold-start cache load, as quoted",
                ROADMAP,
                f"about ${_sig1(seconds['cache_load'])}\\,\\mathrm{{s}}$ more than a",
            )
        )
        checks.append(
            (
                "cold-start compilation, as quoted",
                ROADMAP,
                f"to ${_sig1(seconds['compilation'])}\\,\\mathrm{{s}}$, which is the",
            )
        )

    return checks


def _ms(value: float) -> str:
    """Match the draft's mixed precision: 3 decimals below 1 ms, else 2."""
    return f"{value:.3f} ms" if value < 1.0 else f"{value:.2f} ms"


def _sig1(seconds: float) -> str:
    return f"{seconds:.1f}"


_NUMBER_WORDS = {3: "three", 9: "nine", 18: "eighteen"}


def _words(count: int) -> str:
    return _NUMBER_WORDS.get(count, str(count))


def _run(checks: List[Check], draft: Dict[str, str], verbose: bool = True) -> List[str]:
    failures: List[str] = []
    for name, filename, expected in checks:
        ok = expected in draft.get(filename, "")
        if verbose:
            print(f"{'OK  ' if ok else 'FAIL'} {name}  [{filename}]")
            if not ok:
                print(f"       expected verbatim: {expected!r}")
        if not ok:
            failures.append(name)
    return failures


def _self_test(checks: List[Check], draft: Dict[str, str]) -> int:
    """Mutate one numeric claim at a time and require a check to notice each.

    Mutating every digit at once proves only that each *implemented* check is
    sensitive to some change. It says nothing about numeric claims for which no
    check exists, which is the failure mode that let an earlier version of this
    script pass against a deliberately corrupted draft.

    So this walks every distinct number that appears in any check's expected
    text, perturbs just that number everywhere in the draft, and requires at
    least one check to fail. A number that can be changed with impunity is
    reported: either it is unguarded, or the check that should guard it is
    matching too loosely.
    """
    tokens = sorted({token for _, _, expected in checks for token in re.findall(r"\d+(?:\.\d+)?", expected)})
    words = sorted({word for _, _, expected in checks for word in _NUMBER_WORDS.values() if word in expected})

    unguarded: List[str] = []

    def perturb(token: str) -> str:
        if "." in token:
            return f"{float(token) + 1.0:.{len(token.split('.')[1])}f}"
        return str(int(token) + 1)

    for token in tokens:
        pattern = re.compile(rf"(?<![\d.]){re.escape(token)}(?![\d])")
        mutated = {name: pattern.sub(perturb(token), text) for name, text in draft.items()}
        if not _run(checks, mutated, verbose=False):
            unguarded.append(f"number {token!r}")

    for word in words:
        mutated = {name: text.replace(word, "SOMENUMBER") for name, text in draft.items()}
        if not _run(checks, mutated, verbose=False):
            unguarded.append(f"count word {word!r}")

    # Coverage: numbers present in the guarded sections that no check mentions.
    # These are not failures -- many are configuration values, cross-references
    # or figures sourced from elsewhere -- but listing them is the only way to
    # see which numeric claims this script does *not* speak for.
    guarded = set(tokens)
    present: Dict[str, set] = {}
    for filename in (SPEED, EA, ROADMAP):
        found = set(re.findall(r"\d+(?:\.\d+)?", draft.get(filename, "")))
        uncovered = found - guarded
        if uncovered:
            present[filename] = uncovered

    total = len(tokens) + len(words)
    print(f"self-test: perturbed {total} numeric claims one at a time")
    if unguarded:
        print(f"{len(unguarded)} of them can be changed without any check failing:")
        for item in unguarded:
            print(f"  - {item}")
        return 1
    print("self-test: every one was caught by at least one check")
    print("\ncoverage: numbers in the guarded sections that no check speaks for")
    print("(expected -- configuration values, cross-references, and results from")
    print(" other experiments -- but listed so the boundary is visible)")
    for filename, values in sorted(present.items()):
        sample = ", ".join(sorted(values, key=lambda v: (len(v), v))[:12])
        print(f"  {filename}: {len(values)} distinct, e.g. {sample}")
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
