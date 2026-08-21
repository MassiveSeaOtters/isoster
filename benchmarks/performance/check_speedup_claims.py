"""Check the speedup figures quoted in the repo against the archived sweep.

The README, ``CLAUDE.md`` and ``CITATION.cff`` all quote a photutils speedup.
Those numbers are transcribed by hand from
``benchmarks/performance/reference_speedup.json``, and a transcription that
drifts is worse than no number at all -- it still looks measured. This rebuilds
each quoted clause from the archive and asserts the file contains it verbatim.

It is the same contract ``benchmarks/draft_timings/check_draft_numbers.py``
enforces for the technical chapter, applied to the one headline number that
lives outside it. Both run in the docs CI job.

Usage::

    uv run python benchmarks/performance/check_speedup_claims.py

Exits non-zero if any claim no longer matches the archive.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(__file__).resolve().parent / "reference_speedup.json"


def _squash(text: str) -> str:
    """Collapse whitespace so line-wrapping cannot hide or fake a match."""
    return re.sub(r"\s+", " ", text)


def build_checks(archive: dict) -> list[tuple[str, str, str]]:
    """(name, relative path, expected verbatim text) for each quoted claim.

    Whole clauses, never fragments. An earlier version of this checker matched
    "median 46x over the 237", which left the grid size unguarded: changing 243
    to 999 in CITATION.cff passed every check. Anything short of a complete
    claim can pass by accident, so each expected string here carries every
    number the sentence asserts.
    """
    speed = archive["speedup"]
    median = round(speed["median"])
    attempted = archive["attempted_cases"]
    completed = archive["completed_cases"]
    failed = archive["photutils_failures"]
    return [
        (
            "README headline median",
            "README.md",
            f"a median of ${median}\\times$ on a synthetic Sérsic sweep",
        ),
        (
            "README IQR and slowest case",
            "README.md",
            f"(interquartile range ${round(speed['q1'])}$–${round(speed['q3'])}\\times$; "
            f"slowest case ${round(speed['min'])}\\times$)",
        ),
        (
            "README coverage clause",
            "README.md",
            f"Of {attempted} attempted configurations, `photutils` could not fit {failed}",
        ),
        (
            "README completed-case count",
            "README.md",
            f"the remaining {completed} all passed",
        ),
        (
            "README feature bullet median",
            "README.md",
            f"(median ${median}\\times$ on the synthetic Sérsic sweep; see above)",
        ),
        (
            "CLAUDE.md median and full coverage",
            "CLAUDE.md",
            f"median {median}x over the {completed} of {attempted} synthetic Sersic configurations photutils could fit",
        ),
        (
            "CITATION.cff median and full coverage",
            "CITATION.cff",
            f"median {median}x over the {completed}\n  of {attempted} synthetic Sersic "
            "configurations photutils could fit",
        ),
        (
            "consistency audit: median, IQR and slowest",
            "docs/publication/method-code-consistency-audit.md",
            f"**median {median}x**\n  (IQR {round(speed['q1'])}--{round(speed['q3'])}x, "
            f"slowest {round(speed['min'])}x)",
        ),
        (
            "consistency audit: coverage",
            "docs/publication/method-code-consistency-audit.md",
            f"over the **{completed} of {attempted}** configurations photutils\n  could fit",
        ),
        (
            "consistency audit: excluded-case count",
            "docs/publication/method-code-consistency-audit.md",
            f"The {failed} excluded configurations are systematic, not random",
        ),
        (
            "consistency audit: coverage figures restated in prose",
            "docs/publication/method-code-consistency-audit.md",
            f"including the {attempted}/{completed}/{failed} coverage figures",
        ),
    ]


def self_test(archive: dict) -> int:
    """Perturb one numeric occurrence at a time and require a check to notice.

    A checker nobody has tried to fool is not evidence. This walks every number
    in each guarded file, changes just that occurrence, and reports the ones no
    check objects to. Most are legitimately out of scope -- this file guards
    only the speedup claims -- so the output is a coverage map. What it must not
    contain is a speedup figure.
    """
    guarded = sorted({relative for _, relative, _ in build_checks(archive)})
    unguarded: list[tuple[str, str, str]] = []
    total = 0

    for relative in guarded:
        path = REPO_ROOT / relative
        text = path.read_text()
        for match in re.finditer(r"\d+(?:\.\d+)?", text):
            total += 1
            token = match.group()
            if "." in token:
                bumped = f"{float(token) + 1.0:.{len(token.split('.')[1])}f}"
            else:
                bumped = str(int(token) + 1)
            mutated = {relative: text[: match.start()] + bumped + text[match.end() :]}
            if not _run(build_checks(archive), mutated, verbose=False):
                context = re.sub(r"\s+", " ", text[max(0, match.start() - 45) : match.end() + 25]).strip()
                unguarded.append((relative, token, context))

    print(f"self-test: perturbed {total} numeric occurrences, one at a time")
    print(f"  {total - len(unguarded)} were caught by at least one check; {len(unguarded)} were not")
    if unguarded:
        print("\ncoverage: occurrences no check speaks for (expected for anything")
        print("unrelated to the speedup claims; a speedup figure here is a bug)")
        by_file: dict[str, int] = {}
        for relative, _, _ in unguarded:
            by_file[relative] = by_file.get(relative, 0) + 1
        for relative, count in sorted(by_file.items()):
            print(f"  {relative}: {count}")
    return 0


def _run(checks: list[tuple[str, str, str]], overrides: dict[str, str], verbose: bool = True) -> list[str]:
    """Run every check, reading overridden files from memory rather than disk."""
    failures: list[str] = []
    for name, relative, expected in checks:
        text = overrides.get(relative)
        if text is None:
            text = (REPO_ROOT / relative).read_text()
        ok = _squash(expected) in _squash(text)
        if verbose:
            print(f"{'OK  ' if ok else 'FAIL'} {name}  [{relative}]")
            if not ok:
                print(f"       expected verbatim: {expected!r}")
        if not ok:
            failures.append(name)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Verify the checks fail on corrupted text.")
    args = parser.parse_args()

    if not ARCHIVE.is_file():
        raise SystemExit(
            f"archive not found: {ARCHIVE}\nRun benchmarks/performance/bench_vs_photutils.py, then archive_speedup.py."
        )
    archive = json.loads(ARCHIVE.read_text())

    # The archive must not silently shrink its own grid: the coverage numbers
    # are part of the claim, not decoration.
    completed = archive["completed_cases"]
    attempted = archive["attempted_cases"]
    failures = archive["photutils_failures"]
    if completed + failures != attempted:
        raise SystemExit(
            f"archive is internally inconsistent: {completed} completed + {failures} failed != {attempted} attempted"
        )
    if not archive["all_completed_cases_passed"]:
        raise SystemExit("archive records a completed case that failed its accuracy criteria")
    if archive.get("provenance_warning"):
        print(f"WARNING: {archive['provenance_warning']}\n")

    if args.self_test:
        sys.exit(self_test(archive))

    checks = build_checks(archive)
    failed = _run(checks, overrides={})
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("\nQuoted speedups no longer match the archived sweep:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
