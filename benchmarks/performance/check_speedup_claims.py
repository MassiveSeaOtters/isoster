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
    """(name, relative path, expected verbatim text) for each quoted claim."""
    speed = archive["speedup"]
    median = round(speed["median"])
    checks = [
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
            "README coverage: attempted, failed, completed",
            "README.md",
            f"Of {archive['attempted_cases']} attempted configurations, `photutils` could not fit "
            f"{archive['photutils_failures']}",
        ),
        (
            "README completed-case count",
            "README.md",
            f"the remaining {archive['completed_cases']} all passed",
        ),
        (
            "CLAUDE.md median and coverage",
            "CLAUDE.md",
            f"median {median}x over the {archive['completed_cases']} of "
            f"{archive['attempted_cases']} synthetic Sersic configurations",
        ),
        (
            "CITATION.cff median and coverage",
            "CITATION.cff",
            f"median {median}x over the {archive['completed_cases']}",
        ),
    ]
    return checks


def main() -> None:
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

    failures_found = []
    for name, relative, expected in build_checks(archive):
        text = (REPO_ROOT / relative).read_text()
        ok = _squash(expected) in _squash(text)
        print(f"{'OK  ' if ok else 'FAIL'} {name}  [{relative}]")
        if not ok:
            print(f"       expected verbatim: {expected!r}")
            failures_found.append(name)

    total = len(build_checks(archive))
    print(f"\n{total - len(failures_found)}/{total} checks passed")
    if failures_found:
        print("\nQuoted speedups no longer match the archived sweep:")
        for name in failures_found:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
