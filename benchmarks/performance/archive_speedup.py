"""Promote a bench_vs_photutils run to the committed speedup archive.

``bench_vs_photutils.py`` writes its full results under ``outputs/``, which is
gitignored. This reduces that run to the summary the repository quotes --
median, quartiles, extremes, and the grid-coverage figures -- and writes it to
``reference_speedup.json`` beside this script, where it is tracked.

``check_speedup_claims.py`` then verifies that the README, ``CLAUDE.md`` and
``CITATION.cff`` still match that archive, and runs in the docs CI job. The
three files form the same contract ``benchmarks/draft_timings/`` implements for
the technical chapter: measure, archive, and fail the build if the prose drifts.

Coverage is archived, not just performance. photutils cannot fit every
configuration in the grid, and a summary that silently reported only the
survivors would read as though the grid were smaller than it is.

Usage::

    uv run python benchmarks/performance/bench_vs_photutils.py --no-failure-plots
    uv run python benchmarks/performance/archive_speedup.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "benchmarks_performance" / "bench_vs_photutils" / "benchmark_results.json"
ARCHIVE = Path(__file__).resolve().parent / "reference_speedup.json"

WHAT = (
    "Wall-time ratio photutils.isophote / isoster on a synthetic Sersic grid. "
    "Every completed case also had to pass the script's accuracy criteria "
    "against the true profile. This is a synthetic sweep, not a real-image "
    "benchmark."
)


def describe_failures(failures: list[dict], attempted: int, completed: int) -> str:
    """Describe the excluded configurations *from the records*, not from memory.

    An earlier version hard-coded the failure pattern and the exception text.
    That is fine until a run fails differently, at which point the archive
    carries a confident description of something that did not happen.
    """
    if not failures:
        return f"photutils fitted all {attempted} attempted configurations; nothing is excluded."

    def _spread(key: str) -> str:
        values = sorted({f"{f[key]}" for f in failures})
        return values[0] if len(values) == 1 else "varies (" + ", ".join(values) + ")"

    exceptions = sorted({f"{f.get('exception_type', 'Unknown')}: {f.get('message', '')}" for f in failures})
    exception_text = exceptions[0] if len(exceptions) == 1 else "; ".join(exceptions)

    return (
        f"photutils could not fit {len(failures)} of the {attempted} attempted "
        f"configurations; they are listed in failed_configurations and excluded "
        f"from every statistic here. Across those cases: n={_spread('n')}, "
        f"R_e={_spread('R_e')}, eps={_spread('eps')}, pa={_spread('pa')}, "
        f"noise_snr={_spread('noise_snr')}. Reported failure -- {exception_text}. "
        f"The speedups below therefore describe the {completed} configurations "
        f"photutils could fit and say nothing about the {len(failures)} it could not. "
        f"Read the parameter spread above before assuming the exclusions are random: "
        f"a value that does not vary is a systematic gap in coverage."
    )


PROTOCOL_CAVEAT = (
    "Each configuration is timed ONCE per tool, isoster first and photutils "
    "second, with no warm-up, no repetition and no interleaving. The quartiles "
    "below are spread across grid CONFIGURATIONS, not across repeated timings "
    "of one configuration, so they do not characterise timing noise. This is "
    "an archived single sweep, not a controlled timing study of the kind "
    "benchmarks/draft_timings implements. The measured ratio is large enough "
    "(tens of times) that ordering and warm-up effects are unlikely to change "
    "its order of magnitude, but they have not been quantified."
)

CAVEAT = (
    "Absolute times are machine-specific. The ratio is less machine-dependent "
    "than absolute time, but it is not portable: hardware, BLAS, and library "
    "versions all change ratios too. One run on one machine."
)


def summarize(results_path: Path) -> dict:
    """Reduce one benchmark run to the archived summary."""
    run = json.loads(results_path.read_text())
    summary = run["summary"]
    speedups = sorted(case["speedup"] for case in run["test_cases"] if case.get("speedup"))
    if len(speedups) < 4:
        raise SystemExit(f"only {len(speedups)} timed cases; refusing to archive a summary this thin")
    quartiles = statistics.quantiles(speedups, n=4)

    by_radius: dict[str, list[float]] = collections.defaultdict(list)
    for case in run["test_cases"]:
        if case.get("speedup"):
            by_radius[str(case["params"]["R_e"])].append(case["speedup"])

    attempted = summary["attempted_cases"]
    completed = len(speedups)
    failed = summary["photutils_failures"]
    if completed + failed != attempted:
        raise SystemExit(f"run is internally inconsistent: {completed} + {failed} != {attempted}")

    return {
        "source": "benchmarks/performance/bench_vs_photutils.py",
        "what": WHAT,
        "coverage_caveat": describe_failures(summary["failed_configurations"], attempted, completed),
        "protocol_caveat": PROTOCOL_CAVEAT,
        "environment": run["environment"],
        "provenance_warning": (
            None
            if (run["environment"].get("git_worktree") or {}).get("dirty") is False
            else "Produced from a working tree that did not match its recorded commit, "
            "or from a tree whose state could not be determined. See "
            "environment.git_worktree. Not reproducible provenance."
        ),
        "attempted_cases": attempted,
        "completed_cases": completed,
        "photutils_failures": failed,
        "failed_configurations": summary["failed_configurations"],
        "all_completed_cases_passed": bool(summary["all_completed_cases_passed"]),
        "speedup": {
            "median": round(statistics.median(speedups), 1),
            "q1": round(quartiles[0], 1),
            "q3": round(quartiles[2], 1),
            "min": round(speedups[0], 1),
            "max": round(speedups[-1], 1),
        },
        "speedup_by_R_e": {
            radius: {
                "n": len(values),
                "median": round(statistics.median(values), 1),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
            }
            for radius, values in sorted(by_radius.items(), key=lambda item: float(item[0]))
        },
        "caveat": CAVEAT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Run to archive.")
    args = parser.parse_args()

    if not args.results.is_file():
        raise SystemExit(f"results not found: {args.results}\nRun bench_vs_photutils.py first.")

    archived = summarize(args.results)
    ARCHIVE.write_text(json.dumps(archived, indent=2) + "\n")
    print(f"archived {ARCHIVE}")
    print(json.dumps(archived["speedup"], indent=2))
    print(
        f"coverage: {archived['completed_cases']} completed of "
        f"{archived['attempted_cases']} attempted, {archived['photutils_failures']} photutils failures"
    )


if __name__ == "__main__":
    main()
