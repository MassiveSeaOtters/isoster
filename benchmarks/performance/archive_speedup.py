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
    "its order of magnitude, but they have not been quantified. Quote the "
    "median and quartiles; treat the extremes as indicative only. Pass "
    "--compare-to <previous archive> to have the run-to-run drift measured "
    "and recorded here instead of assumed."
)


def compare_archives(previous: dict, current: dict) -> dict:
    """Report drift between two archives, and whether it can be attributed.

    Two numbers differing is a *measurement*; saying why they differ is an
    *inference*, and the inference only holds if everything except the thing
    being blamed was held fixed. An earlier version of this function reported
    the numbers and then concluded that movement in the extremes "is the
    single-timing protocol showing, not a change in the software" -- without
    checking that the two archives came from the same code, the same grid, the
    same protocol, or clean trees. Run against a dirty archive from a different
    commit, it made that causal claim anyway.

    So: the differences are always reported, plainly. The interpretation is
    attached only when the comparison is controlled, and when it is not, the
    blockers are named instead.
    """
    old_speed, current_speed = previous["speedup"], current["speedup"]
    differences = {
        key: {
            "previous": old_speed[key],
            "current": current_speed[key],
            "delta": round(current_speed[key] - old_speed[key], 2),
        }
        for key in ("median", "q1", "q3", "min", "max")
    }

    def _env(archive: dict, key: str):
        return (archive.get("environment") or {}).get(key)

    def _clean(archive: dict) -> bool:
        return ((archive.get("environment") or {}).get("git_worktree") or {}).get("dirty") is False

    blockers = []
    if _env(previous, "git_sha") != _env(current, "git_sha"):
        blockers.append(f"different code revision ({_env(previous, 'git_sha')} vs {_env(current, 'git_sha')})")
    if not _clean(previous) or not _clean(current):
        which = [label for label, archive in (("previous", previous), ("current", current)) if not _clean(archive)]
        blockers.append(f"unreproducible provenance: {', '.join(which)} archive not from a clean tree")
    for field, label in (
        ("attempted_cases", "grid size"),
        ("completed_cases", "completed cases"),
        ("photutils_failures", "excluded cases"),
    ):
        if previous.get(field) != current.get(field):
            blockers.append(f"different {label} ({previous.get(field)} vs {current.get(field)})")
    if previous.get("source") != current.get("source"):
        blockers.append("different benchmark script")
    if previous.get("protocol_caveat") != current.get("protocol_caveat"):
        blockers.append("different measurement protocol")
    if _env(previous, "platform") != _env(current, "platform"):
        blockers.append(f"different platform ({_env(previous, 'platform')} vs {_env(current, 'platform')})")

    summary = ", ".join(
        f"{key} {value['previous']}x -> {value['current']}x ({value['delta']:+})" for key, value in differences.items()
    )

    if blockers:
        interpretation = (
            "NOT ATTRIBUTABLE. These archives are not a controlled comparison: "
            + "; ".join(blockers)
            + ". The differences above are reported as measurements only -- they cannot be "
            "assigned to timing noise, to the protocol, or to a change in the software, "
            "because more than one thing changed between the runs."
        )
    else:
        interpretation = (
            "Controlled comparison: same commit, both from clean trees, same grid and "
            "completed-case counts, same script and protocol, same platform. With the "
            "code and the inputs held fixed, the differences above are run-to-run "
            "variation of the measurement itself. Movement larger in the extremes than "
            "in the median and quartiles is the expected signature of timing each "
            "configuration once."
        )

    return {
        "differences": differences,
        "summary": summary,
        "controlled": not blockers,
        "blockers": blockers,
        "interpretation": interpretation,
    }


CAVEAT = (
    "Absolute times are machine-specific. The ratio is less machine-dependent "
    "than absolute time, but it is not portable: hardware, BLAS, and library "
    "versions all change ratios too. One run on one machine."
)


def summarize(results_path: Path, previous_archive: Path | None = None) -> dict:
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

    archived = {
        "source": "benchmarks/performance/bench_vs_photutils.py",
        "what": WHAT,
        "coverage_caveat": describe_failures(summary["failed_configurations"], attempted, completed),
        "protocol_caveat": PROTOCOL_CAVEAT,
        # Environment and worktree state are the BENCHMARK RUN's, carried over
        # from the results file -- not this script's. That is the provenance
        # that matters: it says whether the code that produced the timings can
        # be reconstructed from the recorded commit.
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
    if previous_archive is not None:
        if not previous_archive.is_file():
            raise SystemExit(f"--compare-to archive not found: {previous_archive}")
        archived["run_to_run_drift"] = compare_archives(json.loads(previous_archive.read_text()), archived)
    return archived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Run to archive.")
    parser.add_argument(
        "--compare-to",
        type=Path,
        help=(
            "A previous reference_speedup.json. When given, the run-to-run drift "
            "between it and this run is measured and recorded in the archive. "
            "Without it no drift claim is made, because there is nothing to "
            "measure it against."
        ),
    )
    args = parser.parse_args()

    if not args.results.is_file():
        raise SystemExit(f"results not found: {args.results}\nRun bench_vs_photutils.py first.")

    archived = summarize(args.results, previous_archive=args.compare_to)
    ARCHIVE.write_text(json.dumps(archived, indent=2) + "\n")
    print(f"archived {ARCHIVE}")
    print(json.dumps(archived["speedup"], indent=2))
    print(
        f"coverage: {archived['completed_cases']} completed of "
        f"{archived['attempted_cases']} attempted, {archived['photutils_failures']} photutils failures"
    )


if __name__ == "__main__":
    main()
