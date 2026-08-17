"""Check that every timing quoted in the publication draft matches the archive.

The draft's timing tables are transcribed by hand from
``reference_timings.json``, and a transcription that drifts is worse than no
number at all — it looks measured. This script re-derives each quoted value from
the archive and confirms the draft still contains it, so a stale table fails
loudly instead of surviving another review round.

It checks *presence of the exact rendered string*, not a numeric tolerance:
the question is whether the manuscript says what the archive says, so a value
that has been rounded differently is a finding, not a pass.

Usage::

    uv run python benchmarks/draft_timings/check_draft_numbers.py
    uv run python benchmarks/draft_timings/check_draft_numbers.py --results outputs/benchmark_draft_timings/timings.json

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFT_DIR = REPO_ROOT / "docs" / "publication" / "draft"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "reference_timings.json"


def _load_draft() -> Dict[str, str]:
    if not DRAFT_DIR.is_dir():
        raise SystemExit(f"draft directory not found: {DRAFT_DIR} (it is gitignored; nothing to check)")
    return {path.name: path.read_text() for path in DRAFT_DIR.glob("*.md")}


def _thousands(value: int) -> str:
    """Render an integer the way the draft's tables do (thin-space groups)."""
    return f"{value:,}".replace(",", " ")


def build_checks(results: Dict[str, object], draft: Dict[str, str]) -> List[Tuple[str, str, bool]]:
    """Return ``(name, file, ok)`` for every quoted value we can re-derive."""
    checks: List[Tuple[str, str, bool]] = []

    def require(name: str, filename: str, needles: Sequence[str]) -> None:
        text = draft.get(filename, "")
        checks.append((name, filename, all(needle in text for needle in needles)))

    ea = results.get("ea_isofit", {})
    if ea:
        require(
            "EA/ISOFIT absolute times",
            "section-1.4.1-eccentric-anomaly-isofit.md",
            [
                f"{ea[key]['median_ms']:.1f} ms"
                for key in (
                    "default_phi_posthoc",
                    "eccentric_anomaly_only",
                    "isofit_in_loop_only",
                    "eccentric_anomaly_plus_isofit",
                )
            ],
        )

    bands = results.get("joint_solve_bands", {})
    for mode in ("ols", "wls"):
        block = bands.get(mode) if isinstance(bands, dict) else None
        if block:
            require(
                f"joint solve, geometry form, {mode.upper()}",
                "section-1.6-limitations-roadmap.md",
                [f"{block[f'geometry_B={n}']['median_ms']:.3f} ms" for n in (3, 6, 12)],
            )

    maxsma = results.get("maxsma", {})
    if maxsma:
        keys = ("maxsma=100", "maxsma=200", "maxsma=400")
        require(
            "maxsma sampled-point totals",
            "section-1.6-limitations-roadmap.md",
            [_thousands(maxsma[key]["sampled_points"]) for key in keys],
        )
        require(
            "maxsma ring counts",
            "section-1.6-limitations-roadmap.md",
            [f"| {maxsma[key]['n_isophotes']} |" for key in keys],
        )

    lazy = results.get("lazy_gradient", {})
    if lazy:
        require(
            "lazy-gradient evaluation counts",
            "section-1.3-why-fast.md",
            [str(lazy["lazy"]["gradient_evaluations"]), str(lazy["classical"]["gradient_evaluations"])],
        )

    if maxsma:
        require(
            "maxsma wall times",
            "section-1.6-limitations-roadmap.md",
            [f"{maxsma[key]['median_ms']:.1f} ms" for key in ("maxsma=100", "maxsma=200", "maxsma=400")],
        )

    cold = results.get("cold_start", {})
    if cold:
        # Quoted to one significant figure in seconds, so check the bracket the
        # draft states rather than an exact string.
        seconds = {name: cold[name]["median_ms"] / 1000.0 for name in ("import", "compilation", "cache_load")}
        checks.append(
            (
                "cold-start import within the quoted 0.8-0.9 s",
                "section-1.6-limitations-roadmap.md",
                0.75 <= seconds["import"] <= 0.95,
            )
        )
        checks.append(
            (
                "cold-start compilation within the quoted 0.35-0.4 s",
                "section-1.6-limitations-roadmap.md",
                0.30 <= seconds["compilation"] <= 0.45,
            )
        )
        checks.append(
            (
                "cold-start cache load within the quoted 0.1 s",
                "section-1.6-limitations-roadmap.md",
                0.05 <= seconds["cache_load"] <= 0.15,
            )
        )

    across = results.get("across_sessions", {})
    ratio = across.get("lazy_gradient_runtime_ratio") if isinstance(across, dict) else None
    if ratio:
        # The draft quotes a saving as a percentage, i.e. 1 - runtime ratio.
        require(
            "lazy-gradient saving, across-session IQR",
            "section-1.3-why-fast.md",
            [f"{100 * (1 - ratio['q3']):.0f}", f"{100 * (1 - ratio['q1']):.0f}"] if "q3" in ratio else [],
        )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Result JSON to check the draft against (default: the committed archive).",
    )
    args = parser.parse_args()

    if not args.results.is_file():
        raise SystemExit(f"results file not found: {args.results}\nRun run_draft_timings.py --archive first.")

    results = json.loads(args.results.read_text())
    checks = build_checks(results, _load_draft())

    if not checks:
        raise SystemExit("no checks were derivable from this results file")

    for name, filename, ok in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {name}  [{filename}]")

    failures = [name for name, _, ok in checks if not ok]
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
    if failures:
        print("\nThe draft no longer matches the archived measurements:")
        for name in failures:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
