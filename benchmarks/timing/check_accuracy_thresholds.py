"""Gate the Stage 1 accuracy thresholds against the module that derives them.

The bars in the design spec decide which timings enter a headline ratio. They
are computed by ``accuracy_thresholds.py`` and *transcribed* into prose, and a
transcription that drifts still reads as derived --- the same contract the
other four gates enforce, applied before Part B produces any measurement.

Usage::

    uv run python benchmarks/timing/check_accuracy_thresholds.py
    uv run python benchmarks/timing/check_accuracy_thresholds.py --self-test
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.timing.accuracy_thresholds import thresholds  # noqa: E402

SPEC = REPO_ROOT / "docs" / "specs" / "2026-08-22-three-way-benchmark-comparison-design.md"


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def build_checks(computed: Dict[str, object]) -> List[Tuple[str, str, str]]:
    """Whole table rows, with a stem that carries none of the guarded numbers."""
    checks = []
    for metric, key, places in (
        ("amplitude", "systematic_amplitude_error_pct_by_ring", 2),
        ("intensity", "systematic_ring_intensity_error_pct_by_ring", 3),
    ):
        for fixture, bars in computed[key].items():
            cells = " / ".join(f"{value:.{places}f}" for value in bars.values())
            checks.append(
                (
                    f"{metric}_bars_{fixture}",
                    # Stem: the fixture name in a table cell. Stable, and it
                    # carries none of the values it guards, so a corrupted
                    # number fails rather than going dormant.
                    f"| `{fixture}` |",
                    f"{cells} %",
                )
            )
    return checks


def run_checks(checks: List[Tuple[str, str, str]], text: str) -> List[str]:
    squashed = _squash(text)
    failures, fired = [], []
    for name, stem, expected in checks:
        if _squash(stem) not in squashed:
            continue
        fired.append(name)
        if _squash(expected) not in squashed:
            failures.append(
                f"{name}: the spec discusses this row but does not state it as derived.\n       expected: {expected!r}"
            )
    dormant = [name for name, _, _ in checks if name not in fired]
    if dormant:
        print(f"     note: {len(dormant)} threshold row(s) stated in no document: {', '.join(dormant)}")
    else:
        print(f"     note: all {len(checks)} threshold row(s) are stated and match the derivation")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    computed = thresholds()
    checks = build_checks(computed)
    if not SPEC.exists():
        raise SystemExit(f"spec not found: {SPEC}")
    text = SPEC.read_text()

    if args.self_test:
        # Move every derived bar and require each row to stop matching.
        moved = {
            key: {f: {sma: value * 1.5 + 1.0 for sma, value in bars.items()} for f, bars in computed[key].items()}
            for key in ("systematic_amplitude_error_pct_by_ring", "systematic_ring_intensity_error_pct_by_ring")
        }
        corrupted = run_checks(build_checks({**computed, **moved}), text)
        live = [n for n, stem, _ in checks if _squash(stem) in _squash(text)]
        caught = {f.split(":")[0] for f in corrupted}
        missed = [name for name in live if name not in caught]
        print(f"self-test: {len(live) - len(missed)}/{len(live)} threshold rows trip when the derivation moves")
        for name in missed:
            print(f"  MISSED {name}: the spec can drift from the derivation without failing")
        raise SystemExit(1 if missed or not live else 0)

    failures = run_checks(checks, text)
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        raise SystemExit(1)
    print("all Stage 1 accuracy-threshold checks passed")


if __name__ == "__main__":
    main()
