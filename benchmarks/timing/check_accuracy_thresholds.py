"""Gate the whole Stage 1 contract, field by field.

The previous version guarded four transcribed table rows. A review corrupted
the realization count, the family-wise alpha, all three geometry bars, the
target interval, the coverage fraction and the load ceiling, and the checker
passed every time. A contract is not partly frozen.

This compares the **committed** contract against the one the code derives now,
leaf by leaf, and reports every field that moved. Its self-test mutates each
leaf individually and requires the gate to name that leaf --- one global
mutation counted N times is the failure mode this family of gates has already
had twice.

Usage::

    uv run python benchmarks/timing/check_accuracy_thresholds.py
    uv run python benchmarks/timing/check_accuracy_thresholds.py --self-test
    uv run python benchmarks/timing/check_accuracy_thresholds.py --refreeze
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.timing.accuracy_thresholds import stage_1_contract  # noqa: E402

FROZEN = Path(__file__).resolve().parent / "frozen_stage1_contract.json"


def _walk(value: object, path: Tuple[object, ...] = ()) -> List[Tuple[Tuple[object, ...], object]]:
    """Flatten to ``(path_tuple, scalar)``.

    The path stays a tuple rather than a dotted string because several keys are
    radii like ``12.0``: parsing a dotted name back into a path splits that
    into ``12`` and ``0``, and the self-test then cannot reach the field it
    means to mutate.
    """
    if isinstance(value, dict):
        out: List[Tuple[Tuple[object, ...], object]] = []
        for key, item in value.items():
            out.extend(_walk(item, path + (key,)))
        return out
    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            out.extend(_walk(item, path + (index,)))
        return out
    return [(path, value)]


def _display(path: Tuple[object, ...]) -> str:
    return " > ".join(str(part) for part in path)


def _leaves(value: object) -> Dict[str, object]:
    return {_display(path): leaf for path, leaf in _walk(value)}


def compare(frozen: Dict[str, object], computed: Dict[str, object]) -> List[str]:
    failures = []
    frozen_leaves = _leaves({k: v for k, v in frozen.items() if k != "fingerprint"})
    computed_leaves = _leaves({k: v for k, v in computed.items() if k != "fingerprint"})

    for name in sorted(set(frozen_leaves) | set(computed_leaves)):
        if name not in computed_leaves:
            failures.append(f"{name}: frozen but no longer derived")
            continue
        if name not in frozen_leaves:
            failures.append(f"{name}: newly derived but not frozen")
            continue
        want, got = frozen_leaves[name], computed_leaves[name]
        if isinstance(want, (int, float)) and isinstance(got, (int, float)) and not isinstance(want, bool):
            if abs(float(want) - float(got)) > 1e-9:
                failures.append(f"{name}: frozen {want!r}, derived {got!r}")
        elif str(want) != str(got):
            failures.append(f"{name}: frozen {want!r}, derived {got!r}")

    if frozen.get("fingerprint") != computed.get("fingerprint") and not failures:
        failures.append("fingerprint differs though every field matches; the contract shape changed")
    return failures


def _self_test(frozen: Dict[str, object], computed: Dict[str, object]) -> int:
    if compare(frozen, computed):
        print("self-test: the contract already fails unmodified; fix that first")
        return 1
    leaves = [(path, value) for path, value in _walk(frozen) if path[0] != "fingerprint"]
    missed = []
    for path, value in leaves:
        mutated = copy.deepcopy(frozen)
        node = mutated
        for key in path[:-1]:
            node = node[key]
        last = path[-1]
        if isinstance(value, bool):
            node[last] = not value
        elif isinstance(value, (int, float)):
            node[last] = float(value) * 1.5 + 1.0
        else:
            node[last] = f"{value}__mutated"
        name = _display(path)
        if not any(failure.startswith(f"{name}:") for failure in compare(mutated, computed)):
            missed.append(name)
    print(f"self-test: {len(leaves) - len(missed)}/{len(leaves)} contract fields trip when moved")
    for name in missed:
        print(f"  MISSED {name}: it can be edited without the gate noticing")
    return 1 if missed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--refreeze", action="store_true", help="Rewrite the frozen contract. Deliberate act.")
    args = parser.parse_args()

    computed = json.loads(json.dumps(stage_1_contract(), default=str))
    if args.refreeze:
        FROZEN.write_text(json.dumps(computed, indent=2, default=str))
        print(f"refroze {FROZEN.name} ({len(_leaves(computed))} fields)")
        return
    if not FROZEN.exists():
        raise SystemExit(f"no frozen contract at {FROZEN}; run --refreeze deliberately")
    frozen = json.loads(FROZEN.read_text())

    if args.self_test:
        raise SystemExit(_self_test(frozen, computed))

    failures = compare(frozen, computed)
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        print(f"\n{len(failures)} Stage 1 contract field(s) drifted from the frozen contract")
        raise SystemExit(1)
    print(f"OK   all {len(_leaves(frozen)) - 1} Stage 1 contract fields match the frozen contract")
    print(f"OK   fingerprint {frozen['fingerprint'][:16]}")


if __name__ == "__main__":
    main()
