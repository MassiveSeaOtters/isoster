"""Gate the A4 Track 2 archives, on the same terms as the other three gates.

No AutoProf, no fitting: the committed JSON and the committed tolerances only.

What it checks
--------------
1. **The archive describes the measurement this code defines** --- fingerprint
   over the galaxy, radii, planted modes, gradient step and grid. A changed
   fingerprint is a redefinition, not drift, and is reported as such.
2. **Provenance and a clean tree**, both when the run was made and now.
3. **Every claim within its frozen tolerance** of the pilot's value, the
   pilot having drawn from a disjoint seed block.
4. **The licensing verdict is the one the criteria imply.** Recomputed here
   rather than trusted, because a stored verdict that no longer follows from
   the stored numbers is the failure mode a gate exists to catch.

Usage::

    uv run python benchmarks/harmonic_scale/check_gradient_reconstruction.py
    uv run python benchmarks/harmonic_scale/check_gradient_reconstruction.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmarks.harmonic_scale.run_gradient_reconstruction as runner  # noqa: E402

HERE = Path(__file__).resolve().parent


def _campaigns() -> List[tuple[str, Path, Path]]:
    """Every Track 2 campaign that has been archived.

    A campaign named but not yet run is skipped; one archived without frozen
    tolerances is an error, because that is an archive nothing can judge.
    """
    found = []
    for fixture in sorted(runner.SEED_BLOCKS):
        runner.ACTIVE_FIXTURE = fixture
        archive, tolerances = runner.archive_path(), runner.tolerances_path()
        if not archive.exists():
            print(f"note: {fixture} has no Track 2 archive yet; skipping")
            continue
        if not tolerances.exists():
            raise SystemExit(f"{fixture} has an archive but no frozen tolerances ({tolerances.name})")
        found.append((fixture, archive, tolerances))
    if not found:
        raise SystemExit("no Track 2 archives found")
    return found


def check_one(fixture: str, archive: Dict[str, object], tolerances: Dict[str, object]) -> List[str]:
    failures: List[str] = []
    runner.ACTIVE_FIXTURE = fixture

    if archive["environment"]["fixture_fingerprint"] != runner.fixture_fingerprint():
        print("FAIL fingerprint: the archive was produced by a different measurement")
        return ["fingerprint mismatch"]
    print("OK   fingerprint matches the measurement this repository defines")

    git = archive["environment"].get("git") or {}
    if not git.get("commit"):
        failures.append("no commit recorded")
    if git.get("dirty"):
        failures.append("archive produced from a dirty working tree")
    if archive.get("mode") != "validation":
        failures.append(f"mode is {archive.get('mode')!r}, expected 'validation'")
    if archive.get("seed_block") != runner.SEED_BLOCKS[fixture]["validation"]:
        failures.append("archive did not draw from the validation seed block")
    if archive.get("gradient_step") != tolerances["policy"]["gradient_step"]:
        failures.append(
            "the archive and its tolerances used different gradient steps; they describe different quantities"
        )
    if not failures:
        print("OK   provenance complete, validation mode, clean tree, matching gradient step")

    measured = runner.extract_claims(archive)
    for name, entry in sorted(tolerances["claims"].items()):
        if name not in measured:
            failures.append(f"{name}: absent from the archive")
            continue
        actual, expected = float(measured[name]), float(entry["pilot_value"])
        tolerance = float(entry["tolerance"])
        if actual != actual:
            failures.append(f"{name}: archive value is NaN")
            continue
        if abs(actual - expected) > tolerance:
            failures.append(
                f"{name}: archive {actual:.6g} differs from pilot {expected:.6g} by "
                f"{abs(actual - expected):.3g}, over the frozen tolerance {tolerance:.3g}"
            )
    # A tolerance comparable to the value it guards is not a constraint. Say
    # so, for the same reason dormant prose checks are reported: a pass that
    # could not have failed must not read like one that could.
    weak = [
        name
        for name, entry in tolerances["claims"].items()
        if abs(float(entry["pilot_value"])) > 0 and float(entry["tolerance"]) > 0.5 * abs(float(entry["pilot_value"]))
    ]
    print(f"OK   {len(tolerances['claims'])} claims checked against frozen tolerances")
    if weak:
        print(
            f"     note: {len(weak)} claim(s) have a tolerance over half their own value and so "
            f"constrain little: {', '.join(sorted(weak))}"
        )

    # The verdict must follow from the numbers, not merely accompany them.
    recomputed = runner.evaluate_licensing(archive)
    stored = archive.get("licensing") or {}
    for key in ("criterion_1_beats_point_derivative", "licensed_on_reference_configuration"):
        if bool(stored.get(key)) != bool(recomputed[key]):
            failures.append(
                f"stored licensing {key}={stored.get(key)} does not follow from the archived "
                f"numbers (recomputed {recomputed[key]})"
            )
    for name, regime in recomputed["regimes"].items():
        stored_regime = (stored.get("regimes") or {}).get(name, {})
        if stored_regime.get("criterion_2") != regime["criterion_2"]:
            failures.append(f"stored criterion_2 for {name} does not follow from the archived numbers")
    if not any("licensing" in f or "criterion_2" in f for f in failures):
        verdict = "licensed" if recomputed["licensed_on_reference_configuration"] else "NOT licensed"
        print(
            f"OK   licensing verdict recomputes: {verdict} on the reference configuration "
            f"(criterion 1 margin {recomputed['criterion_1_margin']:.0f}x)"
        )
    return failures


def _self_test() -> int:
    """A corrupted archive must fail. A checker nobody has tried to fool is not evidence."""
    worst = 0
    for fixture, archive_path, tolerances_path in _campaigns():
        archive = json.loads(archive_path.read_text())
        tolerances = json.loads(tolerances_path.read_text())
        if check_one(fixture, archive, tolerances):
            print(f"self-test: {fixture} already fails unmodified; fix that first")
            worst = 1
            continue

        mutated = json.loads(json.dumps(archive))
        for case in mutated["cases"]:
            for ring in case["summary"].values():
                for entry in ring.values():
                    if isinstance(entry, dict) and "median" in entry:
                        entry["median"] = float(entry["median"]) + 5.0
        caught_claims = bool(check_one(fixture, mutated, tolerances))

        # And a verdict that no longer follows from its numbers.
        flipped = json.loads(json.dumps(archive))
        flipped["licensing"]["licensed_on_reference_configuration"] = not flipped["licensing"][
            "licensed_on_reference_configuration"
        ]
        caught_verdict = any("licensing" in f for f in check_one(fixture, flipped, tolerances))
        print(f"self-test {fixture}: corrupted numbers caught={caught_claims}, flipped verdict caught={caught_verdict}")
        if not (caught_claims and caught_verdict):
            worst = 1
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(_self_test())

    failures: List[str] = []
    for fixture, archive_path, tolerances_path in _campaigns():
        print(f"=== {archive_path.name}")
        found = check_one(fixture, json.loads(archive_path.read_text()), json.loads(tolerances_path.read_text()))
        for failure in found:
            print(f"FAIL {failure}")
        failures.extend(found)
        print()

    if failures:
        print(f"{len(failures)} check(s) failed")
        raise SystemExit(1)
    print("all Track 2 gradient-reconstruction checks passed")


if __name__ == "__main__":
    main()
