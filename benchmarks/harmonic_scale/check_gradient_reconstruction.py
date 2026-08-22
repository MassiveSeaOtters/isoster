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
5. **The prose quotes the archive.** Any document sentence stating one of
   these numbers must state the archived value, so the write-up cannot drift
   from what was measured. Same contract, and the same stem rule, as the
   harmonic-scale gate.

Usage::

    uv run python benchmarks/harmonic_scale/check_gradient_reconstruction.py
    uv run python benchmarks/harmonic_scale/check_gradient_reconstruction.py --self-test
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmarks.harmonic_scale.run_gradient_reconstruction as runner  # noqa: E402
from benchmarks.harmonic_scale.check_harmonic_scale import (  # noqa: E402
    DOC_PATHS,
    _squash,
    run_doc_checks,
)

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
    # The fixture fingerprint above says the same thing was *measured*. This
    # says the same thing is being *claimed* about it. Without it, editing a
    # reduction while leaving the frozen file in place would judge a validation
    # value computed one way against a pilot value computed another, and pass.
    frozen_claims = tolerances["policy"].get("claims_fingerprint")
    if frozen_claims is None:
        failures.append("the frozen tolerances predate the claim-definition fingerprint; re-freeze from the pilot")
    elif frozen_claims != runner.claims_fingerprint():
        failures.append(
            "the claim definitions have changed since these tolerances were frozen; "
            "the pilot values describe different quantities. Re-freeze from the pilot."
        )
    if not failures:
        print("OK   provenance complete, validation mode, clean tree, matching gradient step and claim definitions")

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
    #
    # Two quite different things land in that bucket and must not be reported
    # as one. A *measured* tolerance that is large says the claim itself is
    # unstable, which is a real weakness. A claim sitting under the
    # deterministic floor is merely small: it reproduces exactly, and the
    # floor is a fixed allowance rather than a statement about its spread.
    # Listing both together would train the reader to skip the line.
    weak_measured, under_floor = [], []
    for name, entry in tolerances["claims"].items():
        value, tolerance = abs(float(entry["pilot_value"])), float(entry["tolerance"])
        if value <= 0 or tolerance <= 0.5 * value:
            continue
        (under_floor if entry.get("basis") == "deterministic_floor" else weak_measured).append(name)
    by_reduction: Dict[str, int] = {}
    for entry in tolerances["claims"].values():
        by_reduction[str(entry.get("reduction", "unspecified"))] = (
            by_reduction.get(str(entry.get("reduction", "unspecified")), 0) + 1
        )
    breakdown = ", ".join(f"{count} {name}" for name, count in sorted(by_reduction.items()))
    print(f"OK   {len(tolerances['claims'])} claims checked against frozen tolerances ({breakdown})")
    if weak_measured:
        print(
            f"     note: {len(weak_measured)} claim(s) have a *measured* tolerance over half their "
            f"own value, so the claim is genuinely unstable and constrains little: "
            f"{', '.join(sorted(weak_measured))}"
        )
    if under_floor:
        floor = tolerances["policy"]["deterministic_floor_pct"]
        print(
            f"     note: {len(under_floor)} deterministic claim(s) sit below the {floor} floor. "
            f"They reproduce exactly; the floor is a fixed allowance, not a measured spread."
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
            f"OK   licensing verdict recomputes from the worst_ring claims: {verdict} on the "
            f"reference configuration (criterion 1 margin {recomputed['criterion_1_margin']:.0f}x)"
        )
    return failures


def build_doc_checks(fixture: str, archive: Dict[str, object]) -> List[tuple[str, str, str]]:
    """Whole clauses the prose must contain, never bare numbers.

    Every stem identifies the campaign and carries **none** of the value it
    guards. That rule is not cosmetic: a stem containing its own number stops
    matching when the number is corrupted, so the check goes dormant instead
    of failing, and a dormant check reads exactly like a passing one. Seven of
    nine numbers were editable in silence for that reason once already.
    """
    from benchmarks.harmonic_scale.run_harmonic_scale import FIXTURES

    runner.ACTIVE_FIXTURE = fixture
    claims = runner.extract_claims(archive)
    licensing = archive["licensing"]
    label = FIXTURES[fixture]["label"]
    checks = [
        (
            f"gradient_agreement_{fixture}",
            f"on the {label} fixture the matched secant reproduces isoster's gradient",
            f"on the {label} fixture the matched secant reproduces isoster's gradient on the clean "
            f"configuration to {claims['worst_ring_gradient_agreement_pct_clean']:.3f}% at the worst "
            f"ring and {claims['typical_ring_gradient_agreement_pct_clean']:.3f}% at the typical one",
        ),
        (
            f"criterion_1_margin_{fixture}",
            f"on the {label} fixture the matched secant beats the point derivative",
            f"on the {label} fixture the matched secant beats the point derivative by "
            f"{licensing['criterion_1_margin']:.0f}x",
        ),
        (
            f"convention_offset_{fixture}",
            f"on the {label} fixture the forward secant and the point derivative disagree",
            f"on the {label} fixture the forward secant and the point derivative disagree by "
            f"{claims['worst_ring_secant_vs_point_derivative_pct']:.1f}% at the worst ring",
        ),
    ]
    # The regime table decides where the licence applies, so every cell is
    # guarded. The stem is the campaign label and the regime name --- both
    # stable, neither a guarded value.
    for name, regime in sorted(licensing["regimes"].items()):
        if regime.get("criterion_2") is None:
            continue
        verdict = "yes" if regime["criterion_2"] else "no"
        checks.append(
            (
                f"regime_{fixture}_{name}",
                f"| {label} | `{name}` |",
                f"| {label} | `{name}` | {regime['gradient_agreement_pct']:.2f}% | "
                f"{regime['raw_agreement_pct']:.2f}% | {regime['bender_agreement_pct']:.2f}% | "
                f"{regime['budget_pct']:.2f}% | {verdict} |",
            )
        )
    return checks


def check_prose(campaigns: List[tuple[str, Dict[str, object]]]) -> List[str]:
    docs = {path.name: path.read_text() for path in DOC_PATHS if path.exists()}
    if not docs:
        print("note: no checked document exists yet; the prose gate has nothing to read")
        return []
    checks = [check for fixture, archive in campaigns for check in build_doc_checks(fixture, archive)]
    failures = run_doc_checks(checks, docs)
    if not failures:
        print(f"OK   prose in {len(docs)} document(s) states the archived Track 2 numbers")
    return failures


def _prose_self_test(campaigns: List[tuple[str, Dict[str, object]]]) -> bool:
    """A stale number in the prose must actually be caught.

    Built by moving each archived value and requiring the claim to fail. A
    claim no document states yet is skipped rather than counted as a pass.
    """
    docs = {path.name: path.read_text() for path in DOC_PATHS if path.exists()}
    live, missed = [], []
    for fixture, archive in campaigns:
        for name, stem, expected in build_doc_checks(fixture, archive):
            if not any(_squash(stem) in _squash(text) for text in docs.values()):
                continue
            live.append(name)
            moved = json.loads(json.dumps(archive))
            for case in moved["cases"]:
                for ring in case["summary"].values():
                    for entry in ring.values():
                        if isinstance(entry, dict) and "median" in entry:
                            entry["median"] = float(entry["median"]) * 1.5 + 1.0
            moved["licensing"] = runner.evaluate_licensing(moved)
            corrupted = {n: e for n, _, e in build_doc_checks(fixture, moved)}
            if corrupted.get(name) == expected:
                missed.append(name)
    if not live:
        print("self-test: no Track 2 prose claims are stated in any checked document")
        return True
    print(f"self-test: {len(live) - len(missed)}/{len(live)} prose claims trip when the archived value moves")
    for name in missed:
        print(f"  MISSED {name}: the prose can drift from the archive without failing")
    return not missed


def _self_test() -> int:
    """A corrupted archive must fail. A checker nobody has tried to fool is not evidence."""
    worst = 0
    campaigns: List[tuple[str, Dict[str, object]]] = []
    for fixture, archive_path, tolerances_path in _campaigns():
        archive = json.loads(archive_path.read_text())
        tolerances = json.loads(tolerances_path.read_text())
        if check_one(fixture, archive, tolerances):
            print(f"self-test: {fixture} already fails unmodified; fix that first")
            worst = 1
            continue

        # One claim at a time, and the failure must name *that* claim.
        #
        # This previously applied a single global mutation and scored a pass
        # whenever any failure appeared, so a claim that survived corruption
        # was invisible. The claim definitions here are declarative, so the
        # mutation can be targeted exactly: perturb only the columns one claim
        # reduces over, and require that claim to be the one that trips.
        runner.ACTIVE_FIXTURE = fixture
        cases = {case["spec"]["name"]: case for case in archive["cases"]}
        survivors = []
        for definition in runner.CLAIM_DEFINITIONS:
            if str(definition["case"]) not in cases:
                continue
            for reduction in runner.CLAIM_REDUCTIONS:
                claim = f"{reduction}_{definition['stem']}"
                if claim not in tolerances["claims"]:
                    continue
                mutated = json.loads(json.dumps(archive))
                target = {c["spec"]["name"]: c for c in mutated["cases"]}[str(definition["case"])]
                keys = runner._selected_keys(target, definition)
                for ring in target["summary"].values():
                    for key in keys:
                        entry = ring.get(key)
                        if isinstance(entry, dict) and "median" in entry:
                            # Scale as well as shift: several Track 2 claims are
                            # ratios, and a pure offset need not move them.
                            entry["median"] = float(entry["median"]) * 1.9 + 7.0
                with contextlib.redirect_stdout(io.StringIO()):
                    found = check_one(fixture, mutated, tolerances)
                if not any(failure.startswith(f"{claim}:") for failure in found):
                    survivors.append(claim)
        caught_claims = not survivors
        for claim in survivors:
            print(f"  MISSED {claim}: corrupting its own columns does not trip it")
        print(f"self-test {fixture}: per-claim mutation, {len(survivors)} survivor(s)")

        # And a verdict that no longer follows from its numbers.
        flipped = json.loads(json.dumps(archive))
        flipped["licensing"]["licensed_on_reference_configuration"] = not flipped["licensing"][
            "licensed_on_reference_configuration"
        ]
        caught_verdict = any("licensing" in f for f in check_one(fixture, flipped, tolerances))
        print(f"self-test {fixture}: corrupted numbers caught={caught_claims}, flipped verdict caught={caught_verdict}")
        if not (caught_claims and caught_verdict):
            worst = 1
        campaigns.append((fixture, archive))
    if not _prose_self_test(campaigns):
        worst = 1
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(_self_test())

    failures: List[str] = []
    campaigns: List[tuple[str, Dict[str, object]]] = []
    for fixture, archive_path, tolerances_path in _campaigns():
        print(f"=== {archive_path.name}")
        archive = json.loads(archive_path.read_text())
        campaigns.append((fixture, archive))
        found = check_one(fixture, archive, json.loads(tolerances_path.read_text()))
        for failure in found:
            print(f"FAIL {failure}")
        failures.extend(found)
        print()

    prose = check_prose(campaigns)
    for failure in prose:
        print(f"FAIL prose: {failure}")
    failures.extend(prose)

    if failures:
        print(f"{len(failures)} check(s) failed")
        raise SystemExit(1)
    print("all Track 2 gradient-reconstruction checks passed")


if __name__ == "__main__":
    main()
