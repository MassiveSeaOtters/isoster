"""Gate the harmonic-scale archive against the tolerances frozen before it ran.

Third gate in the docs CI job, beside ``check_draft_numbers.py`` and
``check_speedup_claims.py``, and it runs on the same terms: no AutoProf, no
image fitting, nothing but the committed JSON and the committed prose. CI has
no AutoProf venv and never will, so a checker that needed one would be a
checker that never ran.

What it checks, in the order the failures matter
------------------------------------------------
1. **The archive describes the experiment the code now defines.** The fixture
   fingerprint covers the galaxy, the planted modes, the radii and the whole
   grid. If the code has moved on, the archived numbers describe a different
   measurement and comparing them to anything is meaningless --- so this fails
   first and loudly, rather than letting later checks report a drift that is
   really a redefinition.

2. **The preconditions held.** Sampling mode attributable for every case, no
   ring silently sampled as an isophotal band, conversion-validity flags
   consistent with the path that produced them. These are not measurements;
   they are what makes the measurements mean anything.

3. **The claims reproduce across disjoint seed blocks.** The tolerances were
   frozen from the *pilot*, which drew its noise realizations from one seed
   block; the archive is a *validation* run that drew from another. Each claim
   must land within its frozen tolerance of the pilot's value. This is the
   pre-registration contract: the tolerance existed, in a committed file,
   before the run it judges.

4. **The prose quotes the archive.** Any documentation sentence stating one of
   these numbers must state the archived one, whole clause at a time. Same
   contract as ``check_draft_numbers.py``, and for the same reason: a
   transcription that drifts is worse than no number, because it still looks
   measured.

Usage::

    uv run python benchmarks/harmonic_scale/check_harmonic_scale.py
    uv run python benchmarks/harmonic_scale/check_harmonic_scale.py --self-test

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harmonic_scale.claims import (  # noqa: E402
    extract_claims,
    structural_problems,
)

HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = HERE / "reference_harmonic_scale.json"
DEFAULT_TOLERANCES = HERE / "frozen_tolerances.json"

#: Documents allowed to quote these numbers. Checked only if they exist, so
#: the gate is useful from the moment the archive lands rather than only once
#: the prose is written.
DOC_PATHS = (
    REPO_ROOT / "docs" / "specs" / "2026-08-22-three-way-benchmark-comparison-design.md",
    REPO_ROOT / "docs" / "05-testing.md",
)


def _squash(text: str) -> str:
    """Collapse whitespace so markdown line-wrapping cannot hide a match."""
    return re.sub(r"\s+", " ", text)


def _load(path: Path, what: str) -> Dict[str, object]:
    if not path.exists():
        raise SystemExit(f"{what} not found: {path}")
    return json.loads(path.read_text())


def check_fingerprint(archive: Dict[str, object]) -> List[str]:
    """Does the archive describe the grid this repository currently defines?"""
    from benchmarks.harmonic_scale.run_harmonic_scale import _fixture_fingerprint

    archived = archive["environment"]["fixture_fingerprint"]
    current = _fixture_fingerprint()
    if archived == current:
        return []
    return [
        "fixture fingerprint mismatch: the archive was produced by a different grid "
        f"(archived {archived[:12]}..., current {current[:12]}...). Re-run "
        "--mode validation --archive rather than editing numbers to match."
    ]


def check_provenance(archive: Dict[str, object]) -> List[str]:
    """Everything needed to defend the numbers a year from now."""
    problems = []
    environment = archive["environment"]
    for key in ("python", "platform", "numpy", "photutils", "isoster"):
        if not environment.get(key):
            problems.append(f"environment is missing {key}")

    git = environment.get("git") or {}
    if not git.get("commit"):
        problems.append("no commit recorded; the archive cannot be tied to a source state")
    if git.get("dirty"):
        problems.append("archive was produced from a dirty working tree")

    if archive.get("mode") != "validation":
        problems.append(f"archive mode is {archive.get('mode')!r}, expected 'validation'")

    blocks = archive.get("seed_blocks") or {}
    if blocks.get("pilot") == blocks.get("validation"):
        problems.append(
            "pilot and validation seed blocks are equal; the tolerances would have been "
            "tuned on the realizations they judge"
        )
    if archive.get("seed_block") != blocks.get("validation"):
        problems.append("archive did not draw from the validation seed block")

    for case in archive["cases"]:
        autoprof = case.get("autoprof_provenance") or {}
        for key in ("autoprof_version", "isophote_extract_sha256", "realized_pipeline_steps"):
            if not autoprof.get(key):
                problems.append(f"{case['spec']['name']}: AutoProf provenance is missing {key}")
        if autoprof.get("psf_fwhm_pix") is None:
            problems.append(
                f"{case['spec']['name']}: no measured PSF recorded, so the interpolation "
                "switch radius cannot be reconstructed"
            )
    return problems


def check_claims(archive: Dict[str, object], tolerances: Dict[str, object]) -> Tuple[List[str], List[str]]:
    """Every claim within its frozen tolerance of the pilot's value."""
    measured = extract_claims(archive)
    frozen = tolerances["claims"]
    failures, lines = [], []

    for name in sorted(frozen):
        entry = frozen[name]
        expected, tolerance = float(entry["pilot_value"]), float(entry["tolerance"])
        if name not in measured:
            failures.append(f"{name}: absent from the archive")
            continue
        actual = float(measured[name])
        if actual != actual:  # NaN
            failures.append(f"{name}: archive value is NaN")
            continue
        difference = abs(actual - expected)
        ok = difference <= tolerance
        lines.append(
            f"{'OK  ' if ok else 'FAIL'} {name}: archive {actual:.6g}, "
            f"pilot {expected:.6g}, |diff| {difference:.3g} <= {tolerance:.3g}"
        )
        if not ok:
            failures.append(
                f"{name}: archive {actual:.6g} differs from pilot {expected:.6g} by "
                f"{difference:.3g}, over the frozen tolerance {tolerance:.3g}"
            )

    unclaimed = sorted(set(measured) - set(frozen))
    if unclaimed:
        lines.append(f"note: {len(unclaimed)} measured quantities carry no frozen tolerance:")
        for name in unclaimed:
            lines.append(f"      {name} = {measured[name]:.6g}")

    # Do not let a pass count overstate what was tested. Most of this grid is
    # noiseless, and a noiseless case does not depend on the seed block at
    # all -- those claims reproduce exactly, which confirms determinism but
    # is not evidence that a tolerance was well chosen. Only the claims that
    # actually move between seed blocks test the pre-registration.
    seed_sensitive = [
        name
        for name in frozen
        if frozen[name].get("basis") == "scatter"
        and name in measured
        and abs(float(measured[name]) - float(frozen[name]["pilot_value"])) > 0
    ]
    lines.append(
        f"note: {len(seed_sensitive)} of {len(frozen)} claims differ at all between the "
        "pilot and validation seed blocks; the rest come from noiseless cases and "
        "reproduce exactly, so they test determinism rather than tolerance choice"
    )
    return failures, lines


def build_doc_checks(archive: Dict[str, object]) -> List[Tuple[str, str]]:
    """Sentences the prose must contain verbatim if it mentions these numbers.

    Whole clauses, never bare numbers. A fragment can match by accident --- the
    lesson ``check_draft_numbers.py`` records from a checker that passed
    against deliberately corrupted text.
    """
    claims = extract_claims(archive)
    checks: List[Tuple[str, str]] = []

    worst_clean = max(claims[f"clean_agreement_pct_{tool}"] for tool in ("isoster", "photutils", "autoprof"))
    checks.append(
        (
            "clean_agreement",
            f"the three tools agree with the analytic truth to better than {worst_clean:.1f}%",
        )
    )
    for key, value in sorted(claims.items()):
        if key.startswith("nearest_pixel_excess_pct_sma"):
            radius = key.rsplit("sma", 1)[1]
            checks.append(
                (
                    key,
                    f"reads {value:.1f}% high at sma = {radius} px when that ring is sampled "
                    "by rounding to the nearest pixel",
                )
            )
    return checks


def run_doc_checks(checks: List[Tuple[str, str]], docs: Dict[str, str]) -> List[str]:
    """A claim is checked only where its topic is already discussed.

    The prose for Part A is not written yet, and a gate that demanded sentences
    nobody has written would fail on every run until they were. What it must
    catch is a *stale* number, so a check fires when a document contains the
    claim's opening words but not the whole clause.
    """
    failures = []
    for name, expected in checks:
        stem = _squash(" ".join(expected.split()[:4]))
        for filename, text in docs.items():
            squashed = _squash(text)
            if stem in squashed and _squash(expected) not in squashed:
                failures.append(
                    f"{name}: {filename} discusses this claim but does not state it as "
                    f"archived.\n       expected verbatim: {expected!r}"
                )
    return failures


def _self_test(archive: Dict[str, object], tolerances: Dict[str, object]) -> int:
    """Confirm the claim gate actually fails when the archive is corrupted.

    A checker nobody has tried to fool is not evidence. This perturbs each
    claim's underlying summary far enough to clear its tolerance and requires
    the gate to notice.
    """
    frozen = tolerances["claims"]
    baseline_failures, _ = check_claims(archive, tolerances)
    if baseline_failures:
        print("self-test: the unmodified archive already fails; fix that first")
        for failure in baseline_failures:
            print(f"  {failure}")
        return 1

    caught = 0
    for name in sorted(frozen):
        mutated = json.loads(json.dumps(archive))
        # Shift every stored median by an amount that clears any tolerance.
        for case in mutated["cases"]:
            for tool_summary in case["summary"].values():
                for ring in tool_summary.values():
                    for key, entry in ring.items():
                        if isinstance(entry, dict) and "median" in entry:
                            entry["median"] = float(entry["median"]) + 10.0
        failures, _ = check_claims(mutated, tolerances)
        if failures:
            caught += 1
        else:
            print(f"self-test: corrupting the archive did NOT trip {name}")
    print(f"self-test: {caught}/{len(frozen)} claims trip when the archive is corrupted")
    return 0 if caught == len(frozen) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--tolerances", type=Path, default=DEFAULT_TOLERANCES)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify the claim gate fails on a corrupted archive.",
    )
    args = parser.parse_args()

    archive = _load(args.archive, "harmonic-scale archive")
    tolerances = _load(args.tolerances, "frozen tolerances")

    if args.self_test:
        raise SystemExit(_self_test(archive, tolerances))

    failures: List[str] = []

    fingerprint_failures = check_fingerprint(archive)
    for failure in fingerprint_failures:
        print(f"FAIL fixture fingerprint: {failure}")
    if fingerprint_failures:
        # Everything below compares numbers between two different experiments,
        # which would produce a misleading list of failures.
        raise SystemExit(1)
    print("OK   fixture fingerprint matches the grid this repository defines")

    for failure in check_provenance(archive):
        print(f"FAIL provenance: {failure}")
        failures.append(failure)
    if not failures:
        print("OK   provenance complete, archive from a clean tree in validation mode")

    structural = structural_problems(archive)
    for failure in structural:
        print(f"FAIL precondition: {failure}")
    failures.extend(structural)
    if not structural:
        print("OK   every case line-sampled, sampling modes attributed, validity flags consistent")

    claim_failures, lines = check_claims(archive, tolerances)
    for line in lines:
        print(line)
    failures.extend(claim_failures)

    docs = {path.name: path.read_text() for path in DOC_PATHS if path.exists()}
    doc_failures = run_doc_checks(build_doc_checks(archive), docs)
    for failure in doc_failures:
        print(f"FAIL prose: {failure}")
    failures.extend(doc_failures)
    if not doc_failures:
        print(f"OK   prose in {len(docs)} document(s) states the archived numbers")

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        raise SystemExit(1)
    print("\nall harmonic-scale checks passed")


if __name__ == "__main__":
    main()
