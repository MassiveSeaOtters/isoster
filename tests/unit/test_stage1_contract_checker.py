"""Fail-closed behavior of the Stage 1 contract and prose checker."""

from __future__ import annotations

from pathlib import Path

from benchmarks.timing import check_accuracy_thresholds as checker
from benchmarks.timing.accuracy_thresholds import stage_1_contract


def _computed_contract():
    return checker.json.loads(checker.json.dumps(stage_1_contract()))


def test_removing_a_guarded_prose_claim_is_a_failure(tmp_path: Path, monkeypatch):
    original = checker.SPEC.read_text()
    missing_claim = original.replace("limits are 1.48253102", "bounds were 1.48253102")
    spec = tmp_path / "spec.md"
    spec.write_text(missing_claim)
    monkeypatch.setattr(checker, "SPEC", spec)

    frozen = _computed_contract()
    failures = checker.check_prose(frozen)

    assert any(failure.startswith("harmonic_rmse_limit:") and "missing" in failure for failure in failures)


def test_each_prose_self_test_moves_only_its_target(monkeypatch):
    frozen = _computed_contract()
    changed_paths = []
    original_check = checker.check_prose

    def recording_check(candidate):
        differences = checker.compare(candidate, frozen)
        changed_paths.append(differences)
        return original_check(candidate)

    monkeypatch.setattr(checker, "check_prose", recording_check)
    assert checker._prose_self_test(frozen)
    assert len(changed_paths) == len(checker.prose_claims(frozen))
    assert all(1 <= len(differences) <= 5 for differences in changed_paths)
