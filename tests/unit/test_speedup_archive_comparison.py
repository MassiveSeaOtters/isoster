"""``--compare-to`` must not call an uncontrolled comparison controlled.

``benchmarks/performance/archive_speedup.py`` can report the drift between two
archived sweeps. Reporting the numbers is a measurement; saying *why* they moved
is an inference, and it only holds if everything except the suspected cause was
held fixed.

Two earlier versions got this wrong in ways that both looked fine:

* the first attached a causal explanation unconditionally, so archives from
  different commits and dirty trees still got one;
* the second checked case *counts* and the *platform string*, which let through
  a changed excluded-case configuration (same count, different case) and a
  changed NumPy version or thread count (same platform, different environment).

These tests pin the boundary: what must block attribution, and what must still
be recognised as a genuinely matched pair.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks" / "performance"))

from archive_speedup import (  # noqa: E402
    case_identity,
    compare_archives,
    fingerprint_cases,
)

ARCHIVE = PROJECT_ROOT / "benchmarks" / "performance" / "reference_speedup.json"


@pytest.fixture
def archive() -> dict:
    """The committed archive, used as a realistic baseline for both sides."""
    if not ARCHIVE.is_file():
        pytest.skip(f"{ARCHIVE} not present")
    return json.loads(ARCHIVE.read_text())


def _matched_pair(archive: dict) -> tuple[dict, dict]:
    """Two archives identical in provenance, differing only in the numbers."""
    previous = copy.deepcopy(archive)
    previous["speedup"] = dict(archive["speedup"], median=archive["speedup"]["median"] + 0.9)
    return previous, archive


class TestControlledPairIsRecognised:
    def test_identical_provenance_is_controlled(self, archive):
        previous, current = _matched_pair(archive)
        result = compare_archives(previous, current)
        assert result["controlled"], result["blockers"]
        assert result["blockers"] == []

    def test_differences_are_always_reported(self, archive):
        previous, current = _matched_pair(archive)
        result = compare_archives(previous, current)
        # Reported whether or not attribution is possible.
        assert set(result["differences"]) == {"median", "q1", "q3", "min", "max"}
        assert result["differences"]["median"]["delta"] == pytest.approx(-0.9, abs=1e-6)

    def test_interpretation_avoids_overclaiming(self, archive):
        previous, current = _matched_pair(archive)
        text = compare_archives(previous, current)["interpretation"]
        assert "consistent with" in text
        assert "expected signature" not in text


class TestCaseIdentityBlocksAttribution:
    def test_changed_excluded_case_with_same_count(self, archive):
        """Same number of exclusions, different configurations excluded."""
        previous = copy.deepcopy(archive)
        previous["failed_configurations"] = copy.deepcopy(archive["failed_configurations"])
        previous["failed_configurations"][0]["eps"] = 0.6
        previous["excluded_fingerprint"] = fingerprint_cases(
            [case_identity(entry) for entry in previous["failed_configurations"]]
        )
        result = compare_archives(previous, archive)
        assert not result["controlled"]
        assert any("excluded cases" in blocker for blocker in result["blockers"])
        assert previous["photutils_failures"] == archive["photutils_failures"]

    def test_changed_completed_set_with_same_count(self, archive):
        previous = copy.deepcopy(archive)
        previous["completed_fingerprint"] = "0" * 64
        result = compare_archives(previous, archive)
        assert not result["controlled"]
        assert any("completed cases" in blocker for blocker in result["blockers"])

    def test_missing_fingerprint_blocks_rather_than_passes(self, archive):
        """An archive predating the field must not be silently trusted."""
        previous = copy.deepcopy(archive)
        del previous["completed_fingerprint"]
        result = compare_archives(previous, archive)
        assert not result["controlled"]
        assert any("cannot verify" in blocker for blocker in result["blockers"])


class TestEnvironmentBlocksAttribution:
    @pytest.mark.parametrize(
        "key, value",
        [
            ("numpy", "1.26.4"),
            ("scipy", "1.10.0"),
            ("python", "3.11.9"),
            ("numba", "0.59.0"),
            ("cpu_count", 4),
            ("processor", "something-else"),
        ],
    )
    def test_library_or_cpu_change_with_same_platform(self, archive, key, value):
        previous = copy.deepcopy(archive)
        previous["environment"] = dict(archive["environment"], **{key: value})
        result = compare_archives(previous, archive)
        assert not result["controlled"]
        assert any(key in blocker for blocker in result["blockers"])
        # The platform string is deliberately untouched: it is not sufficient.
        assert previous["environment"]["platform"] == archive["environment"]["platform"]

    def test_threading_change_with_same_platform(self, archive):
        previous = copy.deepcopy(archive)
        previous["environment"] = copy.deepcopy(archive["environment"])
        previous["environment"]["environment_variables"] = dict(
            archive["environment"]["environment_variables"], OMP_NUM_THREADS="1"
        )
        result = compare_archives(previous, archive)
        assert not result["controlled"]
        assert any("environment_variables" in blocker for blocker in result["blockers"])


class TestProvenanceBlocksAttribution:
    def test_different_commit(self, archive):
        previous = copy.deepcopy(archive)
        previous["environment"] = dict(archive["environment"], git_sha="0" * 40)
        result = compare_archives(previous, archive)
        assert not result["controlled"]
        assert any("code revision" in blocker for blocker in result["blockers"])

    def test_dirty_tree(self, archive):
        previous = copy.deepcopy(archive)
        previous["environment"] = dict(
            archive["environment"],
            git_worktree={"dirty": True, "changed_paths": ["x"], "tracked_diff_sha256": "abc"},
        )
        result = compare_archives(previous, archive)
        assert not result["controlled"]
        assert any("provenance" in blocker for blocker in result["blockers"])

    def test_uncontrolled_interpretation_refuses_attribution(self, archive):
        previous = copy.deepcopy(archive)
        previous["environment"] = dict(archive["environment"], git_sha="0" * 40)
        text = compare_archives(previous, archive)["interpretation"]
        assert "NOT ATTRIBUTABLE" in text
        assert "consistent with" not in text


class TestFingerprint:
    def test_order_independent(self):
        cases = [(1.0, 20, 0.0, 0.0, None, 512), (2.0, 50, 0.3, 0.5, 100, 512)]
        assert fingerprint_cases(cases) == fingerprint_cases(list(reversed(cases)))

    def test_sensitive_to_a_single_parameter(self):
        base = [(1.0, 20, 0.0, 0.0, None, 512)]
        changed = [(1.0, 20, 0.6, 0.0, None, 512)]
        assert fingerprint_cases(base) != fingerprint_cases(changed)

    def test_identity_ignores_unknown_keys(self):
        params = {"n": 1.0, "R_e": 20, "eps": 0.0, "pa": 0.0, "noise_snr": None, "image_size": 512}
        assert case_identity(dict(params, unrelated="x")) == case_identity(params)
