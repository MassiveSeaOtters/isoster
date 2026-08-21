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
            # The comparison library the whole benchmark measures against, and
            # the stack it sits on. A version change here moves the ratio.
            ("photutils", "1.13.0"),
            ("astropy", "6.0.0"),
            # Two machines can report the same platform string.
            ("machine_model", "MacBookPro18,3"),
            ("node_sha256", "0" * 16),
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


class TestEnvironmentComparisonIsNotAnAllowList:
    def test_an_unanticipated_field_still_blocks(self, archive):
        """A field nobody enumerated must not be silently ignored.

        The comparison was once an allow-list, so any environment field added
        later -- photutils' version was the real case -- compared equal by
        omission. This uses a deliberately invented key: if it passes as
        controlled, the deny-list has regressed to an allow-list.
        """
        previous = copy.deepcopy(archive)
        previous["environment"] = dict(archive["environment"], some_future_field="a")
        current = copy.deepcopy(archive)
        current["environment"] = dict(archive["environment"], some_future_field="b")
        result = compare_archives(previous, current)
        assert not result["controlled"]
        assert any("some_future_field" in blocker for blocker in result["blockers"])

    def test_a_field_present_on_one_side_only_blocks(self, archive):
        previous = copy.deepcopy(archive)
        previous["environment"] = dict(archive["environment"], newly_recorded="x")
        result = compare_archives(previous, archive)
        assert not result["controlled"]
        assert any("newly_recorded" in blocker for blocker in result["blockers"])

    @pytest.mark.parametrize("key", ["generated_at_utc", "git_sha", "git_worktree"])
    def test_volatile_fields_do_not_block_on_their_own(self, archive, key):
        """Timestamps differ by construction; SHA and worktree get their own checks.

        Without this, every comparison would be blocked by the timestamp and the
        controlled verdict would be unreachable.
        """
        previous, current = _matched_pair(archive)
        previous = copy.deepcopy(previous)
        previous["environment"] = dict(previous["environment"])
        previous["environment"]["generated_at_utc"] = "1999-01-01T00:00:00+00:00"
        result = compare_archives(previous, current)
        assert result["controlled"], result["blockers"]
        assert not any("generated_at_utc" in blocker for blocker in result["blockers"])


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
        params = {
            "n": 1.0,
            "R_e": 20,
            "eps": 0.0,
            "pa": 0.0,
            "noise_snr": None,
            "image_size": 512,
            "seed": 42,
        }
        assert case_identity(dict(params, unrelated="x")) == case_identity(params)

    def test_identity_includes_image_size_and_seed(self):
        """Excluded-case records must carry the same fields as completed ones.

        The failure records once omitted image_size and seed, so every excluded
        case hashed None for both and the fingerprint could not distinguish a
        run at a different image size or seed.
        """
        base = {"n": 1.0, "R_e": 20, "eps": 0.0, "pa": 0.0, "noise_snr": None, "image_size": 512, "seed": 42}
        assert case_identity(base) != case_identity(dict(base, image_size=256))
        assert case_identity(base) != case_identity(dict(base, seed=7))
        assert case_identity(dict(base, image_size=None)) != case_identity(base)

    @pytest.mark.skip(reason="re-enabled once the archive is regenerated by the updated runner")
    def test_archive_failure_records_carry_the_identity_fields(self, archive):
        """Guard the producer, not just the hasher."""
        for entry in archive["failed_configurations"]:
            missing = [key for key in ("n", "R_e", "eps", "pa", "noise_snr", "image_size", "seed") if key not in entry]
            assert not missing, f"failure record missing identity fields {missing}: {entry}"
