"""A6 coverage for the harmonic-scale grid, claims and gate --- no AutoProf.

CI has no AutoProf venv, so anything that matters must be testable without
one. What is testable without one is the *design*: that the grid is the
designed grid rather than a Cartesian product, that the claim extraction says
what it claims to say, that the archive gate refuses the things it exists to
refuse, and that the tolerance freeze is mechanical rather than hand-fitted.

The measurement itself needs AutoProf and lives in
``tests/integration/test_harmonic_scale_autoprof.py``, which skips cleanly
when the venv is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.harmonic_scale import claims as claims_module
from benchmarks.harmonic_scale.run_harmonic_scale import (
    AUTOPROF_DEFAULT_INTERPOLATE_START,
    FIXTURES,
    INTERPOLATE_EVERYWHERE,
    PILOT_SEED_BLOCK,
    PLANTED_HARMONICS,
    RADII,
    SINGLE_PLANTED_HARMONIC,
    VALIDATION_SEED_BLOCK,
    _fixture_fingerprint,
    _ratio,
    _refuse_archive,
    build_grid,
    freeze_tolerances,
    use_fixture,
)


class TestGridStructure:
    """The grid must be the designed one, not the product of its axes."""

    def test_grid_is_far_smaller_than_the_cartesian_product(self):
        # Six active axes at their listed value counts. The design exists
        # precisely so this number is never run.
        product = 3 * 3 * 2 * 2 * 2 * 2  # interp x eps x pa x clip x background x noise
        assert len(build_grid()) < product

    def test_every_case_is_a_delta_from_the_reference(self):
        grid = {case["name"]: case for case in build_grid()}
        reference = grid["reference"]
        axes = ("eps", "pa_deg", "isoclip", "interpolate_start", "background_offset", "snr", "planted", "set_psf")
        for name, case in grid.items():
            if name == "reference":
                continue
            changed = [axis for axis in axes if case[axis] != reference[axis]]
            assert changed, f"{name} changes nothing; it duplicates the reference"

    def test_one_factor_cases_change_exactly_one_factor(self):
        grid = {case["name"]: case for case in build_grid()}
        reference = grid["reference"]
        axes = ("eps", "pa_deg", "isoclip", "interpolate_start", "background_offset", "snr", "planted", "set_psf")
        for name, case in grid.items():
            if case["kind"] != "one_factor":
                continue
            changed = [axis for axis in axes if case[axis] != reference[axis]]
            assert changed == [changed[0]], f"{name} is labelled one_factor but changes {changed}"

    def test_interactions_change_exactly_two_factors(self):
        grid = {case["name"]: case for case in build_grid()}
        reference = grid["reference"]
        axes = ("eps", "pa_deg", "isoclip", "interpolate_start", "background_offset", "snr", "planted", "set_psf")
        interactions = [c for c in grid.values() if c["kind"] == "interaction"]
        assert len(interactions) == 4, "A3 names four interactions"
        for case in interactions:
            changed = [axis for axis in axes if case[axis] != reference[axis]]
            # radius is an axis of every case rather than a per-case setting,
            # so radius x interpolation shows up as one changed setting.
            assert 1 <= len(changed) <= 2, f"{case['name']} changes {changed}"

    def test_both_interpolation_settings_are_exercised(self):
        settings = {case["interpolate_start"] for case in build_grid()}
        assert AUTOPROF_DEFAULT_INTERPOLATE_START in settings, "the default arm is the whole point"
        assert INTERPOLATE_EVERYWHERE in settings

    def test_radii_straddle_the_switch_at_the_default_setting(self):
        # The measured PSF on this fixture is near 4 px, so the default puts
        # the switch near 20 px. Rings on both sides are what make the
        # radius x interpolation interaction measurable at all.
        switch = AUTOPROF_DEFAULT_INTERPOLATE_START * 4.0
        assert any(r < switch for r in RADII)
        assert any(r > switch for r in RADII)

    def test_unknown_axis_in_a_case_is_rejected(self):
        from benchmarks.harmonic_scale.run_harmonic_scale import _case

        with pytest.raises(ValueError, match="unknown axes"):
            _case("bad", "one_factor", "why", eccentricity=0.3)

    def test_seed_blocks_are_disjoint(self):
        # Enough separation that no realistic realization count could overlap.
        assert abs(PILOT_SEED_BLOCK - VALIDATION_SEED_BLOCK) > 100_000


class TestPlantedModes:
    def test_four_modes_carry_both_components_at_each_order(self):
        for order in (3, 4):
            assert (order, "sin") in PLANTED_HARMONICS
            assert (order, "cos") in PLANTED_HARMONICS

    def test_amplitudes_are_all_distinct(self):
        # A transposed index must show up as a wrong number, not as agreement.
        values = list(PLANTED_HARMONICS.values())
        assert len(set(values)) == len(values)

    def test_control_plants_exactly_one_mode(self):
        assert len(SINGLE_PLANTED_HARMONIC) == 1


class TestRatioFloor:
    """The floor is relative to the ring, and it is on the truth."""

    def test_a_negligible_truth_gives_nan_not_a_huge_ratio(self):
        assert np.isnan(_ratio(1e-4, 1e-9, scale=2.0))

    def test_a_meaningful_truth_divides_normally(self):
        assert _ratio(2.0, 4.0, scale=4.0) == pytest.approx(0.5)

    def test_the_floor_scales_with_the_ring(self):
        # The same truth is calibratable next to a small signal and not next
        # to a large one; no absolute threshold can express that.
        assert not np.isnan(_ratio(1.0, 1e-3, scale=1e-2))
        assert np.isnan(_ratio(1.0, 1e-3, scale=1e3))

    def test_a_non_finite_measurement_is_nan(self):
        assert np.isnan(_ratio(float("nan"), 1.0, scale=1.0))


def _synthetic_results(mode="validation", **overrides):
    """A minimal well-formed run, so the gates can be tested without AutoProf."""

    def ring(value):
        return {key: {"n": 1, "median": value, "min": value, "max": value} for key in claims_module.RATIO_KEYS}

    def case(name, isoclip=True, snr=None, interpolated=True, value=1.0, **spec):
        summary = {
            tool: {f"sma={sma:g}": {**ring(value), "statuses": ["measured"]} for sma in RADII}
            for tool in claims_module.TOOLS
        }
        return {
            "spec": {
                "name": name,
                "kind": "one_factor",
                "eps": 0.3,
                "pa_deg": 0.0,
                "isoclip": isoclip,
                "interpolate_start": INTERPOLATE_EVERYWHERE,
                "background_offset": 0.0,
                "snr": snr,
                "n_realizations": 1,
                "planted": "four_modes",
                **spec,
            },
            "harmonic_conversion_valid": isoclip,
            "harmonic_conversion_reason": "" if isoclip else "eccentric_anomaly_basis_mixes_orders",
            "autoprof_provenance": {
                "autoprof_version": "1.3.4",
                "isophote_extract_sha256": "0" * 64,
                "realized_pipeline_steps": ["background"],
                "psf_fwhm_pix": 4.0,
                "sampling_mode": {
                    "attribution_ok": True,
                    "attribution_note": "",
                    "all_rings_line_sampled": True,
                    "band_sampling_calls": 0,
                    "per_ring_interpolated": [interpolated] * len(RADII),
                },
            },
            "realizations": [],
            "summary": summary,
        }

    results = {
        "mode": mode,
        "seed_block": VALIDATION_SEED_BLOCK if mode == "validation" else PILOT_SEED_BLOCK,
        "seed_blocks": {"pilot": PILOT_SEED_BLOCK, "validation": VALIDATION_SEED_BLOCK},
        "environment": {
            "python": "3.12.0",
            "platform": "test",
            "numpy": "2.0",
            "photutils": "2.3.0",
            "isoster": "1.0.0",
            "git": {"commit": "abc123", "branch": "test", "dirty": False, "dirty_paths": []},
            "fixture_fingerprint": _fixture_fingerprint(),
        },
        "cases": [case("reference"), case("background_offset", background_offset=50.0)],
    }
    results.update(overrides)
    return results


class TestStructuralGate:
    """Preconditions that no numeric tolerance would catch."""

    def test_a_clean_run_has_no_structural_problems(self):
        assert claims_module.structural_problems(_synthetic_results()) == []

    def test_band_sampling_is_reported(self):
        results = _synthetic_results()
        sampling = results["cases"][0]["autoprof_provenance"]["sampling_mode"]
        sampling["all_rings_line_sampled"] = False
        sampling["band_sampling_calls"] = 3
        problems = claims_module.structural_problems(results)
        assert any("isophotal-band" in p for p in problems)

    def test_unattributable_sampling_mode_is_reported(self):
        results = _synthetic_results()
        results["cases"][0]["autoprof_provenance"]["sampling_mode"]["attribution_ok"] = False
        results["cases"][0]["autoprof_provenance"]["sampling_mode"]["attribution_note"] = "counts differ"
        assert any("not attributable" in p for p in claims_module.structural_problems(results))

    def test_validity_flag_inconsistent_with_the_path_is_reported(self):
        results = _synthetic_results()
        # Clipping off puts AutoProf on the eccentric-anomaly basis, where the
        # conversion is not valid; claiming otherwise is the failure A5 exists
        # to prevent.
        results["cases"][0]["spec"]["isoclip"] = False
        results["cases"][0]["harmonic_conversion_valid"] = True
        assert any("harmonic_conversion_valid" in p for p in claims_module.structural_problems(results))

    def test_an_unmeasured_ring_is_reported(self):
        results = _synthetic_results()
        results["cases"][0]["summary"]["isoster"][f"sma={RADII[0]:g}"]["statuses"] = ["gradient_unavailable"]
        assert any("gradient_unavailable" in p for p in claims_module.structural_problems(results))


class TestArchiveRefusal:
    """``--archive`` must refuse anything that cannot back a published claim."""

    def test_a_dirty_tree_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks.harmonic_scale.run_harmonic_scale.TOLERANCES_PATH", tmp_path / "t.json")
        (tmp_path / "t.json").write_text("{}")
        monkeypatch.setattr(
            "benchmarks.harmonic_scale.run_harmonic_scale._git_state",
            lambda: {"commit": "abc", "branch": "b", "dirty": False, "dirty_paths": []},
        )
        results = _synthetic_results()
        results["environment"]["git"] = {
            "commit": "abc",
            "branch": "b",
            "dirty": True,
            "dirty_paths": ["M isoster/fitting.py"],
        }
        assert any("dirty" in p for p in _refuse_archive(results, None))

    def test_a_pilot_run_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks.harmonic_scale.run_harmonic_scale.TOLERANCES_PATH", tmp_path / "t.json")
        (tmp_path / "t.json").write_text("{}")
        monkeypatch.setattr(
            "benchmarks.harmonic_scale.run_harmonic_scale._git_state",
            lambda: {"commit": "abc", "branch": "b", "dirty": False, "dirty_paths": []},
        )
        assert any("validation" in p for p in _refuse_archive(_synthetic_results(mode="pilot"), None))

    def test_a_single_case_run_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks.harmonic_scale.run_harmonic_scale.TOLERANCES_PATH", tmp_path / "t.json")
        (tmp_path / "t.json").write_text("{}")
        monkeypatch.setattr(
            "benchmarks.harmonic_scale.run_harmonic_scale._git_state",
            lambda: {"commit": "abc", "branch": "b", "dirty": False, "dirty_paths": []},
        )
        assert any("whole grid" in p for p in _refuse_archive(_synthetic_results(), "reference"))

    def test_missing_frozen_tolerances_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks.harmonic_scale.run_harmonic_scale.TOLERANCES_PATH", tmp_path / "absent.json")
        problems = _refuse_archive(_synthetic_results(), None)
        assert any("frozen" in p or "before the validation run" in p for p in problems)

    def test_a_clean_validation_run_is_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks.harmonic_scale.run_harmonic_scale.TOLERANCES_PATH", tmp_path / "t.json")
        (tmp_path / "t.json").write_text("{}")
        monkeypatch.setattr(
            "benchmarks.harmonic_scale.run_harmonic_scale._git_state",
            lambda: {"commit": "abc", "branch": "b", "dirty": False, "dirty_paths": []},
        )
        assert _refuse_archive(_synthetic_results(), None) == []

    def test_a_tree_dirtied_since_the_run_is_refused(self, tmp_path, monkeypatch):
        """The hole ``--rearchive`` opened: a clean *recorded* state is not enough.

        Re-archiving promotes an older run, so the state that matters is the
        one now as well as the one then. Reading only the recorded state let a
        dirty tree straight through.
        """
        monkeypatch.setattr("benchmarks.harmonic_scale.run_harmonic_scale.TOLERANCES_PATH", tmp_path / "t.json")
        (tmp_path / "t.json").write_text("{}")
        monkeypatch.setattr(
            "benchmarks.harmonic_scale.run_harmonic_scale._git_state",
            lambda: {"commit": "abc", "branch": "b", "dirty": True, "dirty_paths": ["M x.py"]},
        )
        assert any("now" in problem for problem in _refuse_archive(_synthetic_results(), None))

    def test_an_undeterminable_tree_state_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("benchmarks.harmonic_scale.run_harmonic_scale.TOLERANCES_PATH", tmp_path / "t.json")
        (tmp_path / "t.json").write_text("{}")
        monkeypatch.setattr(
            "benchmarks.harmonic_scale.run_harmonic_scale._git_state",
            lambda: {"commit": None, "branch": None, "dirty": None, "dirty_paths": []},
        )
        problems = _refuse_archive(_synthetic_results(), None)
        assert any("could not determine" in problem for problem in problems)


class TestFingerprint:
    def test_the_fingerprint_moves_when_the_fixture_moves(self, monkeypatch):
        before = _fixture_fingerprint()
        monkeypatch.setattr("benchmarks.harmonic_scale.run_harmonic_scale.RADII", (10.0, 20.0))
        assert _fixture_fingerprint() != before

    def test_the_fingerprint_is_stable_when_nothing_moves(self):
        assert _fixture_fingerprint() == _fixture_fingerprint()


class TestFreezeIsMechanical:
    """The freeze applies one rule; it does not fit a margin to a number."""

    def test_freezing_requires_a_pilot(self):
        frozen = freeze_tolerances(_synthetic_results(mode="pilot"))
        assert frozen["frozen_from"]["mode"] == "pilot"

    def test_every_tolerance_records_which_rule_produced_it(self):
        frozen = freeze_tolerances(_synthetic_results(mode="pilot"))
        for name, entry in frozen["claims"].items():
            assert entry["basis"] in {"scatter", "invariance_floor", "deterministic_floor"}, name
            assert entry["tolerance"] > 0, name

    def test_the_policy_is_recorded_alongside_the_numbers(self):
        frozen = freeze_tolerances(_synthetic_results(mode="pilot"))
        assert set(frozen["policy"]) >= {"safety_factor", "deterministic_floor_pct", "invariance_floor"}


class TestClaimExtraction:
    def test_a_perfect_run_claims_zero_disagreement(self):
        extracted = claims_module.extract_claims(_synthetic_results())
        for tool in claims_module.TOOLS:
            assert extracted[f"clean_agreement_pct_{tool}"] == pytest.approx(0.0)

    def test_a_biased_run_is_reported_in_percent(self):
        results = _synthetic_results()
        for case in results["cases"]:
            for per_tool in case["summary"].values():
                for ring in per_tool.values():
                    for key in claims_module.RATIO_KEYS:
                        ring[key]["median"] = 1.05
        extracted = claims_module.extract_claims(results)
        assert extracted["clean_agreement_pct_isoster"] == pytest.approx(5.0)

    def test_the_amplitude_magnitude_is_not_used_by_any_claim(self):
        # It is rotation-invariant, so it would hide exactly the sign and
        # basis errors several claims exist to catch.
        assert not any("amp" in key for key in claims_module.RATIO_KEYS)

    def test_background_invariance_notices_a_real_shift(self):
        results = _synthetic_results()
        offset_case = next(c for c in results["cases"] if c["spec"]["name"] == "background_offset")
        for per_tool in offset_case["summary"].values():
            for ring in per_tool.values():
                ring["s3_raw_ratio"]["median"] = 1.02
        extracted = claims_module.extract_claims(results)
        assert extracted["background_invariance_isoster"] == pytest.approx(0.02)


class TestFrozenTolerancesFile:
    """The committed tolerance file is part of the pre-registration."""

    def test_it_exists_and_records_its_provenance(self):
        from benchmarks.harmonic_scale.run_harmonic_scale import TOLERANCES_PATH

        if not TOLERANCES_PATH.exists():
            pytest.skip("tolerances not frozen yet")
        frozen = json.loads(TOLERANCES_PATH.read_text())
        assert frozen["frozen_from"]["mode"] == "pilot"
        assert frozen["frozen_from"]["seed_block"] == PILOT_SEED_BLOCK
        assert frozen["claims"]

    def test_it_was_frozen_from_the_pilot_seed_block_not_the_validation_one(self):
        from benchmarks.harmonic_scale.run_harmonic_scale import TOLERANCES_PATH

        if not TOLERANCES_PATH.exists():
            pytest.skip("tolerances not frozen yet")
        frozen = json.loads(TOLERANCES_PATH.read_text())
        assert frozen["frozen_from"]["seed_block"] != VALIDATION_SEED_BLOCK


class TestFixtureRegistry:
    """Each campaign is a separate measurement, and the first one is frozen."""

    def test_the_archived_fixture_fingerprint_still_matches(self):
        """The frozen campaign's grid must not move. Ever.

        Its archive is committed and gated, and the fingerprint is what ties
        the two together. Adding the ``set_psf`` axis broke this once --- every
        case gained a ``set_psf: None`` key, which changed the hash and would
        have invalidated an archive whose numbers had not changed at all. This
        test is the standing guard against repeating that.
        """
        archive_path = (
            Path(__file__).resolve().parents[2] / "benchmarks" / "harmonic_scale" / "reference_harmonic_scale.json"
        )
        if not archive_path.exists():
            pytest.skip("frozen archive not present")
        archived = json.loads(archive_path.read_text())["environment"]["fixture_fingerprint"]
        use_fixture("sersic_n2_compact")
        assert _fixture_fingerprint() == archived

    def test_the_frozen_campaign_adds_no_extra_cases(self):
        # Extra cases would change its grid, hence its fingerprint.
        assert FIXTURES["sersic_n2_compact"]["extra_cases"] == ()

    def test_seed_blocks_are_disjoint_across_every_campaign(self):
        blocks = [spec[key] for spec in FIXTURES.values() for key in ("pilot_seed_block", "validation_seed_block")]
        assert len(set(blocks)) == len(blocks)
        for a in blocks:
            for b in blocks:
                if a != b:
                    assert abs(a - b) > 1000, "seed blocks must not overlap for any run length"

    def test_each_campaign_has_its_own_archive_and_tolerances(self):
        archives = [spec["archive"] for spec in FIXTURES.values()]
        tolerances = [spec["tolerances"] for spec in FIXTURES.values()]
        assert len(set(archives)) == len(archives)
        assert len(set(tolerances)) == len(tolerances)

    def test_the_fixtures_are_actually_different_galaxies(self):
        galaxies = [tuple(sorted(spec["galaxy"].items())) for spec in FIXTURES.values()]
        assert len(set(galaxies)) == len(galaxies)
        # A second fixture that only changed size would not test the gradient
        # the Bender normalization divides by.
        indices = {spec["galaxy"]["n"] for spec in FIXTURES.values()}
        assert len(indices) > 1, "the campaigns must differ in Sersic index, not only in size"

    def test_an_unknown_fixture_is_rejected(self):
        with pytest.raises(SystemExit, match="unknown fixture"):
            use_fixture("no_such_galaxy")

    def test_switching_fixture_switches_the_radii_and_the_archive(self):
        use_fixture("sersic_n4_extended")
        import benchmarks.harmonic_scale.run_harmonic_scale as runner

        assert runner.RADII == FIXTURES["sersic_n4_extended"]["radii"]
        assert runner.ARCHIVE_PATH.name == FIXTURES["sersic_n4_extended"]["archive"]
        use_fixture("sersic_n2_compact")
        assert runner.RADII == FIXTURES["sersic_n2_compact"]["radii"]


class TestSecondFixtureGrid:
    """The PSF axis, which exists because the PSF is assumed rather than measured."""

    @pytest.fixture(autouse=True)
    def _second(self):
        use_fixture("sersic_n4_extended")
        yield
        use_fixture("sersic_n2_compact")

    def test_it_adds_the_psf_cases(self):
        names = {case["name"] for case in build_grid()}
        assert {"psf_set_8", "psf_x_interpolate", "threshold_matched_control"} <= names

    def test_the_matched_threshold_case_differs_in_both_factors(self):
        """Different option values, identical product --- that is the whole point."""
        grid = {case["name"]: case for case in build_grid()}
        default = grid["interpolate_default"]
        matched = grid["threshold_matched_control"]
        assert matched["interpolate_start"] != default["interpolate_start"]
        assert matched["set_psf"] != default["set_psf"]
        # AutoProf assumes 4.0 px when set_psf is None.
        default_threshold = default["interpolate_start"] * 4.0
        matched_threshold = matched["interpolate_start"] * matched["set_psf"]
        assert matched_threshold == pytest.approx(default_threshold)

    def test_the_radii_straddle_both_thresholds(self):
        radii = FIXTURES["sersic_n4_extended"]["radii"]
        for threshold in (5.0 * 4.0, 5.0 * 8.0):
            assert any(r < threshold for r in radii), threshold
            assert any(r > threshold for r in radii), threshold

    def test_its_fingerprint_differs_from_the_frozen_campaign(self):
        second = _fixture_fingerprint()
        use_fixture("sersic_n2_compact")
        assert second != _fixture_fingerprint()
