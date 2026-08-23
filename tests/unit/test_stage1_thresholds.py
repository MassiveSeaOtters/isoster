"""Part B Stage 1: the accuracy contract, and the truth it is built on.

These guard two classes of defect that review found in earlier drafts, both of
which produced authoritative-looking numbers that were wrong: a truth lookup
that silently yielded no value, and a bar derived from it that failed *open* to
infinity so every measurement passed.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from benchmarks.harmonic_scale.run_harmonic_scale import FIXTURES, ORDERS, PLANTED_HARMONICS
from benchmarks.timing import accuracy_thresholds as at
from benchmarks.timing.stage1_fixtures import render_stage1_fixture, stage1_fixtures, stage1_seed
from benchmarks.utils.sersic_model import (
    analytic_truth_on_aperture,
    aperture_displacement_error_px,
    create_sersic_image_with_harmonics,
    integrated_harmonic_truth,
)


def _meta(pa=0.0, eps=None):
    spec = FIXTURES["sersic_n2_compact"]
    galaxy = spec["galaxy"]
    _, meta = create_sersic_image_with_harmonics(
        n=galaxy["n"],
        R_e=galaxy["R_e"],
        I_e=galaxy["I_e"],
        eps=spec["reference_eps"] if eps is None else eps,
        pa=pa,
        shape=galaxy["shape"],
        center=galaxy["center"],
        harmonics=PLANTED_HARMONICS,
    )
    return meta, spec, galaxy


def _direct_polar_truth(meta, x0, y0, sma, eps, pa, orders, n_phi=65536):
    """Independent polar-angle construction of truth on one aperture."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    aperture_axis_ratio = 1.0 - eps
    polar_radius = sma * aperture_axis_ratio / np.sqrt((aperture_axis_ratio * np.cos(theta)) ** 2 + np.sin(theta) ** 2)
    aperture_x = polar_radius * np.cos(theta)
    aperture_y = polar_radius * np.sin(theta)
    x = x0 + aperture_x * np.cos(pa) - aperture_y * np.sin(pa)
    y = y0 + aperture_x * np.sin(pa) + aperture_y * np.cos(pa)

    galaxy_x0, galaxy_y0 = meta["center"]
    dx, dy = x - galaxy_x0, y - galaxy_y0
    galaxy_pa = meta["pa"]
    u = dx * np.cos(galaxy_pa) + dy * np.sin(galaxy_pa)
    v = -dx * np.sin(galaxy_pa) + dy * np.cos(galaxy_pa)
    galaxy_axis_ratio = 1.0 - meta["eps"]
    elliptical_radius = np.hypot(u, v / galaxy_axis_ratio)
    galaxy_polar_angle = np.arctan2(v, u)

    distortion = np.ones_like(theta)
    for (order, kind), amplitude in meta["harmonics"].items():
        wave = np.sin(order * galaxy_polar_angle) if kind == "sin" else np.cos(order * galaxy_polar_angle)
        distortion += amplitude * wave
    intensities = meta["profile"](elliptical_radius / distortion)
    return {
        order: {
            "s_raw": 2.0 * float(np.mean(intensities * np.sin(order * theta))),
            "c_raw": 2.0 * float(np.mean(intensities * np.cos(order * theta))),
            "mean_intensity": float(np.mean(intensities)),
        }
        for order in orders
    }


class TestApertureTruth:
    """`integrated_harmonic_truth` samples the planted ellipse and cannot take
    a returned geometry, so a free fit could not be compared against truth on
    the aperture it actually used."""

    def test_the_planted_geometry_reproduces_the_reference(self):
        for pa in (0.0, 0.4, -1.1):
            meta, spec, galaxy = _meta(pa=pa)
            reference = integrated_harmonic_truth(meta, 25.0, ORDERS)
            aperture = analytic_truth_on_aperture(meta, *galaxy["center"], 25.0, spec["reference_eps"], pa, ORDERS)
            for order in ORDERS:
                assert aperture[order]["s_raw"] == pytest.approx(reference[order]["s_raw"], rel=1e-9)
                assert aperture[order]["c_raw"] == pytest.approx(reference[order]["c_raw"], rel=1e-9)

    def test_a_shifted_centre_changes_the_answer(self):
        meta, spec, galaxy = _meta()
        x0, y0 = galaxy["center"]
        on = analytic_truth_on_aperture(meta, x0, y0, 25.0, spec["reference_eps"], 0.0, ORDERS)
        off = analytic_truth_on_aperture(meta, x0 + 3.0, y0, 25.0, spec["reference_eps"], 0.0, ORDERS)
        assert off[3]["c_raw"] != pytest.approx(on[3]["c_raw"], rel=1e-6)

    def test_a_changed_ellipticity_changes_the_answer(self):
        meta, spec, galaxy = _meta()
        on = analytic_truth_on_aperture(meta, *galaxy["center"], 25.0, spec["reference_eps"], 0.0, ORDERS)
        off = analytic_truth_on_aperture(meta, *galaxy["center"], 25.0, spec["reference_eps"] + 0.15, 0.0, ORDERS)
        assert off[4]["c_raw"] != pytest.approx(on[4]["c_raw"], rel=1e-6)

    @pytest.mark.parametrize(
        "offset_x,offset_y,eps_delta,pa_delta",
        [
            (3.0, -2.0, 0.0, 0.0),
            (0.0, 0.0, 0.12, 0.0),
            (0.0, 0.0, 0.0, 0.20),
        ],
    )
    def test_non_reference_apertures_use_physical_polar_angle(self, offset_x, offset_y, eps_delta, pa_delta):
        meta, spec, galaxy = _meta(pa=0.37)
        x0, y0 = galaxy["center"]
        aperture = (x0 + offset_x, y0 + offset_y, 25.0, spec["reference_eps"] + eps_delta, 0.37 + pa_delta)
        measured = analytic_truth_on_aperture(meta, *aperture, ORDERS, n_phi=65536)
        direct = _direct_polar_truth(meta, *aperture, ORDERS)
        for order in ORDERS:
            assert measured[order]["s_raw"] == pytest.approx(direct[order]["s_raw"], rel=1e-9, abs=1e-11)
            assert measured[order]["c_raw"] == pytest.approx(direct[order]["c_raw"], rel=1e-9, abs=1e-11)
            assert measured[order]["mean_intensity"] == pytest.approx(direct[order]["mean_intensity"], rel=1e-10)

    def test_a_pi_rotation_is_the_same_ellipse_with_a_flipped_angular_origin(self):
        """PA is periodic modulo pi, but the *amplitudes* are not blindly so.

        Rotating the aperture by pi traces the identical ellipse while moving
        its angular origin by pi, so component m picks up ``cos(m*pi)``: even
        orders are unchanged and **odd orders flip sign**. Any cross-tool
        comparison that folds PA into [0, pi) must carry that sign with it, or
        it will read a convention as a disagreement --- the same class of
        mistake Part A found in the Bender gradient convention.
        """
        meta, spec, galaxy = _meta(pa=0.3)
        base = analytic_truth_on_aperture(meta, *galaxy["center"], 25.0, spec["reference_eps"], 0.3, ORDERS)
        turned = analytic_truth_on_aperture(meta, *galaxy["center"], 25.0, spec["reference_eps"], 0.3 + np.pi, ORDERS)
        for order in ORDERS:
            sign = 1.0 if order % 2 == 0 else -1.0
            assert turned[order]["s_raw"] == pytest.approx(sign * base[order]["s_raw"], abs=1e-9)
            assert turned[order]["c_raw"] == pytest.approx(sign * base[order]["c_raw"], abs=1e-9)
            # The ellipse itself is unchanged, so the ring mean must be too.
            assert turned[order]["mean_intensity"] == pytest.approx(base[order]["mean_intensity"], rel=1e-12)


class TestTheDerivationFailsClosed:
    """An infinite bar is a tolerance every measurement passes. The earlier
    version produced exactly that, for every component, when a truth lookup
    read the wrong keys."""

    def test_a_missing_component_raises(self, monkeypatch):
        monkeypatch.setattr(at, "_component_truth", lambda fixture, sma: {})
        with pytest.raises(ValueError, match="cannot derive a bar"):
            at.ring_statistics("sersic_n2_compact")

    def test_a_nan_truth_raises(self, monkeypatch):
        monkeypatch.setattr(at, "_component_truth", lambda fixture, sma: {c: float("nan") for c in at._COMPONENTS})
        with pytest.raises(ValueError, match="cannot derive a bar"):
            at.ring_statistics("sersic_n2_compact")

    def test_a_zero_truth_raises(self, monkeypatch):
        monkeypatch.setattr(at, "_component_truth", lambda fixture, sma: {c: 0.0 for c in at._COMPONENTS})
        with pytest.raises(ValueError, match="cannot derive a bar"):
            at.ring_statistics("sersic_n2_compact")

    def test_no_derived_bar_is_infinite_or_nan(self):
        contract = at.stage_1_contract()
        for bars in contract["descriptive_harmonic_error_pct_by_component"].values():
            for components in bars.values():
                for value in components.values():
                    assert np.isfinite(value) and value > 0

    def test_the_contract_serializes_without_nan(self):
        import json

        json.dumps(at.stage_1_contract(), allow_nan=False, default=str)

    def test_outer_significance_rule_fails_closed(self, monkeypatch):
        monkeypatch.setattr(
            at, "_component_truth", lambda fixture, sma: {component: 0.0 for component in at._COMPONENTS}
        )
        with pytest.raises(ValueError, match="below the frozen"):
            at.outer_limit_significance()


class TestArbitraryRadius:
    def test_a_bar_exists_for_a_radius_not_in_the_table(self):
        value = at.harmonic_absolute_error_limit_at("sersic_n2_compact", 59.25)
        assert np.isfinite(value) and value > 0

    def test_outside_the_target_interval_raises(self):
        with pytest.raises(ValueError, match="outside the target interval"):
            at.harmonic_absolute_error_limit_at("sersic_n2_compact", 200.0)

    def test_the_table_is_a_sample_of_the_function(self):
        contract = at.stage_1_contract()
        bars = contract["systematic_harmonic_absolute_error_by_ring"]["sersic_n2_compact"]
        for sma, value in bars.items():
            assert at.harmonic_absolute_error_limit_at("sersic_n2_compact", float(sma)) == pytest.approx(
                value, abs=5e-9
            )

    def test_harmonic_limit_is_absolute_and_does_not_divide_by_truth(self):
        fixture = "sersic_n2_compact"
        spec = stage1_fixtures()[fixture]
        x0, y0 = spec["galaxy"]["center"]
        # This valid aperture is only 0.43 px from the planted boundary but
        # rotates the s4 truth through zero. A percentage gate explodes there;
        # an absolute raw-amplitude limit remains finite and unchanged.
        zero_crossing_pa = 0.054053054142185836
        truth = analytic_truth_on_aperture(
            at._fixture_meta(fixture), x0, y0, 25.0, spec["reference_eps"], zero_crossing_pa, ORDERS
        )
        assert abs(truth[4]["s_raw"]) < 1e-9
        assert at.harmonic_absolute_error_limit_at(fixture, 25.0) == pytest.approx(
            at.harmonic_absolute_error_limit_on_aperture(fixture, 25.0)
        )
        assert np.isfinite(at.harmonic_absolute_error_limit_on_aperture(fixture, 25.0))

    def test_intensity_limit_is_absolute_and_depends_only_on_radius(self):
        assert at.ring_intensity_absolute_error_limit_at("sersic_n2_compact", 25.0) == pytest.approx(
            at.ring_intensity_absolute_error_limit_on_aperture("sersic_n2_compact", 25.0)
        )


class TestApertureDisplacement:
    REFERENCE = {"x0": 100.0, "y0": 100.0, "sma": 600.0, "eps": 0.3, "pa": 0.2}

    def test_identical_apertures_have_zero_displacement(self):
        assert aperture_displacement_error_px(self.REFERENCE, self.REFERENCE) == pytest.approx(0.0, abs=1e-12)

    def test_position_angle_is_axis_periodic(self):
        turned = {**self.REFERENCE, "pa": self.REFERENCE["pa"] + np.pi}
        assert aperture_displacement_error_px(self.REFERENCE, turned) == pytest.approx(0.0, abs=1e-10)

    def test_ellipticity_error_scales_with_radius(self):
        changed = {**self.REFERENCE, "eps": self.REFERENCE["eps"] + 0.01}
        assert aperture_displacement_error_px(self.REFERENCE, changed) == pytest.approx(6.0, rel=1e-5)


class TestEligibility:
    """A harmonics-off arm must still have its intensity judged."""

    BASE = {
        "execution_status": "ok",
        "coverage_status": "complete",
        "contamination_status": "clean",
        "harmonic_accuracy_status": "not_applicable",
        "intensity_accuracy_status": "pass",
        "geometry_accuracy_status": "not_applicable",
    }

    def test_a_fixed_harmonics_off_arm_with_good_intensity_is_eligible(self):
        assert at.headline_eligible(self.BASE, harmonics_enabled=False, geometry_free=False)

    def test_an_unevaluated_applicable_metric_is_not_a_pass(self):
        assert not at.headline_eligible(
            {**self.BASE, "intensity_accuracy_status": None}, harmonics_enabled=False, geometry_free=False
        )

    def test_a_failed_intensity_disqualifies_a_harmonics_off_arm(self):
        assert not at.headline_eligible(
            {**self.BASE, "intensity_accuracy_status": "fail"}, harmonics_enabled=False, geometry_free=False
        )

    def test_contamination_alone_disqualifies(self):
        assert not at.headline_eligible(
            {**self.BASE, "contamination_status": "contaminated"},
            harmonics_enabled=False,
            geometry_free=False,
        )

    def test_harmonics_on_cannot_claim_not_applicable(self):
        assert not at.headline_eligible(self.BASE, harmonics_enabled=True, geometry_free=False)

    def test_free_geometry_cannot_claim_not_applicable(self):
        assert not at.headline_eligible(self.BASE, harmonics_enabled=False, geometry_free=True)

    def test_applicability_requires_exact_statuses(self):
        outcome = {
            **self.BASE,
            "harmonic_accuracy_status": "pass",
            "geometry_accuracy_status": "pass",
        }
        assert at.headline_eligible(outcome, harmonics_enabled=True, geometry_free=True)


class TestFamilyUnit:
    def test_the_family_is_one_arm_not_the_whole_grid(self):
        contract = at.stage_1_contract()
        rings = len(stage1_fixtures()["sersic_n2_compact"]["radii"])
        assert contract["ensemble_harmonic_tests_per_arm"] == rings * len(at._COMPONENTS) == 20
        assert contract["ensemble_correction"] == "bonferroni_familywise_rmse_envelope"

    def test_family_partition_is_derived_from_the_arm(self):
        radii = [12.0, 18.0, 25.0, 35.0, 45.0]
        families = at.accuracy_family_members(radii, harmonics_enabled=True, geometry_free=True)
        assert len(families["harmonic"]) == 20
        assert len(families["intensity"]) == 5
        assert len(families["geometry"]) == 5

        fixed_without_harmonics = at.accuracy_family_members(radii, harmonics_enabled=False, geometry_free=False)
        assert set(fixed_without_harmonics) == {"intensity"}

    def test_every_fixture_family_member_is_frozen(self):
        contract = at.stage_1_contract()
        for fixture, spec in stage1_fixtures().items():
            expected_fixed = at.accuracy_family_members(
                list(spec["radii"]), harmonics_enabled=True, geometry_free=False
            )
            expected_natural = at.accuracy_family_members(
                [fraction * spec["galaxy"]["R_e"] for fraction in at.END_TO_END_EVALUATION_RADIUS_FRACTIONS],
                harmonics_enabled=True,
                geometry_free=True,
            )
            frozen = contract["ensemble_family_members_by_fixture_and_scope"][fixture]
            assert frozen["fixed_aperture"] == {family: list(members) for family, members in expected_fixed.items()}
            assert frozen["end_to_end"] == {family: list(members) for family, members in expected_natural.items()}

    def test_accuracy_detects_bias_and_excess_scatter(self):
        tiny_constant_bias = {"x": [1e-12] * at.ENSEMBLE_REALIZATIONS}
        large_scattered_bias = {"x": (10.0 + np.array([-100.0, 100.0] * 12 + [0.0])).tolist()}
        scales = {"x": 1.0}
        assert at.evaluate_accuracy_family(tiny_constant_bias, scales)["family_passed"]
        assert not at.evaluate_accuracy_family(large_scattered_bias, scales)["family_passed"]

    def test_familywise_limit_accepts_the_declared_ideal_boundary(self):
        limit = at.familywise_standardized_rmse_limit(20)
        samples = {f"m{i}": [limit] * at.ENSEMBLE_REALIZATIONS for i in range(20)}
        scales = {name: 1.0 for name in samples}
        assert at.evaluate_accuracy_family(samples, scales)["family_passed"]
        samples["m0"] = [np.nextafter(limit, np.inf)] * at.ENSEMBLE_REALIZATIONS
        assert not at.evaluate_accuracy_family(samples, scales)["family_passed"]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
    def test_accuracy_family_fails_closed_on_an_invalid_scale(self, bad):
        with pytest.raises(ValueError, match="scale"):
            at.evaluate_accuracy_family({"x": [0.0] * at.ENSEMBLE_REALIZATIONS}, {"x": bad})

    def test_accuracy_family_requires_the_frozen_realization_count(self):
        with pytest.raises(ValueError, match="exactly"):
            at.evaluate_accuracy_family({"x": [0.0] * 24}, {"x": 1.0})

    def test_family_scales_are_built_from_member_names(self):
        members = ["s3_raw_major@sma=25", "ring_mean@sma=25"]
        scales = at.ideal_sigma_by_family_member("sersic_n2_compact", members)
        assert scales[members[0]] == pytest.approx(at.harmonic_absolute_error_limit_at("sersic_n2_compact", 25))
        assert scales[members[1]] == pytest.approx(at.ring_intensity_absolute_error_limit_at("sersic_n2_compact", 25))
        with pytest.raises(ValueError, match="no ideal"):
            at.ideal_sigma_by_family_member("sersic_n2_compact", ["eps@sma=25"])

    def test_geometry_family_uses_rms_boundary_displacement(self):
        passed = {"sma=25": [0.5] * at.ENSEMBLE_REALIZATIONS}
        failed = {"sma=25": [1.01] * at.ENSEMBLE_REALIZATIONS}
        assert at.evaluate_geometry_accuracy_family(passed)["family_passed"]
        assert not at.evaluate_geometry_accuracy_family(failed)["family_passed"]

    def test_systematic_family_uses_absolute_error(self):
        assert at.evaluate_systematic_accuracy_family({"x": -0.5}, {"x": 0.5})["family_passed"]
        assert not at.evaluate_systematic_accuracy_family({"x": -0.5001}, {"x": 0.5})["family_passed"]


class TestCompleteFixtureIdentity:
    def test_every_fixture_freezes_every_rendering_input(self):
        for fixture in at.stage_1_contract()["fixtures"].values():
            assert set(fixture) >= {
                "n",
                "R_e",
                "I_e",
                "shape",
                "center",
                "eps",
                "pa",
                "harmonics",
                "fixed_aperture_radii",
                "end_to_end_evaluation_radii",
                "scope",
                "scientific_identity",
                "independent_scientific_fixture",
            }

    def test_the_scientific_input_contract_freezes_noise_and_rendering(self):
        scientific_input = at.stage_1_contract()["scientific_input"]
        assert scientific_input["renderer_function"].endswith("create_sersic_image_with_harmonics")
        assert scientific_input["pixel_sampling"] == "pixel_centres_without_subpixel_integration"
        assert scientific_input["psf"] == "none"
        assert scientific_input["background"] == 0.0
        assert scientific_input["noise_arms"]["noiseless"]["distribution"] == "none"
        noisy = scientific_input["noise_arms"]["gaussian_reference"]
        assert noisy["distribution"] == "independent_gaussian"
        assert noisy["snr_at_r_e"] == at.REFERENCE_SNR
        assert noisy["realizations"] == at.ENSEMBLE_REALIZATIONS
        assert scientific_input["seed_derivation"] == "seed_blocks[stage] + realization_index"
        assert scientific_input["harmonic_conversion_requirement"] == "every_source_ring_observed_valid"
        assert scientific_input["tool_harmonic_settings"]["isoster"]["use_eccentric_anomaly"] is False
        assert scientific_input["tool_harmonic_settings"]["autoprof"] == {
            "ap_isoclip": True,
            "ap_iso_interpolate_start": 1000.0,
            "ap_isoband_fixed": True,
            "ap_isoband_width": 0.1,
            "required_observed_sampling_mode": "line_interpolated",
        }

    def test_seed_blocks_are_disjoint_and_exact(self):
        calibration = {stage1_seed("calibration", index) for index in range(at.ENSEMBLE_REALIZATIONS)}
        campaign = {stage1_seed("campaign", index) for index in range(at.ENSEMBLE_REALIZATIONS)}
        assert calibration.isdisjoint(campaign)
        assert min(calibration) == 100_000
        assert min(campaign) == 60_260_822

    def test_the_frozen_noise_renderer_is_repeatable(self):
        first, first_variance, first_meta = render_stage1_fixture(
            "sersic_n2_compact", "gaussian_reference", stage="calibration", realization_index=3
        )
        second, second_variance, second_meta = render_stage1_fixture(
            "sersic_n2_compact", "gaussian_reference", stage="calibration", realization_index=3
        )
        assert np.array_equal(first, second)
        assert np.array_equal(first_variance, second_variance)
        assert first_meta["seed"] == second_meta["seed"] == 100_003
        assert np.all(first_variance == first_meta["noise_sigma"] ** 2)

    def test_every_scope_has_frozen_evaluation_radii(self):
        contract = at.stage_1_contract()
        for fixture, identity in contract["fixtures"].items():
            assert len(identity["fixed_aperture_radii"]) == 5
            assert len(identity["end_to_end_evaluation_radii"]) == 5
            r_e = identity["R_e"]
            assert identity["end_to_end_evaluation_radii"] == pytest.approx(
                [fraction * r_e for fraction in at.END_TO_END_EVALUATION_RADIUS_FRACTIONS]
            )

    def test_host_specific_contamination_rule_names_the_host(self):
        host = at.stage_1_contract()["benchmark_host"]
        assert host == {
            "system": "Darwin",
            "machine": "arm64",
            "machine_model": "MacBookPro18,2",
            "logical_cpu_count": 10,
            "thermal_command": "/usr/bin/pmset -g therm",
        }
        assert at.benchmark_host_mismatches(host) == []
        assert "logical_cpu_count" in at.benchmark_host_mismatches({**host, "logical_cpu_count": 8})[0]

    def test_wide_canvas_is_a_workload_duplicate_not_independent_evidence(self):
        fixture = at.stage_1_contract()["fixtures"]["wide_canvas_961"]
        assert fixture["scientific_identity"] == "sersic_n2_compact"
        assert fixture["independent_scientific_fixture"] is False


def test_human_readable_generator_executes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["accuracy_thresholds.py"])
    at.main()
    output = capsys.readouterr().out
    assert "family-wise normalized RMSE" in output
    assert "geometry=5" in output
