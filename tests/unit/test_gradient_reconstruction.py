"""A4 Track 2: the matched-secant reconstruction, without AutoProf.

The arithmetic and the design are testable in CI; the measurement is not.
What matters most here is the *convention* — that the reconstruction forms
the same quantity isoster divides by, rather than the most accurate gradient
available. Getting that wrong would not fail loudly: it would quietly put
AutoProf's Bender coefficients ~12% away from isoster's and look like a
disagreement between tools.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.harmonic_scale.conventions import (
    DEFAULT_GRADIENT_STEP,
    comparison_radius,
    matched_secant_gradient,
)
from benchmarks.harmonic_scale.run_gradient_reconstruction import (
    CLAIM_DEFINITIONS,
    CLAIM_REDUCTIONS,
    GRADIENT_STEP,
    SEED_BLOCKS,
    build_grid,
    claims_fingerprint,
    evaluate_licensing,
    extract_claims,
    fixture_fingerprint,
    freeze_tolerances,
    structural_validity,
)


class TestMatchedSecant:
    def test_it_is_a_secant_over_the_isoster_interval(self):
        # isoster: gradient_sma = sma*(1+astep); gradient = (mean_g-mean_c)/delta_r
        sma, step = 25.0, 0.1
        near, far = 100.0, 88.0
        expected = (far - near) / (sma * step)
        assert matched_secant_gradient(near, far, sma, step) == pytest.approx(expected)

    def test_the_comparison_radius_matches_isoster_growth(self):
        assert comparison_radius(25.0, 0.1) == pytest.approx(27.5)
        assert comparison_radius(40.0, 0.05) == pytest.approx(42.0)

    def test_the_default_step_is_isoster_default_astep(self):
        from isoster.config import IsosterConfig

        assert DEFAULT_GRADIENT_STEP == pytest.approx(IsosterConfig().astep)

    def test_a_falling_profile_gives_a_negative_gradient(self):
        assert matched_secant_gradient(100.0, 90.0, 25.0) < 0

    def test_a_degenerate_interval_is_nan_not_an_exception(self):
        # sma = 0 is a legal row in isoster's output (a documented sentinel),
        # and it must not raise or return a fabricated number.
        assert np.isnan(matched_secant_gradient(1.0, 2.0, 0.0))

    def test_a_constant_profile_gives_exactly_zero(self):
        assert matched_secant_gradient(50.0, 50.0, 25.0) == 0.0

    def test_it_differs_from_a_point_derivative_on_a_curved_profile(self):
        """The whole reason the campaign exists, expressed as arithmetic.

        On a convex falling profile the forward secant is smaller in magnitude
        than the derivative at the near end. Any reconstruction that returned
        the point derivative would sit systematically away from what isoster
        divides by, and this pins the sign of that offset.
        """
        sma, step = 25.0, 0.1
        scale = 10.0

        def profile(r):
            return 100.0 * np.exp(-r / scale)

        secant = matched_secant_gradient(profile(sma), profile(sma * (1 + step)), sma, step)
        point = -100.0 / scale * np.exp(-sma / scale)
        assert abs(secant) < abs(point)

    def test_an_additive_offset_cancels_exactly(self):
        # A constant contributes nothing to a difference, so a background
        # error must not move the reconstructed gradient at all.
        plain = matched_secant_gradient(100.0, 88.0, 25.0)
        offset = matched_secant_gradient(100.0 + 500.0, 88.0 + 500.0, 25.0)
        assert plain == pytest.approx(offset, abs=1e-12)

    def test_a_multiplicative_rescale_scales_the_gradient(self):
        plain = matched_secant_gradient(100.0, 88.0, 25.0)
        scaled = matched_secant_gradient(1000.0, 880.0, 25.0)
        assert scaled == pytest.approx(10.0 * plain)


class TestGridDesign:
    def test_the_grid_uses_the_shared_interval(self):
        assert GRADIENT_STEP == DEFAULT_GRADIENT_STEP

    def test_no_case_runs_the_eccentric_anomaly_path(self):
        # The reconstruction is scoped to the polar-resampled path. Producing
        # numbers for a conversion the design refuses would invite quoting them.
        assert all(case["isoclip"] for case in build_grid())

    def test_the_background_offset_null_case_exists(self):
        # A constant cancels in a difference, so this case must come out at
        # zero; it is the sharpest check on the grid.
        assert any(case["background_offset"] != 0.0 for case in build_grid())

    def test_noise_cases_carry_many_realizations(self):
        noisy = [case for case in build_grid() if case["snr"] is not None]
        assert noisy, "a secant amplifies noise; the grid must probe it"
        assert all(case["n_realizations"] >= 25 for case in noisy)

    def test_unknown_axis_is_rejected(self):
        from benchmarks.harmonic_scale.run_gradient_reconstruction import _case

        with pytest.raises(ValueError, match="unknown axes"):
            _case("bad", "one_factor", "why", gradient_step=0.2)

    def test_the_fingerprint_covers_the_gradient_step(self, monkeypatch):
        """Changing the interval changes the measurement, so it must change the hash."""
        import benchmarks.harmonic_scale.run_gradient_reconstruction as runner

        before = fixture_fingerprint()
        monkeypatch.setattr(runner, "GRADIENT_STEP", 0.2)
        assert fixture_fingerprint() != before


class TestSeedBlocks:
    def test_they_are_disjoint_from_each_other_and_from_the_a3_campaigns(self):
        from benchmarks.harmonic_scale.run_harmonic_scale import FIXTURES

        blocks = [b for spec in SEED_BLOCKS.values() for b in spec.values()]
        blocks += [spec[key] for spec in FIXTURES.values() for key in ("pilot_seed_block", "validation_seed_block")]
        assert len(set(blocks)) == len(blocks)
        for a in blocks:
            for b in blocks:
                if a != b:
                    assert abs(a - b) > 1000, f"{a} and {b} overlap for any realistic run length"


class TestClaimExtraction:
    def _results(self, gradient_pct=0.05, bender_pct=0.4, raw_pct=0.4, rings=("sma=12",)):
        def ring(scale=1.0):
            entry = {
                key: {"n": 1, "median": value, "min": value, "max": value}
                for key, value in (
                    ("b0_secant_vs_isoster_pct", gradient_pct * scale),
                    ("median_secant_vs_isoster_pct", 0.9),
                    ("isoster_vs_point_derivative_pct", 12.0),
                    ("b0_secant_vs_point_derivative_pct", 12.0),
                )
            }
            for order in (3, 4):
                for prefix in ("a", "b"):
                    entry[f"{prefix}{order}_bender_vs_isoster_pct"] = {"n": 1, "median": bender_pct}
                for prefix in ("s", "c"):
                    entry[f"{prefix}{order}_raw_autoprof_vs_truth_pct"] = {"n": 1, "median": 99.0}
                    entry[f"{prefix}{order}_raw_autoprof_vs_isoster_pct"] = {"n": 1, "median": raw_pct}
            entry["statuses"] = ["measured"]
            return entry

        names = [case["name"] for case in build_grid()]
        summary = {name: ring(scale) for scale, name in enumerate(rings, start=1)}
        # isoclip / interpolate_start are what structural validity reads:
        # polar basis, and interpolated rather than nearest-pixel sampling.
        spec = {"snr": None, "n_realizations": 1, "isoclip": True, "interpolate_start": 100.0}
        return {"cases": [{"spec": {"name": n, **spec}, "summary": summary} for n in names]}

    def test_it_reports_the_gradient_agreement(self):
        claims = extract_claims(self._results(gradient_pct=0.05))
        assert claims["worst_ring_gradient_agreement_pct_clean"] == pytest.approx(0.05)
        assert claims["typical_ring_gradient_agreement_pct_clean"] == pytest.approx(0.05)

    def test_it_reports_the_wrong_estimator_penalty_separately(self):
        # The median column must be quantified, not dismissed.
        claims = extract_claims(self._results())
        assert claims["worst_ring_median_estimator_penalty_pct"] == pytest.approx(0.9)

    def test_it_reports_the_convention_offset(self):
        # The finding that forced the design belongs in the archive.
        claims = extract_claims(self._results())
        assert claims["worst_ring_secant_vs_point_derivative_pct"] == pytest.approx(12.0)

    def test_both_acceptance_quantities_are_present(self):
        claims = extract_claims(self._results())
        assert "worst_ring_bender_agreement_pct_reference" in claims
        assert "worst_ring_raw_agreement_pct_reference" in claims

    def test_the_raw_baseline_is_tool_to_tool_not_tool_to_truth(self):
        """Criterion 2 must compare like with like.

        A raw-versus-truth number and a Bender-versus-isoster number are not
        comparable: under noise both tools see the same realization, so their
        errors correlate and the tool-to-tool gap is far smaller than either
        gap to truth. The pilot read raw 34.5% against Bender 9.9% at
        S/N = 30 under the old formulation, which would have suggested
        normalizing improved matters.
        """
        claims = extract_claims(self._results(raw_pct=0.4))
        # The synthetic fixture sets vs-truth to 99% and vs-isoster to raw_pct;
        # the claim must pick up the latter.
        assert claims["worst_ring_raw_agreement_pct_reference"] == pytest.approx(0.4)

    def test_normalizing_a_worse_bender_than_raw_is_visible(self):
        """Criterion 2 must be able to fail, not only to pass."""
        claims = extract_claims(self._results(bender_pct=9.0, raw_pct=0.4))
        assert claims["worst_ring_bender_agreement_pct_reference"] > claims["worst_ring_raw_agreement_pct_reference"]

    def test_every_claim_is_archived_under_both_reductions(self):
        claims = extract_claims(self._results())
        for definition in CLAIM_DEFINITIONS:
            for reduction in CLAIM_REDUCTIONS:
                assert f"{reduction}_{definition['stem']}" in claims
        assert len(claims) == len(CLAIM_DEFINITIONS) * len(CLAIM_REDUCTIONS)

    def test_the_two_reductions_actually_differ(self):
        """A max and a median that always agree would hide the instability
        the split exists to separate, so the fixture must have several rings."""
        claims = extract_claims(self._results(rings=("sma=12", "sma=20", "sma=30")))
        worst = claims["worst_ring_gradient_agreement_pct_clean"]
        typical = claims["typical_ring_gradient_agreement_pct_clean"]
        assert worst > typical


class TestLicensingRestsOnTheWorstRing:
    """The verdict must not follow whichever reduction happens to be kinder.

    A blanket switch to the median was measured to flip criterion 2 on the
    n=2 fixture's reference case, which would have unlicensed Track 2 there.
    Pinning the criteria to the pre-registered worst-ring statistic is what
    keeps the reduction from being chosen after seeing the verdict.
    """

    def _results(self, **kwargs):
        return TestClaimExtraction()._results(**kwargs)

    def test_criterion_1_reads_the_worst_ring_claims(self):
        results = self._results(gradient_pct=0.05)
        verdict = evaluate_licensing(results)
        # 12.0 point-derivative offset against a 0.05 secant agreement: decisive.
        assert verdict["criterion_1_beats_point_derivative"]
        assert verdict["criterion_1_margin"] == pytest.approx(12.0 / 0.05)

    def test_a_verdict_survives_adding_the_typical_ring_claims(self):
        """The typical-ring family is archived, but nothing may hang on it."""
        results = self._results(rings=("sma=12", "sma=20", "sma=30"))
        claims = extract_claims(results)
        verdict = evaluate_licensing(results)
        assert verdict["regimes"]["reference"]["gradient_agreement_pct"] == pytest.approx(
            claims["worst_ring_gradient_agreement_pct_clean"]
        )


class TestValidityIsObservedNotInferred:
    """Validity must come from realized provenance, never from the request.

    A review found the first version of this reading `spec["isoclip"]` and
    `spec["interpolate_start"]` -- what was *asked for*. An archive whose
    realized behaviour disagreed with its request still reported valid, and the
    tests preserved the defect by only ever mutating the request. So these
    mutate the realized provenance and the recorded per-ring mode, and one test
    makes the two disagree on purpose.
    """

    def _case(
        self,
        basis="polar_from_image_x_axis",
        modes=("line_interpolated",) * 3,
        rad=400.0,
        status="measured",
        partner_interpolated=(True, True, True),
    ):
        rings = [
            {
                "sma": sma,
                "comparison_sma": sma * 1.1,
                "sampling_mode": mode,
                "status": status,
                "autoprof_b0_secant": -1.0,
            }
            for sma, mode in zip((12.0, 18.0, 25.0), modes)
        ]
        # per_ring_interpolated carries the observed mode of every requested
        # ring: the base rings, then their comparison partners, in order.
        observed = [m == "line_interpolated" for m in modes] + list(partner_interpolated)
        return {
            "spec": {"name": "reference", "snr": None, "n_realizations": 1},
            "autoprof_provenance": {
                "harmonic_basis": basis,
                "rad_interp_pix": rad,
                "sampling_mode": {"per_ring_interpolated": observed},
            },
            "realizations": [{"rings": rings}],
            "summary": {},
        }

    def test_a_clean_case_is_valid_on_every_row(self):
        result = structural_validity(self._case())
        assert result["all_rows_valid"] and result["valid_rows"] == 3

    def test_a_realized_eccentric_anomaly_basis_invalidates_every_row(self):
        """The request said nothing; the provenance did. Only the latter counts."""
        result = structural_validity(self._case(basis="eccentric_anomaly"))
        assert result["valid_rows"] == 0
        assert any("mixes orders" in r for r in result["rows"][0]["harmonic_conversion_reasons"])

    def test_a_realized_nearest_pixel_ring_invalidates_only_that_row(self):
        """Row-level, because a case-wide boolean is wrong for some of its rows.

        `interpolate_default` really does contain both modes at once.
        """
        result = structural_validity(self._case(modes=("line_interpolated", "line_nearest_pixel", "line_interpolated")))
        assert [row["harmonic_conversion_valid"] for row in result["rows"]] == [True, False, True]
        assert not result["all_rows_valid"] and result["any_row_valid"]

    def test_a_nearest_pixel_comparison_ring_invalidates_the_pair(self):
        """A secant needs both rings, so the partner's mode is checked too.

        The partner's mode is *observed*, from per_ring_interpolated, so this
        sets it independently of the base ring rather than inferring it from a
        radius threshold.
        """
        case = self._case(partner_interpolated=(False, True, True))
        row = structural_validity(case)["rows"][0]
        assert row["base_sampling_mode"] == "line_interpolated"
        assert row["comparison_sampling_mode"] == "line_nearest_pixel"
        assert row["comparison_sampling_mode_source"] == "observed"
        assert not row["harmonic_conversion_valid"]

    def test_an_unattributed_partner_mode_is_unknown_not_nearest_pixel(self):
        """`bool(None)` is False, so an unattributed ring was being archived as
        an *observed* nearest-pixel measurement. Both make the row invalid, but
        for opposite reasons, and only one of them is true."""
        case = self._case(partner_interpolated=(None, True, True))
        row = structural_validity(case)["rows"][0]
        assert row["comparison_sampling_mode"] is None
        assert row["comparison_sampling_mode_source"] == "observed_unattributed"
        assert not row["harmonic_conversion_valid"]
        assert any("unknown" in reason for reason in row["harmonic_conversion_reasons"])
        assert not any("nearest" in reason for reason in row["harmonic_conversion_reasons"])

    def test_an_unattributed_partner_never_falls_back_to_the_derivation(self):
        """'We could not tell' must not become 'we worked it out'."""
        case = self._case(partner_interpolated=(None, True, True), rad=13.0)
        assert structural_validity(case)["rows"][0]["comparison_sampling_mode_source"] == "observed_unattributed"

    def test_completion_counts_are_matched_by_radius_not_position(self):
        """Assigning summary entries positionally pairs a ring with another
        ring's count whenever the summary is ordered differently."""
        case = self._case()
        case["spec"]["n_realizations"] = 25
        # Deliberately reversed relative to the ring order.
        case["summary"] = {
            "sma=25": {"b0_secant_vs_point_derivative_pct": {"n": 19}},
            "sma=18": {"b0_secant_vs_point_derivative_pct": {"n": 25}},
            "sma=12": {"b0_secant_vs_point_derivative_pct": {"n": 25}},
        }
        rows = structural_validity(case)["rows"]
        assert [(row["sma"], row["realizations_measured"]) for row in rows] == [(12.0, 25), (18.0, 25), (25.0, 19)]

    def test_completion_does_not_depend_on_isoster(self):
        """A zero or invalid isoster gradient makes the cross-tool percentage
        undefined while AutoProf's own secant is perfectly good, so completion
        must not be counted from that column."""
        case = self._case()
        case["spec"]["n_realizations"] = 25
        case["summary"] = {
            "sma=12": {
                "b0_secant_vs_isoster_pct": {"n": 3},
                "b0_secant_vs_point_derivative_pct": {"n": 25},
            }
        }
        row = structural_validity(case)["rows"][0]
        assert row["realizations_measured"] == 25
        assert row["realizations_measured_source"] == "b0_secant_vs_point_derivative_pct"
        assert row["measurement_complete"]

    def test_an_archive_without_observed_partner_modes_falls_back_and_says_so(self):
        """Legacy archives may derive it, but must never claim it was observed."""
        case = self._case(rad=13.0)
        case["autoprof_provenance"].pop("sampling_mode")
        row = structural_validity(case)["rows"][0]
        assert row["comparison_sampling_mode_source"] == "derived_legacy_archive"
        assert row["comparison_sampling_mode"] == "line_nearest_pixel"

    def test_a_realization_that_failed_to_measure_breaks_completion_not_structure(self):
        """The two must be separable: an aperture can be fine and the
        measurement still incomplete in some realizations."""
        case = self._case()
        case["spec"]["n_realizations"] = 25
        case["summary"] = {
            f"sma={sma:g}": {"b0_secant_vs_point_derivative_pct": {"n": n}}
            for sma, n in ((12.0, 25), (18.0, 19), (25.0, 25))
        }
        rows = structural_validity(case)["rows"]
        assert all(row["structurally_applicable"] for row in rows)
        assert [row["measurement_complete"] for row in rows] == [True, False, True]
        assert not rows[1]["harmonic_conversion_valid"]
        assert rows[1]["realizations_measured"] == 19

    def test_a_failed_ring_is_invalid(self):
        result = structural_validity(self._case(status="autoprof_failed"))
        assert result["valid_rows"] == 0

    def test_the_request_cannot_override_the_provenance(self):
        """The exact defect: make them disagree, and provenance must win."""
        case = self._case(basis="eccentric_anomaly")
        case["spec"]["isoclip"] = True  # the request claims the good basis
        case["spec"]["interpolate_start"] = 100.0
        assert structural_validity(case)["valid_rows"] == 0


class TestTheVerdictSeparatesItsThreeConcepts:
    def _results(self, **kwargs):
        return TestClaimExtraction()._results(**kwargs)

    def test_no_licensed_field_survives(self):
        """'licensed' merged method validation, row validity and accuracy."""
        verdict = evaluate_licensing(self._results())
        assert not any("licensed_on" in key for key in verdict)
        assert verdict["withdrawn_criterion_2"]

    def test_method_validation_and_row_validity_are_distinct_fields(self):
        verdict = evaluate_licensing(self._results())
        assert "conversion_method_validated" in verdict
        assert "all_reference_rows_structurally_valid" in verdict

    def test_a_wildly_inaccurate_conversion_does_not_change_row_validity(self):
        """Accuracy must not masquerade as validity."""
        accurate = evaluate_licensing(self._results(bender_pct=0.4))
        awful = evaluate_licensing(self._results(bender_pct=90.0))
        assert accurate["all_reference_rows_structurally_valid"] == awful["all_reference_rows_structurally_valid"]
        assert awful["regimes"]["reference"]["bender_agreement_pct"] == pytest.approx(90.0)

    def test_no_criterion_2_verdict_survives_anywhere(self):
        assert "criterion_2" not in json.dumps(evaluate_licensing(self._results())["regimes"])

    def test_the_consistency_diagnostic_uses_the_exact_bound(self):
        """The linear form B <= R + G is only first order, and its error is
        what produced the tiny paired failures once read as evidence that the
        paired form was unusable."""
        from benchmarks.harmonic_scale.run_gradient_reconstruction import algebraic_consistency

        results = self._results(gradient_pct=5.0, raw_pct=1.0, bender_pct=6.2)
        assert 6.2 > 1.0 + 5.0  # the linear bound would flag this
        assert algebraic_consistency(results["cases"][0])["consistent"]  # the exact one does not


class TestFrozenTolerances:
    def _pilot(self, spread):
        """A pilot with two rings and several realizations, so a bootstrap has
        something to resample."""
        import numpy as np

        rng = np.random.default_rng(7)
        # Five rings, as both real fixtures have: with only two, a max and a
        # median over them are nearly the same statistic and the test could
        # not tell them apart.
        n_real, n_rings = 25, 5
        draws = rng.normal(1.0, spread, size=(n_real, n_rings))
        realizations = [
            {"rings": [{"b0_secant_vs_isoster_pct": float(draws[r, i])} for i in range(n_rings)]} for r in range(n_real)
        ]
        summary = {
            f"sma={10 * (i + 1)}": {
                "b0_secant_vs_isoster_pct": {
                    "n": n_real,
                    "median": float(np.median(draws[:, i])),
                    "stdev": float(np.std(draws[:, i], ddof=1)),
                }
            }
            for i in range(n_rings)
        }
        case = {
            "spec": {"name": "noise_snr100", "snr": 100.0, "n_realizations": n_real},
            "summary": summary,
            "realizations": realizations,
        }
        return {
            "mode": "pilot",
            "fixture": "sersic_n2_compact",
            "seed_block": 500_000,
            "environment": {"git": {"commit": "deadbeef", "dirty": False}, "fixture_fingerprint": "abc"},
            "cases": [case],
        }

    def test_a_noisier_pilot_earns_a_wider_tolerance(self):
        """The tolerance must track the claim's own scatter, not a constant."""
        quiet = freeze_tolerances(self._pilot(spread=0.01))
        loud = freeze_tolerances(self._pilot(spread=0.20))
        name = "worst_ring_gradient_agreement_pct_noise_snr100"
        assert loud["claims"][name]["tolerance"] > quiet["claims"][name]["tolerance"]

    def test_the_worst_ring_scatters_more_than_the_typical_ring(self):
        """The defect this replaced: one number used for both. A max varies
        through *which* ring wins as well as through that ring's noise, so its
        measured spread must come out larger."""
        frozen = freeze_tolerances(self._pilot(spread=0.20))["claims"]
        worst = frozen["worst_ring_gradient_agreement_pct_noise_snr100"]
        typical = frozen["typical_ring_gradient_agreement_pct_noise_snr100"]
        assert worst["basis"] == typical["basis"] == "bootstrap"
        assert worst["bootstrap_stdev"] > typical["bootstrap_stdev"]

    def test_a_deterministic_case_falls_back_to_the_floor(self):
        pilot = self._pilot(spread=0.05)
        pilot["cases"][0]["realizations"] = pilot["cases"][0]["realizations"][:1]
        frozen = freeze_tolerances(pilot)["claims"]
        assert frozen["worst_ring_gradient_agreement_pct_noise_snr100"]["basis"] == "deterministic_floor"

    def test_the_frozen_file_records_the_claim_definitions_it_was_built_from(self):
        frozen = freeze_tolerances(self._pilot(spread=0.05))
        assert frozen["policy"]["claims_fingerprint"] == claims_fingerprint()
        assert frozen["policy"]["claim_reductions"] == dict(CLAIM_REDUCTIONS)

    def test_freezing_is_reproducible(self):
        """A bootstrap seeded from the clock would make the frozen file drift."""
        import json

        first = freeze_tolerances(self._pilot(spread=0.10))
        second = freeze_tolerances(self._pilot(spread=0.10))
        # Compared as text: this toy pilot carries one case, so most claims are
        # NaN, and NaN never equals itself under dict equality.
        assert json.dumps(first["claims"], sort_keys=True) == json.dumps(second["claims"], sort_keys=True)


class TestClaimsFingerprint:
    def test_changing_a_reduction_changes_the_fingerprint(self, monkeypatch):
        import benchmarks.harmonic_scale.run_gradient_reconstruction as runner

        before = claims_fingerprint()
        monkeypatch.setattr(runner, "CLAIM_REDUCTIONS", {"worst_ring": "max over rings"})
        assert claims_fingerprint() != before

    def test_changing_a_claim_definition_changes_the_fingerprint(self, monkeypatch):
        import benchmarks.harmonic_scale.run_gradient_reconstruction as runner

        before = claims_fingerprint()
        monkeypatch.setattr(
            runner,
            "CLAIM_DEFINITIONS",
            runner.CLAIM_DEFINITIONS[:-1],
        )
        assert claims_fingerprint() != before


class TestTheGateRejectsStaleTolerances:
    """The guard added after a reduction change nearly slipped through.

    Editing how a claim is reduced, while leaving the frozen file in place,
    would have compared a validation value computed one way against a pilot
    value computed another -- and passed, because both are just numbers. The
    claim-definition fingerprint is what makes that impossible, so it needs a
    test that watches it fail.
    """

    def _fixtures(self):
        import json
        from pathlib import Path

        here = Path(__file__).resolve().parents[2] / "benchmarks" / "harmonic_scale"
        archive = here / "reference_gradient_reconstruction_sersic_n4_extended.json"
        tolerances = here / "frozen_tolerances_gradient_sersic_n4_extended.json"
        if not (archive.exists() and tolerances.exists()):
            pytest.skip("the n=4 Track 2 archive is not present")
        return json.loads(archive.read_text()), json.loads(tolerances.read_text())

    def _run(self, archive, tolerances):
        import contextlib
        import io

        import benchmarks.harmonic_scale.check_gradient_reconstruction as gate

        with contextlib.redirect_stdout(io.StringIO()):
            return gate.check_one("sersic_n4_extended", archive, tolerances)

    def test_the_archive_passes_unmodified(self):
        archive, tolerances = self._fixtures()
        assert self._run(archive, tolerances) == []

    def test_a_fingerprint_from_different_claim_definitions_is_rejected(self):
        import json

        archive, tolerances = self._fixtures()
        stale = json.loads(json.dumps(tolerances))
        stale["policy"]["claims_fingerprint"] = "0" * 64
        assert any("claim definitions have changed" in f for f in self._run(archive, stale))

    def test_tolerances_predating_the_fingerprint_are_rejected(self):
        import json

        archive, tolerances = self._fixtures()
        old = json.loads(json.dumps(tolerances))
        old["policy"].pop("claims_fingerprint")
        assert any("predate the claim-definition fingerprint" in f for f in self._run(archive, old))
