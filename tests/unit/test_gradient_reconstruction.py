"""A4 Track 2: the matched-secant reconstruction, without AutoProf.

The arithmetic and the design are testable in CI; the measurement is not.
What matters most here is the *convention* — that the reconstruction forms
the same quantity isoster divides by, rather than the most accurate gradient
available. Getting that wrong would not fail loudly: it would quietly put
AutoProf's Bender coefficients ~12% away from isoster's and look like a
disagreement between tools.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.harmonic_scale.conventions import (
    DEFAULT_GRADIENT_STEP,
    comparison_radius,
    matched_secant_gradient,
)
from benchmarks.harmonic_scale.run_gradient_reconstruction import (
    GRADIENT_STEP,
    SEED_BLOCKS,
    build_grid,
    extract_claims,
    fixture_fingerprint,
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
    def _results(self, gradient_pct=0.05, bender_pct=0.4, raw_pct=0.4):
        def ring():
            entry = {
                key: {"n": 1, "median": value, "min": value, "max": value}
                for key, value in (
                    ("b0_secant_vs_isoster_pct", gradient_pct),
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
        return {
            "cases": [
                {"spec": {"name": n, "snr": None, "n_realizations": 1}, "summary": {"sma=12": ring()}} for n in names
            ]
        }

    def test_it_reports_the_gradient_agreement(self):
        claims = extract_claims(self._results(gradient_pct=0.05))
        assert claims["gradient_agreement_pct_clean"] == pytest.approx(0.05)

    def test_it_reports_the_wrong_estimator_penalty_separately(self):
        # The median column must be quantified, not dismissed.
        assert extract_claims(self._results())["median_estimator_penalty_pct"] == pytest.approx(0.9)

    def test_it_reports_the_convention_offset(self):
        # The finding that forced the design belongs in the archive.
        claims = extract_claims(self._results())
        assert claims["secant_vs_point_derivative_pct"] == pytest.approx(12.0)

    def test_both_acceptance_quantities_are_present(self):
        claims = extract_claims(self._results())
        assert "bender_agreement_pct_reference" in claims
        assert "raw_agreement_pct_reference" in claims

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
        assert claims["raw_agreement_pct_reference"] == pytest.approx(0.4)

    def test_normalizing_a_worse_bender_than_raw_is_visible(self):
        """Criterion 2 must be able to fail, not only to pass."""
        claims = extract_claims(self._results(bender_pct=9.0, raw_pct=0.4))
        assert claims["bender_agreement_pct_reference"] > claims["raw_agreement_pct_reference"]
