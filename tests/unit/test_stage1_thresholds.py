"""Part B Stage 1: the accuracy contract, and the truth it is built on.

These guard two classes of defect that review found in earlier drafts, both of
which produced authoritative-looking numbers that were wrong: a truth lookup
that silently yielded no value, and a bar derived from it that failed *open* to
infinity so every measurement passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.harmonic_scale.run_harmonic_scale import FIXTURES, ORDERS, PLANTED_HARMONICS
from benchmarks.timing import accuracy_thresholds as at
from benchmarks.timing.stage1_fixtures import stage1_fixtures
from benchmarks.utils.sersic_model import (
    analytic_truth_on_aperture,
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
        for bars in contract["systematic_amplitude_error_pct_by_component"].values():
            for components in bars.values():
                for value in components.values():
                    assert np.isfinite(value) and value > 0

    def test_the_contract_serializes_without_nan(self):
        import json

        json.dumps(at.stage_1_contract(), allow_nan=False, default=str)


class TestArbitraryRadius:
    def test_a_bar_exists_for_a_radius_not_in_the_table(self):
        value = at.amplitude_bar_at("sersic_n2_compact", 59.25, "s3_raw_major")
        assert np.isfinite(value) and value > 0

    def test_outside_the_target_interval_raises(self):
        with pytest.raises(ValueError, match="outside the target interval"):
            at.amplitude_bar_at("sersic_n2_compact", 200.0, "s3_raw_major")

    def test_the_table_is_a_sample_of_the_function(self):
        contract = at.stage_1_contract()
        bars = contract["systematic_amplitude_error_pct_by_component"]["sersic_n2_compact"]
        for sma, components in bars.items():
            for component, value in components.items():
                # The table stores this function's value rounded to 4 decimals.
                assert at.amplitude_bar_at("sersic_n2_compact", float(sma), component) == pytest.approx(value, abs=5e-5)


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

    def test_a_harmonics_off_arm_with_good_intensity_is_eligible(self):
        assert at.headline_eligible(self.BASE)

    def test_an_unevaluated_applicable_metric_is_not_a_pass(self):
        assert not at.headline_eligible({**self.BASE, "intensity_accuracy_status": None})

    def test_a_failed_intensity_disqualifies_a_harmonics_off_arm(self):
        assert not at.headline_eligible({**self.BASE, "intensity_accuracy_status": "fail"})

    def test_contamination_alone_disqualifies(self):
        assert not at.headline_eligible({**self.BASE, "contamination_status": "contaminated"})


class TestFamilyUnit:
    def test_the_family_is_one_arm_not_the_whole_grid(self):
        contract = at.stage_1_contract()
        rings = len(stage1_fixtures()["sersic_n2_compact"]["radii"])
        assert contract["ensemble_harmonic_tests_per_arm"] == rings * len(at._COMPONENTS) == 20
        assert contract["ensemble_correction"] == "holm_bonferroni"
