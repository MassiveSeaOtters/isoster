"""A2: the three fixed-aperture adapters put every tool on the same rings.

Raw harmonic amplitudes depend on the exact ellipse they were measured on, so a
convention comparison run at free geometry measures the geometry instead. This
was not hypothetical: on the same image scaled by ten, AutoProf's fitted
geometry drifted by up to 0.047 in ellipticity and 8.8 degrees in position
angle, with none of 58 rings landing on identical geometry.

Each adapter therefore imposes the requested rings and then *verifies* it did,
ring by ring. These tests cover the two adapters that need no external
environment; the AutoProf adapter is exercised by the integration test, which
skips when its venv is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.harmonic_scale.adapters import (
    GeometryMismatch,
    assert_rings_match_request,
    measure_isoster_fixed,
    measure_photutils_fixed,
)
from benchmarks.utils.sersic_model import (
    create_sersic_image_with_harmonics,
    integrated_harmonic_truth,
)

SHAPE = (241, 241)
CENTRE = (120.0, 120.0)
EPS = 0.3
PA = 0.0
ORDERS = (3, 4)
SMA_VALUES = (20.0, 30.0, 40.0)
PLANTED = {(4, "cos"): 0.02, (3, "sin"): 0.015}


@pytest.fixture(scope="module")
def fixture_image():
    image, meta = create_sersic_image_with_harmonics(
        n=2.0,
        R_e=25.0,
        I_e=100.0,
        eps=EPS,
        pa=PA,
        shape=SHAPE,
        center=CENTRE,
        harmonics=PLANTED,
    )
    return image, meta


def _request():
    return [{"sma": sma, "x0": CENTRE[0], "y0": CENTRE[1], "eps": EPS, "pa": PA} for sma in SMA_VALUES]


ADAPTERS = {"isoster": measure_isoster_fixed, "photutils": measure_photutils_fixed}


class TestAdaptersHonourTheRequestedRings:
    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_returns_one_row_per_requested_sma_in_order(self, name, fixture_image):
        image, _ = fixture_image
        rows = ADAPTERS[name](image, _request(), orders=ORDERS)
        assert [row["sma"] for row in rows] == list(SMA_VALUES)

    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_reported_geometry_equals_the_request(self, name, fixture_image):
        image, _ = fixture_image
        rows = ADAPTERS[name](image, _request(), orders=ORDERS)
        assert_rings_match_request(rows, _request())

    def test_the_geometry_check_actually_rejects_a_mismatch(self):
        """Guards the check above from passing vacuously."""
        rows = [{"sma": 20.0, "x0": 120.0, "y0": 120.0, "eps": 0.3, "pa": 0.0}]
        drifted = [{"sma": 20.0, "x0": 120.0, "y0": 120.0, "eps": 0.35, "pa": 0.0}]
        with pytest.raises(GeometryMismatch, match="eps"):
            assert_rings_match_request(rows, drifted)


class TestAdaptersRecoverThePlantedSignal:
    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_raw_amplitudes_match_the_integrated_truth(self, name, fixture_image):
        image, meta = fixture_image
        rows = ADAPTERS[name](image, _request(), orders=ORDERS)
        for row in rows:
            truth = integrated_harmonic_truth(meta, sma=row["sma"], orders=ORDERS)
            for order in ORDERS:
                expected = truth[order]
                # Pixelization limits how well a discrete image can reproduce
                # the analytic integral; 3% of the amplitude is comfortably
                # inside that and far below any convention-scale difference.
                scale = max(abs(expected["c_raw"]), abs(expected["s_raw"]), 1e-12)
                assert abs(row[f"c{order}_raw"] - expected["c_raw"]) < 0.03 * scale, (
                    f"{name}: C_{order} off truth at sma={row['sma']}"
                )
                assert abs(row[f"s{order}_raw"] - expected["s_raw"]) < 0.03 * scale, (
                    f"{name}: S_{order} off truth at sma={row['sma']}"
                )

    def test_isoster_and_photutils_agree_ring_by_ring(self, fixture_image):
        """They share a convention exactly, so any gap is a measurement difference."""
        image, _ = fixture_image
        iso_rows = measure_isoster_fixed(image, _request(), orders=ORDERS)
        pht_rows = measure_photutils_fixed(image, _request(), orders=ORDERS)
        for iso, pht in zip(iso_rows, pht_rows):
            scale = max(abs(iso["c4_raw"]), 1e-12)
            assert abs(iso["c4_raw"] - pht["c4_raw"]) < 0.05 * scale, (
                f"isoster and photutils disagree at sma={iso['sma']}"
            )

    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_bender_values_are_the_raw_ones_divided_by_the_shared_factor(self, name, fixture_image):
        image, _ = fixture_image
        rows = ADAPTERS[name](image, _request(), orders=ORDERS)
        for row in rows:
            factor = row["sma"] * abs(row["gradient"])
            assert row["b4_bender"] == pytest.approx(row["c4_raw"] / factor, rel=1e-9)
            assert row["a4_bender"] == pytest.approx(row["s4_raw"] / factor, rel=1e-9, abs=1e-15)


class TestAdapterProvenance:
    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_every_row_reports_a_measurement_status(self, name, fixture_image):
        image, _ = fixture_image
        rows = ADAPTERS[name](image, _request(), orders=ORDERS)
        for row in rows:
            assert row["status"] == "measured"
            assert row["tool"] == name

    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_a_ring_outside_the_frame_is_reported_not_silently_dropped(self, name, fixture_image):
        image, _ = fixture_image
        request = [{"sma": 5000.0, "x0": CENTRE[0], "y0": CENTRE[1], "eps": EPS, "pa": PA}]
        rows = ADAPTERS[name](image, request, orders=ORDERS)
        assert len(rows) == 1, "an unmeasurable ring must still produce a row"
        assert rows[0]["status"] != "measured"
        assert np.isnan(rows[0]["c4_raw"])
