"""A2 integration: the real AutoProf, on the same rings as the other two.

Skips cleanly when the AutoProf venv is absent -- the treatment the campaign
fitter already gives a missing venv, and the reason the pure-arithmetic
convention tests in ``tests/unit/test_harmonic_conventions.py`` carry the
always-running coverage.

Those unit tests reimplement AutoProf's four-line FFT expression and so cannot
notice an upstream change. This file is where the *installed* AutoProf is
pinned: that it still emits ``a_n``, ``b_n`` and ``b0``, that the forced
pipeline honours the requested geometry, and that no ring silently fell into
isophotal-band sampling.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmarks.autoprof_env import resolve_autoprof_python
from benchmarks.harmonic_scale.adapters import (
    assert_rings_match_request,
    measure_autoprof_fixed,
    measure_isoster_fixed,
)
from benchmarks.utils.sersic_model import create_sersic_image_with_harmonics


def _autoprof_available() -> bool:
    interpreter = Path(resolve_autoprof_python())
    if not interpreter.is_file():
        return False
    try:
        probe = subprocess.run([str(interpreter), "-c", "import autoprof"], capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001
        return False
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _autoprof_available(),
    reason="AutoProf venv absent; see benchmarks/exhausted/README.md for the install recipe",
)

SHAPE = (241, 241)
CENTRE = (120.0, 120.0)
EPS = 0.3
PA = 0.0
ORDERS = (3, 4)
PLANTED = {(4, "cos"): 0.03}
SMA_VALUES = (25.0, 35.0)


@pytest.fixture(scope="module")
def fixture_image():
    return create_sersic_image_with_harmonics(
        n=2.0,
        R_e=25.0,
        I_e=100.0,
        eps=EPS,
        pa=PA,
        shape=SHAPE,
        center=CENTRE,
        harmonics=PLANTED,
    )


def _request():
    return [{"sma": sma, "x0": CENTRE[0], "y0": CENTRE[1], "eps": EPS, "pa": PA} for sma in SMA_VALUES]


@pytest.fixture(scope="module")
def autoprof_result(fixture_image, tmp_path_factory):
    image, _ = fixture_image
    workspace = tmp_path_factory.mktemp("autoprof_harmonic_scale")
    return measure_autoprof_fixed(
        image, _request(), orders=ORDERS, workspace=str(workspace), pixel_scale=1.0, isoclip=True
    )


class TestTheForcedPipelineDidWhatWasAsked:
    def test_returns_the_requested_rings(self, autoprof_result):
        rows, _ = autoprof_result
        assert [row["sma"] for row in rows] == pytest.approx(list(SMA_VALUES), abs=1e-6)

    def test_geometry_survived_the_round_trip(self, autoprof_result):
        rows, _ = autoprof_result
        # sma and eps come back from AutoProf; pa and the centre are imposed.
        assert_rings_match_request(rows, _request(), tolerances={"sma": 1e-6, "eps": 1e-6})

    def test_no_ring_fell_into_isophotal_band_sampling(self, autoprof_result):
        """The estimand must be a ring, not a radial band, on every row."""
        _, provenance = autoprof_result
        mode = provenance["sampling_mode"]
        assert mode["all_rings_line_sampled"], (
            f"{mode['band_sampling_calls']} band-sampling calls occurred; the ap_isoband_fixed guard did not hold"
        )
        assert mode["iso_between_total_calls"] > 0, "the sampling probe never fired"

    def test_the_realized_pipeline_is_the_forced_one(self, autoprof_result):
        _, provenance = autoprof_result
        steps = provenance["realized_pipeline_steps"]
        assert steps[2:5] == ["center forced", "isophoteinit forced", "isophoteextract forced"]
        assert "isophotefit forced" not in steps, (
            "isophotefit forced is not part of AutoProf's standard forced pipeline"
        )


class TestTheInstalledAutoProfStillMeansWhatWeThink:
    """The unit tests reimplement the FFT expression; this pins the real thing."""

    def test_emits_the_native_columns_the_conversion_depends_on(self, autoprof_result):
        rows, _ = autoprof_result
        for row in rows:
            assert np.isfinite(row["autoprof_b0"]), "b0 missing: the raw reconstruction needs it"
            for order in ORDERS:
                assert np.isfinite(row[f"autoprof_a{order}_native"])
                assert np.isfinite(row[f"autoprof_b{order}_native"])

    def test_b0_is_positive_and_of_order_the_ring_intensity(self, autoprof_result):
        rows, _ = autoprof_result
        for row in rows:
            assert row["autoprof_b0"] > 0.0

    def test_provenance_pins_the_version_and_the_source(self, autoprof_result):
        _, provenance = autoprof_result
        assert provenance["autoprof_version"]
        assert len(provenance["isophote_extract_sha256"]) == 64
        assert provenance["numpy_version"].startswith("1."), "AutoProf requires numpy<2"


class TestAgreementWithIsoster:
    """What the conversion fixes, and what it leaves for the campaign to measure.

    Measured on this fixture against the integrated analytic truth:

        sma    truth    isoster   photutils   autoprof   autoprof/truth
         25   5.5074     5.4937      5.4859     6.8680        1.247
         35   3.3252     3.3208      3.3180     3.7630        1.132

    isoster and photutils land within 0.3% of truth. AutoProf sits high by a
    *radius-dependent* margin, and ``b0`` matches the ring's mean intensity to
    better than 1%, so the excess is not in the normalization this module
    converts -- it is a difference in what AutoProf samples along the ring.

    These tests therefore pin the conversion (sign, factor of 2, use of
    ``|b0|``) and deliberately do **not** assert close agreement. Asserting a
    tolerance here would bury the residual that the A3 grid exists to measure.
    """

    def test_the_conversion_lands_within_an_order_of_magnitude(self, fixture_image, autoprof_result):
        """Catches a dropped factor of 2, a missing |b0|, or an inverted scale.

        Deliberately loose. A tight bound would be asserting a result we have
        not yet measured; a factor-of-two error cannot hide inside this.
        """
        image, _ = fixture_image
        ap_rows, _ = autoprof_result
        iso_rows = measure_isoster_fixed(image, _request(), orders=ORDERS)

        for ap_row, iso_row in zip(ap_rows, iso_rows):
            ratio = ap_row["c4_raw"] / iso_row["c4_raw"]
            assert 0.6 < ratio < 1.6, (
                f"converted C_4 is off by a factor of {ratio:.2f} at sma={iso_row['sma']}; "
                "that is too large to be a sampling difference and points at the conversion"
            )

    def test_b0_matches_the_ring_mean_so_the_denominator_is_sound(self, fixture_image, autoprof_result):
        """Isolates the normalization from whatever else differs."""
        image, _ = fixture_image
        ap_rows, _ = autoprof_result
        iso_rows = measure_isoster_fixed(image, _request(), orders=ORDERS)
        for ap_row, iso_row in zip(ap_rows, iso_rows):
            ratio = ap_row["autoprof_b0"] / iso_row["mean_intensity"]
            assert 0.95 < ratio < 1.05, (
                f"b0 disagrees with the ring mean by {abs(1 - ratio):.1%}; the raw "
                "reconstruction divides by it, so this must be sound"
            )

    def test_the_residual_excess_shrinks_with_radius(self, fixture_image, autoprof_result):
        """Characterizes the leftover rather than tolerating it silently.

        A constant offset would point at a convention we have not accounted
        for. A radius-dependent one points at sampling or interpolation, which
        is a property of the tools and is what the campaign should quantify.
        """
        image, _ = fixture_image
        ap_rows, _ = autoprof_result
        iso_rows = measure_isoster_fixed(image, _request(), orders=ORDERS)
        ratios = [ap["c4_raw"] / iso["c4_raw"] for ap, iso in zip(ap_rows, iso_rows)]
        assert ratios[0] > ratios[-1], (
            f"expected the excess to fall with radius, got {ratios}; a flat ratio "
            "would suggest an unaccounted convention factor instead"
        )

    def test_the_sign_convention_agrees_after_conversion(self, fixture_image, autoprof_result):
        """Native a_n is negated; the conversion must undo that, not double it."""
        image, _ = fixture_image
        ap_rows, _ = autoprof_result
        iso_rows = measure_isoster_fixed(image, _request(), orders=ORDERS)
        for ap_row, iso_row in zip(ap_rows, iso_rows):
            if abs(iso_row["c4_raw"]) > 1e-6:
                assert np.sign(ap_row["c4_raw"]) == np.sign(iso_row["c4_raw"]), (
                    "converted C_4 has the wrong sign relative to isoster"
                )
