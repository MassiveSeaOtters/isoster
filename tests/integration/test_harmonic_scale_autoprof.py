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
import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.autoprof_env import resolve_autoprof_python
from benchmarks.harmonic_scale.adapters import (
    assert_rings_match_request,
    measure_autoprof_fixed,
    measure_isoster_fixed,
)
from benchmarks.utils.sersic_model import (
    create_sersic_image_with_harmonics,
    integrated_harmonic_truth,
)


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
    """The conversion is correct, and the residual is an interpolation artifact.

    Measured on the planted fixture against the integrated analytic truth,
    varying only ``ap_iso_interpolate_start`` -- the option that decides which
    rings AutoProf samples with Lanczos interpolation and which it samples by
    rounding to the nearest pixel (``SharedFunctions.py:673``):

        ap_iso_interpolate_start   sma=15   sma=25   sma=35   sma=45
        100 (Lanczos everywhere)    0.993    0.999    1.000    1.001
          5 (the default)           0.993    1.247    1.132    1.084
          0 (nearest everywhere)    1.357    1.247    1.132    1.084

    With interpolation enabled everywhere, AutoProf agrees with the analytic
    truth to 0.1%. The apparent 13-25% excess is therefore **not** a scale or
    convention difference -- the conversion in ``conventions.py`` is right --
    but nearest-neighbour sampling of the ring.

    And it is specifically an m=4 effect, because a square pixel grid is
    four-fold symmetric. With equal amplitudes planted at m=3 and m=4:

        Lanczos:           m=3  0.995 1.000 0.997 0.999
                           m=4  0.993 0.999 1.000 1.001
        nearest-neighbour: m=3  0.981 1.063 0.927 0.996   (scatter, no bias)
                           m=4  1.357 1.245 1.136 1.084   (systematic, falls with R)

    m=3 shows only scatter; m=4 is systematically inflated, decreasingly so as
    the ring grows and the half-pixel displacement matters less.
    """

    def test_conversion_recovers_isoster_when_interpolation_is_used_everywhere(self, fixture_image):
        """The decisive test: remove the sampling artifact and the tools agree."""
        image, _ = fixture_image
        with tempfile.TemporaryDirectory() as workspace:
            ap_rows, _ = measure_autoprof_fixed(
                image,
                _request(),
                orders=ORDERS,
                workspace=workspace,
                pixel_scale=1.0,
                isoclip=True,
                extra_options={"ap_iso_interpolate_start": 100},
            )
        iso_rows = measure_isoster_fixed(image, _request(), orders=ORDERS)
        for ap_row, iso_row in zip(ap_rows, iso_rows):
            ratio = ap_row["c4_raw"] / iso_row["c4_raw"]
            assert 0.98 < ratio < 1.02, (
                f"with Lanczos sampling everywhere the tools should agree to ~1%, "
                f"got {ratio:.4f} at sma={iso_row['sma']}"
            )

    def test_b0_matches_the_ring_mean_so_the_denominator_is_sound(self, fixture_image, autoprof_result):
        """Isolates the normalization from the sampling difference."""
        image, _ = fixture_image
        ap_rows, _ = autoprof_result
        iso_rows = measure_isoster_fixed(image, _request(), orders=ORDERS)
        for ap_row, iso_row in zip(ap_rows, iso_rows):
            ratio = ap_row["autoprof_b0"] / iso_row["mean_intensity"]
            assert 0.95 < ratio < 1.05, (
                f"b0 disagrees with the ring mean by {abs(1 - ratio):.1%}; the raw "
                "reconstruction divides by it, so this must be sound"
            )

    def test_nearest_neighbour_sampling_biases_m4_and_not_m3(self, fixture_image):
        """Pins the mechanism: pixel-grid aliasing into the four-fold mode."""
        image, meta = create_sersic_image_with_harmonics(
            n=2.0,
            R_e=25.0,
            I_e=100.0,
            eps=EPS,
            pa=PA,
            shape=SHAPE,
            center=CENTRE,
            harmonics={(3, "cos"): 0.03, (4, "cos"): 0.03},
        )
        request = [{"sma": 25.0, "x0": CENTRE[0], "y0": CENTRE[1], "eps": EPS, "pa": PA}]
        results = {}
        for start in (100, 0):
            with tempfile.TemporaryDirectory() as workspace:
                rows, _ = measure_autoprof_fixed(
                    image,
                    request,
                    orders=(3, 4),
                    workspace=workspace,
                    pixel_scale=1.0,
                    isoclip=True,
                    extra_options={"ap_iso_interpolate_start": start},
                )
            truth = integrated_harmonic_truth(meta, sma=25.0, orders=(3, 4))
            results[start] = {n: rows[0][f"c{n}_raw"] / truth[n]["c_raw"] for n in (3, 4)}

        assert abs(results[100][4] - 1.0) < 0.02, "Lanczos should recover m=4"
        assert results[0][4] > 1.10, "nearest-neighbour should inflate m=4"
        assert abs(results[0][3] - 1.0) < abs(results[0][4] - 1.0), (
            "the bias should hit m=4 harder than m=3; a square grid is four-fold "
            f"symmetric. Got m=3 {results[0][3]:.3f}, m=4 {results[0][4]:.3f}"
        )
