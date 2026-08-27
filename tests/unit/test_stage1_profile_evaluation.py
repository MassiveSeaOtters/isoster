"""Fixed-radius evaluation for the natural end-to-end benchmark scope."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.harmonic_scale.conventions import rotate_raw_to_major_axis
from benchmarks.timing.profile_evaluation import (
    canonicalize_pa_and_harmonics,
    interpolate_profile_to_evaluation_radii,
)


def _major_components(s_sky, c_sky, order, pa):
    return rotate_raw_to_major_axis(s_sky, c_sky, order=order, pa_rad=pa)


def test_pa_canonicalization_carries_the_odd_order_sign():
    pa, components = canonicalize_pa_and_harmonics(
        np.pi + 0.2,
        {"s3_raw_major": 2.0, "c3_raw_major": -3.0, "s4_raw_major": 5.0, "c4_raw_major": 7.0},
    )
    assert pa == pytest.approx(0.2)
    assert components == pytest.approx(
        {"s3_raw_major": -2.0, "c3_raw_major": 3.0, "s4_raw_major": 5.0, "c4_raw_major": 7.0}
    )


def test_interpolation_is_log_radius_and_invariant_across_a_pa_wrap():
    radii = (10.0, 40.0)
    pas = (np.pi - 0.04, 0.04)
    rows = []
    for sma, pa, intensity in zip(radii, pas, (100.0, 25.0)):
        row = {
            "sma": sma,
            "x0": 50.0,
            "y0": 51.0,
            "eps": 0.3,
            "pa": pa,
            "ring_mean": intensity,
            "harmonic_conversion_valid": True,
        }
        for order, sky in ((3, (2.0, -1.0)), (4, (3.0, 5.0))):
            s_major, c_major = _major_components(*sky, order, pa)
            row[f"s{order}_raw_major"] = s_major
            row[f"c{order}_raw_major"] = c_major
        rows.append(row)

    result = interpolate_profile_to_evaluation_radii(rows, [20.0])
    assert len(result) == 1
    middle = result[0]
    assert middle["sma"] == 20.0
    assert middle["ring_mean"] == pytest.approx(62.5)
    assert middle["pa"] == pytest.approx(0.0, abs=1e-12)
    # Sky-frame values were constant, so after conversion to the canonical
    # major axis they remain exactly the same through the PA wrap.
    assert (middle["s3_raw_major"], middle["c3_raw_major"]) == pytest.approx((2.0, -1.0))
    assert (middle["s4_raw_major"], middle["c4_raw_major"]) == pytest.approx((3.0, 5.0))


def test_interpolation_never_extrapolates():
    rows = [
        {
            "sma": 10.0,
            "x0": 0.0,
            "y0": 0.0,
            "eps": 0.3,
            "pa": 0.0,
            "ring_mean": 1.0,
            "harmonic_conversion_valid": True,
            "s3_raw_major": 1.0,
            "c3_raw_major": 1.0,
            "s4_raw_major": 1.0,
            "c4_raw_major": 1.0,
        },
        {
            "sma": 20.0,
            "x0": 0.0,
            "y0": 0.0,
            "eps": 0.3,
            "pa": 0.0,
            "ring_mean": 1.0,
            "harmonic_conversion_valid": True,
            "s3_raw_major": 1.0,
            "c3_raw_major": 1.0,
            "s4_raw_major": 1.0,
            "c4_raw_major": 1.0,
        },
    ]
    with pytest.raises(ValueError, match="not bracketed"):
        interpolate_profile_to_evaluation_radii(rows, [5.0])


def test_harmonics_off_does_not_require_harmonic_columns():
    rows = [
        {"sma": 10.0, "x0": 0.0, "y0": 0.0, "eps": 0.3, "pa": 0.0, "ring_mean": 2.0},
        {"sma": 20.0, "x0": 0.0, "y0": 0.0, "eps": 0.3, "pa": 0.0, "ring_mean": 1.0},
    ]
    result = interpolate_profile_to_evaluation_radii(rows, [15.0], orders=())
    assert set(result[0]) == {"sma", "x0", "y0", "eps", "pa", "ring_mean"}


@pytest.mark.parametrize("validity", [False, None])
def test_harmonic_interpolation_requires_observed_valid_conversion(validity):
    base = {
        "x0": 0.0,
        "y0": 0.0,
        "eps": 0.3,
        "pa": 0.0,
        "ring_mean": 1.0,
        "s3_raw_major": 1.0,
        "c3_raw_major": 1.0,
        "s4_raw_major": 1.0,
        "c4_raw_major": 1.0,
        "harmonic_conversion_valid": validity,
    }
    with pytest.raises(ValueError, match="observed-valid"):
        interpolate_profile_to_evaluation_radii([{**base, "sma": 10.0}, {**base, "sma": 20.0}], [15.0])


def test_duplicate_or_nonfinite_radii_fail_closed():
    base = {
        "x0": 0.0,
        "y0": 0.0,
        "eps": 0.3,
        "pa": 0.0,
        "ring_mean": 1.0,
        "harmonic_conversion_valid": True,
        "s3_raw_major": 1.0,
        "c3_raw_major": 1.0,
        "s4_raw_major": 1.0,
        "c4_raw_major": 1.0,
    }
    with pytest.raises(ValueError, match="unique"):
        interpolate_profile_to_evaluation_radii([{**base, "sma": 10.0}, {**base, "sma": 10.0}], [10.0])
    with pytest.raises(ValueError, match="finite"):
        interpolate_profile_to_evaluation_radii([{**base, "sma": 10.0}, {**base, "sma": np.nan}], [10.0])
