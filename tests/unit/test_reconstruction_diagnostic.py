"""Synthetic sign/phase/normalization gate before interpreting real harmonics."""

import numpy as np
import pytest
from astropy.table import Table

pytest.importorskip("pandas")

from benchmarks.benchmark_baseline.baseline_shared import build_photutils_model_image
from benchmarks.exhausted.analysis.reconstruction_diagnostic import raw_polar_carrier


@pytest.mark.parametrize("ellipticity", [0.0, 0.45])
def test_identical_raw_signal_from_bender_and_autoprof_renders_identically(ellipticity):
    pa = 0.37
    rows, native = [], []
    for sma in np.arange(1.0, 25.0, 0.5):
        row = dict(
            sma=sma,
            intens=100 - sma,
            eps=ellipticity,
            pa=pa,
            x0=30.0,
            y0=30.0,
            grad=-2.0,
            a3=0.4 / (2 * sma),
            b3=0.0,
            a4=0.0,
            b4=0.8 / (2 * sma),
        )
        rows.append(row)
        ap = dict(row, harmonic_basis="polar_from_image_x_axis", autoprof_b0=100 - sma)
        for order, sine, cosine in [(3, 0.4, 0.0), (4, 0.0, 0.8)]:
            phase = order * pa
            sky_sine = sine * np.cos(phase) + cosine * np.sin(phase)
            sky_cosine = -sine * np.sin(phase) + cosine * np.cos(phase)
            ap[f"autoprof_a{order}_native"] = -sky_sine / (2 * ap["autoprof_b0"])
            ap[f"autoprof_b{order}_native"] = sky_cosine / (2 * ap["autoprof_b0"])
        native.append(ap)
    left = raw_polar_carrier(Table(rows=rows), "isoster")
    right = raw_polar_carrier(Table(rows=native), "autoprof")
    a = build_photutils_model_image((61, 61), left, high_harmonics=True, fill=np.nan)
    b = build_photutils_model_image((61, 61), right, high_harmonics=True, fill=np.nan)
    np.testing.assert_allclose(a, b, equal_nan=True, atol=1e-12)
    plain = build_photutils_model_image((61, 61), rows, high_harmonics=False, fill=np.nan)
    y, x = np.mgrid[:61, :61]
    x_rot = (x - 30) * np.cos(pa) + (y - 30) * np.sin(pa)
    y_rot = -(x - 30) * np.sin(pa) + (y - 30) * np.cos(pa)
    radius = np.hypot(x_rot, y_rot / (1 - ellipticity))
    angle = np.arctan2(y - 30, x - 30) - pa
    expected = 0.4 * np.sin(3 * angle) + 0.8 * np.cos(4 * angle)
    support = (radius > 8) & (radius < 20) & np.isfinite(a) & np.isfinite(plain)
    assert np.sqrt(np.mean((a[support] - plain[support] - expected[support]) ** 2)) < 0.05
    bad = Table(rows=rows)
    bad["use_eccentric_anomaly"] = True
    with pytest.raises(ValueError, match="EA harmonics"):
        raw_polar_carrier(bad, "isoster")
    bad = Table(rows=native)
    bad["autoprof_b0"][0] = np.nan
    with pytest.raises(ValueError, match="Missing raw harmonic"):
        raw_polar_carrier(bad, "autoprof")


def test_demo_selection_uses_measured_gap_and_distinct_galaxies():
    import pandas as pd

    from benchmarks.exhausted.analysis.harmonic_demo import select_cases

    rows = []
    for galaxy, scenario, values in [
        ("a", "one", [0.05, 0.03, 0.01]),
        ("a", "two", [0.04, 0.03, 0.01]),
        ("b", "one", [0.03, 0.03, 0.01]),
        ("c", "one", [0.0005, 0.03, 0.00001]),
    ]:
        for tool, value in zip(("isoster", "photutils", "autoprof"), values):
            rows.append(
                dict(
                    galaxy=galaxy, scenario=scenario, primary=True, status="ok", tool=tool, truth_relative_rms_all=value
                )
            )
    chosen = select_cases(pd.DataFrame(rows), 2)
    assert list(zip(chosen.galaxy, chosen.scenario)) == [("a", "one"), ("b", "one")]
    with pytest.raises(ValueError, match="Insufficient"):
        select_cases(pd.DataFrame(rows), 3)


@pytest.mark.parametrize("use_ea", [False, True])
def test_native_isoster_harmonics_preserve_angle_basis(use_ea):
    from isoster.model import build_isoster_model

    pa, axis_ratio = 0.37, 0.55
    rows = [
        dict(
            sma=sma,
            intens=100 - sma,
            eps=1 - axis_ratio,
            pa=pa,
            x0=30.0,
            y0=30.0,
            a3=0.4 / sma,
            b3=0.0,
            a4=0.0,
            b4=0.8 / sma,
            use_eccentric_anomaly=use_ea,
        )
        for sma in np.arange(1.0, 25.0, 0.5)
    ]
    plain = build_isoster_model((61, 61), rows, fill=np.nan, use_harmonics=False)
    model = build_isoster_model((61, 61), rows, fill=np.nan, use_harmonics=True, harmonic_orders=[3, 4])
    y, x = np.mgrid[:61, :61]
    x_rot = (x - 30) * np.cos(pa) + (y - 30) * np.sin(pa)
    y_rot = -(x - 30) * np.sin(pa) + (y - 30) * np.cos(pa)
    radius = np.hypot(x_rot, y_rot / axis_ratio)
    angle = np.arctan2(y_rot / axis_ratio if use_ea else y_rot, x_rot)
    expected = 0.4 * np.sin(3 * angle) + 0.8 * np.cos(4 * angle)
    support = (radius > 8) & (radius < 20)
    assert np.sqrt(np.mean((model[support] - plain[support] - expected[support]) ** 2)) < 0.002
