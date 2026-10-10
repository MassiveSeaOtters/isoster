"""Cubic image sampling, its inner bilinear fallback, and the log-linear renderer."""

import numpy as np

from isoster import IsosterConfig, fit_image
from isoster.model import build_isoster_model
from isoster.sampling import extract_isophote_data, prepare_sampling_image


def _exponential_image(size=101, scale=12.0, eps=0.3):
    y, x = np.mgrid[:size, :size].astype(float)
    c = size // 2
    r = np.sqrt((x - c) ** 2 + ((y - c) / (1 - eps)) ** 2)
    return 100.0 * np.exp(-r / scale)


def test_cubic_sampling_reproduces_pixel_values_and_reduces_bilinear_bias():
    image = _exponential_image(scale=3.0)
    coefficients, order = prepare_sampling_image(image, "cubic")
    assert order == 3 and coefficients.shape == image.shape
    # At pixel centres the interpolating spline returns the pixel values exactly.
    ring = dict(x0=50.0, y0=50.0, sma=4.0, eps=0.3, pa=0.0)
    x = np.array([60.0, 50.0, 40.0])
    y = np.array([50.0, 60.0, 50.0])
    from scipy.ndimage import map_coordinates

    np.testing.assert_allclose(
        map_coordinates(coefficients, [y, x], order=3, prefilter=False), image[y.astype(int), x.astype(int)], rtol=1e-10
    )
    # Between pixel centres the bilinear read overestimates a convex profile; cubic is far closer.
    exact = 100.0 * np.exp(-4.0 / 3.0)
    bilinear = extract_isophote_data(image, None, **ring).intens.mean()
    cubic = extract_isophote_data(coefficients, None, **ring, interpolation_order=3).intens.mean()
    assert bilinear > exact and abs(cubic - exact) < 0.25 * abs(bilinear - exact)


def test_bilinear_default_is_unchanged_and_fallback_uses_bilinear_inside_min_sma():
    image = _exponential_image()
    base = IsosterConfig(x0=50.0, y0=50.0, sma0=8.0, maxsma=20.0, minsma=0.0)
    assert base.sampling_interpolation == "bilinear"
    reference = fit_image(image, None, base)["isophotes"]
    cubic_inner = fit_image(
        image, None, base.model_copy(update={"sampling_interpolation": "cubic", "sampling_cubic_min_sma": 100.0})
    )["isophotes"]
    for a, b in zip(reference, cubic_inner):
        # With the fallback covering every ring, cubic mode must reproduce the bilinear fit.
        np.testing.assert_allclose(a["intens"], b["intens"], rtol=1e-12)
    cubic = fit_image(image, None, base.model_copy(update={"sampling_interpolation": "cubic"}))["isophotes"]
    changed = [abs(a["intens"] - b["intens"]) > 1e-9 for a, b in zip(reference, cubic) if a["sma"] > 0]
    assert any(changed)


def test_outer_fallback_keeps_cubic_inside_max_sma_and_reads_bilinearly_beyond():
    image = _exponential_image()
    cubic_config = IsosterConfig(x0=50.0, y0=50.0, sma0=8.0, maxsma=20.0, minsma=0.0, sampling_interpolation="cubic")
    cubic = fit_image(image, None, cubic_config)["isophotes"]
    switched = fit_image(image, None, cubic_config.model_copy(update={"sampling_cubic_max_sma": 12.0}))["isophotes"]
    assert len(cubic) == len(switched)
    inside = [(a, b) for a, b in zip(cubic, switched) if 0 < a["sma"] <= 12.0]
    beyond = [(a, b) for a, b in zip(cubic, switched) if a["sma"] > 12.0]
    assert inside and beyond
    for a, b in inside:
        # The fit grows outward from sma0 = 8, so rings up to the switch radius are untouched.
        np.testing.assert_allclose(a["intens"], b["intens"], rtol=1e-12)
    assert any(abs(a["intens"] - b["intens"]) > 1e-9 for a, b in beyond)
    bilinear = fit_image(image, None, cubic_config.model_copy(update={"sampling_interpolation": "bilinear"}))[
        "isophotes"
    ]
    everywhere = fit_image(image, None, cubic_config.model_copy(update={"sampling_cubic_max_sma": 0.5}))["isophotes"]
    for a, b in zip(bilinear, everywhere):
        # A switch radius below every ring must reproduce the bilinear fit exactly.
        np.testing.assert_allclose(a["intens"], b["intens"], rtol=1e-12)


def test_log_linear_renderer_is_exact_for_an_exponential_profile_and_keeps_central_pixel():
    scale = 12.0
    sma = np.array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    isophotes = [
        dict(sma=s, x0=40.0, y0=40.0, eps=0.0, pa=0.0, intens=100.0 * np.exp(-s / scale), stop_code=0) for s in sma
    ]
    isophotes[0]["intens"] = 123.0
    y, x = np.mgrid[:81, :81].astype(float)
    truth = 100.0 * np.exp(-np.hypot(x - 40, y - 40) / scale)
    # Inside the first fitted isophote the model is constant by design; test only the interpolated range.
    inside = (np.hypot(x - 40, y - 40) > 2) & (np.hypot(x - 40, y - 40) <= 32)
    linear = build_isoster_model((81, 81), isophotes, fill=np.nan, interp_kind="linear", use_harmonics=False)
    log_linear = build_isoster_model((81, 81), isophotes, fill=np.nan, interp_kind="log_linear", use_harmonics=False)
    assert np.nanmax(np.abs(log_linear[inside] - truth[inside])[truth[inside] > 1]) < 1e-9
    assert np.nansum(linear[inside] - truth[inside]) > 0  # linear-in-intensity overestimates a convex profile
    assert log_linear[40, 40] == 123.0 and linear[40, 40] == 123.0
