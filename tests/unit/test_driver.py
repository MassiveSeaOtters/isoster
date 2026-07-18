"""
Tests for isoster.driver module.
"""

import warnings

import numpy as np
import pytest

from isoster.config import IsosterConfig
from isoster.driver import fit_central_pixel, fit_image


def test_central_pixel_rounding_basic():
    """Test that central pixel uses proper rounding, not truncation."""
    # Create a simple test image with unique values
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 1: x0=5.6, y0=5.6 should round to [6, 6] not truncate to [5, 5]
    # image[6, 6] = 66, image[5, 5] = 55
    result = fit_central_pixel(image, None, x0=5.6, y0=5.6)
    expected_val = image[6, 6]  # Should be 66
    assert result["intens"] == expected_val, (
        f"For (5.6, 5.6): Expected image[6,6]={expected_val}, got {result['intens']}"
    )
    assert result["stop_code"] == 0
    assert result["valid"] == True


def test_central_pixel_rounding_down():
    """Test rounding down case."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 2: x0=5.4, y0=5.4 should round to [5, 5]
    result = fit_central_pixel(image, None, x0=5.4, y0=5.4)
    expected_val = image[5, 5]  # Should be 55
    assert result["intens"] == expected_val, (
        f"For (5.4, 5.4): Expected image[5,5]={expected_val}, got {result['intens']}"
    )


def test_central_pixel_rounding_half():
    """Test rounding at exactly .5 (should round to even, but np.round rounds half up)."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 3: x0=5.5, y0=5.5 should round to [6, 6]
    # np.round uses banker's rounding (round half to even) by default
    result = fit_central_pixel(image, None, x0=5.5, y0=5.5)
    expected_y = int(np.round(5.5))  # 6
    expected_x = int(np.round(5.5))  # 6
    expected_val = image[expected_y, expected_x]
    assert result["intens"] == expected_val, (
        f"For (5.5, 5.5): Expected image[{expected_y},{expected_x}]={expected_val}, got {result['intens']}"
    )


def test_central_pixel_rounding_high():
    """Test rounding with values close to next integer."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 4: x0=5.9, y0=5.9 should definitely round to [6, 6] not truncate to [5, 5]
    result = fit_central_pixel(image, None, x0=5.9, y0=5.9)
    expected_val = image[6, 6]  # Should be 66
    assert result["intens"] == expected_val, (
        f"For (5.9, 5.9): Expected image[6,6]={expected_val}, got {result['intens']}"
    )


def test_central_pixel_rounding_low():
    """Test rounding with values close to current integer."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # Test case 5: x0=5.1, y0=5.1 should round to [5, 5]
    result = fit_central_pixel(image, None, x0=5.1, y0=5.1)
    expected_val = image[5, 5]  # Should be 55
    assert result["intens"] == expected_val, (
        f"For (5.1, 5.1): Expected image[5,5]={expected_val}, got {result['intens']}"
    )


def test_central_pixel_rounding_with_mask():
    """Test that mask also uses proper rounding."""
    image = np.ones((10, 10)) * 100.0
    mask = np.zeros((10, 10), dtype=bool)

    # Mask pixel [6, 6] but not [5, 5]
    mask[6, 6] = True

    # x0=5.6, y0=5.6 should round to [6, 6] which is masked
    result = fit_central_pixel(image, mask, x0=5.6, y0=5.6)
    assert result["valid"] == False, "Coordinate (5.6, 5.6) should round to masked pixel [6, 6]"
    assert result["stop_code"] == -1
    assert np.isnan(result["intens"])

    # x0=5.4, y0=5.4 should round to [5, 5] which is not masked
    result = fit_central_pixel(image, mask, x0=5.4, y0=5.4)
    assert result["valid"] == True, "Coordinate (5.4, 5.4) should round to unmasked pixel [5, 5]"
    assert result["stop_code"] == 0
    assert result["intens"] == 100.0


def test_central_pixel_asymmetric_rounding():
    """Test that x and y round independently."""
    image = np.arange(100).reshape(10, 10).astype(float)

    # x0=5.3 (rounds to 5), y0=5.7 (rounds to 6)
    result = fit_central_pixel(image, None, x0=5.3, y0=5.7)
    expected_val = image[6, 5]  # y=6, x=5 => 65
    assert result["intens"] == expected_val, (
        f"For (5.3, 5.7): Expected image[6,5]={expected_val}, got {result['intens']}"
    )


@pytest.mark.parametrize(
    "x0,y0,expected_y,expected_x",
    [
        (5.1, 5.1, 5, 5),
        (5.4, 5.4, 5, 5),
        (5.5, 5.5, 6, 6),  # np.round behavior
        (5.6, 5.6, 6, 6),
        (5.9, 5.9, 6, 6),
        (4.1, 6.9, 7, 4),  # Asymmetric
        (7.2, 3.8, 4, 7),  # Asymmetric
    ],
)
def test_central_pixel_rounding_parametrized(x0, y0, expected_y, expected_x):
    """Parametrized test for various rounding scenarios."""
    image = np.arange(100).reshape(10, 10).astype(float)
    result = fit_central_pixel(image, None, x0=x0, y0=y0)
    expected_val = image[expected_y, expected_x]
    assert result["intens"] == expected_val, (
        f"For ({x0}, {y0}): Expected image[{expected_y},{expected_x}]={expected_val}, got {result['intens']}"
    )


class TestCentralPixelBoundsCheck:
    """Regression tests for out-of-bounds center coordinates in fit_central_pixel."""

    def test_out_of_bounds_x0(self):
        """x0 far outside image should return invalid, not IndexError."""
        image = np.ones((10, 10)) * 42.0
        result = fit_central_pixel(image, None, x0=100.0, y0=5.0)
        assert result["valid"] is False
        assert np.isnan(result["intens"])
        assert result["stop_code"] == -1

    def test_out_of_bounds_y0(self):
        """y0 far outside image should return invalid, not IndexError."""
        image = np.ones((10, 10)) * 42.0
        result = fit_central_pixel(image, None, x0=5.0, y0=100.0)
        assert result["valid"] is False
        assert np.isnan(result["intens"])
        assert result["stop_code"] == -1

    def test_negative_coords(self):
        """Negative x0/y0 should return invalid, not IndexError."""
        image = np.ones((10, 10)) * 42.0
        result = fit_central_pixel(image, None, x0=-5.0, y0=-5.0)
        assert result["valid"] is False
        assert np.isnan(result["intens"])
        assert result["stop_code"] == -1

    def test_edge_pixel_still_valid(self):
        """Coordinates at the image boundary should still work."""
        image = np.arange(100).reshape(10, 10).astype(float)
        result = fit_central_pixel(image, None, x0=9.0, y0=9.0)
        assert result["valid"] is True
        assert result["intens"] == image[9, 9]

    def test_just_outside_boundary(self):
        """Coordinates rounding to exactly shape should be invalid."""
        image = np.ones((10, 10)) * 42.0
        result = fit_central_pixel(image, None, x0=9.6, y0=5.0)
        assert result["valid"] is False
        assert np.isnan(result["intens"])


def _build_mock_isophote(sma, stop_code=0):
    """Create a minimal fit_isophote-like result for driver-flow tests."""
    return {
        "x0": 20.0,
        "y0": 20.0,
        "eps": 0.2,
        "pa": 0.1,
        "sma": sma,
        "intens": 100.0,
        "rms": 1.0,
        "intens_err": 0.1,
        "x0_err": 0.0,
        "y0_err": 0.0,
        "eps_err": 0.0,
        "pa_err": 0.0,
        "tflux_e": np.nan,
        "tflux_c": np.nan,
        "npix_e": 0,
        "npix_c": 0,
        "a3": 0.0,
        "b3": 0.0,
        "a3_err": 0.0,
        "b3_err": 0.0,
        "a4": 0.0,
        "b4": 0.0,
        "a4_err": 0.0,
        "b4_err": 0.0,
        "stop_code": stop_code,
        "niter": 1,
    }


def test_fit_image_passes_previous_geometry_to_growth_calls(monkeypatch):
    """Regular mode should pass previous_geometry for outward and inward calls."""
    image = np.ones((40, 40), dtype=float)
    config = IsosterConfig(
        x0=20.0,
        y0=20.0,
        sma0=10.0,
        minsma=8.0,
        maxsma=12.0,
        astep=2.0,
        linear_growth=True,
    )

    call_log = []

    def fake_fit_isophote(
        image_arg, mask_arg, sma, start_geometry, cfg_arg, going_inwards=False, previous_geometry=None, **kwargs
    ):
        result = _build_mock_isophote(sma=sma, stop_code=0)
        call_log.append(
            {
                "sma": sma,
                "going_inwards": going_inwards,
                "previous_geometry": previous_geometry,
                "result": result,
            }
        )
        return result

    monkeypatch.setattr("isoster.driver.fit_isophote", fake_fit_isophote)

    fit_image(image, mask=None, config=config)

    assert len(call_log) == 3, "Expected first + one outward + one inward fit call"
    # The first call is always the anchor isophote (no previous_geometry).
    anchor_call = call_log[0]
    assert anchor_call["previous_geometry"] is None
    anchor_result = anchor_call["result"]

    # Inward growth runs before outward growth, so the post-anchor call order
    # is [inward, outward]. Check by role, not position.
    inward_calls = [c for c in call_log[1:] if c["going_inwards"]]
    outward_calls = [c for c in call_log[1:] if not c["going_inwards"]]
    assert len(inward_calls) == 1
    assert len(outward_calls) == 1
    assert inward_calls[0]["previous_geometry"] == anchor_result
    assert outward_calls[0]["previous_geometry"] == anchor_result


def test_fit_image_skips_inward_growth_when_first_isophote_fails(monkeypatch):
    """Inward pass should not start unless first isophote passes quality gating.

    With first_isophote_fail_count=3 (default), the driver probes up to 2 extra
    growth steps after the first isophote fails. All probes here return stop_code=3,
    so no growth happens.
    """
    image = np.ones((40, 40), dtype=float)
    config = IsosterConfig(
        x0=20.0,
        y0=20.0,
        sma0=10.0,
        minsma=0.0,
        maxsma=14.0,
        astep=2.0,
        linear_growth=True,
        max_retry_first_isophote=0,  # isolate probe path from retry path
    )

    call_count = {"n": 0}

    def fake_fit_isophote(
        image_arg, mask_arg, sma, start_geometry, cfg_arg, going_inwards=False, previous_geometry=None, **kwargs
    ):
        call_count["n"] += 1
        return _build_mock_isophote(sma=sma, stop_code=3)

    monkeypatch.setattr("isoster.driver.fit_isophote", fake_fit_isophote)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        results = fit_image(image, mask=None, config=config)

    # 1 first iso + 2 probes (sma=12, sma=14) = 3 calls
    assert call_count["n"] == 3
    assert [iso["sma"] for iso in results["isophotes"]] == [0.0]
    assert results.get("first_isophote_failure") is True


def test_fit_image_treats_stop_code_2_as_acceptable(monkeypatch):
    """Stop code 2 should be treated as acceptable for regular growth propagation."""
    image = np.ones((40, 40), dtype=float)
    config = IsosterConfig(
        x0=20.0,
        y0=20.0,
        sma0=10.0,
        minsma=10.0,
        maxsma=14.0,
        astep=2.0,
        linear_growth=True,
    )

    call_count = {"n": 0}

    def fake_fit_isophote(
        image_arg, mask_arg, sma, start_geometry, cfg_arg, going_inwards=False, previous_geometry=None, **kwargs
    ):
        call_count["n"] += 1
        return _build_mock_isophote(sma=sma, stop_code=2)

    monkeypatch.setattr("isoster.driver.fit_isophote", fake_fit_isophote)

    results = fit_image(image, mask=None, config=config)

    assert call_count["n"] == 3
    assert [iso["sma"] for iso in results["isophotes"]] == [10.0, 12.0, 14.0]


def test_fit_image_raises_on_negative_error_in_regular_mode(monkeypatch):
    """Regular fitting mode should fail if any output error field is negative."""
    image = np.ones((40, 40), dtype=float)
    config = IsosterConfig(
        x0=20.0,
        y0=20.0,
        sma0=10.0,
        minsma=10.0,
        maxsma=10.0,
        astep=2.0,
        linear_growth=True,
    )

    def fake_fit_isophote(
        image_arg, mask_arg, sma, start_geometry, cfg_arg, going_inwards=False, previous_geometry=None, **kwargs
    ):
        iso = _build_mock_isophote(sma=sma, stop_code=0)
        iso["x0_err"] = -0.01
        return iso

    monkeypatch.setattr("isoster.driver.fit_isophote", fake_fit_isophote)

    with pytest.raises(ValueError, match="negative error value"):
        fit_image(image, mask=None, config=config)


def test_fit_image_raises_on_negative_error_in_template_forced_mode(monkeypatch):
    """Template-forced mode should fail if extracted photometry has negative errors."""
    image = np.ones((40, 40), dtype=float)
    config = IsosterConfig()
    template = [
        {"x0": 20.0, "y0": 20.0, "eps": 0.2, "pa": 0.0, "sma": 6.0},
    ]

    def fake_extract_forced_photometry(*args, **kwargs):  # noqa: ANN002, ANN003
        iso = _build_mock_isophote(sma=6.0, stop_code=0)
        iso["pa_err"] = -0.05
        return iso

    monkeypatch.setattr("isoster.fitting.extract_forced_photometry", fake_extract_forced_photometry)

    with pytest.raises(ValueError, match="negative error value"):
        fit_image(image, mask=None, config=config, template=template)


def _make_exponential(size=80, seed=1):
    """Small noisy exponential mock for end-to-end driver tests."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    cx = cy = size / 2.0
    image = np.exp(-np.hypot(x - cx, y - cy) / 8.0)
    return image + rng.normal(0.0, 0.005, image.shape)


def test_central_pixel_follows_harmonic_orders():
    """Regression test for B5: central-pixel harmonic keys follow config.

    The central pixel used to always get a3/b3/a4/b4 regardless of
    config.harmonic_orders, so a fit with orders [3, 4, 5, 6] produced two
    distinct key sets in one result list (and masked FITS cells for the
    central pixel's a5 columns).
    """
    image = _make_exponential()
    cfg = IsosterConfig(
        x0=40.0,
        y0=40.0,
        eps=0.2,
        pa=0.0,
        sma0=8.0,
        minsma=0.0,
        maxsma=25.0,
        harmonic_orders=[3, 4, 5, 6],
    )
    isophotes = fit_image(image, config=cfg)["isophotes"]
    central = next(iso for iso in isophotes if iso["sma"] == 0.0)
    fitted = next(iso for iso in isophotes if iso["sma"] > 0.0)

    expected = {f"{p}{n}{s}" for n in (3, 4, 5, 6) for p in ("a", "b") for s in ("", "_err")}
    assert expected <= set(central), f"central pixel missing {sorted(expected - set(central))}"
    assert expected <= set(fitted), f"fitted isophote missing {sorted(expected - set(fitted))}"


def test_central_pixel_omits_harmonics_when_disabled():
    """B5 counterpart: with compute_deviations=False the central pixel
    carries no harmonic keys, matching the fitted rows' schema."""
    image = _make_exponential()
    cfg = IsosterConfig(
        x0=40.0,
        y0=40.0,
        eps=0.2,
        pa=0.0,
        sma0=8.0,
        minsma=0.0,
        maxsma=25.0,
        compute_deviations=False,
    )
    isophotes = fit_image(image, config=cfg)["isophotes"]
    central = next(iso for iso in isophotes if iso["sma"] == 0.0)
    fitted = next(iso for iso in isophotes if iso["sma"] > 0.0)
    for row in (central, fitted):
        assert "a3" not in row and "b4" not in row


def test_central_pixel_wls_error_from_variance_map():
    """M12 parity: central pixel reports sqrt(var) under WLS, 0.0 under OLS."""
    image = _make_exponential()
    var = np.full_like(image, 0.25)
    cfg = IsosterConfig(x0=40.0, y0=40.0, eps=0.2, pa=0.0, sma0=8.0, minsma=0.0, maxsma=25.0)
    central_wls = next(iso for iso in fit_image(image, config=cfg, variance_map=var)["isophotes"] if iso["sma"] == 0.0)
    assert central_wls["intens_err"] == pytest.approx(0.5, rel=1e-9)

    central_ols = next(iso for iso in fit_image(image, config=cfg)["isophotes"] if iso["sma"] == 0.0)
    assert central_ols["intens_err"] == 0.0
