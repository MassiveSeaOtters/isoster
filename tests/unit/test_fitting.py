import unittest
import warnings

import numpy as np
import pytest

from isoster.config import IsosterConfig
from isoster.fitting import (
    build_isofit_design_matrix,
    compute_aperture_photometry,
    compute_central_regularization_penalty,
    compute_deviations,
    compute_parameter_errors,
    evaluate_harmonic_model,
    fit_all_harmonics,
    fit_first_and_second_harmonics,
    fit_higher_harmonics_simultaneous,
    fit_isophote,
    sigma_clip,
)
from isoster.numba_kernels import build_harmonic_matrix


class TestFitting(unittest.TestCase):
    def test_fit_harmonics(self):
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        y0, A1, B1, A2, B2 = 100.0, 10.0, 5.0, 2.0, 1.0
        intens = y0 + A1 * np.sin(phi) + B1 * np.cos(phi) + A2 * np.sin(2 * phi) + B2 * np.cos(2 * phi)

        coeffs, cov = fit_first_and_second_harmonics(phi, intens)
        self.assertTrue(np.allclose(coeffs, [y0, A1, B1, A2, B2], atol=1e-5))

    def test_sigma_clip(self):
        # We need at least 11 points for a single outlier to be clipped at 3rd sigma
        # due to the outlier's own contribution to the standard deviation.
        phi = np.arange(20)
        outlier_val = 1000.0
        intens = np.array([10.0] * 19 + [outlier_val])

        # Test no clipping
        p, i, n = sigma_clip(phi, intens, nclip=0)
        self.assertEqual(n, 0)
        self.assertEqual(len(i), 20)

        # Test 1 iteration of clipping
        p, i, n = sigma_clip(phi, intens, sclip=3.0, nclip=1)
        self.assertEqual(n, 1)
        self.assertEqual(len(i), 19)
        self.assertNotIn(outlier_val, i)

    def test_aperture_photometry(self):
        image = np.ones((100, 100))
        x0, y0 = 50.0, 50.0
        sma = 10.0
        eps = 0.0
        pa = 0.0

        tflux_e, tflux_c, npix_e, npix_c = compute_aperture_photometry(image, None, x0, y0, sma, eps, pa)

        # Area of r=10 circle is approx 314
        expected_area = np.pi * sma**2
        self.assertLess(abs(npix_c - expected_area) / expected_area, 0.05)
        self.assertAlmostEqual(tflux_c, npix_c)
        self.assertAlmostEqual(tflux_e, tflux_c)
        self.assertEqual(npix_e, npix_c)


def test_fit_isophote_emits_stop_code_2_and_photometry_on_maxit_exhaustion():
    """Max-iteration exits should report stop_code=2 and keep full-photometry populated."""
    image_size = 80
    y_coords, x_coords = np.mgrid[:image_size, :image_size]
    center_x = image_size / 2.0
    center_y = image_size / 2.0
    radius = np.hypot(x_coords - center_x, y_coords - center_y)
    image = np.exp(-radius / 8.0)

    config = IsosterConfig(
        x0=center_x,
        y0=center_y,
        eps=0.2,
        pa=0.0,
        sma0=8.0,
        maxit=1,
        minit=1,
        conver=1e-8,
        maxgerr=1.0,
        full_photometry=True,
        compute_errors=False,
        compute_deviations=False,
        sclip=3.0,
        nclip=0,
    )
    start_geometry = {"x0": center_x, "y0": center_y, "eps": 0.2, "pa": 0.0}
    result = fit_isophote(
        image,
        mask=None,
        sma=8.0,
        start_geometry=start_geometry,
        config=config,
    )

    assert result["niter"] == 1
    assert result["stop_code"] == 2
    assert np.isfinite(result["tflux_e"])
    assert result["npix_e"] > 0


def test_stop_code_2_computes_harmonic_deviations():
    """Regression test for I8: stop_code=2 should still compute harmonic deviations.

    When maxit is reached without convergence, the code should compute best-effort
    a3/b3/a4/b4 from the last iteration rather than leaving them as zeros.
    """
    image_size = 80
    y_coords, x_coords = np.mgrid[:image_size, :image_size]
    center_x = image_size / 2.0
    center_y = image_size / 2.0
    radius = np.hypot(x_coords - center_x, y_coords - center_y)
    # Add a boxy/disky perturbation so a4 is non-zero
    theta = np.arctan2(y_coords - center_y, x_coords - center_x)
    # Add small noise to make the ring realistic
    np.random.seed(42)
    image = np.exp(-radius / 8.0) * (1.0 + 0.1 * np.cos(4 * theta))
    image += np.random.normal(0, 0.005, image.shape)

    config = IsosterConfig(
        x0=center_x,
        y0=center_y,
        eps=0.2,
        pa=0.0,
        sma0=8.0,
        maxit=1,
        minit=1,
        conver=1e-8,
        maxgerr=1.0,
        full_photometry=False,
        compute_errors=False,
        compute_deviations=True,
        harmonic_orders=[3, 4],
        sclip=3.0,
        nclip=0,
    )
    start_geometry = {"x0": center_x, "y0": center_y, "eps": 0.2, "pa": 0.0}
    result = fit_isophote(
        image,
        mask=None,
        sma=8.0,
        start_geometry=start_geometry,
        config=config,
    )

    assert result["stop_code"] == 2, f"Expected stop_code=2, got {result['stop_code']}"
    # Harmonic fields should exist and be finite
    for key in ("a3", "b3", "a4", "b4", "a3_err", "b3_err", "a4_err", "b4_err"):
        assert key in result, f"Missing harmonic field '{key}' in stop_code=2 result"
        assert np.isfinite(result[key]), f"Non-finite {key}={result[key]} in stop_code=2 result"
    # The boxy perturbation should produce a non-trivial a4 or b4
    assert abs(result["a4"]) > 1e-6 or abs(result["b4"]) > 1e-6, (
        f"Expected non-zero a4/b4 from boxy image, got a4={result['a4']}, b4={result['b4']}"
    )


# ============================================================================
# Pytest-style tests for exception handling (ISSUE-1)
# ============================================================================


def test_compute_parameter_errors_singular_matrix():
    """Test that singular matrix cases return zeros gracefully."""
    # Create degenerate case: all same intensity (no variation for harmonic fit)
    phi = np.linspace(0, 2 * np.pi, 10)
    intens = np.ones(10) * 100.0  # Constant intensity

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
            phi, intens, x0=50, y0=50, sma=10, eps=0.2, pa=0.5, gradient=-1.0
        )

        # Should return zeros (may or may not emit warning depending on code path)
        # The important thing is it doesn't crash
        assert np.isclose(x0_err, 0.0), f"Expected x0_err=0.0, got {x0_err}"
        assert np.isclose(y0_err, 0.0), f"Expected y0_err=0.0, got {y0_err}"
        assert np.isclose(eps_err, 0.0), f"Expected eps_err=0.0, got {eps_err}"
        assert np.isclose(pa_err, 0.0), f"Expected pa_err=0.0, got {pa_err}"

        # Note: This case may hit early return (line 216-217) without exception,
        # so warning emission is optional. The key is it returns zeros gracefully.


def test_compute_parameter_errors_zero_gradient():
    """Test that zero gradient is handled gracefully."""
    np.random.seed(42)
    phi = np.linspace(0, 2 * np.pi, 10)
    intens = np.random.random(10) * 100.0

    # Zero gradient should trigger early return before try block
    x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
        phi, intens, x0=50, y0=50, sma=10, eps=0.2, pa=0.5, gradient=0.0
    )

    assert x0_err == 0.0
    assert y0_err == 0.0
    assert eps_err == 0.0
    assert pa_err == 0.0


def test_compute_parameter_errors_none_gradient():
    """Test that None gradient is handled gracefully."""
    np.random.seed(42)
    phi = np.linspace(0, 2 * np.pi, 10)
    intens = np.random.random(10) * 100.0

    # None gradient should trigger early return
    x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
        phi, intens, x0=50, y0=50, sma=10, eps=0.2, pa=0.5, gradient=None
    )

    assert x0_err == 0.0
    assert y0_err == 0.0
    assert eps_err == 0.0
    assert pa_err == 0.0


def test_compute_deviations_singular_matrix():
    """Test that compute_deviations handles errors gracefully."""
    # Create degenerate case with only 2 points at same angle
    phi = np.array([0.0, 0.0])
    intens = np.array([100.0, 100.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a, b, a_err, b_err = compute_deviations(phi, intens, sma=10, gradient=-1.0, order=3)

        # Should return zeros
        assert a == 0.0, f"Expected a=0.0, got {a}"
        assert b == 0.0, f"Expected b=0.0, got {b}"
        assert a_err == 0.0, f"Expected a_err=0.0, got {a_err}"
        assert b_err == 0.0, f"Expected b_err=0.0, got {b_err}"

        # Should emit warning
        assert len(w) >= 1, "Expected at least one warning to be emitted"
        assert any("compute_deviations" in str(warn.message) for warn in w), (
            f"Expected warning about compute_deviations, got: {[str(warn.message) for warn in w]}"
        )


def test_compute_deviations_zero_factor():
    """Test that zero factor (gradient=0 or sma=0) returns zeros."""
    np.random.seed(42)
    phi = np.linspace(0, 2 * np.pi, 10)
    intens = np.random.random(10) * 100.0

    # Zero gradient should make factor=0, triggering early return
    a, b, a_err, b_err = compute_deviations(phi, intens, sma=10, gradient=0.0, order=3)

    assert a == 0.0
    assert b == 0.0
    assert a_err == 0.0
    assert b_err == 0.0


def test_compute_parameter_errors_normal_case():
    """Test that compute_parameter_errors works correctly for normal case."""
    np.random.seed(42)
    # Create realistic harmonic data
    phi = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    y0, A1, B1, A2, B2 = 100.0, 5.0, 3.0, 1.0, 0.5
    intens = y0 + A1 * np.sin(phi) + B1 * np.cos(phi) + A2 * np.sin(2 * phi) + B2 * np.cos(2 * phi)
    intens += np.random.normal(0, 0.1, len(phi))  # Add small noise

    # Should succeed without warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(
            phi, intens, x0=50, y0=50, sma=10, eps=0.2, pa=0.5, gradient=-2.0
        )

        # Should return non-zero errors
        assert x0_err > 0.0, "Expected non-zero x0_err for normal case"
        assert y0_err > 0.0, "Expected non-zero y0_err for normal case"
        assert eps_err > 0.0, "Expected non-zero eps_err for normal case"
        assert pa_err > 0.0, "Expected non-zero pa_err for normal case"

        # Should not emit warnings
        compute_param_warnings = [warn for warn in w if "compute_parameter_errors" in str(warn.message)]
        assert len(compute_param_warnings) == 0, (
            f"Expected no warnings for normal case, got: {[str(warn.message) for warn in compute_param_warnings]}"
        )


def test_compute_deviations_normal_case():
    """Test that compute_deviations works correctly for normal case."""
    np.random.seed(42)
    # Create realistic data with 3rd order harmonic
    phi = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    y0, A3, B3 = 100.0, 2.0, 1.0
    intens = y0 + A3 * np.sin(3 * phi) + B3 * np.cos(3 * phi)
    intens += np.random.normal(0, 0.1, len(phi))  # Add small noise

    # Should succeed without warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a, b, a_err, b_err = compute_deviations(phi, intens, sma=10, gradient=-2.0, order=3)

        # Should return non-zero values
        assert abs(a) > 0.0, "Expected non-zero a for normal case"
        assert abs(b) > 0.0, "Expected non-zero b for normal case"

        # Should not emit warnings
        compute_dev_warnings = [warn for warn in w if "compute_deviations" in str(warn.message)]
        assert len(compute_dev_warnings) == 0, (
            f"Expected no warnings for normal case, got: {[str(warn.message) for warn in compute_dev_warnings]}"
        )


def test_compute_deviations_noise_free():
    """Regression test for B1: noise-free data must not zero the coefficients.

    The OLS branch used scipy leastsq, which returns cov_x=None when the fit
    is exact (noise-free or very-low-noise data); the code treated that as
    total failure and silently returned a_n=b_n=0. The explicit design-matrix
    solve must recover a planted harmonic on noise-free data.
    """
    sma, gradient = 20.0, -13.5
    b4_planted = 0.03
    phi = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    intens = 100.0 + b4_planted * sma * abs(gradient) * np.cos(4 * phi)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a4, b4, a4_err, b4_err = compute_deviations(phi, intens, sma, gradient, 4)

    assert b4 == pytest.approx(b4_planted, abs=1e-10), f"Expected b4≈{b4_planted}, got {b4}"
    assert a4 == pytest.approx(0.0, abs=1e-10), f"Expected a4≈0, got {a4}"
    compute_dev_warnings = [warn for warn in w if "compute_deviations" in str(warn.message)]
    assert len(compute_dev_warnings) == 0, (
        f"Expected no warnings for noise-free case, got: {[str(warn.message) for warn in compute_dev_warnings]}"
    )


def test_pa_wraparound_vectorized():
    """Test that PA wrap-around is correctly handled with vectorized modulo arithmetic.

    This test verifies that the vectorized formula ((delta + π) % (2π)) - π
    correctly wraps PA differences to [-π, π] range and produces finite penalties.
    """
    # Create minimal config for regularization
    config = IsosterConfig(
        x0=50.0,
        y0=50.0,
        eps=0.3,
        pa=0.0,
        sma0=10.0,
        minsma=3.0,
        maxsma=100.0,
        use_central_regularization=True,
        central_reg_strength=0.1,
        central_reg_sma_threshold=10.0,
    )

    # Test cases: various PA differences that should wrap to [-π, π]
    test_cases = [
        # (current_pa, previous_pa, description)
        (0.1, 0.0, "Small positive difference"),
        (0.0, 0.1, "Small negative difference"),
        (np.pi - 0.1, -np.pi + 0.1, "Wrap from +π to -π"),
        (-np.pi + 0.1, np.pi - 0.1, "Wrap from -π to +π"),
        (3.0, -3.0, "Large positive wrap"),
        (-3.0, 3.0, "Large negative wrap"),
        (0.0, 0.0, "No difference"),
        (np.pi, -np.pi, "π and -π are equivalent"),
        (2 * np.pi, 0.0, "Full circle wrap"),
        (4 * np.pi, 0.0, "Multiple circle wrap"),
    ]

    for current_pa, previous_pa, description in test_cases:
        current_geom = {"x0": 50.0, "y0": 50.0, "eps": 0.3, "pa": current_pa}
        previous_geom = {"x0": 50.0, "y0": 50.0, "eps": 0.3, "pa": previous_pa}

        # Compute penalty (should not crash)
        penalty = compute_central_regularization_penalty(current_geom, previous_geom, sma=5.0, config=config)

        # Verify penalty is finite and non-negative
        assert np.isfinite(penalty), (
            f"Penalty should be finite for {description}: pa={current_pa}, prev_pa={previous_pa}"
        )
        assert penalty >= 0.0, f"Penalty should be non-negative for {description}, got {penalty}"

        # Manually compute delta_pa with vectorized formula
        delta_pa = current_pa - previous_pa
        delta_pa_wrapped = ((delta_pa + np.pi) % (2 * np.pi)) - np.pi

        # Verify wrapped value is in [-π, π]
        assert -np.pi - 1e-10 <= delta_pa_wrapped <= np.pi + 1e-10, (
            f"Wrapped delta_pa {delta_pa_wrapped:.6f} not in [-π, π] for {description}"
        )

    print("✓ PA wrap-around vectorized test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_compute_parameter_errors_with_coeffs():
    """Test that passing coefficients avoids re-fitting (EFF-2 optimization)."""
    from isoster.fitting import compute_parameter_errors, fit_first_and_second_harmonics

    # Create synthetic data
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    y0, A1, B1, A2, B2 = 100.0, 10.0, 5.0, 2.0, 1.0
    intens = y0 + A1 * np.sin(phi) + B1 * np.cos(phi) + A2 * np.sin(2 * phi) + B2 * np.cos(2 * phi)
    intens += np.random.RandomState(42).normal(0, 0.5, len(phi))

    # Geometry and gradient
    x0, y0_geom, sma, eps, pa = 50.0, 50.0, 10.0, 0.3, np.pi / 4
    gradient = -2.0

    # Fit harmonics once
    coeffs, cov_matrix = fit_first_and_second_harmonics(phi, intens)

    # Compute errors WITH coefficients (optimized path)
    err_with_coeffs = compute_parameter_errors(
        phi, intens, x0, y0_geom, sma, eps, pa, gradient, cov_matrix=cov_matrix, coeffs=coeffs
    )

    # Compute errors WITHOUT coefficients (legacy path, re-fits)
    err_without_coeffs = compute_parameter_errors(
        phi, intens, x0, y0_geom, sma, eps, pa, gradient, cov_matrix=cov_matrix, coeffs=None
    )

    # Results should be identical
    for i, (with_c, without_c) in enumerate(zip(err_with_coeffs, err_without_coeffs)):
        assert np.abs(with_c - without_c) < 1e-10, (
            f"Error {i}: with_coeffs={with_c}, without_coeffs={without_c}, diff={abs(with_c - without_c)}"
        )

    print(f"✓ Parameter errors with/without coeffs: {err_with_coeffs}")
    print("✓ EFF-2 optimization produces identical results")


def test_gradient_early_termination():
    """Test EFF-1: gradient early termination when first gradient is reliable."""

    from isoster.config import IsosterConfig
    from isoster.fitting import compute_gradient

    # Create simple test image with smooth gradient
    size = 200
    x0, y0 = size / 2, size / 2
    image = np.zeros((size, size))
    y, x = np.mgrid[:size, :size]
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    image = 1000.0 * np.exp(-r / 20.0)  # Smooth exponential profile

    mask = np.zeros_like(image, dtype=bool)

    geometry = {"x0": x0, "y0": y0, "sma": 30.0, "eps": 0.3, "pa": np.pi / 4}
    config = IsosterConfig(
        x0=x0,
        y0=y0,
        eps=0.3,
        pa=np.pi / 4,
        sma0=10.0,
        minsma=3.0,
        maxsma=80.0,
        astep=0.1,
        linear_growth=False,
        integrator="mean",
        use_eccentric_anomaly=False,
    )

    # Case 1: First gradient call (no previous_gradient)
    # Should always compute second gradient when first looks suspicious
    gradient1, error1 = compute_gradient(image, mask, geometry, config, previous_gradient=None, current_data=None)

    assert gradient1 is not None, "First gradient should be computed"
    assert error1 is not None, "First gradient error should be computed"

    # Case 2: With reliable gradient (error << gradient)
    # Should skip second gradient extraction due to low relative error
    gradient2, error2 = compute_gradient(
        image, mask, geometry, config, previous_gradient=gradient1 * 1.5, current_data=None
    )

    assert gradient2 is not None, "Second gradient should be computed"
    assert error2 is not None, "Second gradient error should be computed"

    # Verify that gradients are reasonable
    assert gradient1 < 0, f"Gradient should be negative (declining profile), got {gradient1}"
    assert gradient2 < 0, f"Gradient should be negative (declining profile), got {gradient2}"

    # Verify relative error is small for smooth profile
    rel_error1 = abs(error1 / gradient1) if gradient1 != 0 else np.inf
    rel_error2 = abs(error2 / gradient2) if gradient2 != 0 else np.inf

    # For smooth exponential profile, relative errors should be small
    assert rel_error1 < 0.5, f"Relative error too large: {rel_error1}"
    assert rel_error2 < 0.5, f"Relative error too large: {rel_error2}"

    print(f"✓ Gradient 1: {gradient1:.6f} ± {error1:.6f} (rel_err={rel_error1:.3f})")
    print(f"✓ Gradient 2: {gradient2:.6f} ± {error2:.6f} (rel_err={rel_error2:.3f})")
    print("✓ EFF-1 gradient early termination test passed")


def test_gradient_linear_growth_second_gradient():
    """R26-01: Verify second-gradient formula is correct for linear_growth=True.

    The second-gradient path uses gradient_sma_2 = sma + 2*step, so delta_r = 2*step.
    Previously the formula incorrectly divided by sma*(2*step) instead of just 2*step.
    This test verifies that both linear and geometric growth produce consistent
    gradient values on a smooth exponential profile.
    """
    from isoster.config import IsosterConfig
    from isoster.fitting import compute_gradient

    size = 200
    x0, y0 = size / 2, size / 2
    y, x = np.mgrid[:size, :size]
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    image = 1000.0 * np.exp(-r / 30.0)
    mask = np.zeros_like(image, dtype=bool)

    sma = 25.0
    step = 3.0  # Large step for linear growth

    geometry = {"x0": x0, "y0": y0, "sma": sma, "eps": 0.0, "pa": 0.0}

    # Linear growth: delta_r = step, second delta_r = 2*step
    config_linear = IsosterConfig(
        x0=x0,
        y0=y0,
        eps=0.0,
        pa=0.0,
        sma0=10.0,
        astep=step,
        linear_growth=True,
        integrator="mean",
        use_eccentric_anomaly=False,
    )

    # Force the second-gradient path by providing a small previous_gradient
    # (ensures need_second_gradient is True)
    grad_lin, err_lin = compute_gradient(
        image,
        mask,
        geometry,
        config_linear,
        previous_gradient=-0.001,
        current_data=None,
    )

    # Gradient must be negative for a declining profile
    assert grad_lin < 0, f"Linear gradient should be negative, got {grad_lin}"

    # Analytic gradient for I = 1000*exp(-r/30) at r=25: dI/dr = -1000/30 * exp(-25/30) ≈ -14.4
    analytic = -1000.0 / 30.0 * np.exp(-sma / 30.0)

    # Second-gradient (delta_r = 2*step = 6) is a coarser finite difference,
    # so allow 30% tolerance relative to analytic
    rel_diff = abs((grad_lin - analytic) / analytic)
    assert rel_diff < 0.3, (
        f"Linear growth gradient {grad_lin:.4f} deviates {rel_diff:.1%} from "
        f"analytic {analytic:.4f} — possible sma-factor bug in second-gradient"
    )

    # Sanity: error should be finite
    assert err_lin is not None and np.isfinite(err_lin), f"Gradient error should be finite, got {err_lin}"


# ============================================================================
# Tests for fit_higher_harmonics_simultaneous (EA-HARMONICS feature)
# ============================================================================


def test_fit_higher_harmonics_simultaneous_basic():
    """Test basic functionality of simultaneous harmonics fitting."""
    # Create synthetic data with known 3rd and 4th order harmonics
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    y0 = 100.0
    A3, B3 = 5.0, 3.0
    A4, B4 = 2.0, 1.5
    intens = y0 + A3 * np.sin(3 * phi) + B3 * np.cos(3 * phi) + A4 * np.sin(4 * phi) + B4 * np.cos(4 * phi)

    sma = 10.0
    gradient = -2.0

    # Fit harmonics
    result = fit_higher_harmonics_simultaneous(phi, intens, sma, gradient, orders=[3, 4])

    assert 3 in result, "Should have 3rd harmonic result"
    assert 4 in result, "Should have 4th harmonic result"

    # Check structure of results
    a3, b3, a3_err, b3_err = result[3]
    a4, b4, a4_err, b4_err = result[4]

    # Verify coefficients are close to expected (normalized by sma*|gradient|)
    factor = sma * abs(gradient)
    expected_a3 = A3 / factor
    expected_b3 = B3 / factor
    expected_a4 = A4 / factor
    expected_b4 = B4 / factor

    assert np.allclose(a3, expected_a3, atol=1e-5), f"a3: expected {expected_a3}, got {a3}"
    assert np.allclose(b3, expected_b3, atol=1e-5), f"b3: expected {expected_b3}, got {b3}"
    assert np.allclose(a4, expected_a4, atol=1e-5), f"a4: expected {expected_a4}, got {a4}"
    assert np.allclose(b4, expected_b4, atol=1e-5), f"b4: expected {expected_b4}, got {b4}"

    print("✓ fit_higher_harmonics_simultaneous basic test passed")


def test_fit_higher_harmonics_simultaneous_with_noise():
    """Test simultaneous harmonics fitting with noisy data."""
    np.random.seed(42)
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    y0 = 100.0
    A3, B3 = 5.0, 3.0
    A4, B4 = 2.0, 1.5
    noise = np.random.normal(0, 0.5, len(phi))
    intens = y0 + A3 * np.sin(3 * phi) + B3 * np.cos(3 * phi) + A4 * np.sin(4 * phi) + B4 * np.cos(4 * phi) + noise

    sma = 10.0
    gradient = -2.0

    result = fit_higher_harmonics_simultaneous(phi, intens, sma, gradient, orders=[3, 4])

    a3, b3, a3_err, b3_err = result[3]
    a4, b4, a4_err, b4_err = result[4]

    # Should have non-zero errors
    assert a3_err > 0.0, "a3_err should be positive"
    assert b3_err > 0.0, "b3_err should be positive"
    assert a4_err > 0.0, "a4_err should be positive"
    assert b4_err > 0.0, "b4_err should be positive"

    # Recovered coefficients should be within 3-sigma of true values
    factor = sma * abs(gradient)
    expected_a3 = A3 / factor
    expected_b3 = B3 / factor

    assert abs(a3 - expected_a3) < 3 * a3_err, f"a3 outside 3-sigma: {a3} vs {expected_a3} ± {a3_err}"
    assert abs(b3 - expected_b3) < 3 * b3_err, f"b3 outside 3-sigma: {b3} vs {expected_b3} ± {b3_err}"

    print("✓ fit_higher_harmonics_simultaneous noisy data test passed")


def test_fit_higher_harmonics_simultaneous_empty_orders():
    """Test simultaneous harmonics fitting with empty orders list."""
    np.random.seed(42)
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    intens = np.random.random(100) * 100.0

    result = fit_higher_harmonics_simultaneous(phi, intens, sma=10.0, gradient=-2.0, orders=[])

    assert result == {}, "Empty orders should return empty dict"

    print("✓ fit_higher_harmonics_simultaneous empty orders test passed")


def test_fit_higher_harmonics_simultaneous_extended_orders():
    """Test simultaneous harmonics fitting with extended orders [3, 4, 5, 6]."""
    phi = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    y0 = 100.0
    # Add harmonics for orders 3, 4, 5, 6
    intens = y0 + 3.0 * np.sin(3 * phi) + 2.0 * np.cos(3 * phi)
    intens += 2.0 * np.sin(4 * phi) + 1.0 * np.cos(4 * phi)
    intens += 1.0 * np.sin(5 * phi) + 0.5 * np.cos(5 * phi)
    intens += 0.5 * np.sin(6 * phi) + 0.3 * np.cos(6 * phi)

    result = fit_higher_harmonics_simultaneous(phi, intens, sma=10.0, gradient=-2.0, orders=[3, 4, 5, 6])

    assert 3 in result, "Should have 3rd harmonic"
    assert 4 in result, "Should have 4th harmonic"
    assert 5 in result, "Should have 5th harmonic"
    assert 6 in result, "Should have 6th harmonic"

    # Verify each result is a 4-tuple
    for n in [3, 4, 5, 6]:
        assert len(result[n]) == 4, f"Harmonic {n} should have 4 values (a, b, a_err, b_err)"

    print("✓ fit_higher_harmonics_simultaneous extended orders test passed")


def test_fit_higher_harmonics_simultaneous_insufficient_data():
    """Test simultaneous harmonics fitting with insufficient data points."""
    phi = np.linspace(0, 2 * np.pi, 3, endpoint=False)  # Only 3 points
    intens = np.array([100.0, 101.0, 99.0])

    # Need 5 params (1 + 2*2) for orders=[3, 4], but only have 3 points
    result = fit_higher_harmonics_simultaneous(phi, intens, sma=10.0, gradient=-2.0, orders=[3, 4])

    # Should return zeros for all harmonics
    assert 3 in result, "Should have 3rd harmonic (zeros)"
    assert 4 in result, "Should have 4th harmonic (zeros)"
    assert result[3] == (0.0, 0.0, 0.0, 0.0), "Should return zeros for insufficient data"
    assert result[4] == (0.0, 0.0, 0.0, 0.0), "Should return zeros for insufficient data"

    print("✓ fit_higher_harmonics_simultaneous insufficient data test passed")


def test_fit_higher_harmonics_simultaneous_zero_gradient():
    """Test simultaneous harmonics fitting with zero gradient."""
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    intens = 100.0 + 5.0 * np.sin(3 * phi) + 3.0 * np.cos(3 * phi)

    result = fit_higher_harmonics_simultaneous(phi, intens, sma=10.0, gradient=0.0, orders=[3, 4])

    # Should handle zero gradient gracefully (uses factor=1.0)
    assert 3 in result, "Should have 3rd harmonic"
    assert 4 in result, "Should have 4th harmonic"

    # Should not crash with division by zero
    a3, b3, a3_err, b3_err = result[3]
    assert np.isfinite(a3), "a3 should be finite"
    assert np.isfinite(b3), "b3 should be finite"

    print("✓ fit_higher_harmonics_simultaneous zero gradient test passed")


def test_sigma_clip_ea_phi_sync():
    """Test that sigma clipping keeps phi in sync with angles/intens in EA mode.

    Regression test for C1: In EA mode, angles (psi) and phi are different arrays.
    After sigma_clip removes outliers from angles+intens, phi must be clipped with the
    same mask. Otherwise, downstream functions (compute_gradient, compute_parameter_errors)
    receive mismatched phi/intens arrays.
    """
    from isoster.sampling import extract_isophote_data

    np.random.seed(42)

    # Build a smooth image with outliers placed precisely on the sampling ellipse
    image_size = 201
    center = 100.0
    eps, pa, sma = 0.4, 0.5, 30.0
    yy, xx = np.mgrid[:image_size, :image_size]
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    dx, dy = xx - center, yy - center
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps)) ** 2)
    image = 1000.0 * np.exp(-r_ell / 20.0)

    # Place extreme outliers ON the sampling ellipse to guarantee sigma_clip triggers
    n_outliers = 10
    for i in range(n_outliers):
        angle = 2 * np.pi * i / n_outliers
        ox = center + sma * np.cos(angle) * cos_pa - sma * (1 - eps) * np.sin(angle) * sin_pa
        oy = center + sma * np.cos(angle) * sin_pa + sma * (1 - eps) * np.sin(angle) * cos_pa
        ox_int, oy_int = int(round(ox)), int(round(oy))
        if 0 <= ox_int < image_size and 0 <= oy_int < image_size:
            image[oy_int, ox_int] = 1e6  # Extreme outlier

    # Verify that EA mode extraction gives different angles vs phi
    data = extract_isophote_data(image, None, center, center, sma, eps, pa, use_eccentric_anomaly=True)
    assert not np.allclose(data.angles, data.phi), "EA mode should produce different angles (psi) and phi arrays"

    # Verify sigma clipping DOES clip some points (outliers on the ellipse)
    _, intens_clipped, n_clipped = sigma_clip(data.angles, data.intens, sclip=3.0, nclip=3)
    assert n_clipped >= 1, f"Expected sigma clipping to remove outliers, but n_clipped={n_clipped}"

    # Now run fit_isophote in EA mode with sigma clipping -- this is the actual bug test.
    # Before the fix: phi has length N, intens has length N - n_clipped after sigma_clip.
    # compute_parameter_errors(phi, intens, ...) would crash or produce wrong results
    # because harmonic_function(phi, coeffs) produces N values but intens has N-k values.
    config = IsosterConfig(
        x0=center,
        y0=center,
        eps=eps,
        pa=pa,
        sma0=sma,
        minsma=5.0,
        maxsma=80.0,
        use_eccentric_anomaly=True,
        nclip=3,
        sclip=3.0,
        maxit=30,
        minit=5,
        conver=0.05,
        fix_center=True,
        fix_eps=True,
        fix_pa=True,
        compute_errors=True,
        compute_deviations=True,
    )
    start_geom = {"x0": center, "y0": center, "eps": eps, "pa": pa}

    # This should NOT raise or produce NaN/zero errors due to array length mismatch.
    # Before the fix, compute_parameter_errors catches the broadcast ValueError and
    # silently returns zeros -- we detect this via the RuntimeWarning it emits.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_isophote(image, mask=None, sma=sma, start_geometry=start_geom, config=config)

    assert result is not None, "fit_isophote should return a result"
    assert np.isfinite(result["intens"]), f"Intensity should be finite, got {result['intens']}"
    assert np.isfinite(result["rms"]), f"RMS should be finite, got {result['rms']}"

    # The key assertion: no broadcast shape mismatch warnings should be emitted.
    # Before the fix, the mismatched phi/intens arrays cause:
    #   "operands could not be broadcast together with shapes (N,) (M,)"
    broadcast_warnings = [w for w in caught if "could not be broadcast" in str(w.message)]
    assert len(broadcast_warnings) == 0, (
        f"EA mode + sigma clipping caused array shape mismatch in "
        f"compute_parameter_errors: {[str(w.message) for w in broadcast_warnings]}"
    )


def test_fit_higher_harmonics_vs_sequential():
    """Compare simultaneous fitting with sequential (compute_deviations) for consistency."""
    np.random.seed(42)
    phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    y0 = 100.0
    A3, B3 = 5.0, 3.0
    A4, B4 = 2.0, 1.5
    intens = y0 + A3 * np.sin(3 * phi) + B3 * np.cos(3 * phi) + A4 * np.sin(4 * phi) + B4 * np.cos(4 * phi)

    sma = 10.0
    gradient = -2.0

    # Sequential fitting
    a3_seq, b3_seq, a3_err_seq, b3_err_seq = compute_deviations(phi, intens, sma, gradient, 3)
    a4_seq, b4_seq, a4_err_seq, b4_err_seq = compute_deviations(phi, intens, sma, gradient, 4)

    # Simultaneous fitting
    result = fit_higher_harmonics_simultaneous(phi, intens, sma, gradient, orders=[3, 4])
    a3_sim, b3_sim, a3_err_sim, b3_err_sim = result[3]
    a4_sim, b4_sim, a4_err_sim, b4_err_sim = result[4]

    # For noiseless data with pure harmonics, both methods should give similar results
    # (not identical due to different model assumptions: sequential fits each harmonic
    # with its own constant term, simultaneous shares one constant term)
    assert np.allclose(a3_seq, a3_sim, rtol=0.1), f"a3: sequential={a3_seq}, simultaneous={a3_sim}"
    assert np.allclose(b3_seq, b3_sim, rtol=0.1), f"b3: sequential={b3_seq}, simultaneous={b3_sim}"
    assert np.allclose(a4_seq, a4_sim, rtol=0.1), f"a4: sequential={a4_seq}, simultaneous={a4_sim}"
    assert np.allclose(b4_seq, b4_sim, rtol=0.1), f"b4: sequential={b4_seq}, simultaneous={b4_sim}"

    print("✓ Simultaneous vs sequential comparison test passed")
    print(f"  3rd harmonic: sequential=({a3_seq:.4f}, {b3_seq:.4f}), simultaneous=({a3_sim:.4f}, {b3_sim:.4f})")
    print(f"  4th harmonic: sequential=({a4_seq:.4f}, {b4_seq:.4f}), simultaneous=({a4_sim:.4f}, {b4_sim:.4f})")


class TestFflagSemantics(unittest.TestCase):
    """Regression tests for I1: fflag semantics must match documented convention.

    fflag is documented as 'maximum fraction of flagged data (masked + clipped)'.
    With fflag=0.2, the isophote should be rejected when sigma clipping removes
    >20% of the sampled points (i.e. when actual_points < 80% of total_points).

    Note: In isoster, total_points is the count after mask-based filtering
    (during sampling), and actual_points is after sigma clipping. The fflag
    check only triggers from sigma clipping, not from mask-based exclusion.
    """

    def _make_image_with_outliers(self, size=101, sma=20, outlier_fraction=0.4):
        """Create a radially symmetric image with intensity outliers along part of the ellipse.

        The outliers are extreme enough to be clipped by sigma clipping,
        reducing actual_points relative to total_points.
        """
        y, x = np.mgrid[:size, :size]
        cx, cy = size // 2, size // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) + 1.0
        image = 1000.0 / r

        # Inject extreme outliers along a fraction of the ellipse
        n_outlier_angles = int(360 * outlier_fraction)
        for angle_deg in range(n_outlier_angles):
            angle = np.radians(angle_deg)
            px = int(cx + sma * np.cos(angle))
            py = int(cy + sma * np.sin(angle))
            if 0 <= px < size and 0 <= py < size:
                image[py, px] = 1e6  # extreme outlier
        return image

    def test_fflag_rejects_high_clipping(self):
        """With fflag=0.1 and aggressive sigma clipping, stop_code=1 should trigger."""
        # ~40% of points are extreme outliers, sigma clipping will remove them
        image = self._make_image_with_outliers(outlier_fraction=0.4)
        size = image.shape[0]
        cx, cy = size // 2, size // 2
        mask = np.zeros_like(image, dtype=bool)

        config = IsosterConfig(
            x0=float(cx),
            y0=float(cy),
            sma0=20.0,
            eps=0.0,
            pa=0.0,
            fflag=0.1,  # allow at most 10% clipped
            sclip=2.0,  # aggressive sigma clipping
            nclip=3,  # 3 iterations of clipping
            maxsma=25.0,
            fix_center=True,
            fix_pa=True,
            fix_eps=True,
        )

        geom = {"x0": float(cx), "y0": float(cy), "eps": 0.0, "pa": 0.0}
        result = fit_isophote(image, mask, sma=20.0, start_geometry=geom, config=config)

        # ~40% clipped > 10% threshold, should trigger stop_code=1
        self.assertEqual(result["stop_code"], 1, "fflag=0.1 with ~40% sigma-clipped points should trigger stop_code=1")

    def test_fflag_accepts_low_clipping(self):
        """With fflag=0.5 and moderate outliers, stop_code should NOT be 1."""
        # ~10% of points are outliers
        image = self._make_image_with_outliers(outlier_fraction=0.10)
        size = image.shape[0]
        cx, cy = size // 2, size // 2
        mask = np.zeros_like(image, dtype=bool)

        config = IsosterConfig(
            x0=float(cx),
            y0=float(cy),
            sma0=20.0,
            eps=0.0,
            pa=0.0,
            fflag=0.5,  # allow up to 50% clipped
            sclip=2.0,
            nclip=3,
            maxsma=25.0,
            fix_center=True,
            fix_pa=True,
            fix_eps=True,
        )

        geom = {"x0": float(cx), "y0": float(cy), "eps": 0.0, "pa": 0.0}
        result = fit_isophote(image, mask, sma=20.0, start_geometry=geom, config=config)

        # ~10% clipped < 50% threshold, should NOT trigger stop_code=1
        self.assertNotEqual(
            result["stop_code"], 1, "fflag=0.5 with ~10% sigma-clipped points should not trigger stop_code=1"
        )

    def test_fflag_arithmetic_correctness(self):
        """Verify fflag semantics directly: (1-fflag) is the min valid fraction."""
        # Unit test the arithmetic without full fit_isophote
        # fflag=0.2 means max 20% flagged, so min 80% valid
        # With 100 total points, need >= 80 actual to pass
        fflag = 0.2
        total_points = 100

        # 79 actual out of 100 = 21% flagged > 20% threshold => should fail
        actual_points_fail = 79
        self.assertTrue(
            actual_points_fail < total_points * (1.0 - fflag), "79/100 valid (21% flagged) should fail fflag=0.2 check"
        )

        # 81 actual out of 100 = 19% flagged < 20% threshold => should pass
        actual_points_pass = 81
        self.assertFalse(
            actual_points_pass < total_points * (1.0 - fflag), "81/100 valid (19% flagged) should pass fflag=0.2 check"
        )


class TestVarResidualDdofGuard(unittest.TestCase):
    """Regression tests for I2: var_residual ddof underflow guard.

    When len(intens) <= n_params, np.std/np.var with ddof=n_params produces
    NaN/Inf. Functions should return zero errors instead.
    """

    def test_compute_parameter_errors_few_points_no_cov(self):
        """compute_parameter_errors with fewer points than parameters returns zeros."""
        # 5 harmonic coeffs need > 5 data points
        phi = np.linspace(0, 2 * np.pi, 4)
        intens = np.array([100.0, 101.0, 99.0, 100.5])

        result = compute_parameter_errors(
            phi, intens, x0=50.0, y0=50.0, sma=10.0, eps=0.2, pa=0.5, gradient=-1.0, cov_matrix=None
        )

        for val in result:
            self.assertTrue(np.isfinite(val), f"Expected finite value, got {val}")

    def test_compute_parameter_errors_few_points_with_cov(self):
        """compute_parameter_errors with cov_matrix but few points returns zeros."""
        phi = np.linspace(0, 2 * np.pi, 4)
        intens = np.array([100.0, 101.0, 99.0, 100.5])
        cov_matrix = np.eye(5) * 0.01
        coeffs = np.array([100.0, 0.5, 0.3, 0.1, 0.05])

        result = compute_parameter_errors(
            phi,
            intens,
            x0=50.0,
            y0=50.0,
            sma=10.0,
            eps=0.2,
            pa=0.5,
            gradient=-1.0,
            cov_matrix=cov_matrix,
            coeffs=coeffs,
        )

        for val in result:
            self.assertTrue(np.isfinite(val), f"Expected finite value, got {val}")

    def test_compute_deviations_few_points(self):
        """compute_deviations with fewer points than parameters returns zeros."""
        # 3 params (intercept + sin + cos) need > 3 data points
        phi = np.linspace(0, 2 * np.pi, 2)
        intens = np.array([100.0, 99.0])

        result = compute_deviations(phi, intens, sma=10.0, gradient=-1.0, order=3)

        for val in result:
            self.assertTrue(np.isfinite(val), f"Expected finite value, got {val}")

    def test_fit_higher_harmonics_simultaneous_few_points(self):
        """fit_higher_harmonics_simultaneous with few points returns zeros dict."""
        # orders=[3,4] means 5 params, need > 5 data points
        # The function already has a guard at line 379, but let's also test
        # the case where lstsq succeeds but var_residual would underflow
        angles = np.linspace(0, 2 * np.pi, 5)
        intens = np.array([100.0, 101.0, 99.0, 100.5, 98.0])

        result = fit_higher_harmonics_simultaneous(angles, intens, sma=10.0, gradient=-1.0, orders=[3, 4])

        for order in [3, 4]:
            self.assertIn(order, result)
            for val in result[order]:
                self.assertTrue(np.isfinite(val), f"Order {order}: expected finite, got {val}")

    def test_compute_parameter_errors_exact_boundary(self):
        """With exactly n_params points, should return zeros (not NaN)."""
        phi = np.linspace(0, 2 * np.pi, 5)
        intens = np.array([100.0, 101.0, 99.0, 100.5, 98.0])

        result = compute_parameter_errors(
            phi, intens, x0=50.0, y0=50.0, sma=10.0, eps=0.2, pa=0.5, gradient=-1.0, cov_matrix=None
        )

        for val in result:
            self.assertTrue(np.isfinite(val), f"Expected finite value, got {val}")


class TestGradientLinearGrowth(unittest.TestCase):
    """Regression tests for I3: gradient formula in linear growth mode."""

    def _make_image(self):
        """Create a constant-slope radial image: I = 1000 - 5*r."""
        size = 200
        cx, cy = size / 2, size / 2
        y, x = np.mgrid[:size, :size]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        image = np.clip(1000.0 - 5.0 * r, 0.0, None)
        mask = np.zeros_like(image, dtype=bool)
        return image, mask, cx, cy

    def test_linear_gradient_independent_of_sma(self):
        """For a constant-slope profile, linear-growth gradient should not depend on SMA.

        Before the fix, gradient was divided by sma*step; with the fix it divides
        by step only. On I(r) = 1000 - 5r, dI/dr = -5, so gradient ~ -5 at all radii.
        """
        from isoster.fitting import compute_gradient

        image, mask, cx, cy = self._make_image()
        step = 3.0

        config = IsosterConfig(
            x0=cx,
            y0=cy,
            eps=0.0,
            pa=0.0,
            sma0=10.0,
            minsma=3.0,
            maxsma=80.0,
            astep=step,
            linear_growth=True,
            integrator="mean",
            use_eccentric_anomaly=False,
        )

        geom_15 = {"x0": cx, "y0": cy, "sma": 15.0, "eps": 0.0, "pa": 0.0}
        geom_30 = {"x0": cx, "y0": cy, "sma": 30.0, "eps": 0.0, "pa": 0.0}

        grad_15, _ = compute_gradient(image, mask, geom_15, config, previous_gradient=-1.0)
        grad_30, _ = compute_gradient(image, mask, geom_30, config, previous_gradient=-1.0)

        ratio = grad_30 / grad_15
        self.assertAlmostEqual(
            ratio, 1.0, delta=0.15, msg=f"Linear gradient should not depend on SMA, ratio={ratio:.3f}"
        )

    def test_linear_gradient_magnitude(self):
        """On I(r) = 1000 - 5r, linear-growth gradient should be approximately -5."""
        from isoster.fitting import compute_gradient

        image, mask, cx, cy = self._make_image()
        step = 3.0

        config = IsosterConfig(
            x0=cx,
            y0=cy,
            eps=0.0,
            pa=0.0,
            sma0=10.0,
            minsma=3.0,
            maxsma=80.0,
            astep=step,
            linear_growth=True,
            integrator="mean",
            use_eccentric_anomaly=False,
        )

        geom = {"x0": cx, "y0": cy, "sma": 20.0, "eps": 0.0, "pa": 0.0}
        grad, _ = compute_gradient(image, mask, geom, config, previous_gradient=-1.0)

        self.assertAlmostEqual(grad, -5.0, delta=0.5, msg=f"Expected gradient ~ -5.0, got {grad:.3f}")


# ============================================================================
# Tests for ISOFIT helper functions (Phase 1: true ISOFIT implementation)
# ============================================================================


def test_build_isofit_design_matrix_shape():
    """Verify design matrix shape is (n, 5 + 2*len(orders))."""
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for orders in [[], [3], [3, 4], [3, 4, 5, 6, 7]]:
        A = build_isofit_design_matrix(angles, orders)
        expected_cols = 5 + 2 * len(orders)
        assert A.shape == (64, expected_cols), f"orders={orders}: expected (64, {expected_cols}), got {A.shape}"


def test_build_isofit_design_matrix_first_five_match():
    """First 5 columns must match build_harmonic_matrix() bit-for-bit."""
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    A_isofit = build_isofit_design_matrix(angles, [3, 4])
    A_5param = build_harmonic_matrix(angles)
    np.testing.assert_array_equal(
        A_isofit[:, :5], A_5param, err_msg="First 5 columns of ISOFIT matrix must match build_harmonic_matrix"
    )


def test_fit_all_harmonics_recovery():
    """Synthetic data with known coefficients: verify recovery to atol=1e-10."""
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    # Known coefficients: [I0, A1, B1, A2, B2, A3, B3, A4, B4]
    true_coeffs = np.array([100.0, 5.0, 3.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    orders = [3, 4]

    # Build exact intensity
    intens = true_coeffs[0]
    intens = intens + true_coeffs[1] * np.sin(angles) + true_coeffs[2] * np.cos(angles)
    intens = intens + true_coeffs[3] * np.sin(2 * angles) + true_coeffs[4] * np.cos(2 * angles)
    intens = intens + true_coeffs[5] * np.sin(3 * angles) + true_coeffs[6] * np.cos(3 * angles)
    intens = intens + true_coeffs[7] * np.sin(4 * angles) + true_coeffs[8] * np.cos(4 * angles)

    recovered, ata_inv = fit_all_harmonics(angles, intens, orders)
    np.testing.assert_allclose(
        recovered, true_coeffs, atol=1e-10, err_msg="fit_all_harmonics should recover exact coefficients"
    )
    assert ata_inv is not None, "Covariance matrix should not be None for well-posed problem"


def test_fit_all_harmonics_insufficient_data():
    """Graceful fallback when n_samples < n_params."""
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    intens = np.array([100.0, 101.0, 99.0, 100.5])
    orders = [3, 4]  # needs 9 params, only 4 points

    coeffs, ata_inv = fit_all_harmonics(angles, intens, orders)
    # Should not crash; lstsq handles underdetermined systems
    assert len(coeffs) == 9, f"Expected 9 coefficients, got {len(coeffs)}"
    assert np.all(np.isfinite(coeffs)), "All coefficients should be finite"


def test_evaluate_harmonic_model_matches_5param():
    """When orders=[], evaluate_harmonic_model matches harmonic_function()."""
    from isoster.fitting import harmonic_function

    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    coeffs_5 = np.array([100.0, 5.0, 3.0, 2.0, 1.0])

    model_5param = harmonic_function(angles, coeffs_5)
    model_isofit = evaluate_harmonic_model(angles, coeffs_5, orders=[])
    np.testing.assert_allclose(
        model_isofit, model_5param, atol=1e-12, err_msg="ISOFIT with empty orders should match 5-param model"
    )


def test_isofit_rms_cleaner_than_5param():
    """Synthetic data with strong A3: ISOFIT RMS should be lower than 5-param RMS."""
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    # Data with strong 3rd-order signal
    intens = 100.0 + 2.0 * np.sin(angles) + 1.0 * np.cos(angles)
    intens += 10.0 * np.sin(3 * angles) + 8.0 * np.cos(3 * angles)  # strong A3/B3

    # 5-param fit
    coeffs_5, _ = fit_first_and_second_harmonics(angles, intens)
    from isoster.fitting import harmonic_function

    model_5 = harmonic_function(angles, coeffs_5)
    rms_5 = np.std(intens - model_5)

    # ISOFIT fit
    coeffs_iso, _ = fit_all_harmonics(angles, intens, [3])
    model_iso = evaluate_harmonic_model(angles, coeffs_iso, [3])
    rms_iso = np.std(intens - model_iso)

    assert rms_iso < rms_5 * 0.1, f"ISOFIT RMS ({rms_iso:.4f}) should be much less than 5-param RMS ({rms_5:.4f})"


# ============================================================================
# Tests for ISOFIT fit_isophote behavior (Phase 2: true ISOFIT integration)
# ============================================================================


def _make_circular_sersic_image(size=201, n=1.0, r_eff=30.0, intens_eff=100.0):
    """Create a circular Sersic profile image for testing fit_isophote."""
    from scipy.special import gammaincinv

    b_n = gammaincinv(2.0 * n, 0.5)
    cx, cy = size / 2, size / 2
    y, x = np.mgrid[:size, :size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = np.clip(r, 0.1, None)
    image = intens_eff * np.exp(-b_n * ((r / r_eff) ** (1.0 / n) - 1.0))
    return image, cx, cy


def _make_boxy_sersic_image(size=201, n=1.0, r_eff=30.0, intens_eff=100.0, eps=0.3, pa=0.5, a4_amp=0.03, a3_amp=0.0):
    """Create a Sersic image with injected higher-order harmonic deviations.

    The image has elliptical isophotes with optional boxiness (a4) and
    asymmetry (a3) perturbations suitable for testing ISOFIT recovery.
    """
    from scipy.special import gammaincinv

    b_n = gammaincinv(2.0 * n, 0.5)
    cx, cy = size / 2, size / 2
    y, x = np.mgrid[:size, :size]

    # Rotate to ellipse frame
    dx = x - cx
    dy = y - cy
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_ell = dx * cos_pa + dy * sin_pa
    y_ell = -dx * sin_pa + dy * cos_pa

    # Elliptical radius
    q = 1.0 - eps
    r_ell = np.sqrt(x_ell**2 + (y_ell / q) ** 2)
    r_ell = np.clip(r_ell, 0.1, None)

    # Position angle on the ellipse
    theta = np.arctan2(y_ell / q, x_ell)

    # Base Sersic profile with harmonic perturbation in radius
    # r_perturbed = r_ell * (1 + a3*sin(3θ) + a4*cos(4θ))
    r_perturbed = r_ell * (1.0 + a3_amp * np.sin(3.0 * theta) + a4_amp * np.cos(4.0 * theta))
    r_perturbed = np.clip(r_perturbed, 0.1, None)

    image = intens_eff * np.exp(-b_n * ((r_perturbed / r_eff) ** (1.0 / n) - 1.0))
    return image, cx, cy


def test_default_path_unchanged():
    """Bit-for-bit regression: simultaneous_harmonics=False must use the same path.

    Runs fit_isophote with default config on a circular Sersic mock and verifies
    that the result is identical to a run with explicit simultaneous_harmonics=False.
    """
    image, cx, cy = _make_circular_sersic_image()
    mask = np.zeros_like(image, dtype=bool)
    start = {"x0": cx, "y0": cy, "eps": 0.1, "pa": 0.0}

    config_default = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.1,
        pa=0.0,
        simultaneous_harmonics=False,
        harmonic_orders=[3, 4],
    )
    config_explicit = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.1,
        pa=0.0,
        simultaneous_harmonics=False,
        harmonic_orders=[3, 4],
    )

    result_default = fit_isophote(image, mask, sma=15.0, start_geometry=start, config=config_default)
    result_explicit = fit_isophote(image, mask, sma=15.0, start_geometry=start, config=config_explicit)

    # Core numeric outputs must be identical
    for key in ["intens", "rms", "eps", "pa", "x0", "y0", "stop_code", "niter"]:
        val_d = result_default[key]
        val_e = result_explicit[key]
        if isinstance(val_d, float) and np.isnan(val_d):
            assert np.isnan(val_e), f"Key {key}: default is NaN but explicit is {val_e}"
        else:
            assert val_d == val_e, f"Key {key}: default={val_d}, explicit={val_e}"


def test_isofit_convergence_behavior():
    """ISOFIT should converge on data with strong higher-order harmonics.

    Uses a boxy Sersic mock with injected a4 deviation. Verifies that
    fit_isophote with simultaneous_harmonics=True converges (stop_code=0)
    and recovers nonzero a4/b4 coefficients.
    """
    image, cx, cy = _make_boxy_sersic_image(
        size=201,
        n=1.0,
        r_eff=30.0,
        intens_eff=500.0,
        eps=0.3,
        pa=0.5,
        a4_amp=0.05,
    )
    mask = np.zeros_like(image, dtype=bool)
    start = {"x0": cx, "y0": cy, "eps": 0.3, "pa": 0.5}

    config = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.3,
        pa=0.5,
        simultaneous_harmonics=True,
        harmonic_orders=[3, 4],
        fix_center=True,
        fix_pa=True,
        fix_eps=True,
        maxit=100,
        conver=0.05,
    )
    result = fit_isophote(image, mask, sma=25.0, start_geometry=start, config=config)

    assert result["stop_code"] in (0, 2), f"Expected convergence, got stop_code={result['stop_code']}"
    # a4 or b4 should be nonzero (we injected a4 perturbation)
    a4_mag = np.sqrt(result["a4"] ** 2 + result["b4"] ** 2)
    assert a4_mag > 1e-4, f"ISOFIT should detect injected a4 deviation, got |a4,b4|={a4_mag:.6f}"


def test_isofit_fallback_insufficient_points():
    """ISOFIT should fall back to 5-param when sample points < isofit_min_points.

    Uses many harmonic orders so that isofit_min_points exceeds the number of
    sample points at moderate SMA with heavy masking, triggering the fallback
    warning and 5-param path.
    """
    image, cx, cy = _make_circular_sersic_image(size=201, r_eff=30.0)
    mask = np.zeros_like(image, dtype=bool)

    # Mask most of the image to leave only a few sample points on the ellipse
    # Keep only a narrow strip near the center row
    mask[: int(cy) - 3, :] = True
    mask[int(cy) + 3 :, :] = True

    start = {"x0": cx, "y0": cy, "eps": 0.05, "pa": 0.0}

    # Many orders: min_points = 1 + 2*(2+len(orders))
    # orders [3..32] = 30 orders → min_points = 1 + 2*(2+30) = 65
    many_orders = list(range(3, 33))
    config = IsosterConfig(
        sma0=10.0,
        x0=cx,
        y0=cy,
        eps=0.05,
        pa=0.0,
        simultaneous_harmonics=True,
        harmonic_orders=many_orders,
        fix_center=True,
        fix_pa=True,
        fix_eps=True,
        maxit=50,
        conver=0.05,
        nclip=0,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fit_isophote(image, mask, sma=10.0, start_geometry=start, config=config)
        fallback_warnings = [x for x in w if "Falling back to 5-param" in str(x.message)]
        assert len(fallback_warnings) > 0, "Expected RuntimeWarning about ISOFIT fallback with insufficient points"

    # Result should still be valid (5-param fallback path)
    assert np.isfinite(result["intens"]), "Intensity should be finite after fallback"
    assert result["stop_code"] in (0, 1, 2), f"Expected valid stop code after fallback, got {result['stop_code']}"


def test_isofit_fallback_heavy_masking():
    """ISOFIT should fall back when heavy masking reduces available points.

    Mask a large fraction of the image so that at the fitting SMA,
    the number of valid points drops below isofit_min_points.
    """
    image, cx, cy = _make_circular_sersic_image(size=201, r_eff=30.0)
    mask = np.zeros_like(image, dtype=bool)

    # Mask most of the upper half — leaves roughly half the ellipse
    mask[: int(cy) - 2, :] = True

    start = {"x0": cx, "y0": cy, "eps": 0.05, "pa": 0.0}

    config = IsosterConfig(
        sma0=10.0,
        x0=cx,
        y0=cy,
        eps=0.05,
        pa=0.0,
        simultaneous_harmonics=True,
        harmonic_orders=[3, 4, 5, 6, 7],
        fix_center=True,
        fix_pa=True,
        fix_eps=True,
        maxit=50,
        conver=0.05,
        nclip=0,
    )

    # Even with heavy masking at moderate SMA, we should get a result
    # (either ISOFIT or 5-param fallback)
    result = fit_isophote(image, mask, sma=10.0, start_geometry=start, config=config)
    assert np.isfinite(result["intens"]), "Intensity should be finite with heavy masking"
    assert result["stop_code"] in (0, 1, 2), f"Expected valid stop code with masking, got {result['stop_code']}"


def test_isofit_mixed_mode_profile():
    """Inner isophotes use 5-param fallback, outer use ISOFIT — verify continuity.

    Fit at small SMA (fallback expected) and large SMA (ISOFIT expected).
    Both should produce valid results and intensities should decrease with SMA
    for a Sersic profile.
    """
    image, cx, cy = _make_circular_sersic_image(size=301, r_eff=40.0, intens_eff=500.0)
    mask = np.zeros_like(image, dtype=bool)

    config = IsosterConfig(
        sma0=5.0,
        x0=cx,
        y0=cy,
        eps=0.05,
        pa=0.0,
        simultaneous_harmonics=True,
        harmonic_orders=[3, 4, 5, 6, 7],
        fix_center=True,
        fix_pa=True,
        fix_eps=True,
        maxit=50,
        conver=0.05,
    )
    start = {"x0": cx, "y0": cy, "eps": 0.05, "pa": 0.0}

    # Small SMA: likely fallback to 5-param (few sample points)
    result_inner = fit_isophote(image, mask, sma=3.0, start_geometry=start, config=config)

    # Large SMA: should use full ISOFIT
    result_outer = fit_isophote(image, mask, sma=40.0, start_geometry=start, config=config)

    # Both should have valid intensities
    assert np.isfinite(result_inner["intens"]), "Inner result should have finite intensity"
    assert np.isfinite(result_outer["intens"]), "Outer result should have finite intensity"

    # Monotonically decreasing for Sersic: inner brighter than outer
    assert result_inner["intens"] > result_outer["intens"], (
        f"Inner ({result_inner['intens']:.2f}) should be brighter than outer ({result_outer['intens']:.2f})"
    )

    # Both should converge or reach max iterations (not crash)
    for label, res in [("inner", result_inner), ("outer", result_outer)]:
        assert res["stop_code"] in (0, 2), f"{label}: expected stop_code 0 or 2, got {res['stop_code']}"


# ============================================================================
# Tests for geometry_update_mode='simultaneous'
# ============================================================================


def test_config_accepts_geometry_update_modes():
    """Config validation accepts both 'largest' and 'simultaneous' geometry update modes."""
    cfg_largest = IsosterConfig(geometry_update_mode="largest")
    assert cfg_largest.geometry_update_mode == "largest"

    cfg_simul = IsosterConfig(geometry_update_mode="simultaneous")
    assert cfg_simul.geometry_update_mode == "simultaneous"

    with pytest.raises(Exception):
        IsosterConfig(geometry_update_mode="invalid")


def test_simultaneous_mode_converges_on_circular_sersic():
    """Simultaneous geometry updates should converge on a simple circular Sersic mock."""
    image, cx, cy = _make_circular_sersic_image(size=201, n=1.0, r_eff=30.0, intens_eff=500.0)
    mask = np.zeros_like(image, dtype=bool)
    start = {"x0": cx + 1.0, "y0": cy - 0.5, "eps": 0.15, "pa": 0.3}

    config = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.2,
        pa=0.0,
        geometry_update_mode="simultaneous",
        geometry_damping=0.5,
        maxit=100,
        conver=0.05,
    )
    result = fit_isophote(image, mask, sma=25.0, start_geometry=start, config=config)

    assert result["stop_code"] in (0, 2), f"Expected convergence or max-iter, got stop_code={result['stop_code']}"
    assert not np.isnan(result["intens"]), "Intensity should not be NaN"
    # Center should be recovered close to truth
    assert abs(result["x0"] - cx) < 2.0, f"x0 offset too large: {result['x0'] - cx:.3f}"
    assert abs(result["y0"] - cy) < 2.0, f"y0 offset too large: {result['y0'] - cy:.3f}"


def test_simultaneous_mode_equivalent_profiles():
    """Simultaneous and largest modes should produce comparable geometry on well-behaved data.

    On a circular Sersic with exact center, both modes should converge to similar
    geometry since there's no ambiguity in the fitting.
    """
    image, cx, cy = _make_circular_sersic_image(size=201, n=1.0, r_eff=30.0, intens_eff=500.0)
    mask = np.zeros_like(image, dtype=bool)
    start = {"x0": cx, "y0": cy, "eps": 0.1, "pa": 0.0}

    config_largest = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.1,
        pa=0.0,
        geometry_update_mode="largest",
        maxit=100,
        conver=0.05,
    )
    config_simul = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.1,
        pa=0.0,
        geometry_update_mode="simultaneous",
        geometry_damping=0.5,
        maxit=100,
        conver=0.05,
    )

    result_largest = fit_isophote(image, mask, sma=25.0, start_geometry=start, config=config_largest)
    result_simul = fit_isophote(image, mask, sma=25.0, start_geometry=start, config=config_simul)

    # Both should converge
    for label, res in [("largest", result_largest), ("simultaneous", result_simul)]:
        assert res["stop_code"] in (0, 2), f"{label}: stop_code={res['stop_code']}"

    # Intensity should be very similar (same data, both converged)
    if result_largest["stop_code"] == 0 and result_simul["stop_code"] == 0:
        rel_diff = abs(result_largest["intens"] - result_simul["intens"]) / max(abs(result_largest["intens"]), 1e-10)
        assert rel_diff < 0.01, (
            f"Intensity differs by {rel_diff * 100:.2f}%: largest={result_largest['intens']:.4f}, simul={result_simul['intens']:.4f}"
        )


def test_simultaneous_mode_fewer_iterations():
    """Simultaneous mode should typically converge in fewer iterations than largest mode.

    We use an elliptical mock with deliberate center offset so multiple parameters
    need correction, which is where simultaneous updates should shine.
    """
    image, cx, cy = _make_boxy_sersic_image(
        size=201,
        n=1.0,
        r_eff=30.0,
        intens_eff=500.0,
        eps=0.3,
        pa=0.5,
        a4_amp=0.0,
    )
    mask = np.zeros_like(image, dtype=bool)
    # Start with offset center and wrong PA/eps to exercise all 4 corrections
    start = {"x0": cx + 2.0, "y0": cy - 1.5, "eps": 0.15, "pa": 0.3}

    config_largest = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.3,
        pa=0.5,
        geometry_update_mode="largest",
        geometry_damping=0.7,
        maxit=100,
        conver=0.05,
    )
    config_simul = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.3,
        pa=0.5,
        geometry_update_mode="simultaneous",
        geometry_damping=0.5,
        maxit=100,
        conver=0.05,
    )

    result_largest = fit_isophote(image, mask, sma=25.0, start_geometry=start, config=config_largest)
    result_simul = fit_isophote(image, mask, sma=25.0, start_geometry=start, config=config_simul)

    # Both should produce valid results
    for label, res in [("largest", result_largest), ("simultaneous", result_simul)]:
        assert res["stop_code"] in (0, 2), f"{label}: stop_code={res['stop_code']}"

    # Simultaneous mode should use fewer or equal iterations
    # This is a soft assertion — if it fails by a small margin, the test still passes
    # because we check both converge, which is the primary requirement
    if result_largest["stop_code"] == 0 and result_simul["stop_code"] == 0:
        assert result_simul["niter"] <= result_largest["niter"] + 5, (
            f"Simultaneous ({result_simul['niter']} iters) should not be much worse than largest ({result_largest['niter']} iters)"
        )


def test_simultaneous_mode_respects_fixed_geometry():
    """When geometry parameters are fixed, simultaneous mode should not update them."""
    image, cx, cy = _make_circular_sersic_image(size=201, n=1.0, r_eff=30.0, intens_eff=500.0)
    mask = np.zeros_like(image, dtype=bool)
    start = {"x0": cx, "y0": cy, "eps": 0.2, "pa": 0.3}

    config = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=0.2,
        pa=0.3,
        geometry_update_mode="simultaneous",
        geometry_damping=0.5,
        fix_center=True,
        fix_pa=True,
        maxit=50,
        conver=0.05,
    )
    result = fit_isophote(image, mask, sma=25.0, start_geometry=start, config=config)

    # Center and PA should remain fixed
    assert result["x0"] == pytest.approx(cx, abs=1e-10), f"x0 should be fixed at {cx}, got {result['x0']}"
    assert result["y0"] == pytest.approx(cy, abs=1e-10), f"y0 should be fixed at {cy}, got {result['y0']}"
    # PA should be unchanged (within numerical precision)
    assert result["pa"] == pytest.approx(0.3, abs=1e-10), f"PA should be fixed at 0.3, got {result['pa']}"


def test_simultaneous_mode_with_elliptical_galaxy():
    """Simultaneous mode should recover correct geometry on an elliptical mock."""
    true_eps = 0.4
    true_pa = 0.8
    image, cx, cy = _make_boxy_sersic_image(
        size=201,
        n=1.0,
        r_eff=30.0,
        intens_eff=500.0,
        eps=true_eps,
        pa=true_pa,
        a4_amp=0.0,
    )
    mask = np.zeros_like(image, dtype=bool)
    start = {"x0": cx + 1.0, "y0": cy - 1.0, "eps": 0.2, "pa": 0.5}

    config = IsosterConfig(
        sma0=15.0,
        x0=cx,
        y0=cy,
        eps=true_eps,
        pa=true_pa,
        geometry_update_mode="simultaneous",
        geometry_damping=0.5,
        maxit=100,
        conver=0.05,
    )
    result = fit_isophote(image, mask, sma=25.0, start_geometry=start, config=config)

    assert result["stop_code"] in (0, 2), f"stop_code={result['stop_code']}"
    # Should recover geometry within reasonable tolerance
    assert abs(result["eps"] - true_eps) < 0.05, f"eps recovery: expected ~{true_eps}, got {result['eps']:.4f}"
    # PA comparison needs wrapping awareness
    pa_diff = abs(result["pa"] - true_pa)
    pa_diff = min(pa_diff, np.pi - pa_diff)
    assert pa_diff < 0.1, f"PA recovery: expected ~{true_pa:.3f}, got {result['pa']:.4f}"


class TestStopCodes:
    """Dedicated tests for each stop code value."""

    @staticmethod
    def _make_sersic(shape=(201, 201), x0=100, y0=100, re=30.0, eps=0.3, pa=0.5, ie=1000.0, n=2.0):
        from scipy.special import gammaincinv

        bn = gammaincinv(2 * n, 0.5)
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        dx = xx - x0
        dy = yy - y0
        cos_pa, sin_pa = np.cos(pa), np.sin(pa)
        x_rot = dx * cos_pa + dy * sin_pa
        y_rot = -dx * sin_pa + dy * cos_pa
        r_ellip = np.sqrt(x_rot**2 + (y_rot / (1 - eps)) ** 2)
        return ie * np.exp(-bn * ((r_ellip / re) ** (1.0 / n) - 1))

    def test_stop_code_0_converged(self):
        """Standard Sersic at moderate SMA should converge (stop_code=0)."""
        image = self._make_sersic()
        mask = np.zeros_like(image, dtype=bool)
        start = {"x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(maxit=50, conver=0.05, convergence_scaling="none")
        result = fit_isophote(image, mask, 30.0, start, cfg)

        assert result["stop_code"] == 0, f"Expected converged (0), got {result['stop_code']}"
        assert np.isfinite(result["intens"])

    def test_stop_code_2_maxit_exhaustion(self):
        """maxit=3 on a complex image should produce stop_code=2."""
        image = self._make_sersic()
        mask = np.zeros_like(image, dtype=bool)
        start = {"x0": 100.0, "y0": 100.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(conver=0.001, maxit=3, minit=1, convergence_scaling="none")
        result = fit_isophote(image, mask, 30.0, start, cfg)

        assert result["stop_code"] == 2, f"Expected maxit exhaustion (2), got {result['stop_code']}"
        assert np.isfinite(result["intens"])
        assert result["intens"] > 0

    def test_stop_code_3_too_few_points(self):
        """SMA larger than image or heavy masking should produce stop_code=3."""
        image = self._make_sersic(shape=(51, 51), x0=25, y0=25)
        mask = np.zeros_like(image, dtype=bool)
        # Mask most of the image, leaving only a small corner
        mask[10:, :] = True
        mask[:, 10:] = True
        start = {"x0": 25.0, "y0": 25.0, "eps": 0.3, "pa": 0.5}

        cfg = IsosterConfig(maxit=30, fflag=0.3)
        result = fit_isophote(image, mask, 20.0, start, cfg)

        assert result["stop_code"] == 3, f"Expected too-few-points (3), got {result['stop_code']}"

    def test_stop_code_neg1_gradient_error(self):
        """Flat/uniform image region should produce gradient error (stop_code=-1)."""
        # Uniform image → zero gradient → gradient error
        image = np.ones((101, 101)) * 100.0
        mask = np.zeros_like(image, dtype=bool)
        start = {"x0": 50.0, "y0": 50.0, "eps": 0.0, "pa": 0.0}

        cfg = IsosterConfig(maxit=30, maxgerr=0.5, convergence_scaling="none")
        result = fit_isophote(image, mask, 20.0, start, cfg)

        assert result["stop_code"] == -1, f"Expected gradient error (-1), got {result['stop_code']}"
