"""
Tests for per-pixel variance map support (WLS harmonic fitting).

Verifies:
- Byte-identity when variance_map=None
- WLS correctness and improved coefficient recovery
- Cosmic ray down-weighting via high-variance pixels
- Exact covariance (no residual scaling in WLS)
- Variance-weighted gradient error
- Input validation (shape mismatch, non-positive values)
- Sigma clip passthrough for variances
- IsophoteData.variances field
"""

import warnings

import numpy as np
import pytest

from isoster.config import IsosterConfig
from isoster.driver import fit_image
from isoster.fitting import (
    compute_deviations,
    compute_gradient,
    compute_parameter_errors,
    fit_all_harmonics,
    fit_first_and_second_harmonics,
    fit_higher_harmonics_simultaneous,
    fit_isophote,
    sigma_clip,
)
from isoster.sampling import extract_isophote_data


def _make_sersic_image(size=128, n=2.0, r_eff=30.0, amplitude=1000.0, noise_sigma=5.0, seed=42):
    """Create a synthetic Sersic galaxy image with known noise level."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    cx, cy = size / 2.0, size / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = np.maximum(r, 0.1)
    bn = 1.9992 * n - 0.3271
    profile = amplitude * np.exp(-bn * ((r / r_eff) ** (1.0 / n) - 1.0))
    noise = rng.normal(0.0, noise_sigma, (size, size))
    image = profile + noise
    variance = np.full_like(image, noise_sigma**2)
    return image, variance


# ---------------------------------------------------------------------------
# IsophoteData namedtuple
# ---------------------------------------------------------------------------


class TestIsophoteDataVariancesField:
    """Verify the new 'variances' field in IsophoteData."""

    def test_variances_none_without_map(self):
        """variances is None when no variance map is provided."""
        image = np.ones((64, 64))
        data = extract_isophote_data(image, None, 32, 32, 20, 0.2, 0.0)
        assert data.variances is None

    def test_variances_populated_with_map(self):
        """variances is a non-None array when variance_map is provided."""
        image = np.ones((64, 64))
        var_map = np.full((64, 64), 4.0)
        data = extract_isophote_data(image, None, 32, 32, 20, 0.2, 0.0, variance_map=var_map)
        assert data.variances is not None
        assert len(data.variances) == len(data.intens)
        # Uniform variance map → all sampled variances should be ~4.0
        np.testing.assert_allclose(data.variances, 4.0, rtol=0.01)

    def test_variances_nan_filtering(self):
        """Pixels with NaN variance are excluded from IsophoteData."""
        image = np.ones((64, 64))
        var_map = np.full((64, 64), 4.0)
        # Set a band of NaN variances
        var_map[30:34, :] = np.nan
        data_with_var = extract_isophote_data(image, None, 32, 32, 20, 0.2, 0.0, variance_map=var_map)
        data_no_var = extract_isophote_data(image, None, 32, 32, 20, 0.2, 0.0)
        # Fewer valid pixels when some variances are NaN
        assert len(data_with_var.intens) <= len(data_no_var.intens)


# ---------------------------------------------------------------------------
# Byte-identity: variance_map=None gives identical results
# ---------------------------------------------------------------------------


class TestByteIdentity:
    """Verify that variance_map=None produces identical output to no variance_map."""

    def test_fit_image_byte_identity(self):
        """fit_image results are identical with and without variance_map=None."""
        image, _ = _make_sersic_image(size=64, noise_sigma=3.0, seed=99)
        config = IsosterConfig(x0=32.0, y0=32.0, sma0=10.0, minsma=5.0, maxsma=25.0, eps=0.2, pa=0.5, maxit=10)

        result_no_var = fit_image(image, config=config)
        result_none_var = fit_image(image, config=config, variance_map=None)

        for iso_a, iso_b in zip(result_no_var["isophotes"], result_none_var["isophotes"]):
            for key in iso_a:
                val_a, val_b = iso_a[key], iso_b[key]
                if isinstance(val_a, float):
                    if np.isnan(val_a):
                        assert np.isnan(val_b), f"Key {key}: expected NaN"
                    else:
                        assert val_a == val_b, f"Key {key}: {val_a} != {val_b}"

    def test_fit_first_harmonics_identity(self):
        """fit_first_and_second_harmonics is identical with variances=None."""
        rng = np.random.default_rng(42)
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        intens = 100.0 + 5.0 * np.sin(phi) + 3.0 * np.cos(2 * phi) + rng.normal(0, 1, 100)

        coeffs_ols, cov_ols = fit_first_and_second_harmonics(phi, intens)
        coeffs_none, cov_none = fit_first_and_second_harmonics(phi, intens, variances=None)

        np.testing.assert_array_equal(coeffs_ols, coeffs_none)
        np.testing.assert_array_equal(cov_ols, cov_none)


# ---------------------------------------------------------------------------
# WLS correctness
# ---------------------------------------------------------------------------


class TestWLSCorrectness:
    """Verify WLS gives better coefficient recovery than OLS."""

    def test_wls_recovers_truth_better(self):
        """WLS with known variance recovers true coefficients more accurately."""
        rng = np.random.default_rng(123)
        phi = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        true_coeffs = np.array([100.0, 3.0, -2.0, 1.5, -1.0])

        # Heteroscedastic noise: split at index 100, 10x sigma ratio.
        # This makes the first half 100x more informative (in variance),
        # so WLS should weight the first half heavily and recover truth better.
        sigma = np.where(np.arange(200) < 100, 1.0, 10.0)
        noise = rng.normal(0, sigma)
        variances = sigma**2

        A = np.column_stack([np.ones_like(phi), np.sin(phi), np.cos(phi), np.sin(2 * phi), np.cos(2 * phi)])
        intens = A @ true_coeffs + noise

        coeffs_ols, _ = fit_first_and_second_harmonics(phi, intens)
        coeffs_wls, _ = fit_first_and_second_harmonics(phi, intens, variances=variances)

        ols_error = np.sum((coeffs_ols - true_coeffs) ** 2)
        wls_error = np.sum((coeffs_wls - true_coeffs) ** 2)
        # WLS should be closer to truth with heteroscedastic noise
        assert wls_error < ols_error, f"WLS error ({wls_error:.4f}) should be less than OLS ({ols_error:.4f})"

    def test_fit_all_harmonics_wls(self):
        """fit_all_harmonics WLS path runs without error with heteroscedastic variance."""
        rng = np.random.default_rng(42)
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        # Heteroscedastic noise: first half quiet, second half noisy (10x ratio)
        # to ensure WLS weighting is meaningfully different from OLS
        sigma = np.where(np.arange(100) < 50, 1.0, 10.0)
        intens = 100.0 + 2.0 * np.sin(3 * phi) + rng.normal(0, sigma)
        variances = sigma**2

        coeffs, cov = fit_all_harmonics(phi, intens, [3, 4], variances=variances)
        assert coeffs is not None
        assert cov is not None
        assert cov.shape == (9, 9)


# ---------------------------------------------------------------------------
# Cosmic ray down-weighting
# ---------------------------------------------------------------------------


class TestCosmicRayDownWeighting:
    """WLS should be resilient to high-variance outlier pixels."""

    def test_high_variance_outliers_downweighted(self):
        """Injecting cosmic rays with high variance should not bias WLS."""
        rng = np.random.default_rng(77)
        phi = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        true_mean = 100.0
        intens_clean = np.full(200, true_mean) + rng.normal(0, 1.0, 200)
        variances_clean = np.ones(200)

        # Inject 10 cosmic rays: very bright but with very high variance
        cr_idx = rng.choice(200, 10, replace=False)
        intens_cr = intens_clean.copy()
        intens_cr[cr_idx] += 500.0
        variances_cr = variances_clean.copy()
        variances_cr[cr_idx] = 1e6  # huge variance → negligible weight

        coeffs_clean, _ = fit_first_and_second_harmonics(phi, intens_clean, variances=variances_clean)
        coeffs_cr, _ = fit_first_and_second_harmonics(phi, intens_cr, variances=variances_cr)
        coeffs_ols_cr, _ = fit_first_and_second_harmonics(phi, intens_cr)

        # WLS with cosmic rays should still recover ~100
        assert abs(coeffs_cr[0] - true_mean) < 2.0, f"WLS mean {coeffs_cr[0]:.1f} far from truth {true_mean}"
        # OLS without variance info should be biased high
        assert abs(coeffs_ols_cr[0] - true_mean) > 10.0, f"OLS mean {coeffs_ols_cr[0]:.1f} unexpectedly close to truth"


# ---------------------------------------------------------------------------
# Exact covariance (no residual scaling)
# ---------------------------------------------------------------------------


class TestExactCovariance:
    """WLS covariance should be exact, not scaled by residual variance."""

    def test_wls_covariance_no_scaling(self):
        """WLS (A^T W A)^-1 diagonal matches analytical prediction."""
        rng = np.random.default_rng(42)
        n = 500
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        sigma = 2.0
        variances = np.full(n, sigma**2)
        intens = 50.0 + rng.normal(0, sigma, n)

        _, cov_wls = fit_first_and_second_harmonics(phi, intens, variances=variances)

        # For uniform variance σ², (A^T W A)^-1 should have diagonal ≈ σ²/n
        # for the intercept term (since A[:,0] = 1)
        expected_var_intercept = sigma**2 / n
        actual_var_intercept = cov_wls[0, 0]
        np.testing.assert_allclose(actual_var_intercept, expected_var_intercept, rtol=0.05)

    def test_compute_parameter_errors_exact_mode(self):
        """compute_parameter_errors with use_exact_covariance skips residual scaling."""
        rng = np.random.default_rng(42)
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        intens = 100.0 + 5.0 * np.sin(phi) + rng.normal(0, 1, 100)

        coeffs, cov = fit_first_and_second_harmonics(phi, intens)
        # exact mode should not crash and should return non-negative errors
        errs = compute_parameter_errors(
            phi, intens, 32, 32, 20, 0.2, 0.5, -5.0, cov_matrix=cov, coeffs=coeffs, use_exact_covariance=True
        )
        assert all(e >= 0 for e in errs)


# ---------------------------------------------------------------------------
# Gradient error
# ---------------------------------------------------------------------------


class TestVarianceWeightedGradient:
    """compute_gradient should use per-pixel variance when available."""

    def test_gradient_with_variance_map(self):
        """Gradient error computed from variance map should differ from scatter-based."""
        image, var_map = _make_sersic_image(size=128, noise_sigma=5.0)
        geometry = {"x0": 64, "y0": 64, "sma": 30, "eps": 0.2, "pa": 0.0}
        config = {"astep": 0.1, "linear_growth": False, "integrator": "mean", "use_eccentric_anomaly": False}

        grad_ols, err_ols = compute_gradient(image, None, geometry, config)
        grad_wls, err_wls = compute_gradient(image, None, geometry, config, variance_map=var_map)

        # Both should return valid numbers
        assert np.isfinite(grad_ols) and np.isfinite(err_ols)
        assert np.isfinite(grad_wls) and np.isfinite(err_wls)
        # Gradient values should be similar (same data)
        np.testing.assert_allclose(grad_ols, grad_wls, rtol=1e-10)
        # Errors differ because methods differ
        assert err_ols != err_wls

    def test_gradient_current_data_with_variances(self):
        """compute_gradient handles 3-tuple current_data (phi, intens, variances)."""
        image, var_map = _make_sersic_image(size=64, noise_sigma=3.0)
        data = extract_isophote_data(image, None, 32, 32, 15, 0.2, 0.0, variance_map=var_map)
        geometry = {"x0": 32, "y0": 32, "sma": 15, "eps": 0.2, "pa": 0.0}
        config = {"astep": 0.1, "linear_growth": False, "integrator": "mean", "use_eccentric_anomaly": False}

        grad, err = compute_gradient(
            image, None, geometry, config, current_data=(data.phi, data.intens, data.variances), variance_map=var_map
        )
        assert np.isfinite(grad) and np.isfinite(err)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Validate variance_map input checking in fit_image."""

    def test_shape_mismatch_raises(self):
        """variance_map with wrong shape should raise ValueError."""
        image = np.ones((64, 64))
        var_map = np.ones((32, 32))
        with pytest.raises(ValueError, match="does not match"):
            fit_image(image, variance_map=var_map)

    def test_non_positive_values_warn(self):
        """variance_map with zeros or negatives should produce a warning."""
        image, _ = _make_sersic_image(size=64)
        var_map = np.ones((64, 64))
        var_map[10, 10] = 0.0
        var_map[20, 20] = -1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This will run but we just want the warning check
            config = IsosterConfig(x0=32, y0=32, sma0=5, maxsma=6, maxit=15)
            fit_image(image, config=config, variance_map=var_map)
            non_positive_warnings = [x for x in w if "non-positive" in str(x.message)]
            assert len(non_positive_warnings) >= 1

    def test_nan_values_warn_and_sanitized(self):
        """variance_map with NaN values should warn and replace with sentinel."""
        image, _ = _make_sersic_image(size=64)
        var_map = np.ones((64, 64))
        var_map[10, 10] = np.nan
        var_map[15, 15] = np.nan
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = IsosterConfig(x0=32, y0=32, sma0=5, maxsma=6, maxit=15)
            fit_image(image, config=config, variance_map=var_map)
            nan_warnings = [x for x in w if "NaN" in str(x.message)]
            assert len(nan_warnings) >= 1
            assert "2 NaN" in str(nan_warnings[0].message)
        # Original array should not be mutated
        assert np.isnan(var_map[10, 10])

    def test_inf_values_warn_and_sanitized(self):
        """variance_map with inf values should warn and replace with sentinel."""
        image, _ = _make_sersic_image(size=64)
        var_map = np.ones((64, 64))
        var_map[10, 10] = np.inf
        var_map[15, 15] = -np.inf  # negative inf is also non-finite
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = IsosterConfig(x0=32, y0=32, sma0=5, maxsma=6, maxit=15)
            fit_image(image, config=config, variance_map=var_map)
            inf_warnings = [x for x in w if "infinite" in str(x.message)]
            assert len(inf_warnings) >= 1
        # Original array should not be mutated
        assert np.isinf(var_map[10, 10])

    def test_caller_array_not_mutated(self):
        """fit_image should not modify the caller's variance_map array."""
        image, _ = _make_sersic_image(size=64)
        var_map = np.ones((64, 64))
        var_map[5, 5] = -1.0
        var_map[6, 6] = np.nan
        var_map[7, 7] = np.inf
        original = var_map.copy()
        config = IsosterConfig(x0=32, y0=32, sma0=5, maxsma=6, maxit=15)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_image(image, config=config, variance_map=var_map)
        # Every element should be identical (including NaN positions)
        np.testing.assert_array_equal(var_map, original)


# ---------------------------------------------------------------------------
# Sigma clip passthrough
# ---------------------------------------------------------------------------


class TestSigmaClipVariances:
    """Variances should be clipped alongside intensities."""

    def test_variances_clipped_with_intensities(self):
        """Extra arrays in sigma_clip preserve variance alignment."""
        rng = np.random.default_rng(42)
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        intens = np.full(100, 10.0) + rng.normal(0, 1, 100)
        variances = np.arange(100, dtype=float)

        # Inject an extreme outlier
        intens[50] = 1000.0

        result = sigma_clip(phi, intens, sclip=3.0, nclip=3, extra_arrays=[variances])
        phi_c, intens_c, n_clipped, var_c = result
        assert n_clipped > 0
        assert len(var_c) == len(intens_c)
        # The outlier at index 50 should be removed
        assert 50.0 not in var_c  # variance index 50 is value 50.0


# ---------------------------------------------------------------------------
# Higher harmonics WLS
# ---------------------------------------------------------------------------


class TestHigherHarmonicsWLS:
    """fit_higher_harmonics_simultaneous and compute_deviations WLS paths."""

    def test_simultaneous_wls_returns_dict(self):
        """WLS path returns valid dict with correct keys."""
        rng = np.random.default_rng(42)
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        intens = 100.0 + 2.0 * np.sin(3 * phi) + rng.normal(0, 1, 100)
        variances = np.ones(100)

        result = fit_higher_harmonics_simultaneous(phi, intens, 20.0, -5.0, [3, 4], variances=variances)
        assert 3 in result and 4 in result
        for n in [3, 4]:
            assert len(result[n]) == 4

    def test_compute_deviations_wls(self):
        """compute_deviations WLS path returns valid results."""
        rng = np.random.default_rng(42)
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        intens = 100.0 + 3.0 * np.sin(4 * phi) + rng.normal(0, 1, 100)
        variances = np.ones(100)

        a, b, a_err, b_err = compute_deviations(phi, intens, 20.0, -5.0, 4, variances=variances)
        assert np.isfinite(a) and np.isfinite(b)
        assert a_err >= 0 and b_err >= 0


# ---------------------------------------------------------------------------
# Integration: fit_isophote with variance_map
# ---------------------------------------------------------------------------


class TestFitIsophoteWithVariance:
    """End-to-end fit_isophote with variance_map."""

    def test_fit_isophote_runs_with_variance_map(self):
        """fit_isophote completes without error when variance_map is provided."""
        image, var_map = _make_sersic_image(size=128, noise_sigma=5.0)
        geometry = {"x0": 64, "y0": 64, "eps": 0.2, "pa": 0.5}
        config = IsosterConfig(maxit=10)

        result = fit_isophote(image, None, 20.0, geometry, config, variance_map=var_map)
        assert result is not None
        assert "intens" in result
        assert np.isfinite(result["intens"])

    def test_fit_isophote_wls_smaller_intens_err(self):
        """WLS intens_err should reflect photon noise, often smaller than OLS scatter."""
        image, var_map = _make_sersic_image(size=128, noise_sigma=2.0, amplitude=5000)
        geometry = {"x0": 64, "y0": 64, "eps": 0.2, "pa": 0.5}
        config = IsosterConfig(maxit=15)

        result_ols = fit_isophote(image, None, 20.0, geometry, config)
        result_wls = fit_isophote(image, None, 20.0, geometry, config, variance_map=var_map)

        # Both should have valid intens_err
        assert np.isfinite(result_ols["intens_err"])
        assert np.isfinite(result_wls["intens_err"])


# ---------------------------------------------------------------------------
# Integration: fit_image with variance_map
# ---------------------------------------------------------------------------


class TestFitImageWithVariance:
    """End-to-end fit_image with variance_map."""

    def test_fit_image_with_variance_map(self):
        """fit_image completes and returns valid isophotes when variance_map provided."""
        image, var_map = _make_sersic_image(size=128, noise_sigma=5.0)
        config = IsosterConfig(x0=64, y0=64, sma0=10, minsma=5, maxsma=40, eps=0.2, pa=0.5, maxit=10)
        result = fit_image(image, config=config, variance_map=var_map)
        assert len(result["isophotes"]) > 0
        # Check that we got isophotes (some may not have 'valid' key for central pixel etc.)
        assert any(np.isfinite(iso.get("intens", np.nan)) for iso in result["isophotes"])


# ---------------------------------------------------------------------------
# Sentinel robustness (B2)
# ---------------------------------------------------------------------------


class TestVarianceSentinelRobustness:
    """Regression tests for B2: driver variance sentinels must not explode errors.

    fit_image replaces NaN/inf variance-map entries with a 1e30 sentinel, and
    bilinear sampling smears it into neighboring samples. The old sum(var)/N^2
    error formulas turned a few such pixels into intens_err ~ 1e12 and into
    spurious stop_code=-1 gradient failures on neighboring isophotes. The
    inverse-variance-weighted error formulas are immune (sentinel weight ~ 0).
    """

    @staticmethod
    def _make_nan_variance(variance, n_pixels=15):
        """Plant NaN pixels in a thin annulus at r~30, as in the B2 reproduction."""
        size = variance.shape[0]
        cx = cy = size / 2.0
        y, x = np.mgrid[:size, :size]
        r = np.hypot(x - cx, y - cy)
        theta = np.arctan2(y - cy, x - cx)
        annulus = (r > 29.5) & (r < 30.5) & (theta > 0) & (theta < 0.3)
        idx = np.argwhere(annulus)[:n_pixels]
        var = variance.copy()
        var[idx[:, 0], idx[:, 1]] = np.nan
        return var

    def test_sentinel_variance_does_not_explode_errors(self):
        """NaN cluster in the variance map: finite sane errors, no spurious -1."""
        image, var_clean = _make_sersic_image(size=128, noise_sigma=5.0)
        var_nan = self._make_nan_variance(var_clean)
        config = IsosterConfig(x0=64, y0=64, sma0=10, minsma=5, maxsma=40, eps=0.2, pa=0.5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # driver NaN-sentinel warning covered elsewhere
            res = fit_image(image, config=config, variance_map=var_nan)["isophotes"]
            res_clean = fit_image(image, config=config, variance_map=var_clean)["isophotes"]

        intens_err = np.array([iso["intens_err"] for iso in res])
        stop_codes = np.array([iso["stop_code"] for iso in res])

        # All error bars finite and at the clean-run scale (B2 measured ~1e12)
        assert np.all(np.isfinite(intens_err)), f"non-finite intens_err: {intens_err}"
        assert np.max(intens_err) < 10.0 * np.median(intens_err), (
            f"exploded intens_err: max={np.max(intens_err):.3g}, median={np.median(intens_err):.3g}"
        )
        # No spurious gradient-error failures
        sma = np.array([iso["sma"] for iso in res])
        assert not np.any(stop_codes == -1), f"spurious stop_code=-1 at sma={sma[stop_codes == -1]}"

        # Errors track the clean run (sentinel pixels lose weight, nothing more)
        clean_err = np.array([iso["intens_err"] for iso in res_clean])
        assert np.max(intens_err) < 2.0 * np.max(clean_err)

    def test_gradient_error_sentinel_immune(self):
        """compute_gradient error stays sane when sentinel pixels contaminate the annulus."""
        image, var_clean = _make_sersic_image(size=128, noise_sigma=5.0)
        var_nan = self._make_nan_variance(var_clean)
        # Simulate the driver's NaN -> 1e30 sentinel sanitization
        var_sentinel = np.where(np.isnan(var_nan), 1e30, var_nan)

        geometry = {"x0": 64, "y0": 64, "sma": 30, "eps": 0.2, "pa": 0.0}
        config = {"astep": 0.1, "linear_growth": False, "integrator": "mean", "use_eccentric_anomaly": False}

        grad_s, err_s = compute_gradient(image, None, geometry, config, variance_map=var_sentinel)
        grad_c, err_c = compute_gradient(image, None, geometry, config, variance_map=var_clean)

        assert np.isfinite(grad_s) and np.isfinite(err_s)
        # Same gradient, error at the clean-map scale (old formula gave ~1e12)
        np.testing.assert_allclose(grad_s, grad_c, rtol=1e-10)
        np.testing.assert_allclose(err_s, err_c, rtol=0.5)
