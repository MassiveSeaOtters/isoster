"""
Test unified template-based forced photometry API (R26-05).

This tests the ability to apply geometry from a previously fitted isoster result
(the "template") to extract photometry on a new image. This enables multiband
analysis where one band defines geometry and others use the same geometry.
"""

import os
import tempfile

import numpy as np
import pytest

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.driver import _resolve_template
from isoster.utils import isophote_results_from_fits, isophote_results_to_fits


def make_elliptical_image(shape, x0, y0, eps, pa, intensity_scale, r_scale):
    """
    Create elliptical Sersic-like profile for testing.

    Args:
        shape: (height, width) tuple
        x0, y0: center coordinates
        eps: ellipticity
        pa: position angle in radians
        intensity_scale: peak intensity
        r_scale: exponential scale length

    Returns:
        np.ndarray: 2D image with elliptical profile
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    dx = x - x0
    dy = y - y0
    x_rot = dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
    r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps)) ** 2)
    return intensity_scale * np.exp(-r_ell / r_scale)


def build_template_isophotes(x0, y0, eps, pa, sma_values):
    """Create deterministic template isophotes as a list of geometry dicts."""
    return [{"sma": sma, "x0": x0, "y0": y0, "eps": eps, "pa": pa, "intens": 100.0} for sma in sma_values]


class TestTemplateForced:
    """Tests for template-based forced photometry."""

    def test_template_forced_basic(self):
        """Verify template geometry is applied correctly."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image_g = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        image_r = make_elliptical_image(shape, x0, y0, eps, pa, 800.0, 15.0)

        template_sma_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, minsma=5.0, maxsma=40.0, astep=0.2, linear_growth=True
        )
        template_isophotes = build_template_isophotes(x0=x0, y0=y0, eps=eps, pa=pa, sma_values=template_sma_values)

        results_r = fit_image(image_r, None, config, template=template_isophotes)

        assert len(results_r["isophotes"]) == len(template_isophotes)

        for iso_g, iso_r in zip(template_isophotes, results_r["isophotes"]):
            assert iso_r["sma"] == iso_g["sma"], "SMA should match template"
            assert iso_r["x0"] == iso_g["x0"], "x0 should match template"
            assert iso_r["y0"] == iso_g["y0"], "y0 should match template"

    def test_template_preserves_variable_geometry(self):
        """Verify that x0, y0, eps, pa vary per SMA from template."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)

        template_sma_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, minsma=5.0, maxsma=40.0, astep=5.0, linear_growth=True
        )
        template_list = build_template_isophotes(x0=x0, y0=y0, eps=eps, pa=pa, sma_values=template_sma_values)

        # Manually modify template to have varying geometry
        for i, iso in enumerate(template_list):
            iso["x0"] = 50.0 + i * 0.1
            iso["y0"] = 50.0 - i * 0.05
            iso["eps"] = min(0.3 + i * 0.01, 0.95)

        results_forced = fit_image(image, None, config, template=template_list)

        for iso_templ, iso_forced in zip(template_list, results_forced["isophotes"]):
            if iso_templ["sma"] > 0:
                assert iso_forced["x0"] == iso_templ["x0"]
                assert iso_forced["y0"] == iso_templ["y0"]

    def test_template_intensity_differs(self):
        """Verify intensity comes from target image, not template."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image_g = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        image_r = make_elliptical_image(shape, x0, y0, eps, pa, 800.0, 15.0)

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, minsma=5.0, maxsma=40.0, astep=0.3, linear_growth=True
        )

        template_sma_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

        # Use g-band as template for r-band
        results_g = fit_image(image_g, None, config)
        results_r = fit_image(image_r, None, config, template=results_g)

        for iso_g, iso_r in zip(results_g["isophotes"], results_r["isophotes"]):
            if iso_g["stop_code"] == 0 and iso_r["stop_code"] == 0:
                if iso_g["intens"] > 10:
                    ratio = iso_r["intens"] / iso_g["intens"]
                    assert 0.75 < ratio < 0.85, f"Color ratio at SMA={iso_g['sma']:.1f} should be ~0.8, got {ratio:.3f}"

    def test_template_from_fits(self):
        """Verify round-trip: save -> load -> use as template via file path."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image_g = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        image_r = make_elliptical_image(shape, x0, y0, eps, pa, 800.0, 15.0)

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, minsma=5.0, maxsma=40.0, astep=0.3, linear_growth=True
        )

        results_g = fit_image(image_g, None, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = os.path.join(tmpdir, "template.fits")
            isophote_results_to_fits(results_g, fits_path)

            # Use file path directly as template
            results_r = fit_image(image_r, None, config, template=fits_path)

        assert len(results_r["isophotes"]) == len(results_g["isophotes"])

        for iso_orig, iso_r in zip(results_g["isophotes"], results_r["isophotes"]):
            assert abs(iso_r["sma"] - iso_orig["sma"]) < 1e-6, "SMA mismatch after FITS round-trip"

    def test_template_forced_with_mask(self):
        """Verify masked pixels handled correctly in template mode."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)

        mask = np.zeros(shape, dtype=bool)
        mask[40:60, 60:80] = True

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, minsma=5.0, maxsma=40.0, astep=0.3, linear_growth=True
        )

        template = build_template_isophotes(
            x0=x0, y0=y0, eps=eps, pa=pa, sma_values=[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        )

        results_masked = fit_image(image, mask, config, template=template)

        assert len(results_masked["isophotes"]) == len(template)

        for iso_mask in results_masked["isophotes"]:
            assert "sma" in iso_mask
            assert "intens" in iso_mask

    def test_empty_template_raises_error(self):
        """Verify proper error for empty template list."""
        image = np.ones((50, 50)) * 100.0

        with pytest.raises(ValueError, match="template cannot be empty"):
            fit_image(image, None, {}, template=[])

    def test_none_template_uses_normal_fitting(self):
        """Verify that template=None uses normal fitting mode."""
        image = np.ones((50, 50)) * 100.0

        config = IsosterConfig(x0=25.0, y0=25.0, sma0=5.0, maxsma=15.0)
        results = fit_image(image, None, config, template=None)

        assert "isophotes" in results
        assert len(results["isophotes"]) > 0

    def test_template_with_central_pixel(self):
        """Test that central pixel (sma=0) in template is handled correctly."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.2, 0.0

        image_g = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        image_r = make_elliptical_image(shape, x0, y0, eps, pa, 800.0, 15.0)

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, minsma=0.0, maxsma=30.0, astep=0.3, linear_growth=True
        )

        results_g = fit_image(image_g, None, config)
        results_r = fit_image(image_r, None, config, template=results_g)

        sma_g = [iso["sma"] for iso in results_g["isophotes"]]
        sma_r = [iso["sma"] for iso in results_r["isophotes"]]

        assert 0.0 in sma_g, "Template should have central pixel"
        assert 0.0 in sma_r, "Result should have central pixel"

        central_g = [iso for iso in results_g["isophotes"] if iso["sma"] == 0.0][0]
        central_r = [iso for iso in results_r["isophotes"] if iso["sma"] == 0.0][0]

        if not np.isnan(central_g["intens"]) and not np.isnan(central_r["intens"]):
            ratio = central_r["intens"] / central_g["intens"]
            assert 0.7 < ratio < 0.9, f"Central pixel color ratio should be ~0.8, got {ratio:.3f}"

    def test_template_sorts_by_sma(self):
        """Test that template isophotes are sorted by SMA."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.2, 0.0

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)

        config = IsosterConfig(
            x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, minsma=5.0, maxsma=30.0, astep=0.3, linear_growth=True
        )

        template = build_template_isophotes(x0=x0, y0=y0, eps=eps, pa=pa, sma_values=[10.0, 15.0, 20.0, 25.0, 30.0])

        # Shuffle the template order
        shuffled_template = template.copy()
        np.random.shuffle(shuffled_template)

        results_forced = fit_image(image, None, config, template=shuffled_template)

        sma_values = [iso["sma"] for iso in results_forced["isophotes"]]
        assert sma_values == sorted(sma_values), "Output should be sorted by SMA"


class TestTemplateWithVarianceMap:
    """Regression tests for variance_map support in template/forced mode."""

    def test_variance_map_threaded_to_forced_photometry(self):
        """variance_map should affect intens_err in forced mode (not silently ignored)."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        # Add noise so OLS and WLS errors differ meaningfully
        rng = np.random.default_rng(42)
        image += rng.normal(0, 5.0, shape)

        # Heteroscedastic variance: one half noisier than the other
        variance_map = np.full(shape, 25.0)
        variance_map[:, 50:] = 2500.0  # right half 10x noisier

        template = build_template_isophotes(x0=x0, y0=y0, eps=eps, pa=pa, sma_values=[10.0, 15.0, 20.0, 25.0, 30.0])
        config = IsosterConfig(x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, maxsma=30.0)

        result_no_var = fit_image(image, None, config, template=template)
        result_wls = fit_image(image, None, config, template=template, variance_map=variance_map)

        assert len(result_wls["isophotes"]) == len(template)

        # At least one non-central isophote should have different intens_err
        # (WLS propagated error vs OLS scatter-based error)
        any_differ = False
        for iso_ols, iso_wls in zip(result_no_var["isophotes"], result_wls["isophotes"]):
            if iso_ols["sma"] > 0 and np.isfinite(iso_ols["intens_err"]):
                if iso_ols["intens_err"] != iso_wls["intens_err"]:
                    any_differ = True
                    break
        assert any_differ, "variance_map had no effect on intens_err — it may be silently ignored in forced mode"

    def test_forced_wls_uses_weighted_mean(self):
        """Forced photometry with variance_map should use weighted mean intensity."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.0, 0.0

        # Uniform image — intensity differences come only from weighting
        image = np.full(shape, 100.0)
        # Inject a bright strip that would bias an unweighted mean
        image[48:52, :] = 200.0

        # Give the bright strip very high variance (low weight)
        variance_map = np.full(shape, 1.0)
        variance_map[48:52, :] = 1e6

        template = build_template_isophotes(x0=x0, y0=y0, eps=eps, pa=pa, sma_values=[20.0])
        config = IsosterConfig(x0=x0, y0=y0, eps=eps, pa=pa, sma0=20.0, maxsma=20.0)

        result_ols = fit_image(image, None, config, template=template)
        result_wls = fit_image(image, None, config, template=template, variance_map=variance_map)

        intens_ols = result_ols["isophotes"][0]["intens"]
        intens_wls = result_wls["isophotes"][0]["intens"]

        # Unweighted mean should be pulled toward 200; weighted mean should stay near 100
        assert intens_ols > intens_wls, (
            f"Weighted mean ({intens_wls:.2f}) should be lower than unweighted "
            f"({intens_ols:.2f}) because bright pixels have high variance"
        )

    def test_forced_wls_sigma_clip_preserves_variance_alignment(self):
        """Sigma clipping in forced mode should clip variances alongside intensities."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.2, 0.0

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)
        variance_map = np.full(shape, 25.0)

        template = build_template_isophotes(x0=x0, y0=y0, eps=eps, pa=pa, sma_values=[15.0])
        config = IsosterConfig(x0=x0, y0=y0, eps=eps, pa=pa, sma0=15.0, maxsma=15.0, nclip=3, sclip=3.0)

        # Should not raise any shape mismatch or array length errors
        result = fit_image(image, None, config, template=template, variance_map=variance_map)
        assert len(result["isophotes"]) == 1
        assert np.isfinite(result["isophotes"][0]["intens"])
        assert np.isfinite(result["isophotes"][0]["intens_err"])

    def test_forced_wls_byte_identity_without_variance(self):
        """Forced mode without variance_map should be identical to variance_map=None."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        eps, pa = 0.3, np.pi / 4

        image = make_elliptical_image(shape, x0, y0, eps, pa, 1000.0, 15.0)

        template = build_template_isophotes(x0=x0, y0=y0, eps=eps, pa=pa, sma_values=[10.0, 15.0, 20.0, 25.0, 30.0])
        config = IsosterConfig(x0=x0, y0=y0, eps=eps, pa=pa, sma0=10.0, maxsma=30.0)

        result_a = fit_image(image, None, config, template=template)
        result_b = fit_image(image, None, config, template=template, variance_map=None)

        for iso_a, iso_b in zip(result_a["isophotes"], result_b["isophotes"]):
            for key in iso_a:
                val_a, val_b = iso_a[key], iso_b[key]
                if isinstance(val_a, float):
                    if np.isnan(val_a):
                        assert np.isnan(val_b), f"Key {key}: expected NaN"
                    else:
                        assert val_a == val_b, f"Key {key}: {val_a} != {val_b}"


class TestUnifiedTemplateAPI:
    """Tests for the unified template= parameter (R26-05)."""

    def test_template_accepts_file_path(self):
        """template= should accept a file path string."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        image = make_elliptical_image(shape, x0, y0, 0.2, 0.0, 1000.0, 15.0)

        config = IsosterConfig(x0=x0, y0=y0, sma0=10.0, maxsma=30.0)
        results_g = fit_image(image, None, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = os.path.join(tmpdir, "template.fits")
            isophote_results_to_fits(results_g, fits_path)

            results_r = fit_image(image, None, config, template=fits_path)

        assert len(results_r["isophotes"]) == len(results_g["isophotes"])

    def test_template_accepts_results_dict(self):
        """template= should accept a results dict with 'isophotes' key."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        image = make_elliptical_image(shape, x0, y0, 0.2, 0.0, 1000.0, 15.0)

        config = IsosterConfig(x0=x0, y0=y0, sma0=10.0, maxsma=30.0)
        results_g = fit_image(image, None, config)

        # Pass entire results dict
        results_r = fit_image(image, None, config, template=results_g)

        assert len(results_r["isophotes"]) == len(results_g["isophotes"])

    def test_template_accepts_isophote_list(self):
        """template= should accept a plain list of isophote dicts."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        image = make_elliptical_image(shape, x0, y0, 0.2, 0.0, 1000.0, 15.0)

        template = build_template_isophotes(x0=x0, y0=y0, eps=0.2, pa=0.0, sma_values=[5.0, 10.0, 15.0, 20.0])

        config = IsosterConfig(x0=x0, y0=y0, sma0=10.0, maxsma=30.0)
        results = fit_image(image, None, config, template=template)

        assert len(results["isophotes"]) == len(template)

    def test_template_isophotes_emits_future_warning(self):
        """template_isophotes= should emit FutureWarning."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        image = make_elliptical_image(shape, x0, y0, 0.2, 0.0, 1000.0, 15.0)

        template = build_template_isophotes(x0=x0, y0=y0, eps=0.2, pa=0.0, sma_values=[5.0, 10.0, 15.0])

        config = IsosterConfig(x0=x0, y0=y0, sma0=10.0, maxsma=30.0)
        with pytest.warns(FutureWarning, match="template_isophotes is deprecated"):
            fit_image(image, None, config, template_isophotes=template)

    def test_missing_keys_raise_value_error(self):
        """Template dicts missing required keys should raise ValueError."""
        shape = (100, 100)
        image = make_elliptical_image(shape, 50.0, 50.0, 0.2, 0.0, 1000.0, 15.0)

        bad_template = [{"sma": 5.0, "x0": 50.0}]  # Missing y0, eps, pa

        with pytest.raises(ValueError, match="missing required keys"):
            fit_image(image, None, {}, template=bad_template)

    def test_invalid_type_raises_type_error(self):
        """Non-recognized template type should raise TypeError."""
        image = np.ones((50, 50)) * 100.0

        with pytest.raises(TypeError, match="template must be"):
            fit_image(image, None, {}, template=42)

    def test_no_harmonics_when_deviations_disabled(self):
        """Forced photometry should omit harmonic keys when compute_deviations=False."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        image = make_elliptical_image(shape, x0, y0, 0.2, 0.0, 1000.0, 15.0)

        template = build_template_isophotes(x0=x0, y0=y0, eps=0.2, pa=0.0, sma_values=[10.0, 20.0])

        config = IsosterConfig(
            x0=x0,
            y0=y0,
            compute_deviations=False,
            simultaneous_harmonics=False,
        )
        results = fit_image(image, None, config, template=template)

        for iso in results["isophotes"]:
            for key in ["a3", "b3", "a4", "b4", "a3_err", "b3_err", "a4_err", "b4_err"]:
                assert key not in iso, f"unexpected harmonic key '{key}' when compute_deviations=False"

    def test_harmonics_present_when_deviations_enabled(self):
        """Forced photometry should include harmonic keys when compute_deviations=True."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        image = make_elliptical_image(shape, x0, y0, 0.2, 0.0, 1000.0, 15.0)

        template = build_template_isophotes(x0=x0, y0=y0, eps=0.2, pa=0.0, sma_values=[10.0, 20.0])

        config = IsosterConfig(
            x0=x0,
            y0=y0,
            compute_deviations=True,
            harmonic_orders=[3, 4, 5],
        )
        results = fit_image(image, None, config, template=template)

        for iso in results["isophotes"]:
            for order in [3, 4, 5]:
                assert f"a{order}" in iso, f"missing a{order} key"
                assert f"b{order}" in iso, f"missing b{order} key"

    def test_template_and_template_isophotes_raises(self):
        """Specifying both template= and template_isophotes= should raise ValueError."""
        image = np.ones((50, 50)) * 100.0
        template = build_template_isophotes(x0=25.0, y0=25.0, eps=0.2, pa=0.0, sma_values=[5.0, 10.0])

        with pytest.raises(ValueError, match="Cannot specify both"):
            fit_image(image, None, {}, template=template, template_isophotes=template)


class TestTemplateSelfConsistency:
    """Verify that template-forced photometry on the same image reproduces the
    original fitted profile.  In real use the template is applied to a
    *different* band, but applying it to the *same* image is a strong
    self-consistency check: every non-geometry field (intens, rms, etc.)
    should match the original within numerical noise."""

    def _run_self_consistency(self, eps, pa, use_ea=False):
        """Helper: fit an image, use result as template on the same image,
        compare intensity and geometry."""
        shape = (200, 200)
        x0, y0 = 100.0, 100.0

        image = make_elliptical_image(shape, x0, y0, eps, pa, 5000.0, 25.0)

        config = IsosterConfig(
            x0=x0,
            y0=y0,
            eps=eps,
            pa=pa,
            sma0=6.0,
            minsma=0.0,
            maxsma=80.0,
            astep=0.15,
            linear_growth=False,
            use_eccentric_anomaly=use_ea,
        )

        results_orig = fit_image(image, None, config)
        results_forced = fit_image(image, None, config, template=results_orig)

        iso_orig = results_orig["isophotes"]
        iso_forced = results_forced["isophotes"]

        assert len(iso_forced) == len(iso_orig), f"Expected {len(iso_orig)} isophotes, got {len(iso_forced)}"

        for orig, forced in zip(iso_orig, iso_forced):
            sma = orig["sma"]

            # Geometry should be identical (copied from template)
            assert forced["x0"] == orig["x0"], f"x0 mismatch at SMA={sma}"
            assert forced["y0"] == orig["y0"], f"y0 mismatch at SMA={sma}"
            assert forced["eps"] == orig["eps"], f"eps mismatch at SMA={sma}"
            assert forced["pa"] == orig["pa"], f"pa mismatch at SMA={sma}"

            # Central pixel is handled via fit_central_pixel, intensity exact
            if sma == 0.0:
                if np.isfinite(orig["intens"]):
                    assert forced["intens"] == orig["intens"], "Central pixel intensity mismatch"
                continue

            # Skip problematic isophotes
            if orig["stop_code"] not in {0, 1, 2}:
                continue

            # Intensity: forced mode re-samples the same ellipse so the mean
            # should match very closely.  The fitted result uses the harmonic
            # mean (I_0 coefficient), while forced mode uses np.mean over
            # the same sample points — these differ slightly, so allow 2%.
            if np.isfinite(orig["intens"]) and abs(orig["intens"]) > 1.0:
                rel_diff = abs(forced["intens"] - orig["intens"]) / abs(orig["intens"])
                assert rel_diff < 0.02, (
                    f"Intensity mismatch at SMA={sma:.2f}: "
                    f"orig={orig['intens']:.4f}, forced={forced['intens']:.4f}, "
                    f"rel_diff={rel_diff:.4f}"
                )

    def test_self_consistency_circular(self):
        """Round-trip with nearly circular profile (eps=0.05)."""
        self._run_self_consistency(eps=0.05, pa=0.0)

    def test_self_consistency_elliptical(self):
        """Round-trip with moderate ellipticity (eps=0.3, pa=pi/6)."""
        self._run_self_consistency(eps=0.3, pa=np.pi / 6)

    def test_self_consistency_high_eps_ea(self):
        """Round-trip with high ellipticity in eccentric anomaly mode."""
        self._run_self_consistency(eps=0.55, pa=np.pi / 3, use_ea=True)


class TestResolveTemplate:
    """Unit tests for _resolve_template()."""

    def test_resolve_list(self):
        """List of dicts should pass through and be sorted."""
        template = [
            {"sma": 20.0, "x0": 50.0, "y0": 50.0, "eps": 0.2, "pa": 0.0},
            {"sma": 5.0, "x0": 50.0, "y0": 50.0, "eps": 0.2, "pa": 0.0},
            {"sma": 10.0, "x0": 50.0, "y0": 50.0, "eps": 0.2, "pa": 0.0},
        ]
        result = _resolve_template(template)
        assert [iso["sma"] for iso in result] == [5.0, 10.0, 20.0]

    def test_resolve_results_dict(self):
        """Dict with 'isophotes' key should extract the list."""
        iso_list = [
            {"sma": 5.0, "x0": 50.0, "y0": 50.0, "eps": 0.2, "pa": 0.0},
        ]
        result = _resolve_template({"isophotes": iso_list, "config": None})
        assert len(result) == 1
        assert result[0]["sma"] == 5.0

    def test_resolve_dict_without_isophotes_key(self):
        """Dict without 'isophotes' key should raise ValueError."""
        with pytest.raises(ValueError, match="must contain an 'isophotes' key"):
            _resolve_template({"config": None})

    def test_resolve_file_path(self):
        """File path should load via isophote_results_from_fits."""
        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        image = make_elliptical_image(shape, x0, y0, 0.2, 0.0, 1000.0, 15.0)
        config = IsosterConfig(x0=x0, y0=y0, sma0=10.0, maxsma=20.0)
        results = fit_image(image, None, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = os.path.join(tmpdir, "test.fits")
            isophote_results_to_fits(results, fits_path)

            resolved = _resolve_template(fits_path)

        assert len(resolved) == len(results["isophotes"])

    def test_resolve_pathlib_path(self):
        """pathlib.Path should work the same as str."""
        from pathlib import Path

        shape = (100, 100)
        x0, y0 = 50.0, 50.0
        image = make_elliptical_image(shape, x0, y0, 0.2, 0.0, 1000.0, 15.0)
        config = IsosterConfig(x0=x0, y0=y0, sma0=10.0, maxsma=20.0)
        results = fit_image(image, None, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = Path(tmpdir) / "test.fits"
            isophote_results_to_fits(results, str(fits_path))

            resolved = _resolve_template(fits_path)

        assert len(resolved) == len(results["isophotes"])

    def test_resolve_empty_list_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="template cannot be empty"):
            _resolve_template([])

    def test_resolve_bad_type_raises(self):
        """Non-recognized type should raise TypeError."""
        with pytest.raises(TypeError, match="template must be"):
            _resolve_template(42)

    def test_resolve_missing_keys_raises(self):
        """Missing required keys should raise ValueError."""
        bad_list = [{"sma": 5.0, "x0": 50.0}]
        with pytest.raises(ValueError, match="missing required keys"):
            _resolve_template(bad_list)


class TestIsophoteResultsFromFits:
    """Tests for the isophote_results_from_fits utility function."""

    def test_basic_round_trip(self):
        """Test basic save and load functionality."""
        shape = (100, 100)
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        image = 1000.0 * np.exp(-r / 15.0)

        config = IsosterConfig(x0=50.0, y0=50.0, sma0=10.0, maxsma=40.0)

        results = fit_image(image, None, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = os.path.join(tmpdir, "test.fits")
            isophote_results_to_fits(results, fits_path)

            loaded = isophote_results_from_fits(fits_path)

        assert "isophotes" in loaded
        assert "config" in loaded
        # The CONFIG HDU recovers config as IsosterConfig
        assert isinstance(loaded["config"], IsosterConfig)

        assert len(loaded["isophotes"]) == len(results["isophotes"])

        for orig, load in zip(results["isophotes"], loaded["isophotes"]):
            assert abs(load["sma"] - orig["sma"]) < 1e-10
            assert abs(load["intens"] - orig["intens"]) < 1e-10
            assert load["stop_code"] == orig["stop_code"]

    def test_preserves_all_columns(self):
        """Test that all columns are preserved in round-trip."""
        shape = (100, 100)
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        image = 1000.0 * np.exp(-r / 15.0)

        config = IsosterConfig(x0=50.0, y0=50.0, sma0=10.0, maxsma=40.0, compute_errors=True, compute_deviations=True)

        results = fit_image(image, None, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            fits_path = os.path.join(tmpdir, "test.fits")
            isophote_results_to_fits(results, fits_path)
            loaded = isophote_results_from_fits(fits_path)

        orig_iso = results["isophotes"][5]
        load_iso = loaded["isophotes"][5]

        expected_keys = ["sma", "intens", "intens_err", "eps", "pa", "x0", "y0", "rms", "stop_code", "niter"]

        for key in expected_keys:
            assert key in load_iso, f"Missing key: {key}"
            if isinstance(orig_iso[key], float):
                assert abs(load_iso[key] - orig_iso[key]) < 1e-10, f"Value mismatch for {key}"
            else:
                assert load_iso[key] == orig_iso[key], f"Value mismatch for {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
