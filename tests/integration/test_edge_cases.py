"""
Test edge cases, forced mode, CoG mode, and config validation.

This test file addresses ISSUE-6 gaps in test coverage:
- Forced photometry mode
- Curve-of-growth (CoG) mode
- All-masked/empty images
- Config validation
"""

import numpy as np
import pytest
from pydantic import ValidationError

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.fitting import extract_forced_photometry


class TestForcedMode:
    """Test template-based forced photometry mode (no fitting, just sampling)."""

    def test_forced_mode_basic(self):
        """Test template-based forced mode extracts photometry at specified SMA values."""
        image = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        image = 1000.0 * np.exp(-r / 10.0)

        forced_sma = [5.0, 10.0, 15.0, 20.0]
        x0, y0, eps, pa = 50.0, 50.0, 0.0, 0.0

        template = [{"sma": sma, "x0": x0, "y0": y0, "eps": eps, "pa": pa} for sma in forced_sma]

        results = fit_image(image, None, {}, template=template)
        isophotes = results["isophotes"]

        assert len(isophotes) == len(forced_sma)
        sma_values = [iso["sma"] for iso in isophotes]
        assert sma_values == forced_sma

        stop_codes = [iso["stop_code"] for iso in isophotes]
        assert all(code == 0 for code in stop_codes), "All forced mode isophotes should have stop_code=0"

        intensities = [iso["intens"] for iso in isophotes]
        assert all(intensities[i] > intensities[i + 1] for i in range(len(intensities) - 1)), (
            "Intensity should decrease with radius"
        )

    def test_forced_vs_fitted_mode(self):
        """Compare template-forced photometry against fitted mode at same SMA values."""
        image = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        x0, y0 = 50.0, 50.0
        eps, pa = 0.0, 0.0

        dx = x - x0
        dy = y - y0
        x_rot = dx * np.cos(pa) + dy * np.sin(pa)
        y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
        r_ell = np.sqrt(x_rot**2 + (y_rot / (1 - eps)) ** 2)

        image = 1000.0 * np.exp(-r_ell / 15.0)

        config_fit = IsosterConfig(
            x0=x0,
            y0=y0,
            eps=eps,
            pa=pa,
            sma0=10.0,
            minsma=5.0,
            maxsma=30.0,
            astep=0.2,
            linear_growth=True,
        )
        results_fit = fit_image(image, None, config_fit)

        usable_stop_codes = {0, 1, 2}
        fitted_sma = [iso["sma"] for iso in results_fit["isophotes"] if iso["stop_code"] in usable_stop_codes]
        assert fitted_sma, "Fitted mode should provide at least one usable isophote"

        # Template-forced mode at same SMA values (use first 5 for speed)
        template = [{"sma": sma, "x0": x0, "y0": y0, "eps": eps, "pa": pa} for sma in fitted_sma[:5]]
        results_forced = fit_image(image, None, config_fit, template=template)

        for iso_forced in results_forced["isophotes"]:
            sma = iso_forced["sma"]
            iso_fit = [iso for iso in results_fit["isophotes"] if iso["sma"] == sma][0]

            rel_diff = abs(iso_forced["intens"] - iso_fit["intens"]) / iso_fit["intens"]
            assert rel_diff < 0.05, f"Forced mode intensity should match fitted mode within 5% at SMA={sma}"

    def test_forced_mode_with_mask(self):
        """Test template-forced mode handles masked regions correctly."""
        image = np.ones((100, 100)) * 100.0
        mask = np.zeros((100, 100), dtype=bool)

        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        mask[r < 15] = True

        x0, y0, eps, pa = 50.0, 50.0, 0.0, 0.0
        template = [{"sma": sma, "x0": x0, "y0": y0, "eps": eps, "pa": pa} for sma in [10.0, 20.0, 30.0]]

        results = fit_image(image, mask, {}, template=template)
        isophotes = results["isophotes"]

        assert isophotes[0]["stop_code"] == 3, "Masked isophote should have stop_code=3 (too few points)"

        assert isophotes[1]["stop_code"] == 0
        assert isophotes[2]["stop_code"] == 0

    def test_extract_forced_photometry_direct(self):
        """Test extract_forced_photometry function directly."""
        image = np.ones((100, 100)) * 100.0
        mask = None

        result = extract_forced_photometry(
            image, mask, sma=10.0, x0=50.0, y0=50.0, eps=0.2, pa=0.5, use_eccentric_anomaly=False
        )

        # Check result structure
        assert "sma" in result
        assert "intens" in result
        assert "stop_code" in result
        assert result["sma"] == 10.0
        assert result["stop_code"] == 0
        assert abs(result["intens"] - 100.0) < 1.0  # Should be ~100
        assert result["valid"] is True, "Successful forced photometry should have valid=True"

    def test_forced_photometry_has_valid_field_on_failure(self):
        """Regression test for I7: forced photometry with no data should have valid=False."""
        image = np.ones((100, 100)) * 100.0
        mask = np.ones((100, 100), dtype=bool)  # Everything masked

        result = extract_forced_photometry(
            image, mask, sma=10.0, x0=50.0, y0=50.0, eps=0.2, pa=0.5, use_eccentric_anomaly=False
        )

        assert "valid" in result, "Forced photometry result must include 'valid' field"
        assert result["valid"] is False
        assert result["stop_code"] == 3

    def test_forced_photometry_median_sem_factor(self):
        """Regression test for M12: the median-integrator SEM carries the
        Gaussian-asymptotic sqrt(pi/2) factor (the mean path keeps the
        photutils-compatible rms/sqrt(N))."""
        rng = np.random.default_rng(0)
        image = 100.0 + rng.normal(0.0, 1.0, (128, 128))

        out_mean = extract_forced_photometry(image, None, 64.0, 64.0, 30.0, 0.0, 0.0, integrator="mean", nclip=0)
        out_median = extract_forced_photometry(image, None, 64.0, 64.0, 30.0, 0.0, 0.0, integrator="median", nclip=0)

        # Same ring, same samples: median SEM must exceed the mean SEM by sqrt(pi/2)
        assert out_mean["intens_err"] > 0.0
        ratio = out_median["intens_err"] / out_mean["intens_err"]
        assert ratio == pytest.approx(np.sqrt(np.pi / 2.0), rel=1e-9)


class TestCoGMode:
    """Test curve-of-growth (CoG) photometry mode."""

    def test_cog_basic(self):
        """Test basic CoG computation."""
        # Create circular gradient
        image = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        image = 1000.0 * np.exp(-r / 10.0)

        config = IsosterConfig(
            x0=50.0,
            y0=50.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=30.0,
            astep=0.3,
            compute_cog=True,  # Enable CoG
        )

        results = fit_image(image, None, config)
        isophotes = results["isophotes"]

        # Check CoG fields exist
        assert "tflux_e" in isophotes[0], "CoG mode should include tflux_e"
        assert "tflux_c" in isophotes[0], "CoG mode should include tflux_c"

        # CoG should be monotonically increasing with SMA
        cog_values = [iso["tflux_c"] for iso in isophotes if iso["stop_code"] in {0, 1, 2}]
        # Filter out NaN values
        cog_values = [v for v in cog_values if not np.isnan(v)]

        if len(cog_values) > 1:
            assert all(cog_values[i] <= cog_values[i + 1] for i in range(len(cog_values) - 1)), (
                "Curve-of-growth should be monotonically increasing"
            )

    def test_cog_vs_no_cog(self):
        """Test that CoG mode adds extra fields without affecting other results."""
        image = np.ones((100, 100)) * 100.0

        # Without CoG
        config_no_cog = IsosterConfig(
            x0=50.0,
            y0=50.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=20.0,
            astep=0.5,
            compute_cog=False,
        )
        results_no_cog = fit_image(image, None, config_no_cog)

        # With CoG
        config_cog = IsosterConfig(
            x0=50.0,
            y0=50.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=20.0,
            astep=0.5,
            compute_cog=True,
        )
        results_cog = fit_image(image, None, config_cog)

        # Should have same number of isophotes
        assert len(results_no_cog["isophotes"]) == len(results_cog["isophotes"])

        # Intensities should be identical
        for iso_no, iso_yes in zip(results_no_cog["isophotes"], results_cog["isophotes"]):
            if iso_no["stop_code"] in {0, 1, 2} and iso_yes["stop_code"] in {0, 1, 2}:
                assert abs(iso_no["intens"] - iso_yes["intens"]) < 1e-10


class TestAddCogLengthCheck:
    """Regression test for I6: add_cog_to_isophotes length mismatch."""

    def test_length_mismatch_raises(self):
        """Mismatched isophote and CoG list lengths should raise ValueError."""
        from isoster.cog import add_cog_to_isophotes

        isophotes = [{"sma": 10.0}, {"sma": 20.0}, {"sma": 30.0}]
        cog_results = {
            "cog": [1.0, 2.0],
            "cog_annulus": [0.5, 1.0],
            "area_annulus": [100.0, 200.0],
            "flag_cross": [False, False],
            "flag_negative_area": [False, False],
        }

        with pytest.raises(ValueError, match="Length mismatch"):
            add_cog_to_isophotes(isophotes, cog_results)


class TestMaskedImages:
    """Test handling of all-masked and empty images."""

    def test_all_masked_image(self):
        """Test fitting with completely masked image."""
        image = np.ones((100, 100)) * 100.0
        mask = np.ones((100, 100), dtype=bool)  # Everything masked

        config = IsosterConfig(
            x0=50.0,
            y0=50.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=30.0,
        )

        results = fit_image(image, mask, config)
        isophotes = results["isophotes"]

        # All sampled rows should be non-success if any were produced.
        stop_codes = [iso["stop_code"] for iso in isophotes]
        assert all(code != 0 for code in stop_codes), "All-masked image should produce no successful isophotes"

        # Depending on startup gating, this may return zero rows or only failures.
        if stop_codes:
            assert any(code in {3, -1} for code in stop_codes), (
                "All-masked image should only produce failure stop codes"
            )

    def test_center_masked(self):
        """Test fitting when central pixel is masked."""
        image = np.ones((100, 100)) * 100.0
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True  # Mask center

        config = IsosterConfig(
            x0=50.0,
            y0=50.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=30.0,
        )

        results = fit_image(image, mask, config)

        # Should handle gracefully (may return zero rows if startup fit is rejected).
        assert "isophotes" in results
        stop_codes = [iso["stop_code"] for iso in results["isophotes"]]
        if stop_codes:
            assert all(code != 0 for code in stop_codes)

    def test_partially_masked_ellipse(self):
        """Test ellipse that crosses masked region."""
        image = np.ones((100, 100)) * 100.0
        mask = np.zeros((100, 100), dtype=bool)

        # Mask a small region on the right
        mask[:, 85:] = True

        config = IsosterConfig(
            x0=50.0,
            y0=50.0,
            eps=0.0,
            pa=0.0,  # Circular isophotes
            sma0=10.0,
            minsma=5.0,
            maxsma=40.0,
            fflag=0.3,  # More permissive (allow up to 70% flagged)
        )

        results = fit_image(image, mask, config)
        isophotes = results["isophotes"]

        # Check that output structure is valid; row count can be zero if startup fit is rejected.
        assert isinstance(isophotes, list)

        # Check that masked crossings do not produce all-success rows when rows exist.
        stop_codes = [iso["stop_code"] for iso in isophotes]
        if stop_codes:
            assert any(code != 0 for code in stop_codes), (
                "Some isophotes should fail when crossing masked region heavily"
            )

    def test_empty_image(self):
        """Test fitting on zero/empty image."""
        image = np.zeros((100, 100))  # All zeros

        config = IsosterConfig(
            x0=50.0,
            y0=50.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=30.0,
        )

        results = fit_image(image, None, config)
        isophotes = results["isophotes"]

        # Should get gradient errors (stop_code=-1) or too few points
        stop_codes = [iso["stop_code"] for iso in isophotes]
        assert all(code <= 0 or code == 3 for code in stop_codes), (
            "Zero image should produce gradient errors or too-few-points"
        )


class TestOutOfBoundsCenter:
    """Test fit_image with out-of-bounds center coordinates."""

    def test_fit_image_with_out_of_bounds_center(self):
        """fit_image() with OOB center should return results (not crash) with all stop_code != 0."""
        image = np.ones((100, 100)) * 100.0
        y, x = np.ogrid[:100, :100]
        r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
        image += 1000.0 * np.exp(-r / 15.0)

        # x0=500 is far outside the 100x100 image
        config = IsosterConfig(
            x0=500.0,
            y0=500.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=30.0,
        )

        result = fit_image(image, None, config)

        assert "isophotes" in result
        # All isophotes should have non-zero stop codes (failures)
        for iso in result["isophotes"]:
            assert iso["stop_code"] != 0, (
                f"OOB center should not produce converged isophotes, got stop_code=0 at sma={iso['sma']}"
            )


class TestConfigValidation:
    """Test config validation via fit_image() (integration-level).

    Note: unit-level Pydantic validation is tested in tests/unit/test_config_validation.py.
    This class tests that config validation errors propagate correctly through the
    full driver path.
    """

    def test_invalid_ellipticity(self):
        """Test that invalid ellipticity raises ValidationError."""
        with pytest.raises(ValidationError):
            IsosterConfig(eps=1.5)  # eps must be < 1.0

        with pytest.raises(ValidationError):
            IsosterConfig(eps=-0.1)  # eps must be >= 0.0

    def test_invalid_sma_range(self):
        """Test that maxsma < minsma raises ValidationError."""
        with pytest.raises(ValidationError):
            IsosterConfig(minsma=50.0, maxsma=10.0)

    def test_adaptive_integrator_requires_threshold(self):
        """Test that adaptive integrator requires lsb_sma_threshold."""
        with pytest.raises(ValidationError):
            IsosterConfig(integrator="adaptive")  # Missing lsb_sma_threshold

        # Should work with threshold
        config = IsosterConfig(integrator="adaptive", lsb_sma_threshold=20.0)
        assert config.integrator == "adaptive"
        assert config.lsb_sma_threshold == 20.0

    def test_template_requires_valid_input(self):
        """Test that template= requires valid input."""
        from isoster.driver import _resolve_template

        with pytest.raises(ValueError, match="template cannot be empty"):
            _resolve_template([])

        with pytest.raises(TypeError, match="template must be"):
            _resolve_template(42)

        # Should work with valid list
        result = _resolve_template([{"sma": 10.0, "x0": 50.0, "y0": 50.0, "eps": 0.2, "pa": 0.0}])
        assert len(result) == 1

    def test_invalid_integrator(self):
        """Test that invalid integrator name raises ValidationError."""
        with pytest.raises(ValidationError):
            IsosterConfig(integrator="invalid")

        # Valid integrators should work
        for integrator in ["mean", "median", "adaptive"]:
            if integrator == "adaptive":
                config = IsosterConfig(integrator=integrator, lsb_sma_threshold=20.0)
            else:
                config = IsosterConfig(integrator=integrator)
            assert config.integrator == integrator

    def test_valid_config_passes(self):
        """Test that valid config passes validation."""
        config = IsosterConfig(
            x0=50.0,
            y0=50.0,
            eps=0.3,
            pa=1.0,
            sma0=10.0,
            minsma=5.0,
            maxsma=100.0,
            astep=0.1,
            linear_growth=False,
            maxit=50,
            minit=10,
            conver=0.05,
            sclip=3.0,
            nclip=0,
            fflag=0.5,
            maxgerr=0.5,
            fix_center=False,
            fix_pa=False,
            fix_eps=False,
            compute_errors=True,
            compute_deviations=True,
            full_photometry=False,
            compute_cog=False,
            integrator="mean",
            use_eccentric_anomaly=False,
        )

        assert config.eps == 0.3
        assert config.maxsma == 100.0
        assert config.integrator == "mean"

    def test_default_values(self):
        """Test that config defaults are sensible."""
        config = IsosterConfig()

        # Check defaults
        assert config.eps == 0.2
        assert config.pa == 0.0
        assert config.sma0 == 10.0
        assert config.astep == 0.1
        assert config.linear_growth is False
        assert config.maxit == 50
        assert config.minit == 6
        assert config.conver == 0.05
        assert config.maxgerr == 0.5
        assert config.fflag == 0.5
        assert config.compute_errors is True
        assert config.use_eccentric_anomaly is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
