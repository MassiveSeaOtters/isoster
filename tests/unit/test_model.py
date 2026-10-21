import unittest

import numpy as np

from isoster.model import build_isoster_model


class TestModel(unittest.TestCase):
    def test_build_isoster_model_basic(self):
        """Test basic model building with single isophote."""
        shape = (100, 100)
        # Create a single isophote
        iso = {"x0": 50.0, "y0": 50.0, "sma": 10.0, "eps": 0.0, "pa": 0.0, "intens": 100.0}

        model = build_isoster_model(shape, [iso])

        # At sma=10, intensity should be 100
        r = np.sqrt((np.arange(100) - 50.0) ** 2 + (50.0 - 50.0) ** 2)
        idx_at_sma = np.argmin(np.abs(r - 10.0))  # Find pixel closest to sma=10
        self.assertAlmostEqual(model[50, idx_at_sma], 100.0, places=2)

        # Outside sma, should use boundary value (or fill)
        # With single isophote, pixels beyond sma=10 get fill value (0.0)
        self.assertEqual(model[50, 70], 0.0)

    def test_build_isoster_model_interpolation(self):
        """Test radial interpolation between isophotes."""
        shape = (100, 100)
        isos = [
            {"x0": 50.0, "y0": 50.0, "sma": 10.0, "eps": 0.0, "pa": 0.0, "intens": 100.0},
            {"x0": 50.0, "y0": 50.0, "sma": 20.0, "eps": 0.0, "pa": 0.0, "intens": 50.0},
        ]

        model = build_isoster_model(shape, isos)

        # At sma=10, intensity should be ~100
        self.assertAlmostEqual(model[50, 60], 100.0, places=1)  # r=10

        # At sma=20, intensity should be ~50
        self.assertAlmostEqual(model[50, 70], 50.0, places=1)  # r=20

        # At sma=15 (halfway), intensity should be interpolated (~75 for linear)
        val_mid = model[50, 65]  # r=15
        self.assertGreater(val_mid, 50.0)  # Should be > 50
        self.assertLess(val_mid, 100.0)  # Should be < 100
        self.assertAlmostEqual(val_mid, 75.0, delta=5.0)  # ~75 for linear interp

        # Outside sma=20, should be fill value
        self.assertEqual(model[50, 80], 0.0)

    def test_pa_interpolation_wrap_boundary(self):
        """Regression test for I5: PA interpolation across 0/180 boundary.

        When PA wraps from ~170 to ~10 degrees (pi-boundary), linear interpolation
        without unwrapping produces ~90 degrees. With np.unwrap(period=pi), the
        interpolation correctly goes through 180 (=0).
        """
        shape = (100, 100)
        # Three isophotes with PA wrapping: 170 -> 175 -> 5 degrees (in radians)
        isos = [
            {"x0": 50.0, "y0": 50.0, "sma": 10.0, "eps": 0.3, "pa": np.radians(170.0), "intens": 100.0},
            {"x0": 50.0, "y0": 50.0, "sma": 20.0, "eps": 0.3, "pa": np.radians(175.0), "intens": 80.0},
            {"x0": 50.0, "y0": 50.0, "sma": 30.0, "eps": 0.3, "pa": np.radians(5.0), "intens": 60.0},
        ]

        model = build_isoster_model(shape, isos)

        # The model should not have artifacts from PA interpolation errors.
        # Without unwrap, PA at sma~25 would be ~90 degrees (midpoint of 175 and 5)
        # instead of ~180/0 degrees. This causes a twist artifact visible as
        # asymmetric intensity along the minor axis.
        #
        # Verify by checking that the model is roughly symmetric about the expected PA.
        # At sma~25 the PA should be near 180 (=0) degrees, not 90 degrees.
        # A pixel at (50, 75) is on the major axis for PA~0/180; for PA~90 it would
        # be on the minor axis, producing very different intensities.
        val_along_major = model[50, 75]  # r~25, along PA~0 major axis
        val_along_minor = model[75, 50]  # r~25, along PA~90 direction

        # With correct PA interpolation (~0/180 deg), val_along_major should be
        # noticeably higher than val_along_minor because eps=0.3 means the ellipse
        # is elongated along the PA direction.
        # With buggy interpolation (~90 deg), the roles reverse.
        self.assertGreater(val_along_major, val_along_minor, "PA interpolation should go through 180/0, not through 90")
        # The intensity difference should be substantial (>10% of mean) due to eps=0.3
        mean_val = (val_along_major + val_along_minor) / 2.0
        if mean_val > 0:
            self.assertGreater(
                (val_along_major - val_along_minor) / mean_val,
                0.10,
                "Intensity difference along major vs minor axis should be >10% of mean",
            )

    def test_nan_isophotes_filtered_before_interpolation(self):
        """Regression test for I4: NaN intensities should be filtered out.

        Isophotes with stop_code != 0 may carry NaN intensity. Without filtering,
        interp1d propagates NaN across the whole model.
        """
        shape = (100, 100)
        isos = [
            {"x0": 50.0, "y0": 50.0, "sma": 10.0, "eps": 0.0, "pa": 0.0, "intens": 100.0},
            {"x0": 50.0, "y0": 50.0, "sma": 20.0, "eps": 0.0, "pa": 0.0, "intens": np.nan},
            {"x0": 50.0, "y0": 50.0, "sma": 30.0, "eps": 0.0, "pa": 0.0, "intens": 60.0},
        ]

        model = build_isoster_model(shape, isos)

        # The NaN isophote at sma=20 should be excluded; model should not contain NaN
        # in the region covered by the valid isophotes (sma 10-30)
        val_at_10 = model[50, 60]  # r~10
        val_at_30 = model[50, 80]  # r~30
        self.assertTrue(np.isfinite(val_at_10), f"Model at sma=10 should be finite, got {val_at_10}")
        self.assertTrue(np.isfinite(val_at_30), f"Model at sma=30 should be finite, got {val_at_30}")

    def test_nan_geometry_filtered_before_interpolation(self):
        """Isophotes with NaN geometry columns should be excluded."""
        shape = (100, 100)
        isos = [
            {"x0": 50.0, "y0": 50.0, "sma": 10.0, "eps": 0.0, "pa": 0.0, "intens": 100.0},
            {"x0": np.nan, "y0": 50.0, "sma": 20.0, "eps": 0.0, "pa": 0.0, "intens": 80.0},
            {"x0": 50.0, "y0": 50.0, "sma": 30.0, "eps": 0.0, "pa": 0.0, "intens": 60.0},
        ]

        model = build_isoster_model(shape, isos)

        val_at_10 = model[50, 60]
        self.assertTrue(np.isfinite(val_at_10), f"Model at sma=10 should be finite, got {val_at_10}")

    def test_nan_harmonic_coefficients_treated_as_zero(self):
        """NaN harmonic coefficients should be replaced with 0, not poison the model."""
        shape = (100, 100)
        isos = [
            {
                "x0": 50.0,
                "y0": 50.0,
                "sma": 10.0,
                "eps": 0.0,
                "pa": 0.0,
                "intens": 100.0,
                "a3": 0.01,
                "b3": 0.01,
                "a4": 0.01,
                "b4": 0.01,
            },
            {
                "x0": 50.0,
                "y0": 50.0,
                "sma": 20.0,
                "eps": 0.0,
                "pa": 0.0,
                "intens": 80.0,
                "a3": np.nan,
                "b3": np.inf,
                "a4": 0.01,
                "b4": 0.01,
            },
            {
                "x0": 50.0,
                "y0": 50.0,
                "sma": 30.0,
                "eps": 0.0,
                "pa": 0.0,
                "intens": 60.0,
                "a3": 0.01,
                "b3": 0.01,
                "a4": 0.01,
                "b4": 0.01,
            },
        ]

        model = build_isoster_model(shape, isos, use_harmonics=True, harmonic_orders=[3, 4])

        # Model should be entirely finite despite NaN/Inf harmonic coefficients
        n_nonfinite = np.sum(~np.isfinite(model))
        self.assertEqual(n_nonfinite, 0, f"Model has {n_nonfinite} non-finite pixels from NaN harmonics")

    # ------------------------------------------------------------------
    # Issue 1: Auto-detect harmonic orders from isophote keys
    # ------------------------------------------------------------------

    def test_auto_detect_harmonic_orders_from_keys(self):
        """When harmonic_orders=None, auto-detect available orders from isophote keys."""
        shape = (100, 100)
        # Isophotes with orders 3, 4, 5, 6 present
        isos = [
            {
                "x0": 50.0,
                "y0": 50.0,
                "sma": 10.0,
                "eps": 0.0,
                "pa": 0.0,
                "intens": 100.0,
                "a3": 0.02,
                "b3": 0.01,
                "a4": 0.03,
                "b4": -0.02,
                "a5": 0.01,
                "b5": 0.005,
                "a6": 0.008,
                "b6": -0.003,
            },
            {
                "x0": 50.0,
                "y0": 50.0,
                "sma": 30.0,
                "eps": 0.0,
                "pa": 0.0,
                "intens": 50.0,
                "a3": 0.02,
                "b3": 0.01,
                "a4": 0.03,
                "b4": -0.02,
                "a5": 0.01,
                "b5": 0.005,
                "a6": 0.008,
                "b6": -0.003,
            },
        ]

        # With harmonic_orders=None (default), all available orders should be used
        model_auto = build_isoster_model(shape, isos, use_harmonics=True, harmonic_orders=None)
        # With explicit all orders
        model_explicit = build_isoster_model(shape, isos, use_harmonics=True, harmonic_orders=[3, 4, 5, 6])

        # These should be identical
        np.testing.assert_array_almost_equal(
            model_auto, model_explicit, decimal=10, err_msg="Auto-detected orders should match explicit [3,4,5,6]"
        )

    def test_auto_detect_does_not_silently_drop_higher_orders(self):
        """Default harmonic_orders=None must not silently drop orders > 4."""
        shape = (100, 100)
        # Isophotes with strong a5 signal — dropping it changes the model
        isos = [
            {
                "x0": 50.0,
                "y0": 50.0,
                "sma": 10.0,
                "eps": 0.0,
                "pa": 0.0,
                "intens": 100.0,
                "a3": 0.0,
                "b3": 0.0,
                "a4": 0.0,
                "b4": 0.0,
                "a5": 0.1,
                "b5": 0.0,
            },
            {
                "x0": 50.0,
                "y0": 50.0,
                "sma": 30.0,
                "eps": 0.0,
                "pa": 0.0,
                "intens": 50.0,
                "a3": 0.0,
                "b3": 0.0,
                "a4": 0.0,
                "b4": 0.0,
                "a5": 0.1,
                "b5": 0.0,
            },
        ]

        # Default (harmonic_orders=None) should include a5
        model_default = build_isoster_model(shape, isos, use_harmonics=True)
        # Model with only [3,4] would miss a5 entirely
        model_34_only = build_isoster_model(shape, isos, use_harmonics=True, harmonic_orders=[3, 4])

        # They must differ because a5=0.1 is significant
        max_diff = np.max(np.abs(model_default - model_34_only))
        self.assertGreater(max_diff, 0.1, "Default should include a5, producing different model than [3,4] only")

    def test_auto_detect_with_no_harmonics_present(self):
        """When no harmonic keys present, auto-detect should gracefully use no harmonics."""
        shape = (100, 100)
        isos = [
            {"x0": 50.0, "y0": 50.0, "sma": 10.0, "eps": 0.0, "pa": 0.0, "intens": 100.0},
            {"x0": 50.0, "y0": 50.0, "sma": 30.0, "eps": 0.0, "pa": 0.0, "intens": 50.0},
        ]

        # Should not raise, should produce same result as use_harmonics=False
        model_auto = build_isoster_model(shape, isos, use_harmonics=True, harmonic_orders=None)
        model_no_harm = build_isoster_model(shape, isos, use_harmonics=False)

        np.testing.assert_array_almost_equal(model_auto, model_no_harm, decimal=10)

    # ------------------------------------------------------------------
    # Issue 3: Harmonic detection uses only first isophote row
    # ------------------------------------------------------------------

    def test_harmonic_detected_when_only_later_rows_have_key(self):
        """Harmonics present in later rows but absent from the first must not be dropped.

        Regression test: build_isoster_model() previously checked only
        ``sorted_isos[0]`` to decide whether to create harmonic interpolators.
        If the first valid isophote lacked an ``a{n}`` key, that order was
        silently excluded even when later rows carried it.
        """
        shape = (100, 100)
        # First row has NO a3/b3; later rows do.
        isos = [
            {"x0": 50.0, "y0": 50.0, "sma": 10.0, "eps": 0.0, "pa": 0.0, "intens": 100.0},
            {"x0": 50.0, "y0": 50.0, "sma": 20.0, "eps": 0.0, "pa": 0.0, "intens": 80.0, "a3": 0.1, "b3": 0.05},
            {"x0": 50.0, "y0": 50.0, "sma": 30.0, "eps": 0.0, "pa": 0.0, "intens": 60.0, "a3": 0.1, "b3": 0.05},
        ]

        # Build with harmonics enabled (auto-detect should find a3/b3)
        model_with_harm = build_isoster_model(shape, isos, use_harmonics=True)
        # Build without harmonics for comparison
        model_no_harm = build_isoster_model(shape, isos, use_harmonics=False)

        # The harmonic coefficients are large (a3=0.1), so the models must differ
        max_diff = np.max(np.abs(model_with_harm - model_no_harm))
        self.assertGreater(max_diff, 0.01, "Harmonics present only in later rows should still affect the model")

    def test_harmonic_works_when_all_rows_have_key(self):
        """Harmonics present in every row should still work correctly."""
        shape = (100, 100)
        isos = [
            {"x0": 50.0, "y0": 50.0, "sma": 10.0, "eps": 0.0, "pa": 0.0, "intens": 100.0, "a3": 0.1, "b3": 0.05},
            {"x0": 50.0, "y0": 50.0, "sma": 20.0, "eps": 0.0, "pa": 0.0, "intens": 80.0, "a3": 0.1, "b3": 0.05},
            {"x0": 50.0, "y0": 50.0, "sma": 30.0, "eps": 0.0, "pa": 0.0, "intens": 60.0, "a3": 0.1, "b3": 0.05},
        ]

        model_with_harm = build_isoster_model(shape, isos, use_harmonics=True)
        model_no_harm = build_isoster_model(shape, isos, use_harmonics=False)

        max_diff = np.max(np.abs(model_with_harm - model_no_harm))
        self.assertGreater(max_diff, 0.01, "Harmonics present in all rows should affect the model")

        # Model shape should match input shape
        self.assertEqual(model_with_harm.shape, shape)
        self.assertEqual(model_no_harm.shape, shape)

    # ------------------------------------------------------------------
    # Issue 2: EA mode angle space mismatch
    # ------------------------------------------------------------------

    def test_ea_mode_angle_space_for_harmonics(self):
        """Harmonics fitted in ψ-space (EA mode) must be evaluated in ψ-space in model.

        When use_eccentric_anomaly=True in isophote fitting, harmonic coefficients
        (a3, b3, a4, b4...) are fitted against eccentric anomaly ψ, not position
        angle φ. The model must use ψ = arctan2(y_rot/(1-eps), x_rot) to evaluate
        these harmonics correctly.
        """
        shape = (200, 200)
        eps = 0.6  # High ellipticity to make φ vs ψ difference large

        # Create isophotes with strong b4 (boxy/disky) signal, marked as EA mode
        isos = [
            {
                "x0": 100.0,
                "y0": 100.0,
                "sma": 20.0,
                "eps": eps,
                "pa": 0.0,
                "intens": 100.0,
                "a4": 0.0,
                "b4": 0.08,
                "use_eccentric_anomaly": True,
            },
            {
                "x0": 100.0,
                "y0": 100.0,
                "sma": 60.0,
                "eps": eps,
                "pa": 0.0,
                "intens": 50.0,
                "a4": 0.0,
                "b4": 0.08,
                "use_eccentric_anomaly": True,
            },
        ]

        # Build model with EA-aware angle computation
        model_ea = build_isoster_model(shape, isos, use_harmonics=True, harmonic_orders=[4])

        # Build model forcing φ-space (wrong for EA-fitted data)
        isos_phi = [dict(iso, use_eccentric_anomaly=False) for iso in isos]
        model_phi = build_isoster_model(shape, isos_phi, use_harmonics=True, harmonic_orders=[4])

        # At eps=0.6, ψ and φ differ significantly along minor axis.
        # The two models should differ meaningfully.
        max_diff = np.max(np.abs(model_ea - model_phi))
        self.assertGreater(max_diff, 0.1, f"EA vs φ models should differ at eps={eps}, got max_diff={max_diff:.4f}")

    def test_ea_mode_explicit_parameter_override(self):
        """Explicit use_eccentric_anomaly parameter overrides isophote-stored flag."""
        shape = (200, 200)
        eps = 0.6

        isos = [
            {
                "x0": 100.0,
                "y0": 100.0,
                "sma": 20.0,
                "eps": eps,
                "pa": 0.0,
                "intens": 100.0,
                "a4": 0.0,
                "b4": 0.08,
                "use_eccentric_anomaly": True,
            },
            {
                "x0": 100.0,
                "y0": 100.0,
                "sma": 60.0,
                "eps": eps,
                "pa": 0.0,
                "intens": 50.0,
                "a4": 0.0,
                "b4": 0.08,
                "use_eccentric_anomaly": True,
            },
        ]

        # Explicit override to False should use φ-space despite isophote flag
        model_override = build_isoster_model(
            shape, isos, use_harmonics=True, harmonic_orders=[4], use_eccentric_anomaly=False
        )

        # This should match φ-space model
        isos_phi = [dict(iso, use_eccentric_anomaly=False) for iso in isos]
        model_phi = build_isoster_model(shape, isos_phi, use_harmonics=True, harmonic_orders=[4])

        np.testing.assert_array_almost_equal(
            model_override,
            model_phi,
            decimal=10,
            err_msg="Explicit use_eccentric_anomaly=False should override isophote flag",
        )

    def test_pa_unwrap_flips_odd_harmonics_with_the_frame(self):
        """Inner rings written as pa + pi must not change the rendered model.

        The renderer unwraps pa with period pi so it can interpolate it over sma, so
        every ring outward of a > pi/2 jump is evaluated in a frame rotated by pi.
        The ellipse is unchanged, but sin(n*theta) and cos(n*theta) flip sign for
        odd n, so the odd-order coefficients of the rotated rings must be negated.
        Two lists that describe the same isophotes, one with its four inner rings
        written as pa + pi with a3, b3, a5, b5 negated, must render identically;
        the even orders a4, b4 are left alone.
        """
        shape = (121, 121)
        base = [
            {
                "x0": 60.0,
                "y0": 60.0,
                "sma": float(s),
                "eps": 0.3,
                "pa": 0.4,
                "intens": 100.0 * np.exp(-s / 15.0),
                "a3": 0.02,
                "b3": -0.015,
                "a4": 0.01,
                "b4": 0.005,
                "a5": 0.008,
                "b5": -0.004,
            }
            for s in np.arange(1.0, 41.0)
        ]
        rotated = [dict(iso) for iso in base]
        for iso in rotated[:4]:
            iso["pa"] += np.pi
            for key in ("a3", "b3", "a5", "b5"):
                iso[key] = -iso[key]

        model_base = build_isoster_model(shape, base, use_harmonics=True)
        model_rotated = build_isoster_model(shape, rotated, use_harmonics=True)
        np.testing.assert_allclose(model_rotated, model_base, rtol=0, atol=1e-9)

        # The harmonics are large enough that a sign error would have been visible.
        model_off = build_isoster_model(shape, base, use_harmonics=False)
        self.assertGreater(np.max(np.abs(model_base - model_off)), 1.0)

    def test_ring_with_summed_harmonic_amplitude_above_one_is_ignored(self):
        """A ring with sum_n hypot(a_n, b_n) >= 1 must not drive the radius to zero."""
        shape = (101, 101)
        isos = [
            {"x0": 50.0, "y0": 50.0, "sma": s, "eps": 0.0, "pa": 0.0, "intens": 1000.0 * np.exp(-s / 5.0),
             "a3": 0.0, "b3": 0.0, "a4": 0.0, "b4": 0.0}
            for s in np.arange(1.0, 41.0)
        ]
        isos[-5].update(a3=0.6, b4=0.6)  # amplitudes 0.6 + 0.6: can reach dr/r = 1.2

        with self.assertWarns(UserWarning):
            model = build_isoster_model(shape, isos, use_harmonics=True, harmonic_min_snr=0)
        model_off = build_isoster_model(shape, isos, use_harmonics=False)
        np.testing.assert_allclose(model, model_off, rtol=0, atol=1e-12)

        isos[-5].update(a3=0.3, b4=0.3)  # below the bound: still applied
        model_small = build_isoster_model(shape, isos, use_harmonics=True, harmonic_min_snr=0)
        self.assertGreater(np.max(np.abs(model_small - model_off)), 0.0)

    def test_insignificant_harmonics_are_not_drawn(self):
        """Order n is drawn only where hypot(a_n, b_n) / sqrt((err_a^2 + err_b^2) / 2) >= harmonic_min_snr."""
        shape = (101, 101)

        def rings(a4_err):
            return [
                {"x0": 50.0, "y0": 50.0, "sma": s, "eps": 0.0, "pa": 0.0, "intens": 1000.0 * np.exp(-s / 5.0),
                 "a3": 0.0, "b3": 0.0, "a4": 0.05, "b4": 0.0,
                 "a3_err": 0.01, "b3_err": 0.01, "a4_err": a4_err, "b4_err": a4_err}
                for s in np.arange(1.0, 41.0)
            ]

        model_off = build_isoster_model(shape, rings(0.05), use_harmonics=False)
        noise = build_isoster_model(shape, rings(0.05))  # S/N 1: withheld at the default 2
        np.testing.assert_array_equal(noise, model_off)
        noise_drawn = build_isoster_model(shape, rings(0.05), harmonic_min_snr=0)
        self.assertGreater(np.max(np.abs(noise_drawn - model_off)), 1.0)
        signal = build_isoster_model(shape, rings(0.01))  # S/N 5: drawn
        np.testing.assert_array_equal(signal, noise_drawn)

    def test_twisting_rings_draw_every_pixel_at_a_consistent_radius(self):
        """Each pixel is drawn at a radius r whose interpolated ellipse passes through it.

        Fixed-point steps from the outer ring's geometry oscillate where the position angle
        twists quickly (bars, spiral cores) and left such pixels at a radius no ring passes
        through: issue 2, 2.05 instead of 1.45 sigma on SGA-2020 spiral cores.
        """
        sma = np.arange(1.0, 41.0)
        pa = np.interp(sma, [4.0, 20.0], [0.0, 1.5])
        isos = [{"x0": 50.0, "y0": 50.0, "sma": s, "eps": 0.6, "pa": p, "intens": 1000.0 * np.exp(-s / 5.0)}
                for s, p in zip(sma, pa)]
        model = build_isoster_model((101, 101), isos, use_harmonics=False)

        intens = np.array([iso["intens"] for iso in isos])
        r = np.interp(model, intens[::-1], sma[::-1])  # the radius each pixel was drawn at
        yy, xx = np.mgrid[:101, :101] - 50.0
        pa_r = np.interp(r, sma, pa)
        x_rot = xx * np.cos(pa_r) + yy * np.sin(pa_r)
        y_rot = -xx * np.sin(pa_r) + yy * np.cos(pa_r)
        inside = (r > sma[0]) & (r < sma[-1])
        np.testing.assert_array_less(np.abs(np.hypot(x_rot, y_rot / 0.4) - r)[inside], 0.1)

if __name__ == "__main__":
    unittest.main()
