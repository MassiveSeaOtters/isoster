"""Tests for multi-method comparison QA figure helpers."""

import numpy as np

from isoster.plotting import (
    METHOD_STYLES,
    build_method_profile,
    contour_autoprof,
    contour_isoster_phi,
    contour_isoster_psi,
    contour_pure_ellipse,
    normalize_pa_degrees,
    plot_comparison_qa_figure,
    plot_qa_summary,
    transform_sb_profile,
)


def _make_isophote_list(n=10, pa=0.5):
    """Create a minimal isophote list for testing."""
    return [
        {
            "sma": float(i + 1),
            "intens": 100.0 / (i + 1),
            "eps": 0.3,
            "pa": pa,
            "x0": 50.0,
            "y0": 50.0,
            "stop_code": 0,
            "intens_err": 1.0,
            "rms": 2.0,
        }
        for i in range(n)
    ]


def _make_image(shape=(64, 64), seed=42):
    return np.random.default_rng(seed).normal(100, 10, shape)


# ---------------------------------------------------------------------------
# METHOD_STYLES
# ---------------------------------------------------------------------------


class TestMethodStyles:
    def test_has_required_methods(self):
        assert "isoster" in METHOD_STYLES
        assert "photutils" in METHOD_STYLES
        assert "autoprof" in METHOD_STYLES

    def test_style_keys(self):
        for method, style in METHOD_STYLES.items():
            assert "color" in style
            assert "marker" in style
            assert "label" in style


# ---------------------------------------------------------------------------
# build_method_profile
# ---------------------------------------------------------------------------


class TestBuildMethodProfile:
    def test_from_isophote_list(self):
        isos = _make_isophote_list()
        prof = build_method_profile(isos)
        assert prof is not None
        assert "sma" in prof
        assert "x_axis" in prof
        assert len(prof["sma"]) == 10
        np.testing.assert_allclose(prof["x_axis"], prof["sma"] ** 0.25)

    def test_from_isophote_list_with_stop_codes(self):
        isos = _make_isophote_list()
        isos[5]["stop_code"] = 2
        prof = build_method_profile(isos)
        assert "stop_codes" in prof
        assert prof["stop_codes"][5] == 2

    def test_optional_fields_included(self):
        isos = _make_isophote_list()
        prof = build_method_profile(isos)
        assert "x0" in prof
        assert "y0" in prof
        assert "intens_err" in prof
        assert "rms" in prof

    def test_from_array_dict(self):
        """AutoProf-style: dict with numpy array values."""
        sma = np.arange(1.0, 11.0)
        prof = build_method_profile(
            {
                "sma": sma,
                "intens": 100.0 / sma,
                "eps": np.full(10, 0.3),
                "pa": np.full(10, 0.5),
            }
        )
        assert prof is not None
        assert len(prof["sma"]) == 10
        assert "stop_codes" not in prof

    def test_empty_list_returns_none(self):
        assert build_method_profile([]) is None

    def test_empty_dict_returns_none(self):
        assert build_method_profile({"sma": np.array([])}) is None


# ---------------------------------------------------------------------------
# plot_comparison_qa_figure — layout modes
# ---------------------------------------------------------------------------


class TestPlotComparisonQaFigure:
    """Smoke tests for the three layout modes."""

    def test_mode1_solo_with_model(self, tmp_path):
        """Mode 1: single method with model -> image, model, residual."""
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(20)
        out = tmp_path / "mode1.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            models={"isoster": model},
            title="mode 1 solo",
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_mode1_solo_no_model(self, tmp_path):
        """Mode 1: single method without model -> blank model/residual."""
        image = _make_image()
        isos = _make_isophote_list(20)
        out = tmp_path / "mode1_nomodel.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            title="mode 1 no model",
            output_path=out,
        )
        assert out.exists()

    def test_mode2_one_on_one(self, tmp_path):
        """Mode 2: two methods with models -> image, 2x residual+overlays."""
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(15)
        out = tmp_path / "mode2.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={
                "isoster": build_method_profile(isos),
                "photutils": build_method_profile(isos),
            },
            models={"isoster": model, "photutils": model},
            title="mode 2 one-on-one",
            output_path=out,
        )
        assert out.exists()

    def test_mode3_three_way(self, tmp_path):
        """Mode 3: three methods -> image, 3x residual+overlays."""
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(15)
        sma = np.arange(1.0, 16.0)
        autoprof_prof = build_method_profile(
            {
                "sma": sma,
                "intens": 100.0 / sma,
                "eps": np.full(15, 0.3),
                "pa": np.full(15, 0.5),
            }
        )
        out = tmp_path / "mode3.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={
                "isoster": build_method_profile(isos),
                "photutils": build_method_profile(isos),
                "autoprof": autoprof_prof,
            },
            models={"isoster": model, "photutils": model, "autoprof": model},
            title="mode 3 three-way",
            output_path=out,
        )
        assert out.exists()

    def test_empty_profiles_no_crash(self, tmp_path):
        """Empty profiles dict should not crash."""
        image = _make_image()
        out = tmp_path / "empty.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={},
            title="empty",
            output_path=out,
        )
        assert out.exists()

    def test_mask_overlay(self, tmp_path):
        """Mask is overlaid on the data image."""
        image = _make_image()
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:20, 10:20] = True
        isos = _make_isophote_list(15)
        out = tmp_path / "mask.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            mask=mask,
            title="with mask",
            output_path=out,
        )
        assert out.exists()

    def test_relative_residual(self, tmp_path):
        """relative_residual=True produces a valid figure."""
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(15)
        out = tmp_path / "relative.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            models={"isoster": model},
            relative_residual=True,
            title="relative residual",
            output_path=out,
        )
        assert out.exists()

    def test_method_without_stop_codes(self, tmp_path):
        """Method without stop_codes uses plain scatter."""
        image = _make_image()
        sma = np.arange(1.0, 21.0)
        prof = build_method_profile(
            {
                "sma": sma,
                "intens": 100.0 / sma,
                "eps": np.full(20, 0.3),
                "pa": np.full(20, 0.5),
            }
        )
        out = tmp_path / "no_stop.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"autoprof": prof},
            title="no stop codes",
            output_path=out,
        )
        assert out.exists()

    def test_errorbars_rendered(self, tmp_path):
        """Method with intens_err renders errorbars."""
        image = _make_image()
        isos = _make_isophote_list(15)
        out = tmp_path / "errorbars.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos)},
            title="with errorbars",
            output_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_unknown_method_name_no_keyerror(self, tmp_path):
        """Regression test for P3: methods missing from METHOD_STYLES must not crash.

        style = styles.get(method_name, {}) followed by style["color"] raised
        KeyError for any method not in METHOD_STYLES or user method_styles.
        Unknown methods now fall back to a default color.
        """
        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(15)
        out = tmp_path / "unknown_method.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={
                "isoster": build_method_profile(isos),
                "my_custom_tool": build_method_profile(isos),
            },
            models={"isoster": model, "my_custom_tool": model},
            title="unknown method",
            output_path=out,
        )
        assert out.exists()

    def test_unknown_method_with_partial_user_styles(self, tmp_path):
        """P3 variant: user method_styles that only label (no color) must not crash."""
        image = _make_image()
        isos = _make_isophote_list(15)
        out = tmp_path / "partial_styles.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"my_custom_tool": build_method_profile(isos)},
            title="partial user styles",
            output_path=out,
            method_styles={"my_custom_tool": {"label": "custom"}},
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# Cross-method PA normalization
# ---------------------------------------------------------------------------


class TestCrossMethodPaNormalization:
    """Verify PA profiles from different methods are anchored consistently."""

    def test_pa_offset_by_180_is_resolved(self):
        """Two methods with PAs ~180 deg apart should be normalized close."""
        # Method A: PA ~ 10 deg (in radians)
        pa_a = np.full(10, np.radians(10.0))
        # Method B: PA ~ 190 deg -> same physical direction as 10 deg
        pa_b = np.full(10, np.radians(190.0))

        pa_a_deg = np.degrees(pa_a)
        pa_b_deg = np.degrees(pa_b)

        # Normalize A first (no anchor)
        norm_a = normalize_pa_degrees(pa_a_deg)
        ref_median = float(np.nanmedian(norm_a))

        # Normalize B anchored to A's median
        norm_b = normalize_pa_degrees(pa_b_deg, anchor=ref_median)

        # They should now be within ~5 deg of each other (not 180 apart)
        diff = np.abs(np.nanmedian(norm_a) - np.nanmedian(norm_b))
        assert diff < 10.0, f"PA profiles should be close after anchoring, got diff={diff:.1f} deg"

    def test_pa_no_offset_preserved(self):
        """Two methods with similar PAs should stay similar after anchoring."""
        pa_a = np.full(10, np.radians(45.0))
        pa_b = np.full(10, np.radians(47.0))

        norm_a = normalize_pa_degrees(np.degrees(pa_a))
        ref_median = float(np.nanmedian(norm_a))
        norm_b = normalize_pa_degrees(np.degrees(pa_b), anchor=ref_median)

        diff = np.abs(np.nanmedian(norm_a) - np.nanmedian(norm_b))
        assert diff < 5.0, f"Similar PAs should remain close, got diff={diff:.1f}"


class TestContourHelpers:
    def test_phi_and_psi_contours_differ_at_high_ellipticity(self):
        iso = {
            "sma": 20.0,
            "eps": 0.65,
            "pa": 0.2,
            "x0": 32.0,
            "y0": 31.0,
            "a3": 0.08,
            "b3": 0.03,
        }
        phi = contour_isoster_phi(iso, n_points=180)
        psi = contour_isoster_psi(iso, n_points=180)
        assert phi.shape == psi.shape == (180, 2)
        assert np.nanmax(np.abs(phi - psi)) > 0.5

    def test_large_delta_row_falls_back_to_pure_ellipse(self):
        iso = {
            "sma": 20.0,
            "eps": 0.4,
            "pa": 0.2,
            "x0": 32.0,
            "y0": 31.0,
            "a3": 0.75,
            "b3": 0.0,
        }
        pure = contour_pure_ellipse(iso, n_points=180)
        psi = contour_isoster_psi(iso, n_points=180)
        np.testing.assert_allclose(psi, pure)

    def test_autoprof_contours_are_pure_ellipse_fallback(self):
        iso = {
            "sma": 20.0,
            "eps": 0.3,
            "pa": 0.2,
            "x0": 32.0,
            "y0": 31.0,
            "a4": 0.1,
            "b4": 0.2,
        }
        np.testing.assert_allclose(
            contour_autoprof(iso, n_points=180),
            contour_pure_ellipse(iso, n_points=180),
        )


class TestAsinhSurfaceBrightnessProfile:
    def test_calibrated_asinh_matches_log10_at_high_signal(self):
        intens = np.array([1.0e5, 1.0e4, 10.0])
        intens_err = np.ones_like(intens)
        kwargs = {
            "sb_zeropoint": 27.0,
            "pixel_scale_arcsec": 0.2,
            "sb_asinh_softening": 1.0,
        }
        log_y, *_ = transform_sb_profile(
            intens,
            intens_err,
            sb_profile_scale="log10",
            **kwargs,
        )
        asinh_y, _asinh_err, _label, invert, zero_y = transform_sb_profile(
            intens,
            intens_err,
            sb_profile_scale="asinh",
            **kwargs,
        )
        assert invert is True
        assert np.isfinite(zero_y)
        np.testing.assert_allclose(asinh_y[:2], log_y[:2], atol=2.0e-4)

    def test_plot_qa_summary_draws_asinh_zero_intensity_line(self, tmp_path, monkeypatch):
        from matplotlib.axes import Axes

        image = _make_image()
        model = np.full((64, 64), 100.0)
        isos = _make_isophote_list(20)
        _y, _yerr, _label, _invert, zero_y = transform_sb_profile(
            np.array([iso["intens"] for iso in isos]),
            np.array([iso["intens_err"] for iso in isos]),
            sb_zeropoint=27.0,
            pixel_scale_arcsec=0.2,
            sb_profile_scale="asinh",
            sb_asinh_softening=1.0,
        )

        calls = []
        original_axhline = Axes.axhline

        def recording_axhline(self, y=0, *args, **kwargs):
            calls.append((float(y), kwargs.get("linestyle")))
            return original_axhline(self, y, *args, **kwargs)

        monkeypatch.setattr(Axes, "axhline", recording_axhline)
        out = tmp_path / "asinh_sb.png"
        plot_qa_summary(
            title="asinh sb",
            image=image,
            isoster_model=model,
            isoster_res=isos,
            filename=out,
            sb_zeropoint=27.0,
            pixel_scale_arcsec=0.2,
            sb_profile_scale="asinh",
            sb_asinh_softening=1.0,
        )
        assert out.exists()
        assert any(np.isclose(y, zero_y) and linestyle == "--" for y, linestyle in calls)


# ---------------------------------------------------------------------------
# P4 plotting batch regression tests
# ---------------------------------------------------------------------------


class TestP4PlottingBatch:
    def test_psi_contour_matches_exact_model_radius(self):
        """P4: contour_isoster_psi must draw sma/(1-delta), not sma*(1+delta).

        The model evaluates the profile at r*(1-delta), so the perturbed
        isophote sits exactly at sma/(1-delta); the first-order form
        diverged ~25% at delta=0.2.
        """
        sma, b4 = 10.0, 0.2
        iso = {
            "sma": sma,
            "eps": 0.0,
            "pa": 0.0,
            "x0": 50.0,
            "y0": 50.0,
            "b4": b4,  # delta(psi) = b4*cos(4*psi), max at psi=0
        }
        contour = contour_isoster_psi(iso, n_points=720)
        r = np.hypot(contour[:, 0] - 50.0, contour[:, 1] - 50.0)
        expected_max = sma / (1.0 - b4)  # 12.5 at psi=0
        first_order_max = sma * (1.0 + b4)  # 12.0 — the old wrong value
        np.testing.assert_allclose(np.max(r), expected_max, rtol=1e-3)
        assert abs(np.max(r) - first_order_max) > 0.3, "still using the first-order form"

    def test_overlay_builder_skips_nan_mode_flags(self):
        """P4: NaN use_eccentric_anomaly/tool must not become True/'nan'.

        bool(NaN) is True (silently switching to EA contours) and str(NaN)
        is 'nan' (falling through to the psi default). Missing flags must
        keep isoster's phi-mode defaults.
        """
        from isoster.plotting import _build_isos_for_overlay

        isos = _make_isophote_list(5)
        isos[0]["use_eccentric_anomaly"] = np.nan
        isos[0]["tool"] = np.nan
        prof = build_method_profile(isos)
        rebuilt = _build_isos_for_overlay(prof)
        first = rebuilt[0]
        # NaN flags are dropped so downstream defaults (phi-mode, isoster) apply
        assert "use_eccentric_anomaly" not in first
        assert "tool" not in first
        # Rows with real values keep them
        isos[1]["use_eccentric_anomaly"] = True
        prof2 = build_method_profile(isos)
        rebuilt2 = _build_isos_for_overlay(prof2)
        assert rebuilt2[1]["use_eccentric_anomaly"] is True

    def test_runtime_scalar_passthrough(self):
        """P4: scalar runtime_seconds/retries survive build_method_profile (dict path)."""
        sma = np.arange(1.0, 6.0)
        prof = build_method_profile(
            {
                "sma": sma,
                "intens": 100.0 / sma,
                "eps": np.full(5, 0.3),
                "pa": np.full(5, 0.5),
                "runtime_seconds": 0.15,
                "retries": 2,
            }
        )
        assert prof["runtime_seconds"] == 0.15
        assert prof["retries"] == 2

    def test_absolute_residual_panel_has_no_arbitrary_floor(self, tmp_path, monkeypatch):
        """P4: absolute-mode residual panels must not floor the color range at 0.05."""
        import matplotlib.pyplot as plt

        image = np.full((64, 64), 1e-3)
        model = np.full((64, 64), 1.1e-3)  # residuals ~1e-4
        out = tmp_path / "low_flux.png"
        monkeypatch.setattr(plt, "close", lambda *a, **k: None)
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(_make_isophote_list(10))},
            models={"isoster": model},
            title="low flux",
            output_path=out,
        )
        assert out.exists()
        fig = plt.gcf()
        residual_axes = [ax for ax in fig.axes if ax.images]
        assert residual_axes
        for ax in residual_axes:
            lo, hi = ax.images[0].get_clim()
            assert abs(hi) < 0.05, f"absolute residual floor still applied: clim=({lo}, {hi})"
        plt.close("all")

    def test_overlay_skips_nan_geometry_rows(self):
        """P4: draw_isophote_overlays must skip rows with NaN geometry."""
        import matplotlib.pyplot as plt

        from isoster.plotting import draw_isophote_overlays

        fig, ax = plt.subplots()
        # sma > 1 for all rows so only the NaN-geometry rows are skipped
        isos = [dict(iso, sma=float(i + 2)) for i, iso in enumerate(_make_isophote_list(5))]
        isos[1]["eps"] = np.nan
        isos[2]["pa"] = np.nan
        draw_isophote_overlays(ax, isos, step=1)
        assert len(ax.patches) == 3  # 5 rows minus the 2 NaN-geometry rows
        plt.close(fig)

    def test_asinh_softening_shared_across_methods(self, tmp_path, monkeypatch):
        """P4: all overlaid methods share one asinh softening (from the reference)."""
        import isoster.plotting as plotting_mod

        image = _make_image()
        isos_a = _make_isophote_list(15)
        isos_b = [dict(iso, intens=iso["intens"] * 5.0) for iso in isos_a]

        softenings = []
        real_transform = plotting_mod.transform_sb_profile

        def spy(*args, **kwargs):
            softenings.append(kwargs.get("sb_asinh_softening"))
            return real_transform(*args, **kwargs)

        monkeypatch.setattr(plotting_mod, "transform_sb_profile", spy)
        out = tmp_path / "asinh_shared.png"
        plot_comparison_qa_figure(
            image=image,
            profiles={"isoster": build_method_profile(isos_a), "photutils": build_method_profile(isos_b)},
            title="shared asinh",
            output_path=out,
            sb_profile_scale="asinh",
        )
        assert out.exists()
        assert len(softenings) >= 2
        assert all(s is not None and s > 0 for s in softenings)
        assert len(set(softenings)) == 1, f"per-method softening detected: {softenings}"


def test_p2_extended_figure_fractional_label_matches_sign(tmp_path, monkeypatch):
    """P2: the extended figure's fractional-residual label must match the data.

    compute_fractional_residual_percent returns 100*(model-data)/data but the
    colorbar was labeled '(data - model) / data [%]' — the wrong sign.
    """
    import matplotlib.pyplot as plt

    from isoster.plotting import plot_qa_summary_extended

    image = np.random.default_rng(0).normal(100.0, 1.0, (64, 64))
    model = np.full((64, 64), 100.0)
    isos = _make_isophote_list(15)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    plot_qa_summary_extended(
        "P2 sign",
        image,
        model,
        isos,
        relative_residual=True,
        filename=str(tmp_path / "ext.png"),
    )
    fig = plt.gcf()
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert any("(model - data) / data" in lbl for lbl in labels), f"no correct-sign label found in {labels}"
    assert not any("(data - model) / data" in lbl for lbl in labels), "stale wrong-sign label still present"
    plt.close("all")
