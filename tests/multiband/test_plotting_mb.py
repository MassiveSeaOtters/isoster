"""Smoke tests for ``isoster.multiband.plotting_mb`` (decision D15).

Renders the composite QA figure on a small B=3 result fixture and
verifies the call returns without exception. No pixel comparison.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402  -- must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

from isoster.multiband import IsosterConfigMB, fit_image_multiband  # noqa: E402
from isoster.multiband.plotting_mb import plot_qa_summary_mb  # noqa: E402


def _planted(amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = w = 192
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = x - 96.0, y - 96.0
    cos_pa, sin_pa = np.cos(0.5), np.sin(0.5)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / 0.7) ** 2)
    return amp * np.exp(-3.0 * ((r / 25.0) ** (1 / 1.5) - 1.0)) + rng.normal(0, 0.05, (h, w))


@pytest.fixture
def three_band_result():
    img_g = _planted(100.0, 1)
    img_r = _planted(200.0, 2)
    img_i = _planted(300.0, 3)
    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="r",
        sma0=15.0,
        astep=0.2,
        maxsma=50.0,
        debug=True,
        compute_deviations=True,
        nclip=0,
    )
    return cfg, [img_g, img_r, img_i], fit_image_multiband([img_g, img_r, img_i], None, cfg)


def test_plot_qa_summary_mb_renders_without_exception(three_band_result, tmp_path):
    cfg, images, result = three_band_result
    fig = plot_qa_summary_mb(result, images, output_path=tmp_path / "qa.png")
    assert (tmp_path / "qa.png").exists()
    plt.close(fig)


def test_plot_qa_summary_mb_with_sb_constants(three_band_result, tmp_path):
    cfg, images, result = three_band_result
    fig = plot_qa_summary_mb(
        result,
        images,
        sb_zeropoint=27.0,
        pixel_scale_arcsec=0.168,
        softening_per_band={"g": 0.05, "r": 0.05, "i": 0.05},
        output_path=tmp_path / "qa_sb.png",
        title="multiband QA test",
    )
    assert (tmp_path / "qa_sb.png").exists()
    plt.close(fig)


def test_plot_qa_summary_mb_requires_band_list_when_missing():
    """Empty bands raises a clear error."""
    with pytest.raises(ValueError, match="bands is required"):
        plot_qa_summary_mb({"isophotes": []}, [], bands=[])


def test_plot_qa_summary_mb_image_count_must_match_bands(three_band_result):
    cfg, images, result = three_band_result
    # Drop one band's image
    with pytest.raises(ValueError, match="does not match"):
        plot_qa_summary_mb(result, images[:2])


def test_plot_qa_summary_mb_loose_validity_renders_n_valid_panel(tmp_path):
    """D9 backport: when the result was produced under loose validity,
    the QA figure adds a 4th stacked panel below the geometry block
    showing per-band ``n_valid / n_attempted``."""
    img_g = _planted(100.0, 1)
    img_r = _planted(200.0, 2)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        astep=0.2,
        maxsma=50.0,
        debug=True,
        compute_deviations=True,
        nclip=0,
        loose_validity=True,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    fig = plot_qa_summary_mb(
        result,
        [img_g, img_r],
        output_path=tmp_path / "qa_loose.png",
    )
    assert (tmp_path / "qa_loose.png").exists()
    # The bottom-right gridspec must have 4 stacked rows in loose mode
    # (eps / pa / center / n_valid) instead of the default 3.
    geom_axes = [ax for ax in fig.axes if ax.get_ylabel().startswith(r"$N_{\rm valid}")]
    assert len(geom_axes) == 1
    plt.close(fig)


def _harmonic_panel_lines(fig, ylabel):
    """Return the per-band data lines of the harmonic panel with the given ylabel."""
    ax = next(a for a in fig.axes if a.get_ylabel() == ylabel)
    return [ln for ln in ax.lines if ln.get_marker() == "o"]


def _stored_column(isophotes, key):
    return np.array([float(iso.get(key, np.nan)) for iso in isophotes])


def test_harmonic_panel_plots_stored_values_independent_mode(three_band_result):
    """Regression test for P1 (multiband): per-band post-hoc coefficients are
    stored Bender-normalized, so the panel must plot them as stored — not
    divide by sma*|grad| a second time."""
    cfg, images, result = three_band_result
    assert result["harmonics_shared"] is False  # default 'independent' mode
    fig = plot_qa_summary_mb(result, images)
    lines = _harmonic_panel_lines(fig, r"$a_4$")

    bands = list(result["bands"])
    isophotes = result["isophotes"]
    sma = _stored_column(isophotes, "sma")
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    assert len(lines) == len(bands)
    for b_idx, b in enumerate(bands):
        stored = _stored_column(isophotes, f"a4_{b}")
        m = valid & np.isfinite(stored) & (sma > 0)
        np.testing.assert_allclose(
            lines[b_idx].get_ydata(),
            stored[m],
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"a4_{b}: plotted values differ from stored (double normalization?)",
        )
    plt.close(fig)


def test_harmonic_panel_normalizes_raw_shared_coefficients():
    """P1 counterpart: joint modes store raw shared coefficients, which the
    panel must normalize per band by sma*|grad_b| (decision D16)."""
    img_g = _planted(100.0, 1)
    img_r = _planted(200.0, 2)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        astep=0.2,
        maxsma=50.0,
        debug=True,
        compute_deviations=True,
        nclip=0,
        multiband_higher_harmonics="shared",
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    assert result["harmonics_shared"] is True
    fig = plot_qa_summary_mb(result, [img_g, img_r])
    lines = _harmonic_panel_lines(fig, r"$a_4$")

    bands = list(result["bands"])
    isophotes = result["isophotes"]
    sma = _stored_column(isophotes, "sma")
    valid = np.array([bool(iso.get("valid", True)) for iso in isophotes])
    assert len(lines) == len(bands)
    for b_idx, b in enumerate(bands):
        raw = _stored_column(isophotes, f"a4_{b}")
        grad = _stored_column(isophotes, f"grad_{b}")
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.where((sma * grad) != 0, -raw / (sma * grad), np.nan)
        m = valid & np.isfinite(expected) & (sma > 0)
        np.testing.assert_allclose(
            lines[b_idx].get_ydata(),
            expected[m],
            rtol=1e-6,
            atol=1e-12,
            err_msg=f"a4_{b}: plotted values are not the raw coefficients normalized per band",
        )
    plt.close(fig)


def test_m11_model_inputs_follow_harmonic_orders():
    """Regression test for M11: per-band model inputs must include orders > 4.

    The residual-mosaic row builder hardcoded `for n in (3, 4)`, silently
    dropping orders 5+ from build_isoster_model even though the fit stored
    them. It now follows result['harmonic_orders'] (or detects them from
    the row keys).
    """
    from isoster.multiband.plotting_mb import _isophotes_to_per_band_singleband_lists

    isophotes = [
        {
            "sma": 10.0,
            "x0": 50.0,
            "y0": 50.0,
            "eps": 0.2,
            "pa": 0.3,
            "valid": True,
            "intens_g": 100.0,
            "intens_r": 200.0,
            "a3_g": 0.01,
            "b3_g": 0.0,
            "a4_g": -0.02,
            "b4_g": 0.0,
            "a5_g": 0.005,
            "b5_g": 0.001,
            "a6_g": -0.003,
            "b6_g": 0.002,
            "a3_r": 0.01,
            "b3_r": 0.0,
            "a4_r": -0.02,
            "b4_r": 0.0,
            "a5_r": 0.005,
            "b5_r": 0.001,
            "a6_r": -0.003,
            "b6_r": 0.002,
        }
    ]
    for orders in ([3, 4, 5, 6], None):
        lists = _isophotes_to_per_band_singleband_lists(isophotes, ["g", "r"], orders)
        for b in ("g", "r"):
            row = lists[b][0]
            for n in (3, 4, 5, 6):
                assert f"a{n}" in row and f"b{n}" in row, f"missing order {n} for band {b} (orders={orders})"
            assert row["a5"] == 0.005 and row["b6"] == 0.002
