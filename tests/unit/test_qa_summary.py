"""Tests for the main QA summary figure (plot_qa_summary)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402 -- must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

from isoster.config import IsosterConfig  # noqa: E402
from isoster.driver import fit_image  # noqa: E402
from isoster.model import build_isoster_model  # noqa: E402
from isoster.plotting import plot_qa_summary  # noqa: E402


def _make_boxy_image(size=80, seed=42):
    """Exponential mock with a planted cos(4*theta) perturbation (b4 < 0)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    cx = cy = size / 2.0
    radius = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)
    image = np.exp(-radius / 8.0) * (1.0 + 0.1 * np.cos(4 * theta))
    image += rng.normal(0.0, 0.005, image.shape)
    return image, cx, cy


def test_harmonic_panel_plots_stored_coefficients(tmp_path, monkeypatch):
    """Regression test for P1: the harmonic panel must plot stored values.

    Stored a_n/b_n are already Bender-normalized by the fitter; the panel
    used to divide by sma*|grad| a second time, shrinking the amplitudes
    and distorting their radial shape.
    """
    image, cx, cy = _make_boxy_image()
    config = IsosterConfig(x0=cx, y0=cy, eps=0.2, pa=0.0, sma0=8.0, minsma=2.0, maxsma=25.0)
    isophotes = fit_image(image, config=config)["isophotes"]
    model = build_isoster_model(image.shape, isophotes)

    # plot_qa_summary saves and closes the figure; keep it alive for inspection
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    plot_qa_summary("P1 regression", image, model, isophotes, filename=str(tmp_path / "qa.png"))
    fig = plt.gcf()

    # Locate the harmonic panel by its ylabel
    harm_ax = next(ax for ax in fig.axes if ax.get_ylabel() == r"$a_n, b_n$")

    sma = np.array([iso["sma"] for iso in isophotes])
    x_axis = sma**0.25
    valid_mask = np.isfinite(x_axis) & (sma > 1.0)

    plotted = {}
    for coll in harm_ax.collections:
        label = coll.get_label()
        offsets = coll.get_offsets()
        plotted[label] = np.asarray(offsets[:, 1]) if len(offsets) else np.array([])

    for label, key in (
        (r"$a_3$", "a3"),
        (r"$b_3$", "b3"),
        (r"$a_4$", "a4"),
        (r"$b_4$", "b4"),
    ):
        stored = np.array([iso[key] for iso in isophotes])[valid_mask]
        assert label in plotted, f"missing harmonic collection {label}"
        np.testing.assert_allclose(
            plotted[label],
            stored,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{key}: plotted values differ from stored coefficients (double normalization?)",
        )

    # Sanity: the planted boxiness must show up with the right sign and scale
    b4_plotted = plotted[r"$b_4$"]
    assert np.median(b4_plotted) < -0.01, f"planted b4 not visible in panel (median {np.median(b4_plotted)})"

    plt.close("all")


def test_p4_nan_stop_code_does_not_raise(tmp_path, monkeypatch):
    """P4: an explicit stop_code=NaN row must not crash astype(int)."""
    image, cx, cy = _make_boxy_image()
    config = IsosterConfig(x0=cx, y0=cy, eps=0.2, pa=0.0, sma0=8.0, minsma=2.0, maxsma=25.0)
    isophotes = fit_image(image, config=config)["isophotes"]
    isophotes[3]["stop_code"] = np.nan  # explicit NaN (missing keys were already fine)
    model = build_isoster_model(image.shape, isophotes)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    plot_qa_summary("NaN stop code", image, model, isophotes, filename=str(tmp_path / "qa.png"))
    assert (tmp_path / "qa.png").exists()
    plt.close("all")


def test_p4_asinh_softening_shared_with_photutils_overlay(tmp_path, monkeypatch):
    """P4: the photutils overlay must share isoster's asinh scale.

    Each transform_sb_profile call used to resolve its own softening from
    its own intensities, putting the two series on different asinh scales.
    """
    from types import SimpleNamespace

    import isoster.plotting as plotting_mod

    image, cx, cy = _make_boxy_image()
    config = IsosterConfig(x0=cx, y0=cy, eps=0.2, pa=0.0, sma0=8.0, minsma=2.0, maxsma=25.0)
    isophotes = fit_image(image, config=config)["isophotes"]
    model = build_isoster_model(image.shape, isophotes)
    phot_res = [
        SimpleNamespace(sma=iso["sma"], intens=iso["intens"] * 1.5, int_err=iso["intens_err"]) for iso in isophotes
    ]

    softenings = []
    real_transform = plotting_mod.transform_sb_profile

    def spy(*args, **kwargs):
        softenings.append(kwargs.get("sb_asinh_softening"))
        return real_transform(*args, **kwargs)

    monkeypatch.setattr(plotting_mod, "transform_sb_profile", spy)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    plot_qa_summary(
        "shared asinh",
        image,
        model,
        isophotes,
        photutils_res=phot_res,
        filename=str(tmp_path / "qa.png"),
        sb_profile_scale="asinh",
    )
    plt.close("all")

    assert len(softenings) >= 2, "expected isoster + photutils transform calls"
    assert all(s is not None and s > 0 for s in softenings)
    assert len(set(softenings)) == 1, f"per-series softening detected: {softenings}"
