"""Demo-only figure controls must preserve measurements and existing defaults."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from benchmarks.exhausted.plotting.individual_qa_demo import (
    cross_tool,
    differences,
    metric_matrix,
    residual_thumbnails,
    statistics_table,
)
from isoster.plotting import build_method_profile, plot_comparison_qa_figure, transform_sb_profile


def test_missing_reference_and_failed_measurements_are_not_substituted():
    profile = dict(sma=np.array([1.0, 2.0, 3.0, 4.0]), intens=np.array([3.0, 6.0, 9.0, 12.0]))
    reference = dict(sma=np.array([2.0, 3.0]), intens=np.array([4.0, 6.0]))
    assert np.isnan(differences(profile, None, "intens")).all()
    np.testing.assert_allclose(differences(profile, reference, "intens"), [np.nan, 50, 50, np.nan])
    rows = pd.DataFrame(
        [
            dict(
                tool="autoprof",
                arm="baseline",
                status="error",
                flags="",
                wall_time_fit_s=np.nan,
                truth_relative_rms_all=99.0,
                flux_bias_all=99.0,
                native_max_sma_pix=99.0,
            )
        ]
    )
    fig, axis = plt.subplots()
    table = statistics_table(axis, rows, cross_tool=True)
    assert table[1, 1].get_text().get_text() == "ERROR"
    assert all(table[1, column].get_text().get_text() == "—" for column in range(2, 6))
    plt.close(fig)


def test_metric_matrix_and_residual_thumbnails_preserve_values_and_failures():
    rows = pd.DataFrame(
        [
            dict(
                tool="isoster",
                arm=arm,
                status=status,
                flags="",
                wall_time_fit_s=runtime,
                truth_relative_rms_all=0.02,
                flux_bias_all=bias,
                native_max_sma_pix=20.0,
            )
            for arm, status, runtime, bias in [
                ("ref_default", "ok", 1.0, -0.03),
                ("other", "ok", 2.0, 0.04),
                ("failed", "error", 99.0, 99.0),
            ]
        ]
    )
    fig, axis = plt.subplots()
    values = metric_matrix(axis, rows)
    np.testing.assert_allclose(values[0], [1.0, 2.0, -3.0, 20.0])
    assert np.isnan(values[2]).all()
    assert axis.images[2].get_clim() == (3.0, 4.0)
    np.testing.assert_allclose(axis.images[2].get_array().compressed(), [3.0, 4.0])
    for column, artist in enumerate(axis.images):
        low_color = artist.cmap(0.0)
        high_color = artist.cmap(1.0)
        assert (low_color[0] > low_color[2]) == (column != 3)
        assert (high_color[0] > high_color[2]) == (column == 3)
    assert [label.get_text() for label in axis.get_yticklabels()] == rows.arm.tolist()
    assert all(text.get_color() == "black" for text in axis.texts)
    parent = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    image = np.arange(16.0).reshape(4, 4)
    support = np.ones((4, 4), dtype=bool)
    support[0] = False
    models = {("isoster", "ref_default"): image - 1, ("isoster", "other"): image + 2}
    residual_thumbnails(parent, rows, models, image, support)
    images = [child.images[0] for child in parent.child_axes if child.images]
    assert len(images) == 2
    assert len(parent.child_axes) == len(rows)
    assert images[0].get_clim() == images[1].get_clim()
    np.testing.assert_allclose(images[0].get_array().compressed(), 1)
    np.testing.assert_allclose(images[1].get_array().compressed(), -2)
    assert any(text.get_text() == "ERROR" for child in parent.child_axes for text in child.texts)
    plt.close(fig)


@pytest.mark.parametrize("empty", [False, True])
def test_comparison_can_return_unsaved_figure(tmp_path, empty):
    profile = build_method_profile(
        [dict(sma=float(i), intens=10.0 / i, eps=0.2, pa=0.1, x0=8.0, y0=8.0, stop_code=0) for i in range(1, 8)]
    )
    path = tmp_path / "not_saved.png"
    fig = plot_comparison_qa_figure(
        np.ones((16, 16)), {} if empty else {"isoster": profile}, output_path=path, return_figure=True
    )
    assert isinstance(fig, plt.Figure)
    assert not path.exists()
    assert len(next(axis for axis in fig.axes if axis.images).images) == 1
    plt.close(fig)


@pytest.mark.parametrize("missing_model", [False, True])
def test_cross_tool_zone_table_full_residuals_and_data_only_sb_limits(missing_model):
    profile = build_method_profile(
        [dict(sma=float(i), intens=10.0 / i, eps=0.2, pa=0.1, x0=8.0, y0=8.0, stop_code=0) for i in range(1, 8)]
    )
    rows, models, profiles = [], {}, {}
    for tool, arm in [("isoster", "ref_default"), ("photutils", "baseline_median"), ("autoprof", "baseline")]:
        row = dict(
            tool=tool,
            arm=arm,
            primary=True,
            status="ok",
            flags="",
            wall_time_fit_s=1.0,
            native_max_sma_pix=7.0,
            inner_cut_pix=2.0,
        )
        for index, zone in enumerate(("all", "inner", "mid", "outer"), 1):
            row[f"truth_relative_rms_{zone}"] = index / 100
            row[f"flux_bias_{zone}"] = -index / 100
        rows.append(row)
        models[tool, arm] = np.zeros((16, 16))
        models[tool, arm][0, 0] = np.nan
        profiles[tool, arm] = profile
    rows[0]["truth_relative_rms_mid"] = np.nan
    support = np.zeros((16, 16), bool)
    support[4:12, 4:12] = True
    manifest = dict(
        galaxy_id="synthetic",
        sb_zeropoint=27.0,
        pixel_scale_arcsec=0.168,
        image_sigma=dict(image_sigma_adu=0.01),
        initial_geometry=dict(x0=8.0, y0=8.0, eps=0.2, pa=0.1, maxsma=7.0),
    )
    if missing_model:
        del models["isoster", "ref_default"]
        rows[0]["status"] = "unavailable"
    fig = cross_tool(pd.DataFrame(rows), profiles, models, np.ones((16, 16)), support, manifest)
    fig.canvas.draw()
    table_axis = next(axis for axis in fig.axes if axis.tables)
    table = table_axis.tables[0]
    assert table[1, 3].get_text().get_text() == ("—" if missing_model else "1.000 / 2.000 / — / 4.000")
    assert table[1, 4].get_text().get_text() == ("—" if missing_model else "-1.000 / -2.000 / -3.000 / -4.000")
    panel_top = max(axis.get_position().y1 for axis in fig.axes if axis is not table_axis)
    assert table_axis.get_position().y0 - panel_top >= 0.0119
    residual_axes = [axis for axis in fig.axes if axis.images][1:]
    assert len(residual_axes) == (2 if missing_model else 3)
    for axis in residual_axes:
        displayed = axis.images[0].get_array()
        assert np.ma.getmaskarray(displayed)[0, 0]
        assert not np.ma.getmaskarray(displayed)[1, 1]
        assert displayed[1, 1] == 1
    sb_axis = next(axis for axis in fig.axes if "mag" in axis.get_ylabel())
    assert any("isoster" in text.get_text() for text in sb_axis.get_legend().get_texts())
    assert not any(line.get_label() == "I=0" for line in sb_axis.lines)
    _, _, _, _, zero = transform_sb_profile(
        profile["intens"],
        None,
        sb_zeropoint=27.0,
        pixel_scale_arcsec=0.168,
        sb_profile_scale="asinh",
        sb_asinh_softening=0.01,
    )
    assert max(sb_axis.get_ylim()) < zero
    plt.close(fig)
