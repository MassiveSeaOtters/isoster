"""Direct public API surface tests for isoster."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.table import Table

from isoster import (
    build_ellipse_model,
    fit_isophote,
    isophote_results_to_astropy_tables,
    plot_qa_summary,
)
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model


def create_gaussian_test_image(shape: tuple[int, int] = (120, 120)) -> np.ndarray:
    """Create a smooth elliptical Gaussian image for API tests."""
    y_index, x_index = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    x0 = shape[1] / 2.0
    y0 = shape[0] / 2.0
    sigma_major = 18.0
    sigma_minor = 13.5
    image = 2000.0 * np.exp(-0.5 * (((x_index - x0) / sigma_major) ** 2 + ((y_index - y0) / sigma_minor) ** 2))
    return image


def create_public_api_isophotes() -> list[dict]:
    """Create compact, valid isophote dictionaries for plotting/table tests."""
    return [
        {
            "sma": 2.0,
            "intens": 1900.0,
            "intens_err": 5.0,
            "eps": 0.25,
            "eps_err": 0.01,
            "pa": 0.2,
            "pa_err": 0.01,
            "x0": 60.0,
            "x0_err": 0.02,
            "y0": 60.0,
            "y0_err": 0.02,
            "rms": 10.0,
            "stop_code": 0,
            "niter": 8,
            "a3": 0.0,
            "b3": 0.0,
            "a4": 0.0,
            "b4": 0.0,
        },
        {
            "sma": 5.0,
            "intens": 1200.0,
            "intens_err": 8.0,
            "eps": 0.25,
            "eps_err": 0.01,
            "pa": 0.2,
            "pa_err": 0.01,
            "x0": 60.0,
            "x0_err": 0.02,
            "y0": 60.0,
            "y0_err": 0.02,
            "rms": 12.0,
            "stop_code": 0,
            "niter": 7,
            "a3": 0.0,
            "b3": 0.0,
            "a4": 0.0,
            "b4": 0.0,
        },
        {
            "sma": 9.0,
            "intens": 650.0,
            "intens_err": 10.0,
            "eps": 0.25,
            "eps_err": 0.01,
            "pa": 0.2,
            "pa_err": 0.01,
            "x0": 60.0,
            "x0_err": 0.02,
            "y0": 60.0,
            "y0_err": 0.02,
            "rms": 14.0,
            "stop_code": 0,
            "niter": 7,
            "a3": 0.0,
            "b3": 0.0,
            "a4": 0.0,
            "b4": 0.0,
        },
    ]


def test_fit_isophote_public_api() -> None:
    """Validate direct fit_isophote API call contract."""
    image = create_gaussian_test_image()
    config = IsosterConfig(
        sma0=10.0,
        minsma=3.0,
        maxsma=40.0,
        astep=0.1,
        eps=0.25,
        pa=0.2,
        conver=0.05,
        minit=5,
        maxit=25,
        fix_center=True,
        fix_eps=True,
        fix_pa=True,
    )
    start_geometry = {"x0": 60.0, "y0": 60.0, "eps": 0.25, "pa": 0.2}

    result = fit_isophote(image, mask=None, sma=10.0, start_geometry=start_geometry, config=config)

    required_keys = {"x0", "y0", "eps", "pa", "sma", "intens", "intens_err", "stop_code", "niter"}
    assert required_keys.issubset(result.keys())
    assert result["sma"] == pytest.approx(10.0)
    assert np.isfinite(result["intens"])
    assert result["stop_code"] in {0, 1, 2, 3, -1}


def test_isophote_results_to_astropy_tables_order_and_shape() -> None:
    """Validate table conversion for dict and list inputs."""
    isophotes = create_public_api_isophotes()
    isophotes[0]["custom_metric"] = 1.23

    table_from_dict = isophote_results_to_astropy_tables({"isophotes": isophotes, "config": {}})
    table_from_list = isophote_results_to_astropy_tables(isophotes)

    assert isinstance(table_from_dict, Table)
    assert isinstance(table_from_list, Table)
    assert len(table_from_dict) == len(isophotes)
    assert len(table_from_list) == len(isophotes)

    expected_prefix = ["sma", "intens", "intens_err", "eps", "pa", "x0", "y0", "rms", "stop_code", "niter"]
    present_prefix = [column_name for column_name in expected_prefix if column_name in table_from_dict.colnames]
    assert table_from_dict.colnames[: len(present_prefix)] == present_prefix
    assert "custom_metric" in table_from_dict.colnames

    empty_table = isophote_results_to_astropy_tables({"isophotes": []})
    assert isinstance(empty_table, Table)
    assert len(empty_table) == 0


def test_plot_qa_summary_writes_output_file(tmp_path) -> None:
    """Validate QA summary plotting API produces an artifact."""
    image = create_gaussian_test_image()
    isophotes = create_public_api_isophotes()
    model = build_isoster_model(image.shape, isophotes, use_harmonics=False)
    output_path = tmp_path / "qa_summary.png"

    plot_qa_summary(
        title="api-qa-summary",
        image=image,
        isoster_model=model,
        isoster_res=isophotes,
        photutils_res=None,
        mask=None,
        filename=str(output_path),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_ellipse_model_deprecated_alias_behavior() -> None:
    """Guard compatibility behavior of deprecated build_ellipse_model API."""
    isophotes = create_public_api_isophotes()
    shape = (120, 120)
    current_model = build_isoster_model(shape, isophotes)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_model = build_ellipse_model(shape, isophotes)
    assert any(
        isinstance(item.message, DeprecationWarning) and "build_ellipse_model() is deprecated" in str(item.message)
        for item in caught_warnings
    )

    assert np.allclose(current_model, legacy_model, equal_nan=True)


def test_fit_image_returns_expected_keys() -> None:
    """fit_image() result dict must have 'isophotes' and 'config' keys with documented structure."""
    from isoster.driver import fit_image as driver_fit_image

    image = create_gaussian_test_image()
    config = IsosterConfig(
        x0=60.0,
        y0=60.0,
        sma0=10.0,
        minsma=5.0,
        maxsma=30.0,
        eps=0.25,
        pa=0.2,
        fix_center=True,
        fix_eps=True,
        fix_pa=True,
    )
    result = driver_fit_image(image, config=config)

    # Top-level keys
    assert "isophotes" in result, "Result must contain 'isophotes' key"
    assert "config" in result, "Result must contain 'config' key"
    assert isinstance(result["isophotes"], list)
    assert isinstance(result["config"], IsosterConfig)

    # Per-isophote key set
    required_iso_keys = {
        "sma",
        "intens",
        "intens_err",
        "eps",
        "pa",
        "x0",
        "y0",
        "rms",
        "stop_code",
        "niter",
    }
    for iso in result["isophotes"]:
        missing = required_iso_keys - set(iso.keys())
        assert not missing, f"Isophote at sma={iso.get('sma')} missing keys: {missing}"


def test_isophote_results_fits_round_trip(tmp_path) -> None:
    """Write isophotes to FITS and read back; values should match."""
    from isoster.utils import isophote_results_from_fits, isophote_results_to_fits

    isophotes = create_public_api_isophotes()
    fits_path = str(tmp_path / "round_trip.fits")

    isophote_results_to_fits({"isophotes": isophotes, "config": IsosterConfig()}, fits_path)
    loaded = isophote_results_from_fits(fits_path)

    assert len(loaded["isophotes"]) == len(isophotes)
    for orig, loaded_iso in zip(isophotes, loaded["isophotes"]):
        assert loaded_iso["sma"] == pytest.approx(orig["sma"])
        assert loaded_iso["intens"] == pytest.approx(orig["intens"], rel=1e-5)
        assert loaded_iso["stop_code"] == orig["stop_code"]


def test_optimize_facade_importable() -> None:
    """The isoster.optimize module should be importable (it is a facade)."""
    import importlib

    mod = importlib.import_module("isoster.optimize")
    assert mod is not None
    # The facade re-exports the driver/fitting APIs (review R1)
    for name in ("fit_image", "fit_isophote", "compute_deviations", "compute_gradient", "extract_isophote_data"):
        assert hasattr(mod, name), f"isoster.optimize facade missing {name}"
