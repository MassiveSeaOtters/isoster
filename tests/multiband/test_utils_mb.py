"""Tests for ``isoster.multiband.utils_mb``."""

import numpy as np
import pytest
from astropy.io import fits

from isoster.multiband import (
    IsosterConfigMB,
    fit_image_multiband,
    isophote_results_mb_from_asdf,
    isophote_results_mb_from_fits,
    isophote_results_mb_to_asdf,
    isophote_results_mb_to_astropy_table,
    isophote_results_mb_to_fits,
    load_bands_from_hdus,
)

# ---------------------------------------------------------------------------
# Helpers (lifted from test_driver_mb to avoid cross-fixture coupling)
# ---------------------------------------------------------------------------


def _planted(amp: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = w = 192
    x0 = y0 = 96.0
    eps, pa = 0.3, 0.5
    re = 25.0
    n = 1.5
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = x - x0, y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    bn = 2.0 * n - 0.327
    img = amp * np.exp(-bn * ((r / re) ** (1.0 / n) - 1.0))
    img += rng.normal(0.0, 0.05, size=img.shape)
    return img


def _two_band_fit() -> dict:
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=50.0,
        debug=True,
        compute_deviations=True,
        nclip=0,
    )
    return fit_image_multiband([_planted(100.0, 1), _planted(200.0, 2)], None, cfg)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_to_table_has_per_band_and_shared_columns():
    result = _two_band_fit()
    tbl = isophote_results_mb_to_astropy_table(result)
    cols = set(tbl.colnames)
    # Shared
    for shared in ("sma", "x0", "y0", "eps", "pa", "stop_code", "niter", "valid"):
        assert shared in cols, f"missing shared column {shared}"
    # Per-band suffixes
    for b in ("g", "r"):
        for col in (f"intens_{b}", f"intens_err_{b}", f"rms_{b}", f"a3_{b}", f"b3_{b}", f"a4_{b}", f"b4_{b}"):
            assert col in cols, f"missing per-band column {col}"


def test_fits_roundtrip_preserves_per_band_columns(tmp_path):
    result = _two_band_fit()
    fname = tmp_path / "mb_result.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)

    assert loaded["multiband"] is True
    assert loaded["bands"] == ["g", "r"]
    assert loaded["reference_band"] == "g"
    assert loaded["harmonic_combination"] == "joint"
    assert loaded["variance_mode"] == "ols"
    assert loaded["band_weights"] == {"g": 1.0, "r": 1.0}

    # Same number of isophotes preserved.
    assert len(loaded["isophotes"]) == len(result["isophotes"])
    # Compare per-row per-band intensity columns within float tolerance.
    for orig, restored in zip(result["isophotes"], loaded["isophotes"]):
        for col in ("sma", "x0", "y0", "eps", "pa", "intens_g", "intens_r"):
            if col in orig and orig[col] is not None:
                if isinstance(orig[col], float) and np.isnan(orig[col]):
                    assert np.isnan(float(restored[col]))
                else:
                    np.testing.assert_allclose(
                        float(restored[col]),
                        float(orig[col]),
                        atol=1e-9,
                        rtol=0,
                    )


def test_fits_roundtrip_loose_validity_preserves_n_valid_columns(tmp_path):
    """D9 backport: per-band ``n_valid_<b>`` columns and the top-level
    ``loose_validity`` key round-trip through Schema 1 FITS."""
    img_g = _planted(100.0, 1)
    img_r = _planted(200.0, 2)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=60.0,
        debug=True,
        nclip=0,
        loose_validity=True,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    fname = tmp_path / "mb_loose.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)

    assert loaded["loose_validity"] is True
    cols = list(loaded["isophotes"][0].keys())
    for b in ("g", "r"):
        assert f"n_valid_{b}" in cols, f"missing n_valid_{b} after round-trip"
    # Numerics survive: each per-band count matches original.
    for orig, restored in zip(result["isophotes"], loaded["isophotes"]):
        for b in ("g", "r"):
            key = f"n_valid_{b}"
            if key in orig:
                assert int(restored[key]) == int(orig[key])


def test_fits_roundtrip_records_multiband_keys_in_primary_header(tmp_path):
    result = _two_band_fit()
    fname = tmp_path / "mb_result.fits"
    isophote_results_mb_to_fits(result, fname)
    with fits.open(fname) as hdulist:
        primary = hdulist[0]
        assert primary.header["MULTIBND"] is True
        assert primary.header["BANDS"] == "g,r"
        assert primary.header["REFBAND"] == "g"
        assert primary.header["HARMCMB"] == "joint"
        assert primary.header["VARMODE"] == "ols"


def test_fits_roundtrip_with_variance_maps_records_wls(tmp_path):
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        astep=0.2,
        maxsma=40.0,
        debug=True,
        nclip=0,
    )
    img_g = _planted(100.0, 1)
    img_r = _planted(200.0, 2)
    var = np.full_like(img_g, 0.05**2)
    result = fit_image_multiband(
        [img_g, img_r],
        None,
        cfg,
        variance_maps=[var, var.copy()],
    )
    fname = tmp_path / "mb_result_wls.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)
    assert loaded["variance_mode"] == "wls"


# ---------------------------------------------------------------------------
# load_bands_from_hdus
# ---------------------------------------------------------------------------


def test_load_bands_from_hdus_basic():
    hdus = [
        fits.PrimaryHDU(np.ones((32, 32))),
        fits.PrimaryHDU(np.ones((32, 32)) * 2),
    ]
    hdus[0].header["FILTER"] = "HSC-G"
    hdus[1].header["FILTER"] = "HSC-R"
    images, masks, var_maps, bands = load_bands_from_hdus(hdus)
    assert bands == ["g", "r"]
    assert len(images) == 2
    np.testing.assert_array_equal(images[1], np.ones((32, 32)) * 2)
    assert masks == [None, None]
    assert var_maps is None


def test_load_bands_from_hdus_handles_underscore_filters():
    hdus = [fits.PrimaryHDU(np.ones((8, 8)))]
    hdus[0].header["FILTER"] = "z_band"
    _, _, _, bands = load_bands_from_hdus(hdus)
    assert bands == ["z_band"]


def test_load_bands_from_hdus_missing_filter_rejected():
    hdus = [fits.PrimaryHDU(np.ones((8, 8)))]
    with pytest.raises(KeyError, match="FILTER"):
        load_bands_from_hdus(hdus)


# ---------------------------------------------------------------------------
# Stage-3 Stage-D: compute_cog Schema-1 round-trip
# ---------------------------------------------------------------------------


def test_fits_roundtrip_preserves_compute_cog_columns(tmp_path):
    """compute_cog=True writes per-band ``cog_<b>`` and shared
    ``area_annulus`` / ``flag_*`` columns; round-trip preserves them."""
    img_g = _planted(100.0, 11)
    img_r = _planted(200.0, 12)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=12.0,
        eps=0.2,
        pa=0.3,
        astep=0.2,
        maxsma=60.0,
        compute_cog=True,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    fname = tmp_path / "mb_cog.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)

    assert len(loaded["isophotes"]) == len(result["isophotes"])
    # Verify per-row per-band cog columns + shared geometry-derived
    # columns survive the FITS round-trip.
    for orig, restored in zip(result["isophotes"], loaded["isophotes"]):
        for col in ("cog_g", "cog_r", "cog_annulus_g", "cog_annulus_r", "area_annulus"):
            assert col in orig
            assert col in restored
            np.testing.assert_allclose(
                float(restored[col]),
                float(orig[col]),
                atol=1e-9,
                rtol=0,
            )
        for col in ("flag_cross", "flag_negative_area"):
            assert col in orig
            assert col in restored
            assert bool(restored[col]) == bool(orig[col])


# ---------------------------------------------------------------------------
# Stage I: ASDF round-trip
# ---------------------------------------------------------------------------

# Skip the entire ASDF block if asdf isn't installed in the runner's venv.
asdf = pytest.importorskip("asdf", reason="ASDF is an optional dependency")


def test_asdf_roundtrip_preserves_per_band_columns(tmp_path):
    """Mirror of the FITS Schema-1 round-trip: per-band intens / harmonic
    columns and the multi-band top-level keys survive the ASDF write→read."""
    result = _two_band_fit()
    fname = tmp_path / "mb_result.asdf"
    isophote_results_mb_to_asdf(result, fname)
    loaded = isophote_results_mb_from_asdf(fname)

    assert loaded["multiband"] is True
    assert loaded["bands"] == ["g", "r"]
    assert loaded["reference_band"] == "g"
    assert loaded["harmonic_combination"] == "joint"
    assert loaded["variance_mode"] == "ols"
    assert loaded["band_weights"] == {"g": 1.0, "r": 1.0}

    assert len(loaded["isophotes"]) == len(result["isophotes"])
    for orig, restored in zip(result["isophotes"], loaded["isophotes"]):
        for col in (
            "sma",
            "x0",
            "y0",
            "eps",
            "pa",
            "intens_g",
            "intens_r",
            "a3_g",
            "b3_g",
            "a4_g",
            "b4_g",
            "a3_r",
            "b3_r",
            "a4_r",
            "b4_r",
        ):
            if col in orig and orig[col] is not None:
                if isinstance(orig[col], float) and np.isnan(orig[col]):
                    assert np.isnan(float(restored[col]))
                else:
                    np.testing.assert_allclose(
                        float(restored[col]),
                        float(orig[col]),
                        atol=1e-9,
                        rtol=0,
                    )


def test_asdf_roundtrip_recovers_isoster_config_mb(tmp_path):
    """Config dict serialized via ``model_dump`` reconstructs into an
    :class:`IsosterConfigMB` after ASDF round-trip."""
    result = _two_band_fit()
    fname = tmp_path / "mb_result.asdf"
    isophote_results_mb_to_asdf(result, fname)
    loaded = isophote_results_mb_from_asdf(fname)

    assert isinstance(loaded["config"], IsosterConfigMB)
    assert list(loaded["config"].bands) == ["g", "r"]
    assert loaded["config"].reference_band == "g"


def test_asdf_roundtrip_preserves_compute_cog_columns(tmp_path):
    """Per-band ``cog_<b>`` / shared ``area_annulus`` / ``flag_*`` columns
    written under ``compute_cog=True`` round-trip through ASDF."""
    img_g = _planted(100.0, 11)
    img_r = _planted(200.0, 12)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=12.0,
        eps=0.2,
        pa=0.3,
        astep=0.2,
        maxsma=60.0,
        compute_cog=True,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    fname = tmp_path / "mb_cog.asdf"
    isophote_results_mb_to_asdf(result, fname)
    loaded = isophote_results_mb_from_asdf(fname)

    assert len(loaded["isophotes"]) == len(result["isophotes"])
    for orig, restored in zip(result["isophotes"], loaded["isophotes"]):
        for col in ("cog_g", "cog_r", "cog_annulus_g", "cog_annulus_r", "area_annulus"):
            assert col in restored
            np.testing.assert_allclose(
                float(restored[col]),
                float(orig[col]),
                atol=1e-9,
                rtol=0,
            )
        for col in ("flag_cross", "flag_negative_area"):
            assert col in restored
            assert bool(restored[col]) == bool(orig[col])


def test_asdf_roundtrip_preserves_lsb_auto_lock_metadata(tmp_path):
    """When ``lsb_auto_lock`` triggers, the top-level lock keys and the
    per-row ``lsb_locked`` / ``lsb_auto_lock_anchor`` flags are preserved
    through an ASDF round-trip."""
    import warnings as _warnings

    rng_h = rng_w = 200
    y, x = np.mgrid[0:rng_h, 0:rng_w].astype(np.float64)
    rng = np.random.default_rng(2026)
    img1 = 50.0 * np.exp(-((x - 100.0) ** 2 + ((y - 100.0) / 0.75) ** 2) / (2 * 10.0**2))
    img1 = img1 + rng.normal(0.0, 2.0, size=img1.shape)
    rng = np.random.default_rng(2027)
    img2 = 25.0 * np.exp(-((x - 100.0) ** 2 + ((y - 100.0) / 0.75) ** 2) / (2 * 10.0**2))
    img2 = img2 + rng.normal(0.0, 2.0, size=img2.shape)

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            sma0=5.0,
            minsma=2.0,
            maxsma=80.0,
            astep=0.15,
            lsb_auto_lock=True,
            lsb_auto_lock_integrator="mean",
            lsb_auto_lock_maxgerr=0.2,
            lsb_auto_lock_debounce=2,
        )
    result = fit_image_multiband([img1, img2], None, cfg)
    if not result.get("lsb_auto_lock"):
        pytest.skip("Lock did not trigger on synthetic image; metadata test inapplicable")

    fname = tmp_path / "mb_lsb.asdf"
    isophote_results_mb_to_asdf(result, fname)
    loaded = isophote_results_mb_from_asdf(fname)

    assert loaded.get("lsb_auto_lock") is True
    assert float(loaded["lsb_auto_lock_sma"]) == pytest.approx(
        float(result["lsb_auto_lock_sma"]),
    )
    assert int(loaded["lsb_auto_lock_count"]) == int(result["lsb_auto_lock_count"])

    orig_locked = [iso for iso in result["isophotes"] if iso.get("lsb_locked")]
    loaded_locked = [iso for iso in loaded["isophotes"] if iso.get("lsb_locked")]
    assert len(loaded_locked) == len(orig_locked)
    orig_anchors = sum(1 for iso in result["isophotes"] if iso.get("lsb_auto_lock_anchor"))
    loaded_anchors = sum(1 for iso in loaded["isophotes"] if iso.get("lsb_auto_lock_anchor"))
    assert loaded_anchors == orig_anchors


def test_fits_roundtrip_preserves_lsb_auto_lock_flags(tmp_path):
    """Regression test for B3 (multiband): lock flags must survive FITS exactly.

    FITS logical columns cannot represent masked values, so heterogeneous row
    keys made numpy fill the missing cells with True: every isophote read back
    as locked and as the anchor. With the uniform schema (False stamped on
    inward/central rows) the flags must round-trip exactly.
    """
    import warnings as _warnings

    rng_h = rng_w = 200
    y, x = np.mgrid[0:rng_h, 0:rng_w].astype(np.float64)
    rng = np.random.default_rng(2026)
    img1 = 50.0 * np.exp(-((x - 100.0) ** 2 + ((y - 100.0) / 0.75) ** 2) / (2 * 10.0**2))
    img1 = img1 + rng.normal(0.0, 2.0, size=img1.shape)
    rng = np.random.default_rng(2027)
    img2 = 25.0 * np.exp(-((x - 100.0) ** 2 + ((y - 100.0) / 0.75) ** 2) / (2 * 10.0**2))
    img2 = img2 + rng.normal(0.0, 2.0, size=img2.shape)

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            sma0=5.0,
            minsma=2.0,
            maxsma=80.0,
            astep=0.15,
            lsb_auto_lock=True,
            lsb_auto_lock_integrator="mean",
            lsb_auto_lock_maxgerr=0.2,
            lsb_auto_lock_debounce=2,
        )
    result = fit_image_multiband([img1, img2], None, cfg)
    if not result.get("lsb_auto_lock"):
        pytest.skip("Lock did not trigger on synthetic image; metadata test inapplicable")

    fname = tmp_path / "mb_lsb.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)

    assert len(loaded["isophotes"]) == len(result["isophotes"])
    for orig, restored in zip(result["isophotes"], loaded["isophotes"]):
        assert bool(restored["lsb_locked"]) == bool(orig["lsb_locked"])
        assert bool(restored["lsb_auto_lock_anchor"]) == bool(orig["lsb_auto_lock_anchor"])
    assert sum(1 for iso in loaded["isophotes"] if iso["lsb_auto_lock_anchor"]) == 1

    # Q2: the true geometry anchor's sma is recorded and round-trips
    anchor_sma = result["lsb_auto_lock_anchor_sma"]
    assert anchor_sma is not None and float(anchor_sma) < float(result["lsb_auto_lock_sma"])
    assert float(loaded["lsb_auto_lock_anchor_sma"]) == pytest.approx(float(anchor_sma))


def test_asdf_import_guard(monkeypatch, tmp_path):
    """Helpful ImportError surfaces when the optional ``asdf`` package
    is unavailable at call time."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def mock_import(name, *args, **kwargs):
        if name == "asdf":
            raise ImportError("No module named 'asdf'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ImportError, match="asdf"):
        isophote_results_mb_to_asdf({"isophotes": []}, str(tmp_path / "x.asdf"))

    with pytest.raises(ImportError, match="asdf"):
        isophote_results_mb_from_asdf(str(tmp_path / "x.asdf"))


def test_fits_roundtrip_preserves_extra_top_level_keys(tmp_path):
    """M13: the FITS META HDU preserves non-structural top-level result keys
    (forced-mode bookkeeping, sky offsets, first-isophote diagnostics)."""
    result = _two_band_fit()
    result["forced_photometry_mode"] = True
    result["template_n_isophotes"] = 7
    result["sky_offsets"] = {"g": 1.25, "r": -0.5}
    result["first_isophote_failure"] = False

    fname = tmp_path / "mb_meta.fits"
    isophote_results_mb_to_fits(result, fname)
    loaded = isophote_results_mb_from_fits(fname)

    assert loaded["forced_photometry_mode"] is True
    assert loaded["template_n_isophotes"] == 7
    assert loaded["sky_offsets"] == {"g": 1.25, "r": -0.5}
    assert loaded["first_isophote_failure"] is False


def test_asdf_roundtrip_preserves_extra_top_level_keys(tmp_path):
    """M13: the ASDF tree preserves non-structural top-level result keys."""
    result = _two_band_fit()
    result["forced_photometry_mode"] = True
    result["template_n_isophotes"] = 7
    result["sky_offsets"] = {"g": 1.25, "r": -0.5}

    fname = tmp_path / "mb_meta.asdf"
    isophote_results_mb_to_asdf(result, fname)
    loaded = isophote_results_mb_from_asdf(fname)

    assert loaded["forced_photometry_mode"] is True
    assert loaded["template_n_isophotes"] == 7
    assert loaded["sky_offsets"] == {"g": 1.25, "r": -0.5}
