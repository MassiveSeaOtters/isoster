"""
Tests for FITS and ASDF I/O round-trip, HDU layout, backward compatibility,
and config recovery.
"""

import json
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from isoster.config import IsosterConfig
from isoster.utils import (
    _build_config_hdu,
    isophote_results_from_asdf,
    isophote_results_from_fits,
    isophote_results_to_asdf,
    isophote_results_to_fits,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_results(config_kwargs=None):
    """Build a minimal results dict for testing I/O."""
    config = IsosterConfig(**(config_kwargs or {"x0": 32.0, "y0": 32.0, "sma0": 10.0}))
    isophotes = [
        {
            "sma": 5.0,
            "intens": 100.0,
            "intens_err": 1.0,
            "eps": 0.2,
            "pa": 0.5,
            "x0": 32.0,
            "y0": 32.0,
            "rms": 2.0,
            "stop_code": 0,
            "niter": 15,
        },
        {
            "sma": 10.0,
            "intens": 50.0,
            "intens_err": 0.8,
            "eps": 0.25,
            "pa": 0.6,
            "x0": 32.1,
            "y0": 31.9,
            "rms": 1.5,
            "stop_code": 0,
            "niter": 12,
        },
        {
            "sma": 15.0,
            "intens": 20.0,
            "intens_err": 1.5,
            "eps": 0.3,
            "pa": 0.7,
            "x0": 32.0,
            "y0": 32.0,
            "rms": 3.0,
            "stop_code": 2,
            "niter": 50,
        },
    ]
    return {"isophotes": isophotes, "config": config}


# ---------------------------------------------------------------------------
# FITS tests
# ---------------------------------------------------------------------------


class TestFitsWriter:
    """Tests for the FITS writer (PRIMARY / ISOPHOTES / CONFIG / META)."""

    def test_no_hierarch_warnings(self, tmp_path):
        """Writing results must produce zero VerifyWarning (HIERARCH noise)."""
        results = _make_results()
        fpath = str(tmp_path / "test.fits")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            isophote_results_to_fits(results, fpath)

        verify_warnings = [w for w in caught if issubclass(w.category, fits.verify.VerifyWarning)]
        assert len(verify_warnings) == 0, f"Got {len(verify_warnings)} VerifyWarning(s): " + "; ".join(
            str(w.message) for w in verify_warnings
        )

    def test_hdu_structure(self, tmp_path):
        """Output FITS has 4 HDUs: PRIMARY, ISOPHOTES, CONFIG, META."""
        results = _make_results()
        fpath = str(tmp_path / "test.fits")
        isophote_results_to_fits(results, fpath)

        with fits.open(fpath) as hdulist:
            assert len(hdulist) == 4
            assert hdulist[0].name == "PRIMARY"
            assert hdulist[1].name == "ISOPHOTES"
            assert hdulist[2].name == "CONFIG"
            assert hdulist[3].name == "META"

    def test_round_trip_with_config_recovery(self, tmp_path):
        """Write → read recovers isophotes and an IsosterConfig with matching values."""
        results = _make_results({"x0": 50.0, "y0": 60.0, "sma0": 8.0, "eps": 0.35, "pa": 1.0})
        fpath = str(tmp_path / "test.fits")
        isophote_results_to_fits(results, fpath)

        loaded = isophote_results_from_fits(fpath)
        assert len(loaded["isophotes"]) == len(results["isophotes"])
        assert isinstance(loaded["config"], IsosterConfig)
        assert loaded["config"].x0 == pytest.approx(50.0)
        assert loaded["config"].y0 == pytest.approx(60.0)
        assert loaded["config"].sma0 == pytest.approx(8.0)
        assert loaded["config"].eps == pytest.approx(0.35)

    def test_isophote_data_fidelity(self, tmp_path):
        """Isophote column values survive the round-trip."""
        results = _make_results()
        fpath = str(tmp_path / "test.fits")
        isophote_results_to_fits(results, fpath)
        loaded = isophote_results_from_fits(fpath)

        for orig, loaded_iso in zip(results["isophotes"], loaded["isophotes"]):
            assert loaded_iso["sma"] == pytest.approx(orig["sma"])
            assert loaded_iso["intens"] == pytest.approx(orig["intens"])
            assert loaded_iso["stop_code"] == orig["stop_code"]

    def test_complex_config_values(self, tmp_path):
        """Lists and dicts in config survive JSON round-trip."""
        results = _make_results(
            {
                "x0": 32.0,
                "y0": 32.0,
                "sma0": 10.0,
                "harmonic_orders": [3, 4, 5],
                "central_reg_weights": {"eps": 2.0, "pa": 0.5, "center": 1.0},
            }
        )
        fpath = str(tmp_path / "test.fits")
        isophote_results_to_fits(results, fpath)
        loaded = isophote_results_from_fits(fpath)

        assert loaded["config"].harmonic_orders == [3, 4, 5]
        assert loaded["config"].central_reg_weights == {"eps": 2.0, "pa": 0.5, "center": 1.0}

    def test_none_config_values(self, tmp_path):
        """Config fields with None values round-trip correctly."""
        results = _make_results({"x0": None, "y0": None, "sma0": 10.0, "maxsma": None})
        fpath = str(tmp_path / "test.fits")
        isophote_results_to_fits(results, fpath)
        loaded = isophote_results_from_fits(fpath)

        assert loaded["config"].x0 is None
        assert loaded["config"].y0 is None
        assert loaded["config"].maxsma is None

    def test_overwrite(self, tmp_path):
        """Overwrite=True replaces existing file without error."""
        results = _make_results()
        fpath = str(tmp_path / "test.fits")
        isophote_results_to_fits(results, fpath)
        isophote_results_to_fits(results, fpath, overwrite=True)

    def test_empty_isophotes(self, tmp_path):
        """Empty isophote list still produces valid FITS."""
        results = {"isophotes": [], "config": IsosterConfig(sma0=10.0)}
        fpath = str(tmp_path / "test.fits")
        isophote_results_to_fits(results, fpath)

        with fits.open(fpath) as hdulist:
            assert len(hdulist) == 4
            assert hdulist[2].name == "CONFIG"
            assert hdulist[3].name == "META"


class TestFitsReaderBackwardCompat:
    """Test that the reader handles legacy FITS files (config in header)."""

    def _write_legacy_fits(self, fpath, isophotes, config_dict):
        """Simulate old-format FITS: single table HDU with config in meta."""
        tbl = Table(rows=isophotes)
        for key, value in config_dict.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                tbl.meta[key] = value
            else:
                tbl.meta[key] = str(value)
        tbl.write(fpath, format="fits", overwrite=True)

    def test_legacy_returns_isophotes_and_none_config(self, tmp_path):
        """Legacy file (no CONFIG HDU) returns isophotes with config=None."""
        isophotes = [
            {"sma": 5.0, "intens": 100.0, "eps": 0.2, "pa": 0.5, "x0": 32.0, "y0": 32.0, "stop_code": 0, "niter": 10},
        ]
        fpath = str(tmp_path / "legacy.fits")
        self._write_legacy_fits(fpath, isophotes, {"sma0": 10.0, "eps": 0.2})

        loaded = isophote_results_from_fits(fpath)
        assert len(loaded["isophotes"]) == 1
        assert loaded["config"] is None
        assert loaded["isophotes"][0]["sma"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# ASDF tests
# ---------------------------------------------------------------------------


class TestAsdf:
    """Tests for ASDF I/O."""

    def test_round_trip(self, tmp_path):
        """Write → read recovers isophotes and config."""
        results = _make_results({"x0": 40.0, "y0": 45.0, "sma0": 12.0})
        fpath = str(tmp_path / "test.asdf")
        isophote_results_to_asdf(results, fpath)
        loaded = isophote_results_from_asdf(fpath)

        assert len(loaded["isophotes"]) == len(results["isophotes"])
        for orig, loaded_iso in zip(results["isophotes"], loaded["isophotes"]):
            assert loaded_iso["sma"] == pytest.approx(orig["sma"])
            assert loaded_iso["intens"] == pytest.approx(orig["intens"])

    def test_config_recovery(self, tmp_path):
        """Config is reconstructed as IsosterConfig from ASDF."""
        results = _make_results(
            {
                "x0": 40.0,
                "y0": 45.0,
                "sma0": 12.0,
                "harmonic_orders": [3, 4, 5],
            }
        )
        fpath = str(tmp_path / "test.asdf")
        isophote_results_to_asdf(results, fpath)
        loaded = isophote_results_from_asdf(fpath)

        assert isinstance(loaded["config"], IsosterConfig)
        assert loaded["config"].x0 == pytest.approx(40.0)
        assert loaded["config"].harmonic_orders == [3, 4, 5]

    def test_import_guard(self, monkeypatch, tmp_path):
        """ImportError with helpful message when asdf is not installed."""
        # Monkeypatch __import__ to block asdf
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "asdf":
                raise ImportError("No module named 'asdf'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="asdf"):
            isophote_results_to_asdf({"isophotes": []}, str(tmp_path / "x.asdf"))

        with pytest.raises(ImportError, match="asdf"):
            isophote_results_from_asdf(str(tmp_path / "x.asdf"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestResolveTemplateAsdf:
    """Test that _resolve_template dispatches on .asdf extension."""

    def test_resolve_template_from_asdf(self, tmp_path):
        """_resolve_template loads isophotes from an ASDF file path."""
        from isoster.driver import _resolve_template

        results = _make_results()
        fpath = str(tmp_path / "template.asdf")
        isophote_results_to_asdf(results, fpath)

        resolved = _resolve_template(fpath)
        assert len(resolved) == len(results["isophotes"])
        # Should be sorted by SMA
        sma_values = [r["sma"] for r in resolved]
        assert sma_values == sorted(sma_values)


class TestConfigHduHelpers:
    """Tests for _build_config_hdu / _parse_config_hdu."""

    def test_numpy_scalar_encoding(self):
        """Numpy scalars in config dict are JSON-serializable."""
        results = {
            "config": {
                "x0": np.float64(32.5),
                "sma0": np.int64(10),
                "fix_center": np.bool_(True),
            }
        }
        hdu = _build_config_hdu(results)
        tbl = Table.read(hdu)
        # Verify values round-trip through JSON
        for row in tbl:
            json.loads(row["VALUE"])  # should not raise

    def test_empty_config(self):
        """Results with no config produce a valid (empty) CONFIG HDU."""
        hdu = _build_config_hdu({"config": None})
        assert hdu.name == "CONFIG"
        tbl = Table.read(hdu)
        assert len(tbl) == 0
