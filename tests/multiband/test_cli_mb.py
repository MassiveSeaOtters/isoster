"""Smoke tests for the parallel multi-band CLI ``isoster-mb``.

Stage J locks the CLI as a separate code path from the single-band
``isoster`` entry point (no shared ``main()``). The tests below exercise
the public surface end-to-end: positional images, --bands resolution,
--config YAML loading, output writers (csv / fits / asdf), the
experimental banner, the --template forced-photometry flow, and the
common error paths (missing bands, mismatched counts, mismatched
--reference-band).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.table import Table

from isoster.multiband.cli_mb import main


def _planted_image(amplitude: float = 100.0, seed: int = 0) -> np.ndarray:
    """Compact Sersic-ish image suitable for fast end-to-end CLI runs."""
    rng = np.random.default_rng(seed)
    h = w = 96
    x0 = y0 = 48.0
    eps, pa = 0.2, 0.4
    re = 12.0
    n = 1.5
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = x - x0, y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    bn = 2.0 * n - 0.327
    img = amplitude * np.exp(-bn * ((r / re) ** (1.0 / n) - 1.0))
    img += rng.normal(0.0, 0.05, size=img.shape)
    return img


def _write_fits(path, data):
    fits.writeto(str(path), data, overwrite=True)


def _basic_config_dict():
    return {
        "sma0": 8.0,
        "minsma": 2.0,
        "maxsma": 32.0,
        "astep": 0.2,
        "eps": 0.2,
        "pa": 0.4,
        "minit": 6,
        "maxit": 30,
        "conver": 0.05,
        "fix_center": True,
        "compute_deviations": True,
        "nclip": 0,
    }


def _argv(*items):
    return ["isoster-mb", *items]


# ---------------------------------------------------------------------------
# CLI smoke: csv / fits / asdf
# ---------------------------------------------------------------------------


def test_cli_smoke_fits_output(tmp_path, monkeypatch, capsys):
    img_g = tmp_path / "image_g.fits"
    img_r = tmp_path / "image_r.fits"
    cfg = tmp_path / "config.yaml"
    out = tmp_path / "out.fits"

    _write_fits(img_g, _planted_image(100.0, 1))
    _write_fits(img_r, _planted_image(200.0, 2))
    cfg.write_text(yaml.safe_dump(_basic_config_dict()))

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img_g),
            str(img_r),
            "--bands",
            "g",
            "r",
            "--reference-band",
            "g",
            "--config",
            str(cfg),
            "--output",
            str(out),
        ),
    )
    main()

    assert out.exists()
    # The interface-stability banner appears on stderr by default.
    captured = capsys.readouterr()
    assert "UNSTABLE INTERFACE" in captured.err
    # Schema-1 sanity: PrimaryHDU MULTIBND header + per-band intensity columns.
    with fits.open(out) as hdul:
        assert hdul[0].header["MULTIBND"] is True
        assert hdul[0].header["BANDS"] == "g,r"
        assert "ISOPHOTES" in hdul
        cols = set(hdul["ISOPHOTES"].columns.names)
    for col in ("sma", "intens_g", "intens_r"):
        assert col in cols


def test_cli_smoke_csv_output(tmp_path, monkeypatch):
    img_g = tmp_path / "image_g.fits"
    img_r = tmp_path / "image_r.fits"
    cfg = tmp_path / "config.yaml"
    out = tmp_path / "out.csv"

    _write_fits(img_g, _planted_image(100.0, 3))
    _write_fits(img_r, _planted_image(200.0, 4))
    cfg.write_text(yaml.safe_dump(_basic_config_dict()))

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img_g),
            str(img_r),
            "--bands",
            "g",
            "r",
            "--reference-band",
            "r",
            "--config",
            str(cfg),
            "--output",
            str(out),
            "--quiet",
        ),
    )
    main()

    assert out.exists()
    table = Table.read(out, format="ascii.csv")
    assert len(table) > 0
    for col in ("sma", "intens_g", "intens_r"):
        assert col in table.colnames


def test_cli_smoke_asdf_output(tmp_path, monkeypatch):
    pytest.importorskip("asdf")
    img_g = tmp_path / "image_g.fits"
    img_r = tmp_path / "image_r.fits"
    cfg = tmp_path / "config.yaml"
    out = tmp_path / "out.asdf"

    _write_fits(img_g, _planted_image(100.0, 5))
    _write_fits(img_r, _planted_image(200.0, 6))
    cfg.write_text(yaml.safe_dump(_basic_config_dict()))

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img_g),
            str(img_r),
            "--bands",
            "g",
            "r",
            "--reference-band",
            "g",
            "--config",
            str(cfg),
            "--output",
            str(out),
            "--quiet",
        ),
    )
    main()

    from isoster.multiband import isophote_results_mb_from_asdf

    loaded = isophote_results_mb_from_asdf(out)
    assert loaded["multiband"] is True
    assert loaded["bands"] == ["g", "r"]
    assert len(loaded["isophotes"]) > 0


# ---------------------------------------------------------------------------
# Banner and quiet flag
# ---------------------------------------------------------------------------


def test_cli_quiet_suppresses_banner(tmp_path, monkeypatch, capsys):
    img_g = tmp_path / "image_g.fits"
    img_r = tmp_path / "image_r.fits"
    out = tmp_path / "out.csv"
    _write_fits(img_g, _planted_image(100.0, 7))
    _write_fits(img_r, _planted_image(200.0, 8))

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img_g),
            str(img_r),
            "--bands",
            "g",
            "r",
            "--reference-band",
            "g",
            "--output",
            str(out),
            "--sma0",
            "8",
            "--quiet",
        ),
    )
    main()
    captured = capsys.readouterr()
    assert "UNSTABLE INTERFACE" not in captured.err


# ---------------------------------------------------------------------------
# YAML-only bands (no --bands flag)
# ---------------------------------------------------------------------------


def test_cli_bands_resolved_from_yaml(tmp_path, monkeypatch):
    img_g = tmp_path / "image_g.fits"
    img_r = tmp_path / "image_r.fits"
    cfg = tmp_path / "config.yaml"
    out = tmp_path / "out.csv"

    _write_fits(img_g, _planted_image(100.0, 9))
    _write_fits(img_r, _planted_image(200.0, 10))
    cfg_dict = _basic_config_dict()
    cfg_dict["bands"] = ["g", "r"]
    cfg_dict["reference_band"] = "g"
    cfg.write_text(yaml.safe_dump(cfg_dict))

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img_g),
            str(img_r),
            "--config",
            str(cfg),
            "--output",
            str(out),
            "--quiet",
        ),
    )
    main()

    assert out.exists()
    table = Table.read(out, format="ascii.csv")
    assert "intens_g" in table.colnames
    assert "intens_r" in table.colnames


# ---------------------------------------------------------------------------
# Forced photometry via --template (Stage H integration)
# ---------------------------------------------------------------------------


def test_cli_template_forced_mode(tmp_path, monkeypatch):
    """--template loads a Schema-1 multi-band FITS template and runs the
    forced-photometry path; the output retains the template's geometry."""
    from isoster.multiband import (
        IsosterConfigMB,
        fit_image_multiband,
        isophote_results_mb_to_fits,
    )

    # Step 1: build a template result by fitting the 'g' band joint with itself.
    img_g = _planted_image(100.0, 21)
    img_r = _planted_image(200.0, 22)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=8.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=32.0,
        minit=6,
        maxit=30,
        conver=0.05,
        fix_center=True,
        compute_deviations=True,
        nclip=0,
    )
    template_result = fit_image_multiband([img_g, img_r], None, cfg)
    template_path = tmp_path / "template.fits"
    isophote_results_mb_to_fits(template_result, template_path)

    # Step 2: write the same images, run the CLI in forced-photometry mode.
    img_g_path = tmp_path / "image_g.fits"
    img_r_path = tmp_path / "image_r.fits"
    cfg_yaml = tmp_path / "config.yaml"
    out = tmp_path / "out.fits"
    _write_fits(img_g_path, img_g)
    _write_fits(img_r_path, img_r)
    cfg_yaml.write_text(yaml.safe_dump(_basic_config_dict()))

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img_g_path),
            str(img_r_path),
            "--bands",
            "g",
            "r",
            "--reference-band",
            "g",
            "--config",
            str(cfg_yaml),
            "--template",
            str(template_path),
            "--output",
            str(out),
            "--quiet",
        ),
    )
    main()

    from isoster.multiband import isophote_results_mb_from_fits

    loaded = isophote_results_mb_from_fits(out)
    # Forced-photometry geometry matches template bit-identically.
    assert len(loaded["isophotes"]) == len(template_result["isophotes"])
    for orig, restored in zip(template_result["isophotes"], loaded["isophotes"]):
        for col in ("sma", "x0", "y0", "eps", "pa"):
            np.testing.assert_allclose(
                float(restored[col]),
                float(orig[col]),
                atol=1e-9,
                rtol=0,
            )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_cli_missing_bands_errors(tmp_path, monkeypatch):
    """No --bands and no YAML ``bands`` aborts with SystemExit."""
    img = tmp_path / "image.fits"
    _write_fits(img, _planted_image(100.0, 31))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(str(img), "--output", str(tmp_path / "out.csv"), "--quiet"),
    )
    with pytest.raises(SystemExit, match="--bands"):
        main()


def test_cli_mismatched_image_count_errors(tmp_path, monkeypatch):
    """Two --bands entries but only one positional image is rejected."""
    img = tmp_path / "image_g.fits"
    _write_fits(img, _planted_image(100.0, 32))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img),
            "--bands",
            "g",
            "r",
            "--output",
            str(tmp_path / "out.csv"),
            "--quiet",
        ),
    )
    with pytest.raises(SystemExit, match="positional images"):
        main()


def test_cli_reference_band_not_in_bands_errors(tmp_path, monkeypatch):
    """--reference-band must appear in --bands."""
    img_g = tmp_path / "image_g.fits"
    img_r = tmp_path / "image_r.fits"
    _write_fits(img_g, _planted_image(100.0, 33))
    _write_fits(img_r, _planted_image(200.0, 34))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img_g),
            str(img_r),
            "--bands",
            "g",
            "r",
            "--reference-band",
            "z",
            "--output",
            str(tmp_path / "out.csv"),
            "--quiet",
        ),
    )
    with pytest.raises(SystemExit, match="reference-band"):
        main()


def test_cli_mask_and_masks_mutually_exclusive(tmp_path, monkeypatch):
    img_g = tmp_path / "image_g.fits"
    img_r = tmp_path / "image_r.fits"
    mask = tmp_path / "mask.fits"
    _write_fits(img_g, _planted_image(100.0, 35))
    _write_fits(img_r, _planted_image(200.0, 36))
    _write_fits(mask, np.zeros_like(_planted_image(0.0, 0), dtype=np.uint8))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            str(img_g),
            str(img_r),
            "--bands",
            "g",
            "r",
            "--reference-band",
            "g",
            "--mask",
            str(mask),
            "--masks",
            str(mask),
            str(mask),
            "--output",
            str(tmp_path / "out.csv"),
            "--quiet",
        ),
    )
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main()
