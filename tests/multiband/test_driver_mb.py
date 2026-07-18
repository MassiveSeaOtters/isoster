"""Tests for ``isoster.multiband.driver_mb.fit_image_multiband``."""

import warnings

import numpy as np
import pytest

from isoster.multiband import IsosterConfigMB, fit_image_multiband

# ---------------------------------------------------------------------------
# Synthetic galaxy fixture (Sersic with shared geometry across bands)
# ---------------------------------------------------------------------------


def _planted_galaxy(
    h: int = 192,
    w: int = 192,
    x0: float = 96.0,
    y0: float = 96.0,
    eps: float = 0.3,
    pa: float = 0.5,
    re: float = 25.0,
    n_sersic: float = 1.5,
    amplitude: float = 1.0,
    noise_sigma: float = 0.005,
    seed: int = 0,
) -> np.ndarray:
    """Sersic in elliptical coordinates: truth (x0, y0, eps, pa)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = x - x0
    dy = y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    bn = 2.0 * n_sersic - 0.327
    img = amplitude * np.exp(-bn * ((r / re) ** (1.0 / n_sersic) - 1.0))
    img += rng.normal(0.0, noise_sigma, size=img.shape)
    return img


@pytest.fixture
def planted_two_band():
    """Two bands with shared geometry, different per-band amplitudes."""
    img_g = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=1)
    img_r = _planted_galaxy(amplitude=200.0, noise_sigma=0.05, seed=2)
    return img_g, img_r


# ---------------------------------------------------------------------------
# B=1 fallback: delegation to single-band fit_image
# ---------------------------------------------------------------------------


def test_b1_delegates_to_single_band(planted_two_band):
    img, _ = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=60.0,
        debug=True,
        nclip=0,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_image_multiband([img], None, cfg)
    assert any("delegating to" in str(w.message) for w in captured)
    # Single-band schema: no `intens_g` column (legacy uses bare `intens`).
    iso0 = result["isophotes"][0]
    assert "intens" in iso0
    assert "intens_g" not in iso0
    # Multi-band top-level keys are absent.
    assert "multiband" not in result
    assert "bands" not in result


def test_m6_b1_delegation_forwards_template(planted_two_band):
    """Regression test for M6: B=1 with a template must run forced photometry.

    The B=1 check ran before the forced-photometry dispatch and the
    delegation never received template_isophotes, so templates were
    silently ignored for single-band inputs.
    """
    img, _ = planted_two_band
    template_cfg = IsosterConfigMB(bands=["g"], reference_band="g", sma0=15.0, astep=0.2, maxsma=40.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        template_res = fit_image_multiband([img], None, template_cfg)

    cfg = IsosterConfigMB(bands=["g"], reference_band="g", sma0=15.0, astep=0.2, maxsma=40.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_image_multiband([img], None, cfg, template_isophotes=template_res["isophotes"])

    # Forced photometry extracts without iterating: niter == 0 on ring rows
    ring_rows = [iso for iso in res["isophotes"] if iso["sma"] > 0]
    assert ring_rows and all(iso["niter"] == 0 for iso in ring_rows)


def test_m6_b1_delegation_forwards_feature_flags(planted_two_band):
    """M6: B=1 delegation must forward compute_cog / harmonic_orders / lsb_auto_lock."""
    img, _ = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g"],
        reference_band="g",
        sma0=15.0,
        astep=0.2,
        maxsma=40.0,
        compute_cog=True,
        harmonic_orders=[3, 4, 5, 6],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_image_multiband([img], None, cfg)
    assert any("cog" in iso for iso in res["isophotes"])
    assert any("a5" in iso for iso in res["isophotes"])

    cfg_lock = IsosterConfigMB(
        bands=["g"],
        reference_band="g",
        sma0=15.0,
        astep=0.2,
        maxsma=40.0,
        lsb_auto_lock=True,
        lsb_auto_lock_integrator="mean",  # MB validator: 'median' needs decoupled intercepts
        debug=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_lock = fit_image_multiband([img], None, cfg_lock)
    assert res_lock.get("lsb_auto_lock") is True
    assert any("lsb_locked" in iso for iso in res_lock["isophotes"])


def test_b1_delegation_unwraps_variance_maps_list(planted_two_band):
    """Regression for B17: a length-1 variance_maps list must be unwrapped
    to a single ndarray when delegating to single-band ``fit_image``.

    The single-band sampler accepts only ``variance_map`` (singular,
    ndarray); passing a list would raise. The multi-band B=1 fallback
    must therefore unwrap the list before delegation.
    """
    img, _ = planted_two_band
    var = np.full_like(img, 0.25, dtype=np.float64)
    cfg = IsosterConfigMB(
        bands=["g"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=60.0,
        debug=True,
        nclip=0,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_image_multiband([img], None, cfg, variance_maps=[var])
    assert any("delegating to" in str(w.message) for w in captured)
    iso = next(
        (i for i in result["isophotes"] if float(i.get("sma", 0.0)) > 0.0),
        None,
    )
    assert iso is not None, "Expected at least one non-central isophote"
    assert "intens" in iso
    assert float(iso.get("intens_err", 0.0)) > 0.0  # WLS propagated errors


def test_b1_delegation_unwraps_variance_maps_tuple(planted_two_band):
    """Length-1 tuples are unwrapped just like lists."""
    img, _ = planted_two_band
    var = np.full_like(img, 0.25, dtype=np.float64)
    cfg = IsosterConfigMB(
        bands=["g"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=60.0,
        debug=True,
        nclip=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_image_multiband([img], None, cfg, variance_maps=(var,))
    iso = next(
        (i for i in result["isophotes"] if float(i.get("sma", 0.0)) > 0.0),
        None,
    )
    assert iso is not None
    assert float(iso.get("intens_err", 0.0)) > 0.0


def test_b1_delegation_unwraps_masks_list(planted_two_band):
    """Length-1 mask list is unwrapped to a single ndarray."""
    img, _ = planted_two_band
    mask = np.zeros_like(img, dtype=bool)
    cfg = IsosterConfigMB(
        bands=["g"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=60.0,
        debug=True,
        nclip=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_image_multiband([img], [mask], cfg)
    assert "isophotes" in result and len(result["isophotes"]) > 0


# ---------------------------------------------------------------------------
# B>=2: end-to-end joint fit on planted galaxy
# ---------------------------------------------------------------------------


def test_two_band_end_to_end_recovers_geometry(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.15,
        maxsma=60.0,
        debug=True,
        compute_deviations=True,
        nclip=0,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)

    assert result["multiband"] is True
    assert result["bands"] == ["g", "r"]
    assert result["reference_band"] == "g"
    assert result["harmonic_combination"] == "joint"
    assert result["band_weights"] == {"g": 1.0, "r": 1.0}
    assert result["variance_mode"] == "ols"

    isophotes = result["isophotes"]
    # Filter to acceptable stop codes (0 or 2) at SMA in the well-fit
    # mid-radius range (avoid the unconstrained outermost rings).
    mids = [iso for iso in isophotes if iso["valid"] and iso["stop_code"] in (0, 2) and 15.0 <= iso["sma"] <= 40.0]
    assert len(mids) >= 5
    # Average geometry on these isophotes recovers the truth within
    # the Q17 quality bars.
    x0_avg = np.mean([iso["x0"] for iso in mids])
    y0_avg = np.mean([iso["y0"] for iso in mids])
    eps_avg = np.mean([iso["eps"] for iso in mids])
    pa_avg = np.mean([iso["pa"] for iso in mids])
    assert abs(x0_avg - 96.0) < 0.5
    assert abs(y0_avg - 96.0) < 0.5
    assert abs(eps_avg - 0.3) < 0.02
    pa_diff = abs((pa_avg - 0.5 + np.pi / 2) % np.pi - np.pi / 2)
    assert pa_diff < np.deg2rad(1.0)

    # Per-band intensity columns populated for every isophote.
    for iso in isophotes:
        assert "intens_g" in iso
        assert "intens_r" in iso


def test_two_band_with_variance_maps_runs(planted_two_band):
    img_g, img_r = planted_two_band
    var = np.full_like(img_g, 0.05**2)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=50.0,
        debug=True,
        nclip=0,
    )
    result = fit_image_multiband(
        [img_g, img_r],
        None,
        cfg,
        variance_maps=[var, var.copy()],
    )
    assert result["variance_mode"] == "wls"
    valid_count = sum(1 for iso in result["isophotes"] if iso["valid"])
    assert valid_count > 0


def test_band_weights_passthrough(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        band_weights={"g": 2.0, "r": 0.5},
        sma0=15.0,
        astep=0.2,
        maxsma=40.0,
        nclip=0,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    assert result["band_weights"] == {"g": 2.0, "r": 0.5}


def test_ref_mode_runs(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        harmonic_combination="ref",
        sma0=15.0,
        astep=0.2,
        maxsma=40.0,
        debug=True,
        nclip=0,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    assert result["harmonic_combination"] == "ref"
    assert any(iso["valid"] for iso in result["isophotes"])


def _mask_outside_disk(shape, center, radius):
    """Boolean mask (True = masked) for every pixel outside a central disk."""
    y, x = np.mgrid[: shape[0], : shape[1]]
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius**2


def test_ref_mode_loose_validity_fits_correct_band():
    """Regression test for M1: ref x loose must fit the reference band.

    Under loose validity the solve arrays hold surviving bands only, but the
    ref branch indexed them by full-list position: with band g dropped at
    outer isophotes, position 1 held band i (eps=0.6) instead of the
    reference band r (eps=0.3), and the fit silently converged to the wrong
    band's geometry.
    """
    img_g = _planted_galaxy(eps=0.3, amplitude=100.0, seed=1)
    img_r = _planted_galaxy(eps=0.3, amplitude=200.0, seed=2)
    img_i = _planted_galaxy(eps=0.6, amplitude=150.0, seed=3)
    # Band g keeps too few samples outside 30 px and is dropped there.
    mask_g = _mask_outside_disk(img_g.shape, (96.0, 96.0), 30.0)

    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="r",  # NOT the first band
        harmonic_combination="ref",
        loose_validity=True,
        sma0=10.0,
        minsma=4.0,
        maxsma=60.0,
        astep=0.2,
        eps=0.3,
        pa=0.5,
        debug=True,
        nclip=0,
    )
    result = fit_image_multiband([img_g, img_r, img_i], [mask_g, None, None], cfg)

    outer = [iso for iso in result["isophotes"] if iso["sma"] > 35.0 and iso["stop_code"] == 0]
    assert outer, "no converged outer isophotes (g should be dropped, r+i survive)"
    for iso in outer:
        # Fitting the wrong band would report eps ~ 0.6 (band i's truth).
        assert iso["eps"] == pytest.approx(0.3, abs=0.05)


def test_ref_mode_loose_validity_reference_band_dropped():
    """M1 counterpart: a dropped reference band stops the isophote cleanly.

    Previously this either raised an uncaught IndexError or silently fit the
    wrong band; it must now skip the isophote with a clear stop code.
    """
    img_g = _planted_galaxy(eps=0.3, amplitude=100.0, seed=1)
    img_r = _planted_galaxy(eps=0.3, amplitude=200.0, seed=2)
    img_i = _planted_galaxy(eps=0.3, amplitude=150.0, seed=3)
    # The reference band itself is dropped at outer isophotes.
    mask_r = _mask_outside_disk(img_r.shape, (96.0, 96.0), 30.0)

    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="r",
        harmonic_combination="ref",
        loose_validity=True,
        sma0=10.0,
        minsma=4.0,
        maxsma=60.0,
        astep=0.2,
        eps=0.3,
        pa=0.5,
        debug=True,
        nclip=0,
    )
    result = fit_image_multiband([img_g, img_r, img_i], [None, mask_r, None], cfg)

    outer = [iso for iso in result["isophotes"] if iso["sma"] > 40.0]
    assert outer, "expected isophote rows at radii where r is dropped"
    bad = sorted({iso["stop_code"] for iso in outer} - {3})
    assert not bad, f"expected stop_code=3 where the reference band is dropped, got {bad}"


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_missing_config_raises():
    img = _planted_galaxy()
    with pytest.raises(ValueError, match="config is required"):
        fit_image_multiband([img], None, None)


def test_image_count_mismatch_with_bands_rejected(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r", "i"], reference_band="g")
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband([img_g, img_r], None, cfg)


def test_image_shape_mismatch_rejected():
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    img1 = np.zeros((100, 100), dtype=np.float64)
    img2 = np.zeros((50, 50), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        fit_image_multiband([img1, img2], None, cfg)


def test_variance_maps_count_mismatch_rejected(planted_two_band):
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    var = np.ones_like(img_g)
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband(
            [img_g, img_r],
            None,
            cfg,
            variance_maps=[var, var, var],
        )


def test_variance_maps_tuple_count_mismatch_rejected(planted_two_band):
    """Regression for B1: tuples must be validated, not silently accepted."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    var = np.ones_like(img_g)
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband(
            [img_g, img_r],
            None,
            cfg,
            variance_maps=(var, var, var),
        )


def test_variance_maps_non_sequence_rejected(planted_two_band):
    """Non-ndarray non-sequence inputs are explicitly rejected."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    with pytest.raises(TypeError, match="non-sequence"):
        fit_image_multiband(
            [img_g, img_r],
            None,
            cfg,
            variance_maps=42,  # type: ignore[arg-type]
        )


def test_masks_count_mismatch_rejected(planted_two_band):
    """Regression for B1: per-band mask sequence length is checked."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    bad_mask = np.zeros_like(img_g, dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband([img_g, img_r], [bad_mask, bad_mask, bad_mask], cfg)


def test_masks_tuple_count_mismatch_rejected(planted_two_band):
    """Tuples are accepted and length-checked just like lists."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    bad_mask = np.zeros_like(img_g, dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        fit_image_multiband([img_g, img_r], (bad_mask, bad_mask, bad_mask), cfg)


def test_first_isophote_failure_warning():
    """Sma0 entirely off-image triggers FIRST_FEW_ISOPHOTE_FAILURE."""
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        x0=10.0,
        y0=10.0,
        sma0=200.0,
        maxsma=300.0,
        astep=0.2,
        nclip=0,
        max_retry_first_isophote=0,
    )
    img = np.zeros((128, 128), dtype=np.float64)  # featureless image
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_image_multiband([img, img], None, cfg)
    assert any("FIRST_FEW_ISOPHOTE_FAILURE" in str(w.message) for w in captured)
    assert result.get("first_isophote_failure") is True


def test_m13_ndata_nflag_present_without_debug(planted_two_band):
    """M13: ndata/nflag are written on fitted rows regardless of debug —
    mixed presence used to produce masked FITS cells."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        astep=0.2,
        maxsma=40.0,
        debug=False,
        nclip=0,
    )
    result = fit_image_multiband([img_g, img_r], None, cfg)
    assert result["isophotes"]
    for iso in result["isophotes"]:
        assert "ndata" in iso and "nflag" in iso


def test_m13_forced_mode_stamps_n_valid(planted_two_band):
    """M13: forced-photometry rows carry per-band n_valid_<b> counts."""
    img_g, img_r = planted_two_band
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        astep=0.2,
        maxsma=40.0,
        debug=True,
        nclip=0,
    )
    template_res = fit_image_multiband([img_g, img_r], None, cfg)
    res_forced = fit_image_multiband([img_g, img_r], None, cfg, template_isophotes=template_res["isophotes"])
    ring_rows = [iso for iso in res_forced["isophotes"] if iso["sma"] > 0]
    assert ring_rows
    for iso in ring_rows:
        assert iso["n_valid_g"] > 0 and iso["n_valid_r"] > 0
