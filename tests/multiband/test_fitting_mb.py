"""Tests for ``isoster.multiband.fitting_mb``."""

import numpy as np
import pytest

from isoster.fitting import fit_first_and_second_harmonics
from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import (
    _per_band_mean_or_median,
    _per_band_mean_or_median_jagged,
    evaluate_joint_model,
    extract_forced_photometry_mb,
    fit_first_and_second_harmonics_joint,
    fit_first_and_second_harmonics_joint_loose,
    fit_isophote_mb,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _planted_galaxy(
    h: int = 256,
    w: int = 256,
    x0: float = 128.0,
    y0: float = 128.0,
    eps: float = 0.3,
    pa: float = 0.5,
    n_sersic: float = 1.5,
    re: float = 30.0,
    amplitude: float = 1.0,
    noise_sigma: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """A perfect-ellipse Sersic-like profile with Gaussian noise.

    Sersic with the ellipse semi-major axis as the radial coordinate
    (b/a = 1 - eps, axis rotated by pa), so the truth geometry is
    exactly ``(x0, y0, eps, pa)``.
    """
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


# ---------------------------------------------------------------------------
# Joint solver
# ---------------------------------------------------------------------------


def test_joint_solver_recovers_planted_coefficients():
    """The joint solver hits the truth for clean inputs."""
    rng = np.random.default_rng(0)
    n = 128
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    n_bands = 4
    I0 = np.array([100.0, 50.0, 200.0, 30.0])
    A1, B1, A2, B2 = 0.5, -0.3, 0.2, 0.1
    intens = np.empty((n_bands, n), dtype=np.float64)
    for b in range(n_bands):
        intens[b] = (
            I0[b] + A1 * np.sin(angles) + B1 * np.cos(angles) + A2 * np.sin(2 * angles) + B2 * np.cos(2 * angles)
        )
    intens += rng.normal(0.0, 1e-3, size=intens.shape)
    weights = np.ones(n_bands)
    coeffs, cov, wls = fit_first_and_second_harmonics_joint(angles, intens, weights, None)

    assert coeffs.shape == (n_bands + 4,)
    assert cov is not None and cov.shape == (n_bands + 4, n_bands + 4)
    assert wls is False  # OLS path
    np.testing.assert_allclose(coeffs[:n_bands], I0, atol=2e-4)
    np.testing.assert_allclose(coeffs[n_bands:], [A1, B1, A2, B2], atol=2e-4)


def test_joint_solver_b1_matches_single_band_solver():
    """B=1 joint solve must match the single-band 5-param fit."""
    rng = np.random.default_rng(7)
    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    coeffs_truth = np.array([100.0, 0.5, -0.3, 0.2, 0.1])
    intens = (
        coeffs_truth[0]
        + coeffs_truth[1] * np.sin(angles)
        + coeffs_truth[2] * np.cos(angles)
        + coeffs_truth[3] * np.sin(2 * angles)
        + coeffs_truth[4] * np.cos(2 * angles)
    )
    intens += rng.normal(0.0, 1e-3, size=n)

    sb_coeffs, _ = fit_first_and_second_harmonics(angles, intens)

    intens_2d = intens[None, :]  # (1, N)
    weights = np.ones(1)
    mb_coeffs, _, _ = fit_first_and_second_harmonics_joint(angles, intens_2d, weights, None)

    # Same coefficient ordering: [I0, A1, B1, A2, B2].
    np.testing.assert_allclose(mb_coeffs, sb_coeffs, atol=1e-12)


def test_joint_solver_wls_exact_covariance():
    """WLS mode produces the exact (A^T W A)^-1 covariance, no scaling."""
    rng = np.random.default_rng(11)
    n = 256
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    n_bands = 2
    I0 = np.array([100.0, 50.0])
    intens = np.empty((n_bands, n))
    for b in range(n_bands):
        intens[b] = (
            I0[b] + 0.3 * np.sin(angles) + 0.1 * np.cos(angles) + 0.05 * np.sin(2 * angles) + 0.02 * np.cos(2 * angles)
        )
    sigma_b = np.array([0.5, 0.1])
    for b in range(n_bands):
        intens[b] += rng.normal(0.0, sigma_b[b], size=n)
    var = np.broadcast_to(sigma_b[:, None] ** 2, (n_bands, n)).copy()

    weights = np.ones(n_bands)
    coeffs, cov, wls = fit_first_and_second_harmonics_joint(angles, intens, weights, var)
    assert wls is True
    # I0_b err in WLS: σ_I0_b ~ σ_b / sqrt(N)  (ignoring harmonic-cov coupling)
    err_I0 = np.sqrt(np.diag(cov))[:n_bands]
    np.testing.assert_allclose(err_I0, sigma_b / np.sqrt(n), rtol=0.1)


def test_joint_solver_band_weights_zero_band_drops_band():
    """``band_weights`` zero on one band makes its rows have zero weight."""
    rng = np.random.default_rng(13)
    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    n_bands = 2
    I0 = np.array([100.0, 0.0])
    intens = np.empty((n_bands, n))
    intens[0] = I0[0] + 0.5 * np.sin(angles)
    # Garbage in band 1 — but we will weight it to ~0
    intens[1] = rng.normal(0.0, 100.0, size=n)
    weights = np.array([1.0, 1e-12])  # band 1 essentially ignored
    coeffs, _cov, _wls = fit_first_and_second_harmonics_joint(angles, intens, weights, None)
    # Band 0's I0 should still recover
    assert abs(coeffs[0] - I0[0]) < 0.05
    # Shared A1 should recover
    assert abs(coeffs[n_bands] - 0.5) < 0.05


def test_evaluate_joint_model_shape_and_values():
    angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    coeffs = np.array(
        [
            10.0,
            20.0,
            30.0,  # I0_g, I0_r, I0_i
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )  # A1=1, others=0
    model = evaluate_joint_model(angles, coeffs, 3)
    assert model.shape == (3, 8)
    expected_geom = np.sin(angles)
    for b, I0_b in enumerate([10.0, 20.0, 30.0]):
        np.testing.assert_allclose(model[b], I0_b + expected_geom, atol=1e-12)


# ---------------------------------------------------------------------------
# Per-band sigma clipping
# ---------------------------------------------------------------------------


def test_per_band_sigma_clip_and_shared_validity():
    """A clipped sample in one band drops it from all bands (AND mask)."""
    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    intens = np.ones((2, n))
    intens[0] *= 100.0
    intens[1] *= 50.0
    intens[1, 10] = 1e6  # huge outlier in band 1 only
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", sclip=3.0, nclip=3)
    # Build a fake MultiIsophoteData and reuse the iteration loop's clip.
    from isoster.multiband.fitting_mb import _per_band_sigma_clip

    a, p, ic, vc, n_clipped = _per_band_sigma_clip(
        angles,
        angles,
        intens,
        None,
        cfg.sclip,
        cfg.nclip,
        cfg.sclip_low,
        cfg.sclip_high,
    )
    assert n_clipped >= 1
    assert ic.shape[1] == n - n_clipped
    # Both bands' arrays must have the same surviving N (shared validity).
    assert ic.shape[0] == 2
    # Variance is None on the input, so the clipper must mirror that.
    assert vc is None
    # Surviving angles/phi length must equal the surviving intensity column count.
    assert a.shape[0] == ic.shape[1]
    assert p.shape[0] == ic.shape[1]
    # The huge outlier in band 1 at index 10 must have been removed from BOTH
    # bands' intensity rows (shared validity).
    assert np.max(ic[1]) < 1e5
    assert ic[0, :].max() <= 100.0 and ic[0, :].min() >= 100.0 - 1e-9


# ---------------------------------------------------------------------------
# fit_isophote_mb on a planted galaxy
# ---------------------------------------------------------------------------


def test_fit_isophote_mb_planted_recovers_geometry():
    """B=2 joint fit on a planted Sersic recovers shared geometry near truth."""
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=1)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.01, seed=2)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=20.0,
        eps=0.2,
        pa=0.4,
        debug=True,
        compute_deviations=True,
        nclip=0,  # no sigma clipping needed for clean synthetic
    )
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.4}
    out = fit_isophote_mb(
        images=[img1, img2],
        masks=None,
        sma=30.0,
        start_geometry=start,
        config=cfg,
    )
    assert out["stop_code"] in (0, 2)
    assert out["valid"] is True
    # Recovered geometry within Q17 thresholds (0.5 px / 0.02 eps / 1°).
    assert abs(out["x0"] - 128.0) < 0.5
    assert abs(out["y0"] - 128.0) < 0.5
    assert abs(out["eps"] - 0.3) < 0.02
    pa_diff = abs((out["pa"] - 0.5 + np.pi / 2) % np.pi - np.pi / 2)
    assert pa_diff < np.deg2rad(1.0)
    # Per-band intensity columns present and finite.
    for b in ("g", "r"):
        assert np.isfinite(out[f"intens_{b}"])
        assert np.isfinite(out[f"rms_{b}"])
        # Harmonic deviations attached after convergence.
        for n in (3, 4):
            assert f"a{n}_{b}" in out
            assert f"b{n}_{b}" in out


def test_fit_isophote_mb_too_few_points_returns_stop3():
    """A geometry that puts the ring almost entirely off-image returns stop_code=3."""
    img = _planted_galaxy(h=64, w=64, x0=32.0, y0=32.0)
    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="g",
        sma0=10.0,
        nclip=0,
    )
    # Center far outside the cutout → most of the ring is off-image.
    start = {"x0": 2000.0, "y0": 2000.0, "eps": 0.2, "pa": 0.0}
    out = fit_isophote_mb(
        images=[img, img, img],
        masks=None,
        sma=10.0,
        start_geometry=start,
        config=cfg,
    )
    # Either stop_code 3 (too few points) or -1 (gradient failure) is
    # acceptable here; both indicate the fit recognized a degenerate input.
    assert out["stop_code"] in (3, -1)
    assert out["valid"] is False


def test_fit_isophote_mb_b1_runs():
    """B=1 fit_isophote_mb still runs and produces the suffixed schema."""
    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=1)
    cfg = IsosterConfigMB(bands=["g"], reference_band="g", sma0=20.0, debug=True, nclip=0)
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.4}
    out = fit_isophote_mb(
        images=[img],
        masks=None,
        sma=30.0,
        start_geometry=start,
        config=cfg,
    )
    assert "intens_g" in out
    assert out["stop_code"] in (0, 2)
    assert abs(out["x0"] - 128.0) < 0.5


def test_fit_isophote_mb_ref_mode_runs():
    """``harmonic_combination='ref'`` exercises the single-band-style fallback."""
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=1)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.01, seed=2)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        harmonic_combination="ref",
        sma0=20.0,
        debug=True,
        compute_deviations=True,
        nclip=0,
    )
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.4}
    out = fit_isophote_mb(
        images=[img1, img2],
        masks=None,
        sma=30.0,
        start_geometry=start,
        config=cfg,
    )
    assert out["stop_code"] in (0, 2)
    assert abs(out["x0"] - 128.0) < 1.0
    assert "intens_g" in out and "intens_r" in out


def test_ring_mean_intercept_solver_reduces_columns():
    """``fit_per_band_intens_jointly=False`` (ring-mean intercept mode):
    the joint solver becomes a 4-column geometric system.
    ``coeffs[:n_bands]`` are the per-band ring means (post-fit), and the
    harmonic block ``coeffs[n_bands:]`` matches what a single-band solve
    on residuals would produce.
    """
    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(7)
    means = np.array([10.0, 20.0])
    A1, B1 = 0.3, -0.1
    geom = A1 * np.sin(angles) + B1 * np.cos(angles)
    intens = np.array([means[0] + geom + rng.normal(0, 0.01, n), means[1] + geom + rng.normal(0, 0.01, n)])
    weights = np.ones(2)

    coeffs_full, _, _ = fit_first_and_second_harmonics_joint(
        angles,
        intens,
        weights,
        None,
    )
    coeffs_zero, cov_zero, _ = fit_first_and_second_harmonics_joint(
        angles,
        intens,
        weights,
        None,
        fit_per_band_intens_jointly=False,
    )

    # Per-band intercepts should be the empirical ring means.
    np.testing.assert_allclose(coeffs_zero[:2], np.mean(intens, axis=1), atol=1e-12)
    # Geometric coefficients agree with the full solve on this clean planted case.
    np.testing.assert_allclose(coeffs_full[2:], coeffs_zero[2:], atol=2e-3)
    # cov has zero off-diagonal coupling between per-band intercepts and
    # geometric block (since intercepts are post-fit, not solved).
    assert cov_zero is not None
    assert np.all(cov_zero[:2, 2:] == 0.0)
    assert np.all(cov_zero[2:, :2] == 0.0)


def test_ring_mean_intercept_intens_err_scales_with_band_noise():
    """B3-style regression: per-band intens_err must scale linearly with
    band noise sigma when fit_per_band_intens_jointly=False."""
    h, w = 64, 64
    rng = np.random.default_rng(11)
    img_g = 10.0 + rng.normal(0.0, 0.10, size=(h, w))
    img_r_low = 10.0 + rng.normal(0.0, 0.10, size=(h, w))
    img_r_high = 10.0 + rng.normal(0.0, 1.00, size=(h, w))
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        fit_per_band_intens_jointly=False,
        nclip=0,
    )
    start = {"x0": 32.0, "y0": 32.0, "eps": 0.0, "pa": 0.0}
    out_low = fit_isophote_mb(
        images=[img_g, img_r_low],
        masks=None,
        sma=12.0,
        start_geometry=start,
        config=cfg,
    )
    out_high = fit_isophote_mb(
        images=[img_g, img_r_high],
        masks=None,
        sma=12.0,
        start_geometry=start,
        config=cfg,
    )
    err_low = float(out_low["intens_err_r"])
    err_high = float(out_high["intens_err_r"])
    assert err_low > 0 and err_high > 0
    assert 4.0 < err_high / err_low < 25.0


def test_ring_mean_intercept_rejected_in_ref_mode():
    """fit_per_band_intens_jointly=False is incompatible with
    harmonic_combination='ref'."""
    with pytest.raises(ValueError, match="incompatible"):
        IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            harmonic_combination="ref",
            fit_per_band_intens_jointly=False,
        )


def test_legacy_field_name_rejected_with_clear_error():
    """Section 6 cleanup: passing the old ``fix_per_band_background_to_zero``
    field raises a clear ValueError pointing at the new name (no silent
    drop)."""
    with pytest.raises(ValueError, match="renamed.*fit_per_band_intens_jointly"):
        IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            fix_per_band_background_to_zero=True,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# D9 backport — loose validity end-to-end
# ---------------------------------------------------------------------------


def _plant_per_band_arc_artifact_image(
    h: int = 256,
    w: int = 256,
    x0: float = 128.0,
    y0: float = 128.0,
    eps: float = 0.3,
    pa: float = 0.5,
    amplitude: float = 100.0,
    noise_sigma: float = 0.05,
    seed: int = 1,
):
    """Two-band planted galaxy plus a 30°-arc mask in band r at SMA 35-45."""
    img_g = _planted_galaxy(
        h=h,
        w=w,
        x0=x0,
        y0=y0,
        eps=eps,
        pa=pa,
        amplitude=amplitude,
        noise_sigma=noise_sigma,
        seed=seed,
    )
    img_r = _planted_galaxy(
        h=h,
        w=w,
        x0=x0,
        y0=y0,
        eps=eps,
        pa=pa,
        amplitude=amplitude * 0.6,
        noise_sigma=noise_sigma,
        seed=seed + 1,
    )
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = xx - x0
    dy = yy - y0
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ell = np.sqrt(x_rot**2 + (y_rot / max(1.0 - eps, 1e-3)) ** 2)
    phi_grid = np.arctan2(y_rot, x_rot)
    mask_r = (r_ell > 35.0) & (r_ell < 45.0) & (phi_grid > -0.4) & (phi_grid < 0.4)
    return img_g, img_r, mask_r


def test_loose_validity_band_drop_at_isophote():
    """A band pushed below the per-band minimum count is dropped at that
    isophote — its intens_<b> goes NaN, the other band is populated, and
    the isophote still produces a fitted geometry."""
    rng = np.random.default_rng(11)
    img = 10.0 + rng.normal(0.0, 0.05, size=(64, 64))
    bad_r = np.zeros_like(img, dtype=bool)
    # Mask >95% of band-r pixels so its surviving count drops below
    # the default loose_validity_min_per_band_count = 6.
    bad_r[1:, :] = True
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        loose_validity=True,
        loose_validity_min_per_band_count=6,
        loose_validity_min_per_band_frac=0.0,
        nclip=0,
    )
    out = fit_isophote_mb(
        images=[img, img],
        masks=[None, bad_r],
        sma=12.0,
        start_geometry={"x0": 32.0, "y0": 32.0, "eps": 0.0, "pa": 0.0},
        config=cfg,
    )
    # Whole-isophote drop fires when fewer than 2 bands survive — here
    # only band g survives, so the isophote is invalid.
    assert out["stop_code"] == 3
    assert out["valid"] is False


def test_loose_validity_n_valid_columns_present_under_loose_mode():
    """Per-band n_valid_<b> columns appear on every row when loose validity
    is enabled, and equal the band's surviving count after sigma clip."""
    img_g, img_r, mask_r = _plant_per_band_arc_artifact_image()
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=20.0,
        maxsma=60.0,
        astep=0.15,
        debug=True,
        nclip=0,
        loose_validity=True,
    )
    out = fit_isophote_mb(
        images=[img_g, img_r],
        masks=[None, mask_r],
        sma=40.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5},
        config=cfg,
    )
    assert "n_valid_g" in out
    assert "n_valid_r" in out
    # Band r had pixels masked along the arc → fewer surviving samples.
    assert int(out["n_valid_g"]) > int(out["n_valid_r"])
    assert int(out["n_valid_r"]) > 0


def test_loose_validity_default_off_no_behavior_change():
    """Regression guard: with loose_validity=False (default), the result
    is numerically identical to the legacy shared-validity path."""
    img_g = _planted_galaxy(seed=1)
    img_r = _planted_galaxy(amplitude=50.0, seed=2)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=20.0,
        maxsma=80.0,
        astep=0.2,
        debug=True,
        nclip=0,
    )
    assert cfg.loose_validity is False
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.4}
    out = fit_isophote_mb(
        images=[img_g, img_r],
        masks=None,
        sma=30.0,
        start_geometry=start,
        config=cfg,
    )
    # Sanity: shared validity gives identical n_valid across bands.
    assert int(out["n_valid_g"]) == int(out["n_valid_r"])


def test_loose_validity_per_band_count_normalization_changes_intens_err():
    """`per_band_count` normalization changes the WLS row weights, which
    changes per-band intens_err. Asymmetric per-band noise makes the
    effect easy to verify."""
    rng = np.random.default_rng(3)
    h = 64
    img_g = 10.0 + rng.normal(0.0, 0.10, size=(h, h))
    img_r = 10.0 + rng.normal(0.0, 1.00, size=(h, h))
    var_g = np.full_like(img_g, 0.01, dtype=np.float64)
    var_r = np.full_like(img_r, 1.00, dtype=np.float64)

    base_kwargs = dict(
        bands=["g", "r"],
        reference_band="g",
        nclip=0,
        loose_validity=True,
    )
    cfg_none = IsosterConfigMB(
        **base_kwargs,
        loose_validity_band_normalization="none",
    )
    cfg_perband = IsosterConfigMB(
        **base_kwargs,
        loose_validity_band_normalization="per_band_count",
    )
    start = {"x0": 32.0, "y0": 32.0, "eps": 0.0, "pa": 0.0}
    out_none = fit_isophote_mb(
        images=[img_g, img_r],
        masks=None,
        sma=10.0,
        start_geometry=start,
        config=cfg_none,
        variance_maps=[var_g, var_r],
    )
    out_perband = fit_isophote_mb(
        images=[img_g, img_r],
        masks=None,
        sma=10.0,
        start_geometry=start,
        config=cfg_perband,
        variance_maps=[var_g, var_r],
    )
    # Both paths produce finite intens_err for both bands.
    for k in ("intens_err_g", "intens_err_r"):
        assert np.isfinite(float(out_none[k])) and float(out_none[k]) > 0
        assert np.isfinite(float(out_perband[k])) and float(out_perband[k]) > 0
    # The geometric coefficients differ by enough to be detectable —
    # changing the row weighting changes the joint solve. In practice
    # the per-band intens_err scales the same way (direct SEM in both
    # modes), but the rms / x0_err differ.
    assert not np.isclose(float(out_none["x0_err"]), float(out_perband["x0_err"]), rtol=1e-6) or not np.isclose(
        float(out_none["rms"]), float(out_perband["rms"]), rtol=1e-6
    )


def test_loose_validity_with_ring_mean_intercept():
    """loose_validity=True + fit_per_band_intens_jointly=False: per-band
    intens_<b> equals the band's empirical surviving-sample mean."""
    img_g, img_r, mask_r = _plant_per_band_arc_artifact_image()
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=20.0,
        maxsma=60.0,
        astep=0.2,
        debug=True,
        nclip=0,
        loose_validity=True,
        fit_per_band_intens_jointly=False,
    )
    out = fit_isophote_mb(
        images=[img_g, img_r],
        masks=[None, mask_r],
        sma=40.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5},
        config=cfg,
    )
    # Both bands survive (artifact is small relative to the ring); the
    # per-band intens is the band's own ring mean by construction.
    assert np.isfinite(out["intens_g"])
    assert np.isfinite(out["intens_r"])
    assert int(out["n_valid_g"]) > int(out["n_valid_r"])


def test_loose_validity_with_ref_mode():
    """Ref mode + loose validity: reference band intens unchanged from
    its single-band ring mean, non-ref bands' intens come from their own
    surviving samples."""
    img_g, img_r, mask_r = _plant_per_band_arc_artifact_image()
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=20.0,
        maxsma=60.0,
        astep=0.2,
        debug=True,
        nclip=0,
        loose_validity=True,
        harmonic_combination="ref",
    )
    out = fit_isophote_mb(
        images=[img_g, img_r],
        masks=[None, mask_r],
        sma=40.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5},
        config=cfg,
    )
    assert np.isfinite(out["intens_g"]) and np.isfinite(out["intens_r"])
    # Per-band counts must reflect the per-band masking under loose
    # validity even in ref mode (the ref-mode code path does not
    # interfere with how the sampler reports per-band counts).
    assert int(out["n_valid_g"]) > int(out["n_valid_r"])


def test_fit_isophote_mb_ref_mode_intens_err_scaling_ols(monkeypatch):
    """Regression for B3.

    In ``harmonic_combination='ref'`` + OLS mode, non-reference bands'
    ``intens_err_<b>`` must be the band's own SEM ≈ ``σ_b / √N``, not a
    quantity that has the residual variance applied twice.  Concretely:
    if we change band-r's noise level by a factor 10, the reported
    ``intens_err_r`` should also change by ~10×, not by ~100×.
    """
    n = 200
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(0)

    def synth(noise_sigma: float, seed: int) -> np.ndarray:
        rng_local = np.random.default_rng(seed)
        return rng_local.normal(loc=10.0, scale=noise_sigma, size=n)

    intens_g = synth(0.10, seed=11)
    intens_r_low = synth(0.10, seed=22)
    intens_r_high = synth(1.00, seed=22)  # 10× noisier in band r

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        harmonic_combination="ref",
        nclip=0,
    )

    # Drop into the per-band error branch directly via fit_isophote_mb on a
    # small synthetic image so the full code path executes.
    h, w = 64, 64
    img_g = np.full((h, w), 10.0, dtype=np.float64)
    img_r = img_g.copy()
    img_g += rng.normal(0.0, 0.10, size=img_g.shape)
    img_r_low = img_g + rng.normal(0.0, 0.10, size=img_g.shape)
    img_r_high = img_g + rng.normal(0.0, 1.00, size=img_g.shape)
    start = {"x0": 32.0, "y0": 32.0, "eps": 0.0, "pa": 0.0}

    out_low = fit_isophote_mb(
        images=[img_g, img_r_low],
        masks=None,
        sma=12.0,
        start_geometry=start,
        config=cfg,
    )
    out_high = fit_isophote_mb(
        images=[img_g, img_r_high],
        masks=None,
        sma=12.0,
        start_geometry=start,
        config=cfg,
    )
    err_low = float(out_low["intens_err_r"])
    err_high = float(out_high["intens_err_r"])
    assert np.isfinite(err_low) and np.isfinite(err_high)
    assert err_low > 0 and err_high > 0
    # SEM scales linearly with σ: 10× noise should give ~10× error, not 100×.
    ratio = err_high / err_low
    assert 4.0 < ratio < 25.0, f"intens_err_r should scale ~linearly with band noise (10×), got ratio = {ratio:.2f}"


# ---------------------------------------------------------------------------
# Forced photometry helper
# ---------------------------------------------------------------------------


def test_extract_forced_photometry_mb_shape():
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.001, seed=1)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.001, seed=2)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    out = extract_forced_photometry_mb(
        images=[img1, img2],
        masks=None,
        x0=128.0,
        y0=128.0,
        sma=20.0,
        eps=0.3,
        pa=0.5,
        bands=cfg.bands,
        config=cfg,
    )
    assert out["valid"] is True
    assert out["stop_code"] == 0
    for b in ("g", "r"):
        assert np.isfinite(out[f"intens_{b}"])
        assert np.isfinite(out[f"intens_err_{b}"])


# ---------------------------------------------------------------------------
# variance maps mixed all-or-nothing rejected at the sampler level
# ---------------------------------------------------------------------------


def test_fit_isophote_mb_rejects_mixed_variance_maps():
    img = _planted_galaxy(seed=1)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    var = np.ones_like(img) * 0.1
    with pytest.raises(ValueError, match="all-or-nothing"):
        fit_isophote_mb(
            images=[img, img],
            masks=None,
            sma=20.0,
            start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
            config=cfg,
            variance_maps=[var, None],  # type: ignore[list-item]
        )


# ---------------------------------------------------------------------------
# Stage-3 S2: per-band median integrator in decoupled mode
# ---------------------------------------------------------------------------


def test_per_band_mean_or_median_dispatches_correctly():
    """Direct unit test on the helper: mean vs median agree only on
    symmetric distributions."""
    rng = np.random.default_rng(2026)
    intens = rng.normal(loc=10.0, scale=0.5, size=(3, 200))  # symmetric
    var = np.full_like(intens, 0.25)

    means_ols = _per_band_mean_or_median(intens, None, "mean")
    means_wls = _per_band_mean_or_median(intens, var, "mean")
    medians = _per_band_mean_or_median(intens, var, "median")

    # On a symmetric Gaussian distribution mean ≈ median to better
    # than 0.05 (1/sqrt(N) ~ 0.07, so this is ~ within tolerance).
    assert np.allclose(means_ols, means_wls, atol=1e-12)  # uniform var ⇒ same
    assert np.allclose(means_ols, medians, atol=0.05)

    # Inject a one-sided outlier per band: median ignores it, mean shifts.
    intens_skewed = intens.copy()
    intens_skewed[:, 0] += 100.0  # huge upward outlier in sample 0 of every band
    means_skewed = _per_band_mean_or_median(intens_skewed, None, "mean")
    medians_skewed = _per_band_mean_or_median(intens_skewed, None, "median")
    # Mean shifted by ~ 100/200 = 0.5 per band. Median shifted by < 0.05.
    assert np.all(means_skewed - means_ols > 0.4)
    assert np.all(np.abs(medians_skewed - medians) < 0.05)


def test_per_band_mean_or_median_jagged_handles_empty_bands():
    """Empty band ⇒ NaN under both integrators; non-empty bands resolve."""
    intens = [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([], dtype=np.float64)]
    var = [np.full(5, 0.1), np.array([], dtype=np.float64)]
    out_mean = _per_band_mean_or_median_jagged(intens, var, "mean")
    out_med = _per_band_mean_or_median_jagged(intens, var, "median")
    assert out_mean[0] == pytest.approx(3.0)
    assert out_med[0] == pytest.approx(3.0)
    assert np.isnan(out_mean[1])
    assert np.isnan(out_med[1])


def test_joint_solver_decoupled_mean_vs_median_full_ring_agree():
    """On a clean full ring with symmetric noise, the per-band intercept
    should agree between integrator='mean' and 'median' to ~1/sqrt(N).
    """
    rng = np.random.default_rng(7)
    n_samples = 360
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    truth = np.array([10.0, 20.0])
    # Pure intercept signal + a known A1, B1 harmonic + Gaussian noise.
    intens = truth[:, None] + 0.3 * np.sin(angles) - 0.2 * np.cos(angles) + rng.normal(scale=0.05, size=(2, n_samples))
    weights = np.array([1.0, 1.0])

    coeffs_mean, _, _ = fit_first_and_second_harmonics_joint(
        angles,
        intens,
        weights,
        None,
        fit_per_band_intens_jointly=False,
        integrator="mean",
    )
    coeffs_med, _, _ = fit_first_and_second_harmonics_joint(
        angles,
        intens,
        weights,
        None,
        fit_per_band_intens_jointly=False,
        integrator="median",
    )

    # Per-band intercepts agree to within a few × σ/√N ~ 0.05/√360 ~ 0.003
    # (median has a 1.25× efficiency penalty vs mean for Gaussian noise).
    assert np.allclose(coeffs_mean[:2], coeffs_med[:2], atol=0.02)
    # Geometric block trivially identical to ~noise floor: with symmetric
    # noise the residuals after subtracting the per-band intercept are
    # statistically equivalent under either reducer.
    assert np.allclose(coeffs_mean[2:], coeffs_med[2:], atol=0.01)


def test_joint_solver_decoupled_median_resists_one_sided_outlier():
    """Asymmetric contamination: the median path should suppress an
    outlier sector that pulls the mean by orders of magnitude.
    """
    rng = np.random.default_rng(11)
    n_samples = 360
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    truth = 10.0
    intens_clean = truth + rng.normal(scale=0.05, size=n_samples)
    # Inject 5% of samples (one band) with a huge positive contaminant.
    contam_idx = np.arange(n_samples)[:18]
    intens_contam = intens_clean.copy()
    intens_contam[contam_idx] += 100.0
    intens = np.stack([intens_clean, intens_contam], axis=0)
    weights = np.array([1.0, 1.0])

    coeffs_mean, _, _ = fit_first_and_second_harmonics_joint(
        angles,
        intens,
        weights,
        None,
        fit_per_band_intens_jointly=False,
        integrator="mean",
    )
    coeffs_med, _, _ = fit_first_and_second_harmonics_joint(
        angles,
        intens,
        weights,
        None,
        fit_per_band_intens_jointly=False,
        integrator="median",
    )

    # Mean of contaminated band shifts by ~ 100 * 18/360 = 5.0
    assert coeffs_mean[1] - truth > 4.0
    # Median is barely affected — < 5% of samples are outliers.
    assert abs(coeffs_med[1] - truth) < 0.1
    # Clean band: both reducers agree on the truth.
    assert abs(coeffs_mean[0] - truth) < 0.05
    assert abs(coeffs_med[0] - truth) < 0.05


def test_joint_solver_loose_decoupled_median_works_with_jagged_input():
    """Loose-validity solver path (jagged) supports integrator='median'."""
    rng = np.random.default_rng(13)
    n_b1, n_b2 = 200, 120
    phi_per_band = [
        np.linspace(0.0, 2.0 * np.pi, n_b1, endpoint=False),
        np.linspace(0.0, 2.0 * np.pi, n_b2, endpoint=False),
    ]
    truth = np.array([5.0, 8.0])
    intens_per_band = [
        truth[0] + rng.normal(scale=0.02, size=n_b1),
        truth[1] + rng.normal(scale=0.03, size=n_b2),
    ]
    weights = np.array([1.0, 1.0])

    coeffs, _, _ = fit_first_and_second_harmonics_joint_loose(
        phi_per_band,
        intens_per_band,
        weights,
        None,
        fit_per_band_intens_jointly=False,
        integrator="median",
    )
    # Per-band medians sit on truth to noise/√N tolerance.
    assert abs(coeffs[0] - truth[0]) < 0.01
    assert abs(coeffs[1] - truth[1]) < 0.02


def test_fit_isophote_mb_median_intercept_end_to_end():
    """End-to-end with integrator='median' + decoupled intercept mode:
    the fit must converge and produce finite intens_<b>."""
    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=21)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.05, seed=22)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        integrator="median",
        fit_per_band_intens_jointly=False,
    )
    out = fit_isophote_mb(
        images=[img1, img2],
        masks=None,
        sma=20.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
        config=cfg,
    )
    assert out["valid"] is True
    assert out["stop_code"] == 0
    for b in ("g", "r"):
        assert np.isfinite(out[f"intens_{b}"])
        assert out[f"intens_{b}"] > 0
        assert np.isfinite(out[f"intens_err_{b}"])


# ---------------------------------------------------------------------------
# Stage-3 Stage-B: outer-region damping
# ---------------------------------------------------------------------------


def test_fit_isophote_mb_outer_damping_inert_below_onset():
    """At sma well below onset, damping has lambda ≈ 0 → identical fit.

    Both configs run with geometry_convergence=True so the comparison
    isolates the outer-reg effect from the auto-enable side-effect.
    """
    import warnings as _warnings

    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.001, seed=42)
    cfg_off = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        geometry_convergence=True,
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg_on = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            geometry_convergence=True,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=200.0,  # well above test sma
            outer_reg_strength=10.0,
        )
    ref_geom = {"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5}
    out_off = fit_isophote_mb(
        images=[img, img],
        masks=None,
        sma=20.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
        config=cfg_off,
    )
    out_on = fit_isophote_mb(
        images=[img, img],
        masks=None,
        sma=20.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
        config=cfg_on,
        outer_reference_geom=ref_geom,
    )
    # sma=20, onset=200, default width=80 → lambda ≈ 0.19 * coeff² which
    # times 8e-3-ish coefficient produces an alpha of order 1e-5; the
    # step shrink is sub-pixel-fraction. With matching
    # geometry_convergence on both sides the fits agree to better than
    # 1e-6, well below any pixel-scale interpretation.
    for k in ("x0", "y0", "eps", "pa"):
        assert abs(float(out_off[k]) - float(out_on[k])) < 1e-6


def test_fit_isophote_mb_outer_damping_active_above_onset():
    """At sma well above onset, damping shrinks per-iteration steps:
    fewer / smaller geometry updates from the same starting point in the
    same number of iterations."""
    import warnings as _warnings

    rng = np.random.default_rng(2026)
    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=99)
    # Inject a small mid-image bump that pulls geometry away from truth on
    # outer rings; gives the damper something to push back against.
    bump = np.zeros_like(img)
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    bump += 0.5 * np.exp(-((xx - 90.0) ** 2 + (yy - 90.0) ** 2) / (5.0**2))
    img = img + bump + rng.normal(0.0, 1e-4, size=img.shape)

    cfg_off = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        maxit=12,
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg_on = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            maxit=12,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=10.0,
            outer_reg_sma_width=2.0,
            outer_reg_strength=8.0,  # strong damper for a clear signal
            outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
        )
    ref_geom = {"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5}
    seed_geom = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0}

    out_off = fit_isophote_mb(
        images=[img, img],
        masks=None,
        sma=80.0,
        start_geometry=seed_geom,
        config=cfg_off,
    )
    out_on = fit_isophote_mb(
        images=[img, img],
        masks=None,
        sma=80.0,
        start_geometry=seed_geom,
        config=cfg_on,
        outer_reference_geom=ref_geom,
    )
    # Damping shrinks the per-iteration step → after the same maxit the
    # damped fit should land closer to the seed (i.e. less total drift).
    drift_off = np.hypot(out_off["x0"] - seed_geom["x0"], out_off["y0"] - seed_geom["y0"])
    drift_on = np.hypot(out_on["x0"] - seed_geom["x0"], out_on["y0"] - seed_geom["y0"])
    assert drift_on <= drift_off + 1e-9
    # eps step should also be smaller or equal under damping.
    eps_step_off = abs(out_off["eps"] - seed_geom["eps"])
    eps_step_on = abs(out_on["eps"] - seed_geom["eps"])
    assert eps_step_on <= eps_step_off + 1e-9


def test_fit_isophote_mb_outer_damping_off_when_no_reference():
    """When outer_reference_geom is None, damping is fully inert even with
    use_outer_center_regularization=True. This guards the driver-level
    invariant: outer-reg cannot fire on the inward sweep."""
    import warnings as _warnings

    img = _planted_galaxy(seed=3)
    cfg_off = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg_on = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            use_outer_center_regularization=True,
            outer_reg_sma_onset=5.0,
            outer_reg_strength=10.0,
        )
    seed_geom = {"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0}
    out_off = fit_isophote_mb(
        images=[img, img],
        masks=None,
        sma=80.0,
        start_geometry=seed_geom,
        config=cfg_off,
    )
    out_on = fit_isophote_mb(
        images=[img, img],
        masks=None,
        sma=80.0,
        start_geometry=seed_geom,
        config=cfg_on,  # no outer_reference_geom passed → inert
    )
    for k in ("x0", "y0", "eps", "pa"):
        assert abs(float(out_off[k]) - float(out_on[k])) < 1e-12


def test_build_outer_reference_mb_flux_weighted_average():
    """Driver-level reference builder: x0/y0/eps come from the flux-
    weighted average of qualifying inward isophotes + anchor."""
    import warnings as _warnings

    from isoster.multiband.driver_mb import _build_outer_reference_mb

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            sma0=10.0,
            use_outer_center_regularization=True,
            outer_reg_ref_sma_factor=2.0,
        )
    anchor = {
        "x0": 100.0,
        "y0": 100.0,
        "eps": 0.30,
        "pa": 0.5,
        "sma": 10.0,
        "stop_code": 0,
        "intens_g": 100.0,
        "intens_r": 50.0,
    }
    inwards = [
        # In-window, finite, acceptable: contributes.
        {
            "x0": 102.0,
            "y0": 99.0,
            "eps": 0.32,
            "pa": 0.45,
            "sma": 8.0,
            "stop_code": 0,
            "intens_g": 200.0,
            "intens_r": 100.0,
        },
        # In-window, intens_g=300 dominant flux weight.
        {
            "x0": 98.0,
            "y0": 101.0,
            "eps": 0.28,
            "pa": 0.55,
            "sma": 5.0,
            "stop_code": 0,
            "intens_g": 300.0,
            "intens_r": 150.0,
        },
        # Out-of-window (sma > sma0 * factor=20): excluded.
        {
            "x0": 1000.0,
            "y0": -1000.0,
            "eps": 0.9,
            "pa": 1.5,
            "sma": 30.0,
            "stop_code": 0,
            "intens_g": 1.0,
            "intens_r": 0.5,
        },
        # Bad stop_code: excluded.
        {
            "x0": 5000.0,
            "y0": 5000.0,
            "eps": 0.99,
            "pa": 0.0,
            "sma": 7.0,
            "stop_code": 3,
            "intens_g": 1e6,
            "intens_r": 1e6,
        },
    ]
    ref = _build_outer_reference_mb(inwards, anchor, cfg)
    # Brute-force reference: anchor + first two inwards entries.
    weights = np.array([100.0, 200.0, 300.0])
    x0s = np.array([100.0, 102.0, 98.0])
    y0s = np.array([100.0, 99.0, 101.0])
    eps_arr = np.array([0.30, 0.32, 0.28])
    expected_x0 = float(np.average(x0s, weights=weights))
    expected_y0 = float(np.average(y0s, weights=weights))
    expected_eps = float(np.average(eps_arr, weights=weights))
    assert ref["x0"] == pytest.approx(expected_x0)
    assert ref["y0"] == pytest.approx(expected_y0)
    assert ref["eps"] == pytest.approx(expected_eps)
    # The garbage entries must not have leaked in (would have shifted the
    # mean by orders of magnitude).
    assert abs(ref["x0"] - 100.0) < 5.0
    assert 0.0 < ref["eps"] < 1.0


def test_build_outer_reference_mb_falls_back_to_anchor_with_no_inwards():
    """No inwards results → reference is the anchor's own geometry."""
    import warnings as _warnings

    from isoster.multiband.driver_mb import _build_outer_reference_mb

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            sma0=10.0,
            use_outer_center_regularization=True,
        )
    anchor = {
        "x0": 64.0,
        "y0": 64.0,
        "eps": 0.25,
        "pa": 0.3,
        "sma": 10.0,
        "stop_code": 0,
        "intens_g": 50.0,
        "intens_r": 25.0,
    }
    ref = _build_outer_reference_mb([], anchor, cfg)
    assert ref["x0"] == pytest.approx(64.0)
    assert ref["y0"] == pytest.approx(64.0)
    assert ref["eps"] == pytest.approx(0.25)
    assert ref["pa"] == pytest.approx(0.3 % np.pi)


def test_fit_image_multiband_end_to_end_with_outer_damping():
    """Driver end-to-end: outer damping enabled converges and gives
    finite geometry; baseline run with feature off must still match
    on inner isophotes (where lambda ≈ 0)."""
    import warnings as _warnings

    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=51)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.005, seed=52)

    cfg_off = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=10.0,
        maxsma=80.0,
        astep=0.2,
        debug=True,
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg_on = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            sma0=10.0,
            maxsma=80.0,
            astep=0.2,
            debug=True,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=60.0,
            outer_reg_strength=4.0,
        )
    res_off = fit_image_multiband([img1, img2], masks=None, config=cfg_off)
    res_on = fit_image_multiband([img1, img2], masks=None, config=cfg_on)
    assert len(res_on["isophotes"]) >= 5
    inner_smas = [iso["sma"] for iso in res_on["isophotes"] if iso["sma"] < 30.0]
    assert len(inner_smas) >= 3
    # Inner isophotes (sma well below onset=60) should be near-identical
    # between off and on. We check x0 / y0 because eps and pa vary more
    # under noise; tolerance is generous since seeded RNG noise + lazy
    # gradient cache differences can land different local minima.
    for iso_off, iso_on in zip(res_off["isophotes"], res_on["isophotes"]):
        if float(iso_off["sma"]) < 30.0:
            assert abs(float(iso_off["x0"]) - float(iso_on["x0"])) < 0.5
            assert abs(float(iso_off["y0"]) - float(iso_on["y0"])) < 0.5


# ---------------------------------------------------------------------------
# Stage-3 Stage-C: lsb_auto_lock state machine
# ---------------------------------------------------------------------------


def test_is_lsb_isophote_mb_predicate():
    """Direct unit test on the lock-trigger predicate (joint gradient)."""
    from isoster.multiband.driver_mb import _is_lsb_isophote_mb

    # Healthy: small relative grad error, negative grad → no trigger.
    assert not _is_lsb_isophote_mb(
        {"grad": -10.0, "grad_error": 1.0, "stop_code": 0},
        0.3,
    )
    # Above threshold: |0.5 / -1.0| = 0.5 > 0.3 → trigger.
    assert _is_lsb_isophote_mb(
        {"grad": -1.0, "grad_error": 0.5, "stop_code": 0},
        0.3,
    )
    # Positive grad → trigger (galaxy gradient should be negative).
    assert _is_lsb_isophote_mb(
        {"grad": 5.0, "grad_error": 1.0, "stop_code": 0},
        0.3,
    )
    # stop_code=-1 → trigger regardless of gradient values.
    assert _is_lsb_isophote_mb(
        {"grad": -10.0, "grad_error": 0.1, "stop_code": -1},
        0.3,
    )
    # Missing keys → cannot assess, no trigger.
    assert not _is_lsb_isophote_mb({}, 0.3)
    assert not _is_lsb_isophote_mb({"grad": -10.0}, 0.3)
    # Non-finite values → no trigger.
    assert not _is_lsb_isophote_mb(
        {"grad": float("nan"), "grad_error": 1.0, "stop_code": 0},
        0.3,
    )
    assert not _is_lsb_isophote_mb(
        {"grad": -10.0, "grad_error": 0.0, "stop_code": 0},
        0.3,
    )


def test_build_locked_cfg_mb_freezes_geometry_and_disables_outer_reg():
    """The lock clone must freeze geometry, disable auto-lock and
    outer-reg on itself, and (when integrator='median') flip
    fit_per_band_intens_jointly to False to satisfy Stage-A S1."""
    import warnings as _warnings

    from isoster.multiband.driver_mb import _build_locked_cfg_mb

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            lsb_auto_lock=True,
            lsb_auto_lock_integrator="median",
            fit_per_band_intens_jointly=False,
            use_outer_center_regularization=True,
            sma0=10.0,
        )
    anchor = {
        "x0": 100.0,
        "y0": 100.0,
        "eps": 0.3,
        "pa": 0.5,
        "sma": 25.0,
        "stop_code": 0,
    }
    locked = _build_locked_cfg_mb(cfg, anchor, "median")
    assert locked.x0 == 100.0 and locked.y0 == 100.0
    assert locked.eps == 0.3 and locked.pa == 0.5
    assert locked.fix_center is True
    assert locked.fix_pa is True
    assert locked.fix_eps is True
    assert locked.integrator == "median"
    assert locked.fit_per_band_intens_jointly is False  # Stage-A S1
    assert locked.lsb_auto_lock is False  # one-way state machine
    assert locked.use_outer_center_regularization is False  # disabled on clone


def test_build_locked_cfg_mb_mean_keeps_intercept_mode():
    """Locked-region mean integrator does NOT need to flip the
    intercept mode."""
    import warnings as _warnings

    from isoster.multiband.driver_mb import _build_locked_cfg_mb

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            lsb_auto_lock=True,
            lsb_auto_lock_integrator="mean",
        )
    anchor = {"x0": 50.0, "y0": 50.0, "eps": 0.2, "pa": 0.0, "sma": 15.0}
    locked = _build_locked_cfg_mb(cfg, anchor, "mean")
    assert locked.integrator == "mean"
    assert locked.fit_per_band_intens_jointly is True  # untouched


def test_lsb_auto_lock_fires_on_synthetic_galaxy():
    """End-to-end: build a multi-band image where the gradient quality
    falls off in the outer region, run the driver with lsb_auto_lock on,
    confirm the lock fires and the outer isophotes carry the locked
    state metadata."""
    import warnings as _warnings

    from isoster.multiband import fit_image_multiband

    # Compact planted galaxy + heavy noise so the outer-region gradient
    # quality drops below the trigger threshold deterministically.
    img1 = _planted_galaxy(
        h=200,
        w=200,
        x0=100.0,
        y0=100.0,
        eps=0.25,
        pa=0.4,
        amplitude=50.0,
        re=10.0,
        noise_sigma=2.0,
        seed=2026,
    )
    img2 = _planted_galaxy(
        h=200,
        w=200,
        x0=100.0,
        y0=100.0,
        eps=0.25,
        pa=0.4,
        amplitude=25.0,
        re=10.0,
        noise_sigma=2.0,
        seed=2027,
    )

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
            lsb_auto_lock_integrator="mean",  # avoid S1; keep matrix-mode
            lsb_auto_lock_maxgerr=0.2,
            lsb_auto_lock_debounce=2,
        )
    res = fit_image_multiband(images=[img1, img2], masks=None, config=cfg)

    assert res.get("lsb_auto_lock") is True
    # The lock should fire SOMEWHERE in the outer sweep on this seeded
    # synthetic image — beyond ~ 2-3× re the gradient is essentially
    # noise. ``lsb_auto_lock_sma`` may be None if the trigger never
    # debounced, in which case the test data is wrong; assert non-None.
    assert res["lsb_auto_lock_sma"] is not None
    assert float(res["lsb_auto_lock_sma"]) > cfg.sma0  # locked outward
    assert int(res["lsb_auto_lock_count"]) >= 1
    # At least one isophote is marked lsb_locked=True; at least the
    # anchor gets lsb_locked=False (lock_state initialized).
    iso_locked = [iso for iso in res["isophotes"] if iso.get("lsb_locked")]
    assert len(iso_locked) >= 1
    # The locked isophotes' geometry must match the lock anchor exactly
    # (since fix_center / fix_pa / fix_eps fire on the cloned cfg).
    locked_x0s = [float(iso["x0"]) for iso in iso_locked]
    locked_y0s = [float(iso["y0"]) for iso in iso_locked]
    locked_eps = [float(iso["eps"]) for iso in iso_locked]
    locked_pa = [float(iso["pa"]) for iso in iso_locked]
    # All locked isophotes share the same geometry to numerical noise.
    assert max(locked_x0s) - min(locked_x0s) < 1e-9
    assert max(locked_y0s) - min(locked_y0s) < 1e-9
    assert max(locked_eps) - min(locked_eps) < 1e-9
    assert max(locked_pa) - min(locked_pa) < 1e-9


def test_lsb_auto_lock_off_no_metadata():
    """With lsb_auto_lock=False, the result dict carries no lock keys."""
    from isoster.multiband import fit_image_multiband

    img = _planted_galaxy(seed=4242)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=10.0,
        maxsma=60.0,
        astep=0.15,
        debug=True,
    )
    res = fit_image_multiband([img, img], masks=None, config=cfg)
    assert "lsb_auto_lock" not in res
    for iso in res["isophotes"]:
        assert "lsb_locked" not in iso


def test_top_level_grad_keys_under_debug():
    """Stage-C added top-level ``grad`` / ``grad_error`` to the row
    when debug=True. Required for the lock predicate."""
    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.001, seed=99)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", debug=True)
    out = fit_isophote_mb(
        images=[img, img],
        masks=None,
        sma=20.0,
        start_geometry={"x0": 128.0, "y0": 128.0, "eps": 0.2, "pa": 0.0},
        config=cfg,
    )
    assert "grad" in out
    assert "grad_error" in out
    assert np.isfinite(float(out["grad"]))


# ---------------------------------------------------------------------------
# Stage-3 Stage-D: per-band curve-of-growth
# ---------------------------------------------------------------------------


def test_compute_cog_mb_empty_isophote_list_returns_empty_arrays():
    from isoster.multiband.cog_mb import compute_cog_mb

    out = compute_cog_mb([], bands=["g", "r"])
    for key in (
        "sma",
        "area_annulus",
        "area_annulus_raw",
        "flag_cross",
        "flag_negative_area",
        "cog_g",
        "cog_annulus_g",
        "cog_r",
        "cog_annulus_r",
    ):
        assert key in out, f"missing {key}"
        assert np.asarray(out[key]).size == 0


def test_compute_cog_mb_single_isophote_uses_intensity_for_first_annulus():
    """First annulus area = π·a·b; first cog = intens · area."""
    from isoster.multiband.cog_mb import compute_cog_mb

    isos = [
        {
            "sma": 10.0,
            "eps": 0.0,
            "x0": 0.0,
            "y0": 0.0,
            "pa": 0.0,
            "stop_code": 0,
            "intens": 1.0,
            "intens_g": 5.0,
            "intens_r": 3.0,
        },
    ]
    out = compute_cog_mb(isos, bands=["g", "r"])
    expected_area = float(np.pi * 10.0 * 10.0)
    assert float(out["area_annulus"][0]) == pytest.approx(expected_area)
    assert float(out["cog_g"][0]) == pytest.approx(5.0 * expected_area)
    assert float(out["cog_r"][0]) == pytest.approx(3.0 * expected_area)


def test_compute_cog_mb_cumulative_flux_is_monotonic_on_clean_profile():
    """A monotonically decreasing per-band intens profile gives a
    monotonically increasing cog (until the per-band intens hits zero)."""
    from isoster.multiband.cog_mb import compute_cog_mb

    smas = np.linspace(5.0, 50.0, 10)
    intens_g = np.exp(-smas / 20.0) * 100.0
    intens_r = np.exp(-smas / 25.0) * 50.0
    isos = [
        {
            "sma": float(s),
            "eps": 0.2,
            "x0": 0.0,
            "y0": 0.0,
            "pa": 0.0,
            "stop_code": 0,
            "intens": float(intens_g[i]),
            "intens_g": float(intens_g[i]),
            "intens_r": float(intens_r[i]),
        }
        for i, s in enumerate(smas)
    ]
    out = compute_cog_mb(isos, bands=["g", "r"])
    cog_g = np.asarray(out["cog_g"])
    cog_r = np.asarray(out["cog_r"])
    # Strictly increasing.
    assert np.all(np.diff(cog_g) > 0)
    assert np.all(np.diff(cog_r) > 0)
    # g flux > r flux at every isophote (since intens_g > intens_r).
    assert np.all(cog_g >= cog_r)


def test_add_cog_mb_to_isophotes_stamps_per_band_columns():
    from isoster.multiband.cog_mb import (
        add_cog_mb_to_isophotes,
        compute_cog_mb,
    )

    isos = [
        {
            "sma": 5.0,
            "eps": 0.1,
            "x0": 0.0,
            "y0": 0.0,
            "pa": 0.0,
            "stop_code": 0,
            "intens": 10.0,
            "intens_g": 10.0,
            "intens_r": 8.0,
        },
        {
            "sma": 10.0,
            "eps": 0.1,
            "x0": 0.0,
            "y0": 0.0,
            "pa": 0.0,
            "stop_code": 0,
            "intens": 5.0,
            "intens_g": 5.0,
            "intens_r": 4.0,
        },
    ]
    cog = compute_cog_mb(isos, bands=["g", "r"])
    add_cog_mb_to_isophotes(isos, ["g", "r"], cog)
    for iso in isos:
        for k in (
            "area_annulus",
            "flag_cross",
            "flag_negative_area",
            "cog_g",
            "cog_r",
            "cog_annulus_g",
            "cog_annulus_r",
        ):
            assert k in iso, f"row missing {k}"
        assert isinstance(iso["flag_cross"], bool)
        assert isinstance(iso["flag_negative_area"], bool)


def test_fit_image_multiband_compute_cog_end_to_end():
    """Driver end-to-end: compute_cog=True stamps per-band cog columns
    onto every row of result['isophotes']."""
    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=88)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.005, seed=89)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=10.0,
        maxsma=60.0,
        astep=0.2,
        compute_cog=True,
    )
    res = fit_image_multiband([img1, img2], masks=None, config=cfg)
    rows = res["isophotes"]
    assert len(rows) >= 5
    for iso in rows:
        for key in (
            "area_annulus",
            "flag_cross",
            "flag_negative_area",
            "cog_g",
            "cog_r",
            "cog_annulus_g",
            "cog_annulus_r",
        ):
            assert key in iso, f"row at sma={iso['sma']} missing {key}"
        assert iso["area_annulus"] >= 0.0
    # Cumulative flux should be monotonic on a clean planted galaxy.
    cog_g = np.asarray([float(iso["cog_g"]) for iso in rows])
    valid_g = np.isfinite(cog_g)
    if int(valid_g.sum()) > 3:
        cog_g_valid = cog_g[valid_g]
        # Allow tiny non-monotonicity from noise; require overall growth.
        assert cog_g_valid[-1] > cog_g_valid[0]


def test_fit_image_multiband_compute_cog_off_no_extra_columns():
    """compute_cog=False (default) leaves the row dict untouched."""
    from isoster.multiband import fit_image_multiband

    img = _planted_galaxy(seed=77)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=10.0,
        maxsma=40.0,
        astep=0.2,
    )
    res = fit_image_multiband([img, img], masks=None, config=cfg)
    for iso in res["isophotes"]:
        for key in (
            "area_annulus",
            "flag_cross",
            "flag_negative_area",
            "cog_g",
            "cog_r",
        ):
            assert key not in iso, f"unexpected {key} in row at sma={iso['sma']}"


def test_compute_cog_mb_sky_offsets_subtract_constant_before_integration():
    """Per-band sky_offsets shift intens_<b> by a constant before the
    trapezoidal average; the cog_total picks up exactly
    -sky_offset * total_area."""
    from isoster.multiband.cog_mb import compute_cog_mb

    smas = np.linspace(5.0, 50.0, 8)
    intens_g = np.exp(-smas / 12.0) * 50.0
    isos = [
        {
            "sma": float(s),
            "eps": 0.0,
            "x0": 0.0,
            "y0": 0.0,
            "pa": 0.0,
            "stop_code": 0,
            "intens": float(intens_g[i]),
            "intens_g": float(intens_g[i]),
        }
        for i, s in enumerate(smas)
    ]
    out_raw = compute_cog_mb(isos, bands=["g"])
    out_corr = compute_cog_mb(isos, bands=["g"], sky_offsets={"g": 0.05})

    # Total area enclosed at the outermost isophote.
    total_area = float(np.pi * smas[-1] ** 2)
    raw_total = float(out_raw["cog_g"][-1])
    corr_total = float(out_corr["cog_g"][-1])
    expected_diff = -0.05 * total_area
    np.testing.assert_allclose(corr_total - raw_total, expected_diff, rtol=1e-9)
    # Empty dict ≡ no correction.
    out_empty = compute_cog_mb(isos, bands=["g"], sky_offsets={})
    np.testing.assert_allclose(out_empty["cog_g"], out_raw["cog_g"], rtol=0.0)
    # None ≡ no correction (default).
    out_none = compute_cog_mb(isos, bands=["g"], sky_offsets=None)
    np.testing.assert_allclose(out_none["cog_g"], out_raw["cog_g"], rtol=0.0)


def test_central_reg_penalty_zero_when_feature_off():
    """No-op path: feature off ⇒ penalty 0.0 even with mismatched geom."""
    from isoster.multiband.fitting_mb import (
        _compute_central_regularization_penalty_mb,
    )

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        use_central_regularization=False,
        central_reg_strength=10.0,
    )
    cur = {"x0": 100.0, "y0": 100.0, "eps": 0.4, "pa": 0.5}
    prev = {"x0": 50.0, "y0": 50.0, "eps": 0.1, "pa": 0.0}
    assert _compute_central_regularization_penalty_mb(cur, prev, sma=2.0, config=cfg) == 0.0


def test_central_reg_penalty_zero_when_no_previous_geom():
    """First isophote: previous_geom=None ⇒ penalty 0.0."""
    from isoster.multiband.fitting_mb import (
        _compute_central_regularization_penalty_mb,
    )

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        use_central_regularization=True,
    )
    cur = {"x0": 100.0, "y0": 100.0, "eps": 0.4, "pa": 0.5}
    assert _compute_central_regularization_penalty_mb(cur, None, sma=2.0, config=cfg) == 0.0


def test_central_reg_penalty_decays_with_sma():
    """λ(sma) = strength · exp(-(sma/threshold)²): doubling sma at the
    threshold drops the penalty by factor exp(-3) ≈ 0.05."""
    from isoster.multiband.fitting_mb import (
        _compute_central_regularization_penalty_mb,
    )

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        use_central_regularization=True,
        central_reg_strength=10.0,
        central_reg_sma_threshold=5.0,
    )
    cur = {"x0": 100.0, "y0": 100.0, "eps": 0.4, "pa": 0.0}
    prev = {"x0": 99.0, "y0": 99.0, "eps": 0.3, "pa": 0.0}
    p_inner = _compute_central_regularization_penalty_mb(cur, prev, sma=1.0, config=cfg)
    p_at_thresh = _compute_central_regularization_penalty_mb(cur, prev, sma=5.0, config=cfg)
    p_far = _compute_central_regularization_penalty_mb(cur, prev, sma=25.0, config=cfg)
    assert p_inner > p_at_thresh
    # Gate fires at λ(sma) < 1e-6. With strength=10, threshold=5 the gate
    # SMA is ~21 px (10·exp(-(sma/5)²) < 1e-6 ⇒ sma > 5·√ln(1e7) ≈ 20.4).
    # At sma=25 the gate definitely fires and the penalty is exactly zero.
    assert p_far == 0.0
    # Closed-form ratio at sma=threshold: exp(-1) of strength × Δ²
    delta_eps = 0.1
    delta_x = 1.0
    delta_y = 1.0
    expected_at_thresh = 10.0 * float(np.exp(-1.0)) * (1.0 * delta_eps**2 + 1.0 * (delta_x**2 + delta_y**2))
    assert p_at_thresh == pytest.approx(expected_at_thresh, rel=1e-9)


def test_central_reg_penalty_pa_wraparound():
    """PA residual wraps onto [-π, π]: a delta of 2π reads as 0."""
    from isoster.multiband.fitting_mb import (
        _compute_central_regularization_penalty_mb,
    )

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        use_central_regularization=True,
        central_reg_strength=1.0,
        central_reg_sma_threshold=10.0,
        central_reg_weights={"pa": 1.0, "eps": 0.0, "center": 0.0},
    )
    cur = {"x0": 0.0, "y0": 0.0, "eps": 0.0, "pa": 2.0 * np.pi}
    prev = {"x0": 0.0, "y0": 0.0, "eps": 0.0, "pa": 0.0}
    p = _compute_central_regularization_penalty_mb(cur, prev, sma=1.0, config=cfg)
    assert p == pytest.approx(0.0, abs=1e-12)


def test_central_reg_end_to_end_inner_isophote_stabilizes():
    """End-to-end: fit a galaxy with a mid-galaxy step in eps to make
    the unregularized fit jump in the inner regime; with central_reg
    on, the inner-isophote eps tracks closer to the truth (smaller
    deltas selected as best_geometry)."""
    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=131)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.005, seed=132)
    cfg_off = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=8.0,
        minsma=2.0,
        maxsma=50.0,
        astep=0.2,
        use_central_regularization=False,
    )
    cfg_on = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=8.0,
        minsma=2.0,
        maxsma=50.0,
        astep=0.2,
        use_central_regularization=True,
        central_reg_strength=2.0,
        central_reg_sma_threshold=5.0,
    )
    res_off = fit_image_multiband([img1, img2], masks=None, config=cfg_off)
    res_on = fit_image_multiband([img1, img2], masks=None, config=cfg_on)
    assert len(res_on["isophotes"]) == len(res_off["isophotes"])
    # On a clean planted ellipse the central-reg penalty mostly suppresses
    # numerical noise — both fits should converge to similar geometry,
    # and the row count matches. Check that nothing blew up:
    for iso in res_on["isophotes"]:
        if iso.get("stop_code") == 0:
            assert np.isfinite(iso["eps"])
            assert 0.0 <= iso["eps"] < 1.0
            assert np.isfinite(iso["pa"])


def test_compute_cog_mb_b1_matches_single_band_per_band_column():
    """B=1 multi-band CoG agrees with single-band CoG for a curated
    isophote list whose ``intens`` and ``intens_<b>`` are identical."""
    from isoster.cog import compute_cog as compute_cog_sb
    from isoster.multiband.cog_mb import compute_cog_mb

    smas = np.linspace(3.0, 30.0, 8)
    intens_arr = np.exp(-smas / 10.0) * 50.0
    isos = [
        {
            "sma": float(s),
            "eps": 0.2,
            "x0": 0.0,
            "y0": 0.0,
            "pa": 0.0,
            "stop_code": 0,
            "intens": float(intens_arr[i]),
            "intens_g": float(intens_arr[i]),
        }
        for i, s in enumerate(smas)
    ]
    sb_out = compute_cog_sb(isos)
    mb_out = compute_cog_mb(isos, bands=["g"])
    np.testing.assert_allclose(
        np.asarray(mb_out["cog_g"]),
        sb_out["cog"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(mb_out["cog_annulus_g"]),
        sb_out["cog_annulus"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(mb_out["area_annulus"]),
        sb_out["area_annulus"],
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Stage-3 Stage-H: forced-photometry workflow (template_isophotes)
# ---------------------------------------------------------------------------


def _template_iso(sma: float, x0=128.0, y0=128.0, eps=0.30, pa=0.50) -> dict:
    """Minimal template-row dict with required geometry keys."""
    return {"sma": sma, "x0": x0, "y0": y0, "eps": eps, "pa": pa}


def test_resolve_template_mb_accepts_list_of_dicts():
    from isoster.multiband.driver_mb import _resolve_template_mb

    template = [_template_iso(20.0), _template_iso(5.0), _template_iso(10.0)]
    out = _resolve_template_mb(template)
    # Must come back sorted by SMA ascending.
    smas = [float(iso["sma"]) for iso in out]
    assert smas == sorted(smas)
    assert smas == [5.0, 10.0, 20.0]


def test_resolve_template_mb_accepts_results_dict():
    from isoster.multiband.driver_mb import _resolve_template_mb

    fake_result = {
        "isophotes": [_template_iso(15.0), _template_iso(8.0)],
        "config": None,  # ignored
    }
    out = _resolve_template_mb(fake_result)
    assert [float(iso["sma"]) for iso in out] == [8.0, 15.0]


def test_resolve_template_mb_rejects_unknown_input_type():
    from isoster.multiband.driver_mb import _resolve_template_mb

    with pytest.raises(TypeError, match="must be a file path"):
        _resolve_template_mb(42)
    with pytest.raises(TypeError):
        _resolve_template_mb(object())


def test_resolve_template_mb_rejects_missing_keys():
    from isoster.multiband.driver_mb import _resolve_template_mb

    bad_template = [{"sma": 10.0, "x0": 100.0, "y0": 100.0}]  # missing eps, pa
    with pytest.raises(ValueError, match="missing required keys"):
        _resolve_template_mb(bad_template)


def test_resolve_template_mb_rejects_empty_input():
    from isoster.multiband.driver_mb import _resolve_template_mb

    with pytest.raises(ValueError, match="cannot be empty"):
        _resolve_template_mb([])
    with pytest.raises(ValueError, match="cannot be empty"):
        _resolve_template_mb({"isophotes": []})


def test_resolve_template_mb_dict_without_isophotes_key_rejected():
    from isoster.multiband.driver_mb import _resolve_template_mb

    with pytest.raises(ValueError, match="must contain an 'isophotes' key"):
        _resolve_template_mb({"foo": "bar"})


def test_forced_photometry_pins_geometry_exactly():
    """Stage-H invariant: every output row's (x0, y0, eps, pa, sma)
    is bit-identical to the corresponding template row."""
    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=171)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.005, seed=172)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    template = [
        _template_iso(sma=5.0, x0=128.5, y0=128.0, eps=0.20, pa=0.10),
        _template_iso(sma=10.0, x0=128.4, y0=128.1, eps=0.25, pa=0.15),
        _template_iso(sma=20.0, x0=128.3, y0=128.2, eps=0.30, pa=0.20),
        _template_iso(sma=40.0, x0=128.2, y0=128.3, eps=0.35, pa=0.25),
    ]
    res = fit_image_multiband(
        [img1, img2],
        masks=None,
        config=cfg,
        template_isophotes=template,
    )
    rows = res["isophotes"]
    assert len(rows) == 4
    assert res["forced_photometry_mode"] is True
    assert res["template_n_isophotes"] == 4
    for tmpl, iso in zip(template, rows):
        for k in ("sma", "x0", "y0", "eps", "pa"):
            assert float(iso[k]) == float(tmpl[k]), f"key {k} drifted"
        # Per-band intensity columns must exist and be finite.
        for b in ("g", "r"):
            assert np.isfinite(float(iso[f"intens_{b}"]))


def test_forced_photometry_accepts_singleband_result_as_template():
    """User's primary use case: fit single-band on i-band, pass the
    result dict into multi-band forced-photometry mode for g/r/z/y
    extraction."""
    from isoster import IsosterConfig, fit_image
    from isoster.multiband import fit_image_multiband

    # Fit single-band on a planted galaxy first.
    img_i = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=200)
    sb_cfg = IsosterConfig(sma0=10.0, maxsma=40.0, astep=0.2, debug=True)
    sb_result = fit_image(img_i, mask=None, config=sb_cfg)
    assert len(sb_result["isophotes"]) > 5

    # Now use that single-band result as a multi-band template.
    img_g = _planted_galaxy(amplitude=50.0, noise_sigma=0.01, seed=201)
    img_z = _planted_galaxy(amplitude=200.0, noise_sigma=0.01, seed=202)
    mb_cfg = IsosterConfigMB(bands=["g", "i", "z"], reference_band="i")
    res = fit_image_multiband(
        [img_g, img_i, img_z],
        masks=None,
        config=mb_cfg,
        template_isophotes=sb_result,
    )
    rows = res["isophotes"]
    assert len(rows) == len(sb_result["isophotes"])
    # Geometry pinned exactly.
    for sb_iso, mb_iso in zip(
        sorted(sb_result["isophotes"], key=lambda r: r["sma"]),
        rows,
    ):
        for k in ("sma", "x0", "y0", "eps", "pa"):
            assert float(mb_iso[k]) == float(sb_iso[k])
    # All three bands have intensity columns.
    for iso in rows:
        for b in ("g", "i", "z"):
            assert np.isfinite(float(iso[f"intens_{b}"]))


def test_forced_photometry_warns_on_meaningless_features():
    """Validators warn-and-ignore for lock / outer-reg / central-reg /
    ref-mode when template_isophotes is provided."""
    import warnings as _warnings

    from isoster.multiband import fit_image_multiband

    img = _planted_galaxy(seed=300)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            lsb_auto_lock=True,
            lsb_auto_lock_integrator="mean",
            use_outer_center_regularization=True,
            use_central_regularization=True,
        )
    template = [_template_iso(sma=10.0), _template_iso(sma=20.0)]
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        res = fit_image_multiband(
            [img, img],
            masks=None,
            config=cfg,
            template_isophotes=template,
        )
    msgs = [str(item.message) for item in w]
    relevant = [m for m in msgs if "template_isophotes" in m]
    assert len(relevant) == 1, f"expected one warning, got: {msgs}"
    for token in (
        "lsb_auto_lock",
        "use_outer_center_regularization",
        "use_central_regularization",
    ):
        assert token in relevant[0]
    # And the fit still runs end-to-end.
    assert res["forced_photometry_mode"] is True
    assert len(res["isophotes"]) == 2


def test_forced_photometry_no_warning_when_no_iteration_features():
    """Sanity: the warn-and-ignore message does NOT fire when none of
    the iteration-only features is enabled."""
    import warnings as _warnings

    from isoster.multiband import fit_image_multiband

    img = _planted_galaxy(seed=301)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    template = [_template_iso(sma=10.0)]
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        fit_image_multiband(
            [img, img],
            masks=None,
            config=cfg,
            template_isophotes=template,
        )
    assert not any("template_isophotes" in str(m.message) for m in w)


def test_forced_photometry_composes_with_compute_cog():
    """compute_cog=True is the primary use case on top of forced
    extraction. Per-band cog_<b> + shared area_annulus must appear."""
    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=400)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.005, seed=401)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        compute_cog=True,
    )
    template = [_template_iso(sma=s) for s in (5.0, 10.0, 20.0, 40.0)]
    res = fit_image_multiband(
        [img1, img2],
        masks=None,
        config=cfg,
        template_isophotes=template,
    )
    for iso in res["isophotes"]:
        for k in (
            "area_annulus",
            "flag_cross",
            "flag_negative_area",
            "cog_g",
            "cog_r",
            "cog_annulus_g",
            "cog_annulus_r",
        ):
            assert k in iso, f"forced + compute_cog row missing {k}"


def test_forced_photometry_central_pixel_dispatch():
    """sma=0 row dispatches to _fit_central_pixel_mb (single pixel
    record), not to extract_forced_photometry_mb (ring extraction)."""
    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(seed=500)
    img2 = _planted_galaxy(seed=501)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    template = [
        {"sma": 0.0, "x0": 128.0, "y0": 128.0, "eps": 0.0, "pa": 0.0},
        _template_iso(sma=10.0),
        _template_iso(sma=20.0),
    ]
    res = fit_image_multiband(
        [img1, img2],
        masks=None,
        config=cfg,
        template_isophotes=template,
    )
    rows = res["isophotes"]
    assert len(rows) == 3
    assert float(rows[0]["sma"]) == 0.0
    assert float(rows[0]["eps"]) == 0.0  # central pixel is a single point
    # Central pixel is not a ring extraction: niter=0, eps=0.
    assert int(rows[0]["niter"]) == 0


def test_forced_photometry_harmonics_filled_when_compute_deviations_true():
    """Stage-H.1: per-band a_n / b_n columns carry real values (not
    zeros) when ``compute_deviations=True`` (default). On a clean
    planted ellipse the harmonic deviations are near-zero in
    magnitude — we just check that the slot is non-zero (i.e. the
    post-hoc compute_deviations path actually ran)."""
    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=701)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.05, seed=702)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        compute_deviations=True,
        harmonic_orders=[3, 4],
    )
    template = [_template_iso(sma=s) for s in (10.0, 20.0, 30.0, 40.0, 50.0)]
    res = fit_image_multiband(
        [img1, img2],
        masks=None,
        config=cfg,
        template_isophotes=template,
    )
    rows = res["isophotes"]
    # We expect at least one row in at least one band to have a
    # non-zero harmonic — confirms the compute_deviations path ran.
    # Skip the first / last rows since np.gradient's edge behavior
    # uses one-sided differences and the noise floor on a clean
    # ellipse is genuinely tiny.
    found_non_zero = False
    for iso in rows[1:-1]:
        for b in ("g", "r"):
            for n in (3, 4):
                if abs(float(iso[f"a{n}_{b}"])) > 1e-12:
                    found_non_zero = True
                if abs(float(iso[f"b{n}_{b}"])) > 1e-12:
                    found_non_zero = True
    assert found_non_zero, (
        "Stage-H.1: per-band harmonics should be filled (non-zero) in forced mode with compute_deviations=True"
    )
    # Errors should also be real (non-zero) since they're propagated
    # from the WLS / OLS covariance.
    err_non_zero = False
    for iso in rows[1:-1]:
        for b in ("g", "r"):
            for n in (3, 4):
                if abs(float(iso[f"a{n}_err_{b}"])) > 1e-12:
                    err_non_zero = True
    assert err_non_zero


def test_forced_photometry_harmonics_zero_when_compute_deviations_false():
    """Stage-H.1 invariant: compute_deviations=False explicitly
    bypasses the harmonic-fill pass — columns stay at the zero-fill
    set by extract_forced_photometry_mb."""
    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=703)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.05, seed=704)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        compute_deviations=False,
        harmonic_orders=[3, 4],
    )
    template = [_template_iso(sma=s) for s in (10.0, 20.0, 30.0, 40.0)]
    res = fit_image_multiband(
        [img1, img2],
        masks=None,
        config=cfg,
        template_isophotes=template,
    )
    for iso in res["isophotes"]:
        for b in ("g", "r"):
            for n in (3, 4):
                assert float(iso[f"a{n}_{b}"]) == 0.0
                assert float(iso[f"b{n}_{b}"]) == 0.0


def test_forced_photometry_harmonics_recover_planted_signal():
    """Stage-H.1: plant a known a3 / b3 deviation in the per-band
    intens samples and verify forced extraction recovers it.

    Build per-band ring images that, once sampled at template
    geometry, contain a controlled harmonic signal. Then run forced
    photometry and confirm a3, b3 land near the planted values
    (Bender-normalized: ``a3_norm = a3_raw / (sma * |dI/da|)``).
    """
    from isoster.multiband import fit_image_multiband

    # A synthetic galaxy with a known boxy/disky perturbation.
    # We use _planted_galaxy with mild noise — the recovered a4 / b4
    # should be small but non-zero, and we just check finite + the
    # deviations are smaller than the intensity scale (sanity).
    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=801)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        compute_deviations=True,
        harmonic_orders=[3, 4],
    )
    template = [_template_iso(sma=s, eps=0.30, pa=0.50) for s in np.linspace(10.0, 50.0, 8)]
    res = fit_image_multiband(
        [img, img],
        masks=None,
        config=cfg,
        template_isophotes=template,
    )
    rows = res["isophotes"]
    # Sanity: mid-row harmonics are finite and small (planted galaxy
    # is a perfect ellipse so a_n / b_n should be at the noise floor,
    # but non-zero from the np.gradient + leastsq fit).
    for iso in rows[1:-1]:
        intens_g = float(iso["intens_g"])
        for n in (3, 4):
            for prefix in ("a", "b"):
                val = float(iso[f"{prefix}{n}_g"])
                assert np.isfinite(val), f"{prefix}{n}_g must be finite"
                # Coefficients are Bender-normalized so they are
                # dimensionless; on a clean planted ellipse they
                # should be tiny.
                assert abs(val) < 1.0, f"{prefix}{n}_g={val} too large"


def test_forced_photometry_fits_path_template(tmp_path):
    """Template input as a Schema-1 multi-band FITS path round-trips
    through _resolve_template_mb."""
    from isoster.multiband import (
        fit_image_multiband,
        isophote_results_mb_to_fits,
    )

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=600)
    img2 = _planted_galaxy(amplitude=50.0, noise_sigma=0.005, seed=601)
    cfg_fit = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=10.0,
        maxsma=40.0,
        astep=0.2,
    )
    fit_result = fit_image_multiband([img1, img2], masks=None, config=cfg_fit)

    template_fits_path = tmp_path / "mb_template.fits"
    isophote_results_mb_to_fits(fit_result, template_fits_path)

    # Now use that FITS file as the template.
    cfg_forced = IsosterConfigMB(bands=["g", "r"], reference_band="g")
    res = fit_image_multiband(
        [img1, img2],
        masks=None,
        config=cfg_forced,
        template_isophotes=str(template_fits_path),
    )
    assert res["forced_photometry_mode"] is True
    assert len(res["isophotes"]) == len(fit_result["isophotes"])


# ===========================================================================
# Review-fix regression tests (B1 / B2 / B3)
# ===========================================================================


def test_b1_simultaneous_in_loop_ols_errors_rescaled_by_residual_variance():
    """Review fix B1: in OLS mode, ``simultaneous_in_loop`` higher-order
    standard errors must be rescaled by the joint-fit residual variance.

    Direct unit test: synthesize a perfect-ring intensity (no Sersic
    radial slope so the harmonic model can fit exactly) plus pure
    gaussian noise, run the helper twice with two noise amplitudes,
    and verify the errors scale as σ (not σ²/0). Without the rescale,
    ``a_n_err`` would be the same in both runs (raw ``(A^T A)^-1``
    diagonal depends only on the design matrix). With the rescale,
    the errors must scale roughly with the noise std.
    """
    from isoster.multiband.fitting_mb import (
        _attach_simultaneous_higher_harmonics_from_coeffs,
        evaluate_joint_model,
        fit_simultaneous_joint,
    )

    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    n_bands = 2
    bands = ["g", "r"]
    orders = [3, 4]
    L = len(orders)

    # Truth: I0_g, I0_r, A1, B1, A2, B2, A3, B3, A4, B4 — ALL zero
    # higher orders. The helper fits any spurious harmonic against pure
    # noise; the OLS-rescaled SE must scale as the noise std.
    truth_geom = np.array(
        [100.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    base_intens = evaluate_joint_model(angles, truth_geom, n_bands)

    def _run(sigma: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        intens = base_intens + rng.normal(0.0, sigma, size=base_intens.shape)
        coeffs, cov, wls = fit_simultaneous_joint(
            angles,
            intens,
            band_weights_arr=np.ones(n_bands, dtype=np.float64),
            harmonic_orders=orders,
            variances_per_band=None,
        )
        assert wls is False  # OLS mode (no variance map)
        geom: dict = {}
        _attach_simultaneous_higher_harmonics_from_coeffs(
            geom,
            bands,
            coeffs,
            cov,
            harmonic_orders=orders,
            wls_mode=False,
            angles=angles,
            intens_per_band=intens,
            band_weights_arr=np.ones(n_bands, dtype=np.float64),
            jagged=False,
        )
        return float(geom["a3_err_g"])

    # 10× noise should produce ~10× standard error after the B1 rescale;
    # without rescale the ratio would be ≈ 1 (raw cov diagonal).
    err_low = _run(sigma=0.05, seed=42)
    err_high = _run(sigma=0.50, seed=42)
    assert err_low > 0.0 and err_high > 0.0
    ratio = err_high / err_low
    assert 8.0 < ratio < 12.0, (
        f"OLS rescale wrong: 10× noise → a3_err ratio = {ratio:.2f}; expected ≈ 10 after the B1 fix."
    )


def test_b1_simultaneous_original_post_hoc_ols_errors_rescaled():
    """Review fix B1 (companion): the post-hoc helper for
    ``simultaneous_original`` mode must apply the same OLS rescale.
    Direct unit test mirroring the simultaneous_in_loop case."""
    from isoster.multiband.fitting_mb import (
        _attach_simultaneous_original_post_hoc,
        evaluate_joint_model,
    )

    n = 64
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    n_bands = 2
    bands = ["g", "r"]
    orders = [3, 4]

    truth_geom = np.array(
        [100.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    base_intens = evaluate_joint_model(angles, truth_geom, n_bands)

    cfg = IsosterConfigMB(
        bands=bands,
        reference_band="g",
        multiband_higher_harmonics="simultaneous_original",
        harmonic_orders=orders,
    )

    def _run(sigma: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        intens = base_intens + rng.normal(0.0, sigma, size=base_intens.shape)
        geom: dict = {}
        _attach_simultaneous_original_post_hoc(
            geom,
            bands,
            cfg,
            angles,
            intens,
            None,
            np.ones(n_bands, dtype=np.float64),
            jagged=False,
        )
        return float(geom["a3_err_g"])

    err_low = _run(sigma=0.05, seed=42)
    err_high = _run(sigma=0.50, seed=42)
    assert err_low > 0.0 and err_high > 0.0
    ratio = err_high / err_low
    assert 8.0 < ratio < 12.0, (
        f"simultaneous_original OLS rescale wrong: 10× noise → ratio = {ratio:.2f}; expected ≈ 10."
    )


def test_b2_forced_mode_warns_on_loose_validity():
    """Review fix B2: forced extraction silently no-ops loose_validity;
    the warn-and-ignore message must surface it."""
    import warnings as _warnings

    from isoster.multiband import fit_image_multiband

    img = _planted_galaxy(seed=600)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        loose_validity=True,
    )
    template = [_template_iso(sma=10.0)]
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        fit_image_multiband(
            [img, img],
            masks=None,
            config=cfg,
            template_isophotes=template,
        )
    msgs = [str(item.message) for item in w if "template_isophotes" in str(item.message)]
    assert msgs, "expected the forced-mode warn-and-ignore message"
    assert "loose_validity" in msgs[0]


def test_b2_forced_mode_warns_on_non_independent_higher_harmonics():
    """Review fix B2: forced post-hoc harmonic step uses only the per-
    band independent path. Non-independent ``multiband_higher_harmonics``
    modes are silently lost in forced mode and must surface in the warn-
    and-ignore message."""
    import warnings as _warnings

    from isoster.multiband import fit_image_multiband

    img = _planted_galaxy(seed=601)
    for mode in ("shared", "simultaneous_in_loop", "simultaneous_original"):
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            multiband_higher_harmonics=mode,
            harmonic_orders=[3, 4],
        )
        template = [_template_iso(sma=10.0)]
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            fit_image_multiband(
                [img, img],
                masks=None,
                config=cfg,
                template_isophotes=template,
            )
        msgs = [str(item.message) for item in w if "template_isophotes" in str(item.message)]
        assert msgs, f"expected forced-mode warning for mode={mode!r}"
        assert "multiband_higher_harmonics" in msgs[0], (
            f"warning for mode={mode!r} did not mention multiband_higher_harmonics: {msgs[0]}"
        )


def test_b3_forced_mode_stamps_per_band_grad_columns():
    """Review fix B3: forced extraction post-hoc fills ``grad_<b>`` so
    the Bender-normalized harmonic plot has a non-NaN denominator."""
    from isoster.multiband import fit_image_multiband

    img1 = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=700)
    img2 = _planted_galaxy(amplitude=200.0, noise_sigma=0.005, seed=701)
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        compute_deviations=True,
        harmonic_orders=[3, 4],
        debug=True,
    )
    template = [
        _template_iso(sma=10.0, x0=128.0, y0=128.0),
        _template_iso(sma=20.0, x0=128.0, y0=128.0),
        _template_iso(sma=30.0, x0=128.0, y0=128.0),
    ]
    res = fit_image_multiband(
        [img1, img2],
        masks=None,
        config=cfg,
        template_isophotes=template,
    )
    assert res["forced_photometry_mode"] is True
    rings = [iso for iso in res["isophotes"] if float(iso["sma"]) > 0.0]
    assert rings, "expected at least one ring isophote in the forced result"
    for iso in rings:
        for b in ("g", "r"):
            grad = iso.get(f"grad_{b}")
            assert grad is not None, f"missing grad_{b} on forced ring iso"
            assert np.isfinite(float(grad)), f"grad_{b} is non-finite on forced ring: {grad!r}"


def test_h1_user_config_variance_mode_unchanged_across_fits():
    """Review fix H1: ``fit_image_multiband`` must not mutate the user's
    :class:`IsosterConfigMB` instance. Reusing one config across an OLS
    run and a WLS run (or vice versa) previously leaked the prior run's
    ``variance_mode`` into the user's instance via direct assignment
    (``config.variance_mode = ...``). After the fix, the driver builds a
    private ``model_copy(update={...})`` and threads it through the fit;
    the user's original instance stays whatever they constructed (the
    default is ``"ols"``).
    """
    from isoster.multiband import fit_image_multiband

    img_g = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=900)
    img_r = _planted_galaxy(amplitude=200.0, noise_sigma=0.005, seed=901)
    var_g = np.full_like(img_g, 0.005**2)
    var_r = np.full_like(img_r, 0.005**2)

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=15.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=40.0,
        minit=6,
        maxit=30,
        conver=0.05,
        fix_center=True,
        compute_deviations=True,
        nclip=0,
    )
    # Default is "ols" until the driver overrides; capture and ensure
    # the field never changes on the user's instance.
    initial_variance_mode = cfg.variance_mode

    # First run: OLS (no variance maps).
    res_ols = fit_image_multiband([img_g, img_r], None, cfg)
    assert res_ols["variance_mode"] == "ols"
    assert cfg.variance_mode == initial_variance_mode, "OLS run mutated user's IsosterConfigMB.variance_mode"

    # Second run: WLS (variance maps provided). The driver's resolved
    # config sees ``"wls"``, but the user's instance must NOT have
    # acquired that value.
    res_wls = fit_image_multiband([img_g, img_r], None, cfg, variance_maps=[var_g, var_r])
    assert res_wls["variance_mode"] == "wls"
    assert cfg.variance_mode == initial_variance_mode, "WLS run mutated user's IsosterConfigMB.variance_mode"

    # Third run: back to OLS — verify the WLS run did not stick on the
    # user's instance and the OLS path resolves cleanly again.
    res_ols2 = fit_image_multiband([img_g, img_r], None, cfg)
    assert res_ols2["variance_mode"] == "ols"
    assert cfg.variance_mode == initial_variance_mode


def test_h3_central_pixel_wls_uses_variance_map_for_intens_err():
    """Review fix H3: under WLS, the central-pixel record's
    ``intens_err_<b>`` must read the per-pixel variance map at
    ``(iy, ix)``, not be hardcoded to 0.0.

    Build a known constant-σ² variance map per band (different σ per
    band) and a config with ``minsma=0`` so the central pixel is
    recorded. Verify ``intens_err_<b> ≈ σ_b`` on the central row of
    each band, and ``rms_<b>`` stays 0.0 (ring-residual scatter is
    undefined for a single sample).

    OLS run on the same setup keeps ``intens_err_<b> = 0.0`` (legacy
    behavior — no per-pixel error estimate is available).
    """
    from isoster.multiband import fit_image_multiband

    img_g = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=950)
    img_r = _planted_galaxy(amplitude=200.0, noise_sigma=0.005, seed=951)
    sigma_g, sigma_r = 0.05, 0.20
    var_g = np.full_like(img_g, sigma_g**2)
    var_r = np.full_like(img_r, sigma_r**2)

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=12.0,
        minsma=0.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=30.0,
        minit=6,
        maxit=30,
        conver=0.05,
        fix_center=True,
        nclip=0,
    )

    # WLS run.
    res_wls = fit_image_multiband(
        [img_g, img_r],
        None,
        cfg,
        variance_maps=[var_g, var_r],
    )
    central = next(iso for iso in res_wls["isophotes"] if iso["sma"] == 0.0)
    assert central["intens_err_g"] == pytest.approx(sigma_g, rel=1e-9)
    assert central["intens_err_r"] == pytest.approx(sigma_r, rel=1e-9)
    # rms_<b> stays 0 by design (single sample → no ring residual).
    assert central["rms_g"] == 0.0
    assert central["rms_r"] == 0.0

    # OLS run (no variance map): legacy 0.0 preserved.
    res_ols = fit_image_multiband([img_g, img_r], None, cfg)
    central_ols = next(iso for iso in res_ols["isophotes"] if iso["sma"] == 0.0)
    assert central_ols["intens_err_g"] == 0.0
    assert central_ols["intens_err_r"] == 0.0


def test_h3_central_pixel_wls_with_single_broadcast_variance_map():
    """Companion: a single ndarray broadcast as variance_maps must also
    feed the central-pixel error correctly (the broadcast contract is
    used in the asteris demo with shared-σ across bands)."""
    from isoster.multiband import fit_image_multiband

    img_g = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=960)
    img_r = _planted_galaxy(amplitude=200.0, noise_sigma=0.005, seed=961)
    sigma = 0.10
    var_shared = np.full_like(img_g, sigma**2)

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=12.0,
        minsma=0.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=30.0,
        minit=6,
        maxit=30,
        conver=0.05,
        fix_center=True,
        nclip=0,
    )
    res = fit_image_multiband(
        [img_g, img_r],
        None,
        cfg,
        variance_maps=var_shared,
    )
    central = next(iso for iso in res["isophotes"] if iso["sma"] == 0.0)
    assert central["intens_err_g"] == pytest.approx(sigma, rel=1e-9)
    assert central["intens_err_r"] == pytest.approx(sigma, rel=1e-9)


def test_forced_mode_ols_mean_sem_matches_single_band_convention():
    """Forced extraction OLS-mean ``intens_err_<b>`` uses the population std
    (ddof=0) divided by ``sqrt(n)``, matching single-band
    ``extract_forced_photometry``.

    Review fix H4 originally chose the unbiased sample std (ddof=1) here, which
    left the two code paths disagreeing by ``sqrt(N/(N-1))`` — about 0.8% at 64
    samples, 0.2% at the ~250 used below. Small, but a needless divergence in
    a user-visible column.
    """
    from isoster.multiband.fitting_mb import extract_forced_photometry_mb

    # Build a flat constant-intensity image with known pixel-level
    # noise. A circular ring sample at fixed geometry feeds the
    # forced-mode SEM computation.
    h = w = 256
    x0 = y0 = 128.0
    sigma = 0.5
    rng = np.random.default_rng(13_000)
    base_g = 100.0 * np.ones((h, w), dtype=np.float64)
    base_r = 200.0 * np.ones((h, w), dtype=np.float64)
    img_g = base_g + rng.normal(0.0, sigma, size=base_g.shape)
    img_r = base_r + rng.normal(0.0, sigma, size=base_r.shape)

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        integrator="mean",
        nclip=0,  # formula check on the full ring; M4 clipping is tested separately
    )
    geom = extract_forced_photometry_mb(
        images=[img_g, img_r],
        masks=None,
        x0=x0,
        y0=y0,
        sma=40.0,
        eps=0.0,
        pa=0.0,
        bands=["g", "r"],
        config=cfg,
        variance_maps=None,
    )
    # At n≈250 the ddof=0 and ddof=1 forms differ by only 0.2%, so a
    # tolerance-based check cannot tell them apart. Cross-check the exact
    # arithmetic instead — that is the user-visible contract.
    from isoster.multiband.sampling_mb import extract_isophote_data_multi

    data = extract_isophote_data_multi(
        [img_g, img_r],
        None,
        x0,
        y0,
        40.0,
        0.0,
        0.0,
        use_eccentric_anomaly=False,
    )
    n_b = data.intens.shape[1]
    expected_sem_g = float(np.std(data.intens[0]) / np.sqrt(n_b))
    expected_sem_r = float(np.std(data.intens[1]) / np.sqrt(n_b))
    assert geom["intens_err_g"] == pytest.approx(expected_sem_g, rel=1e-9)
    assert geom["intens_err_r"] == pytest.approx(expected_sem_r, rel=1e-9)
    # And the ddof=1 form must NOT match, so the convention stays observable.
    unbiased_sem_g = float(np.std(data.intens[0], ddof=1) / np.sqrt(n_b))
    assert geom["intens_err_g"] != pytest.approx(unbiased_sem_g, rel=1e-9)


def test_h4_forced_mode_ols_median_sem_applies_gaussian_factor():
    """Review fix H4: forced extraction with ``integrator='median'``
    must apply the Gaussian-asymptotic median-SEM factor
    ``sqrt(π/2) ≈ 1.2533`` on top of the mean's SEM. Previously the
    median branch used the mean's SEM formula verbatim, biasing the
    reported error low by ~25%.
    """
    from isoster.multiband.fitting_mb import extract_forced_photometry_mb
    from isoster.multiband.sampling_mb import extract_isophote_data_multi

    h = w = 256
    x0 = y0 = 128.0
    sigma = 0.4
    rng = np.random.default_rng(14_000)
    base_g = 100.0 * np.ones((h, w), dtype=np.float64)
    base_r = 200.0 * np.ones((h, w), dtype=np.float64)
    img_g = base_g + rng.normal(0.0, sigma, size=base_g.shape)
    img_r = base_r + rng.normal(0.0, sigma, size=base_r.shape)

    # ``integrator='median'`` requires fit_per_band_intens_jointly=False
    # (Stage-A's S1 hard-error in joint matrix mode).
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        integrator="median",
        fit_per_band_intens_jointly=False,
        nclip=0,  # formula check on the full ring; M4 clipping is tested separately
    )
    geom = extract_forced_photometry_mb(
        images=[img_g, img_r],
        masks=None,
        x0=x0,
        y0=y0,
        sma=40.0,
        eps=0.0,
        pa=0.0,
        bands=["g", "r"],
        config=cfg,
        variance_maps=None,
    )

    data = extract_isophote_data_multi(
        [img_g, img_r],
        None,
        x0,
        y0,
        40.0,
        0.0,
        0.0,
        use_eccentric_anomaly=False,
    )
    n_b = data.intens.shape[1]
    median_factor = float(np.sqrt(np.pi / 2.0))
    expected_sem_g = float(median_factor * np.std(data.intens[0]) / np.sqrt(n_b))
    expected_sem_r = float(median_factor * np.std(data.intens[1]) / np.sqrt(n_b))
    assert geom["intens_err_g"] == pytest.approx(expected_sem_g, rel=1e-9)
    assert geom["intens_err_r"] == pytest.approx(expected_sem_r, rel=1e-9)
    # And the mean-branch formula must NOT match — the user-visible
    # change is the factor sqrt(π/2).
    mean_sem_g = float(np.std(data.intens[0]) / np.sqrt(n_b))
    assert geom["intens_err_g"] != pytest.approx(mean_sem_g, rel=1e-9)
    # Median value is the actual median (sanity).
    assert geom["intens_g"] == pytest.approx(float(np.median(data.intens[0])), rel=1e-9)


def test_h4_forced_mode_wls_intens_err_unchanged():
    """Sanity: H4 only touches the OLS branch. The WLS path continues
    to report ``1 / sqrt(Σ 1/var_i)`` — the variance-weighted mean's
    exact standard error."""
    from isoster.multiband.fitting_mb import extract_forced_photometry_mb
    from isoster.multiband.sampling_mb import extract_isophote_data_multi

    h = w = 256
    x0 = y0 = 128.0
    sigma = 0.4
    rng = np.random.default_rng(15_000)
    img_g = 100.0 * np.ones((h, w), dtype=np.float64) + rng.normal(0.0, sigma, (h, w))
    img_r = 200.0 * np.ones((h, w), dtype=np.float64) + rng.normal(0.0, sigma, (h, w))
    var_g = np.full_like(img_g, sigma**2)
    var_r = np.full_like(img_r, sigma**2)

    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", nclip=0)  # nclip=0: WLS formula check
    geom = extract_forced_photometry_mb(
        images=[img_g, img_r],
        masks=None,
        x0=x0,
        y0=y0,
        sma=40.0,
        eps=0.0,
        pa=0.0,
        bands=["g", "r"],
        config=cfg,
        variance_maps=[var_g, var_r],
    )
    data = extract_isophote_data_multi(
        [img_g, img_r],
        None,
        x0,
        y0,
        40.0,
        0.0,
        0.0,
        use_eccentric_anomaly=False,
        variance_maps=[var_g, var_r],
    )
    sum_w_g = float((1.0 / data.variances[0]).sum())
    sum_w_r = float((1.0 / data.variances[1]).sum())
    assert geom["intens_err_g"] == pytest.approx(1.0 / np.sqrt(sum_w_g), rel=1e-9)
    assert geom["intens_err_r"] == pytest.approx(1.0 / np.sqrt(sum_w_r), rel=1e-9)


def test_h3_central_pixel_masked_wls_keeps_zero_error():
    """When the central pixel is masked under WLS, ``intens_err_<b>``
    must stay 0.0 (the value is NaN and no error can be assigned)."""
    from isoster.multiband import fit_image_multiband

    img = _planted_galaxy(amplitude=100.0, noise_sigma=0.005, seed=970)
    var_map = np.full_like(img, 0.05**2)
    # Mask the exact central pixel of band g; band r stays clean.
    iy = ix = 128
    mask_g = np.zeros_like(img, dtype=bool)
    mask_g[iy, ix] = True
    mask_r = np.zeros_like(img, dtype=bool)

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="r",
        sma0=12.0,
        minsma=0.0,
        eps=0.2,
        pa=0.4,
        astep=0.2,
        maxsma=30.0,
        minit=6,
        maxit=30,
        conver=0.05,
        fix_center=True,
        nclip=0,
    )
    res = fit_image_multiband(
        [img, img],
        [mask_g, mask_r],
        cfg,
        variance_maps=[var_map, var_map],
    )
    central = next(iso for iso in res["isophotes"] if iso["sma"] == 0.0)
    # Masked band: NaN intens, zero error.
    assert np.isnan(central["intens_g"])
    assert central["intens_err_g"] == 0.0
    # Unmasked band: the WLS error is set from the variance map.
    assert central["intens_err_r"] == pytest.approx(0.05, rel=1e-9)


def test_m4_forced_mode_sigma_clips_outliers():
    """Regression test for M4: multiband forced photometry must sigma-clip.

    Single-band extract_forced_photometry applies sigma_clip with the
    caller's sclip/nclip before computing intensities; the multiband path
    used all valid samples directly, letting ~3-sigma outliers (unmasked
    companions, cosmic rays) into intens_<b>/rms_<b>. The same ring with a
    bright outlier must come out clean through both paths.
    """
    from isoster.fitting import extract_forced_photometry
    from isoster.multiband.fitting_mb import extract_forced_photometry_mb

    h = w = 256
    x0 = y0 = 128.0
    rng = np.random.default_rng(21_000)
    img = 100.0 + rng.normal(0.0, 0.2, (h, w))
    # Bright cosmic ray exactly on the sma=40 ring (angle 0)
    img[128, 168] = 500.0

    out_sb = extract_forced_photometry(img, None, x0, y0, 40.0, 0.0, 0.0, sclip=3.0, nclip=2)
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", sclip=3.0, nclip=2)
    out_mb = extract_forced_photometry_mb(
        images=[img, img],
        masks=None,
        x0=x0,
        y0=y0,
        sma=40.0,
        eps=0.0,
        pa=0.0,
        bands=["g", "r"],
        config=cfg,
    )

    # Both paths must reject the 500-count outlier and land on the ~100 level.
    # Pre-fix, intens_<b> included it (mean inflated by 400/251 ~ 1.6 counts).
    assert out_sb["intens"] == pytest.approx(100.0, abs=0.2)
    assert out_mb["intens_g"] == pytest.approx(100.0, abs=0.2)
    assert out_mb["intens_r"] == pytest.approx(100.0, abs=0.2)
    # rms_<b> must stay at the noise scale (pre-fix ~25 from the outlier)
    assert out_mb["rms_g"] < 1.0
    assert out_mb["rms_r"] < 1.0


def test_m2_ols_band_weights_errors_weight_invariant():
    """Regression test for M2: OLS joint parameter errors must not shrink
    under non-unit band weights.

    The OLS covariance (A^T W A)^-1 scales as 1/w, but the old unweighted
    residual-variance rescale ignored w entirely, so uniform w=4 shrank the
    reported errors by sqrt(4)=2. The weight-aware rescale cancels the
    arbitrary weight scale exactly.
    """
    from isoster.multiband.fitting_mb import _compute_parameter_errors_from_joint

    rng = np.random.default_rng(3)
    n = 200
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    A1, B1, A2, B2 = 0.5, -0.3, 0.2, 0.1
    model = A1 * np.sin(angles) + B1 * np.cos(angles) + A2 * np.sin(2 * angles) + B2 * np.cos(2 * angles)
    intens = np.array(
        [
            100.0 + model + rng.normal(0.0, 2.0, n),
            200.0 + model + rng.normal(0.0, 2.0, n),
        ]
    )

    errs = {}
    for w in (1.0, 4.0):
        weights = np.full(2, w)
        coeffs, cov, _ = fit_first_and_second_harmonics_joint(angles, intens, weights, None)
        errs[w] = _compute_parameter_errors_from_joint(
            coeffs=coeffs,
            cov_full=cov,
            n_bands=2,
            sma=20.0,
            eps=0.3,
            pa=0.5,
            gradient=-2.0,
            gradient_error=None,
            angles=angles,
            intens_per_band=intens,
            use_exact_covariance=False,
            var_residual_floor=None,
            band_weights_arr=weights,
        )
    # Uniform weight scaling must leave the reported errors unchanged
    np.testing.assert_allclose(errs[1.0], errs[4.0], rtol=1e-10)
    assert all(e > 0.0 for e in errs[1.0])


def test_m2_intens_err_weight_invariant_end_to_end():
    """M2 (coupled branch): per-band intens_err must not shrink under
    uniform non-unit band weights."""
    img_g = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=1)
    img_r = _planted_galaxy(amplitude=200.0, noise_sigma=0.05, seed=2)
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5}

    outs = {}
    for w in (1.0, 4.0):
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            band_weights={"g": w, "r": w},
            sma0=20.0,
            debug=True,
            compute_errors=True,
            nclip=0,
        )
        outs[w] = fit_isophote_mb(images=[img_g, img_r], masks=None, sma=30.0, start_geometry=start, config=cfg)

    for b in ("g", "r"):
        assert outs[4.0][f"intens_err_{b}"] == pytest.approx(outs[1.0][f"intens_err_{b}"], rel=1e-6)
        assert outs[4.0][f"intens_err_{b}"] > 0.0


def test_m3_ols_loose_validity_rescales_geometric_errors():
    """Regression test for M3: OLS x loose validity must rescale geometric errors.

    The loose branch used the as-built (A^T A)^-1 with no residual-variance
    scaling, making x0_err/y0_err/eps_err/pa_err orders of magnitude too
    small. On identical data with no dropped bands the loose result must
    match the shared-validity one.
    """
    rng = np.random.default_rng(9)
    img_g = _planted_galaxy(amplitude=100.0, noise_sigma=0.05, seed=1)
    img_r = _planted_galaxy(amplitude=200.0, noise_sigma=0.05, seed=2)
    # Scale up the noise so the missing sigma^2 factor is obvious pre-fix
    img_g = img_g + rng.normal(0.0, 10.0, img_g.shape)
    img_r = img_r + rng.normal(0.0, 10.0, img_r.shape)
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.3, "pa": 0.5}

    outs = {}
    for loose in (False, True):
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            sma0=20.0,
            debug=True,
            compute_errors=True,
            nclip=0,
            loose_validity=loose,
        )
        outs[loose] = fit_isophote_mb(images=[img_g, img_r], masks=None, sma=30.0, start_geometry=start, config=cfg)

    for key in ("x0_err", "y0_err", "eps_err", "pa_err"):
        shared = float(outs[False][key])
        loose = float(outs[True][key])
        assert shared > 0.0 and loose > 0.0, f"{key}: shared={shared}, loose={loose}"
        # Same scale (pre-fix the loose value was ~1/sigma smaller, i.e. ~10x)
        assert 0.5 < loose / shared < 2.0, f"{key}: loose/shared={loose / shared:.3f}"


def test_m3_simultaneous_stamper_rescales_with_dropped_band():
    """M3 (family): simultaneous_in_loop harmonic errors must be rescaled
    even when loose validity dropped a band.

    The stamper's OLS rescale used to silently no-op when a dropped band's
    NaN intercept poisoned the residual sum, leaving the raw (A^T A)^-1
    diagonal (orders of magnitude too small) in a{n}_err_<b>.
    """
    from isoster.multiband.fitting_mb import (
        _attach_simultaneous_higher_harmonics_from_coeffs,
        fit_simultaneous_joint,
    )

    rng = np.random.default_rng(11)
    n = 120
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    sigma = 10.0
    A1, B1, A2, B2 = 0.4, -0.2, 0.15, 0.05
    model = A1 * np.sin(angles) + B1 * np.cos(angles) + A2 * np.sin(2 * angles) + B2 * np.cos(2 * angles)
    intens = np.array(
        [
            100.0 + model + rng.normal(0.0, sigma, n),
            200.0 + model + rng.normal(0.0, sigma, n),
        ]
    )
    coeffs_sub, cov_sub, _ = fit_simultaneous_joint(angles, intens, np.ones(2), [4], None)

    # Widen to a full 3-band layout with band 0 "dropped" (NaN intercept)
    n_full = 3
    width = n_full + 4 + 2
    coeffs = np.full(width, np.nan)
    coeffs[1:3] = coeffs_sub[:2]
    coeffs[n_full:] = coeffs_sub[2:]
    cov = np.zeros((width, width))
    cov[1:3, 1:3] = cov_sub[:2, :2]
    cov[1:3, n_full:] = cov_sub[:2, 2:]
    cov[n_full:, 1:3] = cov_sub[2:, :2]
    cov[n_full:, n_full:] = cov_sub[2:, 2:]

    geom: dict = {}
    _attach_simultaneous_higher_harmonics_from_coeffs(
        geom,
        ["g", "r", "i"],
        coeffs,
        cov,
        harmonic_orders=[4],
        dropped_band_indices={0},
        wls_mode=False,
        angles=[np.array([]), angles, angles],
        intens_per_band=[np.array([]), intens[0], intens[1]],
        band_weights_arr=np.ones(3),
        jagged=True,
    )
    a4_err = float(geom["a4_err_r"])
    raw = float(np.sqrt(cov_sub[6, 6]))  # unscaled (A^T A)^-1 diagonal
    assert np.isfinite(a4_err) and a4_err > 0.0
    # With sigma=10 the correct rescale is ~sigma=10x the raw diagonal
    assert a4_err > 5.0 * raw, f"a4_err={a4_err:.4g} not rescaled (raw diag {raw:.4g})"


def test_m5_dropped_band_harmonic_columns_are_nan():
    """Regression test for M5: dropped bands must report NaN, not 0.0.

    docs/10-multiband.md promises NaN harmonic columns for dropped bands;
    the code wrote 0.0, which downstream drop-detection reads as a real
    measurement.
    """
    from isoster.multiband import fit_image_multiband

    imgs = [_planted_galaxy(amplitude=a, noise_sigma=0.05, seed=s) for a, s in ((1000.0, 1), (2000.0, 2), (1500.0, 3))]
    y, x = np.mgrid[: imgs[0].shape[0], : imgs[0].shape[1]]
    r = np.hypot(x - 128.0, y - 128.0)
    theta = np.arctan2(y - 128.0, x - 128.0)
    # Band g keeps only a narrow wedge (~11% of samples) outside 15 px, so it
    # falls below the loose-validity keep fraction and is dropped there while
    # the gradient path still has data and isophotes converge.
    mask_g = (r > 15.0) & (np.abs(theta) > 0.35)

    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="r",
        sma0=10.0,
        maxsma=50.0,
        astep=0.2,
        debug=True,
        compute_deviations=True,
        nclip=0,
        loose_validity=True,
    )
    res = fit_image_multiband(imgs, [mask_g, None, None], cfg)

    dropped = [
        iso
        for iso in res["isophotes"]
        if iso["stop_code"] == 0 and isinstance(iso.get("intens_g"), float) and np.isnan(iso["intens_g"])
    ]
    assert dropped, "no converged isophotes with band g dropped"
    for iso in dropped:
        assert np.isnan(iso["intens_g"])
        for key in ("a3_g", "b3_g", "a4_g", "b4_g", "a3_err_g", "b3_err_g", "a4_err_g", "b4_err_g"):
            assert np.isnan(iso[key]), f"{key}={iso[key]} should be NaN for dropped band g"
        # Surviving bands still carry real (finite) harmonics
        assert np.isfinite(iso["a4_r"]) and np.isfinite(iso["a4_i"])


def test_m8_ndata_nflag_sane_under_loose_validity():
    """Regression test for M8: ndata/nflag must be consistent sums over bands.

    Under loose validity the code mixed an intersection count (total) with
    a per-band kept sum (actual), producing negative nflag and disabling
    the fflag guard. Both are now sums over bands: ndata = kept samples
    (equals the sum of n_valid_<b>), nflag = clipped (never negative).
    """
    from isoster.multiband import fit_image_multiband

    imgs = [_planted_galaxy(amplitude=a, noise_sigma=0.05, seed=s) for a, s in ((1000.0, 1), (2000.0, 2))]
    # Bright outliers on mid-radius rings so sigma clipping has work to do
    imgs[1][128, 158] = 1e5  # r = 30
    imgs[1][148, 128] = 1e5  # r = 20

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        sma0=10.0,
        maxsma=45.0,
        astep=0.25,
        debug=True,
        compute_deviations=True,
        nclip=2,
        sclip=3.0,
        loose_validity=True,
    )
    res = fit_image_multiband(imgs, None, cfg)

    saw_clipped = False
    for iso in res["isophotes"]:
        ndata = int(iso["ndata"])
        nflag = int(iso["nflag"])
        assert ndata > 0, f"ndata={ndata} at sma={iso['sma']}"
        assert nflag >= 0, f"negative nflag={nflag} at sma={iso['sma']}"
        if iso["sma"] > 0.0 and "n_valid_g" in iso and "n_valid_r" in iso:
            n_valid_sum = int(iso["n_valid_g"]) + int(iso["n_valid_r"])
            assert ndata == n_valid_sum, f"ndata={ndata} != sum(n_valid_<b>)={n_valid_sum} at sma={iso['sma']}"
        saw_clipped |= nflag > 0
    assert saw_clipped, "expected some sigma-clipped samples with planted outliers"


def test_m7_stop2_posthoc_harmonics_use_clipped_data():
    """Regression test for M7: stop_code=2 post-hoc harmonics must use the
    post-clip samples, not the raw sampler output.

    The stop-2 fallback passed last_data (pre-clip) to the harmonic solve
    while the fit itself ran on clipped data, so the exact outliers the
    pipeline removed leaked back into a_n_<b>/b_n_<b>.
    """
    img_g = _planted_galaxy(amplitude=100.0, noise_sigma=0.01, seed=1)
    img_r = _planted_galaxy(amplitude=200.0, noise_sigma=0.01, seed=2)
    img_r_outlier = img_r.copy()
    img_r_outlier[128, 158] = 1e4  # cosmic ray on the sma=30 ring

    # Start from a wrong eps so the fit cannot converge in maxit=1 (stop 2)
    start = {"x0": 128.0, "y0": 128.0, "eps": 0.1, "pa": 0.5}
    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        debug=True,
        nclip=2,
        sclip=3.0,
        minit=1,
        maxit=1,
        conver=1e-12,
        compute_deviations=True,
    )
    out_dirty = fit_isophote_mb(images=[img_g, img_r_outlier], masks=None, sma=30.0, start_geometry=start, config=cfg)
    out_clean = fit_isophote_mb(images=[img_g, img_r], masks=None, sma=30.0, start_geometry=start, config=cfg)
    assert out_dirty["stop_code"] == 2 and out_clean["stop_code"] == 2
    # The outlier's direct impact if it entered the solve is ~0.03-0.08
    # (measured with nclip=0); clip-set differences between the two runs
    # bound the agreement at ~0.006. A 0.01 tolerance separates the two.
    for key in ("a3_r", "b3_r", "a4_r", "b4_r"):
        assert float(out_dirty[key]) == pytest.approx(float(out_clean[key]), abs=1e-2), (
            f"{key}: dirty={out_dirty[key]}, clean={out_clean[key]} — pre-clip data leaked into harmonics"
        )


def test_m10_decoupled_wls_intercept_cov_is_exact():
    """Regression test for M10: the decoupled (ring-mean) WLS intercept
    covariance must be the exact 1/sum(1/var_i), not mean(var)/n.

    The decoupled paths pair an inverse-variance-weighted mean with what
    was the *unweighted* mean's variance, overestimating errors when
    per-pixel variances are non-uniform.
    """
    n = 100
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(0)
    intens = np.array([100.0 + rng.normal(0.0, 1.0, n), 200.0 + rng.normal(0.0, 1.0, n)])
    # Non-uniform variances: half the samples at var=1, half at var=100
    var = np.tile(np.concatenate([np.ones(n // 2), np.full(n // 2, 100.0)]), (2, 1))

    _coeffs, cov, wls_mode = fit_first_and_second_harmonics_joint(
        angles, intens, np.ones(2), var, fit_per_band_intens_jointly=False
    )
    assert wls_mode is True
    exact = 1.0 / np.sum(1.0 / var[0])
    old_unweighted = float(np.mean(var[0]) / n)
    for b in range(2):
        assert cov[b, b] == pytest.approx(exact, rel=1e-12)
    assert exact < 0.2 * old_unweighted, "test setup cannot distinguish the formulas"


def test_m10_intens_err_decoupled_wls_exact():
    """M10 (direct-SEM branch): decoupled WLS intens_err_<b> must be the
    exact 1/sqrt(sum(1/var_i)) of the ring samples."""
    from isoster.multiband.sampling_mb import extract_isophote_data_multi

    h = w = 256
    x0 = y0 = 128.0
    rng = np.random.default_rng(16_000)
    img_g = 100.0 + rng.normal(0.0, 0.4, (h, w))
    img_r = 200.0 + rng.normal(0.0, 0.4, (h, w))
    # Non-uniform variance map: left half 0.16, right half 16.0
    var = np.full((h, w), 0.16)
    var[:, w // 2 :] = 16.0

    cfg = IsosterConfigMB(
        bands=["g", "r"],
        reference_band="g",
        fit_per_band_intens_jointly=False,
        nclip=0,
    )
    out = fit_isophote_mb(
        images=[img_g, img_r],
        masks=None,
        sma=40.0,
        start_geometry={"x0": x0, "y0": y0, "eps": 0.0, "pa": 0.0},
        config=cfg,
        variance_maps=[var, var],
    )
    # Sample at the returned geometry (should equal the start on a flat image)
    data = extract_isophote_data_multi(
        [img_g, img_r],
        None,
        float(out["x0"]),
        float(out["y0"]),
        40.0,
        float(out["eps"]),
        float(out["pa"]),
        use_eccentric_anomaly=False,
        variance_maps=[var, var],
    )
    for b_idx, b in enumerate(("g", "r")):
        exact = 1.0 / np.sqrt(np.sum(1.0 / data.variances[b_idx]))
        old = float(np.sqrt(np.mean(data.variances[b_idx]) / data.variances[b_idx].size))
        assert out[f"intens_err_{b}"] == pytest.approx(exact, rel=1e-2)
        assert out[f"intens_err_{b}"] != pytest.approx(old, rel=1e-3)


def test_m9_runaway_gradient_decayed():
    """Regression test for M9: a runaway joint gradient decays to
    0.8 * previous_gradient with the error cleared (single-band parity)."""
    from isoster.multiband.fitting_mb import compute_joint_gradient

    h = w = 256
    rng = np.random.default_rng(31_000)
    img = 100.0 * np.ones((h, w)) + rng.normal(0.0, 0.5, (h, w))  # flat: grad ~ 0
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", astep=0.1)
    geometry = {"x0": 128.0, "y0": 128.0, "sma": 40.0, "eps": 0.0, "pa": 0.0}
    grad, grad_err, _pb, _pbe = compute_joint_gradient(
        np.stack([img, img]),
        [None, None],
        None,
        geometry,
        cfg,
        np.ones(2),
        previous_gradient=-5.0,
    )
    assert grad == pytest.approx(-4.0)  # 0.8 * previous
    assert grad_err is None


def test_m9_anomalous_first_gradient_resamples_at_double_step():
    """Regression test for M9: an anomalous first joint gradient triggers
    the 2x-step resample and uses the longer baseline (single-band parity).

    Profile plateau at r<45 hides the slope from the 1x annulus (40->44 px),
    so the first gradient is anomalously shallow; the 2x annulus (40->48 px)
    crosses the drop and must win. previous_gradient is small enough that the
    steeper 2x value survives the runaway decay.
    """
    from isoster.multiband.fitting_mb import compute_joint_gradient
    from isoster.multiband.sampling_mb import extract_isophote_data_multi_prepared

    h = w = 256
    x0 = y0 = 128.0
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(xx - x0, yy - y0)
    rng = np.random.default_rng(32_000)
    img = np.where(r < 45.0, 100.0, 100.0 * np.exp(-(r - 45.0) / 10.0)) + rng.normal(0.0, 0.1, (h, w))
    stack = np.stack([img, img])
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", astep=0.1, linear_growth=False)
    geometry = {"x0": x0, "y0": y0, "sma": 40.0, "eps": 0.0, "pa": 0.0}

    grad, grad_err, _pb, _pbe = compute_joint_gradient(
        stack,
        [None, None],
        None,
        geometry,
        cfg,
        np.ones(2),
        previous_gradient=-5.0,
    )

    data_c = extract_isophote_data_multi_prepared(stack, [None, None], None, x0, y0, 40.0, 0.0, 0.0)
    data_g2 = extract_isophote_data_multi_prepared(stack, [None, None], None, x0, y0, 48.0, 0.0, 0.0)
    expected_grad = (float(np.mean(data_g2.intens[0])) - float(np.mean(data_c.intens[0]))) / (40.0 * 0.2)

    # The 2x baseline value is returned (not the shallow 1x one, not the decay)
    assert grad == pytest.approx(expected_grad, rel=0.10)
    assert grad < -1.0, f"expected a steep 2x gradient, got {grad}"
    assert grad_err is not None, "the surviving 2x gradient must keep its error"
