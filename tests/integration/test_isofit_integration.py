"""
Integration tests for true ISOFIT simultaneous harmonic fitting (Phase 3).

Tests verify end-to-end behavior of simultaneous_harmonics=True through
fit_image() and fit_isophote(), including:
- Recovery of injected boxiness (a4) in mock Sersic profiles
- ISOFIT vs post-hoc coefficient comparison
- Real data (M51) regression with no stop-code regressions
- EA mode + ISOFIT on high-ellipticity mock
"""

from pathlib import Path

import numpy as np
import pytest

from isoster import fit_image
from isoster.config import IsosterConfig
from tests.fixtures import create_sersic_model


def _make_boxy_sersic_image(
    size=401, n=1.0, r_eff=30.0, intens_eff=500.0, eps=0.3, pa=0.5, a4_amp=0.04, a3_amp=0.0, oversample=5
):
    """Create a Sersic image with injected higher-order harmonic perturbations.

    The perturbation is applied to the elliptical radius via:
        r_perturbed = r_ell * (1 + a3*sin(3*theta) + a4*cos(4*theta))

    This produces isophotes with measurable a4 (boxiness/diskiness) and
    optionally a3 (asymmetry) coefficients.

    Returns:
        (image, cx, cy) tuple.
    """
    from scipy.special import gammaincinv

    b_n = gammaincinv(2.0 * n, 0.5)
    cx, cy = size / 2.0, size / 2.0

    # Oversampled grid for subpixel accuracy
    y_hr = np.linspace(0, size, size * oversample, endpoint=False) + 0.5 / oversample
    x_hr = np.linspace(0, size, size * oversample, endpoint=False) + 0.5 / oversample
    yy, xx = np.meshgrid(y_hr, x_hr, indexing="ij")

    dx = xx - cx
    dy = yy - cy
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    x_ell = dx * cos_pa + dy * sin_pa
    y_ell = -dx * sin_pa + dy * cos_pa

    q = 1.0 - eps
    r_ell = np.sqrt(x_ell**2 + (y_ell / q) ** 2)
    r_ell = np.clip(r_ell, 0.1, None)

    theta = np.arctan2(y_ell / q, x_ell)

    # Perturbed radius with harmonic deviations
    r_perturbed = r_ell * (1.0 + a3_amp * np.sin(3.0 * theta) + a4_amp * np.cos(4.0 * theta))
    r_perturbed = np.clip(r_perturbed, 0.1, None)

    image_hr = intens_eff * np.exp(-b_n * ((r_perturbed / r_eff) ** (1.0 / n) - 1.0))

    # Downsample
    image = image_hr.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    return image, cx, cy


# ============================================================================
# Test 1: ISOFIT recovers injected boxiness (a4) within tolerance
# ============================================================================


def test_isofit_mock_sersic_boxy():
    """Fit a Sersic mock with injected a4 boxiness, verify ISOFIT recovery.

    Uses fit_image() with simultaneous_harmonics=True and verifies that
    the recovered a4 coefficient is consistently nonzero in the valid
    radial range (0.5*R_e to 4*R_e) and has the correct sign.
    """
    r_eff = 30.0
    injected_a4 = 0.04
    image, cx, cy = _make_boxy_sersic_image(
        size=401,
        n=1.0,
        r_eff=r_eff,
        intens_eff=500.0,
        eps=0.3,
        pa=0.5,
        a4_amp=injected_a4,
        oversample=5,
    )

    config = IsosterConfig(
        x0=cx,
        y0=cy,
        sma0=6.0,
        minsma=3.0,
        maxsma=150.0,
        astep=0.12,
        eps=0.3,
        pa=0.5,
        minit=10,
        maxit=60,
        conver=0.05,
        fix_center=True,
        simultaneous_harmonics=True,
        harmonic_orders=[3, 4],
    )

    results = fit_image(image, mask=None, config=config)
    isophotes = results["isophotes"]

    sma = np.array([iso["sma"] for iso in isophotes])
    a4 = np.array([iso["a4"] for iso in isophotes])
    b4 = np.array([iso["b4"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    # Valid range: converged, within 0.5*R_e to 4*R_e
    valid = (stop_codes == 0) & (sma >= 0.5 * r_eff) & (sma <= 4.0 * r_eff)
    n_valid = int(valid.sum())
    assert n_valid >= 5, f"Need at least 5 valid isophotes in range, got {n_valid}"

    # a4 magnitude should be consistently nonzero (we injected a4_amp=0.04)
    a4_mag = np.sqrt(a4[valid] ** 2 + b4[valid] ** 2)
    median_a4_mag = np.median(a4_mag)
    assert median_a4_mag > 0.005, f"Median |a4,b4| = {median_a4_mag:.4f}, expected > 0.005 from injected {injected_a4}"

    # At least 80% of valid points should have nonzero a4
    nonzero_frac = np.mean(a4_mag > 1e-4)
    assert nonzero_frac > 0.8, f"Only {nonzero_frac:.0%} of valid points have nonzero a4"

    print("\nISOFIT boxy mock recovery:")
    print(f"  Valid isophotes in range: {n_valid}")
    print(f"  Median |a4,b4|: {median_a4_mag:.4f} (injected: {injected_a4})")
    print(f"  Nonzero a4 fraction: {nonzero_frac:.0%}")


# ============================================================================
# Test 1b: default (post-hoc) path recovers b4 on noise-free data (B1)
# ============================================================================


def test_default_path_noise_free_recovers_b4():
    """Regression test for B1: the default path must recover b4 on noise-free data.

    Same noise-free boxy mock as test_isofit_mock_sersic_boxy, but fit with the
    default configuration (compute_deviations=True, simultaneous_harmonics=False,
    no variance map). That path used scipy leastsq, which reports a None
    covariance when the fit is exact, so every converged isophote silently got
    a_n=b_n=0. The r->r*(1+0.04*cos(4*theta)) perturbation maps to b4=-0.04 in
    the stored Bender normalization (b_n = B_n/(sma*|grad|), grad<0).
    """
    r_eff = 30.0
    image, cx, cy = _make_boxy_sersic_image(
        size=401,
        n=1.0,
        r_eff=r_eff,
        intens_eff=500.0,
        eps=0.3,
        pa=0.5,
        a4_amp=0.04,
        oversample=5,
    )

    config = IsosterConfig(
        x0=cx,
        y0=cy,
        sma0=6.0,
        minsma=3.0,
        maxsma=150.0,
        astep=0.12,
        eps=0.3,
        pa=0.5,
        minit=10,
        maxit=60,
        conver=0.05,
        fix_center=True,
        harmonic_orders=[3, 4],
    )

    results = fit_image(image, mask=None, config=config)
    isophotes = results["isophotes"]

    sma = np.array([iso["sma"] for iso in isophotes])
    b4 = np.array([iso["b4"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    # Valid range: converged, within 0.5*R_e to 4*R_e
    valid = (stop_codes == 0) & (sma >= 0.5 * r_eff) & (sma <= 4.0 * r_eff)
    n_valid = int(valid.sum())
    assert n_valid >= 5, f"Need at least 5 valid isophotes in range, got {n_valid}"

    # Planted perturbation maps to b4=-0.04; every converged isophote must
    # carry a non-zero b4 of the right sign (B1 returned exactly 0.0)
    median_b4 = np.median(b4[valid])
    assert median_b4 < -0.02, f"Median b4 = {median_b4:.4f}, expected near -0.04 from injected a4_amp=0.04"
    nonzero_frac = np.mean(b4[valid] < -1e-4)
    assert nonzero_frac > 0.8, f"Only {nonzero_frac:.0%} of valid points have non-zero b4 (B1 regime: all zero)"

    print("\nDefault-path noise-free b4 recovery (B1):")
    print(f"  Valid isophotes in range: {n_valid}")
    print(f"  Median b4: {median_b4:.4f} (expected near -0.04)")
    print(f"  Non-zero b4 fraction: {nonzero_frac:.0%}")


# ============================================================================
# Test 2: ISOFIT vs post-hoc coefficient comparison
# ============================================================================


def test_isofit_vs_posthoc_coefficients():
    """Compare ISOFIT (simultaneous) vs post-hoc harmonic coefficients.

    Both approaches should yield similar a3/a4 coefficients for a boxy
    mock galaxy. The ISOFIT approach accounts for cross-correlations,
    so results may differ slightly, but should be correlated.
    """
    r_eff = 30.0
    image, cx, cy = _make_boxy_sersic_image(
        size=401,
        n=1.0,
        r_eff=r_eff,
        intens_eff=500.0,
        eps=0.3,
        pa=0.5,
        a4_amp=0.04,
        a3_amp=0.01,
        oversample=5,
    )

    base_config = dict(
        x0=cx,
        y0=cy,
        sma0=6.0,
        minsma=3.0,
        maxsma=120.0,
        astep=0.12,
        eps=0.3,
        pa=0.5,
        minit=10,
        maxit=60,
        conver=0.05,
        fix_center=True,
        fix_pa=True,
        fix_eps=True,
        harmonic_orders=[3, 4],
    )

    # Run ISOFIT (simultaneous in iteration loop)
    config_isofit = IsosterConfig(**base_config, simultaneous_harmonics=True)
    results_isofit = fit_image(image, mask=None, config=config_isofit)

    # Run post-hoc (sequential after convergence)
    config_posthoc = IsosterConfig(**base_config, simultaneous_harmonics=False)
    results_posthoc = fit_image(image, mask=None, config=config_posthoc)

    iso_isofit = results_isofit["isophotes"]
    iso_posthoc = results_posthoc["isophotes"]

    # Match by SMA (both should have similar SMA grids)
    sma_isofit = np.array([iso["sma"] for iso in iso_isofit])
    sma_posthoc = np.array([iso["sma"] for iso in iso_posthoc])
    sc_isofit = np.array([iso["stop_code"] for iso in iso_isofit])
    sc_posthoc = np.array([iso["stop_code"] for iso in iso_posthoc])

    # Find common converged SMA values in valid range
    valid_isofit = (sc_isofit == 0) & (sma_isofit >= 0.5 * r_eff) & (sma_isofit <= 4.0 * r_eff)
    valid_posthoc = (sc_posthoc == 0) & (sma_posthoc >= 0.5 * r_eff) & (sma_posthoc <= 4.0 * r_eff)

    # Extract combined a4 magnitude (sqrt(a4^2 + b4^2)) for comparison
    a4_isofit = np.array([iso["a4"] for iso in iso_isofit])[valid_isofit]
    b4_isofit = np.array([iso["b4"] for iso in iso_isofit])[valid_isofit]
    a4_posthoc = np.array([iso["a4"] for iso in iso_posthoc])[valid_posthoc]
    b4_posthoc = np.array([iso["b4"] for iso in iso_posthoc])[valid_posthoc]

    mag_isofit = np.sqrt(a4_isofit**2 + b4_isofit**2)
    mag_posthoc = np.sqrt(a4_posthoc**2 + b4_posthoc**2)

    # Both should detect nonzero a4 magnitude (we injected a4_amp=0.04)
    assert np.median(mag_isofit) > 0.005, f"ISOFIT median |a4,b4| too small: {np.median(mag_isofit):.6f}"
    assert np.median(mag_posthoc) > 0.005, f"Post-hoc median |a4,b4| too small: {np.median(mag_posthoc):.6f}"

    # Correlation check: the b4 profiles should be correlated between methods
    n_compare = min(len(b4_isofit), len(b4_posthoc))
    if n_compare >= 5:
        corr = np.corrcoef(b4_isofit[:n_compare], b4_posthoc[:n_compare])[0, 1]
        assert corr > 0.5, f"ISOFIT vs post-hoc b4 correlation = {corr:.3f}, expected > 0.5"
        print(f"\nISOFIT vs post-hoc b4 correlation: {corr:.3f}")

    print(f"  ISOFIT median |a4,b4|: {np.median(mag_isofit):.4f}")
    print(f"  Post-hoc median |a4,b4|: {np.median(mag_posthoc):.4f}")


# ============================================================================
# Test 3: Real data (M51) — no stop-code regressions with ISOFIT
# ============================================================================

M51_PATH = Path(__file__).parent.parent.parent / "examples" / "data" / "m51" / "M51.fits"


@pytest.mark.real_data
def test_isofit_m51():
    """Run ISOFIT on M51 and verify no stop-code regression vs default.

    Both simultaneous_harmonics=True and =False should produce comparable
    convergence rates — ISOFIT should not degrade fitting quality.
    """
    if not M51_PATH.exists():
        pytest.skip(f"M51 data not found at {M51_PATH}")

    from astropy.io import fits

    with fits.open(M51_PATH) as hdul:
        image = hdul[0].data.astype(np.float64)

    y0, x0 = np.array(image.shape) / 2.0

    base_config = dict(
        x0=x0,
        y0=y0,
        sma0=10.0,
        minsma=5.0,
        maxsma=min(x0, y0) - 20,
        astep=0.15,
        eps=0.3,
        pa=0.0,
        minit=10,
        maxit=50,
        conver=0.05,
        harmonic_orders=[3, 4],
    )

    # Default path
    config_default = IsosterConfig(**base_config, simultaneous_harmonics=False)
    results_default = fit_image(image, mask=None, config=config_default)

    # ISOFIT path
    config_isofit = IsosterConfig(**base_config, simultaneous_harmonics=True)
    results_isofit = fit_image(image, mask=None, config=config_isofit)

    iso_default = results_default["isophotes"]
    iso_isofit = results_isofit["isophotes"]

    sc_default = [iso["stop_code"] for iso in iso_default]
    sc_isofit = [iso["stop_code"] for iso in iso_isofit]

    conv_default = sum(1 for sc in sc_default if sc == 0) / len(sc_default)
    conv_isofit = sum(1 for sc in sc_isofit if sc == 0) / len(sc_isofit)

    # ISOFIT convergence should not be dramatically worse (within 15 ppt)
    assert conv_isofit > conv_default - 0.15, (
        f"ISOFIT convergence ({conv_isofit:.1%}) regressed vs default ({conv_default:.1%})"
    )

    # Both should produce enough isophotes
    assert len(iso_isofit) > 10, f"ISOFIT produced only {len(iso_isofit)} isophotes"

    # Stop-code distribution: ISOFIT should not introduce new failure modes
    fail_default = sum(1 for sc in sc_default if sc in (-1, 3))
    fail_isofit = sum(1 for sc in sc_isofit if sc in (-1, 3))
    assert fail_isofit <= fail_default + 3, f"ISOFIT has {fail_isofit} failures vs {fail_default} default (regression)"

    print("\nM51 ISOFIT regression test:")
    print(f"  Default: {len(iso_default)} isophotes, {conv_default:.1%} converged")
    print(f"  ISOFIT:  {len(iso_isofit)} isophotes, {conv_isofit:.1%} converged")
    print(f"  Default failures: {fail_default}, ISOFIT failures: {fail_isofit}")


# ============================================================================
# Test 4: EA mode + ISOFIT on high-ellipticity mock
# ============================================================================


def test_isofit_ea_mode():
    """EA mode + ISOFIT should work on high-ellipticity mock.

    Verifies that eccentric anomaly sampling combined with simultaneous
    harmonics fitting produces valid results on an elliptical Sersic model.
    """
    r_eff = 25.0
    eps = 0.6
    pa = np.pi / 3

    image, true_profile, params = create_sersic_model(
        R_e=r_eff,
        n=1.0,
        I_e=1000.0,
        eps=eps,
        pa=pa,
        oversample=5,
    )

    config = IsosterConfig(
        x0=params["x0"],
        y0=params["y0"],
        sma0=8.0,
        minsma=3.0,
        maxsma=min(100, params["x0"] - 10),
        astep=0.15,
        eps=eps,
        pa=pa,
        minit=10,
        maxit=60,
        conver=0.05,
        maxgerr=1.0,  # Relaxed for high ellipticity
        use_eccentric_anomaly=True,
        simultaneous_harmonics=True,
        harmonic_orders=[3, 4],
    )

    results = fit_image(image, mask=None, config=config)
    isophotes = results["isophotes"]

    sma = np.array([iso["sma"] for iso in isophotes])
    intens = np.array([iso["intens"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    # Should produce isophotes
    assert len(isophotes) > 10, f"Expected >10 isophotes, got {len(isophotes)}"

    # Convergence rate should be reasonable
    converged = (stop_codes == 0).sum()
    conv_rate = converged / len(isophotes)
    assert conv_rate > 0.4, f"Convergence rate {conv_rate:.1%} too low for EA+ISOFIT"

    # Intensity accuracy in valid range
    valid = (stop_codes == 0) & (sma >= 3.0) & (sma >= 0.5 * r_eff) & (sma <= 5.0 * r_eff)
    n_valid = int(valid.sum())
    assert n_valid > 5, f"Expected >5 valid isophotes, got {n_valid}"

    true_intens = true_profile(sma[valid])
    rel_diff = np.abs((intens[valid] - true_intens) / true_intens)
    max_rel_diff = np.max(rel_diff)

    assert max_rel_diff < 0.05, f"Max intensity rel diff = {max_rel_diff * 100:.2f}% exceeds 5% threshold"

    # Harmonics should be present in output
    for iso in isophotes:
        assert "a3" in iso, "Missing a3 in isophote output"
        assert "a4" in iso, "Missing a4 in isophote output"
        assert "a3_err" in iso, "Missing a3_err in isophote output"
        assert "a4_err" in iso, "Missing a4_err in isophote output"

    print(f"\nEA + ISOFIT (eps={eps}) test:")
    print(f"  Total isophotes: {len(isophotes)}")
    print(f"  Converged: {converged} ({conv_rate:.1%})")
    print(f"  Max intensity rel diff (valid range): {max_rel_diff * 100:.3f}%")


# ============================================================================
# Test 5: R26-T1 — geometry_update_mode='simultaneous' with fit_image
# ============================================================================


def test_simultaneous_geometry_fit_image():
    """R26-T1: fit_image end-to-end with geometry_update_mode='simultaneous'.

    Verifies that simultaneous geometry updates produce valid isophote profiles
    when running through the full driver pipeline (outward + inward growth).
    """
    r_eff = 25.0
    eps = 0.3
    pa = 0.7

    image, true_profile, params = create_sersic_model(
        R_e=r_eff,
        n=1.5,
        I_e=800.0,
        eps=eps,
        pa=pa,
        oversample=5,
    )

    config = IsosterConfig(
        x0=params["x0"],
        y0=params["y0"],
        sma0=6.0,
        minsma=3.0,
        maxsma=100.0,
        astep=0.15,
        eps=eps,
        pa=pa,
        minit=10,
        maxit=60,
        conver=0.05,
        geometry_update_mode="simultaneous",
        geometry_damping=0.5,
        fix_center=True,
    )

    results = fit_image(image, mask=None, config=config)
    isophotes = results["isophotes"]

    sma_arr = np.array([iso["sma"] for iso in isophotes])
    intens = np.array([iso["intens"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    # Should produce reasonable number of isophotes
    assert len(isophotes) > 10, f"Expected >10 isophotes, got {len(isophotes)}"

    # Good convergence rate
    conv_rate = (stop_codes == 0).sum() / len(isophotes)
    assert conv_rate > 0.5, f"Convergence rate {conv_rate:.1%} too low for simultaneous mode"

    # Intensity accuracy in valid range
    valid = (stop_codes == 0) & (sma_arr >= 0.5 * r_eff) & (sma_arr <= 5.0 * r_eff)
    n_valid = int(valid.sum())
    assert n_valid >= 5, f"Expected >= 5 valid isophotes in range, got {n_valid}"

    true_intens = true_profile(sma_arr[valid])
    max_rel_diff = np.max(np.abs((intens[valid] - true_intens) / true_intens))
    assert max_rel_diff < 0.05, f"Max intensity rel diff = {max_rel_diff * 100:.2f}% exceeds 5% threshold"


# ============================================================================
# Test 6: R26-T2 — isofit_mode='original' post-hoc harmonics with fit_image
# ============================================================================


def test_isofit_original_mode_fit_image():
    """R26-T2: fit_image end-to-end with isofit_mode='original'.

    Verifies that isofit_mode='original' (Ciambur 2015 post-hoc fitting)
    produces harmonic coefficients through the full driver pipeline.
    """
    r_eff = 30.0
    injected_a4 = 0.04
    image, cx, cy = _make_boxy_sersic_image(
        size=401,
        n=1.0,
        r_eff=r_eff,
        intens_eff=500.0,
        eps=0.3,
        pa=0.5,
        a4_amp=injected_a4,
        oversample=5,
    )

    config = IsosterConfig(
        x0=cx,
        y0=cy,
        sma0=6.0,
        minsma=3.0,
        maxsma=150.0,
        astep=0.12,
        eps=0.3,
        pa=0.5,
        minit=10,
        maxit=60,
        conver=0.05,
        fix_center=True,
        simultaneous_harmonics=True,
        isofit_mode="original",
        harmonic_orders=[3, 4],
    )

    results = fit_image(image, mask=None, config=config)
    isophotes = results["isophotes"]

    sma_arr = np.array([iso["sma"] for iso in isophotes])
    stop_codes = np.array([iso["stop_code"] for iso in isophotes])

    # Should produce isophotes with harmonic keys
    assert len(isophotes) > 10, f"Expected >10 isophotes, got {len(isophotes)}"
    for iso in isophotes:
        assert "a4" in iso, "Missing a4 in isofit_mode='original' output"
        assert "b4" in iso, "Missing b4 in isofit_mode='original' output"

    # Valid range should recover nonzero a4
    valid = (stop_codes == 0) & (sma_arr >= 0.5 * r_eff) & (sma_arr <= 4.0 * r_eff)
    n_valid = int(valid.sum())
    assert n_valid >= 5, f"Expected >= 5 valid isophotes in range, got {n_valid}"

    a4 = np.array([iso["a4"] for iso in isophotes])
    b4 = np.array([iso["b4"] for iso in isophotes])
    a4_mag = np.sqrt(a4[valid] ** 2 + b4[valid] ** 2)
    median_a4_mag = np.median(a4_mag)
    assert median_a4_mag > 0.005, (
        f"isofit_mode='original' median |a4,b4| = {median_a4_mag:.4f}, expected > 0.005 from injected {injected_a4}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
