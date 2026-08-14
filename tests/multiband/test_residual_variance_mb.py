"""One solve yields one OLS error scale.

Three rules, all mirroring single-band ``fit_isophote``:

* The residual variance is measured against the **complete** fitted model. A
  truncated model charges the fitted higher-order structure to the noise.
* The ``sigma_bg**2`` floor bounds a fit's residual variance, so every error
  derived from that fit shares it. Multi-band used to floor the geometry errors
  only, leaving the per-band intensity errors and the higher-order harmonic
  errors on a different, unfloored scale.
* An exactly-determined solve carries no information about the noise, so its
  residual variance is zero. Returning ``None`` there made the harmonic
  attachers skip the OLS rescale and publish a raw ``(A^T A)^-1`` diagonal,
  which is an arbitrary scale rather than an error.
"""

import numpy as np
import pytest

from isoster._shared import _weighted_mean_variance
from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import (
    _attach_simultaneous_higher_harmonics_from_coeffs,
    _compute_joint_residual_variance,
    _compute_parameter_errors_from_joint,
    _reference_residual_variance,
    _sigma_bg_variance_floor,
    evaluate_joint_model,
    fit_isophote_mb,
)
from isoster.multiband.sampling_mb import extract_isophote_data_multi_prepared, prepare_inputs
from isoster.numba_kernels import build_harmonic_matrix

START = {"x0": 60.0, "y0": 60.0, "eps": 0.3, "pa": 0.4}
SMA = 20.0


def make_sersic_image(size=121, amplitude=1000.0, r_eff=20.0):
    from scipy.special import gammaincinv

    bn = gammaincinv(4.0, 0.5)
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((xx - size // 2) ** 2 + ((yy - size // 2) / 0.7) ** 2)
    return amplitude * np.exp(-bn * ((r / r_eff) ** 0.5 - 1.0))


# ---------------------------------------------------------------------------
# The residual is measured against the complete fitted model
# ---------------------------------------------------------------------------

ORDERS = (3, 4)


def simultaneous_coeffs(n_bands=2, orders=ORDERS):
    """A wide coefficient vector with large, non-negligible higher-order terms."""
    width = n_bands + 4 + 2 * len(orders)
    coeffs = np.zeros(width)
    coeffs[:n_bands] = [100.0, 70.0]
    coeffs[n_bands : n_bands + 4] = [1.5, -2.0, 0.8, 1.2]
    coeffs[n_bands + 4 :] = [5.0, -4.0, 3.0, 2.5]
    return coeffs


@pytest.mark.parametrize("jagged", [False, True])
def test_noiseless_simultaneous_model_has_zero_residual_variance(jagged):
    """A model evaluated from its own coefficients has no residual, by definition.

    The retired helper skipped the fitted higher-order block, so it charged all
    of that structure to the noise and returned 30.5 here — its comment called
    that "conservative", which is true only in sign.
    """
    n_bands, n_samples = 2, 64
    coeffs = simultaneous_coeffs(n_bands)
    width = coeffs.size
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    model = evaluate_joint_model(angles, coeffs, n_bands, harmonic_orders=ORDERS)

    if jagged:
        args = ([angles, angles], [model[0], model[1]])
    else:
        args = (angles, model)

    result = _compute_joint_residual_variance(
        coeffs,
        *args,
        np.ones(n_bands),
        n_bands=n_bands,
        n_geom_params=width,
        jagged=jagged,
        harmonic_orders=ORDERS,
    )
    assert result == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize("jagged", [False, True])
def test_truncated_model_would_charge_the_harmonics_to_the_noise(jagged):
    """Guard the failure mode directly: omitting the orders must be observable."""
    n_bands, n_samples = 2, 64
    coeffs = simultaneous_coeffs(n_bands)
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    model = evaluate_joint_model(angles, coeffs, n_bands, harmonic_orders=ORDERS)
    args = ([angles, angles], [model[0], model[1]]) if jagged else (angles, model)

    truncated = _compute_joint_residual_variance(
        coeffs, *args, np.ones(n_bands), n_bands=n_bands, n_geom_params=coeffs.size, jagged=jagged
    )
    # Half the summed squared amplitude of the (3, 4) terms, per band.
    expected = 0.5 * (5.0**2 + 4.0**2 + 3.0**2 + 2.5**2) * (2 * n_samples) / (2 * n_samples - coeffs.size)
    assert truncated == pytest.approx(expected, rel=1e-9)


def test_noise_only_residual_is_recovered_with_the_full_model():
    """Adding known noise to the exact model recovers that noise, not the signal."""
    n_bands, n_samples = 2, 256
    coeffs = simultaneous_coeffs(n_bands)
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    model = evaluate_joint_model(angles, coeffs, n_bands, harmonic_orders=ORDERS)
    rng = np.random.default_rng(11)
    sigma = 0.7
    noisy = model + rng.normal(0.0, sigma, model.shape)

    result = _compute_joint_residual_variance(
        coeffs,
        angles,
        noisy,
        np.ones(n_bands),
        n_bands=n_bands,
        n_geom_params=coeffs.size,
        jagged=False,
        harmonic_orders=ORDERS,
    )
    assert result == pytest.approx(sigma**2, rel=0.15)


def test_simultaneous_in_loop_geometry_and_harmonics_share_one_scale():
    """End-to-end: one solve, one error scale, across all three error families.

    The geometry path counted ``n_bands + 4`` parameters while the harmonic
    attacher counted ``n_bands + 4 + 2L``, and both measured a truncated model,
    so a single solve produced two different residual variances. With the floor
    dominating, every error derived from that solve must now move with
    ``sigma_bg`` by the same factor.
    """
    images = [make_sersic_image(), make_sersic_image(amplitude=600.0)]
    results = {}
    for sigma_bg in (5.0, 50.0):
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            sigma_bg=sigma_bg,
            multiband_higher_harmonics="simultaneous_in_loop",
        )
        results[sigma_bg] = fit_isophote_mb(images, None, SMA, START, cfg)

    for key in ("eps_err", "pa_err", "a3_err_g", "b4_err_g", "intens_err_g"):
        low, high = results[5.0][key], results[50.0][key]
        assert low > 0.0, key
        assert high / low == pytest.approx(10.0, rel=0.01), key


def test_truncated_and_full_residual_variance_differ_on_real_harmonics():
    """The two forms must be distinguishable, or the test above proves nothing."""
    n_bands, n_samples = 2, 96
    coeffs = simultaneous_coeffs(n_bands)
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    model = evaluate_joint_model(angles, coeffs, n_bands, harmonic_orders=ORDERS)
    rng = np.random.default_rng(12)
    noisy = model + rng.normal(0.0, 0.5, model.shape)

    full = _compute_joint_residual_variance(
        coeffs,
        angles,
        noisy,
        np.ones(n_bands),
        n_bands=n_bands,
        n_geom_params=coeffs.size,
        jagged=False,
        harmonic_orders=ORDERS,
    )
    truncated = _compute_joint_residual_variance(
        coeffs, angles, noisy, np.ones(n_bands), n_bands=n_bands, n_geom_params=n_bands + 4, jagged=False
    )
    assert full == pytest.approx(0.25, rel=0.15)  # recovers sigma**2 = 0.5**2
    assert truncated > 50.0 * full


# ---------------------------------------------------------------------------
# Reference mode: the scale comes from the band that was actually solved
# ---------------------------------------------------------------------------


def reference_ring(n=128, sigma=0.3, seed=4):
    coeffs_ref = np.array([100.0, 1.5, -2.0, 0.8, 1.2])
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(seed)
    intens = build_harmonic_matrix(angles) @ coeffs_ref + rng.normal(0.0, sigma, n)
    return angles, intens, coeffs_ref


def test_reference_residual_variance_recovers_the_reference_band_noise():
    """Five fitted parameters, measured against the model the ref solve fitted."""
    angles, intens, coeffs_ref = reference_ring(sigma=0.3)
    assert _reference_residual_variance(angles, intens, coeffs_ref, None) == pytest.approx(0.09, rel=0.15)


def test_reference_residual_variance_is_zero_when_exactly_determined():
    angles, _intens, coeffs_ref = reference_ring(n=5)
    exact = build_harmonic_matrix(angles) @ coeffs_ref
    assert _reference_residual_variance(angles, exact, coeffs_ref, None) == 0.0
    assert _reference_residual_variance(angles, exact, coeffs_ref, 9.0) == 9.0


@pytest.mark.parametrize("other_noise", [0.2, 2.0, 20.0, 200.0])
def test_reference_geometry_errors_ignore_bands_outside_the_solve(other_noise):
    """A band excluded from the harmonic solve must not set that solve's scale.

    The pooled joint helper measured every band's residuals against a model
    whose geometry only the reference band had constrained, so non-reference
    noise drove the geometry errors: across this parametrisation the pooled form
    spans a factor of ~99, and end-to-end a 100x noise change moved ``eps_err``
    by 18x. Geometry is held fixed here so the scaling is the only variable.
    """
    n_bands = 2
    angles, intens_ref, coeffs_ref = reference_ring()
    other = 70.0 + np.random.default_rng(9).normal(0.0, other_noise, angles.size)

    coeffs = np.zeros(n_bands + 4)
    coeffs[0] = coeffs_ref[0]
    coeffs[1] = float(other.mean())
    coeffs[n_bands:] = coeffs_ref[1:5]

    errors = _compute_parameter_errors_from_joint(
        coeffs=coeffs,
        cov_full=np.eye(n_bands + 4) * 1e-4,
        n_bands=n_bands,
        sma=20.0,
        eps=0.3,
        pa=0.4,
        gradient=-50.0,
        gradient_error=None,
        angles=angles,
        intens_per_band=np.vstack([intens_ref, other]),
        use_exact_covariance=False,
        var_residual_floor=None,
        band_weights_arr=np.ones(n_bands),
        residual_variance=_reference_residual_variance(angles, intens_ref, coeffs_ref, None),
    )
    # Bit-identical regardless of the other band: pinned to the value the
    # reference band alone implies.
    assert errors[2] == pytest.approx(4.310919409e-06, rel=1e-9)


def test_reference_mode_geometry_and_reference_intensity_share_one_scale():
    """End-to-end: `intens_<ref>` is a fitted intercept, so it takes the floor.

    The non-reference bands report a plain ring mean, which is a different
    quantity and correctly stays unfloored.
    """
    images = [make_sersic_image(), make_sersic_image(amplitude=600.0)]
    results = {}
    for sigma_bg in (5.0, 50.0):
        cfg = IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            harmonic_combination="ref",
            sigma_bg=sigma_bg,
        )
        results[sigma_bg] = fit_isophote_mb(images, None, SMA, START, cfg)

    for key in ("eps_err", "pa_err", "intens_err_g"):
        assert results[50.0][key] / results[5.0][key] == pytest.approx(10.0, rel=0.01), key
    # Non-reference band: a plain ring statistic, unaffected by the floor.
    assert results[50.0]["intens_err_r"] == pytest.approx(results[5.0]["intens_err_r"], rel=1e-12)


def azimuthal_variance_map(size=121, low=0.5, high=200.0):
    """Strongly non-uniform variance, so the weighted and unweighted means
    cannot coincide by accident."""
    yy, _xx = np.mgrid[:size, :size]
    return np.where(yy < size // 2, low, high).astype(np.float64)


def test_reference_mode_non_reference_bands_report_the_weighted_mean():
    """Under WLS the value and its error must describe one estimator.

    Ref mode widened its coefficient vector with a plain ``np.mean`` while
    ``intens_err_<b>`` reported the inverse-variance-weighted mean's variance.
    On the map below the reported error understated the reported mean's own
    error tenfold (0.0904 against 0.9026). The variance map is deliberately
    bimodal so the two means differ.
    """
    base = make_sersic_image()
    images = [base, base * 0.6]
    variance_maps = [np.full((121, 121), 1.0), azimuthal_variance_map()]
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", harmonic_combination="ref")

    out = fit_isophote_mb(images, None, SMA, START, cfg, variance_maps=variance_maps)

    image_stack, masks, var_stack = prepare_inputs(images, None, variance_maps)
    ring = extract_isophote_data_multi_prepared(
        image_stack, masks, var_stack, out["x0"], out["y0"], SMA, out["eps"], out["pa"]
    )
    intens, variances = ring.intens[1], ring.variances[1]
    weights = 1.0 / variances
    weighted_mean = float((intens * weights).sum() / weights.sum())
    plain_mean = float(intens.mean())

    # The fixture must actually distinguish the two estimators.
    assert abs(weighted_mean - plain_mean) > 1e-3

    assert out["intens_r"] == pytest.approx(weighted_mean, rel=1e-9)
    assert out["intens_err_r"] == pytest.approx(np.sqrt(_weighted_mean_variance(variances)), rel=1e-9)
    # And the plain mean's own error is the value that would have been wrong to
    # report next to a weighted mean.
    plain_mean_error = float(np.sqrt(variances.sum() / intens.size**2))
    assert plain_mean_error > 5.0 * out["intens_err_r"]


def test_reference_mode_ols_non_reference_bands_are_unchanged():
    """Without a variance map both reducers give the plain mean, so nothing moves."""
    base = make_sersic_image()
    images = [base, base * 0.6]
    cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", harmonic_combination="ref")

    out = fit_isophote_mb(images, None, SMA, START, cfg)

    image_stack, masks, var_stack = prepare_inputs(images, None, None)
    ring = extract_isophote_data_multi_prepared(
        image_stack, masks, var_stack, out["x0"], out["y0"], SMA, out["eps"], out["pa"]
    )
    assert out["intens_r"] == pytest.approx(float(ring.intens[1].mean()), rel=1e-9)


# ---------------------------------------------------------------------------
# sigma_bg floor
# ---------------------------------------------------------------------------


def test_floor_helper_reads_sigma_bg_as_a_variance():
    assert _sigma_bg_variance_floor(IsosterConfigMB(bands=["g"], reference_band="g", sigma_bg=None)) is None
    assert _sigma_bg_variance_floor(IsosterConfigMB(bands=["g"], reference_band="g", sigma_bg=3.0)) == 9.0


def test_residual_variance_is_raised_to_the_floor():
    """The helper applies the floor itself, so no caller can forget it."""
    n_bands, n_samples = 2, 64
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    coeffs = np.zeros(n_bands + 4)
    coeffs[:n_bands] = [100.0, 70.0]
    rng = np.random.default_rng(0)
    intens = np.vstack([100.0 + rng.normal(0, 0.1, n_samples), 70.0 + rng.normal(0, 0.1, n_samples)])
    weights = np.ones(n_bands)

    unfloored = _compute_joint_residual_variance(
        coeffs, angles, intens, weights, n_bands=n_bands, n_geom_params=n_bands + 4, jagged=False
    )
    floored = _compute_joint_residual_variance(
        coeffs, angles, intens, weights, n_bands=n_bands, n_geom_params=n_bands + 4, jagged=False, floor=25.0
    )

    assert unfloored < 25.0
    assert floored == 25.0


def test_sigma_bg_reaches_both_the_geometry_and_the_intensity_errors():
    """Geometry and per-band intensity errors move together with sigma_bg.

    Before the fix only the geometry errors carried the floor, so a single fit
    reported its parameters on two different error scales.
    """
    images = [make_sersic_image(), make_sersic_image(amplitude=600.0)]

    results = {}
    for sigma_bg in (5.0, 50.0):
        cfg = IsosterConfigMB(bands=["g", "r"], reference_band="g", sigma_bg=sigma_bg)
        results[sigma_bg] = fit_isophote_mb(images, None, SMA, START, cfg)

    # sigma_bg dominates the residual variance at both levels, so both errors
    # scale linearly with it (variance floor sigma_bg**2 -> error ~ sigma_bg).
    eps_ratio = results[50.0]["eps_err"] / results[5.0]["eps_err"]
    intens_ratio = results[50.0]["intens_err_g"] / results[5.0]["intens_err_g"]

    assert intens_ratio == pytest.approx(10.0, rel=0.05)
    assert eps_ratio == pytest.approx(10.0, rel=0.05)


def test_intensity_error_ignores_sigma_bg_when_the_data_dominate():
    """The floor is a lower bound, not a rescale — a noisy ring is unaffected."""
    images = [make_sersic_image(), make_sersic_image(amplitude=600.0)]
    cfg_none = IsosterConfigMB(bands=["g", "r"], reference_band="g", sigma_bg=None)
    cfg_tiny = IsosterConfigMB(bands=["g", "r"], reference_band="g", sigma_bg=1e-6)

    out_none = fit_isophote_mb(images, None, SMA, START, cfg_none)
    out_tiny = fit_isophote_mb(images, None, SMA, START, cfg_tiny)

    assert out_tiny["intens_err_g"] == pytest.approx(out_none["intens_err_g"], rel=1e-12)


# ---------------------------------------------------------------------------
# Exactly-determined solves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [2, 3])
def test_exactly_and_under_determined_solves_report_zero_not_none(n_samples):
    """N <= P carries no noise information, so the residual variance is zero.

    Rows are ``n_bands * n_samples``, so with 6 parameters and 2 bands the
    boundary sits at 3 samples per band: 2 is under-determined, 3 exact.
    """
    n_bands = 2
    n_geom_params = n_bands + 4
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    coeffs = np.zeros(n_geom_params)
    intens = np.vstack([np.arange(n_samples, dtype=float), np.arange(n_samples, dtype=float) + 10.0])

    result = _compute_joint_residual_variance(
        coeffs, angles, intens, np.ones(n_bands), n_bands=n_bands, n_geom_params=n_geom_params, jagged=False
    )
    assert result == 0.0


def test_exactly_determined_solve_still_takes_the_floor():
    """Single-band floors the zero too, so the error scale stays bounded."""
    n_bands, n_samples = 2, 3
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    coeffs = np.zeros(n_bands + 4)
    intens = np.zeros((n_bands, n_samples))

    result = _compute_joint_residual_variance(
        coeffs,
        angles,
        intens,
        np.ones(n_bands),
        n_bands=n_bands,
        n_geom_params=n_bands + 4,
        jagged=False,
        floor=16.0,
    )
    assert result == 16.0


def test_malformed_input_still_returns_none():
    """None is now reserved for inputs the helper cannot use at all."""
    result = _compute_joint_residual_variance(
        np.zeros(6), "not-an-array", np.zeros((2, 8)), np.ones(2), n_bands=2, n_geom_params=6, jagged=False
    )
    assert result is None


def test_exactly_determined_harmonics_report_zero_rather_than_an_unscaled_diagonal():
    """The attacher must not publish a raw (A^T A)^-1 diagonal as an error.

    With ``2 * n_samples == n_bands + 4 + 2L`` the solve is exactly determined.
    The retired code got ``None`` back, skipped the rescale, and stamped
    ``sqrt(diag(cov))`` — here 1.0 — as if it were a standard error.
    """
    bands = ["g", "r"]
    orders = (3, 4)
    n_bands, L = len(bands), len(orders)
    width = n_bands + 4 + 2 * L  # 10
    n_samples = width // n_bands  # 5 rows per band -> 10 total == width

    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    coeffs = np.arange(width, dtype=np.float64)
    cov = np.eye(width)  # sqrt(diag) == 1.0, an obvious sentinel
    intens = np.vstack([np.linspace(100.0, 101.0, n_samples), np.linspace(70.0, 71.0, n_samples)])

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
        band_weights_arr=np.ones(n_bands),
        jagged=False,
    )

    for band in bands:
        for order in orders:
            assert geom[f"a{order}_err_{band}"] == 0.0
            assert geom[f"b{order}_err_{band}"] == 0.0


def test_over_determined_harmonics_still_get_a_real_error():
    """The zero above is specific to N <= P, not a blanket suppression."""
    bands = ["g", "r"]
    orders = (3, 4)
    n_bands, L = len(bands), len(orders)
    width = n_bands + 4 + 2 * L
    n_samples = 64

    rng = np.random.default_rng(5)
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    coeffs = np.zeros(width)
    coeffs[:n_bands] = [100.0, 70.0]
    intens = np.vstack([100.0 + rng.normal(0, 1.0, n_samples), 70.0 + rng.normal(0, 1.0, n_samples)])

    geom: dict = {}
    _attach_simultaneous_higher_harmonics_from_coeffs(
        geom,
        bands,
        coeffs,
        np.eye(width),
        harmonic_orders=orders,
        wls_mode=False,
        angles=angles,
        intens_per_band=intens,
        band_weights_arr=np.ones(n_bands),
        jagged=False,
    )

    assert geom["a3_err_g"] > 0.0
    assert geom["a3_err_g"] != 1.0  # not the unscaled diagonal
