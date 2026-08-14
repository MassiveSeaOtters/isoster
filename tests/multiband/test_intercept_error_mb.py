"""The per-band intensity error must belong to the per-band statistic reported.

In decoupled intercept mode ``intens_<b>`` is a plain ring mean or median, not
a joint-solve output. The retired code always attached the variance of an
inverse-variance-weighted mean to it, so under ``integrator='median'`` it
described a statistic nobody reported — wrong by exactly sqrt(pi/2) even for a
uniform variance map — and under OLS it used the raw ring scatter (which still
contains the fitted m=1 / m=2 signal) instead of the residual scatter
single-band uses.
"""

import numpy as np
import pytest

from isoster._shared import _ring_statistic_and_variance, _weighted_mean_variance
from isoster.multiband.fitting_mb import (
    _per_band_intercept_variance,
    fit_first_and_second_harmonics_joint,
    fit_first_and_second_harmonics_joint_loose,
    fit_simultaneous_joint,
    fit_simultaneous_joint_loose,
)

N_SAMPLES = 128
ORDERS = (3, 4)


def ring_angles(n=N_SAMPLES):
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)


def ring_with_harmonic(level=100.0, amp=5.0, noise=0.3, seed=0, n=N_SAMPLES):
    """A ring carrying a genuine m=2 signal on top of noise.

    The harmonic is deterministic structure the model fits, so it inflates the
    raw scatter but not the uncertainty of the ring's mean.
    """
    rng = np.random.default_rng(seed)
    phi = ring_angles(n)
    return level + amp * np.cos(2.0 * phi) + rng.normal(0.0, noise, n)


def azimuthal_variances(low=1.0, high=100.0, n=N_SAMPLES):
    v = np.full(n, low)
    v[n // 2 :] = high
    return v


# ---------------------------------------------------------------------------
# The estimator-matching rule itself
# ---------------------------------------------------------------------------


def test_wls_mean_intercept_keeps_the_weighted_mean_variance():
    """WLS with integrator='mean' reports the IVW mean, so 1/sum(1/v) is right."""
    intens = ring_with_harmonic()
    variances = azimuthal_variances()
    assert _per_band_intercept_variance(intens, variances, "mean") == pytest.approx(
        _weighted_mean_variance(variances), rel=1e-12
    )


def test_wls_median_intercept_uses_the_median_variance():
    """WLS with integrator='median' reports a plain median, so it needs the
    median's own heteroscedastic variance — not the weighted mean's."""
    intens = ring_with_harmonic()
    variances = azimuthal_variances()
    expected = _ring_statistic_and_variance(intens, variances, "median")[1]

    assert _per_band_intercept_variance(intens, variances, "median") == pytest.approx(expected, rel=1e-12)
    # And it genuinely differs from the retired value.
    assert _per_band_intercept_variance(intens, variances, "median") != pytest.approx(
        _weighted_mean_variance(variances), rel=1e-6
    )


def test_wls_median_is_wrong_by_pi_over_two_even_for_uniform_variance():
    """The retired pairing failed in the simplest possible case.

    With uniform variance the weighted and unweighted means coincide, so the
    only remaining difference is the median's pi/2 penalty.
    """
    intens = ring_with_harmonic()
    variances = np.full(N_SAMPLES, 4.0)

    new = _per_band_intercept_variance(intens, variances, "median")
    retired = _weighted_mean_variance(variances)
    assert new / retired == pytest.approx(np.pi / 2.0, rel=1e-12)


def test_ols_uses_residual_scatter_not_raw_ring_scatter():
    """OLS matches single-band's np.std(intens - model) / sqrt(N).

    The ring's real m=2 signal averages out of the intercept, so leaving it in
    the scatter would overstate the error.
    """
    phi = ring_angles()
    intens = ring_with_harmonic(amp=5.0, noise=0.3)
    residuals = intens - (np.mean(intens) + 5.0 * np.cos(2.0 * phi) - np.mean(5.0 * np.cos(2.0 * phi)))

    from_residual = _per_band_intercept_variance(intens, None, "mean", residuals_b=residuals)
    from_raw = _per_band_intercept_variance(intens, None, "mean")

    assert from_residual == pytest.approx(np.std(residuals) ** 2 / N_SAMPLES, rel=1e-12)
    # The harmonic dominates the raw scatter, so ignoring it inflates the error.
    assert from_raw > 10.0 * from_residual


def test_ols_median_carries_the_pi_over_two_penalty():
    intens = ring_with_harmonic()
    mean_var = _per_band_intercept_variance(intens, None, "mean")
    median_var = _per_band_intercept_variance(intens, None, "median")
    assert median_var / mean_var == pytest.approx(np.pi / 2.0, rel=1e-12)


def test_empty_band_contributes_zero_not_infinity():
    """An infinite entry would poison the joint covariance matrix."""
    empty = np.empty(0, dtype=np.float64)
    assert _per_band_intercept_variance(empty, None, "mean") == 0.0
    assert _per_band_intercept_variance(empty, empty, "median") == 0.0


# ---------------------------------------------------------------------------
# The four decoupled solvers must all write the matching variance
# ---------------------------------------------------------------------------


def solver_cases():
    """(name, callable) for each decoupled solver, normalised to one signature."""
    phi = ring_angles()
    weights = np.ones(2)

    def shared(intens_2d, var_2d, integrator):
        return fit_first_and_second_harmonics_joint(
            phi, intens_2d, weights, var_2d, fit_per_band_intens_jointly=False, integrator=integrator
        )

    def shared_higher(intens_2d, var_2d, integrator):
        return fit_simultaneous_joint(
            phi, intens_2d, weights, ORDERS, var_2d, fit_per_band_intens_jointly=False, integrator=integrator
        )

    def loose(intens_2d, var_2d, integrator):
        return fit_first_and_second_harmonics_joint_loose(
            [phi, phi],
            [intens_2d[0], intens_2d[1]],
            weights,
            None if var_2d is None else [var_2d[0], var_2d[1]],
            fit_per_band_intens_jointly=False,
            integrator=integrator,
        )

    def loose_higher(intens_2d, var_2d, integrator):
        return fit_simultaneous_joint_loose(
            [phi, phi],
            [intens_2d[0], intens_2d[1]],
            weights,
            ORDERS,
            None if var_2d is None else [var_2d[0], var_2d[1]],
            fit_per_band_intens_jointly=False,
            integrator=integrator,
        )

    return [
        ("first_second", shared),
        ("first_second_loose", loose),
        ("simultaneous", shared_higher),
        ("simultaneous_loose", loose_higher),
    ]


@pytest.mark.parametrize("name,solver", solver_cases(), ids=[n for n, _ in solver_cases()])
def test_solver_intercept_covariance_matches_the_reported_statistic_wls(name, solver):
    """Every decoupled solver writes the reported statistic's variance."""
    intens_2d = np.vstack([ring_with_harmonic(seed=1), ring_with_harmonic(level=70.0, seed=2)])
    var_2d = np.vstack([azimuthal_variances(), azimuthal_variances(low=2.0, high=8.0)])

    for integrator in ("mean", "median"):
        _coeffs, cov, wls_mode = solver(intens_2d, var_2d, integrator)
        assert wls_mode is True
        assert cov is not None
        for b in range(2):
            expected = _per_band_intercept_variance(intens_2d[b], var_2d[b], integrator)
            assert cov[b, b] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name,solver", solver_cases(), ids=[n for n, _ in solver_cases()])
def test_solver_intercept_covariance_uses_residual_scatter_ols(name, solver):
    """Under OLS the solvers score the intercept on residuals, not raw scatter.

    Each ring here carries a strong m=2 signal, which the geometric block fits,
    so the raw-scatter form the retired code used was far too large.
    """
    intens_2d = np.vstack([ring_with_harmonic(amp=5.0, seed=3), ring_with_harmonic(amp=5.0, level=70.0, seed=4)])

    _coeffs, cov, wls_mode = solver(intens_2d, None, "mean")
    assert wls_mode is False
    assert cov is not None
    for b in range(2):
        raw = float(np.var(intens_2d[b], ddof=1) / N_SAMPLES)
        assert cov[b, b] < raw / 10.0
        # It is a real scatter estimate, not zero.
        assert cov[b, b] > 0.0
