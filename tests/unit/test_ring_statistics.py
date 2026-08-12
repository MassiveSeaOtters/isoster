"""A ring's location and the variance reported for it must describe the same
estimator computed from the same samples.

The inverse-variance-weighted mean's variance, 1/sum(1/v), is only correct for
the inverse-variance-weighted mean. Isoster's gradient reports an unweighted
mean or an unweighted median, which need their own variances.
"""

import numpy as np
import pytest

from isoster._shared import _ring_statistic_and_variance


def test_mean_variance_matches_the_unweighted_mean():
    """Var(mean) = sum(v)/N^2, not the inverse-variance-weighted 1/sum(1/v)."""
    variances = np.concatenate([np.full(64, 1.0), np.full(64, 100.0)])
    intens = np.full(variances.size, 5.0)

    location, variance = _ring_statistic_and_variance(intens, variances, "mean")

    assert location == pytest.approx(5.0)
    assert variance == pytest.approx(np.sum(variances) / variances.size**2)

    inverse_weighted = 1.0 / np.sum(1.0 / variances)
    assert variance > 20.0 * inverse_weighted  # measured factor is 25.5


def test_mean_variance_agrees_with_inverse_weighting_for_uniform_variance():
    """For uniform variance the two formulas coincide exactly."""
    variances = np.full(128, 7.0)
    intens = np.full(128, 5.0)

    _, variance = _ring_statistic_and_variance(intens, variances, "mean")

    assert variance == pytest.approx(1.0 / np.sum(1.0 / variances), rel=1e-12)


def test_median_variance_carries_the_pi_over_two_penalty():
    """A median is noisier than a mean by pi/2 in variance under uniform noise."""
    variances = np.full(101, 3.0)
    intens = np.zeros(101)

    _, median_variance = _ring_statistic_and_variance(intens, variances, "median")
    _, mean_variance = _ring_statistic_and_variance(intens, variances, "mean")

    assert median_variance / mean_variance == pytest.approx(np.pi / 2.0, rel=1e-12)


def test_median_variance_tracks_monte_carlo_for_uniform_noise():
    """The analytic median variance must match simulation."""
    rng = np.random.default_rng(20260812)
    sigma, n, trials = 2.0, 101, 20000
    draws = rng.normal(0.0, sigma, size=(trials, n))
    empirical = np.var(np.median(draws, axis=1))

    _, analytic = _ring_statistic_and_variance(np.zeros(n), np.full(n, sigma**2), "median")

    assert analytic == pytest.approx(empirical, rel=0.06)


def test_median_variance_tracks_monte_carlo_for_heteroscedastic_noise():
    """The documented approximation must hold when variances differ across the ring."""
    rng = np.random.default_rng(20260812)
    variances = np.concatenate([np.full(50, 1.0), np.full(51, 100.0)])
    trials = 20000
    draws = rng.normal(0.0, np.sqrt(variances), size=(trials, variances.size))
    empirical = np.var(np.median(draws, axis=1))

    _, analytic = _ring_statistic_and_variance(np.zeros(variances.size), variances, "median")

    assert analytic == pytest.approx(empirical, rel=0.10)


def test_scatter_based_mean_variance_without_a_variance_map():
    """Without variances the estimate stays the existing scatter-based one."""
    rng = np.random.default_rng(3)
    intens = rng.normal(10.0, 2.0, 200)

    location, variance = _ring_statistic_and_variance(intens, None, "mean")

    assert location == pytest.approx(np.mean(intens))
    assert variance == pytest.approx(np.std(intens) ** 2 / intens.size, rel=1e-12)


def test_scatter_based_median_variance_without_a_variance_map():
    """The median's pi/2 penalty applies to the scatter-based estimate too."""
    rng = np.random.default_rng(3)
    intens = rng.normal(10.0, 2.0, 200)

    location, variance = _ring_statistic_and_variance(intens, None, "median")

    assert location == pytest.approx(np.median(intens))
    assert variance == pytest.approx((np.pi / 2.0) * np.std(intens) ** 2 / intens.size, rel=1e-12)


def test_empty_ring_returns_unknown():
    """An empty ring has no location and no usable uncertainty."""
    location, variance = _ring_statistic_and_variance(np.array([]), None, "mean")
    assert np.isnan(location)
    assert np.isinf(variance)


def test_unknown_integrator_falls_back_to_mean():
    """compute_gradient passes 'adaptive' through unresolved in direct calls."""
    intens = np.arange(10.0)
    location, variance = _ring_statistic_and_variance(intens, None, "adaptive")
    assert location == pytest.approx(np.mean(intens))
    assert variance == pytest.approx(np.std(intens) ** 2 / intens.size, rel=1e-12)
