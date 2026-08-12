"""The gradient error must be built from the same ring statistics the gradient
uses, on the same samples.
"""

import numpy as np
import pytest

from isoster._shared import _ring_statistic_and_variance
from isoster.fitting import compute_gradient
from isoster.sampling import extract_isophote_data

CENTER = 60.0
SMA = 20.0
GEOMETRY = {"x0": CENTER, "y0": CENTER, "sma": SMA, "eps": 0.3, "pa": 0.4}


def make_sersic_image(size=121, amplitude=1000.0, r_eff=20.0):
    from scipy.special import gammaincinv

    bn = gammaincinv(4.0, 0.5)
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((xx - size // 2) ** 2 + ((yy - size // 2) / 0.7) ** 2)
    return amplitude * np.exp(-bn * ((r / r_eff) ** 0.5 - 1.0))


def gradient_config(integrator="mean"):
    return {
        "astep": 0.1,
        "linear_growth": False,
        "integrator": integrator,
        "use_eccentric_anomaly": False,
    }


def expected_error(image, variance_map, integrator, sma=SMA, step=0.1):
    """Rebuild the error from the same public pieces compute_gradient uses."""
    delta_r = sma * step
    _, var_c = _ring_statistic_and_variance(*_ring_inputs(image, variance_map, sma), integrator)
    _, var_g = _ring_statistic_and_variance(*_ring_inputs(image, variance_map, sma * (1.0 + step)), integrator)
    return np.sqrt(var_c + var_g) / delta_r


def _ring_inputs(image, variance_map, sma):
    data = extract_isophote_data(image, None, CENTER, CENTER, sma, 0.3, 0.4, variance_map=variance_map)
    return data.intens, data.variances


@pytest.mark.parametrize("integrator", ["mean", "median"])
def test_gradient_error_matches_the_reported_ring_statistic(integrator):
    """Heteroscedastic ring: the error must follow the statistic, not a weighted mean."""
    image = make_sersic_image()
    variance = np.full(image.shape, 4.0)
    variance[:, 75:] = 400.0  # a noisy strip crossing the rings

    _, error = compute_gradient(image, None, GEOMETRY, gradient_config(integrator), variance_map=variance)

    assert error == pytest.approx(expected_error(image, variance, integrator), rel=1e-9)


def test_heteroscedastic_mean_error_exceeds_the_old_inverse_weighted_result():
    """Explicitly reject the superseded formula."""
    from isoster._shared import _weighted_mean_variance

    image = make_sersic_image()
    variance = np.full(image.shape, 4.0)
    variance[:, 75:] = 400.0

    _, error = compute_gradient(image, None, GEOMETRY, gradient_config("mean"), variance_map=variance)

    delta_r = SMA * 0.1
    intens_c, var_c = _ring_inputs(image, variance, SMA)
    intens_g, var_g = _ring_inputs(image, variance, SMA * 1.1)
    old = np.sqrt(_weighted_mean_variance(var_c) + _weighted_mean_variance(var_g)) / delta_r

    assert error > 1.5 * old


def test_uniform_variance_matches_the_old_result():
    """With uniform variance the two formulas coincide, so nothing may move."""
    from isoster._shared import _weighted_mean_variance

    image = make_sersic_image()
    variance = np.full(image.shape, 4.0)

    _, error = compute_gradient(image, None, GEOMETRY, gradient_config("mean"), variance_map=variance)

    delta_r = SMA * 0.1
    _, var_c = _ring_inputs(image, variance, SMA)
    _, var_g = _ring_inputs(image, variance, SMA * 1.1)
    old = np.sqrt(_weighted_mean_variance(var_c) + _weighted_mean_variance(var_g)) / delta_r

    assert error == pytest.approx(old, rel=1e-9)


def test_median_error_exceeds_mean_error_for_the_same_ring():
    """The median's pi/2 penalty must reach the gradient error."""
    image = make_sersic_image()
    variance = np.full(image.shape, 4.0)

    _, mean_error = compute_gradient(image, None, GEOMETRY, gradient_config("mean"), variance_map=variance)
    _, median_error = compute_gradient(image, None, GEOMETRY, gradient_config("median"), variance_map=variance)

    assert median_error == pytest.approx(np.sqrt(np.pi / 2.0) * mean_error, rel=1e-9)


def test_no_variance_map_mean_path_is_unchanged():
    """The scatter-based mean estimate must stay exactly as it was."""
    image = make_sersic_image()
    _, error = compute_gradient(image, None, GEOMETRY, gradient_config("mean"))

    delta_r = SMA * 0.1
    intens_c, _ = _ring_inputs(image, None, SMA)
    intens_g, _ = _ring_inputs(image, None, SMA * 1.1)
    old = np.sqrt(np.std(intens_c) ** 2 / intens_c.size + np.std(intens_g) ** 2 / intens_g.size) / delta_r

    assert error == pytest.approx(old, rel=1e-12)


def test_cached_current_data_matches_freshly_sampled_data():
    """Passing the current ring in via current_data must not change the answer."""
    image = make_sersic_image()
    variance = np.full(image.shape, 4.0)
    variance[:, 75:] = 400.0
    data = extract_isophote_data(image, None, CENTER, CENTER, SMA, 0.3, 0.4, variance_map=variance)

    _, fresh = compute_gradient(image, None, GEOMETRY, gradient_config("mean"), variance_map=variance)
    _, cached = compute_gradient(
        image,
        None,
        GEOMETRY,
        gradient_config("mean"),
        current_data=(data.phi, data.intens, data.variances),
        variance_map=variance,
    )

    assert cached == pytest.approx(fresh, rel=1e-12)


def make_truncated_disk_image(size=121, amplitude=30.0, disk_scale=30.0, r_truncation=23.5, truncation_rate=8.0):
    """An exponential disk with a sharp outer edge.

    The plain Sersic bulge in ``make_sersic_image`` steepens monotonically
    outward, so its two-step gradient (SMA to 1.2*SMA) is always shallower
    than its one-step gradient (SMA to 1.1*SMA) and the EFF-1 early-exit in
    ``compute_gradient`` never keeps the second baseline's result. This
    profile is flat out to ``r_truncation`` (between 1.0*SMA and 1.1*SMA)
    and then drops steeply, so the two-step gradient is the steeper one and
    can survive the final ``gradient >= previous_gradient / 3`` check.
    """
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((xx - size // 2) ** 2 + ((yy - size // 2) / 0.7) ** 2)
    inner = amplitude * np.exp(-r / disk_scale)
    edge_value = amplitude * np.exp(-r_truncation / disk_scale)
    outer = edge_value * np.exp(-truncation_rate * (r - r_truncation) / disk_scale)
    return np.where(r < r_truncation, inner, outer)


def test_second_baseline_uses_the_matched_variance():
    """The longer two-step baseline must use the same helper as the first.

    With this fixture the one-step gradient is -1.017 and the two-step
    gradient is -1.185 (steeper, because the truncation edge falls between
    1.1*SMA and 1.2*SMA). previous_gradient=-3.2 places previous_gradient/3
    (-1.067) strictly between them, so compute_gradient takes the second
    baseline (-1.017 >= -1.067) and keeps it (-1.185 < -1.067, the final
    early-exit override does not fire).
    """
    image = make_truncated_disk_image()
    variance = np.full(image.shape, 4.0)
    variance[:, 75:] = 400.0

    grad, error = compute_gradient(
        image, None, GEOMETRY, gradient_config("mean"), previous_gradient=-3.2, variance_map=variance
    )

    assert error is not None, "fixture must reach and keep the second baseline"
    assert grad == pytest.approx(-1.1851899005215216, rel=1e-9)

    delta_r_2 = SMA * 0.2
    _, var_c = _ring_statistic_and_variance(*_ring_inputs(image, variance, SMA), "mean")
    _, var_g2 = _ring_statistic_and_variance(*_ring_inputs(image, variance, SMA * 1.2), "mean")
    assert error == pytest.approx(np.sqrt(var_c + var_g2) / delta_r_2, rel=1e-9)


def test_corrected_error_can_change_which_baseline_is_selected():
    """A more honest ring error can change which baseline compute_gradient reports.

    This is intended, not a regression. gradient_error is not just descriptive:
    it gates two decisions downstream, ``need_second_gradient`` at
    isoster/fitting.py:908 (via ``relative_error >= 0.3``) and the final
    baseline-selection override at isoster/fitting.py:940 (via
    ``gradient >= previous_gradient / 3``). The old inverse-variance-weighted
    formula understated this ring's error enough to keep ``relative_error``
    under 0.3, so the first (one-step) baseline was always kept. The
    corrected formula reports the unweighted mean's true, larger error, which
    pushes ``relative_error`` past 0.3 and lets the second (two-step) baseline
    be taken and kept instead. A more truthful per-ring uncertainty can
    therefore change which ring's gradient compute_gradient reports.
    """
    from isoster._shared import _weighted_mean_variance

    image = make_truncated_disk_image()
    variance = np.full(image.shape, 4.0)
    variance[:, 75:] = 400.0

    delta_r = SMA * 0.1
    intens_c, var_c = _ring_inputs(image, variance, SMA)
    intens_g, var_g = _ring_inputs(image, variance, SMA * 1.1)
    one_step_gradient = (np.mean(intens_g) - np.mean(intens_c)) / delta_r

    old_relative_error = abs(
        np.sqrt(_weighted_mean_variance(var_c) + _weighted_mean_variance(var_g)) / delta_r / one_step_gradient
    )
    assert old_relative_error < 0.3, "fixture must reproduce the old formula staying under threshold"

    _, new_var_c = _ring_statistic_and_variance(intens_c, var_c, "mean")
    _, new_var_g = _ring_statistic_and_variance(intens_g, var_g, "mean")
    new_relative_error = abs(np.sqrt(new_var_c + new_var_g) / delta_r / one_step_gradient)
    assert new_relative_error >= 0.3, "fixture must reproduce the corrected formula crossing threshold"

    gradient, error = compute_gradient(
        image, None, GEOMETRY, gradient_config("mean"), previous_gradient=-3.2, variance_map=variance
    )

    assert error is not None, "the corrected error must let the second baseline survive"
    assert gradient != pytest.approx(one_step_gradient, rel=1e-6), "the reported gradient must be the second baseline"
