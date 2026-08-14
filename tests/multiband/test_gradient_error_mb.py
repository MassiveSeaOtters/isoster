"""The multi-band gradient error must be built from the same ring statistics
the gradient uses, on the same samples — matching single-band.

The retired code always paired the reported per-band statistic with the
variance of an inverse-variance-weighted mean, so under ``integrator='median'``
it described an estimator nobody reported, and under OLS it dropped the
median's pi/2 penalty entirely.
"""

import numpy as np
import pytest

from isoster._shared import _ring_statistic_and_variance, _weighted_mean_variance
from isoster.fitting import compute_gradient
from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import compute_joint_gradient
from isoster.multiband.sampling_mb import extract_isophote_data_multi_prepared, prepare_inputs

CENTER = 60.0
SMA = 20.0
EPS = 0.3
PA = 0.4
STEP = 0.1
GEOMETRY = {"x0": CENTER, "y0": CENTER, "sma": SMA, "eps": EPS, "pa": PA}


def make_sersic_image(size=121, amplitude=1000.0, r_eff=20.0):
    from scipy.special import gammaincinv

    bn = gammaincinv(4.0, 0.5)
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((xx - size // 2) ** 2 + ((yy - size // 2) / 0.7) ** 2)
    return amplitude * np.exp(-bn * ((r / r_eff) ** 0.5 - 1.0))


def make_config(integrator="mean", bands=("g", "r")):
    # A median cannot come out of a linear least-squares intercept column, so
    # the config validator requires the decoupled intercept mode alongside it.
    return IsosterConfigMB(
        bands=list(bands),
        reference_band=bands[0],
        astep=STEP,
        linear_growth=False,
        integrator=integrator,
        fit_per_band_intens_jointly=(integrator != "median"),
        use_eccentric_anomaly=False,
    )


def gradient_inputs(images, variance_maps):
    """Resolve the driver-level stacks the joint gradient consumes."""
    return prepare_inputs(images, None, variance_maps)


def ring(image_stack, var_stack, sma):
    return extract_isophote_data_multi_prepared(
        image_stack, [None] * image_stack.shape[0], var_stack, CENTER, CENTER, sma, EPS, PA
    )


def expected_per_band_error(image_stack, var_stack, band, integrator, sma=SMA, step=STEP):
    """Rebuild one band's error from the same public pieces the code uses."""
    delta_r = sma * step
    data_c = ring(image_stack, var_stack, sma)
    data_g = ring(image_stack, var_stack, sma * (1.0 + step))
    var_c = data_c.variances[band] if data_c.variances is not None else None
    var_g = data_g.variances[band] if data_g.variances is not None else None
    _, ring_var_c = _ring_statistic_and_variance(data_c.intens[band], var_c, integrator)
    _, ring_var_g = _ring_statistic_and_variance(data_g.intens[band], var_g, integrator)
    return np.sqrt(ring_var_c + ring_var_g) / delta_r


def smooth_variance_map(size=121, low=0.5, high=40.0):
    """A variance map that varies smoothly across the image, so the weighted and
    unweighted forms genuinely disagree."""
    yy, xx = np.mgrid[:size, :size]
    ramp = (xx + yy) / float(2 * (size - 1))
    return low + (high - low) * ramp


def azimuthal_variance_map(size=121, low=1.0, high=100.0):
    """A variance map split across the ring, giving the two forms maximum spread.

    A ramp across the image barely varies along a single ring, so it makes the
    weighted and unweighted means differ only slightly. Splitting by azimuth
    puts half of each ring's samples at each level.
    """
    yy, _xx = np.mgrid[:size, :size]
    return np.where(yy < size // 2, low, high).astype(np.float64)


@pytest.mark.parametrize("integrator", ["mean", "median"])
def test_per_band_error_matches_the_reported_ring_statistic(integrator):
    """Each band's gradient error is the variance of the statistic it reported."""
    images = [make_sersic_image(), make_sersic_image(amplitude=600.0)]
    variance_maps = [smooth_variance_map(), smooth_variance_map(low=1.0, high=9.0)]
    image_stack, masks, var_stack = gradient_inputs(images, variance_maps)
    cfg = make_config(integrator)

    _, _, _per_band_grad, per_band_err = compute_joint_gradient(
        image_stack, masks, var_stack, GEOMETRY, cfg, np.ones(2), previous_gradient=None
    )

    for band in range(2):
        assert per_band_err[band] == pytest.approx(
            expected_per_band_error(image_stack, var_stack, band, integrator), rel=1e-12
        )


def test_heteroscedastic_mean_error_exceeds_the_old_inverse_weighted_result():
    """The unweighted mean's variance is never below the weighted mean's.

    Cauchy-Schwarz gives ``1/sum(1/v) <= sum(v)/N**2``, so the retired formula
    always understated the error of the mean that was actually reported.
    """
    for variance_map, min_ratio in ((smooth_variance_map(), 1.0), (azimuthal_variance_map(), 4.0)):
        image_stack, masks, var_stack = gradient_inputs([make_sersic_image()], [variance_map])

        _, _, _grads, errs = compute_joint_gradient(
            image_stack,
            masks,
            var_stack,
            GEOMETRY,
            make_config("mean", bands=("g",)),
            np.ones(1),
            previous_gradient=None,
        )

        delta_r = SMA * STEP
        data_c = ring(image_stack, var_stack, SMA)
        data_g = ring(image_stack, var_stack, SMA * (1.0 + STEP))
        retired = np.sqrt(_weighted_mean_variance(data_c.variances[0]) + _weighted_mean_variance(data_g.variances[0]))
        retired /= delta_r

        # The inequality is guaranteed by Cauchy-Schwarz; how far apart the two
        # forms land depends on how much the variance varies around the ring.
        assert errs[0] > retired
        assert errs[0] / retired > min_ratio


def test_uniform_variance_reproduces_the_old_result():
    """With uniform variance both forms coincide, so nothing moves."""
    images = [make_sersic_image()]
    variance_maps = [np.full((121, 121), 4.0)]
    image_stack, masks, var_stack = gradient_inputs(images, variance_maps)

    _, _, _grads, errs = compute_joint_gradient(
        image_stack, masks, var_stack, GEOMETRY, make_config("mean", bands=("g",)), np.ones(1), previous_gradient=None
    )

    delta_r = SMA * STEP
    data_c = ring(image_stack, var_stack, SMA)
    data_g = ring(image_stack, var_stack, SMA * (1.0 + STEP))
    retired = np.sqrt(_weighted_mean_variance(data_c.variances[0]) + _weighted_mean_variance(data_g.variances[0]))
    retired /= delta_r

    assert errs[0] == pytest.approx(retired, rel=1e-12)


def test_ols_median_error_carries_the_pi_over_two_penalty():
    """Without a variance map the median's error is sqrt(pi/2) times the mean's.

    The retired code used the mean's scatter formula for both integrators, so
    the median's error was reported about 20% too small.
    """
    images = [make_sersic_image()]
    image_stack, masks, var_stack = gradient_inputs(images, None)

    _, _, _g_mean, errs_mean = compute_joint_gradient(
        image_stack, masks, var_stack, GEOMETRY, make_config("mean", bands=("g",)), np.ones(1), previous_gradient=None
    )
    _, _, _g_med, errs_median = compute_joint_gradient(
        image_stack, masks, var_stack, GEOMETRY, make_config("median", bands=("g",)), np.ones(1), previous_gradient=None
    )

    assert errs_median[0] / errs_mean[0] == pytest.approx(np.sqrt(np.pi / 2.0), rel=1e-12)


@pytest.mark.parametrize("integrator", ["mean", "median"])
@pytest.mark.parametrize("with_variance", [False, True])
def test_b1_gradient_matches_single_band(integrator, with_variance):
    """At B=1 the joint gradient and its error equal single-band's exactly.

    This is the alignment the whole item is about: one band, one image, one
    variance map, and the two code paths must not disagree.
    """
    image = make_sersic_image()
    variance_map = smooth_variance_map() if with_variance else None
    image_stack, masks, var_stack = gradient_inputs([image], [variance_map] if with_variance else None)

    grad_mb, err_mb, _pb_g, _pb_e = compute_joint_gradient(
        image_stack,
        masks,
        var_stack,
        GEOMETRY,
        make_config(integrator, bands=("g",)),
        np.ones(1),
        previous_gradient=None,
    )
    grad_sb, err_sb = compute_gradient(
        image,
        None,
        GEOMETRY,
        {
            "astep": STEP,
            "linear_growth": False,
            "integrator": integrator,
            "use_eccentric_anomaly": False,
        },
        previous_gradient=None,
        variance_map=variance_map,
    )

    assert grad_mb == pytest.approx(grad_sb, rel=1e-12)
    assert err_mb == pytest.approx(err_sb, rel=1e-12)


def test_runaway_gradient_decays_and_clears_the_error():
    """A runaway gradient reports no uncertainty rather than a stale one.

    When the fresh gradient fails the ``previous_gradient / 3`` sanity check the
    value is replaced by a decayed estimate, so the error computed alongside the
    discarded value must not travel with it. This is the one reachable path that
    returns ``None`` for the joint error: an empty ring is caught earlier, and
    under shared validity no single band can be empty while the intersection is
    not.
    """
    images = [make_sersic_image()]
    image_stack, masks, var_stack = gradient_inputs(images, [smooth_variance_map()])

    # A hugely negative previous_gradient puts previous/3 below the real
    # gradient (about -82 here), so the runaway branch fires.
    previous_gradient = -1.0e9
    grad, err, _pb_g, per_band_err = compute_joint_gradient(
        image_stack,
        masks,
        var_stack,
        GEOMETRY,
        make_config("mean", bands=("g",)),
        np.ones(1),
        previous_gradient=previous_gradient,
    )

    assert err is None
    assert grad == pytest.approx(previous_gradient * 0.8, rel=1e-12)
    # The per-band values are untouched by the decay, so they still describe the
    # ring that was actually measured.
    assert per_band_err[0] == pytest.approx(expected_per_band_error(image_stack, var_stack, 0, "mean"), rel=1e-12)
