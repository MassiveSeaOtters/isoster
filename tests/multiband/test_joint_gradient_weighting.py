"""The joint gradient must be pooled with the weights the harmonic solve used.

A common geometry step ``delta`` produces, in band ``b``, a ring harmonic of
amplitude ``delta * grad_b`` — proportional to *that band's* radial gradient. The
joint solve forces one shared amplitude across bands, so what it fits is

    A1 = delta * sum_b(W_b * grad_b) / sum_b(W_b),     W_b = w_b / var_b

i.e. ``delta`` times a weight-averaged gradient. Recovering ``delta`` therefore
requires dividing by a gradient pooled with those same ``W_b``. Pooling with the
bare band weight leaves the variance factors uncancelled: measured recoveries of
0.180 and 0.045 against a truth of 0.100 at different band configurations, so the
error is not even one-signed.

Reference mode has the same defect in a different form — it fits harmonics from
the reference band alone but divided by an all-band pooled gradient, which is
wrong under OLS as well as WLS.
"""

import numpy as np
import pytest

from isoster.multiband.fitting_mb import (
    fit_first_and_second_harmonics_joint,
    fit_first_and_second_harmonics_ref,
    joint_gradient_pooling_weights,
)

N_SAMPLES = 256
DELTA = 0.100  # the common geometry step every test must recover
PHI = np.linspace(0.0, 2.0 * np.pi, N_SAMPLES, endpoint=False)


def planted_rings(gradients, levels=None, noise=None, seed=0):
    """Rings carrying a common geometry step ``DELTA``.

    Band b shows amplitude ``DELTA * gradients[b]``, which is what a real shared
    displacement produces — unlike the existing joint-solver test, which plants
    identical amplitudes in every band and so cannot see this class of defect.
    """
    gradients = np.asarray(gradients, dtype=np.float64)
    if levels is None:
        levels = np.full(gradients.size, 100.0)
    rows = [levels[b] + DELTA * gradients[b] * np.sin(PHI) for b in range(gradients.size)]
    out = np.vstack(rows)
    if noise is not None:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0.0, 1.0, out.shape) * np.asarray(noise)[:, None]
    return out


def uniform_variances(values):
    return np.vstack([np.full(N_SAMPLES, v) for v in values])


def recovered_delta(gradients, variances, band_weights):
    """Fit the shared amplitude, then divide by the consistently pooled gradient."""
    gradients = np.asarray(gradients, dtype=np.float64)
    band_weights = np.asarray(band_weights, dtype=np.float64)
    var2d = None if variances is None else uniform_variances(variances)
    intens = planted_rings(gradients)
    coeffs, _cov, _wls = fit_first_and_second_harmonics_joint(PHI, intens, band_weights, var2d)
    shared_a1 = float(coeffs[gradients.size])
    weights = joint_gradient_pooling_weights(band_weights, var2d, gradients.size)
    pooled_gradient = float(np.sum(weights * gradients) / np.sum(weights))
    return shared_a1 / pooled_gradient


# ---------------------------------------------------------------------------
# T1.1 / T1.6 — joint mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gradients,variances",
    [
        ([-100.0, -10.0], [1.0, 100.0]),  # bright/quiet + faint/noisy
        ([-100.0, -10.0], [100.0, 1.0]),  # the opposite pairing
        ([-80.0, -40.0, -5.0], [1.0, 9.0, 400.0]),  # three bands
        ([-100.0, -10.0], [1.0, 1.0]),  # equal variance: the easy case
    ],
)
def test_joint_wls_recovers_the_planted_geometry_step(gradients, variances):
    """T1.1: exact recovery for band-dependent variance."""
    assert recovered_delta(gradients, variances, np.ones(len(gradients))) == pytest.approx(DELTA, rel=1e-9)


def test_uniform_variance_leaves_the_pooling_weights_proportional(gradients=(-100.0, -10.0)):
    """T1.6: with one variance for every band the weights stay proportional to w_b.

    The pooled gradient is then identical to the retired band-weight pooling, so
    nothing moves in that configuration — which is exactly why the existing tests
    never caught this.
    """
    band_weights = np.array([1.0, 3.0])
    var2d = uniform_variances([4.0, 4.0])
    weights = joint_gradient_pooling_weights(band_weights, var2d, 2)
    grads = np.asarray(gradients)
    assert np.sum(weights * grads) / np.sum(weights) == pytest.approx(
        np.sum(band_weights * grads) / np.sum(band_weights), rel=1e-12
    )


def test_ols_pooling_weights_are_the_band_weights():
    """T1.6: no variance map means the solve weights by w_b, so pooling must too."""
    band_weights = np.array([1.0, 2.5, 0.5])
    np.testing.assert_array_equal(joint_gradient_pooling_weights(band_weights, None, 3), band_weights)


def test_retired_pooling_is_measurably_wrong():
    """Guard the failure mode itself, so a regression cannot pass silently."""
    gradients = np.array([-100.0, -10.0])
    band_weights = np.ones(2)
    var2d = uniform_variances([1.0, 100.0])
    intens = planted_rings(gradients)
    coeffs, _cov, _wls = fit_first_and_second_harmonics_joint(PHI, intens, band_weights, var2d)
    retired = float(coeffs[2]) / float(np.sum(band_weights * gradients) / np.sum(band_weights))
    assert retired == pytest.approx(0.180198, rel=1e-4)
    assert abs(retired - DELTA) > 0.5 * DELTA


# ---------------------------------------------------------------------------
# T1.2 — reference mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_variance", [False, True])
def test_reference_mode_recovers_the_planted_step_from_its_own_gradient(with_variance):
    """T1.2: ref mode is wrong under OLS too, because it pooled across bands."""
    gradients = np.array([-100.0, -10.0])
    variances = uniform_variances([1.0, 100.0]) if with_variance else None
    intens = planted_rings(gradients)
    ref_variances = variances[0] if variances is not None else None

    coeffs_ref, _cov, _wls = fit_first_and_second_harmonics_ref(PHI, intens[0], ref_variances)
    shared_a1 = float(coeffs_ref[1])

    # What the code does now: a one-hot pooling weight on the reference band.
    one_hot = np.array([1.0, 0.0])
    ref_gradient = float(np.sum(one_hot * gradients) / np.sum(one_hot))
    assert shared_a1 / ref_gradient == pytest.approx(DELTA, rel=1e-9)

    # The retired all-band pooling, kept so the defect stays visible.
    pooled = float(np.mean(gradients))
    assert abs(shared_a1 / pooled - DELTA) > 0.5 * DELTA


# ---------------------------------------------------------------------------
# T1.3 — unit invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [10.0, 0.1, 1000.0])
def test_recovered_step_is_invariant_to_a_bands_flux_units(scale):
    """T1.3: re-expressing one band's flux units cannot change the geometry.

    Flux x scale means gradient x scale and variance x scale**2 — the same
    physical measurement in different numerical units.
    """
    base_gradients = np.array([-100.0, -10.0])
    base_variances = np.array([1.0, 100.0])
    band_weights = np.ones(2)

    reference = recovered_delta(base_gradients, base_variances, band_weights)
    rescaled = recovered_delta(
        base_gradients * np.array([1.0, scale]),
        base_variances * np.array([1.0, scale**2]),
        band_weights,
    )
    assert reference == pytest.approx(DELTA, rel=1e-9)
    assert rescaled == pytest.approx(reference, rel=1e-9)


# ---------------------------------------------------------------------------
# T1.5 — the weights come from the samples the solve saw
# ---------------------------------------------------------------------------


def test_pooling_weights_use_the_supplied_post_clipping_samples():
    """T1.5: weights must reflect the clipped set, not the raw ring.

    Sigma clipping removes samples before the solve. Passing the unfiltered ring
    would put the pooled gradient and the fitted amplitude back on different
    sample sets — the class of mismatch this work exists to remove.
    """
    band_weights = np.ones(2)
    full = uniform_variances([1.0, 100.0])
    clipped = np.vstack([full[0][:200], full[1][:200]])

    weights_full = joint_gradient_pooling_weights(band_weights, full, 2)
    weights_clipped = joint_gradient_pooling_weights(band_weights, clipped, 2)

    # The weight is a sum over samples, so dropping samples must lower it.
    assert weights_clipped[0] == pytest.approx(200.0, rel=1e-12)
    assert weights_full[0] == pytest.approx(256.0, rel=1e-12)
    # The ratio between bands is what actually drives the pooling, and it is
    # unchanged here because both bands lost the same samples.
    assert weights_clipped[0] / weights_clipped[1] == pytest.approx(weights_full[0] / weights_full[1], rel=1e-12)


def test_pooling_weights_handle_jagged_bands_and_dropped_indices():
    """Loose validity: per-band sample counts differ and bands can be absent."""
    band_weights = np.array([1.0, 1.0, 1.0])
    jagged = [np.full(100, 1.0), np.full(50, 4.0)]  # only bands 0 and 2 survive
    weights = joint_gradient_pooling_weights(band_weights, jagged, 3, band_indices=[0, 2])
    assert weights[0] == pytest.approx(100.0, rel=1e-12)
    assert weights[2] == pytest.approx(50.0 / 4.0, rel=1e-12)
    assert weights[1] == 1.0  # untouched; band 1 is not in the solve
