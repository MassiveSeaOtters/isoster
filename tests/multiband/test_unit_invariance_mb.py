"""Phase 3: a full multi-band fit must not depend on a band's flux units.

Expressing one band in different numerical units — flux x s, and therefore
variance x s**2 — is the same physical measurement written differently. Every
reported quantity should be unchanged: geometry, uncertainties, the stop code,
the convergence decision and the iteration count.

Two independent defects broke that, and both had to be fixed:

* the convergence test compared a variance-weighted shared amplitude against an
  *unweighted* pooled scatter, so the stopping decision moved with a band's raw
  numbers (iteration count 8 vs 7);
* the amplitude parameterisation's covariance is not unit-invariant, so the
  reported geometry uncertainties moved by ~15% even once convergence was fixed.

The first is fixed for every configuration by the variance-weighted convergence
scatter; the second requires ``geometry_parameterized_solve=True``.
"""

import numpy as np
import pytest

from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import fit_isophote_mb

TRUE_GEOMETRY = {"x0": 128.0, "y0": 128.0, "eps": 0.25, "pa": 0.45}
BAND_AMPLITUDES = [12.0, 45.0, 80.0]
BAND_SIGMA = [0.023, 0.052, 0.046]
START_GEOMETRY = {"x0": 128.0, "y0": 128.0, "eps": 0.20, "pa": 0.40}
RESCALED_BAND = 1

GEOMETRY_KEYS = ("x0", "y0", "eps", "pa")
ERROR_KEYS = ("x0_err", "y0_err", "eps_err", "pa_err")
DECISION_KEYS = ("stop_code", "niter")


def planted_galaxy(h, w, x0, y0, eps, pa, amplitude, re, noise_sigma, seed, n_sersic=1.5):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = x - x0, y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    bn = 2.0 * n_sersic - 0.327
    img = amplitude * np.exp(-bn * ((r / re) ** (1.0 / n_sersic) - 1.0))
    return img + rng.normal(0.0, noise_sigma, size=img.shape)


def fit_with_units(scale, geometry_parameterized, sma=30.0):
    """Fit the same galaxy with band ``RESCALED_BAND`` in different flux units.

    The noise draw is seeded per band and multiplied by the same factor as the
    signal, so the two runs differ *only* by a change of units — not by a
    different noise realisation.
    """
    factors = [1.0, 1.0, 1.0]
    factors[RESCALED_BAND] = scale
    images = [
        planted_galaxy(
            256,
            256,
            TRUE_GEOMETRY["x0"],
            TRUE_GEOMETRY["y0"],
            TRUE_GEOMETRY["eps"],
            TRUE_GEOMETRY["pa"],
            BAND_AMPLITUDES[b] * factors[b],
            25.0,
            BAND_SIGMA[b] * factors[b],
            seed=1000 * b + 7,
        )
        for b in range(3)
    ]
    variance_maps = [np.full((256, 256), (BAND_SIGMA[b] * factors[b]) ** 2) for b in range(3)]
    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="i",
        nclip=0,
        geometry_parameterized_solve=geometry_parameterized,
    )
    return fit_isophote_mb(images, None, sma, START_GEOMETRY, cfg, variance_maps=variance_maps)


def relative_change(a, b, key):
    va, vb = float(a[key]), float(b[key])
    return abs(vb - va) / abs(va) if va else abs(vb - va)


@pytest.mark.parametrize("scale", [10.0, 0.1])
def test_full_fit_is_unit_invariant_under_the_geometry_parameterisation(scale):
    """Geometry, uncertainties, stop code and iteration count all invariant."""
    reference = fit_with_units(1.0, geometry_parameterized=True)
    rescaled = fit_with_units(scale, geometry_parameterized=True)

    for key in GEOMETRY_KEYS + ERROR_KEYS:
        assert relative_change(reference, rescaled, key) < 1e-5, key
    for key in DECISION_KEYS:
        assert int(reference[key]) == int(rescaled[key]), key


def test_convergence_decision_is_unit_invariant_in_both_parameterisations():
    """The weighted convergence scatter fixes the stopping decision on its own.

    Before it, this same comparison gave 8 iterations against 7.
    """
    for geometry_parameterized in (False, True):
        reference = fit_with_units(1.0, geometry_parameterized)
        rescaled = fit_with_units(10.0, geometry_parameterized)
        assert int(reference["niter"]) == int(rescaled["niter"])
        assert int(reference["stop_code"]) == int(rescaled["stop_code"])


def test_amplitude_parameterisation_still_has_unit_dependent_uncertainties():
    """Pin the remaining reason to prefer the geometry parameterisation.

    Fixing convergence is not enough: the amplitude form's covariance is not
    unit-invariant, so the reported geometry errors still move by ~15%. If this
    ever stops being true the geometry parameterisation has lost one of its
    justifications and the plan should be revisited.
    """
    reference = fit_with_units(1.0, geometry_parameterized=False)
    rescaled = fit_with_units(10.0, geometry_parameterized=False)

    error_drift = max(relative_change(reference, rescaled, key) for key in ERROR_KEYS)
    assert error_drift > 0.05

    geometry_drift = max(relative_change(reference, rescaled, key) for key in GEOMETRY_KEYS)
    assert geometry_drift < 1e-3  # convergence fix already helps the geometry


def test_reported_rms_stays_the_physical_ring_dispersion():
    """The weighted scatter is for the convergence test only.

    ``rms`` is a documented output column — the ring-residual dispersion in flux
    units — so it must still scale with the band whose units changed.
    """
    reference = fit_with_units(1.0, geometry_parameterized=True)
    rescaled = fit_with_units(10.0, geometry_parameterized=True)
    assert float(rescaled["rms"]) > 2.0 * float(reference["rms"])


# ---------------------------------------------------------------------------
# The reconstructed model must be the model that was actually fitted
# ---------------------------------------------------------------------------


def _joint_fit_and_reconstruct(gradients, angles):
    """Fit a planted geometry step, then rebuild the model both ways."""
    from isoster.multiband.fitting_mb import (
        evaluate_joint_model,
        fit_first_and_second_harmonics_joint,
        joint_gradient_pooling_weights,
    )

    delta = 0.100
    gradients = np.asarray(gradients, dtype=np.float64)
    intens = np.vstack([100.0 + delta * g * np.sin(angles) for g in gradients])
    coeffs, _cov, _wls = fit_first_and_second_harmonics_joint(
        angles, intens, np.ones(gradients.size), None, per_band_gradients=gradients
    )
    pooling = joint_gradient_pooling_weights(np.ones(gradients.size), None, gradients.size)
    pooled_gradient = float(np.sum(pooling * gradients) / np.sum(pooling))
    equivalent = coeffs.copy()
    equivalent[gradients.size :] = coeffs[gradients.size :] * pooled_gradient
    band_scale = gradients / pooled_gradient
    corrected = evaluate_joint_model(angles, equivalent, gradients.size, band_scale=band_scale)
    uncorrected = evaluate_joint_model(angles, equivalent, gradients.size)
    return delta, float(coeffs[gradients.size]), intens, corrected, uncorrected


@pytest.mark.parametrize("coverage", [1.0, 0.6])
def test_geometry_solve_model_reconstruction_is_exact(coverage):
    """Zero residual against the model the solve actually fitted.

    Under the geometry parameterisation band ``b``'s fitted amplitude is
    ``delta * grad_b``, so reconstructing with one pooled amplitude for every
    band describes a model that was never fitted. That false residual reached
    ``rms`` and ``rms_<b>``, the convergence scatter, the OLS residual-variance
    scaling and therefore the formal errors, and the shared higher-harmonic
    subtraction — where on an incomplete ring it leaked into non-zero third and
    fourth order coefficients with no higher-order signal planted.
    """
    n = 256
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)[: int(n * coverage)]
    delta, recovered, intens, corrected, uncorrected = _joint_fit_and_reconstruct([-1.0, -10.0], angles)

    assert recovered == pytest.approx(delta, rel=1e-9)
    assert np.abs(intens - corrected).max() < 1e-10
    # And the previous reconstruction was badly wrong, so this cannot pass by
    # the two happening to coincide.
    assert np.abs(intens - uncorrected).max() > 0.4


def test_equal_gradients_make_both_reconstructions_agree():
    """The defect needs unequal gradients; with equal ones nothing moves."""
    angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    _delta, _rec, _intens, corrected, uncorrected = _joint_fit_and_reconstruct([-4.0, -4.0], angles)
    np.testing.assert_allclose(corrected, uncorrected, rtol=1e-12)
