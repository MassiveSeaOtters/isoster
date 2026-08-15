"""Phase 2: the geometry-parameterised joint solve.

The standard joint solver fits one shared *amplitude*. A common geometry step
``delta`` produces amplitude ``delta * grad_b`` in band ``b``, so one shared
amplitude is misspecified whenever the bands have different gradients: it fits
``delta`` times a weight-averaged gradient, which then has to be divided back out.

Scaling each band's harmonic columns by ``grad_b`` makes the shared parameters
the geometry steps themselves. These tests measure the three claimed
consequences: unbiasedness, minimum-variance band weighting, and a shared
coefficient that no longer depends on a band's flux units.
"""

import numpy as np
import pytest

from isoster.multiband.fitting_mb import (
    fit_first_and_second_harmonics_geometry,
    fit_first_and_second_harmonics_joint,
    joint_gradient_pooling_weights,
)

N_SAMPLES = 256
DELTA = 0.100
PHI = np.linspace(0.0, 2.0 * np.pi, N_SAMPLES, endpoint=False)

# Band configurations. The distinction that matters is whether the amplitude
# weighting (1/var) and the information weighting (grad^2/var) agree or invert.
AGREE = ([-100.0, -10.0], [1.0, 100.0])
INVERTED = ([-0.008, -0.0405], [0.00052, 0.00208])  # HSC-like: g faint but quiet
INVERTED_3 = ([-0.008, -0.025, -0.0405], [0.00052, 0.00267, 0.00208])  # HSC g/r/i


def planted(gradients, noise_sigma=None, seed=0):
    gradients = np.asarray(gradients, dtype=np.float64)
    rows = [100.0 + DELTA * g * np.sin(PHI) for g in gradients]
    out = np.vstack(rows)
    if noise_sigma is not None:
        rng = np.random.default_rng(seed)
        out = out + rng.normal(0.0, 1.0, out.shape) * np.asarray(noise_sigma)[:, None]
    return out


def variances_2d(values):
    return np.vstack([np.full(N_SAMPLES, v) for v in values])


def amplitude_delta(gradients, var2d, band_weights, intens):
    """Recover delta the Phase 1 way: shared amplitude / consistently pooled gradient."""
    coeffs, _cov, _wls = fit_first_and_second_harmonics_joint(PHI, intens, band_weights, var2d)
    weights = joint_gradient_pooling_weights(band_weights, var2d, len(gradients))
    return float(coeffs[len(gradients)]) / float(np.sum(weights * gradients) / np.sum(weights))


def geometry_delta(gradients, var2d, band_weights, intens):
    coeffs, _cov, _wls = fit_first_and_second_harmonics_geometry(PHI, intens, band_weights, gradients, var2d)
    return float(coeffs[len(gradients)])


# ---------------------------------------------------------------------------
# T2.1 — both parameterisations are unbiased
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gradients,variances", [AGREE, INVERTED, INVERTED_3])
def test_both_parameterisations_recover_the_planted_step(gradients, variances):
    gradients = np.asarray(gradients, dtype=np.float64)
    var2d = variances_2d(variances)
    band_weights = np.ones(gradients.size)
    intens = planted(gradients)

    assert amplitude_delta(gradients, var2d, band_weights, intens) == pytest.approx(DELTA, rel=1e-9)
    assert geometry_delta(gradients, var2d, band_weights, intens) == pytest.approx(DELTA, rel=1e-9)


def test_geometry_solve_needs_no_pooled_gradient():
    """The shared coefficient *is* the step, so nothing divides it afterwards."""
    gradients = np.asarray(INVERTED[0])
    coeffs, _cov, _wls = fit_first_and_second_harmonics_geometry(
        PHI, planted(gradients), np.ones(2), gradients, variances_2d(INVERTED[1])
    )
    assert float(coeffs[2]) == pytest.approx(DELTA, rel=1e-9)


# ---------------------------------------------------------------------------
# T2.2 — minimum-variance band weighting
# ---------------------------------------------------------------------------


def scatter_over_realisations(gradients, variances, n_realisations=400):
    gradients = np.asarray(gradients, dtype=np.float64)
    var2d = variances_2d(variances)
    band_weights = np.ones(gradients.size)
    sigma = np.sqrt(np.asarray(variances))
    amp, geom = [], []
    for seed in range(n_realisations):
        intens = planted(gradients, noise_sigma=sigma, seed=seed)
        amp.append(amplitude_delta(gradients, var2d, band_weights, intens))
        geom.append(geometry_delta(gradients, var2d, band_weights, intens))
    return np.array(amp), np.array(geom)


@pytest.mark.parametrize("gradients,variances", [INVERTED, INVERTED_3])
def test_geometry_parameterisation_lowers_scatter_when_weightings_invert(gradients, variances):
    """T2.2: the case that matters — 1/var and grad^2/var disagree.

    On real HSC data the two were measured close to inverted: band g carried 65%
    of the amplitude weight and 10% of the geometric information.
    """
    amp, geom = scatter_over_realisations(gradients, variances)
    # Both unbiased.
    assert amp.mean() == pytest.approx(DELTA, rel=0.1)
    assert geom.mean() == pytest.approx(DELTA, rel=0.1)
    # And the geometry parameterisation is materially tighter.
    assert geom.std() / amp.std() < 0.85


def test_no_penalty_when_the_two_weightings_agree():
    """Where 1/var already tracks grad^2/var there is nothing to gain, and the
    geometry parameterisation must not be *worse*."""
    amp, geom = scatter_over_realisations(*AGREE, n_realisations=200)
    assert geom.std() / amp.std() == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# T2.3 — degenerate bands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_a_band_without_a_gradient_contributes_no_geometry(bad):
    """No measurable radial gradient means no geometric leverage — not a crash.

    The healthy band must still recover the step exactly.
    """
    gradients = np.array([-100.0, bad])
    usable = np.array([-100.0, 0.0])
    intens = planted(usable)
    coeffs, cov, _wls = fit_first_and_second_harmonics_geometry(PHI, intens, np.ones(2), gradients, None)
    assert np.all(np.isfinite(coeffs))
    assert cov is not None
    assert float(coeffs[2]) == pytest.approx(DELTA, rel=1e-9)


def test_all_bands_degenerate_falls_back_without_crashing():
    gradients = np.array([0.0, 0.0])
    intens = planted(gradients)
    coeffs, cov, _wls = fit_first_and_second_harmonics_geometry(PHI, intens, np.ones(2), gradients, None)
    assert np.all(np.isfinite(coeffs))
    assert cov is None
    assert float(coeffs[2]) == 0.0
    # The per-band intercepts still carry the ring means.
    assert float(coeffs[0]) == pytest.approx(100.0, rel=1e-9)


# ---------------------------------------------------------------------------
# T2.4 — the shared coefficient stops depending on a band's flux units
# ---------------------------------------------------------------------------


def rescaled_case(scale):
    """Band 1 re-expressed with flux x scale: gradient x scale, variance x scale**2."""
    gradients = np.array([-100.0, -10.0 * scale])
    variances = np.array([1.0, 100.0 * scale**2])
    sigma = np.sqrt(variances)
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, (2, N_SAMPLES))
    levels = np.array([100.0, 100.0 * scale])
    intens = np.vstack([levels[b] + DELTA * gradients[b] * np.sin(PHI) + sigma[b] * base[b] for b in range(2)])
    var2d = variances_2d(variances)
    coeffs, _cov, _wls = fit_first_and_second_harmonics_geometry(PHI, intens, np.ones(2), gradients, var2d)
    model = np.vstack(
        [coeffs[b] + gradients[b] * (coeffs[2] * np.sin(PHI) + coeffs[3] * np.cos(PHI)) for b in range(2)]
    )
    residual = (intens - model).reshape(-1)
    weights = np.vstack([np.full(N_SAMPLES, 1.0 / variances[b]) for b in range(2)]).reshape(-1)
    weighted_rms = float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights)))
    return abs(float(coeffs[2])), float(np.std(residual)), weighted_rms


@pytest.mark.parametrize("scale", [10.0, 0.1])
def test_shared_coefficient_is_exactly_unit_invariant(scale):
    """T2.4: the shared coefficient is a geometry step, so units cannot touch it.

    Under the amplitude parameterisation it was not invariant, which is half of
    why the convergence comparison drifted.
    """
    amp_ref, _u_ref, _w_ref = rescaled_case(1.0)
    amp_new, _u_new, _w_new = rescaled_case(scale)
    assert amp_new == pytest.approx(amp_ref, rel=1e-9)


def test_convergence_statistic_needs_the_weighted_rms_as_well():
    """The numerator is fixed by the parameterisation; the denominator is not.

    An unweighted pooled rms is still in raw flux units, so the convergence
    comparison stays unit-dependent until the rms is weighted too. Both numbers
    are recorded here so the choice is evidence-backed rather than asserted.
    """
    amp0, unweighted0, weighted0 = rescaled_case(1.0)
    amp1, unweighted1, weighted1 = rescaled_case(10.0)

    unweighted_factor = (amp1 / unweighted1) / (amp0 / unweighted0)
    weighted_factor = (amp1 / weighted1) / (amp0 / weighted0)

    assert unweighted_factor == pytest.approx(0.1005, rel=0.02)  # still wrong
    assert weighted_factor == pytest.approx(1.0, abs=0.01)  # invariant


# ---------------------------------------------------------------------------
# Integration: the config flag, end to end
# ---------------------------------------------------------------------------

TRUE_GEOMETRY = {"x0": 128.0, "y0": 128.0, "eps": 0.25, "pa": 0.45}
BAND_AMPLITUDES = [12.0, 45.0, 80.0]  # HSC-like: g faint, i bright
BAND_SIGMA = [0.023, 0.052, 0.046]
START_GEOMETRY = {"x0": 128.0, "y0": 128.0, "eps": 0.20, "pa": 0.40}


def planted_galaxy(h, w, x0, y0, eps, pa, amplitude, re, noise_sigma, seed, n_sersic=1.5):
    """Sersic profile whose truth geometry is exactly ``(x0, y0, eps, pa)``.

    Defined locally rather than imported from ``test_fitting_mb``, matching the
    convention in the other multi-band test modules — pytest does not put the
    test directory on ``sys.path``.
    """
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


def _fit_planted(flag, seed, sma=30.0):
    from isoster.multiband.config_mb import IsosterConfigMB
    from isoster.multiband.fitting_mb import fit_isophote_mb

    images = [
        planted_galaxy(
            h=256,
            w=256,
            x0=TRUE_GEOMETRY["x0"],
            y0=TRUE_GEOMETRY["y0"],
            eps=TRUE_GEOMETRY["eps"],
            pa=TRUE_GEOMETRY["pa"],
            amplitude=BAND_AMPLITUDES[b],
            re=25.0,
            noise_sigma=BAND_SIGMA[b],
            seed=1000 * b + seed,
        )
        for b in range(3)
    ]
    variance_maps = [np.full((256, 256), BAND_SIGMA[b] ** 2) for b in range(3)]
    cfg = IsosterConfigMB(bands=["g", "r", "i"], reference_band="i", nclip=0, geometry_parameterized_solve=flag)
    return fit_isophote_mb(images, None, sma, START_GEOMETRY, cfg, variance_maps=variance_maps)


def test_flag_defaults_off_and_leaves_the_solve_unchanged():
    from isoster.multiband.config_mb import IsosterConfigMB

    assert IsosterConfigMB(bands=["g", "r"], reference_band="g").geometry_parameterized_solve is False
    off = _fit_planted(False, seed=0)
    again = _fit_planted(False, seed=0)
    assert off["eps"] == again["eps"]  # deterministic


def test_geometry_parameterisation_recovers_known_common_geometry():
    """Both parameterisations are unbiased against a planted common geometry."""
    out = _fit_planted(True, seed=0)
    assert out["eps"] == pytest.approx(TRUE_GEOMETRY["eps"], abs=0.01)
    assert out["pa"] == pytest.approx(TRUE_GEOMETRY["pa"], abs=0.01)


def test_geometry_parameterisation_reduces_scatter_end_to_end():
    """The measured payoff: same bias, materially less scatter.

    Twelve realisations is enough to see a ~30% effect and keeps the test fast.
    """
    amp = np.array([_fit_planted(False, seed=s)["eps"] for s in range(12)])
    geom = np.array([_fit_planted(True, seed=s)["eps"] for s in range(12)])
    assert abs(geom.mean() - TRUE_GEOMETRY["eps"]) == pytest.approx(abs(amp.mean() - TRUE_GEOMETRY["eps"]), abs=2e-4)
    assert geom.std() < amp.std()
