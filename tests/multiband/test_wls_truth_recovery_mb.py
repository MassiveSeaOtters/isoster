"""Phase 4: WLS truth recovery — the coverage gap that hid the weighting defect.

Two existing tests came close and each missed for a different reason:

* ``test_joint_solver_recovers_planted_coefficients`` plants *identical*
  amplitudes in all four bands, which is the one configuration where an
  inconsistently weighted gradient cancels out;
* ``test_fit_isophote_mb_planted_recovers_geometry`` uses different band
  amplitudes but runs OLS, where the solve and the pooling already agree.

What was never tested is the combination that real multi-band data always has:
**different band gradients and different band noise, under WLS**. These tests
plant a known common geometry and check it comes back, in joint and reference
mode, for both parameterisations, including a spatially-varying-variance stress
case that the scalar pooling weight is only an approximation for.

Measured tolerances (2026-08-15), so the thresholds below are evidence rather
than guesses: eps and pa recover to within 3e-4 in every configuration; the
centre to within 4.3e-3 px, worst case being the azimuthally split variance.
"""

import numpy as np
import pytest

from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import fit_isophote_mb

TRUE = {"x0": 128.0, "y0": 128.0, "eps": 0.25, "pa": 0.45}
# Different gradients (via amplitude) AND different noise, HSC-like.
BAND_AMPLITUDES = [12.0, 45.0, 80.0]
BAND_SIGMA = [0.023, 0.052, 0.046]
# Deliberately offset from truth so the fit has to travel.
START = {"x0": 127.0, "y0": 129.0, "eps": 0.20, "pa": 0.40}

EPS_PA_TOL = 1e-3
CENTRE_TOL = 1e-2


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


def images(seed_offset=0):
    return [
        planted_galaxy(
            256,
            256,
            TRUE["x0"],
            TRUE["y0"],
            TRUE["eps"],
            TRUE["pa"],
            BAND_AMPLITUDES[b],
            25.0,
            BAND_SIGMA[b],
            seed=1000 * b + 3 + seed_offset,
        )
        for b in range(3)
    ]


def variance_maps(kind):
    """Three per-band variance patterns, increasingly hostile to a scalar weight."""
    if kind == "uniform":
        return [np.full((256, 256), BAND_SIGMA[b] ** 2) for b in range(3)]
    if kind == "ramp":
        # Smoothly varying, 20x across the image — realistic for coadd depth.
        yy, xx = np.mgrid[:256, :256]
        ramp = 1.0 + 19.0 * (xx + yy) / (2 * 255.0)
        return [np.full((256, 256), BAND_SIGMA[b] ** 2) * ramp for b in range(3)]
    if kind == "azimuthal":
        # Sharp split, so variance varies *within* a ring. This is exactly the
        # case the scalar pooling weight is documented as approximate for.
        yy, _xx = np.mgrid[:256, :256]
        return [np.where(yy < 128, BAND_SIGMA[b] ** 2, BAND_SIGMA[b] ** 2 * 50.0) for b in range(3)]
    raise ValueError(kind)


def fit(kind, geometry_parameterized=False, harmonic_combination="joint", seed_offset=0):
    cfg = IsosterConfigMB(
        bands=["g", "r", "i"],
        reference_band="i",
        nclip=0,
        geometry_parameterized_solve=geometry_parameterized,
        harmonic_combination=harmonic_combination,
    )
    return fit_isophote_mb(images(seed_offset), None, 30.0, START, cfg, variance_maps=variance_maps(kind))


def assert_recovers_truth(out, eps_pa_tol=EPS_PA_TOL, centre_tol=CENTRE_TOL):
    assert int(out["stop_code"]) in (0, 2)
    assert out["eps"] == pytest.approx(TRUE["eps"], abs=eps_pa_tol)
    assert out["pa"] == pytest.approx(TRUE["pa"], abs=eps_pa_tol)
    assert out["x0"] == pytest.approx(TRUE["x0"], abs=centre_tol)
    assert out["y0"] == pytest.approx(TRUE["y0"], abs=centre_tol)


# ---------------------------------------------------------------------------
# T4.1 — joint mode, WLS, different gradients and different noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geometry_parameterized", [False, True])
@pytest.mark.parametrize("kind", ["uniform", "ramp"])
def test_joint_wls_recovers_planted_geometry(kind, geometry_parameterized):
    assert_recovers_truth(fit(kind, geometry_parameterized=geometry_parameterized))


@pytest.mark.parametrize("geometry_parameterized", [False, True])
def test_joint_wls_recovery_is_stable_across_noise_realisations(geometry_parameterized):
    """Not a single lucky seed."""
    for seed_offset in (0, 17, 41):
        assert_recovers_truth(fit("uniform", geometry_parameterized=geometry_parameterized, seed_offset=seed_offset))


# ---------------------------------------------------------------------------
# T4.2 — reference mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["uniform", "ramp", "azimuthal"])
def test_reference_mode_wls_recovers_planted_geometry(kind):
    """Ref mode drives geometry from one band; it must still find the truth.

    This is the mode whose gradient denominator was wrong in OLS *and* WLS until
    Phase 1, and no truth-recovery test covered it.
    """
    assert_recovers_truth(fit(kind, harmonic_combination="ref"))


# ---------------------------------------------------------------------------
# T4.3 — spatially varying variance: a documented stress test
# ---------------------------------------------------------------------------


def test_azimuthally_varying_variance_degrades_gracefully():
    """Variance varying *within* a ring is where the scalar weight is approximate.

    The exactly-correct per-band weight is harmonic-specific in that case (see
    ``joint_gradient_pooling_weights``), so some degradation is expected and the
    point of this test is that it stays small and bounded rather than that it is
    absent. Measured: centre error grows from ~6e-4 px to ~4.3e-3 px, with eps
    and pa essentially unaffected.
    """
    benign = fit("uniform")
    hostile = fit("azimuthal")

    assert_recovers_truth(hostile)

    centre_error = max(abs(hostile["x0"] - TRUE["x0"]), abs(hostile["y0"] - TRUE["y0"]))
    benign_error = max(abs(benign["x0"] - TRUE["x0"]), abs(benign["y0"] - TRUE["y0"]))
    assert centre_error > benign_error  # the approximation does cost something
    assert centre_error < CENTRE_TOL  # but it stays bounded


def test_geometry_parameterisation_is_more_robust_to_within_ring_variance():
    """The stress case is also where the geometry parameterisation helps most.

    Measured: centre error 4.3e-3 px (amplitude) against 1.6e-3 px (geometry).
    """
    amplitude = fit("azimuthal", geometry_parameterized=False)
    geometry = fit("azimuthal", geometry_parameterized=True)

    def centre_error(out):
        return max(abs(out["x0"] - TRUE["x0"]), abs(out["y0"] - TRUE["y0"]))

    assert centre_error(geometry) < centre_error(amplitude)


# T4.4 — the same under ``loose_validity`` is deferred to Phase 5, which decides
# whether that mode is repaired or rejected at configuration time.
