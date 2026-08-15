"""Phase 7: B=1 multi-band must equal single-band, across a config matrix.

Multi-band is a ~3,300-line parallel implementation of the single-band fitting
loop with its own config class. That duplication is the standing drift risk, and
three rounds of review found every real defect in a *less-used* mode. A matrix
harness is what converts "we happened to check" into coverage.

**This harness calls ``fit_isophote_mb`` directly, on purpose.**
``fit_image_multiband`` delegates to ``fit_image`` when ``len(bands) == 1``, so a
B=1 parity test written against that entry point compares single-band with
itself and would pass against any multi-band implementation whatsoever. The same
failure mode made an earlier B=1 gradient test vacuous.

Measured parity (2026-08-15): agreement is at machine precision (~5e-15) in 14
of the 16 combinations below. The exception is documented and bounded rather
than hidden — see ``MEDIAN_OLS_UNFLOORED``.
"""

import itertools

import numpy as np
import pytest

from isoster.config import IsosterConfig
from isoster.fitting import fit_isophote
from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import fit_isophote_mb

TRUE = {"x0": 128.0, "y0": 128.0, "eps": 0.25, "pa": 0.45}
START = {"x0": 128.0, "y0": 128.0, "eps": 0.20, "pa": 0.40}
SMA = 30.0
NOISE_SIGMA = 0.4

# Geometry, photometry, uncertainties and the control flow that produced them.
COMPARED_KEYS = (
    "x0",
    "y0",
    "eps",
    "pa",
    "rms",
    "x0_err",
    "y0_err",
    "eps_err",
    "pa_err",
    "stop_code",
    "niter",
)

MACHINE_TOL = 1e-12
# The one corner where the two paths differ structurally. Multi-band cannot
# report a median from a least-squares intercept column, so `integrator='median'`
# forces the decoupled intercept mode: it subtracts the per-band median and fits
# a 4-column geometric system, where single-band fits its usual 5-parameter
# system and reports the median separately. The OLS residual variances therefore
# come from slightly different models, which moves the geometry *errors* (not
# the geometry) by <1%. Setting `sigma_bg` makes the floor dominate and the two
# agree exactly again, which is what identifies the residual variance as the
# cause.
MEDIAN_OLS_UNFLOORED = 1e-2


def planted_galaxy(h=256, w=256, amplitude=60.0, re=25.0, seed=5, n_sersic=1.5):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = x - TRUE["x0"], y - TRUE["y0"]
    cos_pa, sin_pa = np.cos(TRUE["pa"]), np.sin(TRUE["pa"])
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - TRUE["eps"])) ** 2)
    bn = 2.0 * n_sersic - 0.327
    img = amplitude * np.exp(-bn * ((r / re) ** (1.0 / n_sersic) - 1.0))
    return img + rng.normal(0.0, NOISE_SIGMA, size=img.shape)


IMAGE = planted_galaxy()
VARIANCE = np.full((256, 256), NOISE_SIGMA**2)

MATRIX = list(
    itertools.product(
        ("mean", "median"),  # integrator
        (False, True),  # weighted least squares
        (False, True),  # eccentric-anomaly sampling
        (None, 0.5),  # sigma_bg
    )
)


def run_pair(integrator, wls, eccentric_anomaly, sigma_bg):
    """Fit the same ring through both code paths with matched configuration."""
    # A median cannot come out of a least-squares intercept column, so the
    # multi-band validator requires the decoupled intercept mode alongside it.
    single = IsosterConfig(
        integrator=integrator,
        use_eccentric_anomaly=eccentric_anomaly,
        sigma_bg=sigma_bg,
        nclip=0,
    )
    multi = IsosterConfigMB(
        bands=["g"],
        reference_band="g",
        integrator=integrator,
        use_eccentric_anomaly=eccentric_anomaly,
        sigma_bg=sigma_bg,
        nclip=0,
        fit_per_band_intens_jointly=(integrator != "median"),
    )
    variance_map = VARIANCE if wls else None
    sb = fit_isophote(IMAGE, None, SMA, START, single, variance_map=variance_map)
    mb = fit_isophote_mb([IMAGE], None, SMA, START, multi, variance_maps=[variance_map] if wls else None)
    return sb, mb


def tolerance_for(integrator, wls, sigma_bg):
    if integrator == "median" and not wls and sigma_bg is None:
        return MEDIAN_OLS_UNFLOORED
    return MACHINE_TOL


@pytest.mark.parametrize("integrator,wls,eccentric_anomaly,sigma_bg", MATRIX)
def test_b1_multiband_matches_single_band(integrator, wls, eccentric_anomaly, sigma_bg):
    sb, mb = run_pair(integrator, wls, eccentric_anomaly, sigma_bg)
    tol = tolerance_for(integrator, wls, sigma_bg)

    for key in COMPARED_KEYS:
        a, b = float(sb[key]), float(mb[key])
        if not (np.isfinite(a) and np.isfinite(b)):
            assert np.isfinite(a) == np.isfinite(b), key
            continue
        relative = abs(b - a) / abs(a) if a else abs(b - a)
        assert relative < tol, f"{key}: single={a!r} multi={b!r} rel={relative:.3e}"


@pytest.mark.parametrize("integrator,wls,eccentric_anomaly,sigma_bg", MATRIX)
def test_b1_reported_intensity_matches_single_band(integrator, wls, eccentric_anomaly, sigma_bg):
    """``intens_g`` is the multi-band spelling of single-band ``intens``."""
    sb, mb = run_pair(integrator, wls, eccentric_anomaly, sigma_bg)
    assert float(mb["intens_g"]) == pytest.approx(float(sb["intens"]), rel=tolerance_for(integrator, wls, sigma_bg))


def test_the_mean_integrator_agrees_to_machine_precision_everywhere():
    """The default integrator has no documented exception; keep it that way."""
    worst = 0.0
    for wls, eccentric_anomaly, sigma_bg in itertools.product((False, True), (False, True), (None, 0.5)):
        sb, mb = run_pair("mean", wls, eccentric_anomaly, sigma_bg)
        for key in COMPARED_KEYS:
            a, b = float(sb[key]), float(mb[key])
            if a and np.isfinite(a) and np.isfinite(b):
                worst = max(worst, abs(b - a) / abs(a))
    assert worst < MACHINE_TOL, f"worst relative disagreement {worst:.3e}"


def test_the_documented_exception_is_real_and_bounded():
    """Pin the median/OLS/unfloored corner from both sides.

    If it silently becomes exact, the tolerance above is dead weight and the
    comment explaining it is wrong; if it grows, that is a regression. Either
    way this test should be the thing that says so.
    """
    sb, mb = run_pair("median", wls=False, eccentric_anomaly=False, sigma_bg=None)
    disagreement = max(
        abs(float(mb[k]) - float(sb[k])) / abs(float(sb[k]))
        for k in ("x0_err", "y0_err", "eps_err", "pa_err")
        if float(sb[k])
    )
    assert MACHINE_TOL < disagreement < MEDIAN_OLS_UNFLOORED

    # Setting sigma_bg makes the floor dominate the residual variance, and the
    # two paths agree exactly again — which is what identifies the cause.
    sb_floored, mb_floored = run_pair("median", wls=False, eccentric_anomaly=False, sigma_bg=0.5)
    for key in ("x0_err", "y0_err", "eps_err", "pa_err"):
        assert float(mb_floored[key]) == pytest.approx(float(sb_floored[key]), rel=MACHINE_TOL)


def test_harness_does_not_go_through_the_delegating_entry_point():
    """Guard the guard: ``fit_image_multiband`` would make this vacuous.

    At B=1 it delegates to single-band, so a parity test written against it
    compares single-band with itself.
    """
    import inspect

    from isoster.multiband import fit_image_multiband

    source = inspect.getsource(fit_image_multiband)
    assert "_delegate_single_band" in source
    # ...which is precisely why this module imports fit_isophote_mb instead.
    assert fit_isophote_mb.__module__ == "isoster.multiband.fitting_mb"
