"""Tests for residual-variance scaling of the OLS harmonic covariance.

Under ordinary least squares the inverse normal-equation matrix returned by the
harmonic solvers is only a shape: it must be multiplied by the residual variance
of the fit to become a covariance. Two things therefore have to match the fit
exactly:

- the angle array the model is evaluated at must be the one the coefficients
  were fitted in (the eccentric anomaly psi when ``use_eccentric_anomaly=True``,
  not the polar angle phi), and
- the model must be the complete fitted model, including any higher-order terms
  that ``isofit_mode='in_loop'`` fits simultaneously.

Weighted least squares is unaffected: its covariance is exact and the residual
rescaling is skipped entirely.

See docs/agent/journal/2026-08-12_ea-ols-review.md.
"""

import numpy as np
import pytest

from isoster.config import IsosterConfig
from isoster.fitting import (
    compute_parameter_errors,
    evaluate_harmonic_model,
    fit_all_harmonics,
    fit_first_and_second_harmonics,
    fit_isophote,
    harmonic_function,
)
from isoster.numba_kernels import compute_ellipse_coords
from isoster.sampling import extract_isophote_data

SHAPE = (241, 241)
CENTER = 120.0
SMA = 40.0


def make_sersic_image(
    shape=SHAPE,
    x0=CENTER,
    y0=CENTER,
    eps=0.5,
    pa=0.6,
    re=35.0,
    n=3.0,
    ie=1000.0,
    a4=0.0,
    noise_sigma=0.0,
    seed=1,
):
    """Noise-free-by-default Sersic galaxy with an optional fourth-order term.

    ``a4`` perturbs the elliptical radius as ``r -> r * (1 + a4*cos(4*theta))``
    where theta is the eccentric anomaly, which is how a disky or boxy isophote
    is usually parameterised.
    """
    from scipy.special import gammaincinv

    bn = gammaincinv(2 * n, 0.5)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    dx, dy = xx - x0, yy - y0
    x_rot = dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    theta = np.arctan2(y_rot / (1.0 - eps), x_rot)
    r = r * (1.0 + a4 * np.cos(4.0 * theta))
    image = ie * np.exp(-bn * ((r / re) ** (1.0 / n) - 1.0))
    if noise_sigma > 0:
        image = image + np.random.default_rng(seed).normal(0.0, noise_sigma, shape)
    return image


def add_ring_harmonic(image, amplitude, order, coordinate, x0=CENTER, y0=CENTER, eps=0.5, pa=0.6):
    """Multiply the image by ``1 + amplitude*cos(order*angle)``.

    Along any ring of the given geometry this is exactly a constant plus one
    harmonic term, so a fit that includes ``order`` can represent it perfectly.
    ``coordinate`` selects the angle: ``"psi"`` (eccentric anomaly) or ``"phi"``
    (polar). It has to match the sampling mode under test, because a pure
    harmonic in one coordinate is spread over many harmonics in the other.

    Note this is *not* the same as perturbing the elliptical radius. A radial
    perturbation ``r -> r*(1 + a4*cos(4*theta))`` produces eighth- and
    twelfth-order intensity terms as well, which orders [3, 4] genuinely cannot
    fit, so its residual is real rather than an artefact of the rescaling.
    """
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    dx, dy = xx - x0, yy - y0
    x_rot = dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
    if coordinate == "psi":
        angle = np.arctan2(y_rot / (1.0 - eps), x_rot)
    else:
        angle = np.arctan2(y_rot, x_rot)
    return image * (1.0 + amplitude * np.cos(order * angle))


def frozen_config(**overrides):
    """Config with every geometry parameter held fixed.

    Freezing the geometry makes the sampled ring identical on every iteration,
    so a test can re-extract the same data and reproduce the expected errors
    exactly. It also keeps the first- and second-order amplitudes large instead
    of letting the geometry iteration drive them towards zero, which is the
    regime where the psi/phi coordinate mismatch is measurable.
    """
    params = {
        "compute_errors": True,
        "fix_center": True,
        "fix_pa": True,
        "fix_eps": True,
        "nclip": 0,
        "use_lazy_gradient": False,
        "use_corrected_errors": False,
        "debug": True,
    }
    params.update(overrides)
    return IsosterConfig(**params)


def expected_errors(image, geom, cfg, gradient, orders=None):
    """Recompute the geometry errors the way the fit itself should.

    Re-extracts the ring at the frozen geometry, refits the harmonics through
    the same public solvers the fit uses, and scales the covariance by the
    residual variance of the complete fitted model in the fitted angle basis.
    """
    data = extract_isophote_data(
        image,
        None,
        geom["x0"],
        geom["y0"],
        SMA,
        geom["eps"],
        geom["pa"],
        use_eccentric_anomaly=cfg.use_eccentric_anomaly,
    )
    if orders:
        coeffs, cov = fit_all_harmonics(data.angles, data.intens, orders)
        model = evaluate_harmonic_model(data.angles, coeffs, orders)
    else:
        coeffs, cov = fit_first_and_second_harmonics(data.angles, data.intens)
        model = harmonic_function(data.angles, coeffs)

    var_residual = float(np.var(data.intens - model, ddof=len(coeffs)))
    covariance = cov[:5, :5] * var_residual

    return compute_parameter_errors(
        data.angles,
        data.intens,
        geom["x0"],
        geom["y0"],
        SMA,
        geom["eps"],
        geom["pa"],
        gradient,
        cov_matrix=covariance,
        coeffs=coeffs[:5],
        use_exact_covariance=True,  # already scaled above
    )


# ---------------------------------------------------------------------------
# Bug 1: covariance fitted in psi must be scaled by residuals in psi
# ---------------------------------------------------------------------------


def test_ea_five_parameter_errors_scale_with_psi_residuals():
    """EA-mode geometry errors must use the psi residuals, not the phi ones.

    A deliberately wrong, frozen centre keeps the first-order amplitudes large,
    which is the regime where the eccentric anomaly and the polar angle give
    materially different residual variances.
    """
    image = make_sersic_image(eps=0.7, a4=0.0)
    geom = {"x0": CENTER + 4.0, "y0": CENTER + 4.0, "eps": 0.7, "pa": 0.6}
    cfg = frozen_config(use_eccentric_anomaly=True)

    result = fit_isophote(image, None, SMA, geom, cfg)
    x0_err, y0_err, eps_err, pa_err = expected_errors(image, geom, cfg, result["grad"])

    assert result["x0_err"] == pytest.approx(x0_err, rel=1e-9)
    assert result["y0_err"] == pytest.approx(y0_err, rel=1e-9)
    assert result["eps_err"] == pytest.approx(eps_err, rel=1e-9)
    assert result["pa_err"] == pytest.approx(pa_err, rel=1e-9)


def test_ea_five_parameter_errors_differ_from_phi_scaling():
    """Guard the old behaviour explicitly, so a regression cannot pass silently.

    Without this the previous test could be satisfied by a change that happened
    to make psi and phi scaling agree.
    """
    image = make_sersic_image(eps=0.7, a4=0.0)
    geom = {"x0": CENTER + 4.0, "y0": CENTER + 4.0, "eps": 0.7, "pa": 0.6}
    cfg = frozen_config(use_eccentric_anomaly=True)

    result = fit_isophote(image, None, SMA, geom, cfg)
    data = extract_isophote_data(
        image, None, geom["x0"], geom["y0"], SMA, geom["eps"], geom["pa"], use_eccentric_anomaly=True
    )
    coeffs, cov = fit_first_and_second_harmonics(data.angles, data.intens)

    var_psi = np.var(data.intens - harmonic_function(data.angles, coeffs), ddof=5)
    var_phi = np.var(data.intens - harmonic_function(data.phi, coeffs), ddof=5)

    # The scenario has to actually separate the two, or the test proves nothing.
    assert var_phi > 1.5 * var_psi

    stale = compute_parameter_errors(
        data.phi,
        data.intens,
        geom["x0"],
        geom["y0"],
        SMA,
        geom["eps"],
        geom["pa"],
        result["grad"],
        cov_matrix=cov,
        coeffs=coeffs,
    )
    assert result["x0_err"] != pytest.approx(stale[0], rel=1e-6)


# ---------------------------------------------------------------------------
# Bug 2: in_loop must rescale with the complete fitted model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_ea", [True, False])
def test_in_loop_errors_scale_with_full_model(use_ea):
    """The nine-parameter fit must be rescaled by its own residuals.

    Runs in both angle bases: the truncated-model defect is independent of the
    eccentric anomaly and shows up in ordinary polar sampling too.
    """
    image = make_sersic_image(a4=0.06)
    geom = {"x0": CENTER, "y0": CENTER, "eps": 0.5, "pa": 0.6}
    cfg = frozen_config(
        use_eccentric_anomaly=use_ea,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
        harmonic_orders=[3, 4],
    )

    result = fit_isophote(image, None, SMA, geom, cfg)
    x0_err, y0_err, eps_err, pa_err = expected_errors(image, geom, cfg, result["grad"], orders=[3, 4])

    assert result["x0_err"] == pytest.approx(x0_err, rel=1e-9)
    assert result["y0_err"] == pytest.approx(y0_err, rel=1e-9)
    assert result["eps_err"] == pytest.approx(eps_err, rel=1e-9)
    assert result["pa_err"] == pytest.approx(pa_err, rel=1e-9)


@pytest.mark.parametrize("use_ea", [True, False])
def test_in_loop_errors_not_inflated_by_fitted_fourth_order_signal(use_ea):
    """A fitted fourth-order term is signal, not noise.

    The planted term is a pure fourth-order harmonic in the coordinate being
    fitted, so on noise-free data the nine-parameter model absorbs it entirely
    and the reported geometry errors should stay close to those of a smooth
    galaxy. Rescaling by the truncated five-term model instead counts the whole
    fourth-order amplitude as noise.
    """
    geom = {"x0": CENTER, "y0": CENTER, "eps": 0.5, "pa": 0.6}
    cfg = frozen_config(
        use_eccentric_anomaly=use_ea,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
        harmonic_orders=[3, 4],
    )

    base = make_sersic_image()
    perturbed_image = add_ring_harmonic(base, 0.10, 4, "psi" if use_ea else "phi")

    smooth = fit_isophote(base, None, SMA, geom, cfg)
    perturbed = fit_isophote(perturbed_image, None, SMA, geom, cfg)

    # The fourth-order term must actually be present and fitted, or the test
    # would pass trivially on data that has no higher-order signal at all.
    assert abs(perturbed["b4"]) > 0.04

    assert perturbed["eps_err"] < 2.0 * smooth["eps_err"]
    assert perturbed["x0_err"] < 2.0 * smooth["x0_err"]


def test_in_loop_geometry_and_harmonic_errors_share_residual_variance(use_ea=True):
    """Both error families come from one fit, so one residual variance scales both.

    The higher-order harmonic errors were already computed correctly; this pins
    the geometry errors to the same scale factor.
    """
    image = make_sersic_image(a4=0.06, noise_sigma=1.0, seed=11)
    geom = {"x0": CENTER, "y0": CENTER, "eps": 0.5, "pa": 0.6}
    cfg = frozen_config(
        use_eccentric_anomaly=use_ea,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
        harmonic_orders=[3, 4],
    )

    result = fit_isophote(image, None, SMA, geom, cfg)
    data = extract_isophote_data(
        image, None, geom["x0"], geom["y0"], SMA, geom["eps"], geom["pa"], use_eccentric_anomaly=use_ea
    )
    coeffs, cov = fit_all_harmonics(data.angles, data.intens, [3, 4])
    model = evaluate_harmonic_model(data.angles, coeffs, [3, 4])
    var_residual = np.var(data.intens - model, ddof=len(coeffs))

    # b4_err is stored as sqrt(cov[8,8] * var_residual) / (sma * |grad|).
    factor = SMA * abs(result["grad"])
    expected_b4_err = np.sqrt(cov[8, 8] * var_residual) / factor
    assert result["b4_err"] == pytest.approx(expected_b4_err, rel=1e-9)

    # The geometry block must be scaled by that same residual variance.
    expected = expected_errors(image, geom, cfg, result["grad"], orders=[3, 4])
    assert result["eps_err"] == pytest.approx(expected[2], rel=1e-9)


# ---------------------------------------------------------------------------
# Paths that must not change
# ---------------------------------------------------------------------------


def test_regular_polar_five_parameter_errors_unchanged():
    """Ordinary polar angle with the five-parameter model was already correct.

    Here ``angles`` and ``phi`` are the same array and the fitted model is the
    five-term one, so neither defect applies and results must be untouched.
    """
    image = make_sersic_image(noise_sigma=1.0, seed=7)
    geom = {"x0": CENTER, "y0": CENTER, "eps": 0.5, "pa": 0.6}
    cfg = frozen_config(use_eccentric_anomaly=False)

    result = fit_isophote(image, None, SMA, geom, cfg)
    expected = expected_errors(image, geom, cfg, result["grad"])

    assert result["x0_err"] == pytest.approx(expected[0], rel=1e-9)
    assert result["y0_err"] == pytest.approx(expected[1], rel=1e-9)
    assert result["eps_err"] == pytest.approx(expected[2], rel=1e-9)
    assert result["pa_err"] == pytest.approx(expected[3], rel=1e-9)


@pytest.mark.parametrize("in_loop", [False, True])
def test_wls_errors_use_exact_covariance_without_rescaling(in_loop):
    """With a variance map the covariance is exact and must not be rescaled."""
    image = make_sersic_image(a4=0.06, noise_sigma=1.0, seed=13)
    variance_map = np.full(image.shape, 1.0)
    geom = {"x0": CENTER, "y0": CENTER, "eps": 0.5, "pa": 0.6}
    extra = {"simultaneous_harmonics": True, "isofit_mode": "in_loop", "harmonic_orders": [3, 4]} if in_loop else {}
    cfg = frozen_config(use_eccentric_anomaly=True, **extra)

    result = fit_isophote(image, None, SMA, geom, cfg, variance_map=variance_map)

    data = extract_isophote_data(
        image,
        None,
        geom["x0"],
        geom["y0"],
        SMA,
        geom["eps"],
        geom["pa"],
        use_eccentric_anomaly=True,
        variance_map=variance_map,
    )
    if in_loop:
        coeffs, cov = fit_all_harmonics(data.angles, data.intens, [3, 4], variances=data.variances)
    else:
        coeffs, cov = fit_first_and_second_harmonics(data.angles, data.intens, variances=data.variances)

    expected = compute_parameter_errors(
        data.angles,
        data.intens,
        geom["x0"],
        geom["y0"],
        SMA,
        geom["eps"],
        geom["pa"],
        result["grad"],
        cov_matrix=cov[:5, :5],
        coeffs=coeffs[:5],
        use_exact_covariance=True,
    )
    assert result["x0_err"] == pytest.approx(expected[0], rel=1e-9)
    assert result["eps_err"] == pytest.approx(expected[2], rel=1e-9)


def test_compute_parameter_errors_accepts_explicit_residual_variance():
    """The caller can supply the residual variance instead of having it rebuilt."""
    angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    intens = 100.0 + 2.0 * np.sin(angles) + 1.0 * np.cos(2.0 * angles)
    coeffs, cov = fit_first_and_second_harmonics(angles, intens)

    supplied = 4.0
    errors = compute_parameter_errors(
        angles,
        intens,
        x0=CENTER,
        y0=CENTER,
        sma=SMA,
        eps=0.5,
        pa=0.6,
        gradient=-1.0,
        cov_matrix=cov,
        coeffs=coeffs,
        residual_variance=supplied,
    )
    reference = compute_parameter_errors(
        angles,
        intens,
        x0=CENTER,
        y0=CENTER,
        sma=SMA,
        eps=0.5,
        pa=0.6,
        gradient=-1.0,
        cov_matrix=cov * supplied,
        coeffs=coeffs,
        use_exact_covariance=True,
    )
    assert errors == pytest.approx(reference, rel=1e-12)


def test_residual_variance_floor_still_applies_to_supplied_value():
    """``sigma_bg`` sets a lower bound on the residual variance however it arrives."""
    angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    intens = 100.0 + 2.0 * np.sin(angles) + 1.0 * np.cos(2.0 * angles)
    coeffs, cov = fit_first_and_second_harmonics(angles, intens)

    floored = compute_parameter_errors(
        angles,
        intens,
        CENTER,
        CENTER,
        SMA,
        0.5,
        0.6,
        -1.0,
        cov_matrix=cov,
        coeffs=coeffs,
        residual_variance=1e-8,
        var_residual_floor=9.0,
    )
    reference = compute_parameter_errors(
        angles,
        intens,
        CENTER,
        CENTER,
        SMA,
        0.5,
        0.6,
        -1.0,
        cov_matrix=cov * 9.0,
        coeffs=coeffs,
        use_exact_covariance=True,
    )
    assert floored == pytest.approx(reference, rel=1e-12)


# ---------------------------------------------------------------------------
# Degenerate fits: no residual degrees of freedom
# ---------------------------------------------------------------------------


def mask_ring_to_exactly(image, n_keep, sma, eps, pa, x0=CENTER, y0=CENTER):
    """Mask everything except enough pixels for exactly ``n_keep`` ring samples.

    The mask is sampled with nearest-neighbour interpolation, so one unmasked
    pixel can revive several adjacent samples. Pixels are therefore added one at
    a time and any that would overshoot the target are put back, which makes the
    surviving count exact rather than approximate.
    """
    n_samples = max(64, int(2 * np.pi * sma))
    x, y, _, _ = compute_ellipse_coords(n_samples, sma, eps, pa, x0, y0, True)
    rows = np.round(y).astype(int)
    cols = np.round(x).astype(int)
    mask = np.ones(image.shape, dtype=bool)

    for i in range(n_samples):
        if not mask[rows[i], cols[i]]:
            continue
        mask[rows[i], cols[i]] = False
        surviving = len(
            extract_isophote_data(
                image, mask.astype(np.float64), x0, y0, sma, eps, pa, use_eccentric_anomaly=True
            ).intens
        )
        if surviving > n_keep:
            mask[rows[i], cols[i]] = True
            continue
        if surviving == n_keep:
            return mask
    raise RuntimeError(f"could not construct a ring with exactly {n_keep} samples")


def test_exactly_determined_in_loop_fit_reports_zero_geometry_errors():
    """With as many samples as parameters there is no information about the noise.

    ``isofit_min_points`` equals the parameter count exactly (1 + 2*(2 + L) is
    5 + 2*L), so the ISOFIT model switches on precisely at N == P. The fit then
    passes through every point, leaving no residual degrees of freedom. Falling
    back to a reduced five-term model here manufactures an error out of the
    higher-order signal instead of reporting that none can be measured.
    """
    sma, eps, pa = 10.0, 0.5, 0.6
    image = make_sersic_image(noise_sigma=1.0, seed=5)
    # orders=[3] gives 7 parameters, so 7 surviving samples is the boundary.
    mask = mask_ring_to_exactly(image, 7, sma, eps, pa)
    cfg = frozen_config(
        use_eccentric_anomaly=True,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
        harmonic_orders=[3],
        fflag=1.0,
    )

    result = fit_isophote(image, mask, sma, {"x0": CENTER, "y0": CENTER, "eps": eps, "pa": pa}, cfg)

    # The scenario has to be the intended one, or the test proves nothing.
    assert result["ndata"] == 7
    assert result["rms"] < 1e-6  # the seven-parameter model passes through all seven points

    assert result["x0_err"] == 0.0
    assert result["y0_err"] == 0.0
    assert result["eps_err"] == 0.0
    assert result["pa_err"] == 0.0
    # The higher-order errors already reported zero here; the geometry errors
    # must agree rather than contradict them.
    assert result["a3_err"] == 0.0
    assert result["b3_err"] == 0.0


def test_zero_residual_variance_gives_zero_errors():
    """Zero is how the caller says "no residual degrees of freedom"."""
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    intens = 100.0 + 2.0 * np.sin(angles) + np.cos(2.0 * angles)
    coeffs, cov = fit_first_and_second_harmonics(angles, intens)

    errors = compute_parameter_errors(
        angles,
        intens,
        CENTER,
        CENTER,
        SMA,
        0.5,
        0.6,
        -1.0,
        cov_matrix=cov,
        coeffs=coeffs,
        residual_variance=0.0,
    )
    assert errors == (0.0, 0.0, 0.0, 0.0)


def test_omitted_residual_variance_keeps_legacy_five_parameter_behaviour():
    """External callers that omit the keyword still get the old rebuild path.

    ``compute_parameter_errors`` is public, so ``residual_variance=None`` has to
    keep meaning "not supplied" rather than "unavailable".
    """
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    rng = np.random.default_rng(4)
    intens = 100.0 + 2.0 * np.sin(angles) + np.cos(2.0 * angles) + rng.normal(0.0, 1.0, angles.size)
    coeffs, cov = fit_first_and_second_harmonics(angles, intens)

    legacy_variance = np.var(intens - harmonic_function(angles, coeffs), ddof=5)
    omitted = compute_parameter_errors(
        angles, intens, CENTER, CENTER, SMA, 0.5, 0.6, -1.0, cov_matrix=cov, coeffs=coeffs
    )
    explicit = compute_parameter_errors(
        angles,
        intens,
        CENTER,
        CENTER,
        SMA,
        0.5,
        0.6,
        -1.0,
        cov_matrix=cov,
        coeffs=coeffs,
        residual_variance=legacy_variance,
    )
    assert omitted == pytest.approx(explicit, rel=1e-12)
    assert omitted[0] > 0.0


# ---------------------------------------------------------------------------
# One fit, one error scale: the sigma_bg floor must reach both families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_bg", [1.0, 30.0])
def test_sigma_bg_floor_applies_to_geometry_and_harmonic_errors_alike(sigma_bg):
    """Both error families are scaled by one residual variance, floor included.

    The floor is a statement about the residual variance of the fit, not about
    which parameter is being reported, so applying it to the geometry block but
    not to the higher-order block would leave one fit with two error scales.
    """
    image = make_sersic_image(noise_sigma=1.0, seed=5)
    geom = {"x0": CENTER, "y0": CENTER, "eps": 0.5, "pa": 0.6}
    base = dict(
        use_eccentric_anomaly=True,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
        harmonic_orders=[3, 4],
    )

    unfloored = fit_isophote(image, None, SMA, geom, frozen_config(**base))
    floored = fit_isophote(image, None, SMA, geom, frozen_config(sigma_bg=sigma_bg, **base))

    # The floor has to actually bite in this scenario.
    assert floored["eps_err"] > unfloored["eps_err"]

    # Both families must move by the same factor.
    geometry_ratio = floored["eps_err"] / unfloored["eps_err"]
    harmonic_ratio = floored["b4_err"] / unfloored["b4_err"]
    assert harmonic_ratio == pytest.approx(geometry_ratio, rel=1e-9)


def test_sigma_bg_absent_leaves_both_families_on_the_same_scale():
    """Without a floor the two families already share a scale; keep it that way."""
    image = make_sersic_image(noise_sigma=1.0, seed=5)
    geom = {"x0": CENTER, "y0": CENTER, "eps": 0.5, "pa": 0.6}
    cfg = frozen_config(
        use_eccentric_anomaly=True,
        simultaneous_harmonics=True,
        isofit_mode="in_loop",
        harmonic_orders=[3, 4],
    )
    result = fit_isophote(image, None, SMA, geom, cfg)

    data = extract_isophote_data(image, None, CENTER, CENTER, SMA, 0.5, 0.6, use_eccentric_anomaly=True)
    coeffs, cov = fit_all_harmonics(data.angles, data.intens, [3, 4])
    var_residual = np.var(data.intens - evaluate_harmonic_model(data.angles, coeffs, [3, 4]), ddof=len(coeffs))
    factor = SMA * abs(result["grad"])

    assert result["b4_err"] == pytest.approx(np.sqrt(cov[8, 8] * var_residual) / factor, rel=1e-9)
    assert result["eps_err"] == pytest.approx(
        expected_errors(image, geom, cfg, result["grad"], orders=[3, 4])[2], rel=1e-9
    )
