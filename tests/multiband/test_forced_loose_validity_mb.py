"""Phase 5: forced photometry honours ``loose_validity``.

Forced extraction used to sample shared-validity unconditionally, and the driver
listed ``loose_validity`` among the features it silently ignored. That mattered
because shared validity drops a sample from *every* band whenever *any* band
rejects it: measured on the HSC demo with its real per-band masks, 18-27% of
usable samples routinely, and at one radius a fully-masked g band left zero
usable samples for r and i, which each had 370.

Forced photometry runs no joint solve — each band's intensity, rms and error is
computed independently — so there is no rectangular design matrix to preserve
and per-band sampling needed no new machinery.
"""

import numpy as np
import pytest

from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.fitting_mb import extract_forced_photometry_mb

BANDS = ["g", "r", "i"]
AMPLITUDES = [20.0, 50.0, 80.0]
GEOMETRY = {"x0": 128.0, "y0": 128.0, "eps": 0.25, "pa": 0.45}
SMA = 40.0


def planted_galaxy(amplitude, seed, h=256, w=256, re=25.0, noise_sigma=0.05, n_sersic=1.5):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = x - GEOMETRY["x0"], y - GEOMETRY["y0"]
    cos_pa, sin_pa = np.cos(GEOMETRY["pa"]), np.sin(GEOMETRY["pa"])
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r = np.sqrt(x_rot**2 + (y_rot / (1.0 - GEOMETRY["eps"])) ** 2)
    bn = 2.0 * n_sersic - 0.327
    img = amplitude * np.exp(-bn * ((r / re) ** (1.0 / n_sersic) - 1.0))
    return img + rng.normal(0.0, noise_sigma, size=img.shape)


def images():
    return [planted_galaxy(AMPLITUDES[b], seed=b) for b in range(3)]


def masks_with_g_annulus(inner, outer):
    """Mask only band g, over an annulus that crosses the ring at ``SMA``."""
    masks = [np.zeros((256, 256), dtype=bool) for _ in range(3)]
    yy, xx = np.mgrid[:256, :256]
    radius = np.hypot(xx - GEOMETRY["x0"], yy - GEOMETRY["y0"])
    masks[0][(radius > inner) & (radius < outer)] = True
    return masks


def forced(masks, loose_validity, **kw):
    cfg = IsosterConfigMB(bands=BANDS, reference_band="i", nclip=0, loose_validity=loose_validity, **kw)
    return extract_forced_photometry_mb(
        images(), masks, GEOMETRY["x0"], GEOMETRY["y0"], SMA, GEOMETRY["eps"], GEOMETRY["pa"], BANDS, cfg
    )


def test_partially_masked_band_no_longer_thins_the_others():
    """Shared validity clips every band to the intersection; loose does not."""
    masks = masks_with_g_annulus(30.0, 55.0)
    shared = forced(masks, loose_validity=False)
    loose = forced(masks, loose_validity=True)

    # Shared: all three bands collapse to the samples g happened to keep.
    assert shared["n_valid_g"] == shared["n_valid_r"] == shared["n_valid_i"]
    # Loose: the clean bands keep their own, far larger, sample sets.
    assert loose["n_valid_r"] > 10 * shared["n_valid_r"]
    assert loose["n_valid_i"] > 10 * shared["n_valid_i"]
    assert loose["n_valid_g"] == shared["n_valid_g"]


def test_a_fully_masked_band_no_longer_destroys_the_isophote():
    """The case that motivated the repair.

    With band g fully masked across the ring, shared validity yields an empty
    intersection and the whole isophote fails — discarding r and i, which are
    perfectly measurable. This reproduces on the real HSC demo at sma=80.
    """
    masks = masks_with_g_annulus(20.0, 70.0)

    shared = forced(masks, loose_validity=False)
    assert int(shared["stop_code"]) == 3
    assert not np.isfinite(shared["intens_r"])
    assert not np.isfinite(shared["intens_i"])

    loose = forced(masks, loose_validity=True)
    assert int(loose["stop_code"]) == 0
    assert loose["n_valid_r"] > 200
    assert loose["n_valid_i"] > 200
    assert np.isfinite(loose["intens_r"])
    assert np.isfinite(loose["intens_i"])
    # The genuinely absent band is NaN, not 0.0 — 0.0 would read as a real
    # measurement of zero flux.
    assert loose["n_valid_g"] == 0
    assert not np.isfinite(loose["intens_g"])
    assert not np.isfinite(loose["intens_err_g"])
    assert not np.isfinite(loose["rms_g"])


def test_loose_and_shared_agree_when_no_band_is_masked():
    """With nothing masked the two modes must produce the same measurement."""
    masks = [np.zeros((256, 256), dtype=bool) for _ in range(3)]
    shared = forced(masks, loose_validity=False)
    loose = forced(masks, loose_validity=True)
    for b in BANDS:
        assert loose[f"n_valid_{b}"] == shared[f"n_valid_{b}"]
        assert loose[f"intens_{b}"] == pytest.approx(shared[f"intens_{b}"], rel=1e-12)
        assert loose[f"intens_err_{b}"] == pytest.approx(shared[f"intens_err_{b}"], rel=1e-12)


def test_absent_band_harmonics_are_nan_not_zero():
    """Dropped-band convention (M5): NaN marks absence across every column."""
    masks = masks_with_g_annulus(20.0, 70.0)
    loose = forced(masks, loose_validity=True)
    for order in (3, 4):
        assert not np.isfinite(loose[f"a{order}_g"])
        assert not np.isfinite(loose[f"b{order}_g"])
        assert np.isfinite(loose[f"a{order}_r"]) or loose[f"a{order}_r"] == 0.0


def test_variance_maps_compose_with_loose_forced_photometry():
    """WLS forced extraction keeps per-band samples too."""
    masks = masks_with_g_annulus(20.0, 70.0)
    variance_maps = [np.full((256, 256), 0.05**2) for _ in range(3)]
    cfg = IsosterConfigMB(bands=BANDS, reference_band="i", nclip=0, loose_validity=True)
    out = extract_forced_photometry_mb(
        images(),
        masks,
        GEOMETRY["x0"],
        GEOMETRY["y0"],
        SMA,
        GEOMETRY["eps"],
        GEOMETRY["pa"],
        BANDS,
        cfg,
        variance_maps=variance_maps,
    )
    assert int(out["stop_code"]) == 0
    assert out["n_valid_r"] > 200
    assert np.isfinite(out["intens_err_r"])
    assert not np.isfinite(out["intens_g"])


def test_loose_validity_requires_at_least_two_bands():
    """B=1 + loose validity is rejected at configuration time.

    Per-band validity only means something when bands can disagree, and the
    joint solve needs two surviving bands — so a single-band loose run failed
    every isophote with ``stop_code=3`` partway through. An immediate error
    naming the alternative beats a silent whole-image failure.
    """
    with pytest.raises(ValueError, match="requires at least 2 bands"):
        IsosterConfigMB(bands=["g"], reference_band="g", loose_validity=True)

    # Neither half alone is a problem.
    assert IsosterConfigMB(bands=["g"], reference_band="g").loose_validity is False
    assert IsosterConfigMB(bands=["g", "r"], reference_band="g", loose_validity=True).loose_validity is True


# ---------------------------------------------------------------------------
# The joint gradient must sample in the same validity mode as the solve
# ---------------------------------------------------------------------------


def gradient_case(masks, loose_validity):
    from isoster.multiband.fitting_mb import compute_joint_gradient
    from isoster.multiband.sampling_mb import prepare_inputs

    image_stack, masks_resolved, var_stack = prepare_inputs(images(), masks, None)
    cfg = IsosterConfigMB(bands=BANDS, reference_band="i", loose_validity=loose_validity)
    geometry = {"x0": GEOMETRY["x0"], "y0": GEOMETRY["y0"], "sma": SMA, "eps": GEOMETRY["eps"], "pa": GEOMETRY["pa"]}
    return compute_joint_gradient(
        image_stack, masks_resolved, var_stack, geometry, cfg, np.ones(3), previous_gradient=None
    )


def test_gradient_survives_a_band_fully_masked_at_this_radius():
    """The gradient used to sample shared-validity regardless of the flag.

    It is the denominator of every geometry correction *and* the source of the
    pooling weights, so sampling it on a different set from the solve is the
    same class of mismatch this work removes. In the worst case it collapsed to
    the "no usable ring" sentinel while two bands had full rings.
    """
    masks = masks_with_g_annulus(20.0, 70.0)

    shared_grad, _err, shared_per_band, _ = gradient_case(masks, loose_validity=False)
    assert shared_grad == -1.0  # the sentinel
    assert shared_per_band == []

    loose_grad, _err, loose_per_band, _ = gradient_case(masks, loose_validity=True)
    assert np.isfinite(loose_grad)
    assert loose_grad < 0.0  # a real, falling profile
    assert not np.isfinite(loose_per_band[0])  # band g contributes nothing
    assert np.isfinite(loose_per_band[1]) and np.isfinite(loose_per_band[2])


def test_gradient_modes_agree_when_nothing_is_masked():
    """Loose sampling must not perturb the gradient when it cannot matter."""
    masks = [np.zeros((256, 256), dtype=bool) for _ in range(3)]
    shared_grad, _e1, shared_per_band, _ = gradient_case(masks, loose_validity=False)
    loose_grad, _e2, loose_per_band, _ = gradient_case(masks, loose_validity=True)
    assert loose_grad == pytest.approx(shared_grad, rel=1e-12)
    for a, b in zip(shared_per_band, loose_per_band):
        assert b == pytest.approx(a, rel=1e-12)


@pytest.mark.parametrize("loose", [False, True])
@pytest.mark.parametrize("wls", [False, True])
def test_reference_mode_convergence_ignores_passive_bands(loose, wls):
    """Reference mode drives geometry from one band, so only that band's
    residuals may decide when the fit stops.

    The rectangular branch was fixed first; the jagged branch still pooled every
    band, so under loose validity a passive band's noise moved the iteration
    count (11 -> 7 in OLS) and shifted reported errors by ~3%. The reference
    band is deliberately not first in the band list here, and the surviving
    counts differ between bands.
    """
    from isoster.multiband.fitting_mb import fit_isophote_mb

    def galaxy(amplitude, seed, sigma):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:256, 0:256].astype(np.float64)
        dx, dy = x - 128.0, y - 128.0
        c, s = np.cos(0.45), np.sin(0.45)
        xr, yr = dx * c + dy * s, -dx * s + dy * c
        r = np.sqrt(xr**2 + (yr / 0.75) ** 2)
        bn = 2.0 * 1.5 - 0.327
        return amplitude * np.exp(-bn * ((r / 25.0) ** (1.0 / 1.5) - 1.0)) + rng.normal(0.0, sigma, (256, 256))

    results = []
    for passive_noise in (0.05, 5.0):
        images = [galaxy(30.0, 1, passive_noise), galaxy(50.0, 0, 0.05), galaxy(80.0, 2, 0.05)]
        variance_maps = (
            [np.full((256, 256), passive_noise**2), np.full((256, 256), 0.05**2), np.full((256, 256), 0.05**2)]
            if wls
            else None
        )
        cfg = IsosterConfigMB(
            bands=["r", "g", "i"],  # reference band is NOT first
            reference_band="g",
            harmonic_combination="ref",
            nclip=0,
            loose_validity=loose,
        )
        results.append(
            fit_isophote_mb(
                images,
                None,
                30.0,
                {"x0": 128.0, "y0": 128.0, "eps": 0.20, "pa": 0.40},
                cfg,
                variance_maps=variance_maps,
            )
        )

    quiet, noisy = results
    assert int(quiet["niter"]) == int(noisy["niter"])
    for key in ("eps", "pa", "eps_err", "pa_err"):
        assert float(noisy[key]) == pytest.approx(float(quiet[key]), rel=1e-9), key
