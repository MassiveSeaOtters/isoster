"""The multi-band ring read honours ``sampling_interpolation`` like single-band.

Before 1.1.1 the multi-band sampler had its own bilinear kernel and ignored the
field, so a band fitted on its own (which delegates to the single-band driver and
read cubically) disagreed with the same band fitted inside a stack by the whole
bilinear read bias, 1.0 % RMS on a PSF-convolved n=2 Sersic. These tests pin the
kernel at three levels: the numba stack kernel against scipy, the prepared
sampler against the single-band sampler, and a two-band stack of one image
against the single-band fit of that image.
"""

import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates, spline_filter

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.multiband import fit_image_multiband
from isoster.multiband.config_mb import IsosterConfigMB
from isoster.multiband.numba_kernels_mb import (
    _bilinear_sample_stack_numba,
    _cubic_sample_stack_numba,
    _cubic_sample_stack_numpy,
)
from isoster.multiband.sampling_mb import (
    extract_isophote_data_multi_prepared,
    prepare_sampling_stack,
)
from isoster.sampling import extract_isophote_data, prepare_sampling_image


def _sersic_image(h=300, w=300, n=2.0, re=20.0, eps=0.3, pa=0.6, peak=1000.0, psf_sigma=1.5, noise=1.0, seed=0):
    y, x = np.mgrid[:h, :w]
    dx, dy = x - w / 2.0, y - h / 2.0
    xr = dx * np.cos(pa) + dy * np.sin(pa)
    yr = -dx * np.sin(pa) + dy * np.cos(pa)
    r = np.sqrt(xr**2 + (yr / (1.0 - eps)) ** 2)
    bn = 2.0 * n - 1.0 / 3.0
    image = gaussian_filter(peak * np.exp(-bn * ((r / re) ** (1.0 / n) - 1.0)), psf_sigma)
    return image + np.random.default_rng(seed).normal(0.0, noise, image.shape)


GEOMETRY = dict(x0=150.0, y0=150.0, eps=0.25, pa=0.5, sma0=10.0, maxsma=120.0)


def _intensities(result, key):
    return {round(float(iso["sma"]), 6): float(iso[key]) for iso in result["isophotes"] if iso["stop_code"] == 0}


def _relative_differences(a, b):
    common = sorted(set(a) & set(b))
    assert len(common) > 20
    return np.array([(b[s] - a[s]) / a[s] for s in common])


# ---------------------------------------------------------------------------
# Kernel level: the numba cubic stack read reproduces scipy, edges included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", [_cubic_sample_stack_numba, _cubic_sample_stack_numpy])
def test_cubic_stack_kernel_matches_scipy_including_the_edge_footprint(kernel):
    rng = np.random.default_rng(1)
    stack = rng.normal(size=(2, 40, 50)) + 10.0
    coeffs = np.stack([spline_filter(band, order=3) for band in stack])
    # Interior points, points whose 4x4 footprint crosses the border, points on the
    # border itself, and points outside the image.
    y = np.concatenate(
        [rng.uniform(2, 37, 500), rng.uniform(0, 1.5, 100), rng.uniform(37.5, 39, 100), [0.0, 39.0, -0.5, 39.5]]
    )
    x = np.concatenate(
        [rng.uniform(2, 47, 500), rng.uniform(0, 1.5, 100), rng.uniform(47.5, 49, 100), [0.0, 49.0, 10.0, 10.0]]
    )
    out = np.empty((2, y.size))
    kernel(coeffs, stack, y, x, out)
    for band in range(2):
        reference = map_coordinates(
            coeffs[band], np.vstack([y, x]), order=3, mode="constant", cval=np.nan, prefilter=False
        )
        assert np.array_equal(np.isnan(out[band]), np.isnan(reference))
        finite = np.isfinite(reference)
        assert np.max(np.abs(out[band][finite] - reference[finite])) < 1e-12


def test_cubic_stack_kernel_drops_samples_touching_non_finite_pixels_like_the_bilinear_one():
    rng = np.random.default_rng(2)
    stack = rng.normal(size=(2, 60, 60)) + 10.0
    stack[1, 25:30, 30:36] = np.nan
    sampling_stack, order = prepare_sampling_stack(stack, "cubic")
    assert order == 3 and np.isfinite(sampling_stack).all()
    y = rng.uniform(20, 35, 2000)
    x = rng.uniform(25, 40, 2000)
    cubic = np.empty((2, y.size))
    bilinear = np.empty((2, y.size))
    _cubic_sample_stack_numba(sampling_stack, stack, y, x, cubic)
    _bilinear_sample_stack_numba(stack, y, x, bilinear)
    assert np.isnan(cubic[1]).sum() > 100
    assert np.array_equal(np.isnan(cubic[1]), np.isnan(bilinear[1]))
    assert np.isfinite(cubic[0]).all()


# ---------------------------------------------------------------------------
# Sampler level: one band through the prepared multi-band sampler equals the
# single-band sampler with the same kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interpolation", ["bilinear", "cubic"])
def test_prepared_sampler_matches_the_single_band_sampler(interpolation):
    image = _sersic_image(h=200, w=200)
    mask = np.zeros(image.shape, dtype=bool)
    stack = image[None].astype(np.float64)
    sampling_stack, order = prepare_sampling_stack(stack, interpolation)
    sampling_image, order_sb = prepare_sampling_image(image, interpolation)
    assert order == order_sb
    for sma in (5.0, 20.0, 60.0):
        single = extract_isophote_data(sampling_image, mask, 100.0, 100.0, sma, 0.3, 0.4, interpolation_order=order_sb)
        multi = extract_isophote_data_multi_prepared(
            stack, [None], None, 100.0, 100.0, sma, 0.3, 0.4, sampling_stack=sampling_stack, interpolation_order=order
        )
        assert multi.intens.shape[1] == single.intens.size
        np.testing.assert_allclose(multi.intens[0], single.intens, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Fit level: the same band in a two-band stack agrees with its single-band fit
# ---------------------------------------------------------------------------


def _fit_pair(interpolation, **extra):
    image = _sersic_image()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        single = fit_image(image, None, IsosterConfig(**GEOMETRY, sampling_interpolation=interpolation, **extra))
        multi = fit_image_multiband(
            [image, image],
            None,
            IsosterConfigMB(
                bands=["g", "r"], reference_band="r", **GEOMETRY, sampling_interpolation=interpolation, **extra
            ),
        )
    return _intensities(single, "intens"), _intensities(multi, "intens_r")


@pytest.mark.parametrize("interpolation", ["bilinear", "cubic"])
def test_two_band_stack_of_one_image_matches_the_single_band_fit(interpolation):
    single, multi = _fit_pair(interpolation)
    difference = _relative_differences(single, multi)
    # The joint solve converges along a slightly different path than the single-band
    # solve, which moves the intensities at the 1e-4 level; the kernel moves them at
    # the 1e-2 level (see the next test), so this tolerance separates the two.
    assert np.sqrt(np.mean(difference**2)) < 5e-4
    assert np.max(np.abs(difference)) < 2e-3


def test_the_kernel_changes_the_multi_band_intensities_as_much_as_the_single_band_ones():
    single_bilinear, multi_bilinear = _fit_pair("bilinear")
    single_cubic, multi_cubic = _fit_pair("cubic")
    gap_single = _relative_differences(single_cubic, single_bilinear)
    gap_multi = _relative_differences(multi_cubic, multi_bilinear)
    assert np.sqrt(np.mean(gap_single**2)) > 5e-3
    assert np.sqrt(np.mean(gap_multi**2)) == pytest.approx(np.sqrt(np.mean(gap_single**2)), rel=0.1)


def test_the_inner_fallback_reads_bilinearly():
    image = _sersic_image()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bilinear = fit_image_multiband(
            [image, image],
            None,
            IsosterConfigMB(bands=["g", "r"], reference_band="r", **GEOMETRY, sampling_interpolation="bilinear"),
        )
        fallback = fit_image_multiband(
            [image, image],
            None,
            IsosterConfigMB(bands=["g", "r"], reference_band="r", **GEOMETRY, sampling_cubic_min_sma=1000.0),
        )
        cubic = fit_image_multiband(
            [image, image], None, IsosterConfigMB(bands=["g", "r"], reference_band="r", **GEOMETRY)
        )
    assert _intensities(fallback, "intens_r") == _intensities(bilinear, "intens_r")
    assert _intensities(cubic, "intens_r") != _intensities(bilinear, "intens_r")


def test_single_band_delegation_passes_the_kernel_through():
    image = _sersic_image()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        single = fit_image(image, None, IsosterConfig(**GEOMETRY, sampling_interpolation="bilinear"))
        delegated = fit_image_multiband(
            [image],
            None,
            IsosterConfigMB(bands=["g"], reference_band="g", **GEOMETRY, sampling_interpolation="bilinear"),
        )
    assert _intensities(delegated, "intens") == _intensities(single, "intens")


def test_forced_photometry_reads_with_the_fit_kernel():
    image = _sersic_image()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = IsosterConfigMB(bands=["g", "r"], reference_band="r", **GEOMETRY)
        free = fit_image_multiband([image, image], None, config)
        template = [
            {"sma": iso["sma"], "x0": iso["x0"], "y0": iso["y0"], "eps": iso["eps"], "pa": iso["pa"]}
            for iso in free["isophotes"]
            if iso["sma"] > 0 and iso["stop_code"] == 0
        ]
        forced_cubic = fit_image_multiband([image, image], None, config, template_isophotes=template)
        forced_bilinear = fit_image_multiband(
            [image, image],
            None,
            IsosterConfigMB(bands=["g", "r"], reference_band="r", **GEOMETRY, sampling_interpolation="bilinear"),
            template_isophotes=template,
        )
    free_intens = _intensities(free, "intens_r")
    agreement = _relative_differences(free_intens, _intensities(forced_cubic, "intens_r"))
    kernel_gap = _relative_differences(
        _intensities(forced_cubic, "intens_r"), _intensities(forced_bilinear, "intens_r")
    )
    # Forced extraction at the free fit's own geometry re-reads the converged ring
    # with the same kernel; switching the kernel moves it by the read bias.
    assert np.max(np.abs(agreement)) < 2e-5
    assert np.sqrt(np.mean(kernel_gap**2)) > 5e-3
