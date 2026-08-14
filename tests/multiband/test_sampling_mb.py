"""Tests for ``isoster.multiband.sampling_mb``."""

import warnings

import numpy as np
import pytest

from isoster.multiband.numba_kernels_mb import (
    _build_joint_design_matrix_numpy,
    build_joint_design_matrix,
)
from isoster.multiband.sampling_mb import (
    MultiIsophoteData,
    extract_isophote_data_multi,
)
from isoster.sampling import extract_isophote_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_image(h=128, w=128, x0=64.0, y0=64.0, sigma=20.0, amplitude=100.0):
    """A radially-symmetric Gaussian — easy ground truth for sampler tests."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return amplitude * np.exp(-r2 / (2.0 * sigma**2))


# ---------------------------------------------------------------------------
# B=1 numerical parity with single-band sampler (decision D14 delegation)
# ---------------------------------------------------------------------------


def test_b1_intensity_matches_single_band():
    """Multi-band sampler with B=1 should reproduce single-band intensities."""
    img = _gaussian_image()
    mask = np.zeros_like(img, dtype=bool)

    sb = extract_isophote_data(img, mask, x0=64.0, y0=64.0, sma=15.0, eps=0.2, pa=0.3)
    mb = extract_isophote_data_multi([img], mask, x0=64.0, y0=64.0, sma=15.0, eps=0.2, pa=0.3)

    assert isinstance(mb, MultiIsophoteData)
    assert mb.intens.shape == (1, len(sb.intens))
    np.testing.assert_allclose(mb.intens[0], sb.intens, atol=1e-12)
    np.testing.assert_allclose(mb.angles, sb.angles, atol=1e-12)
    np.testing.assert_allclose(mb.phi, sb.phi, atol=1e-12)


def test_b1_with_variance_matches_single_band():
    img = _gaussian_image()
    mask = np.zeros_like(img, dtype=bool)
    var = np.ones_like(img) * 0.1

    sb = extract_isophote_data(img, mask, x0=64.0, y0=64.0, sma=15.0, eps=0.2, pa=0.3, variance_map=var)
    mb = extract_isophote_data_multi([img], mask, x0=64.0, y0=64.0, sma=15.0, eps=0.2, pa=0.3, variance_maps=[var])

    np.testing.assert_allclose(mb.intens[0], sb.intens, atol=1e-12)
    assert mb.variances is not None
    np.testing.assert_allclose(mb.variances[0], sb.variances, atol=1e-12)


# ---------------------------------------------------------------------------
# Shared-validity rule
# ---------------------------------------------------------------------------


def test_shared_validity_via_per_band_masks():
    """A mask hit in any band drops the sample from all bands."""
    img1 = _gaussian_image()
    img2 = _gaussian_image() * 0.5

    mask1 = np.zeros_like(img1, dtype=bool)
    mask2 = np.zeros_like(img2, dtype=bool)
    # Block out a wedge of band 2
    mask2[60:68, 80:96] = True

    mb_unmasked = extract_isophote_data_multi(
        [img1, img2],
        masks=[mask1, np.zeros_like(img1, dtype=bool)],
        x0=64.0,
        y0=64.0,
        sma=20.0,
        eps=0.0,
        pa=0.0,
    )
    mb_masked = extract_isophote_data_multi(
        [img1, img2],
        masks=[mask1, mask2],
        x0=64.0,
        y0=64.0,
        sma=20.0,
        eps=0.0,
        pa=0.0,
    )

    # Some samples should have been dropped from both bands.
    assert mb_masked.valid_count < mb_unmasked.valid_count
    # Both bands have the same N (shared validity).
    assert mb_masked.intens.shape[1] == mb_masked.valid_count


def test_shared_validity_via_nan_in_one_band():
    """A NaN pixel hit in one band drops the sample from all bands."""
    img1 = _gaussian_image()
    img2 = _gaussian_image() * 0.7
    # Plant NaN values in band 2 at the location an SMA-20 ring would reach.
    img2[64, 84] = np.nan

    mb = extract_isophote_data_multi(
        [img1, img2],
        None,
        x0=64.0,
        y0=64.0,
        sma=20.0,
        eps=0.0,
        pa=0.0,
    )
    # All retained intensities are finite in *both* bands.
    assert np.all(np.isfinite(mb.intens))


def test_non_positive_variance_dropped_and_warned():
    """Non-positive variance entries are invalid: their samples are dropped.

    A variance of zero or below is not physical, so it carries no usable
    information about the noise. The retired policy clamped such entries to
    ``1e-30``, which kept the sample in every ring statistic while giving it
    near-infinite weight. Mirrors single-band ``fit_image``.
    """
    img1 = _gaussian_image()
    img2 = _gaussian_image() * 0.7
    var1 = np.ones_like(img1) * 0.5
    var2 = np.ones_like(img2) * 0.5
    var2[64, 84] = -1.0

    mb_clean = extract_isophote_data_multi(
        [img1, img2],
        None,
        x0=64.0,
        y0=64.0,
        sma=20.0,
        eps=0.0,
        pa=0.0,
        variance_maps=[var1, var1.copy()],
    )
    with pytest.warns(RuntimeWarning, match="non-positive"):
        mb_sanit = extract_isophote_data_multi(
            [img1, img2],
            None,
            x0=64.0,
            y0=64.0,
            sma=20.0,
            eps=0.0,
            pa=0.0,
            variance_maps=[var1, var2],
        )
    # The bad pixel is excluded, so strictly fewer samples survive than on
    # the clean map, and every surviving variance is usable.
    assert mb_sanit.valid_count < mb_clean.valid_count
    assert mb_sanit.variances is not None
    assert np.all(mb_sanit.variances > 0.0)
    assert np.all(np.isfinite(mb_sanit.variances))


def test_all_masked_returns_zero_valid():
    """When every sample is masked in some band, valid_count is zero."""
    img1 = _gaussian_image()
    img2 = _gaussian_image()
    mask_all = np.ones_like(img1, dtype=bool)
    mb = extract_isophote_data_multi(
        [img1, img2],
        [None, mask_all],
        x0=64.0,
        y0=64.0,
        sma=20.0,
        eps=0.0,
        pa=0.0,
    )
    assert mb.valid_count == 0
    assert mb.intens.shape == (2, 0)


# ---------------------------------------------------------------------------
# Mask broadcasting and variance all-or-nothing semantics
# ---------------------------------------------------------------------------


def test_single_mask_broadcasts():
    img1 = _gaussian_image()
    img2 = _gaussian_image() * 0.5
    mask = np.zeros_like(img1, dtype=bool)
    mask[60:68, 80:96] = True

    mb_broadcast = extract_isophote_data_multi(
        [img1, img2],
        mask,
        x0=64.0,
        y0=64.0,
        sma=20.0,
        eps=0.0,
        pa=0.0,
    )
    mb_per_band = extract_isophote_data_multi(
        [img1, img2],
        [mask, mask],
        x0=64.0,
        y0=64.0,
        sma=20.0,
        eps=0.0,
        pa=0.0,
    )
    np.testing.assert_array_equal(mb_broadcast.intens, mb_per_band.intens)


def test_variance_all_or_nothing_rejects_none_in_list():
    img = _gaussian_image()
    var = np.ones_like(img)
    with pytest.raises(ValueError, match="all-or-nothing"):
        extract_isophote_data_multi(
            [img, img],
            None,
            x0=64.0,
            y0=64.0,
            sma=15.0,
            eps=0.0,
            pa=0.0,
            variance_maps=[var, None],  # type: ignore[list-item]
        )


def test_variance_single_ndarray_broadcasts():
    img1 = _gaussian_image()
    img2 = _gaussian_image() * 0.5
    var = np.ones_like(img1) * 0.2
    mb = extract_isophote_data_multi(
        [img1, img2],
        None,
        x0=64.0,
        y0=64.0,
        sma=15.0,
        eps=0.0,
        pa=0.0,
        variance_maps=var,
    )
    assert mb.variances is not None
    np.testing.assert_allclose(mb.variances[0], mb.variances[1], atol=1e-12)


# ---------------------------------------------------------------------------
# Shape mismatches
# ---------------------------------------------------------------------------


def test_shape_mismatch_in_images_rejected():
    img1 = _gaussian_image()
    img2 = _gaussian_image(h=64, w=64)
    with pytest.raises(ValueError, match="shape"):
        extract_isophote_data_multi(
            [img1, img2],
            None,
            x0=64.0,
            y0=64.0,
            sma=15.0,
            eps=0.0,
            pa=0.0,
        )


def test_mask_list_wrong_length_rejected():
    img = _gaussian_image()
    mask = np.zeros_like(img, dtype=bool)
    with pytest.raises(ValueError, match="length"):
        extract_isophote_data_multi(
            [img, img],
            [mask],
            x0=64.0,
            y0=64.0,
            sma=15.0,
            eps=0.0,
            pa=0.0,
        )


def test_variance_list_wrong_length_rejected():
    img = _gaussian_image()
    var = np.ones_like(img)
    with pytest.raises(ValueError, match="length"):
        extract_isophote_data_multi(
            [img, img, img],
            None,
            x0=64.0,
            y0=64.0,
            sma=15.0,
            eps=0.0,
            pa=0.0,
            variance_maps=[var, var],
        )


def test_variance_invalid_entries_marked_nan_with_warning():
    """Non-finite and non-positive variance entries are marked NaN, with a warning.

    Replaces the retired sentinel scheme. ``1e30`` was finite and strictly
    positive, so the sampler's validity rule kept it rather than dropping it
    as its comment claimed; ``1e-30`` gave one pixel near-infinite weight.
    """
    from isoster.multiband.sampling_mb import _resolve_variance_maps

    h, w = 32, 32
    var = np.ones((h, w), dtype=np.float64)
    var[0, 0] = np.nan
    var[0, 1] = np.inf
    var[1, 0] = -1.0
    var[1, 1] = 0.0

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        out = _resolve_variance_maps([var, var.copy()], n_bands=2, h=h, w=w)
    msgs = [str(w.message) for w in captured if issubclass(w.category, RuntimeWarning)]
    assert any("non-finite" in m for m in msgs)
    assert any("non-positive" in m for m in msgs)
    assert out is not None
    for arr in out:
        assert np.isnan(arr[0, 0])
        assert np.isnan(arr[0, 1])
        assert np.isnan(arr[1, 0])
        assert np.isnan(arr[1, 1])
        # Everything else is untouched and still usable.
        assert np.isfinite(arr[2:, 2:]).all()
        assert (arr[2:, 2:] > 0.0).all()


def test_retired_variance_sentinels_are_gone():
    """Nothing may reintroduce the 1e30 sentinel or the 1e-30 clamp."""
    from isoster.multiband import sampling_mb

    assert not hasattr(sampling_mb, "BAD_PIXEL_VARIANCE")
    assert not hasattr(sampling_mb, "MIN_POSITIVE_VARIANCE")

    h, w = 16, 16
    var = np.ones((h, w), dtype=np.float64)
    var[3, 3] = np.nan
    var[4, 4] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = sampling_mb._resolve_variance_maps([var], n_bands=1, h=h, w=w)
    assert out is not None
    finite = out[0][np.isfinite(out[0])]
    assert finite.max() <= 1.0
    assert finite.min() >= 1.0


def test_variance_clean_arrays_emit_no_warning():
    """Sanitization is silent when the input is already clean."""
    from isoster.multiband.sampling_mb import _resolve_variance_maps

    h, w = 16, 16
    var = np.full((h, w), 0.25, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _resolve_variance_maps([var, var.copy()], n_bands=2, h=h, w=w)
    assert out is not None
    np.testing.assert_array_equal(out[0], var)


def test_variance_broadcast_ndarray_marked():
    """The single-ndarray broadcast path also marks invalid entries."""
    from isoster.multiband.sampling_mb import _resolve_variance_maps

    h, w = 24, 24
    var = np.ones((h, w), dtype=np.float64)
    var[5, 5] = np.nan
    with pytest.warns(RuntimeWarning, match="non-finite"):
        out = _resolve_variance_maps(var, n_bands=3, h=h, w=w)
    assert out is not None and len(out) == 3
    for arr in out:
        assert np.isnan(arr[5, 5])


# ---------------------------------------------------------------------------
# Joint design matrix kernel
# ---------------------------------------------------------------------------


def test_joint_design_matrix_numba_matches_numpy():
    phi = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    for B in (1, 2, 3, 5):
        A_njit = build_joint_design_matrix(phi, B)
        A_np = _build_joint_design_matrix_numpy(phi, B)
        np.testing.assert_allclose(A_njit, A_np, atol=0.0)
        assert A_njit.shape == (B * len(phi), B + 4)


def test_joint_design_matrix_b1_equals_single_band_matrix():
    """B=1 joint matrix has shape (N, 5) and equals [1, sin, cos, sin2, cos2]."""
    from isoster.numba_kernels import build_harmonic_matrix

    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    A_mb = build_joint_design_matrix(phi, 1)
    A_sb = build_harmonic_matrix(phi)
    assert A_mb.shape == A_sb.shape
    np.testing.assert_allclose(A_mb, A_sb, atol=1e-15)


def test_joint_design_matrix_indicator_block_is_band_diagonal():
    """The leading B columns are 1.0 only on the band's own row block."""
    phi = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    A = build_joint_design_matrix(phi, 3)
    # band 0 rows: indicator is [1, 0, 0]
    np.testing.assert_array_equal(A[0:8, :3], np.tile([[1.0, 0.0, 0.0]], (8, 1)))
    # band 1 rows: indicator is [0, 1, 0]
    np.testing.assert_array_equal(A[8:16, :3], np.tile([[0.0, 1.0, 0.0]], (8, 1)))
    # band 2 rows: indicator is [0, 0, 1]
    np.testing.assert_array_equal(A[16:24, :3], np.tile([[0.0, 0.0, 1.0]], (8, 1)))
    # geometric block repeats per band
    geom_band0 = A[0:8, 3:]
    geom_band1 = A[8:16, 3:]
    geom_band2 = A[16:24, 3:]
    np.testing.assert_array_equal(geom_band0, geom_band1)
    np.testing.assert_array_equal(geom_band0, geom_band2)


def test_joint_design_matrix_invalid_args_rejected():
    with pytest.raises(ValueError):
        build_joint_design_matrix(np.array([]), 2)
    with pytest.raises(ValueError):
        build_joint_design_matrix(np.linspace(0, 1, 5), 0)
