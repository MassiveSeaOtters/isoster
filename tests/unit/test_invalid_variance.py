"""Samples with unusable variance are dropped, so every ring statistic and its
uncertainty describe one identical sample set.

A variance that is not finite or not strictly positive carries no usable
information, so the sample is treated exactly like a masked pixel and removed
during sampling. See docs/agent/plans/2026-08-12-gradient-error-ring-statistics.md.
"""

import numpy as np
import pytest

from isoster.sampling import extract_isophote_data

SHAPE = (121, 121)
CENTER = 60.0
SMA = 20.0


def make_flat_image(value=100.0):
    return np.full(SHAPE, value, dtype=np.float64)


def sample(image, variance_map):
    return extract_isophote_data(image, None, CENTER, CENTER, SMA, 0.3, 0.4, variance_map=variance_map)


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf, 0.0, -1.0],
    ids=["nan", "posinf", "neginf", "zero", "negative"],
)
def test_unusable_variance_samples_are_dropped(bad_value):
    """Every unusable variance value removes its samples from the ring."""
    image = make_flat_image()
    clean = np.full(SHAPE, 4.0)
    dirty = clean.copy()
    dirty[50:70, 70:85] = bad_value

    n_clean = sample(image, clean).intens.size
    dirty_data = sample(image, dirty)

    assert dirty_data.intens.size < n_clean
    assert np.all(np.isfinite(dirty_data.variances))
    assert np.all(dirty_data.variances > 0.0)


def test_all_arrays_stay_aligned_after_dropping():
    """Dropping must remove the same positions from every returned array."""
    image = make_flat_image()
    variance = np.full(SHAPE, 4.0)
    variance[50:70, 70:85] = 0.0

    data = sample(image, variance)
    n = data.intens.size
    assert data.angles.size == n
    assert data.phi.size == n
    assert data.radii.size == n
    assert data.variances.size == n


def test_clean_variance_map_drops_nothing():
    """A fully valid variance map must not change the sample count."""
    image = make_flat_image()
    variance = np.full(SHAPE, 4.0)

    with_variance = sample(image, variance)
    without_variance = sample(image, None)

    assert with_variance.intens.size == without_variance.intens.size
    np.testing.assert_array_equal(with_variance.intens, without_variance.intens)


def test_no_variance_map_path_is_unchanged():
    """variance_map=None must be byte-identical and carry no variances."""
    image = make_flat_image()
    data = sample(image, None)
    assert data.variances is None
    assert data.intens.size > 0


import warnings

from isoster.config import IsosterConfig
from isoster.driver import fit_image


def make_sersic_image(size=121, amplitude=1000.0, r_eff=20.0):
    from scipy.special import gammaincinv

    bn = gammaincinv(4.0, 0.5)
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((xx - size // 2) ** 2 + ((yy - size // 2) / 0.7) ** 2)
    return amplitude * np.exp(-bn * ((r / r_eff) ** 0.5 - 1.0))


def test_fit_image_does_not_mutate_the_callers_variance_map():
    """The driver must work on a copy, whatever it marks invalid."""
    image = make_sersic_image()
    variance = np.full(image.shape, 4.0)
    variance[30:40, 30:40] = -1.0
    original = variance.copy()

    cfg = IsosterConfig(sma0=6.0, minsma=3.0, maxsma=40.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_image(image, None, cfg, variance_map=variance)

    np.testing.assert_array_equal(variance, original)


@pytest.mark.parametrize(
    "bad_value, expected_phrase",
    [(np.nan, "non-finite"), (np.inf, "non-finite"), (-1.0, "non-positive"), (0.0, "non-positive")],
    ids=["nan", "inf", "negative", "zero"],
)
def test_fit_image_warns_and_excludes_unusable_variance(bad_value, expected_phrase):
    """Unusable entries are reported and excluded, never clamped or substituted."""
    image = make_sersic_image()
    variance = np.full(image.shape, 4.0)
    variance[30:40, 30:40] = bad_value

    cfg = IsosterConfig(sma0=6.0, minsma=3.0, maxsma=40.0)
    with pytest.warns(RuntimeWarning, match=expected_phrase):
        result = fit_image(image, None, cfg, variance_map=variance)

    assert len(result["isophotes"]) > 0


def test_variance_sentinel_constant_is_gone():
    """The 1e30 sentinel is retired; nothing may reintroduce it."""
    import isoster._shared as shared

    assert not hasattr(shared, "VARIANCE_SENTINEL")


def test_fit_image_grad_error_survives_unmassaged_variance_map():
    """grad_error, produced end-to-end by fit_image -> driver -> sampling, must
    stay finite and close to a clean-run value when the caller's variance map
    is an unmassaged array: NaN, inf, and non-positive entries sit in it
    exactly as a survey pipeline would hand them over, in one contiguous block
    that spans the radii of several isophote rings.

    This is the end-to-end path that earlier tests miss:
    ``test_gradient_error_sentinel_immune`` (test_variance_map.py) calls
    ``compute_gradient`` directly with a raw NaN array, bypassing the
    driver's own NaN/inf/non-positive marking in ``driver.py``, and
    ``TestVarianceSentinelRobustness`` runs ``fit_image`` end to end but only
    checks ``intens_err`` and stop codes, never ``grad_error`` or ``ndata``.
    """
    image = make_sersic_image()
    clean_variance = np.full(image.shape, 4.0)
    dirty_variance = clean_variance.copy()
    # One contiguous block, split into the three ways a caller's variance map
    # goes bad unmassaged. Its radii (~20-37 pixels from the image center)
    # cross several isophote rings as sma sweeps from sma0=6 to maxsma=40.
    dirty_variance[50:56, 80:100] = np.nan
    dirty_variance[56:63, 80:100] = np.inf
    dirty_variance[63:66, 80:100] = 0.0
    dirty_variance[66:70, 80:100] = -1.0

    cfg = IsosterConfig(sma0=6.0, minsma=3.0, maxsma=40.0, debug=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = fit_image(image, None, cfg, variance_map=clean_variance)["isophotes"]
        dirty = fit_image(image, None, cfg, variance_map=dirty_variance)["isophotes"]

    assert len(dirty) == len(clean)
    # The two runs share the same deterministic sma grid (sma steps come
    # from the config, not from the data), so isophotes at the same index
    # describe the same ring in both runs.
    sma_dirty = np.array([iso["sma"] for iso in dirty])
    sma_clean = np.array([iso["sma"] for iso in clean])
    np.testing.assert_allclose(sma_dirty, sma_clean)

    affected = [i for i, iso in enumerate(dirty) if 20.0 <= iso["sma"] <= 38.0]
    assert len(affected) >= 5, "test block must cross several rings in this sma range"

    for i in affected:
        grad_err_dirty = dirty[i]["grad_error"]
        grad_err_clean = clean[i]["grad_error"]
        assert np.isfinite(grad_err_dirty), f"non-finite grad_error at sma={dirty[i]['sma']}"
        # Sane magnitude: a handful of dropped samples perturbs the ring
        # statistic by a modest, bounded amount, nowhere near the
        # many-orders-of-magnitude blow-up a kept large-variance entry (or
        # the retired sentinel/clamp) would cause.
        assert grad_err_dirty < 3.0 * grad_err_clean, (
            f"grad_error inflated at sma={dirty[i]['sma']}: dirty={grad_err_dirty:.6g}, clean={grad_err_clean:.6g}"
        )
        # The exclusion, not down-weighting, is what did the work: the
        # affected rings must show a real drop in sample count.
        assert dirty[i]["ndata"] < clean[i]["ndata"], (
            f"ndata did not drop at sma={dirty[i]['sma']}: dirty={dirty[i]['ndata']}, clean={clean[i]['ndata']}"
        )


# ---------------------------------------------------------------------------
# Invalid source pixels must not survive bilinear interpolation
# ---------------------------------------------------------------------------


def ring_pixel(sma=SMA, eps=0.3, pa=0.4, index=10):
    """Row and column of one image pixel the sampling ring passes through."""
    from isoster.numba_kernels import compute_ellipse_coords

    n_samples = max(64, int(2 * np.pi * sma))
    x, y, _, _ = compute_ellipse_coords(n_samples, sma, eps, pa, CENTER, CENTER, False)
    return int(round(y[index])), int(round(x[index]))


@pytest.mark.parametrize("bad_value", [0.0, -1.0], ids=["zero", "negative"])
def test_isolated_invalid_source_pixel_is_dropped(bad_value):
    """A lone zero or negative variance pixel must not be blended back into validity.

    Bilinear interpolation mixes an invalid source pixel with its positive
    neighbours, producing a positive interpolated variance that passes a
    value-only check. NaN and inf propagate through interpolation on their own,
    but zero and negative values do not, so the four source pixels behind each
    sample have to be checked directly.
    """
    image = make_flat_image()
    variance = np.full(SHAPE, 4.0)
    row, col = ring_pixel()

    clean_count = sample(image, variance).intens.size

    dirty = variance.copy()
    dirty[row, col] = bad_value
    dirty_data = sample(image, dirty)

    marked_invalid = variance.copy()
    marked_invalid[row, col] = np.nan
    expected_count = sample(image, marked_invalid).intens.size

    assert dirty_data.intens.size < clean_count
    assert dirty_data.intens.size == expected_count
    assert np.all(dirty_data.variances > 0.0)


def test_support_check_does_not_drop_samples_on_a_clean_map():
    """The check must not cost valid samples."""
    image = make_flat_image()
    variance = np.full(SHAPE, 4.0)

    assert sample(image, variance).intens.size == sample(image, None).intens.size


def test_prepared_variance_map_skips_the_support_check():
    """fit_image already sanitises, so it may opt out of the per-sample check.

    Declaring a map prepared is a promise that unusable entries are already
    non-finite; a raw zero would then survive, which is exactly why the opt-out
    is off by default.
    """
    image = make_flat_image()
    variance = np.full(SHAPE, 4.0)
    row, col = ring_pixel()
    dirty = variance.copy()
    dirty[row, col] = 0.0

    checked = extract_isophote_data(image, None, CENTER, CENTER, SMA, 0.3, 0.4, variance_map=dirty)
    skipped = extract_isophote_data(
        image, None, CENTER, CENTER, SMA, 0.3, 0.4, variance_map=dirty, variance_map_prepared=True
    )

    assert skipped.intens.size > checked.intens.size


def test_fit_image_still_excludes_invalid_pixels_through_the_prepared_path():
    """fit_image opts out of the check but must still exclude invalid samples.

    Its own sanitisation converts unusable entries to NaN before sampling, and
    NaN propagates through interpolation, so the exclusion still happens.
    """
    image = make_sersic_image()
    variance = np.full(image.shape, 4.0)
    # Off-centre, so it clips mid-radius rings without starving the inner ones.
    variance[70:78, 70:78] = 0.0

    cfg = IsosterConfig(sma0=6.0, minsma=3.0, maxsma=40.0, debug=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dirty = fit_image(image, None, cfg, variance_map=variance)["isophotes"]
        clean = fit_image(image, None, cfg, variance_map=np.full(image.shape, 4.0))["isophotes"]

    dirty_by_sma = {round(i["sma"], 4): i for i in dirty}
    dropped_somewhere = any(
        dirty_by_sma[round(i["sma"], 4)]["ndata"] < i["ndata"] for i in clean if round(i["sma"], 4) in dirty_by_sma
    )
    assert dropped_somewhere


# ---------------------------------------------------------------------------
# The source-pixel check must match SciPy's interpolation footprint exactly
# ---------------------------------------------------------------------------


def reference_validity(variance_map, x, y):
    """Which samples survive when invalid entries are marked NaN first.

    This is exactly what ``fit_image`` does, so it is the behaviour the
    per-sample source-pixel check has to reproduce for a raw map.
    """
    from scipy.ndimage import map_coordinates

    marked = np.where(np.isfinite(variance_map) & (variance_map > 0.0), variance_map, np.nan)
    interpolated = map_coordinates(marked, np.vstack([y, x]), order=1, mode="constant", cval=np.nan)
    return np.isfinite(interpolated) & (interpolated > 0.0)


@pytest.mark.parametrize("bad_value", [0.0, -1.0], ids=["zero", "negative"])
def test_support_check_covers_the_penultimate_cell_on_the_final_row_and_column(bad_value):
    """Sampling exactly on the last row or column still reaches the cell before it.

    At an exact final index SciPy shifts its interpolation interval to the last
    two cells and gives the penultimate one zero weight -- but a NaN there still
    propagates, because zero times NaN is NaN. Clipping each neighbour
    independently would inspect the final cell twice and miss it.
    """
    from isoster.sampling import _bilinear_support_is_valid

    size = 6
    variance = np.full((size, size), 4.0)
    variance[2, size - 2] = bad_value  # penultimate column
    variance[size - 2, 2] = bad_value  # penultimate row

    # Sample exactly on the final column, and exactly on the final row.
    x = np.array([float(size - 1), 2.0])
    y = np.array([2.0, float(size - 1)])

    support = _bilinear_support_is_valid(variance, x, y)
    assert not support.any()
    np.testing.assert_array_equal(support, reference_validity(variance, x, y))


def test_support_check_matches_scipy_across_shapes_and_edges():
    """The raw-map path and the sanitise-first path must exclude the same samples."""
    from isoster.sampling import _bilinear_support_is_valid

    rng = np.random.default_rng(20260812)
    compared = 0
    for _ in range(200):
        height, width = rng.integers(4, 9, size=2)
        variance = np.full((height, width), 4.0)
        for _ in range(rng.integers(1, 4)):
            variance[rng.integers(0, height), rng.integers(0, width)] = rng.choice([0.0, -1.0])

        # Interior coordinates plus every exact corner and edge.
        x = np.concatenate([rng.uniform(0, width - 1, 20), [width - 1.0, width - 1.0, 0.0, 0.0]])
        y = np.concatenate([rng.uniform(0, height - 1, 20), [height - 1.0, 0.0, height - 1.0, 0.0]])
        compared += x.size

        np.testing.assert_array_equal(_bilinear_support_is_valid(variance, x, y), reference_validity(variance, x, y))

    assert compared > 4000


def test_support_check_handles_a_single_row_or_column_image():
    """An axis of length one has no penultimate cell to fall back to."""
    from isoster.sampling import _bilinear_support_is_valid

    single_row = np.full((1, 5), 4.0)
    single_row[0, 3] = 0.0
    x = np.array([0.0, 3.0, 4.0])
    y = np.zeros(3)
    np.testing.assert_array_equal(_bilinear_support_is_valid(single_row, x, y), reference_validity(single_row, x, y))

    single_pixel = np.full((1, 1), 4.0)
    assert _bilinear_support_is_valid(single_pixel, np.array([0.0]), np.array([0.0])).tolist() == [True]
