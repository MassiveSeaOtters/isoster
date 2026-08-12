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
