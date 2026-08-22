"""Forced photometry must measure its harmonics, not fabricate zeros.

Template-based forced photometry (``fit_image(..., template=...)``) reports
``a_n`` / ``b_n`` columns whenever ``compute_deviations`` is on. Those columns
used to be filled with literal ``0.0`` and never replaced: the ring was
sampled, the intensity was measured, and the harmonic solve was simply never
run. A user fitting a boxy galaxy in forced mode got a confident-looking
``a4 = 0.0`` for every isophote.

Zeros are the worst possible placeholder here, because zero is also the
correct answer for a perfect ellipse. There is no way for a caller to tell a
measured null result from an unmeasured one.

These tests plant a known fourth-order distortion and require that forced mode
recover it, that it agree with the free fit that measured the same rings, and
that an unmeasurable ring say so with NaN rather than with a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from isoster import fit_image
from isoster.config import IsosterConfig
from isoster.fitting import extract_forced_photometry

PLANTED_B4 = 0.06


def _planted_harmonic_image(size=241, r_eff_pix=25.0, sersic_n=2.0, eps=0.3, pa=0.0, eps4_cos=PLANTED_B4):
    """A Sersic galaxy whose isophotes carry a known fourth-order distortion.

    The isophote of value ``I(a)`` is placed at ``r = a * D(phi)`` with
    ``D(phi) = 1 + eps4_cos * cos(4 phi)``, phi measured from the major axis.
    To first order this makes the Bender-normalized ``b4`` equal to
    ``eps4_cos`` — see the harmonic-scale design note for the derivation.
    """
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[:size, :size]
    dx, dy = xx - centre, yy - centre
    x_rot = dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
    q = 1.0 - eps
    r_ellipse = np.sqrt(x_rot**2 + (y_rot / q) ** 2)
    phi = np.arctan2(y_rot / q, x_rot)
    distortion = 1.0 + eps4_cos * np.cos(4.0 * phi)
    r_scaled = np.maximum(r_ellipse / distortion, 1e-6)
    b_n = 2.0 * sersic_n - 1.0 / 3.0
    image = 500.0 * np.exp(-b_n * ((r_scaled / r_eff_pix) ** (1.0 / sersic_n) - 1.0))
    return image, centre


def _config(centre, **overrides):
    params = dict(
        sma0=12.0,
        maxsma=55.0,
        x0=centre,
        y0=centre,
        eps=0.3,
        pa=0.0,
        compute_deviations=True,
        harmonic_orders=[3, 4],
    )
    params.update(overrides)
    return IsosterConfig(**params)


class TestForcedPhotometryMeasuresHarmonics:
    def test_recovers_the_planted_fourth_order_signal(self):
        """Forced mode must measure b4, not report a fabricated zero."""
        image, centre = _planted_harmonic_image()
        config = _config(centre)

        free = fit_image(image, config=config)["isophotes"]
        forced = fit_image(image, config=config, template=free)["isophotes"]

        measured = [iso["b4"] for iso in forced if iso["sma"] > 0]
        assert measured, "no forced isophotes with positive sma"
        assert np.nanmax(np.abs(measured)) > 0.5 * PLANTED_B4, (
            "forced photometry reported no fourth-order signal on an image "
            f"with a planted b4 of {PLANTED_B4}; largest |b4| was "
            f"{np.nanmax(np.abs(measured)):.6f}"
        )

    def test_reproduces_the_free_fit_exactly_when_the_gradient_is_not_lazy(self):
        """Same image, same rings, same gradient: the harmonics must be identical.

        With ``use_lazy_gradient=False`` both paths measure the gradient
        fresh at every isophote, so there is nothing left to differ and the
        agreement is exact rather than approximate.
        """
        image, centre = _planted_harmonic_image()
        config = _config(centre, use_lazy_gradient=False)

        free = fit_image(image, config=config)["isophotes"]
        forced = fit_image(image, config=config, template=free)["isophotes"]

        assert len(free) == len(forced)
        compared = 0
        for free_iso, forced_iso in zip(free, forced):
            if free_iso["sma"] <= 0 or not np.isfinite(free_iso["b4"]):
                continue
            assert forced_iso["b4"] == free_iso["b4"], f"forced b4 differs from free b4 at sma={free_iso['sma']:.2f}"
            compared += 1
        assert compared > 5, "too few isophotes compared to be meaningful"

    def test_matches_the_free_fit_within_the_lazy_gradient_difference(self):
        """With the default lazy gradient, agreement is close but not exact.

        The free path may reuse a cached gradient across iterations while
        forced photometry always measures one fresh, so the shared Bender
        denominator can differ slightly. The residual is bounded and is a
        property of the lazy gradient, not of the harmonic solve -- the
        companion test above pins exact agreement once that cache is off.
        """
        image, centre = _planted_harmonic_image()
        config = _config(centre, use_lazy_gradient=True)

        free = fit_image(image, config=config)["isophotes"]
        forced = fit_image(image, config=config, template=free)["isophotes"]

        differences = [
            abs(forced_iso["b4"] - free_iso["b4"])
            for free_iso, forced_iso in zip(free, forced)
            if forced_iso["sma"] > 0 and np.isfinite(free_iso["b4"])
        ]
        assert differences
        assert max(differences) < 0.1 * PLANTED_B4, (
            f"lazy-gradient difference is larger than a tenth of the planted signal: {max(differences):.3e}"
        )

    def test_reports_nan_not_zero_when_the_ring_cannot_be_measured(self):
        """An unmeasurable ring must be distinguishable from a round one."""
        image, centre = _planted_harmonic_image(size=81)
        # A semi-major axis far outside the frame samples nothing.
        result = extract_forced_photometry(
            image,
            None,
            centre,
            centre,
            sma=400.0,
            eps=0.3,
            pa=0.0,
        )
        assert not result["valid"]
        assert np.isnan(result["a4"]), "unmeasured harmonic reported as a number"
        assert np.isnan(result["b4"]), "unmeasured harmonic reported as a number"

    def test_exposes_the_gradient_under_debug_like_the_free_path(self):
        """The Bender denominator must be inspectable, as in free fitting."""
        image, centre = _planted_harmonic_image()
        config = _config(centre, debug=True)

        free = fit_image(image, config=config)["isophotes"]
        forced = fit_image(image, config=config, template=free)["isophotes"]

        graded = [iso for iso in forced if iso["sma"] > 0 and "grad" in iso]
        assert graded, "forced photometry exposed no gradient under debug=True"
        assert np.isfinite(graded[0]["grad"])

    def test_no_harmonic_keys_when_deviations_are_switched_off(self):
        """Unchanged behavior: no columns at all rather than empty ones."""
        image, centre = _planted_harmonic_image()
        config = _config(centre, compute_deviations=False)

        free = fit_image(image, config=_config(centre))["isophotes"]
        forced = fit_image(image, config=config, template=free)["isophotes"]

        assert "a4" not in forced[-1]
        assert "b4" not in forced[-1]


def _masked_beyond(image, centre, radius, eps=0.3):
    """Mask everything outside an elliptical radius, in pixels."""
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    r = np.sqrt((xx - centre) ** 2 + ((yy - centre) / (1.0 - eps)) ** 2)
    return r > radius


class TestForcedPhotometryRefusesToNormalizeByAFallback:
    """A harmonic is only reported when everything it depends on was measured.

    ``compute_gradient`` returns a ``-1.0`` sentinel when it cannot form a
    comparison ring, and ``compute_deviations`` returns zeros when its solve is
    singular or underdetermined. Both look like ordinary numbers. Dividing a
    real harmonic amplitude by a sentinel gradient produces a plausible,
    well-scaled, entirely meaningless coefficient -- the most dangerous kind of
    wrong answer, because nothing downstream can detect it.
    """

    def test_reports_nan_when_the_comparison_ring_is_unavailable(self):
        """A measured ring plus an unmeasurable gradient is still unmeasured."""
        image, centre = _planted_harmonic_image()
        # The ring at sma=44 is clean; its comparison ring at 44*(1+astep)
        # lies entirely inside the masked region, so no gradient exists.
        mask = _masked_beyond(image, centre, 45.0)
        config = _config(centre)

        result = extract_forced_photometry(image, mask, centre, centre, sma=44.0, eps=0.3, pa=0.0, config=config)

        assert np.isnan(result["a4"]), "harmonic normalized by a fallback gradient"
        assert np.isnan(result["b4"]), "harmonic normalized by a fallback gradient"

    def test_reports_nan_when_the_harmonic_solve_is_singular(self):
        """A singular solve must not be reported as a measured zero."""
        image, centre = _planted_harmonic_image()
        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
        theta = np.arctan2(yy - centre, xx - centre)
        mask = np.abs(((theta + np.pi) % (2.0 * np.pi)) - np.pi) > 0.02
        config = _config(centre)

        with pytest.warns(RuntimeWarning, match="compute_deviations"):
            result = extract_forced_photometry(image, mask, centre, centre, sma=30.0, eps=0.3, pa=0.0, config=config)

        assert np.isnan(result["a4"]), "singular solve reported as a4 = 0.0"
        assert np.isnan(result["b4"]), "singular solve reported as b4 = 0.0"

    def test_reports_nan_when_there_are_too_few_samples_to_solve(self):
        """Underdetermined is not the same as round, and must not look like it."""
        image, centre = _planted_harmonic_image()
        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
        theta = np.arctan2(yy - centre, xx - centre)
        mask = np.abs(((theta + np.pi) % (2.0 * np.pi)) - np.pi) > 0.05
        config = _config(centre)

        result = extract_forced_photometry(image, mask, centre, centre, sma=30.0, eps=0.3, pa=0.0, config=config)

        assert np.isnan(result["a4"]), "underdetermined solve reported as a4 = 0.0"
        assert np.isnan(result["b4"]), "underdetermined solve reported as b4 = 0.0"


class TestForcedPhotometryHonoursWhatTheCallerAsked:
    def test_gradient_follows_the_effective_arguments_when_config_is_omitted(self):
        """The Bender denominator must use the caller's ring statistic.

        The reported intensity follows the ``integrator`` argument, so the
        gradient that normalizes the harmonics has to follow it too. Otherwise
        numerator and denominator describe two different ring statistics.
        """
        image, centre = _planted_harmonic_image()

        without_config = extract_forced_photometry(
            image,
            None,
            centre,
            centre,
            sma=30.0,
            eps=0.3,
            pa=0.0,
            integrator="median",
        )
        with_config = extract_forced_photometry(
            image,
            None,
            centre,
            centre,
            sma=30.0,
            eps=0.3,
            pa=0.0,
            integrator="median",
            config=_config(centre, integrator="median"),
        )

        assert without_config["b4"] == pytest.approx(with_config["b4"], rel=1e-12), (
            "omitting config silently changed the gradient's ring statistic"
        )

    def test_warns_that_simultaneous_harmonics_do_not_apply_at_fixed_geometry(self):
        """Requesting a strategy that cannot run must not be silently downgraded."""
        image, centre = _planted_harmonic_image()
        config = _config(centre, compute_deviations=False, simultaneous_harmonics=True)
        template = [{"sma": s, "x0": centre, "y0": centre, "eps": 0.3, "pa": 0.0} for s in (15.0, 25.0, 35.0)]

        with pytest.warns(UserWarning, match="simultaneous_harmonics"):
            fit_image(image, config=config, template=template)
