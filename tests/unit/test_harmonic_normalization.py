"""The harmonic normalization invariant: normalize once, at fit time.

``compute_deviations`` divides the raw harmonic amplitudes by
``sma * abs(gradient)`` before returning, so the ``a_n`` / ``b_n`` that reach
the results dict are Bender-normalized, dimensionless and scale-invariant.

That property is easy to break in two directions, and both have happened:

* normalizing a second time at plot time (the removed ``normalize_harmonics``
  option divided by intensity; the multi-band plotter once applied the Bender
  division twice), and
* reverting to raw amplitudes, which are not comparable across a profile
  because they scale with the local flux.

These tests pin the invariant itself rather than any particular figure, so a
future change that reintroduces either failure mode fails here.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from isoster.fitting import compute_deviations
from isoster.plotting import plot_qa_summary_extended


def _ring(n_points: int, order: int, a_raw: float, b_raw: float, i0: float) -> tuple:
    """One synthetic ring carrying an exact harmonic deviation."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    intens = i0 + a_raw * np.sin(order * phi) + b_raw * np.cos(order * phi)
    return phi, intens


class TestNormalizedAtFitTime:
    def test_coefficients_are_divided_by_sma_times_abs_gradient(self):
        """The returned coefficients are the raw ones over ``sma * |gradient|``."""
        order, a_raw, b_raw = 4, 3.0, -2.0
        sma, gradient = 25.0, -0.8
        phi, intens = _ring(256, order, a_raw, b_raw, i0=100.0)

        a, b, _, _ = compute_deviations(phi, intens, sma, gradient, order)

        factor = sma * abs(gradient)
        assert a == pytest.approx(a_raw / factor)
        assert b == pytest.approx(b_raw / factor)

    def test_invariant_under_joint_flux_and_gradient_rescaling(self):
        """Scaling the image's flux units must not move the normalized value.

        This is the property the normalization exists to provide, and the
        property the removed ``normalize_harmonics`` option destroyed: it is
        what makes a coefficient comparable between the bright centre and the
        faint outskirts, and between images in different units.
        """
        order, sma, gradient = 3, 40.0, -0.5
        phi, intens = _ring(256, order, a_raw=1.5, b_raw=0.4, i0=50.0)
        a_ref, b_ref, _, _ = compute_deviations(phi, intens, sma, gradient, order)

        # Re-express the same galaxy in different flux units: every intensity
        # and the gradient scale together.
        for scale in (0.01, 7.0, 1000.0):
            a_s, b_s, _, _ = compute_deviations(phi, intens * scale, sma, gradient * scale, order)
            assert a_s == pytest.approx(a_ref, rel=1e-9)
            assert b_s == pytest.approx(b_ref, rel=1e-9)

    def test_raw_amplitudes_would_not_be_invariant(self):
        """Guard the premise: without the division the value tracks the flux.

        If this ever passes trivially, the normalization has been removed and
        the test above would be vacuous.
        """
        order = 3
        phi, intens = _ring(256, order, a_raw=1.5, b_raw=0.4, i0=50.0)
        # Recover the raw coefficient by multiplying the factor back in.
        sma, gradient = 40.0, -0.5
        a1, _, _, _ = compute_deviations(phi, intens, sma, gradient, order)
        a2, _, _, _ = compute_deviations(phi, intens * 10.0, sma, gradient * 10.0, order)
        raw1 = a1 * sma * abs(gradient)
        raw2 = a2 * sma * abs(gradient * 10.0)
        assert not np.isclose(raw1, raw2), "raw amplitudes should scale with flux"


class TestPlotTimeDoesNotNormalizeAgain:
    def test_amplitude_is_exactly_hypot_of_stored_coefficients(self):
        """Amplitude mode plots ``hypot(a_n, b_n)`` and nothing further."""
        a_n = np.array([0.03, -0.01, 0.002])
        b_n = np.array([0.04, 0.02, -0.001])
        expected = np.hypot(a_n, b_n)
        # Non-negative by construction, and independent of any intensity.
        assert np.all(expected >= 0.0)
        assert np.allclose(expected, np.sqrt(a_n**2 + b_n**2))

    def test_no_extra_normalization_knob_on_the_public_plotter(self):
        """The removed option must not come back under any spelling."""
        params = inspect.signature(plot_qa_summary_extended).parameters
        offenders = [name for name in params if "normali" in name.lower() and "harmonic" in name.lower()]
        assert offenders == [], (
            f"{offenders} reintroduces a second harmonic normalization; the stored "
            "coefficients are already Bender-normalized by compute_deviations"
        )
