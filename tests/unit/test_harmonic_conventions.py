"""A1: what each tool means by ``a_n`` and ``b_n``, measured on known rings.

This is the cheapest and most conclusive level of the harmonic-scale work
(``docs/specs/2026-08-22-three-way-benchmark-comparison-design.md``). It feeds
exactly-known ring samples to each tool's harmonic implementation and compares
against closed-form truth. No galaxy, no noise, no geometry fitting, no image:
whatever these tests find is a convention difference and nothing else.

Two of the three arms call the real library. The AutoProf arm reimplements its
four-line FFT expression, because AutoProf cannot be imported into this
environment (it pins ``numpy<2``). That is a deliberate limitation of a unit
test, and it is why the integration test pins the *installed* AutoProf's output
separately -- these tests would not notice an upstream convention change.

The response matrix is the point. Injecting one mode at a time and recording
the full matrix separates a sign flip, a phase rotation and cross-order leakage
from one another; injecting several modes at once lets them hide inside each
other.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.harmonic_scale.conventions import (
    HARMONIC_KINDS,
    autoprof_native_coefficients,
    bender_coefficients_isoster,
    bender_coefficients_photutils,
    raw_from_autoprof,
    raw_from_bender,
    rotate_raw_to_major_axis,
)

ORDERS = (3, 4)
N_SAMPLES = 256
BASE_INTENSITY = 100.0
UNIT_AMPLITUDE = 1.0

#: Numerical floor of each tool's harmonic solve, as a fraction of the ring's
#: pedestal intensity. Measured, not guessed -- see ``TestSolverPrecision``.
#:
#: isoster (``np.linalg.lstsq``) and the AutoProf FFT expression are direct
#: solves and land near machine precision. photutils runs
#: ``scipy.optimize.leastsq`` -- an *iterative* optimizer -- on a problem that
#: is exactly linear in its three parameters, so it converges to a relative
#: tolerance against the DC term rather than solving exactly. That costs about
#: seven orders of magnitude of precision. It is far below any noise level in
#: real data and is not a defect, but a shared 1e-9 tolerance would fail on it
#: for reasons that have nothing to do with convention.
SOLVER_FLOOR = {"isoster": 1.0e-11, "photutils": 5.0e-9, "autoprof": 1.0e-11}


def _tolerance(tool, base=BASE_INTENSITY):
    """Absolute tolerance for ``tool`` on a ring with the given pedestal."""
    return SOLVER_FLOOR[tool] * base


def _ring(order, kind, amplitude=UNIT_AMPLITUDE, base=BASE_INTENSITY, n_samples=N_SAMPLES, offset=0.0):
    """One exactly-known ring: a single injected harmonic on a flat pedestal."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    wave = np.sin(order * phi) if kind == "sin" else np.cos(order * phi)
    return phi, base + offset + amplitude * wave


def _raw_by_tool(phi, intensities, order):
    """Raw (S_n, C_n) as each tool would report them, reconstructed to a common scale.

    ``sma=1`` and ``gradient=-1`` make the Bender normalization factor
    ``sma * |gradient|`` exactly 1, so the Bender arms return raw amplitudes
    untouched. The AutoProf arm is reconstructed through its own ``b0``.
    """
    iso_a, iso_b = bender_coefficients_isoster(phi, intensities, sma=1.0, gradient=-1.0, order=order)
    pht_a, pht_b = bender_coefficients_photutils(phi, intensities, sma=1.0, gradient=-1.0, order=order)
    native = autoprof_native_coefficients(intensities, ORDERS)
    ap_s, ap_c = raw_from_autoprof(native[f"a{order}"], native[f"b{order}"], native["b0"])
    return {
        "isoster": raw_from_bender(iso_a, iso_b, sma=1.0, gradient=-1.0),
        "photutils": raw_from_bender(pht_a, pht_b, sma=1.0, gradient=-1.0),
        "autoprof": (ap_s, ap_c),
    }


class TestResponseMatrix:
    """Inject one mode at a time; every tool must see that mode and no other."""

    @pytest.mark.parametrize("tool", ["isoster", "photutils", "autoprof"])
    def test_diagonal_response_recovers_the_injected_amplitude(self, tool):
        for order in ORDERS:
            for kind in HARMONIC_KINDS:
                phi, intensities = _ring(order, kind)
                s_n, c_n = _raw_by_tool(phi, intensities, order)[tool]
                measured = s_n if kind == "sin" else c_n
                assert measured == pytest.approx(UNIT_AMPLITUDE, abs=_tolerance(tool)), (
                    f"{tool} did not recover the injected {kind} amplitude at n={order}"
                )

    @pytest.mark.parametrize("tool", ["isoster", "photutils", "autoprof"])
    def test_no_leakage_into_the_orthogonal_kind_or_the_other_order(self, tool):
        for injected_order in ORDERS:
            for injected_kind in HARMONIC_KINDS:
                phi, intensities = _ring(injected_order, injected_kind)
                for measured_order in ORDERS:
                    s_n, c_n = _raw_by_tool(phi, intensities, measured_order)[tool]
                    for measured_kind, value in (("sin", s_n), ("cos", c_n)):
                        if measured_order == injected_order and measured_kind == injected_kind:
                            continue
                        assert value == pytest.approx(0.0, abs=_tolerance(tool)), (
                            f"{tool}: injecting {injected_kind} at n={injected_order} leaked "
                            f"{value:.3e} into {measured_kind} at n={measured_order}"
                        )

    def test_all_three_tools_agree_on_the_same_ring(self):
        """Raw amplitudes are the common scale; on one ring they must coincide."""
        phi, intensities = _ring(4, "cos", amplitude=2.5)
        by_tool = _raw_by_tool(phi, intensities, 4)
        reference = by_tool["isoster"]
        for tool, (s_n, c_n) in by_tool.items():
            atol = _tolerance(tool)
            assert s_n == pytest.approx(reference[0], abs=atol), f"{tool} S_4 disagrees"
            assert c_n == pytest.approx(reference[1], abs=atol), f"{tool} C_4 disagrees"

    def test_superposition_holds_for_all_four_modes_at_once(self):
        """The combined case must be the sum of the individual responses."""
        phi = np.linspace(0.0, 2.0 * np.pi, N_SAMPLES, endpoint=False)
        planted = {(3, "sin"): 0.015, (3, "cos"): 0.030, (4, "sin"): 0.010, (4, "cos"): 0.020}
        combined = np.full_like(phi, BASE_INTENSITY)
        for (order, kind), amplitude in planted.items():
            wave = np.sin(order * phi) if kind == "sin" else np.cos(order * phi)
            combined = combined + amplitude * wave

        for order in ORDERS:
            s_n, c_n = _raw_by_tool(phi, combined, order)["isoster"]
            atol = _tolerance("isoster")
            assert s_n == pytest.approx(planted[(order, "sin")], abs=atol)
            assert c_n == pytest.approx(planted[(order, "cos")], abs=atol)


class TestAutoProfDiffersInThreeSpecificWays:
    """The differences the spec derived, each pinned separately."""

    def test_a_is_the_negated_sine_coefficient(self):
        phi, intensities = _ring(4, "sin", amplitude=1.0)
        native = autoprof_native_coefficients(intensities, ORDERS)
        assert native["a4"] < 0.0, "AutoProf's a_n should carry the opposite sign to the sine amplitude"

    def test_b_is_the_cosine_coefficient_and_is_not_negated(self):
        phi, intensities = _ring(4, "cos", amplitude=1.0)
        native = autoprof_native_coefficients(intensities, ORDERS)
        assert native["b4"] > 0.0, "AutoProf's b_n should share the sign of the cosine amplitude"

    def test_the_normalization_is_twice_the_mean_intensity(self):
        amplitude = 3.0
        phi, intensities = _ring(4, "cos", amplitude=amplitude, base=BASE_INTENSITY)
        native = autoprof_native_coefficients(intensities, ORDERS)
        assert native["b0"] == pytest.approx(BASE_INTENSITY, abs=1e-9)
        assert native["b4"] == pytest.approx(amplitude / (2.0 * BASE_INTENSITY), rel=1e-9)

    def test_b0_is_the_exact_mean_of_the_fft_input(self):
        """The raw reconstruction depends on this, so it is pinned directly."""
        phi, intensities = _ring(3, "sin", amplitude=0.7, base=42.0)
        native = autoprof_native_coefficients(intensities, ORDERS)
        assert native["b0"] == pytest.approx(float(np.mean(intensities)), rel=1e-12)


class TestSensitivityToFluxScaleAndBackground:
    """The corrected claim: both conventions are scale-invariant; only one is offset-sensitive."""

    @pytest.mark.parametrize("factor", [1.0, 10.0, 1.0e6])
    def test_autoprof_coefficients_are_invariant_under_multiplicative_scaling(self, factor):
        phi, intensities = _ring(4, "cos", amplitude=2.0)
        reference = autoprof_native_coefficients(intensities, ORDERS)
        scaled = autoprof_native_coefficients(intensities * factor, ORDERS)
        assert scaled["b4"] == pytest.approx(reference["b4"], rel=1e-12)
        assert scaled["a4"] == pytest.approx(reference["a4"], abs=1e-12)

    def test_autoprof_coefficients_move_under_an_additive_background_offset(self):
        phi, intensities = _ring(4, "cos", amplitude=2.0)
        reference = autoprof_native_coefficients(intensities, ORDERS)
        offset = autoprof_native_coefficients(intensities + BASE_INTENSITY, ORDERS)
        expected = reference["b4"] * BASE_INTENSITY / (2.0 * BASE_INTENSITY)
        assert offset["b4"] == pytest.approx(expected, rel=1e-9)
        assert offset["b4"] != pytest.approx(reference["b4"], rel=1e-6)

    def test_bender_coefficients_are_unmoved_by_an_additive_background_offset(self):
        """A constant does not change the radial gradient, so Bender does not care."""
        phi, intensities = _ring(4, "cos", amplitude=2.0)
        reference = bender_coefficients_isoster(phi, intensities, sma=1.0, gradient=-1.0, order=4)
        shifted = bender_coefficients_isoster(phi, intensities + BASE_INTENSITY, sma=1.0, gradient=-1.0, order=4)
        assert shifted[0] == pytest.approx(reference[0], abs=1e-9)
        assert shifted[1] == pytest.approx(reference[1], abs=1e-9)

    def test_the_raw_reconstruction_survives_the_offset_that_moved_the_native_values(self):
        """b0 moves with the offset, and the reconstruction divides it back out."""
        phi, intensities = _ring(4, "cos", amplitude=2.0)
        shifted = intensities + BASE_INTENSITY
        native = autoprof_native_coefficients(shifted, ORDERS)
        _, c_4 = raw_from_autoprof(native["a4"], native["b4"], native["b0"])
        assert c_4 == pytest.approx(2.0, abs=1e-9)


class TestPositionAngleRotation:
    """The formula the AutoProf polar-resampled path needs, pinned rather than described."""

    @pytest.mark.parametrize("pa_deg", [0.0, 30.0, 57.3, 90.0, -45.0])
    @pytest.mark.parametrize("order", ORDERS)
    def test_rotation_recovers_the_major_axis_coefficients(self, pa_deg, order):
        """A signal defined on the major axis, sampled in a frame offset by PA."""
        pa = np.deg2rad(pa_deg)
        s_major, c_major = 0.015, 0.030

        # Sample against theta_sky = theta_major + PA, which is what AutoProf's
        # polar resampling does. In that frame the same physical signal has
        # rotated coefficients; the helper must undo exactly that.
        theta_sky = np.linspace(0.0, 2.0 * np.pi, N_SAMPLES, endpoint=False)
        theta_major = theta_sky - pa
        intensities = BASE_INTENSITY + s_major * np.sin(order * theta_major) + c_major * np.cos(order * theta_major)

        native = autoprof_native_coefficients(intensities, ORDERS)
        s_sky, c_sky = raw_from_autoprof(native[f"a{order}"], native[f"b{order}"], native["b0"])
        s_rot, c_rot = rotate_raw_to_major_axis(s_sky, c_sky, order=order, pa_rad=pa)

        assert s_rot == pytest.approx(s_major, abs=1e-9), "rotation did not recover S on the major axis"
        assert c_rot == pytest.approx(c_major, abs=1e-9), "rotation did not recover C on the major axis"

    def test_rotation_is_the_identity_at_zero_position_angle(self):
        s_rot, c_rot = rotate_raw_to_major_axis(0.4, -0.2, order=4, pa_rad=0.0)
        assert s_rot == pytest.approx(0.4, abs=1e-12)
        assert c_rot == pytest.approx(-0.2, abs=1e-12)

    def test_an_unrotated_conversion_would_be_wrong_at_nonzero_pa(self):
        """Guards the test above from passing for the trivial reason."""
        pa, order = np.deg2rad(30.0), 4
        s_major, c_major = 0.015, 0.030
        theta_sky = np.linspace(0.0, 2.0 * np.pi, N_SAMPLES, endpoint=False)
        theta_major = theta_sky - pa
        intensities = BASE_INTENSITY + s_major * np.sin(order * theta_major) + c_major * np.cos(order * theta_major)
        native = autoprof_native_coefficients(intensities, ORDERS)
        s_sky, c_sky = raw_from_autoprof(native[f"a{order}"], native[f"b{order}"], native["b0"])
        assert abs(s_sky - s_major) > 1e-3, "PA=30 deg should visibly rotate the coefficients"


class TestSolverPrecision:
    """Characterization, not a feature test: pins the measured numerical floors.

    These pass against today's code by construction — that is the point. They
    exist so that ``SOLVER_FLOOR`` above is a recorded measurement rather than a
    number someone tuned until the suite went green, and so a change in either
    direction is visible. If photutils ever switches to a direct solve, the
    ``photutils`` case here fails and the tolerance can be tightened.
    """

    @pytest.mark.parametrize("tool", ["isoster", "autoprof"])
    def test_the_direct_solvers_reach_near_machine_precision(self, tool):
        phi, intensities = _ring(4, "cos", amplitude=2.5, base=1.0e4)
        s_n, c_n = _raw_by_tool(phi, intensities, 4)[tool]
        error = max(abs(s_n), abs(c_n - 2.5))
        assert error < 1.0e-11 * 1.0e4, f"{tool} is no longer a direct solve: error {error:.3e}"

    def test_photutils_error_scales_with_the_pedestal_not_the_amplitude(self):
        """The signature of an iterative solver converging against the DC term."""
        errors = {}
        for base in (1.0e2, 1.0e4):
            phi, intensities = _ring(4, "cos", amplitude=2.5, base=base)
            s_n, c_n = _raw_by_tool(phi, intensities, 4)["photutils"]
            errors[base] = max(abs(s_n), abs(c_n - 2.5))

        # Two decades more pedestal, at fixed amplitude, costs ~two decades of
        # absolute accuracy. An amplitude-driven error would not do this.
        ratio = errors[1.0e4] / errors[1.0e2]
        assert 10.0 < ratio < 1000.0, f"unexpected photutils error scaling: ratio {ratio:.1f}"
        assert errors[1.0e4] < SOLVER_FLOOR["photutils"] * 1.0e4, "photutils floor has regressed"
