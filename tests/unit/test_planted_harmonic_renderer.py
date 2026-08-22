"""A2: a Sersic renderer whose isophotes carry a *known* harmonic distortion.

The whole harmonic-scale calibration rests on knowing the right answer. Neither
existing generator can plant a harmonic, and the truth cannot come from the
first-order expansion in the design note -- Sersic curvature and products
between simultaneously planted modes add terms the linearization drops.

So the renderer is paired with a truth function that integrates the *analytic*
profile densely around the sampling ellipse. These tests check the two against
each other and against the linear approximation, and pin the regime where the
approximation is trustworthy so its error is reported rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.utils.sersic_model import (
    create_sersic_image_with_harmonics,
    integrated_harmonic_truth,
    linearized_harmonic_truth,
)

SHAPE = (241, 241)
CENTRE = (120.0, 120.0)
R_EFF = 25.0
SERSIC_N = 2.0
I_EFF = 100.0


def _params(eps=0.3, pa=0.0, harmonics=None):
    return dict(
        n=SERSIC_N,
        R_e=R_EFF,
        I_e=I_EFF,
        eps=eps,
        pa=pa,
        shape=SHAPE,
        center=CENTRE,
        harmonics=harmonics or {},
    )


class TestRenderer:
    def test_no_harmonics_reproduces_a_plain_ellipse(self):
        """The distortion must be an addition, not a change of the base model."""
        from benchmarks.utils.sersic_model import create_sersic_image_vectorized

        planted, _ = create_sersic_image_with_harmonics(**_params(harmonics={}))
        plain, _ = create_sersic_image_vectorized(
            n=SERSIC_N,
            R_e=R_EFF,
            I_e=I_EFF,
            eps=0.3,
            pa=0.0,
            shape=SHAPE,
            center=CENTRE,
            oversample=1,
        )
        assert np.allclose(planted, plain, rtol=1e-12, atol=1e-12)

    def test_the_planted_isophote_sits_where_it_was_planted(self):
        """The defining property: the level set of I(a) is r = a * D(phi)."""
        amplitude = 0.05
        image, meta = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): amplitude}))
        sma = 30.0
        target = meta["profile"](sma)

        # Walk out along phi=0 (where D is maximal) and phi=pi/4 (where the
        # n=4 cosine vanishes) and find where the image crosses that level.
        for phi, expected_scale in ((0.0, 1.0 + amplitude), (np.pi / 8.0, 1.0)):
            radii = np.linspace(sma * 0.7, sma * 1.3, 4001)
            x = CENTRE[0] + radii * np.cos(phi)
            y = CENTRE[1] + radii * (1.0 - 0.3) * np.sin(phi)
            values = meta["profile"](
                np.sqrt((x - CENTRE[0]) ** 2 + ((y - CENTRE[1]) / (1.0 - 0.3)) ** 2)
                / (1.0 + amplitude * np.cos(4.0 * phi))
            )
            crossing = radii[np.argmin(np.abs(values - target))]
            assert crossing == pytest.approx(sma * expected_scale, rel=2e-3), (
                f"planted isophote is not at a*D(phi) for phi={phi:.3f}"
            )

    def test_harmonics_are_keyed_by_order_and_kind(self):
        """sin and cos at the same order must be independent inputs."""
        cos_only, _ = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): 0.05}))
        sin_only, _ = create_sersic_image_with_harmonics(**_params(harmonics={(4, "sin"): 0.05}))
        assert not np.allclose(cos_only, sin_only)


class TestIntegratedTruth:
    def test_recovers_a_single_planted_cosine_mode(self):
        amplitude = 0.02
        _, meta = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): amplitude}))
        truth = integrated_harmonic_truth(meta, sma=30.0, orders=(3, 4))

        # b_n is the Bender-normalized cosine amplitude, which to first order
        # equals the planted epsilon.
        assert truth[4]["b_bender"] == pytest.approx(amplitude, rel=0.05)
        assert truth[4]["a_bender"] == pytest.approx(0.0, abs=1e-6)
        assert truth[3]["b_bender"] == pytest.approx(0.0, abs=1e-6)

    def test_recovers_a_single_planted_sine_mode(self):
        amplitude = 0.02
        _, meta = create_sersic_image_with_harmonics(**_params(harmonics={(4, "sin"): amplitude}))
        truth = integrated_harmonic_truth(meta, sma=30.0, orders=(3, 4))
        assert truth[4]["a_bender"] == pytest.approx(amplitude, rel=0.05)
        assert truth[4]["b_bender"] == pytest.approx(0.0, abs=1e-6)

    def test_reports_raw_amplitudes_consistent_with_its_own_bender_values(self):
        """The two tracks must describe one measurement, not two."""
        _, meta = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): 0.03}))
        truth = integrated_harmonic_truth(meta, sma=30.0, orders=(4,))
        entry = truth[4]
        factor = entry["sma"] * abs(entry["gradient"])
        assert entry["c_raw"] / factor == pytest.approx(entry["b_bender"], rel=1e-9)
        assert entry["s_raw"] / factor == pytest.approx(entry["a_bender"], rel=1e-9, abs=1e-12)

    def test_is_stable_against_refining_the_integration_grid(self):
        """A converged integral, not an artifact of the sample count."""
        _, meta = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): 0.03}))
        coarse = integrated_harmonic_truth(meta, sma=30.0, orders=(4,), n_phi=2048)[4]
        fine = integrated_harmonic_truth(meta, sma=30.0, orders=(4,), n_phi=16384)[4]
        assert fine["b_bender"] == pytest.approx(coarse["b_bender"], rel=1e-6)


class TestTheLinearApproximationIsReportedNotAssumed:
    def test_the_linear_estimate_is_simply_the_planted_amplitude(self):
        amplitude = 0.02
        _, meta = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): amplitude}))
        linear = linearized_harmonic_truth(meta, orders=(3, 4))
        assert linear[4]["b_bender"] == amplitude
        assert linear[4]["a_bender"] == 0.0

    def test_the_two_truths_diverge_measurably_at_large_amplitude(self):
        """If they never differed, integrating would be pointless."""
        _, small = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): 0.01}))
        _, large = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): 0.20}))

        def relative_gap(meta, amplitude):
            integrated = integrated_harmonic_truth(meta, sma=30.0, orders=(4,))[4]["b_bender"]
            return abs(integrated - amplitude) / amplitude

        assert relative_gap(small, 0.01) < relative_gap(large, 0.20)
        assert relative_gap(large, 0.20) > 1e-3, "no measurable non-linearity even at eps=0.2"

    def test_the_nonlinearity_is_small_in_the_calibration_regime(self):
        """Justifies the amplitudes the campaign plants."""
        for amplitude in (0.010, 0.015, 0.020, 0.030):
            _, meta = create_sersic_image_with_harmonics(**_params(harmonics={(4, "cos"): amplitude}))
            integrated = integrated_harmonic_truth(meta, sma=30.0, orders=(4,))[4]["b_bender"]
            assert abs(integrated - amplitude) / amplitude < 0.05, (
                f"planted amplitude {amplitude} is outside the near-linear regime"
            )


class TestThePlantingAngleIsPolarNotEccentricAnomaly:
    """Regression: the renderer once planted in the wrong angular variable.

    ``arctan2(y_rot / q, x_rot)`` circularizes the ellipse, so it is the
    *eccentric anomaly*, not the polar angle. The two coincide only for a
    circle, which is why the first version of the renderer looked right at
    eps=0 and was wrong everywhere else: the tools recovered 87% of the planted
    amplitude at eps=0.3 and 34% at eps=0.6, while the renderer and its own
    truth function agreed with each other because both used the wrong variable.

    The invariant below is what makes the bug visible: a planted epsilon is
    dimensionless and defined in the polar-angle basis, so the recovered
    Bender coefficient must not depend on the ellipticity of the host ellipse.
    """

    @pytest.mark.parametrize("eps", [0.0, 0.1, 0.3])
    def test_recovered_amplitude_does_not_depend_on_ellipticity(self, eps):
        amplitude = 0.02
        _, meta = create_sersic_image_with_harmonics(
            n=SERSIC_N,
            R_e=R_EFF,
            I_e=I_EFF,
            eps=eps,
            pa=0.0,
            shape=SHAPE,
            center=CENTRE,
            harmonics={(4, "cos"): amplitude},
        )
        recovered = integrated_harmonic_truth(meta, sma=30.0, orders=(4,))[4]["b_bender"]
        assert recovered == pytest.approx(amplitude, rel=0.05), (
            f"planted amplitude not recovered at eps={eps}; the planting angle "
            "is probably eccentric anomaly rather than polar angle"
        )

    def test_the_eccentric_anomaly_basis_would_give_a_different_answer(self):
        """Guards the test above from passing for a trivial reason."""
        eps = 0.3
        q = 1.0 - eps
        phi = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
        # Eccentric anomaly corresponding to each polar angle on this ellipse.
        psi = np.arctan2(np.sin(phi) / q, np.cos(phi))
        assert np.max(np.abs(psi - phi)) > 0.1, "polar angle and eccentric anomaly should differ materially at eps=0.3"
