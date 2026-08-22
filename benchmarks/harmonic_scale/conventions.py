"""One place that knows what each tool means by ``a_n`` and ``b_n``.

isoster and photutils share a convention exactly. AutoProf differs in four
ways, and this module is where those four differences live so that no caller
has to rediscover them:

1. **Normalization.** AutoProf divides by twice the mean ring intensity;
   isoster and photutils divide by ``sma * |dI/da|`` (Bender). Both are
   invariant to multiplicative flux scaling -- numerator and denominator scale
   together in each case. They differ in response to an *additive* background
   error, which moves AutoProf's denominator and leaves the radial gradient
   alone.
2. **A factor of 2**, from the FFT convention.
3. **A sign flip on ``a`` only.** AutoProf's ``a`` is the negated sine
   coefficient; its ``b`` is the cosine coefficient, unnegated.
4. **The angle basis, chosen at runtime.** With a mask or ``ap_isoclip=True``,
   AutoProf re-interpolates onto polar angle measured from the image x-axis,
   which is offset from the major axis by the position angle. That offset
   rotates ``(S_n, C_n)`` by ``n * PA``.

Everything here works in **raw amplitudes** -- the sine and cosine amplitudes
of the ring signal in image intensity units, before any normalization. Raw
amplitudes are the common scale on which the three tools can be compared,
because reaching them from each tool's output is exact. They are *not*
portable across different images, intensity units or apertures; see the
design note for why the calibration fixes geometry.

Sign and scale conventions used throughout, for a ring signal

    I(phi) = I0 + S_n * sin(n phi) + C_n * cos(n phi)

* isoster / photutils:  ``a_n = S_n / (sma |dI/da|)``, ``b_n = C_n / (sma |dI/da|)``
* AutoProf:             ``a_n = -S_n / (2 |b0|)``,     ``b_n = +C_n / (2 |b0|)``
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

#: The two harmonic components, in the order used by every response matrix here.
HARMONIC_KINDS: tuple[str, str] = ("sin", "cos")


def autoprof_native_coefficients(
    intensities: Sequence[float],
    orders: Iterable[int],
) -> dict[str, float]:
    """Reproduce AutoProf's harmonic expression on an already-sampled ring.

    Mirrors ``autoprof/pipeline_steps/Isophote_Extract.py:170-181`` exactly::

        coefs = fft(intensities)
        a_n   = Im(coefs[n]) / |coefs[0]|
        b_n   = Re(coefs[n]) / |coefs[0]|
        a_0   = Im(coefs[0]) / N          (identically zero for real input)
        b_0   = Re(coefs[0]) / N          (the mean of the input vector)

    This is a reimplementation, not a call into AutoProf, because AutoProf pins
    ``numpy<2`` and cannot be imported alongside isoster. It is therefore blind
    to an upstream change: the integration test pins the *installed* AutoProf's
    columns separately, and the archive records its version and source digest.

    Args:
        intensities: The ring samples, in the order AutoProf would FFT them.
        orders: Harmonic orders to report, e.g. ``(3, 4)``.

    Returns:
        ``{"a0": ..., "b0": ..., "a3": ..., "b3": ..., ...}`` in AutoProf's
        native scale.
    """
    values = np.asarray(intensities, dtype=np.float64)
    coefficients = np.fft.fft(values)
    dc_magnitude = np.abs(coefficients[0])

    result = {
        "a0": float(np.imag(coefficients[0]) / len(coefficients)),
        "b0": float(np.real(coefficients[0]) / len(coefficients)),
    }
    for order in orders:
        result[f"a{order}"] = float(np.imag(coefficients[order]) / dc_magnitude)
        result[f"b{order}"] = float(np.real(coefficients[order]) / dc_magnitude)
    return result


def raw_from_autoprof(a_n: float, b_n: float, b0: float) -> tuple[float, float]:
    """Recover raw ``(S_n, C_n)`` from AutoProf's native coefficients.

    Exact, not approximate: ``b0`` is the mean of the very vector that entered
    the FFT, so no estimate of the ring's central intensity is needed. An
    earlier draft of the design proposed deriving it from the ``SB`` column and
    carried a mean-versus-median error term; ``b0`` removes that entirely.

    ``|b0|`` rather than ``b0`` because AutoProf divides by ``|coefs[0]|``. A
    noisy outer ring can have a non-positive mean, and using the signed value
    would flip the reconstructed sign exactly where the data are worst.
    """
    scale = 2.0 * abs(b0)
    return -scale * a_n, scale * b_n


def raw_from_bender(a_n: float, b_n: float, sma: float, gradient: float) -> tuple[float, float]:
    """Recover raw ``(S_n, C_n)`` from Bender-normalized coefficients.

    Inverts ``a_n = S_n / (sma * |gradient|)``. Exact for isoster and photutils
    alike, since both store that same normalization.
    """
    factor = sma * abs(gradient)
    return a_n * factor, b_n * factor


def bender_from_raw(s_n: float, c_n: float, sma: float, gradient: float) -> tuple[float, float]:
    """Normalize raw amplitudes the way isoster and photutils do."""
    factor = sma * abs(gradient)
    if factor == 0:
        return float("nan"), float("nan")
    return s_n / factor, c_n / factor


def bender_coefficients_isoster(
    phi: Sequence[float],
    intensities: Sequence[float],
    sma: float,
    gradient: float,
    order: int,
) -> tuple[float, float]:
    """isoster's own harmonic solve, called directly on a supplied ring."""
    from isoster.fitting import compute_deviations

    a_n, b_n, _, _ = compute_deviations(
        np.asarray(phi, dtype=np.float64),
        np.asarray(intensities, dtype=np.float64),
        sma,
        gradient,
        order,
    )
    return float(a_n), float(b_n)


def bender_coefficients_photutils(
    phi: Sequence[float],
    intensities: Sequence[float],
    sma: float,
    gradient: float,
    order: int,
) -> tuple[float, float]:
    """photutils' own harmonic solve, normalized as photutils normalizes it.

    ``photutils/isophote/isophote.py::_compute_deviations`` divides
    ``fit_upper_harmonic``'s coefficients by ``sma`` and ``abs(gradient)``; we
    apply the same two divisions here so the returned pair is on the same scale
    as :func:`bender_coefficients_isoster`.
    """
    from photutils.isophote.harmonics import fit_upper_harmonic

    coefficients, _ = fit_upper_harmonic(
        np.asarray(phi, dtype=np.float64),
        np.asarray(intensities, dtype=np.float64),
        order,
    )
    factor = sma * abs(gradient)
    return float(coefficients[1] / factor), float(coefficients[2] / factor)


#: The gradient interval isoster and photutils both use, as a fraction of the
#: semi-major axis. Matches ``IsosterConfig.astep``'s default.
DEFAULT_GRADIENT_STEP = 0.1


def matched_secant_gradient(
    profile_at_sma: float,
    profile_at_comparison: float,
    sma: float,
    step: float = DEFAULT_GRADIENT_STEP,
) -> float:
    """The gradient isoster means, computed from any tool's radial profile.

    isoster and photutils do not divide by ``dI/da``. They divide by a forward
    secant over ``sma -> sma*(1 + astep)``::

        gradient = (mean(sma * (1 + astep)) - mean(sma)) / (sma * astep)

    On the Part A fixture that sits 11-14% below the point derivative at every
    radius, systematically. Reconstructing AutoProf's denominator as a point
    derivative would therefore have made its Bender coefficients differ from
    isoster's by that 12% *by construction*, and a definition mismatch would
    have been reported as a disagreement between tools.

    What is shared here is the **interval**, which both tools must use or the
    comparison is meaningless. What is measured is the **value**, taken from
    the supplying tool's own profile with nothing from isoster entering it.

    Args:
        profile_at_sma: The tool's ring statistic at ``sma``. For AutoProf this
            must be ``b0`` -- the mean of the exact vector that entered the
            FFT, and so the estimator consistent with the harmonic numerator.
            Its ``I``/``SB`` column is a *median* and is the wrong choice.
        profile_at_comparison: The same statistic at ``sma * (1 + step)``.
        sma: Semi-major axis of the measured ring.
        step: The interval, which must match the config being compared against.

    Returns:
        The secant gradient, or NaN if the interval is degenerate.
    """
    delta_r = sma * step
    if delta_r == 0 or not np.isfinite(delta_r):
        return float("nan")
    return float((profile_at_comparison - profile_at_sma) / delta_r)


def comparison_radius(sma: float, step: float = DEFAULT_GRADIENT_STEP) -> float:
    """The ring isoster pairs with ``sma`` to form its gradient.

    Exposed so a campaign can request both rings explicitly: the reconstruction
    needs the tool measured at both radii, which doubles the ring count.
    """
    return float(sma) * (1.0 + float(step))


def rotate_raw_to_major_axis(
    s_sky: float,
    c_sky: float,
    order: int,
    pa_rad: float,
) -> tuple[float, float]:
    """Rotate raw coefficients from the sky frame into the major-axis frame.

    AutoProf's polar-resampled path measures against ``theta_sky =
    theta_major + PA`` (``SharedFunctions.py:640``). Substituting that into
    ``S sin(n theta_sky) + C cos(n theta_sky)`` and collecting terms in
    ``theta_major`` gives, with ``alpha = n * PA``::

        S_major = S_sky cos(alpha) - C_sky sin(alpha)
        C_major = S_sky sin(alpha) + C_sky cos(alpha)

    Only valid for the polar-resampled path. The eccentric-anomaly path is a
    *different basis*, not a rotated one: changing between them mixes harmonic
    orders, so no same-order two-component rotation can express it. Convert
    that path by resampling the ring signal, or leave it native and labelled.
    """
    alpha = order * pa_rad
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    return (
        float(s_sky * cos_alpha - c_sky * sin_alpha),
        float(s_sky * sin_alpha + c_sky * cos_alpha),
    )
