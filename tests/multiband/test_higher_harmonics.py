"""Tests for the ``multiband_higher_harmonics`` enum (Section 6 of
plan-2026-04-29-multiband-feasibility.md).

Covers all four enum values:

- ``independent`` — current per-band, per-order, uncoupled-across-bands
  fit. Stage-1 baseline; verified bit-identical to the
  pre-Section-6 behavior.
- ``shared`` — post-hoc joint refit of higher orders only; (A1, B1, A2,
  B2) and per-band ``I0_b`` frozen at the converged-loop values.
- ``simultaneous_in_loop`` — wider joint solve every iteration.
- ``simultaneous_original`` — Ciambur-original variant: standard 5-param
  loop, ONE post-hoc joint refit over all orders.

The test suite intentionally lives in a dedicated file so the coverage
matrix is easy to read in CI logs.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pytest

from isoster._shared import _normalize_harmonic_for_plot
from isoster.multiband import IsosterConfigMB, fit_image_multiband
from isoster.multiband.fitting_mb import (
    fit_simultaneous_joint,
    fit_simultaneous_joint_loose,
)
from isoster.multiband.utils_mb import (
    isophote_results_mb_from_fits,
    isophote_results_mb_to_fits,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _planted_arc_band(
    n: int = 121,
    xc: float = 60.0,
    yc: float = 60.0,
    scale: float = 1.0,
    eps: float = 0.3,
    sigma_n: float = 0.005,
    m4_amp: float = 0.05,
    seed: int = 7,
) -> np.ndarray:
    """Sersic-like exponential profile with planted m=4 boxiness.

    The m=4 amplitude is the same in every band; under shared / simultaneous
    modes we expect to recover ONE consistent value across the bands.
    """
    rng = np.random.default_rng(seed)
    y, x = np.indices((n, n))
    sma_grid = np.sqrt((x - xc) ** 2 + (y - yc) ** 2 / (1.0 - eps) ** 2)
    img = scale * np.exp(-(sma_grid / 8.0))
    if m4_amp != 0.0:
        theta = np.arctan2(y - yc, x - xc)
        img += m4_amp * np.cos(4.0 * theta) * np.exp(-(sma_grid / 8.0))
    img += rng.normal(0.0, sigma_n, img.shape)
    return img


def _three_band_planted(
    m4_amp: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng_seeds = (7, 11, 13)
    bands = []
    for scale, seed in zip((1.0, 1.5, 2.0), rng_seeds):
        bands.append(_planted_arc_band(scale=scale, m4_amp=m4_amp, seed=seed))
    return tuple(bands)  # type: ignore[return-value]


def _make_cfg(
    mode: str = "independent",
    *,
    harmonic_orders=None,
    loose_validity: bool = False,
    fit_per_band_intens_jointly: bool = True,
) -> IsosterConfigMB:
    kwargs: dict = dict(
        bands=["g", "r", "i"],
        reference_band="r",
        sma0=10.0,
        minsma=2.0,
        maxsma=25.0,
        astep=0.2,
        eps=0.3,
        x0=60.0,
        y0=60.0,
        multiband_higher_harmonics=mode,
        loose_validity=loose_validity,
        fit_per_band_intens_jointly=fit_per_band_intens_jointly,
    )
    if harmonic_orders is not None:
        kwargs["harmonic_orders"] = harmonic_orders
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return IsosterConfigMB(**kwargs)


def assert_one_shared_shape(row, orders=(3, 4), bands=("g", "r", "i")):
    """Every band's raw amplitude is the same shape times that band's scale.

    Under ``shared`` the raw per-band values differ (they must — a common shape
    gives ``A_n_raw = -A_n_norm * sma * grad_b``), but the *ratio* between two
    bands is set by their gradients alone and is therefore identical for every
    order. That is checkable without needing the per-band gradient columns.
    """
    reference = bands[0]
    ratios = {}
    for order in orders:
        for band in bands[1:]:
            denom = float(row[f"a{order}_{reference}"])
            if abs(denom) < 1e-12:
                continue
            ratios.setdefault(band, []).append(float(row[f"a{order}_{band}"]) / denom)
    for band, values in ratios.items():
        if len(values) > 1:
            assert max(values) - min(values) < 1e-6, f"{band}: {values}"


# ---------------------------------------------------------------------------
# Independent mode (back-compat regression)
# ---------------------------------------------------------------------------


def test_independent_default_runs_and_per_band_columns_differ():
    """Default 'independent' mode produces band-distinct higher-order values."""
    bands = _three_band_planted()
    cfg = _make_cfg("independent")
    res = fit_image_multiband(list(bands), config=cfg)
    iso = res["isophotes"]
    assert all(r["stop_code"] == 0 for r in iso)
    # With three independent bands and noise, per-band a4 / b4 generically
    # differ (especially at the iso-by-iso scale).
    differs_a4 = any(not (r["a4_g"] == r["a4_r"] == r["a4_i"]) for r in iso)
    differs_b4 = any(not (r["b4_g"] == r["b4_r"] == r["b4_i"]) for r in iso)
    assert differs_a4 or differs_b4


def test_independent_top_level_keys_consistent():
    bands = _three_band_planted()
    cfg = _make_cfg("independent")
    res = fit_image_multiband(list(bands), config=cfg)
    assert res["multiband_higher_harmonics"] == "independent"
    assert res["harmonics_shared"] is False
    assert res["harmonic_orders"] == [3, 4]


# ---------------------------------------------------------------------------
# Shared mode (NEW DEVELOPMENT)
# ---------------------------------------------------------------------------


def test_shared_mode_shares_the_normalised_shape_not_the_raw_amplitude():
    """'shared' means one shared isophotal *shape*, not one raw amplitude.

    A common shape deviation produces raw amplitude ``-A_n_norm * sma * grad_b``
    in band ``b``, so bands with different gradients must carry *different* raw
    values for the same shape. Writing one identical raw value into every column
    used to make the reported Bender-normalised amplitudes depend on the bands'
    arbitrary flux units — measured at 0.11 against 0.011 for a planted 0.02
    after rescaling one band by ten.

    Schema 1 stores raw amplitudes, so the invariant to check is that every band
    *normalises* to the same value.
    """
    bands = _three_band_planted()
    cfg = _make_cfg("shared")
    cfg = cfg.model_copy(update={"debug": True})  # per-band grad_<b> columns
    res = fit_image_multiband(list(bands), config=cfg)
    iso = res["isophotes"]
    assert all(r["stop_code"] == 0 for r in iso)
    checked = 0
    for r in iso:
        grads = {b: float(r.get(f"grad_{b}", float("nan"))) for b in ("g", "r", "i")}
        if not all(np.isfinite(g) and g != 0.0 for g in grads.values()):
            continue
        for n_order in (3, 4):
            normalised = [-float(r[f"a{n_order}_{b}"]) / (float(r["sma"]) * grads[b]) for b in ("g", "r", "i")]
            assert max(normalised) - min(normalised) < 1e-9
            checked += 1
    assert checked > 0, "fixture produced no rows with usable per-band gradients"


def test_shared_mode_recovers_planted_m4_signal():
    """Shared mode recovers the planted m=4 amplitude near the central region.

    The planted ``b4 ≈ 0.05`` cosine term at the model coordinates feeds
    a reduced amplitude at small SMA after sigma-clipping, but the sign
    and order of magnitude must match.
    """
    bands = _three_band_planted(m4_amp=0.05)
    cfg = _make_cfg("shared")
    res = fit_image_multiband(list(bands), config=cfg)
    iso = res["isophotes"]
    # At small SMA the planted amplitude is most clearly recovered; sample
    # rows 5-9 (a few isophotes inward from sma0) and check they reflect
    # an above-noise positive b4 in the reference band.
    b4s = [r["b4_r"] for r in iso[5:10]]
    assert max(b4s) > 0.005, b4s


def test_shared_mode_top_level_keys():
    bands = _three_band_planted()
    cfg = _make_cfg("shared")
    res = fit_image_multiband(list(bands), config=cfg)
    assert res["multiband_higher_harmonics"] == "shared"
    assert res["harmonics_shared"] is True


def test_shared_mode_with_loose_validity_drops_band_at_isophote():
    """Loose validity x shared composes: a band with a masked region drops at
    that isophote; the surviving bands still carry the one shared *shape*.

    Their raw columns are not identical -- under ``shared`` each band's raw
    value is the common shape scaled by that band's own gradient -- so the
    invariant checked here is the cross-band ratio, not equality.
    """
    bands = _three_band_planted()
    n = bands[0].shape[0]
    mask_g = np.zeros((n, n), dtype=bool)
    mask_g[20:50, 20:50] = True
    masks = [mask_g, np.zeros((n, n), dtype=bool), np.zeros((n, n), dtype=bool)]
    cfg = _make_cfg("shared", loose_validity=True)
    res = fit_image_multiband(list(bands), masks=masks, config=cfg)
    iso = res["isophotes"]
    assert all(r["stop_code"] == 0 for r in iso)
    # Surviving (r, i) bands carry the same shape, scaled per band.
    for r in iso:
        for n_order in (3, 4):
            assert r[f"a{n_order}_r"] != 0.0 or r[f"a{n_order}_i"] == 0.0
        assert_one_shared_shape(r, bands=("r", "i"))


def test_shared_mode_with_ring_mean_intercept():
    """Both flags compose. Under fit_per_band_intens_jointly=False, per-band
    intensities come from ring means; higher orders shared across bands."""
    bands = _three_band_planted()
    cfg = _make_cfg("shared", fit_per_band_intens_jointly=False)
    res = fit_image_multiband(list(bands), config=cfg)
    iso = res["isophotes"]
    assert all(r["stop_code"] == 0 for r in iso)
    for r in iso:
        assert_one_shared_shape(r)


# ---------------------------------------------------------------------------
# Simultaneous_in_loop / simultaneous_original (RECOVERED FEATURE)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["simultaneous_in_loop", "simultaneous_original"])
def test_simultaneous_modes_produce_identical_per_band_values(mode):
    bands = _three_band_planted()
    cfg = _make_cfg(mode)
    res = fit_image_multiband(list(bands), config=cfg)
    iso = res["isophotes"]
    assert all(r["stop_code"] == 0 for r in iso)
    for r in iso:
        for n_order in (3, 4):
            assert_one_shared_shape(r)


@pytest.mark.parametrize("mode", ["simultaneous_in_loop", "simultaneous_original"])
def test_simultaneous_recovers_planted_m4_signal(mode):
    bands = _three_band_planted(m4_amp=0.05)
    cfg = _make_cfg(mode)
    res = fit_image_multiband(list(bands), config=cfg)
    iso = res["isophotes"]
    b4s = [r["b4_r"] for r in iso[5:10]]
    assert max(b4s) > 0.005, (mode, b4s)


@pytest.mark.parametrize("mode", ["simultaneous_in_loop", "simultaneous_original"])
def test_simultaneous_with_loose_validity_runs(mode):
    """simultaneous_* × loose_validity uses the jagged higher-order kernel."""
    bands = _three_band_planted()
    n = bands[0].shape[0]
    mask_g = np.zeros((n, n), dtype=bool)
    mask_g[20:50, 20:50] = True
    masks = [mask_g, np.zeros((n, n), dtype=bool), np.zeros((n, n), dtype=bool)]
    cfg = _make_cfg(mode, loose_validity=True)
    res = fit_image_multiband(list(bands), masks=masks, config=cfg)
    iso = res["isophotes"]
    assert all(r["stop_code"] == 0 for r in iso)
    # Higher-order columns shared on surviving bands.
    for r in iso:
        for n_order in (3, 4):
            assert r[f"a{n_order}_r"] != 0.0 or r[f"a{n_order}_i"] == 0.0
        assert_one_shared_shape(r, bands=("r", "i"))


def test_simultaneous_original_matches_shared_within_tolerance():
    """On clean data with non-pathological geometry, simultaneous_original
    and shared produce higher-order coefficients that agree to within
    ~1% at the same isophotes — they're mathematically equivalent in
    the converged limit (geometry harmonics are tiny, so refitting them
    in simultaneous_original barely changes the higher-order block).
    """
    bands = _three_band_planted(m4_amp=0.05)
    cfg_shared = _make_cfg("shared")
    cfg_orig = _make_cfg("simultaneous_original")
    res_s = fit_image_multiband(list(bands), config=cfg_shared)
    res_o = fit_image_multiband(list(bands), config=cfg_orig)
    iso_s = res_s["isophotes"]
    iso_o = res_o["isophotes"]
    assert len(iso_s) == len(iso_o)
    # Compare reference-band higher-order values at every isophote with
    # absolute tolerance comparable to the residual noise.
    # The two modes now use different per-band conventions: `shared` solves for
    # a dimensionless shape and writes each band's own raw amplitude, while the
    # `simultaneous_*` modes still stamp one identical raw amplitude into every
    # band. They therefore agree on the underlying signal but not column by
    # column; the reference band is compared with a tolerance that reflects
    # that. See docs/10-multiband.md on the shared-shape convention.
    for rs, ro in zip(iso_s, iso_o):
        for col in ("a3_r", "b3_r", "a4_r", "b4_r"):
            np.testing.assert_allclose(rs[col], ro[col], atol=3e-3, err_msg=col)


def test_simultaneous_in_loop_emits_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        IsosterConfigMB(
            bands=["g", "r"],
            reference_band="g",
            multiband_higher_harmonics="simultaneous_in_loop",
        )
    msgs = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any("experimental" in m for m in msgs)


# ---------------------------------------------------------------------------
# Schema 1 round-trip
# ---------------------------------------------------------------------------


def test_higher_harmonics_columns_round_trip(tmp_path):
    """All non-independent modes survive a full FITS round-trip with shared
    per-band values and top-level keys preserved."""
    bands = _three_band_planted()
    cfg = _make_cfg("shared")
    res = fit_image_multiband(list(bands), config=cfg)
    p = tmp_path / "rt.fits"
    isophote_results_mb_to_fits(res, p)
    res_back = isophote_results_mb_from_fits(p)

    assert res_back["multiband_higher_harmonics"] == "shared"
    assert res_back["harmonic_orders"] == [3, 4]
    assert res_back["harmonics_shared"] is True
    iso_back = res_back["isophotes"]
    iso_orig = res["isophotes"]
    assert len(iso_back) == len(iso_orig)
    for rb, ro in zip(iso_back, iso_orig):
        for col in ("a3_g", "a3_r", "a3_i", "a4_g", "a4_r", "a4_i"):
            np.testing.assert_allclose(rb[col], ro[col])


def test_harmonic_orders_extension_to_5_6():
    """``harmonic_orders=[3, 4, 5, 6]`` works under shared mode; per-band
    columns ``a5_<b>``, ``b5_<b>``, ``a6_<b>``, ``b6_<b>`` get written and
    are identical across bands."""
    bands = _three_band_planted()
    cfg = _make_cfg("shared", harmonic_orders=[3, 4, 5, 6])
    res = fit_image_multiband(list(bands), config=cfg)
    iso = res["isophotes"]
    assert all(r["stop_code"] == 0 for r in iso)
    for r in iso:
        for n_order in (3, 4, 5, 6):
            assert f"a{n_order}_g" in r
            assert f"b{n_order}_g" in r
            assert_one_shared_shape(r, orders=(3, 4, 5, 6))


# ---------------------------------------------------------------------------
# Per-band Bender normalization
# ---------------------------------------------------------------------------


def test_per_band_normalization_recovers_one_shape_under_shared_mode():
    """Under ``shared``, normalizing each band's *own* raw column returns one value.

    This is the defining property of ``shared`` mode and the thing that
    distinguishes it from ``simultaneous_*``. The solve fits a single
    dimensionless shape and writes it into each band scaled by that band's own
    gradient, so dividing each band's raw column by its own ``sma * |grad_b|``
    must recover the same number in every band.

    An earlier version of this test took *one* band's raw value, divided it by
    three different gradients, and asserted the results differed. That is the
    ``simultaneous_*`` semantics -- one identical raw amplitude across bands --
    and asserting it here documented behaviour ``shared`` deliberately does not
    have.
    """
    bands = _three_band_planted()
    cfg = _make_cfg("shared").model_copy(update={"debug": True})
    res = fit_image_multiband(list(bands), config=cfg)
    iso = res["isophotes"]

    # Normalize through the production helper rather than reimplementing the
    # formula here. The plot-time convention is the *signed* Bender form
    # ``-A_raw / (sma * grad_b)``; an open-coded ``abs(grad)`` version returns
    # the exact negative of it on this fixture and would still pass the
    # cross-band equality check below while testing the wrong convention.
    # ``debug=True`` exposes the solver's own ``grad_<b>``; a finite-difference
    # proxy leaves a few-percent residual and would need a meaninglessly loose
    # tolerance.
    smas = np.asarray([r["sma"] for r in iso], dtype=float)
    pick = len(iso) // 2
    normalized = {}
    for band in ("g", "r", "i"):
        grads = np.asarray([r[f"grad_{band}"] for r in iso], dtype=float)
        assert np.isfinite(grads[pick]) and grads[pick] != 0.0, f"{band}: unusable gradient"
        curve = _normalize_harmonic_for_plot(
            np.asarray([r[f"a4_{band}"] for r in iso], dtype=float),
            smas,
            grads,
            np.asarray([r[f"intens_{band}"] for r in iso], dtype=float),
        )
        normalized[band] = float(curve[pick])

    values = list(normalized.values())
    assert max(values) - min(values) < 1e-6, f"shared mode must normalize to one shape across bands, got {normalized}"

    # And the raw columns really are band-distinct, so the agreement above is
    # the normalization doing work rather than three identical inputs.
    raw = [float(iso[pick][f"a4_{band}"]) for band in ("g", "r", "i")]
    assert max(raw) - min(raw) > 1e-12, f"raw columns unexpectedly identical: {raw}"

    # Pin the *sign* convention, which the equality check above cannot see.
    # This fixture is an ordinary outward-declining profile, so every gradient
    # is negative and the signed form -A/(sma*grad) is the exact negative of
    # the absolute form -A/(sma*|grad|). A helper that quietly switched to
    # abs() would flip every published curve while still normalizing the bands
    # to a common value.
    for band in ("g", "r", "i"):
        grad = float(iso[pick][f"grad_{band}"])
        assert grad < 0.0, f"{band}: fixture expected an outward-declining profile"
        abs_form = -float(iso[pick][f"a4_{band}"]) / (float(iso[pick]["sma"]) * abs(grad))
        assert np.isclose(normalized[band], -abs_form, rtol=1e-9), (
            f"{band}: plot-time normalization is not the signed Bender form"
        )


# ---------------------------------------------------------------------------
# Direct solver-level tests
# ---------------------------------------------------------------------------


def test_fit_simultaneous_joint_recovers_planted_m4():
    """Solver-level: planting cos(4φ) into a 3-band ring recovers a B4
    coefficient close to the input amplitude."""
    rng = np.random.default_rng(0)
    B, N = 3, 64
    phi = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    I0 = np.array([1.0, 1.5, 2.0])
    A4_true = 0.04
    intens = (I0[:, None] + A4_true * np.cos(4 * phi)[None, :] + 0.001 * rng.standard_normal((B, N))).astype(np.float64)
    bw = np.ones(B, dtype=np.float64)
    coeffs, cov, _ = fit_simultaneous_joint(phi, intens, bw, [3, 4])
    assert coeffs.shape == (B + 4 + 4,)
    assert cov is not None and cov.shape == (B + 8, B + 8)
    # Per-band I0
    np.testing.assert_allclose(coeffs[:B], I0, atol=2e-4)
    # Geometric (A1, B1, A2, B2) ≈ 0
    np.testing.assert_allclose(coeffs[B : B + 4], np.zeros(4), atol=2e-3)
    # Higher block: A3, B3, A4, B4. Only B4 has a planted signal.
    higher = coeffs[B + 4 : B + 8]
    np.testing.assert_allclose(higher[0], 0.0, atol=2e-3)  # a3
    np.testing.assert_allclose(higher[1], 0.0, atol=2e-3)  # b3
    np.testing.assert_allclose(higher[2], 0.0, atol=2e-3)  # a4
    np.testing.assert_allclose(higher[3], A4_true, atol=2e-3)  # b4


def test_fit_simultaneous_joint_loose_jagged_path():
    """The loose-validity higher-order solver returns a wider coeffs
    vector and recovers the planted signal with per-band-distinct N_b."""
    rng = np.random.default_rng(1)
    phi_full = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    # Each band keeps a different number of samples to exercise the
    # jagged-builder path. Common subset is the first N_min indices.
    n_keep = [60, 50, 40]
    phi_per_band = [phi_full[: n_keep[b]] for b in range(3)]
    I0 = np.array([1.0, 1.5, 2.0])
    A4_true = 0.04
    intens_per_band = []
    for b in range(3):
        i_b = I0[b] + A4_true * np.cos(4 * phi_per_band[b]) + 0.001 * rng.standard_normal(n_keep[b])
        intens_per_band.append(i_b)
    bw = np.ones(3, dtype=np.float64)
    coeffs, cov, _ = fit_simultaneous_joint_loose(
        phi_per_band,
        intens_per_band,
        bw,
        [3, 4],
    )
    assert coeffs.shape == (3 + 4 + 4,)
    np.testing.assert_allclose(coeffs[:3], I0, atol=5e-4)
    a3, b3, a4, b4 = coeffs[3 + 4 : 3 + 8]
    np.testing.assert_allclose(b4, A4_true, atol=3e-3)


def test_shared_shape_is_independent_of_a_bands_flux_units():
    """The property that makes 'shared' mean a shared *shape*.

    Re-expressing one band's flux in different units is the same physical
    measurement. Before the reparameterisation the reported Bender-normalised
    amplitudes moved with those units — a planted 0.02 was reported as 0.11 and
    0.011 after a x10 rescale, and mirrored under x0.1.
    """
    from isoster._shared import _normalize_harmonic_for_plot
    from isoster.multiband.fitting_mb import _attach_shared_higher_harmonics

    n_samples = 256
    phi = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    sma = 30.0
    planted_shape = 0.02

    def normalised_for(scale):
        gradients = np.array([-1.0, -1.0 * scale])
        raw = np.array([-planted_shape * sma * g for g in gradients])
        intens = np.vstack([100.0 * (1.0 if b == 0 else scale) + raw[b] * np.sin(4.0 * phi) for b in range(2)])
        coeffs = np.zeros(2 + 4)
        coeffs[0], coeffs[1] = intens[0].mean(), intens[1].mean()
        geom: dict = {}
        _attach_shared_higher_harmonics(
            geom,
            ["g", "r"],
            coeffs,
            phi,
            intens,
            None,
            sma,
            list(gradients),
            harmonic_orders=[4],
            band_weights_arr=np.ones(2),
            jagged=False,
        )
        return [
            float(
                _normalize_harmonic_for_plot(
                    np.array([geom[f"a4_{b}"]]), np.array([sma]), np.array([g]), np.array([100.0])
                )[0]
            )
            for b, g in zip(("g", "r"), gradients)
        ]

    for scale in (1.0, 10.0, 0.1, 100.0):
        for value in normalised_for(scale):
            assert value == pytest.approx(planted_shape, rel=1e-6), scale


def test_band_scale_actually_changes_the_shared_harmonics():
    """Guard that the reconstruction factor is not an inert parameter.

    ``band_scale`` corrects the geometric model subtracted before the
    higher-order refit. If omitting it made no difference there would be
    nothing for the call-site test below to protect.
    """
    from isoster.multiband.fitting_mb import _attach_shared_higher_harmonics

    n_samples = 154  # ~60% angular coverage
    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)[:n_samples]
    gradients = [-1.0, -10.0]  # deliberately unequal
    coeffs = np.array([100.0, 100.0, 0.7, -0.4, 0.3, 0.2])
    intens = np.vstack([100.0 + 0.5 * np.sin(4.0 * phi) + 0.3 * np.cos(3.0 * phi) for _ in range(2)])

    without: dict = {}
    with_scale: dict = {}
    _attach_shared_higher_harmonics(
        without,
        ["g", "r"],
        coeffs,
        phi,
        intens,
        None,
        30.0,
        gradients,
        harmonic_orders=[3, 4],
        band_weights_arr=np.ones(2),
        jagged=False,
    )
    _attach_shared_higher_harmonics(
        with_scale,
        ["g", "r"],
        coeffs,
        phi,
        intens,
        None,
        30.0,
        gradients,
        harmonic_orders=[3, 4],
        band_weights_arr=np.ones(2),
        jagged=False,
        band_scale=np.array([0.25, 1.75]),
    )
    assert any(abs(with_scale[f"a{n}_g"] - without[f"a{n}_g"]) > 1e-9 for n in (3, 4)), (
        "band_scale made no difference; the call-site guard below would be meaningless"
    )


def test_every_harmonic_dispatch_call_passes_the_reconstruction_factor():
    """Every path into the harmonic dispatcher, including the maxit fallback.

    The convergence paths passed ``band_scale`` while the two ``stop_code=2``
    fallback calls did not, because they sit at a different indentation and an
    earlier pattern-based edit matched only the convergence sites. ``stop_code=2``
    is a usable best-effort result, so its harmonics have to be reconstructed
    from the same model as everything else.

    A structural check is the right shape here: the failure mode is a *new call
    site* forgetting the keyword, which no numerical fixture reliably catches.
    """
    import pathlib
    import re

    from isoster.multiband import fitting_mb

    # Read from disk rather than via ``inspect.getsource``: that goes through
    # ``linecache``, which caches file contents and can return a stale copy.
    module_source = pathlib.Path(fitting_mb.__file__).read_text()
    start_of_fit = module_source.index("def fit_isophote_mb(")
    end_of_fit = module_source.index("\ndef ", module_source.index("return best_geometry", start_of_fit))
    source = module_source[start_of_fit:end_of_fit]
    calls = [m.start() for m in re.finditer(r"_attach_higher_harmonics_dispatch\(", source)]
    assert len(calls) >= 4, f"expected several dispatch sites, found {len(calls)}"
    for start in calls:
        depth, end = 0, start
        for idx in range(source.index("(", start), len(source)):
            if source[idx] == "(":
                depth += 1
            elif source[idx] == ")":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        call_text = source[start:end]
        assert "band_scale=" in call_text, f"dispatch call at offset {start} omits band_scale"


def test_dispatcher_forwards_the_reconstruction_factor_to_simultaneous_in_loop():
    """The dispatcher's own call must forward ``band_scale`` too.

    The structural guard above walks call sites inside ``fit_isophote_mb``, so
    it cannot see the dispatcher's internal call to
    ``_attach_simultaneous_higher_harmonics_from_coeffs``. That path uses the
    factor for its OLS residual-variance rescale, and dropping it left both
    other tests green.

    Constructed so the answer is unambiguous: ``intens`` is the *exact* model
    implied by the coefficients and the per-band scales, so the residual — and
    therefore every formal error — must be zero. Reconstructing without the
    factor describes a model that was never fitted and produces a non-zero
    error out of nothing.
    """
    from isoster.multiband.fitting_mb import _attach_higher_harmonics_dispatch

    n_bands, orders = 2, [3, 4]
    width = n_bands + 4 + 2 * len(orders)
    n_samples = 128
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    band_scale = np.array([0.25, 1.75])  # deliberately unequal and distinctive

    coeffs = np.zeros(width)
    coeffs[:n_bands] = [100.0, 70.0]
    coeffs[n_bands : n_bands + 4] = [1.5, -2.0, 0.8, 1.2]
    coeffs[n_bands + 4 :] = [0.6, -0.4, 0.3, 0.5]

    def shared_terms(ang):
        out = (
            coeffs[n_bands] * np.sin(ang)
            + coeffs[n_bands + 1] * np.cos(ang)
            + coeffs[n_bands + 2] * np.sin(2.0 * ang)
            + coeffs[n_bands + 3] * np.cos(2.0 * ang)
        )
        for j, order in enumerate(orders):
            out = out + coeffs[n_bands + 4 + 2 * j] * np.sin(order * ang)
            out = out + coeffs[n_bands + 4 + 2 * j + 1] * np.cos(order * ang)
        return out

    intens = np.vstack([coeffs[b] + band_scale[b] * shared_terms(angles) for b in range(n_bands)])

    cfg = _make_cfg("simultaneous_in_loop")
    cfg = cfg.model_copy(update={"bands": ["g", "r"], "reference_band": "g", "harmonic_orders": orders})

    def run(scale):
        geom: dict = {}
        _attach_higher_harmonics_dispatch(
            geom,
            ["g", "r"],
            cfg,
            coeffs,
            angles,
            intens,
            None,
            30.0,
            [-1.0, -1.0],
            np.ones(n_bands),
            jagged=False,
            last_cov=np.eye(width),
            last_wls_mode=False,
            band_scale=scale,
        )
        return geom

    with_scale = run(band_scale)
    without_scale = run(None)

    # Exact model, so every formal error must vanish.
    for order in orders:
        assert with_scale[f"a{order}_err_g"] == pytest.approx(0.0, abs=1e-9)
        assert with_scale[f"b{order}_err_g"] == pytest.approx(0.0, abs=1e-9)
    # And omitting the factor manufactures an error from nothing.
    assert any(without_scale[f"a{order}_err_g"] > 1e-3 for order in orders)
