"""
Tests for the outer-region center regularization feature.

Covers:
- Backward compatibility (use_outer_center_regularization=False is untouched,
  including the inward-first loop reorder that is always active).
- Clean exponential galaxy is not over-constrained at any strength value.
- Contaminated galaxy has reduced pre-reference center drift with the feature on.
- Inner reference fallback when no inward isophotes exist.
- Config validation (warnings for onset < sma0, minsma >= sma0, and
  fix_center combined with the feature).
- LSB auto-lock still fires when both features are active.
- Forced-photometry guard: the feature must warn and no-op when a template
  is provided.
"""

import warnings

import numpy as np
import pytest

from isoster.config import IsosterConfig
from isoster.driver import _build_outer_reference, fit_image


def _make_exponential_galaxy(
    shape=(200, 200),
    center=None,
    scale=15.0,
    peak=5000.0,
    background=0.0,
    rng_seed=0,
    noise_sigma=0.0,
):
    """Simple exponential profile galaxy image with optional Gaussian noise.

    Duplicated from test_lsb_auto_lock.py because tests/integration has no
    __init__.py and thus does not support intra-package imports.
    """
    h, w = shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    image = peak * np.exp(-r / scale) + background
    if noise_sigma > 0.0:
        rng = np.random.default_rng(rng_seed)
        image = image + rng.normal(0.0, noise_sigma, size=image.shape)
    return image


def _make_truncated_galaxy(
    shape=(200, 200),
    center=None,
    scale=10.0,
    peak=8000.0,
    truncate_r=40.0,
    background=100.0,
    noise_sigma=30.0,
    rng_seed=42,
):
    """Bright galaxy inside ``truncate_r`` that fades abruptly into noisy
    background outside, so outward growth slides into the LSB regime.

    Duplicated from test_lsb_auto_lock.py (see _make_exponential_galaxy note).
    """
    h, w = shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    galaxy = peak * np.exp(-r / scale)
    galaxy[r > truncate_r] = 0.0
    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(background, noise_sigma, size=(h, w))
    return galaxy + noise


def _make_contaminated_truncated_galaxy(
    shape=(200, 200),
    center=None,
    scale=10.0,
    peak=8000.0,
    truncate_r=40.0,
    background=100.0,
    noise_sigma=30.0,
    contaminant_center=(150.0, 100.0),
    contaminant_flux=3000.0,
    contaminant_scale=8.0,
    rng_seed=123,
):
    """Truncated galaxy with a bright off-center point source as contaminant.

    The point source is placed well outside truncate_r so it only affects the
    LSB outer isophotes once the fit starts to drift.
    """
    h, w = shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    y, x = np.ogrid[:h, :w]
    r_gal = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    galaxy = peak * np.exp(-r_gal / scale)
    galaxy[r_gal > truncate_r] = 0.0

    r_cont = np.sqrt((x - contaminant_center[0]) ** 2 + (y - contaminant_center[1]) ** 2)
    contaminant = contaminant_flux * np.exp(-r_cont / contaminant_scale)

    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(background, noise_sigma, size=(h, w))
    return galaxy + contaminant + noise


def _pre_lock_outward_isophotes(result):
    """Return outward isophotes excluding any hybrid-lock-frozen tail."""
    sma0 = result["config"].sma0
    out = []
    for iso in result["isophotes"]:
        if iso.get("sma", 0.0) < sma0:
            continue
        if iso.get("lsb_locked", False):
            continue
        out.append(iso)
    return out


def _combined_drift(isophotes, x0_ref, y0_ref):
    if not isophotes:
        return 0.0, 0.0, 0.0
    dx = np.array([iso["x0"] - x0_ref for iso in isophotes])
    dy = np.array([iso["y0"] - y0_ref for iso in isophotes])
    return (
        float(np.max(np.abs(dx))),
        float(np.max(np.abs(dy))),
        float(np.sqrt(np.max(np.abs(dx)) ** 2 + np.max(np.abs(dy)) ** 2)),
    )


class TestBackwardCompatibility:
    """With the feature off, nothing should observably change. This also
    catches regressions from the inward-first loop reorder."""

    def test_default_is_disabled(self):
        cfg = IsosterConfig(x0=100, y0=100, sma0=10.0, maxsma=60.0)
        assert cfg.use_outer_center_regularization is False

        image = _make_exponential_galaxy()
        result = fit_image(image, None, cfg)

        assert "use_outer_center_regularization" not in result
        assert "outer_reg_x0_ref" not in result
        assert "outer_reg_y0_ref" not in result

    def test_inward_first_reorder_preserves_inward_results(self):
        """The inward loop's outputs must match between old and new driver
        order. Since the old order is gone, we check that the inward isophotes
        are independent of the feature flag (feature off vs on).
        """
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0, noise_sigma=0.0)
        base = dict(x0=100.0, y0=100.0, sma0=10.0, minsma=2.0, maxsma=80.0, astep=0.2)

        result_off = fit_image(image, None, IsosterConfig(**base))
        result_on = fit_image(
            image,
            None,
            IsosterConfig(
                **base,
                use_outer_center_regularization=True,
                outer_reg_strength=0.0,  # strength=0 => penalty is zero
            ),
        )

        inward_off = [iso for iso in result_off["isophotes"] if iso["sma"] < 10.0]
        inward_on = [iso for iso in result_on["isophotes"] if iso["sma"] < 10.0]
        assert len(inward_off) == len(inward_on)
        for a, b in zip(inward_off, inward_on):
            assert a["x0"] == pytest.approx(b["x0"])
            assert a["y0"] == pytest.approx(b["y0"])
            # The galaxy is circular, so both fits converge to eps = 0; with cubic reads the
            # two runs can stop one iteration apart (eps 1e-11 against 1e-9), which is far
            # below any measurable ellipticity.
            assert a["eps"] == pytest.approx(b["eps"], abs=1e-6)


class TestCleanGalaxyNoOverConstraint:
    """The feature must not introduce artificial drift on a clean galaxy."""

    @pytest.mark.parametrize("strength", [0.5, 2.0, 8.0])
    def test_clean_galaxy_center_stays_stable(self, strength):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0, noise_sigma=0.0)
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            minsma=2.0,
            maxsma=60.0,
            astep=0.25,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=25.0,
            outer_reg_sma_width=8.0,
            outer_reg_strength=strength,
        )
        result = fit_image(image, None, cfg)

        assert result["use_outer_center_regularization"] is True
        x0_ref = result["outer_reg_x0_ref"]
        y0_ref = result["outer_reg_y0_ref"]

        # Reference should sit near the true center (within 0.5 px).
        assert abs(x0_ref - 100.0) < 0.5
        assert abs(y0_ref - 100.0) < 0.5

        # Every outward isophote should sit within 1 px of the reference —
        # a clean noiseless galaxy should never drift.
        outward = _pre_lock_outward_isophotes(result)
        assert len(outward) > 0
        max_dx = max(abs(iso["x0"] - x0_ref) for iso in outward)
        max_dy = max(abs(iso["y0"] - y0_ref) for iso in outward)
        assert max_dx < 1.0, f"clean galaxy drifted {max_dx} px in x at strength={strength}"
        assert max_dy < 1.0, f"clean galaxy drifted {max_dy} px in y at strength={strength}"


class TestContaminatedGalaxyDriftReduced:
    """Pre-reference center drift should drop with the feature on."""

    def test_contaminated_galaxy_drift_decreases(self):
        image = _make_contaminated_truncated_galaxy()
        base = dict(
            x0=100.0,
            y0=100.0,
            sma0=8.0,
            minsma=2.0,
            maxsma=90.0,
            astep=0.15,
        )

        cfg_off = IsosterConfig(**base)
        cfg_on = IsosterConfig(
            **base,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=30.0,
            outer_reg_sma_width=8.0,
            outer_reg_strength=5.0,
        )

        result_off = fit_image(image, None, cfg_off)
        result_on = fit_image(image, None, cfg_on)

        # Reference used for both metrics: the feature-on reference centroid
        # (this is the "ground truth" inner center either driver builds).
        x0_ref = result_on["outer_reg_x0_ref"]
        y0_ref = result_on["outer_reg_y0_ref"]

        out_off = _pre_lock_outward_isophotes(result_off)
        out_on = _pre_lock_outward_isophotes(result_on)

        _, _, drift_off = _combined_drift(out_off, x0_ref, y0_ref)
        _, _, drift_on = _combined_drift(out_on, x0_ref, y0_ref)

        assert drift_on < drift_off, (
            f"Feature-on drift {drift_on:.2f} px should be less than "
            f"feature-off drift {drift_off:.2f} px on contaminated galaxy"
        )


class TestInnerReferenceFallback:
    """When there are no inward isophotes, the reference must fall back to
    the anchor isophote center without error."""

    def test_fallback_to_anchor_when_no_inward(self):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0, noise_sigma=0.0)
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            minsma=10.0,  # no inward growth
            maxsma=40.0,
            astep=0.25,
            use_outer_center_regularization=True,
            outer_reg_strength=1.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_image(image, None, cfg)

        assert result["use_outer_center_regularization"] is True
        anchor = next(iso for iso in result["isophotes"] if iso["sma"] == 10.0)
        assert result["outer_reg_x0_ref"] == pytest.approx(anchor["x0"])
        assert result["outer_reg_y0_ref"] == pytest.approx(anchor["y0"])


class TestConfigValidation:
    """Validator warnings for the outer-center-regularization fields."""

    def test_onset_below_sma0_warns(self):
        with pytest.warns(UserWarning, match="outer_reg_sma_onset"):
            IsosterConfig(
                x0=100,
                y0=100,
                sma0=50.0,
                maxsma=200.0,
                use_outer_center_regularization=True,
                outer_reg_sma_onset=10.0,
            )

    def test_minsma_ge_sma0_warns(self):
        with pytest.warns(UserWarning, match="inward isophotes to build the reference"):
            IsosterConfig(
                x0=100,
                y0=100,
                sma0=10.0,
                minsma=10.0,
                maxsma=60.0,
                use_outer_center_regularization=True,
                outer_reg_sma_onset=30.0,
            )

    def test_fix_center_warns(self):
        with pytest.warns(UserWarning, match="use_outer_center_regularization"):
            IsosterConfig(
                x0=100,
                y0=100,
                sma0=10.0,
                maxsma=60.0,
                use_outer_center_regularization=True,
                fix_center=True,
            )


class TestLsbAutoLockStillFires:
    """With both use_outer_center_regularization and lsb_auto_lock enabled,
    the auto-lock must still commit at roughly the same sma on the existing
    truncated fixture."""

    def test_lock_still_fires_with_outer_reg(self):
        image = _make_truncated_galaxy()
        base = dict(
            x0=100.0,
            y0=100.0,
            sma0=8.0,
            minsma=2.0,
            maxsma=90.0,
            astep=0.15,
            lsb_auto_lock=True,
            lsb_auto_lock_debounce=2,
            lsb_auto_lock_maxgerr=0.3,
            debug=True,
        )
        result_lock = fit_image(image, None, IsosterConfig(**base))
        result_both = fit_image(
            image,
            None,
            IsosterConfig(
                **base,
                use_outer_center_regularization=True,
                outer_reg_sma_onset=20.0,
                outer_reg_sma_width=6.0,
                outer_reg_strength=2.0,
            ),
        )

        assert result_lock["lsb_auto_lock_sma"] is not None, "baseline lock should fire"
        assert result_both["lsb_auto_lock_sma"] is not None, "lock should still fire with outer_reg on"

        sma_lock = result_lock["lsb_auto_lock_sma"]
        sma_both = result_both["lsb_auto_lock_sma"]

        # Allow up to 2 growth steps difference.
        ratio = sma_both / sma_lock
        assert 1.0 / (1.15**2) <= ratio <= (1.15**2), (
            f"lock sma shifted too far: baseline={sma_lock:.1f}, with outer_reg={sma_both:.1f}"
        )


class TestForcedPhotometryGuard:
    """use_outer_center_regularization must warn and no-op in forced mode."""

    def test_forced_photometry_warns_and_runs(self):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0)

        template_cfg = IsosterConfig(x0=100.0, y0=100.0, sma0=10.0, maxsma=40.0, astep=0.25)
        template_result = fit_image(image, None, template_cfg)
        template = template_result["isophotes"]

        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            maxsma=40.0,
            astep=0.25,
            use_outer_center_regularization=True,
            outer_reg_strength=2.0,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit_image(image, None, cfg, template=template)

        assert any(
            "use_outer_center_regularization" in str(w.message) and "forced photometry" in str(w.message)
            for w in caught
        ), "expected a forced-photometry guard warning for outer_center_regularization"
        assert "use_outer_center_regularization" not in result
        assert "outer_reg_x0_ref" not in result
        assert "outer_reg_y0_ref" not in result
        assert "outer_reg_eps_ref" not in result
        assert "outer_reg_pa_ref" not in result


# --- outer_reg_weights (center / eps / pa) coverage ------------------------


def _count_saturated_outer_eps_steps(isophotes, sma_threshold, clip_max_eps, frac=0.9):
    """Count adjacent-isophote |delta eps| steps at or above frac*clip_max_eps.

    Restricted to outward isophotes with sma >= sma_threshold.
    """
    smas = np.array([iso["sma"] for iso in isophotes])
    eps = np.array([iso["eps"] for iso in isophotes])
    order = np.argsort(smas)
    smas = smas[order]
    eps = eps[order]
    mask = smas >= sma_threshold
    eps_outer = eps[mask]
    if eps_outer.size < 2:
        return 0
    deps = np.abs(np.diff(eps_outer))
    return int(np.sum(deps >= frac * clip_max_eps))


class TestOuterReferenceCarriesEpsAndPa:
    """The reference dict populates all four geometry fields, and the result
    dict exposes the eps/pa references for downstream QA."""

    def test_reference_geom_has_all_four_fields(self):
        image = _make_exponential_galaxy(scale=20.0, peak=10000.0)
        cfg = IsosterConfig(
            x0=100.0,
            y0=100.0,
            sma0=10.0,
            minsma=1.0,
            maxsma=40.0,
            astep=0.25,
            use_outer_center_regularization=True,
            outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
        )
        result = fit_image(image, None, cfg)
        assert "outer_reg_eps_ref" in result
        assert "outer_reg_pa_ref" in result
        assert np.isfinite(result["outer_reg_eps_ref"])
        assert np.isfinite(result["outer_reg_pa_ref"])
        assert 0.0 <= result["outer_reg_eps_ref"] < 1.0
        assert 0.0 <= result["outer_reg_pa_ref"] < np.pi


class TestPaCircularMean:
    """PA reference must use a circular mean on 2*pa so that inner isophotes
    with pa near 0 and pa near pi do not average to pi/2 (orthogonal to both)."""

    def test_circular_mean_wraps_correctly(self):
        inwards = [
            {
                "sma": 5.0,
                "x0": 100.0,
                "y0": 100.0,
                "eps": 0.2,
                "pa": 0.05,
                "intens": 1000.0,
                "stop_code": 0,
            },
            {
                "sma": 7.0,
                "x0": 100.0,
                "y0": 100.0,
                "eps": 0.2,
                "pa": np.pi - 0.05,
                "intens": 1000.0,
                "stop_code": 0,
            },
        ]
        anchor = {
            "sma": 10.0,
            "x0": 100.0,
            "y0": 100.0,
            "eps": 0.2,
            "pa": 0.0,
            "intens": 1000.0,
            "stop_code": 0,
        }
        cfg = IsosterConfig(x0=100.0, y0=100.0, maxsma=40.0)
        ref = _build_outer_reference(inwards, anchor, cfg)
        dist_from_pi_over_2 = abs(ref["pa"] - 0.5 * np.pi)
        assert dist_from_pi_over_2 > 0.5, (
            f"pa_ref={ref['pa']:.3f} rad landed near pi/2 - circular mean "
            "appears to have used a naive arithmetic average over axis-like "
            "angles."
        )


class TestFullWeightsSuppressEpsSaturation:
    """On a contaminated noisy galaxy, the center-only outer-reg can produce
    saturated delta eps steps because the penalty redirects the outer random
    walk from (x0, y0) into (eps, pa). Turning on the eps weight should not
    increase the saturated-step count."""

    def test_full_weights_reduce_eps_saturation(self):
        image = _make_contaminated_truncated_galaxy()
        base = dict(
            x0=100.0,
            y0=100.0,
            sma0=8.0,
            minsma=2.0,
            maxsma=90.0,
            astep=0.15,
        )
        outer_kwargs = dict(
            use_outer_center_regularization=True,
            outer_reg_sma_onset=30.0,
            outer_reg_sma_width=8.0,
            outer_reg_strength=5.0,
        )

        cfg_center_only = IsosterConfig(**base, **outer_kwargs)
        cfg_full = IsosterConfig(
            **base,
            **outer_kwargs,
            outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
        )

        res_center = fit_image(image, None, cfg_center_only)
        res_full = fit_image(image, None, cfg_full)

        clip_max_eps = cfg_center_only.clip_max_eps
        assert clip_max_eps is not None
        n_sat_center = _count_saturated_outer_eps_steps(
            res_center["isophotes"], sma_threshold=30.0, clip_max_eps=clip_max_eps
        )
        n_sat_full = _count_saturated_outer_eps_steps(
            res_full["isophotes"], sma_threshold=30.0, clip_max_eps=clip_max_eps
        )
        assert n_sat_full <= n_sat_center, (
            f"full weights should not produce MORE saturated delta eps steps: "
            f"center-only={n_sat_center}, full={n_sat_full}"
        )


class TestOuterRegWeightsValidation:
    """New config-level warnings for degenerate weight combinations."""

    def test_all_zero_weights_warns(self):
        with pytest.warns(UserWarning, match="all outer_reg_weights are zero"):
            IsosterConfig(
                x0=100.0,
                y0=100.0,
                sma0=10.0,
                maxsma=40.0,
                use_outer_center_regularization=True,
                outer_reg_weights={"center": 0.0, "eps": 0.0, "pa": 0.0},
            )

    def test_fix_eps_with_eps_weight_warns(self):
        with pytest.warns(UserWarning, match="fix_eps=True"):
            IsosterConfig(
                x0=100.0,
                y0=100.0,
                sma0=10.0,
                maxsma=40.0,
                fix_eps=True,
                eps=0.3,
                use_outer_center_regularization=True,
                outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 0.0},
            )

    def test_fix_pa_with_pa_weight_warns(self):
        with pytest.warns(UserWarning, match="fix_pa=True"):
            IsosterConfig(
                x0=100.0,
                y0=100.0,
                sma0=10.0,
                maxsma=40.0,
                fix_pa=True,
                use_outer_center_regularization=True,
                outer_reg_weights={"center": 1.0, "eps": 0.0, "pa": 1.0},
            )


# --- outer_reg_mode = 'solver' (Tikhonov) coverage -------------------------


def _pa_step_std_outer(result, sma_threshold):
    """Std of adjacent-isophote delta pa (radians) for outward iso sma >= threshold.

    Smoothness metric: baseline free fits show std ~0.01-0.05 rad; the
    selector-mode quantized jumps push std up because of occasional
    0.5-rad clipped steps.
    """
    isos = result["isophotes"]
    sma = np.array([iso["sma"] for iso in isos])
    pa = np.array([iso["pa"] for iso in isos])
    order = np.argsort(sma)
    sma, pa = sma[order], pa[order]
    mask = sma >= sma_threshold
    pa_o = pa[mask]
    if pa_o.size < 3:
        return float("nan")
    dpa = np.diff(pa_o)
    # Wrap each delta to (-pi/2, pi/2] so mod-pi artefacts don't blow up std.
    dpa = ((dpa + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
    return float(np.std(dpa))


class TestSolverModeTikhonov:
    """Tests for the solver-level Tikhonov form of outer regularization."""

    def test_default_mode_is_damping(self):
        """Default outer_reg_mode is 'damping' and the only other valid
        value is 'solver'. The historical 'selector' string is no longer
        accepted."""
        cfg = IsosterConfig(x0=100.0, y0=100.0, maxsma=40.0)
        assert cfg.outer_reg_mode == "damping"
        with pytest.raises(Exception):
            # Pydantic raises ValidationError (a subclass) on pattern mismatch
            IsosterConfig(x0=100.0, y0=100.0, maxsma=40.0, outer_reg_mode="selector")

    def test_default_weights_are_full(self):
        """Default outer_reg_weights damp center, eps, and pa uniformly.
        This is the cleanup default (was {1, 0, 0} in an earlier revision
        of this branch, which caused saturated eps/PA jumps on real data)."""
        cfg = IsosterConfig(x0=100.0, y0=100.0, maxsma=40.0)
        assert cfg.outer_reg_weights == {"center": 1.0, "eps": 1.0, "pa": 1.0}

    def test_sma_width_auto_computes_when_none(self):
        """outer_reg_sma_width defaults to None and is auto-computed as
        0.4 * onset inside the fitting code. Expert users can still set it."""
        cfg_default = IsosterConfig(
            x0=100.0,
            y0=100.0,
            maxsma=40.0,
            use_outer_center_regularization=True,
        )
        assert cfg_default.outer_reg_sma_width is None
        # Explicit override still works.
        cfg_explicit = IsosterConfig(
            x0=100.0,
            y0=100.0,
            maxsma=40.0,
            use_outer_center_regularization=True,
            outer_reg_sma_width=15.0,
        )
        assert cfg_explicit.outer_reg_sma_width == 15.0

    def test_solver_mode_zero_weights_equals_no_reg(self):
        """With all weights zero, solver mode must reproduce the no-reg
        fit exactly - this is the pixel-identity gate that proves the
        solver branch is dormant at zero weights (regardless of mode)."""
        image = _make_contaminated_truncated_galaxy()
        base = dict(x0=100.0, y0=100.0, sma0=8.0, minsma=2.0, maxsma=90.0, astep=0.15)
        cfg_off = IsosterConfig(**base)
        cfg_solver_zero = IsosterConfig(
            **base,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=30.0,
            outer_reg_sma_width=8.0,
            outer_reg_strength=5.0,
            outer_reg_mode="solver",
            outer_reg_weights={"center": 0.0, "eps": 0.0, "pa": 0.0},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_off = fit_image(image, None, cfg_off)
            res_solver_zero = fit_image(image, None, cfg_solver_zero)
        iso_off = res_off["isophotes"]
        iso_solver = res_solver_zero["isophotes"]
        assert len(iso_off) == len(iso_solver)
        for a, b in zip(iso_off, iso_solver):
            for key in ("x0", "y0", "eps", "pa", "sma"):
                assert a[key] == pytest.approx(b[key], abs=1e-12), (
                    f"solver+zero-weights diverged from no-reg at key={key}"
                )

    def test_damping_mode_reduces_pa_step_variance(self):
        """On the contaminated mock, damping mode (the default) should
        yield smaller PA step variance than no regularization. This is
        the smoothness gate that motivated the Tikhonov solver-level
        rewrite of the outer-reg feature."""
        image = _make_contaminated_truncated_galaxy()
        base = dict(x0=100.0, y0=100.0, sma0=8.0, minsma=2.0, maxsma=90.0, astep=0.15)
        cfg_off = IsosterConfig(**base)
        cfg_damping = IsosterConfig(
            **base,
            use_outer_center_regularization=True,
            outer_reg_mode="damping",
            outer_reg_sma_onset=30.0,
            outer_reg_sma_width=8.0,
            outer_reg_strength=5.0,
            outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_off = fit_image(image, None, cfg_off)
            res_damping = fit_image(image, None, cfg_damping)
        std_off = _pa_step_std_outer(res_off, sma_threshold=30.0)
        std_damping = _pa_step_std_outer(res_damping, sma_threshold=30.0)
        assert std_damping <= std_off + 1e-9, (
            f"damping PA-step std ({std_damping:.4f} rad) should be <= "
            f"no-reg ({std_off:.4f} rad) on the contaminated mock"
        )

    def test_solver_mode_preserves_center_stability(self):
        """Solver mode must still reduce pre-reference center drift on
        the contaminated galaxy (the property that motivated the feature
        in the first place)."""
        image = _make_contaminated_truncated_galaxy()
        base = dict(x0=100.0, y0=100.0, sma0=8.0, minsma=2.0, maxsma=90.0, astep=0.15)
        cfg_off = IsosterConfig(**base)
        cfg_solver = IsosterConfig(
            **base,
            use_outer_center_regularization=True,
            outer_reg_sma_onset=30.0,
            outer_reg_sma_width=8.0,
            outer_reg_strength=5.0,
            outer_reg_mode="solver",
            outer_reg_weights={"center": 1.0, "eps": 1.0, "pa": 1.0},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_off = fit_image(image, None, cfg_off)
            res_solver = fit_image(image, None, cfg_solver)
        x0_ref = res_solver["outer_reg_x0_ref"]
        y0_ref = res_solver["outer_reg_y0_ref"]
        out_off = _pre_lock_outward_isophotes(res_off)
        out_solver = _pre_lock_outward_isophotes(res_solver)
        _, _, drift_off = _combined_drift(out_off, x0_ref, y0_ref)
        _, _, drift_solver = _combined_drift(out_solver, x0_ref, y0_ref)
        assert drift_solver < drift_off, (
            f"solver mode center drift {drift_solver:.2f} px must be less than no-reg drift {drift_off:.2f} px"
        )

    def test_solver_mode_with_simultaneous_harmonics_warns(self):
        """solver + simultaneous_harmonics is not supported in this phase;
        config should warn that the Tikhonov term is dropped."""
        with pytest.warns(UserWarning, match="7-parameter ISOFIT"):
            IsosterConfig(
                x0=100.0,
                y0=100.0,
                sma0=10.0,
                maxsma=40.0,
                use_outer_center_regularization=True,
                outer_reg_mode="solver",
                simultaneous_harmonics=True,
                harmonic_orders=[3],
            )


class TestTikhonovAlphaHelper:
    """Unit-level sanity checks for the _tikhonov_alpha blend fraction."""

    def test_alpha_zero_when_weight_zero(self):
        from isoster.fitting import _tikhonov_alpha

        assert _tikhonov_alpha(coeff=1.0, lambda_sma=2.0, weight=0.0) == 0.0

    def test_alpha_zero_when_lambda_zero(self):
        from isoster.fitting import _tikhonov_alpha

        assert _tikhonov_alpha(coeff=1.0, lambda_sma=0.0, weight=1.0) == 0.0

    def test_alpha_in_unit_interval(self):
        from isoster.fitting import _tikhonov_alpha

        for coeff, lam, w in [(0.5, 1.0, 1.0), (3.0, 5.0, 2.0), (1e-6, 100.0, 100.0)]:
            a = _tikhonov_alpha(coeff, lam, w)
            assert 0.0 <= a < 1.0

    def test_alpha_monotone_in_lambda(self):
        from isoster.fitting import _tikhonov_alpha

        a_low = _tikhonov_alpha(coeff=1.0, lambda_sma=0.1, weight=1.0)
        a_hi = _tikhonov_alpha(coeff=1.0, lambda_sma=10.0, weight=1.0)
        assert a_hi > a_low
