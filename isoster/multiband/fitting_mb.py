"""
Multi-band joint per-isophote fitter.

Implements the joint design-matrix solve and the per-isophote iteration
loop that produces a single shared geometry per SMA along with per-band
intensities and per-band harmonic deviations. See
``docs/agent/plan-2026-04-29-multiband-feasibility.md`` decisions
D2 (joint design matrix), D9 (sigma clipping), D10 (combined gradient),
D11 (per-band I0_b), and D12 (band weights).

The iteration loop is a port of :func:`isoster.fitting.fit_isophote`
extended with the joint design-matrix solve, central regularization,
outer-center Tikhonov damping, and the joint combined-gradient surface
the driver's LSB auto-lock triggers on. The single-band ISOFIT API
(``simultaneous_harmonics``) is not carried over; higher-order
harmonics are handled per-band or shared via the config's
``multiband_higher_harmonics`` enum.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .._shared import _ring_statistic_and_variance, _tikhonov_alpha, _weighted_mean_variance
from ..fitting import (
    compute_aperture_photometry,
    compute_deviations,
    sigma_clip,
)
from ..numba_kernels import build_harmonic_matrix
from .config_mb import IsosterConfigMB
from .numba_kernels_mb import (
    build_joint_design_matrix,
    build_joint_design_matrix_higher,
    build_joint_design_matrix_jagged,
    build_joint_design_matrix_jagged_higher,
)
from .sampling_mb import (
    MultiIsophoteData,
    extract_isophote_data_multi,
    extract_isophote_data_multi_prepared,
    prepare_inputs,
)

# Per-band column-key naming. Centralized so ``_empty_isophote_dict``,
# ``extract_forced_photometry_mb``, and the driver's central-pixel
# helper all agree on which columns exist; Schema-1 readers rely on
# the exact suffix layout.
_PER_BAND_INTENSITY_KEYS: Tuple[str, ...] = ("intens", "intens_err", "rms")
_PER_BAND_DEBUG_KEYS: Tuple[str, ...] = ("grad", "grad_error", "grad_r_error")


def _per_band_harmonic_keys_for_orders(orders: Sequence[int]) -> Tuple[str, ...]:
    """Return per-band harmonic-key suffixes for the given orders.

    With the default ``orders=[3, 4]`` this matches the Stage-1 ``[3, 4]``
    key set exactly. Used by :func:`_empty_isophote_dict` and the
    iteration-loop's per-band column initialization so callers extending
    ``harmonic_orders`` (e.g. to ``[3, 4, 5, 6]``) get all four key sets
    generated automatically.
    """
    keys: List[str] = []
    for n in orders:
        n_int = int(n)
        keys.extend([f"a{n_int}", f"b{n_int}", f"a{n_int}_err", f"b{n_int}_err"])
    return tuple(keys)


# ---------------------------------------------------------------------------
# Joint solve
# ---------------------------------------------------------------------------


def _compute_central_regularization_penalty_mb(
    current_geom: Dict[str, float],
    previous_geom: Optional[Dict[str, float]],
    sma: float,
    config: IsosterConfigMB,
) -> float:
    """Stage-3 Stage-F: central-region regularization penalty (multi-band).

    Geometry is shared in multi-band, so this is a verbatim port of the
    single-band ``compute_central_regularization_penalty``: a Gaussian-
    decaying penalty that adds to the best-iteration selector
    (``effective_amp``) for SMA below the threshold, discouraging
    large per-iteration geometry jumps in the low-S/N central region.
    No per-band design choice — the penalty operates on the shared
    ``(x0, y0, eps, pa)`` only.

    Returns 0.0 when the feature is off, the previous-isophote geometry
    is None (e.g. the first isophote), or the SMA is far enough above
    the threshold that ``λ(sma) < 1e-6``.
    """
    if not config.use_central_regularization:
        return 0.0
    if previous_geom is None:
        return 0.0
    lambda_sma = config.central_reg_strength * float(np.exp(-((sma / config.central_reg_sma_threshold) ** 2)))
    if lambda_sma < 1e-6:
        return 0.0
    weights = config.central_reg_weights
    delta_eps = current_geom["eps"] - previous_geom["eps"]
    delta_pa = current_geom["pa"] - previous_geom["pa"]
    # Wrap PA residual onto [-π, π] (single-band convention).
    delta_pa = ((delta_pa + np.pi) % (2.0 * np.pi)) - np.pi
    delta_x0 = current_geom["x0"] - previous_geom["x0"]
    delta_y0 = current_geom["y0"] - previous_geom["y0"]
    penalty = lambda_sma * (
        float(weights.get("eps", 1.0)) * delta_eps**2
        + float(weights.get("pa", 1.0)) * delta_pa**2
        + float(weights.get("center", 1.0)) * (delta_x0**2 + delta_y0**2)
    )
    return float(penalty)


def _per_band_mean_or_median(
    intens_per_band: NDArray[np.float64],
    variances_per_band: Optional[NDArray[np.float64]],
    integrator: str = "mean",
) -> NDArray[np.float64]:
    """Per-band intercept reducer used in decoupled intercept mode.

    ``integrator='mean'`` (default): inverse-variance weighted mean
    under WLS, simple mean under OLS — preserves the original
    ``_per_band_mean`` semantics.

    ``integrator='median'`` (Stage-3 S1/S2): plain ``np.median`` of
    each band's ring samples. Variances are intentionally ignored —
    medians are not weighted statistics, and the ring samples have
    already been sigma-clipped upstream (sclip/nclip pipeline). Only
    legal under ``fit_per_band_intens_jointly=False`` (config
    validator enforces this).
    """
    n_bands = intens_per_band.shape[0]
    out = np.empty(n_bands, dtype=np.float64)
    for b in range(n_bands):
        if integrator == "median":
            out[b] = float(np.median(intens_per_band[b])) if intens_per_band[b].size else float("nan")
        elif variances_per_band is None:
            out[b] = float(np.mean(intens_per_band[b]))
        else:
            w = 1.0 / variances_per_band[b]
            denom = float(np.sum(w))
            out[b] = float(np.sum(intens_per_band[b] * w) / denom) if denom > 0 else float("nan")
    return out


def _per_band_intercept_variance(
    intens_b: NDArray[np.float64],
    variances_b: Optional[NDArray[np.float64]],
    integrator: str,
    residuals_b: Optional[NDArray[np.float64]] = None,
) -> float:
    """Variance of the per-band ring intercept that decoupled mode reports.

    :func:`_per_band_mean_or_median` writes one of three statistics into
    ``coeffs[b]``: an inverse-variance-weighted mean (WLS with
    ``integrator='mean'``), a plain mean (OLS), or a plain median. The
    uncertainty has to belong to whichever one was actually reported, matching
    single-band ``fit_isophote``. The retired code used the weighted mean's
    variance for every combination, so under ``integrator='median'`` it
    described a statistic nobody reported, and under OLS it dropped the
    median's pi/2 penalty.

    ``residuals_b`` supplies the samples' departures from the fitted harmonic
    model, so the OLS scatter term matches single-band's
    ``np.std(intens - model)``. A ring's real m=1 / m=2 signal averages out of
    the intercept, so leaving it in the scatter would overstate the error.
    Falls back to the raw samples when residuals are unavailable.

    An empty band returns 0.0 rather than an infinite entry that would poison
    the joint covariance matrix; callers report ``intens_err_<b>`` as NaN.
    """
    n_b = int(intens_b.size)
    if n_b == 0:
        return 0.0
    if variances_b is not None:
        if integrator == "median":
            # The reported intercept is a plain median, so it needs the
            # median's own heteroscedastic variance.
            variance = float(_ring_statistic_and_variance(intens_b, variances_b, "median")[1])
        else:
            # The reported intercept is the inverse-variance-weighted mean,
            # whose variance is exactly 1 / sum(1/v).
            variance = float(_weighted_mean_variance(variances_b))
        return variance if np.isfinite(variance) else 0.0
    values = intens_b if residuals_b is None else residuals_b
    variance = float(np.std(values) ** 2 / n_b)
    if integrator == "median":
        # The median's standard error is sqrt(pi/2) larger than the mean's
        # for Gaussian noise.
        variance *= np.pi / 2.0
    return variance


def _split_per_band(flat: NDArray[np.float64], counts: NDArray[np.int64]) -> List[NDArray[np.float64]]:
    """Split a row-stacked jagged vector back into its per-band pieces."""
    return list(np.split(flat, np.cumsum(counts)[:-1]))


def _sigma_bg_variance_floor(config: IsosterConfigMB) -> Optional[float]:
    """The background-noise floor on an OLS residual variance, or ``None``.

    ``sigma_bg`` bounds how small a fit's residual variance can honestly be, so
    every OLS error scale derived from that fit has to share the floor or the
    reported parameters end up on different scales. Single-band computes the
    residual variance once and hands the floored value to all its consumers;
    multi-band derives one per solve (geometry, per-band intensity, higher-order
    harmonics), so the floor travels with each instead.
    """
    return config.sigma_bg**2 if config.sigma_bg is not None else None


def _apply_variance_floor(var_residual: float, floor: Optional[float]) -> float:
    """Raise a residual variance to the ``sigma_bg**2`` floor when one is set."""
    if floor is None:
        return float(var_residual)
    return float(max(var_residual, floor))


def _reference_residual_variance(
    angles_ref: NDArray[np.float64],
    intens_ref: NDArray[np.float64],
    coeffs_ref: NDArray[np.float64],
    floor: Optional[float],
) -> float:
    """OLS residual variance of the reference-band-only harmonic solve.

    Under ``harmonic_combination='ref'`` the geometric coefficients come from the
    reference band alone (:func:`fit_first_and_second_harmonics_ref`), so the
    covariance they carry must be scaled by *that band's* residual variance. The
    pooled joint helper instead measured every band's residuals against a model
    whose geometry only one band had constrained, letting a band excluded from
    the solve set its noise scale: holding the reference band byte-identical and
    raising a second band's noise from 0.2 to 20 moved ``eps_err`` by 18x.

    Five fitted parameters — one intercept plus ``A1, B1, A2, B2`` — matching the
    reference solve, and the same exact model it fitted. Mirrors single-band
    ``fit_isophote``, which computes one residual variance from the exact fitted
    model and hands it to ``compute_parameter_errors`` via ``residual_variance=``.
    """
    n_params = 5
    if int(intens_ref.size) <= n_params:
        # Exactly determined: the model passes through every point, so the data
        # carry no information about the noise.
        var_residual = 0.0
    else:
        model = build_harmonic_matrix(angles_ref) @ coeffs_ref[:n_params]
        var_residual = float(np.var(intens_ref - model, ddof=n_params))
    return _apply_variance_floor(var_residual, floor)


def fit_first_and_second_harmonics_joint(
    angles: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    band_weights_arr: NDArray[np.float64],
    variances_per_band: Optional[NDArray[np.float64]] = None,
    *,
    fit_per_band_intens_jointly: bool = True,
    integrator: str = "mean",
    per_band_gradients: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], bool]:
    """
    Solve the joint multi-band 1st+2nd harmonic system in WLS or OLS mode.

    ``per_band_gradients`` switches the shared parameters from harmonic
    *amplitudes* to geometry *steps* by scaling every shared column by that
    band's radial gradient; see :func:`_geometry_row_scaling`. ``None`` keeps the
    amplitude parameterisation byte-identically.

    Parameters
    ----------
    angles : (N,) float64
        Shared angle array along the ellipse (psi in EA mode, phi in
        regular mode). Same for every band by construction.
    intens_per_band : (B, N) float64
        Per-band intensity samples. Order matches the band order in the
        IsosterConfigMB.bands list.
    band_weights_arr : (B,) float64
        Per-band scalar weights ``w_b`` (already resolved). Must be > 0.
    variances_per_band : (B, N) float64 or None
        Per-pixel variances. ``None`` triggers OLS mode; otherwise WLS.
    fit_per_band_intens_jointly : bool, default True
        Default ``True`` keeps the full ``(B + 4)``-column joint solve
        (per-band intercepts ``I0_b`` co-fit with the shared geometric
        harmonics). When ``False``, the leading ``B`` per-band intercept
        columns are dropped; the solve becomes a 4-column
        ``(A1, B1, A2, B2)`` system over ring-mean residuals, and
        ``coeffs[b]`` is filled post-fit with the band's IVW (WLS) or
        simple (OLS) mean. ``cov`` for those rows is the band's own
        SEM² (no joint coupling). Renamed from the deprecated
        ``fix_per_band_background_to_zero=True`` (Section 6 cleanup).

    Returns
    -------
    coeffs : (B + 4,) float64
        Coefficient vector ``[I0_0, I0_1, ..., I0_{B-1}, A1, B1, A2, B2]``.
    cov : (B + 4, B + 4) float64 or None
        Covariance matrix from the joint solve. WLS: ``(A^T W A)^-1`` is
        the exact covariance. OLS: ``(A^T A)^-1`` (caller must scale by
        residual variance for true covariance). ``None`` on solver failure.
    wls_mode : bool
        True when ``variances_per_band`` was provided.

    Notes
    -----
    Decision D12: ``band_weights`` enter the joint normal equations as a
    diagonal weight matrix ``W = diag(w_eff)`` where each band's row block
    receives the band's scalar weight ``w_b`` (``w_eff = w_b`` in OLS,
    ``w_eff = w_b / variance_b(pixel)`` in WLS). The implementation
    forms ``A^T W A`` via the equivalent one-sided product
    ``AW.T @ A`` with ``AW = A * w_eff[:, None]``; this matches the
    standard ``A^T W A`` rather than a literal ``sqrt(W) A`` row scaling.
    """
    n_bands, n_samples = intens_per_band.shape

    # Per-row effective weights w_eff: in WLS, w_eff = w_b / variance_b(pixel).
    # In OLS, w_eff = w_b. Either way w_eff is a length (B*N) vector with
    # the band's scalar weight applied to every sample of that band.
    w_band_per_row = np.repeat(band_weights_arr, n_samples)  # (B*N,)
    if variances_per_band is not None:
        var_flat = variances_per_band.reshape(n_bands * n_samples)
        w_eff = w_band_per_row / var_flat
        wls_mode = True
    else:
        w_eff = w_band_per_row
        wls_mode = False

    if not fit_per_band_intens_jointly:
        # Drop the per-band intercept columns. Per-band ring means or
        # medians are computed up front and subtracted from the RHS so
        # the geometric 4-column solve fits residuals only.
        means = _per_band_mean_or_median(intens_per_band, variances_per_band, integrator)
        residuals = intens_per_band - means[:, None]
        y_geom = residuals.reshape(n_bands * n_samples)
        # Geometric block: drop the band-indicator columns from the joint
        # design matrix. The remaining 4 columns are identical for every
        # band so we can build them once and tile.
        A_full = build_joint_design_matrix(angles, n_bands)  # (B*N, B+4)
        A_geom = A_full[:, n_bands:]  # (B*N, 4)
        row_scale = _geometry_row_scaling(per_band_gradients, [n_samples] * n_bands)
        if row_scale is not None:
            A_geom = A_geom * row_scale[:, None]
        AW_geom = A_geom * w_eff[:, None]
        ATWA = AW_geom.T @ A_geom
        ATWy = AW_geom.T @ y_geom
        try:
            geom_coeffs = np.linalg.solve(ATWA, ATWy)
            geom_cov = np.linalg.inv(ATWA)
        except np.linalg.LinAlgError:
            geom_coeffs = np.zeros(4, dtype=np.float64)
            geom_cov = None

        coeffs = np.zeros(n_bands + 4, dtype=np.float64)
        coeffs[:n_bands] = means
        coeffs[n_bands:] = geom_coeffs

        if geom_cov is None:
            return coeffs, None, wls_mode

        cov = np.zeros((n_bands + 4, n_bands + 4), dtype=np.float64)
        cov[n_bands:, n_bands:] = geom_cov
        # Per-band intercept covariance: each band's ring SEM². Mirrors
        # the ref-mode B3 fix — the per-band intercept does not flow
        # through the joint solve so its covariance must come from the
        # band's own statistics, for the statistic it actually reported.
        geom_model = (A_geom @ geom_coeffs).reshape(n_bands, n_samples)
        for b in range(n_bands):
            cov[b, b] = _per_band_intercept_variance(
                intens_per_band[b],
                variances_per_band[b] if variances_per_band is not None else None,
                integrator,
                residuals_b=residuals[b] - geom_model[b],
            )
        return coeffs, cov, wls_mode

    # --- Default: full (B + 4)-column joint solve --- #
    A = build_joint_design_matrix(angles, n_bands).astype(np.float64, copy=True)  # (B*N, B+4)
    row_scale = _geometry_row_scaling(per_band_gradients, [n_samples] * n_bands)
    if row_scale is not None:
        A[:, n_bands:] *= row_scale[:, None]
    y = intens_per_band.reshape(n_bands * n_samples)
    AW = A * w_eff[:, None]
    ATWA = AW.T @ A
    ATWy = AW.T @ y
    try:
        coeffs = np.linalg.solve(ATWA, ATWy)
        cov = np.linalg.inv(ATWA)
        return coeffs, cov, wls_mode
    except np.linalg.LinAlgError:
        # Fallback: per-band means as I0_b, zeros for harmonic coefficients.
        fallback = np.zeros(n_bands + 4, dtype=np.float64)
        for b in range(n_bands):
            fallback[b] = float(np.mean(intens_per_band[b]))
        return fallback, None, wls_mode


def _per_band_mean_or_median_jagged(
    intens_per_band: List[NDArray[np.float64]],
    variances_per_band: Optional[List[NDArray[np.float64]]],
    integrator: str = "mean",
) -> NDArray[np.float64]:
    """Per-band intercept reducer for jagged inputs (loose validity).

    Same semantics as :func:`_per_band_mean_or_median` but accepts
    ragged per-band sample lists. Bands with zero surviving samples
    return NaN under both integrators.
    """
    n_bands = len(intens_per_band)
    out = np.empty(n_bands, dtype=np.float64)
    for b in range(n_bands):
        if intens_per_band[b].size == 0:
            out[b] = float("nan")
            continue
        if integrator == "median":
            out[b] = float(np.median(intens_per_band[b]))
        elif variances_per_band is None:
            out[b] = float(np.mean(intens_per_band[b]))
        else:
            w = 1.0 / variances_per_band[b]
            denom = float(np.sum(w))
            out[b] = float(np.sum(intens_per_band[b] * w) / denom) if denom > 0 else float("nan")
    return out


def fit_first_and_second_harmonics_joint_loose(
    phi_per_band: List[NDArray[np.float64]],
    intens_per_band: List[NDArray[np.float64]],
    band_weights_arr: NDArray[np.float64],
    variances_per_band: Optional[List[NDArray[np.float64]]] = None,
    *,
    normalize: bool = False,
    fit_per_band_intens_jointly: bool = True,
    integrator: str = "mean",
    per_band_gradients: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], bool]:
    """
    Loose-validity counterpart to :func:`fit_first_and_second_harmonics_joint`.

    Each band b contributes ``N_b`` rows to the jagged design matrix
    ``(Σ N_b, B + 4)``; per-band intercept columns are 1 only on that
    band's row block. ``band_weights_arr`` row-scaling and the
    ``per_band_count`` normalization both compose into the row weights
    so the math stays a single weighted-least-squares solve. WLS row
    weights divide by per-pixel variance; OLS uses ``w_b`` only.

    The ``fit_per_band_intens_jointly=False`` semantics mirror the shared
    path: subtract per-band ring means from RHS, drop intercept columns,
    fit the 4-column geometric system, and write per-band SEM into the
    intercept block of the returned covariance.

    Returns
    -------
    coeffs : (B + 4,) float64
        Coefficient vector (per-band intercepts then geometric block).
    cov : (B + 4, B + 4) float64 or None
    wls_mode : bool
    """
    n_bands = len(phi_per_band)
    n_per_band = np.array([int(p.size) for p in phi_per_band], dtype=np.int64)
    n_total = int(n_per_band.sum())
    wls_mode = variances_per_band is not None

    # Per-row band weights & per-pixel WLS weights composed together.
    band_weight_per_row = np.concatenate(
        [np.full(n_per_band[b], band_weights_arr[b], dtype=np.float64) for b in range(n_bands)]
    )
    if wls_mode:
        var_concat = np.concatenate(variances_per_band)  # type: ignore[arg-type]
        w_eff = band_weight_per_row / var_concat
    else:
        w_eff = band_weight_per_row

    # Apply per-band-count normalization (Q7-(b)). Multiplies each band's
    # row block by 1/N_b so the band's total contribution to A^T W A
    # equals w_b regardless of N_b. Composes multiplicatively with the
    # per-row WLS / band weight.
    if normalize:
        norm_per_row = np.concatenate(
            [np.full(n_per_band[b], 1.0 / max(n_per_band[b], 1), dtype=np.float64) for b in range(n_bands)]
        )
        w_eff = w_eff * norm_per_row

    if not fit_per_band_intens_jointly:
        means = _per_band_mean_or_median_jagged(intens_per_band, variances_per_band, integrator)
        residuals_per_band = [
            intens_per_band[b] - means[b] if intens_per_band[b].size else intens_per_band[b] for b in range(n_bands)
        ]
        y_geom = np.concatenate(residuals_per_band) if residuals_per_band else np.empty(0)
        # Geometric block only — no intercept columns. Build from each
        # band's own phi and stack.
        sin1 = np.concatenate([np.sin(p) for p in phi_per_band]) if n_total else np.empty(0)
        cos1 = np.concatenate([np.cos(p) for p in phi_per_band]) if n_total else np.empty(0)
        sin2 = np.concatenate([np.sin(2.0 * p) for p in phi_per_band]) if n_total else np.empty(0)
        cos2 = np.concatenate([np.cos(2.0 * p) for p in phi_per_band]) if n_total else np.empty(0)
        A_geom = np.column_stack([sin1, cos1, sin2, cos2])
        row_scale = _geometry_row_scaling(per_band_gradients, n_per_band)
        if row_scale is not None:
            A_geom = A_geom * row_scale[:, None]
        AW = A_geom * w_eff[:, None]
        ATWA = AW.T @ A_geom
        ATWy = AW.T @ y_geom
        try:
            geom_coeffs = np.linalg.solve(ATWA, ATWy)
            geom_cov = np.linalg.inv(ATWA)
        except np.linalg.LinAlgError:
            geom_coeffs = np.zeros(4, dtype=np.float64)
            geom_cov = None

        coeffs = np.zeros(n_bands + 4, dtype=np.float64)
        coeffs[:n_bands] = means
        coeffs[n_bands:] = geom_coeffs
        if geom_cov is None:
            return coeffs, None, wls_mode
        cov = np.zeros((n_bands + 4, n_bands + 4), dtype=np.float64)
        cov[n_bands:, n_bands:] = geom_cov
        geom_model_per_band = _split_per_band(A_geom @ geom_coeffs, n_per_band)
        for b in range(n_bands):
            cov[b, b] = _per_band_intercept_variance(
                intens_per_band[b],
                variances_per_band[b] if variances_per_band is not None else None,
                integrator,
                residuals_b=residuals_per_band[b] - geom_model_per_band[b],
            )
        return coeffs, cov, wls_mode

    # Default joint loose path: full (B + 4) column solve.
    A = build_joint_design_matrix_jagged(phi_per_band, n_bands, normalize=False).astype(np.float64, copy=True)
    row_scale = _geometry_row_scaling(per_band_gradients, n_per_band)
    if row_scale is not None:
        A[:, n_bands:] *= row_scale[:, None]
    y = np.concatenate(intens_per_band) if n_total else np.empty(0)
    AW = A * w_eff[:, None]
    ATWA = AW.T @ A
    ATWy = AW.T @ y
    try:
        coeffs = np.linalg.solve(ATWA, ATWy)
        cov = np.linalg.inv(ATWA)
        return coeffs, cov, wls_mode
    except np.linalg.LinAlgError:
        fallback = np.zeros(n_bands + 4, dtype=np.float64)
        for b in range(n_bands):
            if intens_per_band[b].size:
                fallback[b] = float(np.mean(intens_per_band[b]))
        return fallback, None, wls_mode


def fit_simultaneous_joint(
    angles: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    band_weights_arr: NDArray[np.float64],
    harmonic_orders: Sequence[int],
    variances_per_band: Optional[NDArray[np.float64]] = None,
    *,
    fit_per_band_intens_jointly: bool = True,
    integrator: str = "mean",
    per_band_gradients: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], bool]:
    """Joint solve over per-band ``I0_b`` + (A1,B1,A2,B2) + shared higher orders.

    Extends :func:`fit_first_and_second_harmonics_joint` with ``2*L`` extra
    columns for shared higher-order coefficients ``(A_n, B_n)`` per
    ``n in harmonic_orders``. Used by ``simultaneous_in_loop`` (called
    every iteration) and ``simultaneous_original`` (called once post-hoc).

    Returns
    -------
    coeffs : (B + 4 + 2*L,) float64
        ``[I0_0, ..., I0_{B-1}, A1, B1, A2, B2,
        A_{orders[0]}, B_{orders[0]}, A_{orders[1]}, B_{orders[1]}, ...]``.
    cov : (B + 4 + 2*L, B + 4 + 2*L) float64 or None
        Joint covariance. WLS: exact ``(A^T W A)^-1``. OLS: ``(A^T A)^-1``
        (caller scales by residual variance).
    wls_mode : bool
    """
    orders_arr = np.asarray(list(harmonic_orders), dtype=np.int64)
    L = int(orders_arr.size)
    n_bands, n_samples = intens_per_band.shape
    n_extra = 4 + 2 * L

    w_band_per_row = np.repeat(band_weights_arr, n_samples)
    if variances_per_band is not None:
        var_flat = variances_per_band.reshape(n_bands * n_samples)
        w_eff = w_band_per_row / var_flat
        wls_mode = True
    else:
        w_eff = w_band_per_row
        wls_mode = False

    if not fit_per_band_intens_jointly:
        means = _per_band_mean_or_median(intens_per_band, variances_per_band, integrator)
        residuals = intens_per_band - means[:, None]
        y_geom = residuals.reshape(n_bands * n_samples)
        A_full = build_joint_design_matrix_higher(angles, n_bands, orders_arr)
        A_geom = A_full[:, n_bands:]  # (B*N, 4 + 2L)
        row_scale = _geometry_row_scaling(per_band_gradients, [n_samples] * n_bands)
        if row_scale is not None:
            A_geom = A_geom * row_scale[:, None]
        AW_geom = A_geom * w_eff[:, None]
        ATWA = AW_geom.T @ A_geom
        ATWy = AW_geom.T @ y_geom
        try:
            geom_coeffs = np.linalg.solve(ATWA, ATWy)
            geom_cov = np.linalg.inv(ATWA)
        except np.linalg.LinAlgError:
            geom_coeffs = np.zeros(n_extra, dtype=np.float64)
            geom_cov = None

        coeffs = np.zeros(n_bands + n_extra, dtype=np.float64)
        coeffs[:n_bands] = means
        coeffs[n_bands:] = geom_coeffs
        if geom_cov is None:
            return coeffs, None, wls_mode
        cov = np.zeros((n_bands + n_extra, n_bands + n_extra), dtype=np.float64)
        cov[n_bands:, n_bands:] = geom_cov
        geom_model = (A_geom @ geom_coeffs).reshape(n_bands, n_samples)
        for b in range(n_bands):
            cov[b, b] = _per_band_intercept_variance(
                intens_per_band[b],
                variances_per_band[b] if variances_per_band is not None else None,
                integrator,
                residuals_b=residuals[b] - geom_model[b],
            )
        return coeffs, cov, wls_mode

    A = build_joint_design_matrix_higher(angles, n_bands, orders_arr).astype(np.float64, copy=True)
    row_scale = _geometry_row_scaling(per_band_gradients, [n_samples] * n_bands)
    if row_scale is not None:
        A[:, n_bands:] *= row_scale[:, None]
    y = intens_per_band.reshape(n_bands * n_samples)
    AW = A * w_eff[:, None]
    ATWA = AW.T @ A
    ATWy = AW.T @ y
    try:
        coeffs = np.linalg.solve(ATWA, ATWy)
        cov = np.linalg.inv(ATWA)
        return coeffs, cov, wls_mode
    except np.linalg.LinAlgError:
        fallback = np.zeros(n_bands + n_extra, dtype=np.float64)
        for b in range(n_bands):
            fallback[b] = float(np.mean(intens_per_band[b]))
        return fallback, None, wls_mode


def fit_simultaneous_joint_loose(
    phi_per_band: List[NDArray[np.float64]],
    intens_per_band: List[NDArray[np.float64]],
    band_weights_arr: NDArray[np.float64],
    harmonic_orders: Sequence[int],
    variances_per_band: Optional[List[NDArray[np.float64]]] = None,
    *,
    normalize: bool = False,
    fit_per_band_intens_jointly: bool = True,
    integrator: str = "mean",
    per_band_gradients: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], bool]:
    """Loose-validity higher-order joint solver.

    Extends :func:`fit_first_and_second_harmonics_joint_loose` with shared
    higher-order columns per ``harmonic_orders``. Required for
    ``simultaneous_*`` modes under ``loose_validity=True``.
    """
    orders_arr = np.asarray(list(harmonic_orders), dtype=np.int64)
    L = int(orders_arr.size)
    n_bands = len(phi_per_band)
    n_extra = 4 + 2 * L
    n_per_band = np.array([int(p.size) for p in phi_per_band], dtype=np.int64)
    n_total = int(n_per_band.sum())
    wls_mode = variances_per_band is not None

    band_weight_per_row = np.concatenate(
        [np.full(n_per_band[b], band_weights_arr[b], dtype=np.float64) for b in range(n_bands)]
    )
    if wls_mode:
        var_concat = np.concatenate(variances_per_band)  # type: ignore[arg-type]
        w_eff = band_weight_per_row / var_concat
    else:
        w_eff = band_weight_per_row

    if normalize:
        norm_per_row = np.concatenate(
            [np.full(n_per_band[b], 1.0 / max(n_per_band[b], 1), dtype=np.float64) for b in range(n_bands)]
        )
        w_eff = w_eff * norm_per_row

    if not fit_per_band_intens_jointly:
        means = _per_band_mean_or_median_jagged(intens_per_band, variances_per_band, integrator)
        residuals_per_band = [
            intens_per_band[b] - means[b] if intens_per_band[b].size else intens_per_band[b] for b in range(n_bands)
        ]
        y_geom = np.concatenate(residuals_per_band) if residuals_per_band else np.empty(0)
        A_full = build_joint_design_matrix_jagged_higher(
            phi_per_band,
            n_bands,
            orders_arr,
            normalize=False,
        )
        A_geom = A_full[:, n_bands:]
        row_scale = _geometry_row_scaling(per_band_gradients, n_per_band)
        if row_scale is not None:
            A_geom = A_geom * row_scale[:, None]
        AW_geom = A_geom * w_eff[:, None]
        ATWA = AW_geom.T @ A_geom
        ATWy = AW_geom.T @ y_geom
        try:
            geom_coeffs = np.linalg.solve(ATWA, ATWy)
            geom_cov = np.linalg.inv(ATWA)
        except np.linalg.LinAlgError:
            geom_coeffs = np.zeros(n_extra, dtype=np.float64)
            geom_cov = None

        coeffs = np.zeros(n_bands + n_extra, dtype=np.float64)
        coeffs[:n_bands] = means
        coeffs[n_bands:] = geom_coeffs
        if geom_cov is None:
            return coeffs, None, wls_mode
        cov = np.zeros((n_bands + n_extra, n_bands + n_extra), dtype=np.float64)
        cov[n_bands:, n_bands:] = geom_cov
        geom_model_per_band = _split_per_band(A_geom @ geom_coeffs, n_per_band)
        for b in range(n_bands):
            cov[b, b] = _per_band_intercept_variance(
                intens_per_band[b],
                variances_per_band[b] if variances_per_band is not None else None,
                integrator,
                residuals_b=residuals_per_band[b] - geom_model_per_band[b],
            )
        return coeffs, cov, wls_mode

    A = build_joint_design_matrix_jagged_higher(
        phi_per_band,
        n_bands,
        orders_arr,
        normalize=False,
    ).astype(np.float64, copy=True)
    row_scale = _geometry_row_scaling(per_band_gradients, n_per_band)
    if row_scale is not None:
        A[:, n_bands:] *= row_scale[:, None]
    y = np.concatenate(intens_per_band) if n_total else np.empty(0)
    AW = A * w_eff[:, None]
    ATWA = AW.T @ A
    ATWy = AW.T @ y
    try:
        coeffs = np.linalg.solve(ATWA, ATWy)
        cov = np.linalg.inv(ATWA)
        return coeffs, cov, wls_mode
    except np.linalg.LinAlgError:
        fallback = np.zeros(n_bands + n_extra, dtype=np.float64)
        for b in range(n_bands):
            if intens_per_band[b].size:
                fallback[b] = float(np.mean(intens_per_band[b]))
        return fallback, None, wls_mode


def fit_first_and_second_harmonics_ref(
    angles: NDArray[np.float64],
    intens_ref: NDArray[np.float64],
    variances_ref: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], bool]:
    """
    Reference-band-only fallback for ``harmonic_combination='ref'``.

    Wraps :func:`isoster.numba_kernels.build_harmonic_matrix` and the
    standard 5-param solve so the calling iteration loop sees the same
    coefficient layout as the joint solver: a ``(B + 4,)`` vector with
    placeholder ``I0_b = mean(intens_b)`` for every non-reference band.

    Returns
    -------
    coeffs : (5,)
        ``[I0_ref, A1, B1, A2, B2]``. Caller widens to ``(B + 4,)`` by
        filling per-band means.
    cov : (5, 5) or None
    wls_mode : bool
    """
    A = build_harmonic_matrix(angles)
    if variances_ref is not None:
        weights = 1.0 / variances_ref
        AW = A * weights[:, None]
        ATWA = AW.T @ A
        ATWy = AW.T @ intens_ref
        try:
            coeffs = np.linalg.solve(ATWA, ATWy)
            cov = np.linalg.inv(ATWA)
            return coeffs, cov, True
        except np.linalg.LinAlgError:
            return np.array([np.mean(intens_ref), 0.0, 0.0, 0.0, 0.0]), None, True
    try:
        coeffs, _residuals, _rank, _sv = np.linalg.lstsq(A, intens_ref, rcond=None)
        cov = np.linalg.inv(A.T @ A)
        return coeffs, cov, False
    except np.linalg.LinAlgError:
        return np.array([np.mean(intens_ref), 0.0, 0.0, 0.0, 0.0]), None, False


def evaluate_joint_model(
    angles: NDArray[np.float64],
    coeffs: NDArray[np.float64],
    n_bands: int,
    harmonic_orders: Optional[Sequence[int]] = None,
) -> NDArray[np.float64]:
    """Evaluate the joint model intensities for every band at every angle.

    With ``harmonic_orders`` left at the default ``None`` (or empty), the
    model is the standard 5-parameter geometric form ``I0_b + A1·sin(φ)
    + B1·cos(φ) + A2·sin(2φ) + B2·cos(2φ)``. When orders are supplied,
    extra shared terms ``Σ A_n·sin(nφ) + B_n·cos(nφ)`` are added,
    matching the layout of :func:`fit_simultaneous_joint`. ``coeffs``
    must have shape ``(B + 4 + 2*len(harmonic_orders),)``.

    Returns shape ``(B, N)``: each row b is
    ``I0_b + A1·sin(φ) + B1·cos(φ) + A2·sin(2φ) + B2·cos(2φ)``.
    """
    A1, B1, A2, B2 = coeffs[n_bands], coeffs[n_bands + 1], coeffs[n_bands + 2], coeffs[n_bands + 3]
    geom = A1 * np.sin(angles) + B1 * np.cos(angles) + A2 * np.sin(2.0 * angles) + B2 * np.cos(2.0 * angles)
    if harmonic_orders:
        for j, n_order in enumerate(harmonic_orders):
            an = float(coeffs[n_bands + 4 + 2 * j])
            bn = float(coeffs[n_bands + 4 + 2 * j + 1])
            geom = geom + an * np.sin(int(n_order) * angles) + bn * np.cos(int(n_order) * angles)
    out = np.empty((n_bands, len(angles)), dtype=np.float64)
    for b in range(n_bands):
        out[b] = coeffs[b] + geom
    return out


# ---------------------------------------------------------------------------
# Per-band sigma clipping (decision D9)
# ---------------------------------------------------------------------------


def _per_band_sigma_clip(
    angles: NDArray[np.float64],
    phi: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    variances_per_band: Optional[NDArray[np.float64]],
    sclip: float,
    nclip: int,
    sclip_low: Optional[float],
    sclip_high: Optional[float],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Optional[NDArray[np.float64]],
    int,
]:
    """
    Per-band sigma clipping with shared-validity AND across bands.

    Each band is clipped independently against its own intensity
    statistics; the resulting per-band survivor masks are AND-ed and
    applied to angles, phi, and every band's intens/variance arrays.
    Reduces to the existing single-band clip when B=1.
    """
    n_bands, n_samples = intens_per_band.shape
    if nclip <= 0 or n_samples == 0:
        return angles, phi, intens_per_band, variances_per_band, 0

    keep = np.ones(n_samples, dtype=bool)
    for b in range(n_bands):
        # Use the existing single-band sigma_clip on (angles, intens_b)
        # to get an index mask back. sigma_clip returns clipped arrays,
        # not a mask, so we re-derive by tracking which samples survive
        # via index alignment.
        idx = np.arange(n_samples)
        clipped = sigma_clip(
            idx.astype(np.float64),
            intens_per_band[b].copy(),
            sclip=sclip,
            nclip=nclip,
            sclip_low=sclip_low,
            sclip_high=sclip_high,
        )
        idx_keep = clipped[0].astype(np.int64)
        survivor = np.zeros(n_samples, dtype=bool)
        survivor[idx_keep] = True
        keep &= survivor

    if keep.all():
        return angles, phi, intens_per_band, variances_per_band, 0

    n_clipped = int(n_samples - keep.sum())
    intens_clipped = intens_per_band[:, keep]
    variances_clipped: Optional[NDArray[np.float64]]
    variances_clipped = variances_per_band[:, keep] if variances_per_band is not None else None
    return angles[keep], phi[keep], intens_clipped, variances_clipped, n_clipped


def _per_band_sigma_clip_loose(
    phi_per_band: List[NDArray[np.float64]],
    intens_per_band: List[NDArray[np.float64]],
    variances_per_band: Optional[List[NDArray[np.float64]]],
    sclip: float,
    nclip: int,
    sclip_low: Optional[float],
    sclip_high: Optional[float],
) -> Tuple[
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    Optional[List[NDArray[np.float64]]],
    int,
]:
    """
    Independent per-band sigma clipping for loose-validity (no AND across bands).

    Each band's clip uses only its own surviving samples; bands do not
    propagate clip rejections to each other (decision Q6 of the D9
    backport interview).
    """
    n_bands = len(intens_per_band)
    out_phi: List[NDArray[np.float64]] = []
    out_intens: List[NDArray[np.float64]] = []
    out_variances: Optional[List[NDArray[np.float64]]] = [] if variances_per_band is not None else None
    n_clipped_total = 0

    for b in range(n_bands):
        n_b = int(intens_per_band[b].size)
        if nclip <= 0 or n_b == 0:
            out_phi.append(phi_per_band[b])
            out_intens.append(intens_per_band[b])
            if out_variances is not None:
                out_variances.append(variances_per_band[b])  # type: ignore[index]
            continue
        idx = np.arange(n_b)
        clipped = sigma_clip(
            idx.astype(np.float64),
            intens_per_band[b].copy(),
            sclip=sclip,
            nclip=nclip,
            sclip_low=sclip_low,
            sclip_high=sclip_high,
        )
        idx_keep = clipped[0].astype(np.int64)
        keep = np.zeros(n_b, dtype=bool)
        keep[idx_keep] = True
        n_clipped_total += int(n_b - keep.sum())
        out_phi.append(phi_per_band[b][keep])
        out_intens.append(intens_per_band[b][keep])
        if out_variances is not None:
            out_variances.append(variances_per_band[b][keep])  # type: ignore[index]
    return out_phi, out_intens, out_variances, n_clipped_total


# ---------------------------------------------------------------------------
# Combined gradient (decision D10)
# ---------------------------------------------------------------------------


def _geometry_row_scaling(
    per_band_gradients: Optional[NDArray[np.float64]],
    samples_per_band: Sequence[int],
) -> Optional[NDArray[np.float64]]:
    """Per-row gradient factors that turn shared amplitudes into geometry steps.

    A common geometry step ``delta`` produces amplitude ``delta * grad_b`` in
    band ``b``, so scaling every *shared* design column by that band's gradient
    makes the shared free parameters the geometry steps themselves. The per-band
    intercept columns are never scaled — ``I0_b`` stays an intensity.

    Returns ``None`` when no gradients were supplied, so callers can skip the
    multiply entirely and stay byte-identical to the amplitude parameterisation.
    Non-finite gradients become 0.0: a band with no measurable radial gradient
    carries no geometric leverage, which is the correct contribution rather than
    a NaN that would poison the solve.
    """
    if per_band_gradients is None:
        return None
    gradients = np.asarray(per_band_gradients, dtype=np.float64).reshape(-1)
    gradients = np.where(np.isfinite(gradients), gradients, 0.0)
    return np.repeat(gradients, np.asarray(samples_per_band, dtype=np.intp))


def fit_first_and_second_harmonics_geometry(
    angles: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    band_weights_arr: NDArray[np.float64],
    per_band_gradients: NDArray[np.float64],
    variances_per_band: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], bool]:
    """Geometry-parameterised joint solve: shared parameters are geometry steps.

    The standard joint solver fits one shared *amplitude* across bands. But a
    common geometry step ``delta`` produces amplitude ``delta * grad_b`` in band
    ``b``, so a single shared amplitude is a misspecified model whenever the
    bands have different gradients: what it fits is ``delta`` times a
    weight-averaged gradient, which then has to be divided back out by a
    separately pooled gradient.

    Scaling each band's harmonic columns by ``grad_b`` instead makes the shared
    parameters the geometry steps themselves:

        y_{b,i} = I0_b + grad_b * (D1 sin p + E1 cos p + D2 sin 2p + E2 cos 2p)

    Three consequences, all of which remove a defect rather than trade one off:

    * No division by a pooled gradient, so no way for the two to disagree.
    * The effective weight on each geometry parameter becomes
      ``w_b * grad_b**2 / var_b`` — the minimum-variance weighting — instead of
      ``w_b / var_b``, which on real HSC data was measured close to *inverted*
      against the bands' information content (band g: 65% of the weight, 10% of
      the information).
    * The shared coefficients and the residual scatter end up in the same units,
      so the convergence comparison stops depending on a band's flux units.

    Returns the same ``(coeffs, cov, wls_mode)`` triple as
    :func:`fit_first_and_second_harmonics_joint`, with ``coeffs`` laid out as
    ``[I0_0, ..., I0_{B-1}, D1, E1, D2, E2]``. A band whose gradient is zero or
    non-finite contributes zero columns and therefore no geometry information,
    which is the correct behaviour: no measurable radial gradient means no
    geometric leverage. If *every* band is degenerate the normal equations are
    singular and the fallback path returns per-band means with zero geometry.
    """
    n_bands, n_samples = intens_per_band.shape
    gradients = np.asarray(per_band_gradients, dtype=np.float64).reshape(-1)
    gradients = np.where(np.isfinite(gradients), gradients, 0.0)

    w_band_per_row = np.repeat(band_weights_arr, n_samples)
    if variances_per_band is not None:
        w_eff = w_band_per_row / variances_per_band.reshape(n_bands * n_samples)
        wls_mode = True
    else:
        w_eff = w_band_per_row
        wls_mode = False

    design = build_joint_design_matrix(angles, n_bands).astype(np.float64, copy=True)
    # Scale only the four shared geometric columns, per band row-block. The
    # per-band intercept columns are untouched: I0_b is still an intensity.
    design[:, n_bands:] *= np.repeat(gradients, n_samples)[:, None]

    y = intens_per_band.reshape(n_bands * n_samples)
    design_weighted = design * w_eff[:, None]
    normal_matrix = design_weighted.T @ design
    normal_rhs = design_weighted.T @ y
    try:
        coeffs = np.linalg.solve(normal_matrix, normal_rhs)
        cov = np.linalg.inv(normal_matrix)
        return coeffs, cov, wls_mode
    except np.linalg.LinAlgError:
        fallback = np.zeros(n_bands + 4, dtype=np.float64)
        for b in range(n_bands):
            fallback[b] = float(np.mean(intens_per_band[b]))
        return fallback, None, wls_mode


def _band_ring(data: MultiIsophoteData, band: int):
    """One band's kept samples and variances, in either validity layout.

    Under loose validity the bands no longer share a sample set, so the
    rectangular ``intens`` field holds only the cross-band intersection and the
    per-band lists carry the truth. Reading the rectangular view there would
    measure a gradient on samples other bands happened to share.
    """
    if data.intens_per_band is not None:
        intens = data.intens_per_band[band]
        variances = data.variances_per_band[band] if data.variances_per_band is not None else None
        return intens, variances
    return data.intens[band], (data.variances[band] if data.variances is not None else None)


def _ring_is_empty(data: MultiIsophoteData) -> bool:
    """True when no band has a usable ring.

    Under shared validity that is the intersection being empty. Under loose
    validity the isophote survives as long as *some* band has samples: a band
    fully masked at this radius must not discard the bands that are clean, which
    is the same rule the forced-photometry path applies.
    """
    if data.intens_per_band is not None:
        return all(arr.size == 0 for arr in data.intens_per_band)
    return data.intens.shape[1] == 0


def joint_gradient_pooling_weights(
    band_weights_arr: NDArray[np.float64],
    variances_per_band: Union[None, NDArray[np.float64], List[NDArray[np.float64]]],
    n_bands: int,
    band_indices: Optional[Sequence[int]] = None,
) -> NDArray[np.float64]:
    """Per-band weights for pooling gradients, matching the harmonic solve.

    The joint solve weights every sample by ``w_b / var_{b,i}``, so band ``b``
    contributes total weight ``W_b = w_b * sum_i (1 / var_{b,i})``. A common
    geometry step ``delta`` produces amplitude ``delta * grad_b`` in band ``b``,
    so the shared fitted amplitude is ``delta * sum(W_b grad_b) / sum(W_b)``.
    Dividing that by a gradient pooled with the *same* ``W_b`` returns ``delta``
    exactly; pooling with the bare ``w_b`` does not, and the error is not even
    one-signed — measured recoveries of 0.180 and 0.045 against a truth of 0.100
    at different band configurations.

    Under OLS there is no variance map, the solve weights by ``w_b`` alone, and
    this returns ``band_weights_arr`` unchanged.

    ``variances_per_band`` must be the **post-clipping** arrays handed to the
    solve. Taking them from the sampler's unfiltered ring would put the pooled
    gradient and the fitted amplitude back on different sample sets, which is
    the class of bug this whole exercise removes.

    ``band_indices`` maps jagged entries to their position in the full band list
    under loose validity with dropped bands; absent bands keep their bare weight,
    which is unused because they do not enter the solve.

    Exactness
    ---------
    The exactly-correct weight is ``sum_i w_b * sin^2(phi_i) / var_{b,i}``, which
    differs per harmonic when the variance varies *azimuthally within* a band —
    no single scalar can then be right for all of ``A1, B1, A2, B2`` at once.
    ``W_b`` as defined here is exact for shared angular coverage with
    azimuthally-uniform variance inside each band (the common case: bands differ
    in sky level, each band's own ring is uniform) and an approximation
    otherwise. See ``docs/10-multiband.md``, "Joint gradient weighting".
    """
    weights = np.array(band_weights_arr, dtype=np.float64, copy=True)
    if variances_per_band is None:
        return weights
    for position in range(len(variances_per_band)):
        var_b = variances_per_band[position]
        if var_b is None or np.size(var_b) == 0:
            continue
        b_idx = int(band_indices[position]) if band_indices is not None else position
        if b_idx >= n_bands:
            continue
        inverse_variance_sum = float(np.sum(1.0 / np.asarray(var_b, dtype=np.float64)))
        if np.isfinite(inverse_variance_sum) and inverse_variance_sum > 0.0:
            weights[b_idx] = float(band_weights_arr[b_idx]) * inverse_variance_sum
    return weights


def compute_joint_gradient(
    image_stack: NDArray[np.float64],
    masks_resolved: List[Optional[NDArray[np.float64]]],
    var_stack: Optional[NDArray[np.float64]],
    geometry: Dict[str, float],
    config: IsosterConfigMB,
    band_weights_arr: NDArray[np.float64],
    previous_gradient: Optional[float] = None,
    current_data: Optional[MultiIsophoteData] = None,
    pooling_weights: Optional[NDArray[np.float64]] = None,
) -> Tuple[float, Optional[float], List[float], List[Optional[float]]]:
    """
    Compute one combined-scalar radial gradient from multiple bands.

    Mirrors the single-band ``compute_gradient`` refinement logic (M9):
    when the first joint gradient looks anomalous (relative error >= 0.3
    and it falls short of a third of the previous gradient), the annulus
    is resampled at 2x step and the longer baseline is used; runaway
    gradients (>= previous/3) are decayed to ``0.8 * previous_gradient``
    with the error cleared.

    ``pooling_weights`` are the per-band weights used to combine the per-band
    gradients into the joint scalar. They must match the weights each band
    carried in the harmonic solve whose coefficients this gradient will divide,
    because a common geometry step ``delta`` produces amplitude
    ``delta * grad_b`` in band ``b``, so the shared fitted amplitude is
    ``delta`` times the *weight-averaged* gradient. Pooling the gradient with
    different weights leaves the variance factors uncancelled: on an exactly
    solvable two-band ring with band-dependent variance, a truth of 0.100 was
    recovered as 0.180. See :func:`joint_gradient_pooling_weights`. ``None``
    falls back to ``band_weights_arr``, which is exactly right under OLS, where
    the solve also weights by ``w_b`` alone.

    Returns
    -------
    gradient_joint : float
        Weighted-mean gradient ``Σ W_b grad_b / Σ W_b``.
    gradient_error_joint : float or None
        ``sqrt(Σ W_b^2 σ_b^2 / (Σ W_b)^2)``, or None when no per-band
        errors are available.
    per_band_gradients : list[float]
        Length B. Each entry is ``grad_b`` (raw per-band gradient).
    per_band_gradient_errors : list[float|None]
        Length B. Each entry is ``σ_b`` (per-band gradient error) or None.
    """
    x0 = geometry["x0"]
    y0 = geometry["y0"]
    sma = geometry["sma"]
    eps = geometry["eps"]
    pa = geometry["pa"]
    # Sample the gradient rings the same way the harmonic solve sampled its own,
    # so the numerator and the denominator of every geometry correction describe
    # one sample set. Sampling shared-validity under loose validity also made a
    # single fully-masked band collapse the joint gradient to a sentinel.
    loose_validity = bool(config.loose_validity)

    # Sample at the current SMA (reuse cached data when available).
    if current_data is not None:
        data_c = current_data
    else:
        data_c = extract_isophote_data_multi_prepared(
            image_stack,
            masks_resolved,
            var_stack,
            x0,
            y0,
            sma,
            eps,
            pa,
            use_eccentric_anomaly=config.use_eccentric_anomaly,
            loose_validity=loose_validity,
        )

    if _ring_is_empty(data_c):
        if previous_gradient is not None:
            return previous_gradient * 0.8, None, [], []
        return -1.0, None, [], []

    # Sample at sma + step (linear or geometric).
    if config.linear_growth:
        gradient_sma = sma + config.astep
    else:
        gradient_sma = sma * (1.0 + config.astep)

    data_g = extract_isophote_data_multi_prepared(
        image_stack,
        masks_resolved,
        var_stack,
        x0,
        y0,
        gradient_sma,
        eps,
        pa,
        use_eccentric_anomaly=config.use_eccentric_anomaly,
        loose_validity=loose_validity,
    )

    if _ring_is_empty(data_g):
        if previous_gradient is not None:
            return previous_gradient * 0.8, None, [], []
        return -1.0, None, [], []

    delta_r = config.astep if config.linear_growth else sma * config.astep
    n_bands = len(data_c.intens_per_band) if data_c.intens_per_band is not None else data_c.intens.shape[0]

    def _per_band_gradient(data_g, delta_r_b):
        """Per-band two-point gradients and errors against the current ring.

        Each ring's uncertainty comes from the statistic the gradient actually
        used, through the same shared helper as single-band ``compute_gradient``.
        The retired code paired a reported median with the variance of an
        inverse-variance-weighted mean, which describes a different estimator,
        and under OLS it dropped the median's pi/2 penalty. An infinite ring
        variance means no usable uncertainty rather than an enormous one.
        """
        grads: List[float] = []
        errs: List[Optional[float]] = []
        for b in range(n_bands):
            intens_c_b, var_c_b = _band_ring(data_c, b)
            intens_g_b, var_g_b = _band_ring(data_g, b)
            mean_c, var_mean_c = _ring_statistic_and_variance(intens_c_b, var_c_b, config.integrator)
            mean_g, var_mean_g = _ring_statistic_and_variance(intens_g_b, var_g_b, config.integrator)
            if intens_c_b.size == 0 or intens_g_b.size == 0:
                # This band has no usable ring at one of the two radii, while
                # others may. Contribute nothing rather than a NaN gradient.
                grads.append(float("nan"))
                errs.append(None)
                continue
            grads.append((mean_g - mean_c) / delta_r_b)
            if np.isfinite(var_mean_c) and np.isfinite(var_mean_g):
                errs.append(float(np.sqrt(var_mean_c + var_mean_g)) / delta_r_b)
            else:
                errs.append(None)
        return grads, errs

    pool_w = band_weights_arr if pooling_weights is None else np.asarray(pooling_weights, dtype=np.float64)

    def _pool(grads, errs):
        """Weight-combine per-band gradients into the joint scalar + error.

        gradient_joint = Σ W_b grad_b / Σ W_b
        σ²_joint = Σ W²_b σ²_b / (Σ W_b)²   (independent measurements)

        ``W_b`` is ``pooling_weights`` — the weight each band carried in the
        harmonic solve — not the bare band weight, so that the shared amplitude
        and this gradient describe the same weighted average of the bands.
        """
        grad_arr = np.asarray(grads, dtype=np.float64)
        # Under loose validity a band can have no usable ring at this radius
        # while others do; it contributes nothing rather than poisoning the sum.
        usable = np.isfinite(grad_arr)
        if not usable.any():
            return float("nan"), None
        weights = pool_w[usable]
        w_sum = float(weights.sum())
        if w_sum <= 0.0:
            return float("nan"), None
        grad = float(np.sum(weights * grad_arr[usable])) / w_sum
        contributing_errs = [e for e, keep in zip(errs, usable) if keep]
        if any(e is None for e in contributing_errs):
            return grad, None
        err_arr = np.array(contributing_errs, dtype=np.float64)
        var_joint = float(np.sum((weights**2) * (err_arr**2))) / (w_sum**2)
        return grad, float(np.sqrt(var_joint))

    per_band_grad, per_band_err = _per_band_gradient(data_g, delta_r)
    grad_joint, grad_err_joint = _pool(per_band_grad, per_band_err)

    if previous_gradient is None:
        previous_gradient = grad_joint + grad_err_joint if grad_err_joint is not None else grad_joint

    # EFF-1 (single-band parity, review M9): when the first joint gradient
    # looks anomalous, resample at 2x step and use the longer baseline.
    relative_error = (
        abs(grad_err_joint / grad_joint)
        if (grad_err_joint is not None and grad_joint is not None and grad_joint != 0)
        else np.inf
    )
    need_second_gradient = (grad_joint >= (previous_gradient / 3.0)) and (relative_error >= 0.3)

    if need_second_gradient:
        if config.linear_growth:
            gradient_sma_2 = sma + 2 * config.astep
        else:
            gradient_sma_2 = sma * (1.0 + 2 * config.astep)
        data_g2 = extract_isophote_data_multi_prepared(
            image_stack,
            masks_resolved,
            var_stack,
            x0,
            y0,
            gradient_sma_2,
            eps,
            pa,
            use_eccentric_anomaly=config.use_eccentric_anomaly,
            loose_validity=loose_validity,
        )
        if not _ring_is_empty(data_g2):
            delta_r_2 = 2 * config.astep if config.linear_growth else sma * 2 * config.astep
            per_band_grad, per_band_err = _per_band_gradient(data_g2, delta_r_2)
            grad_joint, grad_err_joint = _pool(per_band_grad, per_band_err)

    # Runaway-gradient decay (single-band parity, review M9).
    if grad_joint >= (previous_gradient / 3.0):
        grad_joint = previous_gradient * 0.8
        grad_err_joint = None

    return grad_joint, grad_err_joint, per_band_grad, per_band_err


# ---------------------------------------------------------------------------
# Parameter errors from the joint covariance matrix
# ---------------------------------------------------------------------------


def _compute_parameter_errors_from_joint(
    coeffs: NDArray[np.float64],
    cov_full: Optional[NDArray[np.float64]],
    n_bands: int,
    sma: float,
    eps: float,
    pa: float,
    gradient: float,
    gradient_error: Optional[float],
    angles: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    use_exact_covariance: bool,
    var_residual_floor: Optional[float],
    band_weights_arr: Optional[NDArray[np.float64]] = None,
    band_indices: Optional[Sequence[int]] = None,
    harmonic_orders: Optional[Sequence[int]] = None,
    residual_variance: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Map the joint (B+4)x(B+4) covariance into geometric parameter errors.

    The shared geometric coefficients (A1, B1, A2, B2) live at indices
    ``[n_bands : n_bands+4]`` of the joint coefficient vector. Their
    diagonal variances (after residual-variance scaling in OLS mode)
    feed the standard Jedrzejewski-1987 error propagation, identical
    to single-band. Returns (x0_err, y0_err, eps_err, pa_err).

    ``band_weights_arr``: per-band weights of the joint solve, needed for
    the weight-aware OLS residual-variance rescale (review M2). Full
    (unsubsetted) array. ``band_indices``: full-list band index for each
    jagged entry under loose validity with dropped bands (review M3).

    ``harmonic_orders`` must be passed whenever the iteration loop solved the
    wider ``simultaneous_in_loop`` system, so this rescale measures the same
    model, and counts the same parameters, as the higher-order attacher does.
    Without it the geometry errors and the harmonic errors from one solve were
    scaled by two different residual variances.

    ``residual_variance`` supplies the OLS scale directly, already floored, for
    callers whose covariance did not come from a pooled all-band solve —
    ``harmonic_combination='ref'`` being the case that needs it. The pooled
    helper is then not called at all. Same role as the ``residual_variance=``
    keyword on single-band ``compute_parameter_errors``.
    """
    if cov_full is None or gradient is None or abs(gradient) < 1e-10:
        return 0.0, 0.0, 0.0, 0.0
    # `intens_per_band` may be a rectangular ndarray (shared validity)
    # or a list of jagged per-band arrays (loose validity). Both shapes
    # carry a meaningful total pixel count; we accept either.
    if isinstance(intens_per_band, np.ndarray):
        n_pixels = int(intens_per_band.size)
    else:
        n_pixels = int(sum(arr.size for arr in intens_per_band))
    # Fitted-parameter count for the OLS degrees of freedom: surviving
    # per-band intercepts, 4 shared geometric coefficients, and the shared
    # higher-order pair per order when the loop solved the wider system.
    n_higher_params = 2 * len(harmonic_orders) if harmonic_orders else 0
    n_geom_params = (len(band_indices) if band_indices is not None else n_bands) + 4 + n_higher_params
    if n_pixels <= n_geom_params:
        return 0.0, 0.0, 0.0, 0.0

    g_err_sq = gradient_error**2 if gradient_error is not None else 0.0
    g_sq = gradient**2

    try:
        if use_exact_covariance:
            covariance = cov_full
        elif residual_variance is not None:
            # Caller-supplied scale, already floored: the covariance did not come
            # from a pooled all-band solve, so pooling residuals here would let
            # bands outside the solve set its noise level.
            if not np.isfinite(residual_variance):
                return 0.0, 0.0, 0.0, 0.0
            covariance = cov_full * float(residual_variance)
        else:
            # OLS: scale the inverse normal equations by the weight-aware
            # residual variance from the shared helper (review M2/M3).
            # Previously the shared-validity branch used an unweighted
            # np.var (mis-scaled by ~1/w under non-unit band weights) and
            # the loose-validity branch skipped the rescale entirely.
            var_residual = _compute_joint_residual_variance(
                coeffs,
                angles,
                intens_per_band,
                band_weights_arr,
                n_bands=n_bands,
                n_geom_params=n_geom_params,
                jagged=not isinstance(intens_per_band, np.ndarray),
                band_indices=band_indices,
                floor=var_residual_floor,
                harmonic_orders=harmonic_orders,
            )
            if var_residual is None or not np.isfinite(var_residual):
                return 0.0, 0.0, 0.0, 0.0
            covariance = cov_full * var_residual
        errors = np.sqrt(np.diagonal(covariance))

        sig_a1_sq = float(errors[n_bands] ** 2)
        sig_b1_sq = float(errors[n_bands + 1] ** 2)
        sig_a2_sq = float(errors[n_bands + 2] ** 2)
        sig_b2_sq = float(errors[n_bands + 3] ** 2)

        a1 = float(coeffs[n_bands])
        b1 = float(coeffs[n_bands + 1])
        a2 = float(coeffs[n_bands + 2])
        b2 = float(coeffs[n_bands + 3])

        var_major = (sig_b1_sq + (b1**2 / g_sq) * g_err_sq) / g_sq
        var_minor = ((1.0 - eps) ** 2 / g_sq) * (sig_a1_sq + (a1**2 / g_sq) * g_err_sq)

        x0_err = float(np.sqrt(var_minor * np.sin(pa) ** 2 + var_major * np.cos(pa) ** 2))
        y0_err = float(np.sqrt(var_minor * np.cos(pa) ** 2 + var_major * np.sin(pa) ** 2))

        var_eps = (2.0 * (1.0 - eps) / (sma * gradient)) ** 2 * (sig_b2_sq + (b2**2 / g_sq) * g_err_sq)
        eps_err = float(np.sqrt(var_eps))

        if abs(eps) > np.finfo(float).resolution:
            denom = (1.0 - eps) ** 2 - 1.0
            if abs(denom) < 1e-10:
                denom = -1e-10
            var_pa = (2.0 * (1.0 - eps) / (sma * gradient * denom)) ** 2 * (sig_a2_sq + (a2**2 / g_sq) * g_err_sq)
            pa_err = float(np.sqrt(var_pa))
        else:
            pa_err = 0.0

        return x0_err, y0_err, eps_err, pa_err
    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(
            f"_compute_parameter_errors_from_joint failed: {e}. Returning zero errors.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Per-isophote multi-band fit
# ---------------------------------------------------------------------------


def _stamp_n_valid_per_band(geom: Dict[str, object], bands: Sequence[str], counts: NDArray[np.int64]) -> None:
    """Stamp per-band surviving-sample counts (``n_valid_<b>``) onto a row."""
    for b_idx, b in enumerate(bands):
        geom[f"n_valid_{b}"] = int(counts[b_idx])


def _empty_isophote_dict(
    sma: float,
    x0: float,
    y0: float,
    eps: float,
    pa: float,
    bands: Sequence[str],
    use_eccentric_anomaly: bool,
    stop_code: int,
    niter: int,
    debug: bool,
    harmonic_orders: Sequence[int] = (3, 4),
) -> Dict[str, object]:
    """Build a degenerate isophote row with NaN intensities for every band.

    ``harmonic_orders`` controls which per-band ``a{n}_<b>`` /
    ``b{n}_<b>`` (and matching ``_err``) columns get zero-initialized.
    Defaults to ``(3, 4)`` so existing call sites that do not yet thread
    the config through retain Stage-1 behavior.
    """
    row: Dict[str, object] = {
        "sma": sma,
        "x0": x0,
        "y0": y0,
        "eps": eps,
        "pa": pa,
        "x0_err": 0.0,
        "y0_err": 0.0,
        "eps_err": 0.0,
        "pa_err": 0.0,
        "rms": float("nan"),
        "stop_code": stop_code,
        "niter": niter,
        "valid": False,
        "use_eccentric_anomaly": use_eccentric_anomaly,
        "tflux_e": float("nan"),
        "tflux_c": float("nan"),
        "npix_e": 0,
        "npix_c": 0,
        "ndata": 0,
        "nflag": 0,
    }
    harmonic_keys = _per_band_harmonic_keys_for_orders(harmonic_orders)
    for b in bands:
        for key in _PER_BAND_INTENSITY_KEYS:
            row[f"{key}_{b}"] = float("nan")
        for key in harmonic_keys:
            row[f"{key}_{b}"] = 0.0
        if debug:
            for key in _PER_BAND_DEBUG_KEYS:
                row[f"{key}_{b}"] = float("nan")
        # D9 backport: per-band surviving-sample count (zero on the
        # degenerate "no fit" path).
        row[f"n_valid_{b}"] = 0
    return row


def fit_isophote_mb(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    sma: float,
    start_geometry: Dict[str, float],
    config: IsosterConfigMB,
    going_inwards: bool = False,
    previous_geometry: Optional[Dict[str, float]] = None,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
    *,
    image_stack: Optional[NDArray[np.float64]] = None,
    masks_resolved: Optional[List[Optional[NDArray[np.float64]]]] = None,
    var_stack: Optional[NDArray[np.float64]] = None,
    outer_reference_geom: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Fit a single multi-band isophote at the given semi-major axis.

    Mirrors :func:`isoster.fitting.fit_isophote` but with the joint
    design-matrix solver, the combined-scalar radial gradient, per-band
    intensity / harmonic columns, and the experimental-feature reductions
    documented in the Stage-1 plan.
    """
    n_bands = len(config.bands)
    bands = list(config.bands)
    band_weights = config.resolved_band_weights()
    band_weights_arr = np.array([band_weights[b] for b in bands], dtype=np.float64)
    debug = bool(config.debug)

    # Pre-resolve image / mask / variance arrays once. The driver
    # passes pre-resolved arrays through to amortize the cost across
    # all isophote iterations; standalone callers (and the existing
    # tests) hit the resolver here instead.
    if image_stack is None or masks_resolved is None:
        image_stack, masks_resolved, var_stack = prepare_inputs(
            images,
            masks,
            variance_maps,
        )
    elif var_stack is None and variance_maps is not None:
        # Caller passed image_stack and masks but not var_stack: resolve.
        _, _, var_stack = prepare_inputs(images, masks, variance_maps)

    x0 = start_geometry["x0"]
    y0 = start_geometry["y0"]
    eps = start_geometry["eps"]
    pa = start_geometry["pa"]

    stop_code = 0
    niter = 0
    best_geometry: Optional[Dict[str, object]] = None
    converged = False
    min_amplitude = float("inf")
    previous_gradient: Optional[float] = None
    lexceed = False

    if config.convergence_scaling == "sector_area":
        n_samples_for_scale = max(64, int(2.0 * np.pi * sma))
        angular_width = 2.0 * np.pi / n_samples_for_scale
        delta_sma_for_scale = sma * config.astep if not config.linear_growth else config.astep
        convergence_scale = max(1.0, sma * delta_sma_for_scale * angular_width)
    elif config.convergence_scaling == "sqrt_sma":
        convergence_scale = max(1.0, float(np.sqrt(sma)))
    else:
        convergence_scale = 1.0

    prev_geom = (x0, y0, eps, pa)
    stable_count = 0
    cached_gradient: Optional[float] = None
    cached_gradient_error: Optional[float] = None
    cached_per_band_grad: List[float] = []
    cached_per_band_grad_err: List[Optional[float]] = []
    no_improvement_count = 0

    last_data: Optional[MultiIsophoteData] = None
    last_per_band_grad: List[float] = []
    # Captures the most recent iteration's joint-solve coefficient vector
    # ``[I0_0, ..., I0_{B-1}, A1, B1, A2, B2]`` so the shared-mode post-hoc
    # higher-order refit can subtract the frozen geometric model. ``None``
    # while the loop has not yet reached the joint solver.
    last_joint_coeffs: Optional[NDArray[np.float64]] = None
    # Captures the most recent iteration's joint-solve covariance so the
    # simultaneous_in_loop dispatcher can read shared higher-order standard
    # errors directly without re-running the solve.
    last_joint_cov: Optional[NDArray[np.float64]] = None
    # Tracks whether the most recent joint-solve was WLS or OLS. Required
    # so the simultaneous_in_loop dispatcher knows whether ``last_joint_cov``
    # is the exact MLE covariance (WLS) or raw ``(A^T A)^-1`` that needs
    # rescaling by the residual variance (OLS) before sqrt for SE.
    last_joint_wls_mode: bool = False

    # ISOFIT and forced-photometry handling are out of scope; we always
    # fit a 5-param model in joint or ref form.
    min_points = 6 + n_bands  # Decision D9 for joint mode
    use_ref_only = config.harmonic_combination == "ref"

    loose_validity = bool(config.loose_validity)
    loose_normalize = config.loose_validity_band_normalization == "per_band_count"

    # Section 6: simultaneous_in_loop widens the joint solver every iteration
    # to (B*N, B + 4 + 2*L) where L = len(harmonic_orders). The geometry
    # update math reads coeffs[n_bands..n_bands+3] which is unchanged.
    simul_in_loop = config.multiband_higher_harmonics == "simultaneous_in_loop"
    tail_width = 4 + 2 * len(config.harmonic_orders) if simul_in_loop else 4
    eval_orders_in_loop = list(config.harmonic_orders) if simul_in_loop else None

    # Track the per-band surviving counts for the most recent iteration
    # so we can stamp ``n_valid_<b>`` on the result row.  Under shared
    # validity these all equal ``actual_points``; under loose validity
    # they reflect each band's own surviving count after the per-band
    # sigma clip.
    last_n_valid_per_band = np.zeros(n_bands, dtype=np.int64)
    # Bands dropped at the most recent isophote because they fell
    # below the per-band thresholds.  Empty under shared validity.
    last_dropped_band_indices: List[int] = []
    # Per-band kept arrays from the most recent iteration; needed by
    # the per-band intens_err path under loose validity.
    last_intens_per_band_loose: Optional[List[NDArray[np.float64]]] = None
    last_variances_per_band_loose: Optional[List[NDArray[np.float64]]] = None
    last_phi_per_band_loose: Optional[List[NDArray[np.float64]]] = None

    # --- Stage-3 Stage-B: outer-region damping setup (once per isophote) ---
    # Mirrors single-band ``isoster.fitting.fit_isophote`` lines 1255–1290:
    # compute the sigmoid lambda(sma), per-axis weights, and arm the alpha
    # blend. ``solver`` mode (with reference pull) is not yet supported in
    # multi-band; the config validator restricts ``outer_reg_mode`` to
    # ``'damping'``. simultaneous_in_loop is excluded because the wider
    # joint matrix has no Tikhonov hook (Stage E will revisit).
    outer_damp_on = False
    outer_damp_lambda = 0.0
    outer_damp_w_center = 0.0
    outer_damp_w_eps = 0.0
    outer_damp_w_pa = 0.0
    if (
        bool(config.use_outer_center_regularization)
        and config.outer_reg_mode == "damping"
        and outer_reference_geom is not None
        and not simul_in_loop
    ):
        _ow = config.outer_reg_weights
        outer_damp_w_center = float(_ow.get("center", 0.0))
        outer_damp_w_eps = float(_ow.get("eps", 0.0))
        outer_damp_w_pa = float(_ow.get("pa", 0.0))
        _onset = config.outer_reg_sma_onset
        _width = config.outer_reg_sma_width if config.outer_reg_sma_width is not None else 0.4 * _onset
        outer_damp_lambda = config.outer_reg_strength / (1.0 + float(np.exp(-(sma - _onset) / _width)))
        outer_damp_on = outer_damp_lambda >= 1e-6

    for i in range(config.maxit):
        niter = i + 1
        # Set when this iteration's solve returned geometry steps rather than
        # amplitudes, so the pooled-gradient rescale below knows to run.
        solved_in_geometry_units = False
        # Per-band gradients that turn the shared columns into geometry steps.
        # ``None`` keeps the amplitude parameterisation. The scaling needs a
        # gradient at solve time and the loop only produces one afterwards, so
        # the previous iteration's is used: these are slowly-varying nuisance
        # scale factors, not the estimand. Iteration 0 has none and falls back
        # to amplitudes. Threaded into *every* solver — shared and loose, joint
        # and decoupled, plain and simultaneous — so the parameterisation cannot
        # depend on which other options are set.
        geometry_gradients = (
            np.asarray(last_per_band_grad, dtype=np.float64)
            if (config.geometry_parameterized_solve and last_per_band_grad)
            else None
        )
        # Scatter used by the convergence test. Defaults to the reported `rms`
        # and is replaced by the variance-weighted form under WLS below.
        convergence_rms = float("nan")
        data = extract_isophote_data_multi_prepared(
            image_stack,
            masks_resolved,
            var_stack,
            x0,
            y0,
            sma,
            eps,
            pa,
            use_eccentric_anomaly=config.use_eccentric_anomaly,
            loose_validity=loose_validity,
        )
        last_data = data
        total_points = data.valid_count

        if loose_validity:
            # Per-band independent sigma clip on jagged arrays.
            phi_pb, intens_pb, vars_pb, _n_clipped_loose = _per_band_sigma_clip_loose(
                data.phi_per_band,  # type: ignore[arg-type]
                data.intens_per_band,  # type: ignore[arg-type]
                data.variances_per_band,
                config.sclip,
                config.nclip,
                config.sclip_low,
                config.sclip_high,
            )
            # M8: express total/actual in matching units under loose
            # validity — sums over bands. data.valid_count is the
            # intersection count; mixing it with the per-band kept sums
            # made nflag negative and left the fflag guard inert.
            total_points = int(np.sum(data.n_valid_per_band))
            n_valid_after_clip = np.array([int(p.size) for p in phi_pb], dtype=np.int64)
            # Per-band drop logic: a band falling below either the
            # absolute count or the fraction threshold is dropped from
            # the joint solve at this isophote.
            min_count = int(config.loose_validity_min_per_band_count)
            min_frac = float(config.loose_validity_min_per_band_frac)
            n_attempted = max(int(data.n_samples), 1)
            surviving_mask = np.array(
                [(n_b >= min_count) and (n_b / n_attempted >= min_frac) for n_b in n_valid_after_clip], dtype=bool
            )
            surviving_idx = np.where(surviving_mask)[0]
            dropped_idx_list: List[int] = [int(i_b) for i_b in range(n_bands) if not surviving_mask[i_b]]
            actual_points = int(n_valid_after_clip.sum())
            last_n_valid_per_band = n_valid_after_clip
            last_dropped_band_indices = dropped_idx_list
            last_intens_per_band_loose = intens_pb
            last_variances_per_band_loose = vars_pb
            last_phi_per_band_loose = phi_pb

            # Whole-isophote drop: fewer than 2 surviving bands means
            # the joint solve is meaningless.
            if surviving_idx.size < 2:
                if best_geometry is not None:
                    best_geometry["stop_code"] = 3
                    best_geometry["niter"] = niter
                    # M13: the end-of-fit n_valid_<b> stamping is bypassed
                    # on this early return.
                    _stamp_n_valid_per_band(best_geometry, bands, last_n_valid_per_band)
                    return best_geometry
                stop_code = 3
                break

            # Subset jagged arrays to surviving bands.
            phi_solve = [phi_pb[i_b] for i_b in surviving_idx]
            intens_solve = [intens_pb[i_b] for i_b in surviving_idx]
            vars_solve = [vars_pb[i_b] for i_b in surviving_idx] if vars_pb is not None else None

            # The downstream code reads `intens_per_band` (rectangular)
            # to drive evaluate_joint_model + RMS + harmonic stamping.
            # Under loose validity these are jagged; we keep separate
            # lists and skip the rectangular evaluation path later.
            angles = phi_solve[0]  # placeholder for fflag check (unused)
            phi = phi_solve[0]
            intens_per_band = intens_solve  # type: ignore[assignment]
            variances_per_band = vars_solve  # type: ignore[assignment]
        else:
            # Per-band sigma clip + AND across bands (shared validity).
            angles, phi, intens_per_band, variances_per_band, _n_clipped = _per_band_sigma_clip(
                data.angles,
                data.phi,
                data.intens,
                data.variances,
                config.sclip,
                config.nclip,
                config.sclip_low,
                config.sclip_high,
            )
            actual_points = len(angles)
            last_n_valid_per_band = np.full(n_bands, actual_points, dtype=np.int64)
            last_dropped_band_indices = []
            last_intens_per_band_loose = None
            last_variances_per_band_loose = None
            last_phi_per_band_loose = None
            surviving_idx = np.arange(n_bands)
            phi_solve: List[NDArray[np.float64]] = []  # only used in loose path
            intens_solve: List[NDArray[np.float64]] = []
            vars_solve = None

        if total_points > 0 and actual_points < total_points * (1.0 - config.fflag):
            if best_geometry is not None:
                best_geometry["stop_code"] = 1
                best_geometry["niter"] = niter
                # M13: the end-of-fit n_valid_<b> stamping is bypassed on
                # this early return.
                _stamp_n_valid_per_band(best_geometry, bands, last_n_valid_per_band)
                return best_geometry
            stop_code = 1
            break

        # In ref mode we use the reference band's intensity vector for the
        # 5-param solve; in joint mode we use the (B+4)-column joint solve.
        # Both modes share validation against `min_points`.
        ref_min_points = 6
        # Set by the reference-only solve below; stays None in joint mode, where
        # the pooled residual variance is the correct scale.
        ref_ols_var_residual: Optional[float] = None
        if use_ref_only:
            if actual_points < ref_min_points:
                stop_code = 3
                break
        else:
            if actual_points < min_points:
                stop_code = 3
                break

        if use_ref_only:
            ref_idx = bands.index(config.reference_band)
            if loose_validity:
                # The solve arrays hold surviving bands only, so locate the
                # reference band by position within them (review M1: indexing
                # by full-list position silently fit the wrong band or
                # crashed, and the lstsq failure was swallowed into zeroed
                # harmonics by fit_first_and_second_harmonics_ref).
                ref_positions = np.flatnonzero(surviving_idx == ref_idx)
                ref_pos = int(ref_positions[0]) if ref_positions.size else None
                if ref_pos is None or intens_solve[ref_pos].size < ref_min_points:
                    # Reference band dropped (or too few samples) at this
                    # isophote: the ref-mode solve is undefined — skip with
                    # a clear stop code instead of fitting the wrong band.
                    if best_geometry is not None:
                        best_geometry["stop_code"] = 3
                        best_geometry["niter"] = niter
                        return best_geometry
                    stop_code = 3
                    break
                angles_ref = phi_solve[ref_pos]
                intens_ref = intens_solve[ref_pos]
                variances_ref = vars_solve[ref_pos] if vars_solve is not None else None
            else:
                angles_ref = angles
                intens_ref = intens_per_band[ref_idx]
                variances_ref = variances_per_band[ref_idx] if variances_per_band is not None else None
            coeffs_ref, cov_ref, wls_mode = fit_first_and_second_harmonics_ref(angles_ref, intens_ref, variances_ref)
            # The geometry covariance below comes from this reference-only
            # solve, so its OLS scale must come from the same band's residuals.
            ref_ols_var_residual = (
                None
                if wls_mode
                else _reference_residual_variance(angles_ref, intens_ref, coeffs_ref, _sigma_bg_variance_floor(config))
            )
            # Widen ref coeffs to a full (B + 4,) layout so downstream
            # bookkeeping is uniform with joint mode. Per-band I0_b for
            # non-reference bands comes from the same reducer the decoupled
            # joint path uses, so under WLS it is the inverse-variance-weighted
            # mean whose variance `intens_err_<b>` already reports. A plain
            # `np.mean` here discarded the variance map the caller supplied and
            # left the value and its uncertainty describing different
            # estimators: on a heterogeneous map the reported error understated
            # the reported mean's own error tenfold. Dropped bands (loose
            # validity) stay NaN, matching the loose joint solver's widening
            # below. Ref mode cannot select `integrator='median'` — the config
            # validators exclude that combination — so this is always a mean.
            coeffs = np.zeros(n_bands + 4, dtype=np.float64)
            if loose_validity:
                coeffs[:n_bands] = np.nan
                means_solve = _per_band_mean_or_median_jagged(intens_solve, vars_solve, config.integrator)
                for new_idx, orig_idx in enumerate(surviving_idx):
                    coeffs[orig_idx] = float(means_solve[new_idx])
            else:
                coeffs[:n_bands] = _per_band_mean_or_median(intens_per_band, variances_per_band, config.integrator)
            coeffs[ref_idx] = float(coeffs_ref[0])
            coeffs[n_bands:] = coeffs_ref[1:5]
            # cov: only the harmonic block is meaningful in ref mode.
            if cov_ref is not None:
                cov_full = np.zeros((n_bands + 4, n_bands + 4), dtype=np.float64)
                # Per-band I0 diagonal stub variance, from the same helper that
                # produces `intens_err_<b>`, so the two cannot disagree. The
                # retired form (`mean(v)/N`, the unweighted mean's variance) was
                # a third opinion alongside the plain-mean value and the
                # inverse-variance error. Only the harmonic block is read
                # downstream, but a wrong number here is a trap for the next
                # reader.
                if loose_validity:
                    stub_bands = [
                        (
                            int(orig_idx),
                            intens_solve[new_idx],
                            vars_solve[new_idx] if vars_solve is not None else None,
                        )
                        for new_idx, orig_idx in enumerate(surviving_idx)
                    ]
                else:
                    stub_bands = [
                        (
                            b_idx,
                            intens_per_band[b_idx],
                            variances_per_band[b_idx] if variances_per_band is not None else None,
                        )
                        for b_idx in range(n_bands)
                    ]
                for b_idx, intens_b, var_b in stub_bands:
                    if b_idx == ref_idx:
                        cov_full[b_idx, b_idx] = float(cov_ref[0, 0])
                    else:
                        cov_full[b_idx, b_idx] = _per_band_intercept_variance(intens_b, var_b, config.integrator)
                cov_full[n_bands:, n_bands:] = cov_ref[1:5, 1:5]
            else:
                cov_full = None
        elif loose_validity:
            n_surviving = int(surviving_idx.size)
            surviving_weights = band_weights_arr[surviving_idx]
            # Geometry parameterisation, restricted to the bands in this solve.
            if geometry_gradients is not None:
                solve_gradients = np.asarray(geometry_gradients, dtype=np.float64)[surviving_idx]
                solved_in_geometry_units = True
            else:
                solve_gradients = None
            if simul_in_loop:
                coeffs_sub, cov_sub, wls_mode = fit_simultaneous_joint_loose(
                    phi_solve,
                    intens_solve,
                    surviving_weights,
                    config.harmonic_orders,
                    vars_solve,
                    normalize=loose_normalize,
                    fit_per_band_intens_jointly=config.fit_per_band_intens_jointly,
                    integrator=config.integrator,
                    per_band_gradients=solve_gradients,
                )
            else:
                coeffs_sub, cov_sub, wls_mode = fit_first_and_second_harmonics_joint_loose(
                    phi_solve,
                    intens_solve,
                    surviving_weights,
                    vars_solve,
                    normalize=loose_normalize,
                    fit_per_band_intens_jointly=config.fit_per_band_intens_jointly,
                    integrator=config.integrator,
                    per_band_gradients=solve_gradients,
                )
            # Widen the surviving-bands solution back to a full coefficient
            # vector with NaN for dropped bands. Trailing block is
            # (B+4) wide for the standard solver, (B+4+2L) for simultaneous.
            coeffs = np.full(n_bands + tail_width, np.nan, dtype=np.float64)
            for new_idx, orig_idx in enumerate(surviving_idx):
                coeffs[orig_idx] = coeffs_sub[new_idx]
            coeffs[n_bands:] = coeffs_sub[n_surviving:]
            if cov_sub is not None:
                # Review fix H2: widen the FULL surviving-bands covariance
                # block (including the surviving-band cross-correlations
                # cov_sub[i, j] for i ≠ j) into the full ``(n_bands +
                # tail_width)``-square matrix. The previous version
                # dropped the off-diagonals, which would silently bias
                # downstream cross-band uncertainty propagation. Dropped
                # bands stay at zero (their I0_b is NaN — no cov
                # information to propagate). The geometric ↔ surviving-
                # band cross-block (cov_sub[surviving rows, n_surviving:])
                # is also preserved at the surviving bands' original-index
                # rows / cols.
                cov_full = np.zeros(
                    (n_bands + tail_width, n_bands + tail_width),
                    dtype=np.float64,
                )
                for new_i, orig_i in enumerate(surviving_idx):
                    for new_j, orig_j in enumerate(surviving_idx):
                        cov_full[orig_i, orig_j] = cov_sub[new_i, new_j]
                    # Cross-block: row orig_i (intercept) ↔ trailing geom.
                    cov_full[orig_i, n_bands:] = cov_sub[new_i, n_surviving:]
                    cov_full[n_bands:, orig_i] = cov_sub[n_surviving:, new_i]
                # Trailing geometric block (shared across bands).
                cov_full[n_bands:, n_bands:] = cov_sub[n_surviving:, n_surviving:]
            else:
                cov_full = None
        else:
            if geometry_gradients is not None:
                solved_in_geometry_units = True
            if simul_in_loop:
                coeffs, cov_full, wls_mode = fit_simultaneous_joint(
                    angles,
                    intens_per_band,
                    band_weights_arr,
                    config.harmonic_orders,
                    variances_per_band,
                    fit_per_band_intens_jointly=config.fit_per_band_intens_jointly,
                    integrator=config.integrator,
                    per_band_gradients=geometry_gradients,
                )
            else:
                coeffs, cov_full, wls_mode = fit_first_and_second_harmonics_joint(
                    angles,
                    intens_per_band,
                    band_weights_arr,
                    variances_per_band,
                    fit_per_band_intens_jointly=config.fit_per_band_intens_jointly,
                    integrator=config.integrator,
                    per_band_gradients=geometry_gradients,
                )

        A1 = float(coeffs[n_bands])
        B1 = float(coeffs[n_bands + 1])
        A2 = float(coeffs[n_bands + 2])
        B2 = float(coeffs[n_bands + 3])
        last_joint_coeffs = coeffs
        last_joint_cov = cov_full
        last_joint_wls_mode = bool(wls_mode)

        # Combined gradient.
        if (
            i == 0
            or not config.use_lazy_gradient
            or no_improvement_count >= 3
            or cached_gradient_error is None
            or lexceed
        ):
            geom = {"x0": x0, "y0": y0, "sma": sma, "eps": eps, "pa": pa}
            # Pool the per-band gradients with the weights each band carried in
            # the harmonic solve above, using the post-clipping variances that
            # solve actually saw. Under OLS this reduces to band_weights_arr.
            #
            # Reference mode fits the harmonics from the reference band alone,
            # so that band's solve weight is 1 and every other band's is 0.
            # Expressing it as a one-hot pooling weight makes the geometry
            # denominator, the gradient error, the runaway check, the maxgerr
            # gate and the reported `grad` all describe the same band. Dividing
            # a reference-band coefficient by an all-band pooled gradient
            # recovered 0.182 against a truth of 0.100 — in OLS as well as WLS.
            if use_ref_only:
                pooling_weights = np.zeros(n_bands, dtype=np.float64)
                pooling_weights[bands.index(config.reference_band)] = 1.0
            else:
                pooling_weights = joint_gradient_pooling_weights(
                    band_weights_arr,
                    vars_solve if loose_validity else variances_per_band,
                    n_bands,
                    band_indices=surviving_idx if loose_validity else None,
                )
            grad_joint, grad_err_joint, per_band_grad, per_band_err = compute_joint_gradient(
                image_stack,
                masks_resolved,
                var_stack,
                geom,
                config,
                band_weights_arr,
                previous_gradient=previous_gradient,
                current_data=data,
                pooling_weights=pooling_weights,
            )
            cached_gradient = grad_joint
            cached_gradient_error = grad_err_joint
            cached_per_band_grad = per_band_grad
            cached_per_band_grad_err = per_band_err
            if no_improvement_count >= 3:
                no_improvement_count = 0
        else:
            grad_joint = cached_gradient
            grad_err_joint = cached_gradient_error
            per_band_grad = cached_per_band_grad
            per_band_err = cached_per_band_grad_err

        if solved_in_geometry_units and grad_joint is not None and np.isfinite(grad_joint):
            # Convert the geometry steps back into amplitude-equivalent
            # coefficients by multiplying by the same pooled gradient the
            # downstream geometry update divides by. That update then recovers
            # the fitted step exactly, and every consumer — max_amp, the
            # convergence test, the harmonic attachers, the covariance-based
            # geometry errors — keeps working unchanged in its own units. The
            # only thing that differs is the *value*, which now carries the
            # minimum-variance band weighting.
            scale = np.ones(coeffs.size, dtype=np.float64)
            # Every shared column was scaled by grad_b in the solve, including
            # the simultaneous higher-order block, so all of them come back.
            scale[n_bands:] = float(grad_joint)
            coeffs = coeffs * scale
            if cov_full is not None:
                cov_full = cov_full * np.outer(scale, scale)
            A1 = float(coeffs[n_bands])
            B1 = float(coeffs[n_bands + 1])
            A2 = float(coeffs[n_bands + 2])
            B2 = float(coeffs[n_bands + 3])
            last_joint_coeffs = coeffs
            last_joint_cov = cov_full

        last_per_band_grad = per_band_grad

        if grad_err_joint is not None:
            previous_gradient = grad_joint

        if grad_joint == 0.0 or grad_joint is None:
            stop_code = -1
            break

        # Gradient-error gate (mirrors single-band behavior).
        gradient_relative_error: Optional[float]
        if grad_err_joint is not None and grad_joint < 0.0:
            gradient_relative_error = abs(grad_err_joint / grad_joint)
        else:
            gradient_relative_error = None
        if not going_inwards:
            if config.permissive_geometry and gradient_relative_error is None:
                pass
            elif gradient_relative_error is None or gradient_relative_error > config.maxgerr or grad_joint >= 0.0:
                if lexceed:
                    stop_code = -1
                    break
                lexceed = True

        # RMS of the joint model fit (per-band-stacked residuals).
        # Under loose validity, ``intens_per_band`` is a jagged list and
        # the bands sample at potentially different angles, so we
        # evaluate the model per band on each band's own *post-clip*
        # kept angles (must match the post-clip intensities; using the
        # pre-clip ``data.phi_per_band`` here silently produces shape
        # mismatches that turn the residual concat into NaN, which then
        # blocks the convergence test from ever firing).
        if loose_validity:
            model_per_band_loose: List[NDArray[np.float64]] = []
            residual_chunks: List[NDArray[np.float64]] = []
            A1c, B1c, A2c, B2c = (
                float(coeffs[n_bands]),
                float(coeffs[n_bands + 1]),
                float(coeffs[n_bands + 2]),
                float(coeffs[n_bands + 3]),
            )
            higher_in_loop_terms: List[Tuple[int, float, float]] = []
            if eval_orders_in_loop:
                for j, n_order in enumerate(eval_orders_in_loop):
                    higher_in_loop_terms.append(
                        (
                            int(n_order),
                            float(coeffs[n_bands + 4 + 2 * j]),
                            float(coeffs[n_bands + 4 + 2 * j + 1]),
                        )
                    )
            phi_post_clip = last_phi_per_band_loose or []
            intens_post_clip = last_intens_per_band_loose or []
            for b_idx in range(n_bands):
                if b_idx >= len(phi_post_clip) or phi_post_clip[b_idx].size == 0 or np.isnan(coeffs[b_idx]):
                    model_per_band_loose.append(np.empty(0, dtype=np.float64))
                    continue
                p_b = phi_post_clip[b_idx]
                m_b = (
                    float(coeffs[b_idx])
                    + A1c * np.sin(p_b)
                    + B1c * np.cos(p_b)
                    + A2c * np.sin(2.0 * p_b)
                    + B2c * np.cos(2.0 * p_b)
                )
                for n_order, an, bn in higher_in_loop_terms:
                    m_b = m_b + an * np.sin(n_order * p_b) + bn * np.cos(n_order * p_b)
                model_per_band_loose.append(m_b)
                i_b = intens_post_clip[b_idx]
                # Both arrays now come from the same post-clip kept set
                # so the size match is guaranteed; the explicit check
                # is a defensive guard.
                if i_b.size == m_b.size:
                    residual_chunks.append(i_b - m_b)
            rms = float(np.std(np.concatenate(residual_chunks))) if residual_chunks else float("nan")
            model_per_band = model_per_band_loose  # type: ignore[assignment]
        else:
            model_per_band = evaluate_joint_model(
                angles,
                coeffs,
                n_bands,
                harmonic_orders=eval_orders_in_loop,
            )
            residuals_flat = (intens_per_band - model_per_band).reshape(-1)
            rms = float(np.std(residuals_flat))
            if variances_per_band is not None:
                # Convergence compares the shared harmonic amplitude — which the
                # solve produced under per-sample weights w_b/var_{b,i} — against
                # this scatter. An unweighted pooled scatter is in raw flux units
                # and is dominated by whichever band happens to carry the largest
                # numbers, so the comparison changed by a factor of 8 under a pure
                # change of one band's flux units. Weighting the scatter the same
                # way the solve weighted the data puts both sides in one unit
                # system. Reduces to the plain rms under OLS with equal band
                # weights, so the default path is untouched; `rms` itself is a
                # reported column and stays the physical ring dispersion.
                convergence_weights = np.repeat(
                    band_weights_arr, intens_per_band.shape[1]
                ) / variances_per_band.reshape(-1)
                weight_sum = float(np.sum(convergence_weights))
                if np.isfinite(weight_sum) and weight_sum > 0.0:
                    convergence_rms = float(np.sqrt(np.sum(convergence_weights * residuals_flat**2) / weight_sum))

        if not np.isfinite(convergence_rms):
            # OLS, or the loose-validity path, which has no rectangular
            # variance block to weight with: fall back to the plain scatter.
            convergence_rms = rms

        harmonics = [A1, B1, A2, B2]
        if config.fix_center:
            harmonics[0] = 0.0
            harmonics[1] = 0.0
        if config.fix_pa:
            harmonics[2] = 0.0
        if config.fix_eps:
            harmonics[3] = 0.0
        max_idx = int(np.argmax(np.abs(harmonics)))
        max_amp = harmonics[max_idx]

        # Stage-3 Stage-F: central-region regularization penalty enters
        # the best-iteration selector. Iterations whose geometry jumped
        # far from the previous isophote at low SMA look worse to the
        # selector and are not chosen as best_geometry. No-op when the
        # feature is off, when previous_geometry is None (first
        # isophote), or above ~3× the threshold SMA.
        central_reg_penalty = _compute_central_regularization_penalty_mb(
            {"x0": x0, "y0": y0, "eps": eps, "pa": pa},
            previous_geometry,
            sma,
            config,
        )
        effective_amp = abs(max_amp) + central_reg_penalty
        if effective_amp < min_amplitude:
            min_amplitude = effective_amp
            no_improvement_count = 0

            if config.compute_errors:
                x0_err, y0_err, eps_err, pa_err = _compute_parameter_errors_from_joint(
                    coeffs=coeffs,
                    cov_full=cov_full,
                    n_bands=n_bands,
                    sma=sma,
                    eps=eps,
                    pa=pa,
                    gradient=grad_joint,
                    gradient_error=grad_err_joint if config.use_corrected_errors else None,
                    # Under loose validity the OLS rescale needs the full
                    # jagged per-band angle lists plus the surviving bands'
                    # positions in the full vectors (review M3).
                    angles=phi_solve if loose_validity else angles,
                    intens_per_band=intens_per_band,
                    use_exact_covariance=wls_mode,
                    var_residual_floor=config.sigma_bg**2 if config.sigma_bg is not None else None,
                    band_weights_arr=band_weights_arr,
                    band_indices=surviving_idx if loose_validity else None,
                    # Under simultaneous_in_loop the loop solved the wider
                    # system, so the geometry rescale must measure that same
                    # model and count those same parameters.
                    harmonic_orders=eval_orders_in_loop,
                    # In ref mode the geometry came from the reference band
                    # alone, so its scale must too. None in joint mode, where
                    # the pooled estimate is correct.
                    residual_variance=ref_ols_var_residual,
                )
            else:
                x0_err = y0_err = eps_err = pa_err = 0.0

            # Per-band reported intensity: use the joint coefficient I0_b
            # (which is the WLS / OLS-fitted background level for that
            # band along this isophote, parallel to single-band's `intens`).
            best_geometry = {
                "sma": sma,
                "x0": x0,
                "y0": y0,
                "eps": eps,
                "pa": pa,
                "x0_err": x0_err,
                "y0_err": y0_err,
                "eps_err": eps_err,
                "pa_err": pa_err,
                "rms": rms,
                "valid": True,
                "use_eccentric_anomaly": config.use_eccentric_anomaly,
                "tflux_e": float("nan"),
                "tflux_c": float("nan"),
                "npix_e": 0,
                "npix_c": 0,
            }
            ref_idx_for_err = bands.index(config.reference_band) if use_ref_only else -1
            # Pooled weight-aware OLS residual variance for the coupled
            # intens_err_<b> branch below (review M2: the per-band unweighted
            # np.var mis-scaled errors by ~1/w under non-unit band weights).
            # Only meaningful for the joint (non-ref) shared-validity path.
            ols_var_residual_joint: Optional[float] = None
            if (
                not wls_mode
                and cov_full is not None
                and not loose_validity
                and not use_ref_only
                and config.fit_per_band_intens_jointly
            ):
                ols_var_residual_joint = _compute_joint_residual_variance(
                    coeffs,
                    angles,
                    intens_per_band,
                    band_weights_arr,
                    n_bands=n_bands,
                    # Same model, parameter count, and floor as the geometry
                    # errors above: one solve, one scale.
                    n_geom_params=n_bands + 4 + (2 * len(eval_orders_in_loop) if eval_orders_in_loop else 0),
                    jagged=False,
                    floor=_sigma_bg_variance_floor(config),
                    harmonic_orders=eval_orders_in_loop,
                )
            # When the per-band intercept is computed post-fit (ref mode for
            # non-ref bands, or fit_per_band_intens_jointly=False for every
            # band), `intens_err_b` is the band's own SEM and does NOT flow
            # through the joint covariance.  Routing it through
            # cov_full[b_idx, b_idx] in OLS would double-apply the residual
            # variance (B3 regression).
            for b_idx, b in enumerate(bands):
                # Loose-validity dropped bands: NaN every per-band field.
                if loose_validity and b_idx in last_dropped_band_indices:
                    best_geometry[f"intens_{b}"] = float("nan")
                    best_geometry[f"intens_err_{b}"] = float("nan")
                    best_geometry[f"rms_{b}"] = float("nan")
                    for n_order in config.harmonic_orders:
                        # NaN (not 0.0) so downstream consumers can detect
                        # the drop — 0.0 reads as a real measurement (M5).
                        best_geometry[f"a{int(n_order)}_{b}"] = float("nan")
                        best_geometry[f"b{int(n_order)}_{b}"] = float("nan")
                        best_geometry[f"a{int(n_order)}_err_{b}"] = float("nan")
                        best_geometry[f"b{int(n_order)}_err_{b}"] = float("nan")
                    if debug:
                        for key in _PER_BAND_DEBUG_KEYS:
                            best_geometry[f"{key}_{b}"] = float("nan")
                    continue
                intens_b = float(coeffs[b_idx])
                # Per-band rms from band b residuals; intens_err_b from
                # diagonal of the joint covariance at row b (already an
                # exact covariance under WLS, residual-scaled under OLS).
                if loose_validity:
                    band_intens_kept = (
                        last_intens_per_band_loose[b_idx]
                        if last_intens_per_band_loose is not None
                        else np.empty(0, dtype=np.float64)
                    )
                    band_model_kept = model_per_band[b_idx]
                    if band_intens_kept.size and band_intens_kept.size == band_model_kept.size:
                        rms_b = float(np.std(band_intens_kept - band_model_kept))
                    else:
                        rms_b = float("nan")
                else:
                    rms_b = float(np.std(intens_per_band[b_idx] - model_per_band[b_idx]))
                # A band's intensity error comes from its own ring statistic only
                # when the intercept did not flow through the joint solve. Loose
                # validity used to force that unconditionally, but the two are
                # independent: with `fit_per_band_intens_jointly=True` the jagged
                # solve still fits `I0_b` as a coupled parameter, and its
                # covariance is mapped back into `cov_full` for surviving bands.
                # Taking the direct SEM there ignored the coupling — a measured
                # ~31% underestimate under incomplete angular coverage. Bands
                # dropped at this isophote never reach here; they are NaN-filled
                # above.
                use_direct_sem = (not config.fit_per_band_intens_jointly) or (use_ref_only and b_idx != ref_idx_for_err)
                if use_direct_sem:
                    # The reported intens_<b> is this band's own ring statistic,
                    # not a joint-solve output, so its error comes from that same
                    # statistic — an inverse-variance-weighted mean, a plain mean,
                    # or a median. See _per_band_intercept_variance.
                    if loose_validity:
                        band_intens_kept = (
                            last_intens_per_band_loose[b_idx]
                            if last_intens_per_band_loose is not None
                            else np.empty(0, dtype=np.float64)
                        )
                        band_var_kept = (
                            last_variances_per_band_loose[b_idx] if last_variances_per_band_loose is not None else None
                        )
                    else:
                        band_intens_kept = intens_per_band[b_idx]
                        band_var_kept = variances_per_band[b_idx] if variances_per_band is not None else None
                    if int(band_intens_kept.size) <= 0:
                        intens_err_b = float("nan")
                    else:
                        # rms_b already holds this band's residuals around the
                        # fitted model, which is the OLS scatter single-band uses.
                        residuals_b = (
                            band_intens_kept - band_model_kept
                            if loose_validity
                            else intens_per_band[b_idx] - model_per_band[b_idx]
                        )
                        if residuals_b.size != band_intens_kept.size:
                            residuals_b = None
                        intens_err_b = float(
                            np.sqrt(
                                _per_band_intercept_variance(
                                    band_intens_kept,
                                    band_var_kept,
                                    config.integrator,
                                    residuals_b=residuals_b,
                                )
                            )
                        )
                elif cov_full is not None:
                    if wls_mode:
                        intens_err_b = float(np.sqrt(max(cov_full[b_idx, b_idx], 0.0)))
                    else:
                        # OLS: scale the (A^T A)^-1 diagonal by a residual
                        # variance. Joint mode uses the pooled weight-aware
                        # estimate (review M2); ref mode keeps the band's own
                        # (the ref solve is unweighted).
                        if use_ref_only:
                            # Only the reference band reaches here: the others
                            # take the direct-SEM branch above. `intens_<ref>`
                            # is the fitted intercept coeffs_ref[0], so it is
                            # scaled by the same reference-only residual
                            # variance as the geometry errors from that solve —
                            # one solve, one scale, floor included.
                            if ref_ols_var_residual is not None:
                                var_res_b = ref_ols_var_residual
                            else:
                                ddof_eff = 1 + 4  # one I0_b + 4 shared geometric
                                if len(intens_per_band[b_idx]) > ddof_eff:
                                    var_res_b = float(
                                        np.var(intens_per_band[b_idx] - model_per_band[b_idx], ddof=ddof_eff)
                                    )
                                else:
                                    var_res_b = 0.0
                        else:
                            var_res_b = ols_var_residual_joint if ols_var_residual_joint is not None else 0.0
                        intens_err_b = float(np.sqrt(max(cov_full[b_idx, b_idx], 0.0) * var_res_b))
                else:
                    intens_err_b = float("nan")
                best_geometry[f"intens_{b}"] = intens_b
                best_geometry[f"intens_err_{b}"] = intens_err_b
                best_geometry[f"rms_{b}"] = rms_b
                # Harmonic deviation placeholders; computed post-hoc on
                # convergence below. Initialize to zeros for unconverged
                # exit paths (matches single-band convention).
                for n_order in config.harmonic_orders:
                    best_geometry[f"a{int(n_order)}_{b}"] = 0.0
                    best_geometry[f"b{int(n_order)}_{b}"] = 0.0
                    best_geometry[f"a{int(n_order)}_err_{b}"] = 0.0
                    best_geometry[f"b{int(n_order)}_err_{b}"] = 0.0
                if debug:
                    grad_b = per_band_grad[b_idx] if b_idx < len(per_band_grad) else float("nan")
                    err_b = per_band_err[b_idx] if b_idx < len(per_band_err) else None
                    grad_r_err_b = (
                        abs(err_b / grad_b)
                        if (err_b is not None and grad_b is not None and grad_b != 0.0)
                        else float("nan")
                    )
                    best_geometry[f"grad_{b}"] = float(grad_b)
                    best_geometry[f"grad_error_{b}"] = float(err_b) if err_b is not None else float("nan")
                    best_geometry[f"grad_r_error_{b}"] = float(grad_r_err_b)
            best_geometry["ndata"] = actual_points
            best_geometry["nflag"] = total_points - actual_points
            if debug:
                # Stage-3 Stage-C: top-level joint gradient scalars mirror
                # the single-band ``grad`` / ``grad_error`` API. The
                # multi-band lsb_auto_lock trigger reads these (per S3:
                # joint combined gradient). Single-band downstream tooling
                # that consumes ``iso['grad']`` works on multi-band too.
                best_geometry["grad"] = float(grad_joint) if grad_joint is not None else float("nan")
                best_geometry["grad_error"] = float(grad_err_joint) if grad_err_joint is not None else float("nan")
        else:
            no_improvement_count += 1

        # Effective rms for convergence check (decision D17 — sigma_bg honored
        # in multi-band even though variance maps already encode pixel noise).
        # Under WLS this is the variance-weighted scatter, so the convergence
        # comparison stops depending on a band's flux units; identical to `rms`
        # under OLS with equal band weights.
        effective_rms = convergence_rms
        if config.sigma_bg is not None and len(angles) > 0:
            noise_floor = config.sigma_bg / np.sqrt(len(angles))
            effective_rms = max(rms, noise_floor)

        if abs(max_amp) < config.conver * convergence_scale * effective_rms and i >= config.minit:
            stop_code = 0
            converged = True
            if config.compute_deviations and best_geometry is not None:
                if loose_validity and last_intens_per_band_loose is not None:
                    _attach_higher_harmonics_dispatch(
                        best_geometry,
                        bands,
                        config,
                        last_joint_coeffs,
                        last_phi_per_band_loose or [],
                        last_intens_per_band_loose,
                        last_variances_per_band_loose,
                        sma,
                        per_band_grad,
                        band_weights_arr,
                        jagged=True,
                        last_cov=last_joint_cov,
                        last_wls_mode=last_joint_wls_mode,
                        dropped_band_indices=set(last_dropped_band_indices),
                    )
                else:
                    _attach_higher_harmonics_dispatch(
                        best_geometry,
                        bands,
                        config,
                        last_joint_coeffs,
                        angles,
                        intens_per_band,
                        variances_per_band,
                        sma,
                        per_band_grad,
                        band_weights_arr,
                        jagged=False,
                        last_cov=last_joint_cov,
                        last_wls_mode=last_joint_wls_mode,
                    )
            break

        # Geometry update (Jedrzejewski-1987, joint gradient denominator).
        damping = config.geometry_damping
        if grad_err_joint is not None and abs(grad_joint) > 0.0:
            grad_snr = abs(grad_joint / grad_err_joint)
            snr_damping = float(np.clip(grad_snr / 3.0, 0.1, 1.0))
            damping *= snr_damping

        if config.geometry_update_mode == "simultaneous":
            # All four parameters update each iteration.
            if not config.fix_center:
                coeff_c_minor = (1.0 - eps) / grad_joint
                coeff_c_major = 1.0 / grad_joint
                aux_minor = -A1 * coeff_c_minor * damping
                aux_major = -B1 * coeff_c_major * damping
                if outer_damp_on and outer_damp_w_center > 0.0:
                    alpha_minor = _tikhonov_alpha(coeff_c_minor, outer_damp_lambda, outer_damp_w_center)
                    alpha_major = _tikhonov_alpha(coeff_c_major, outer_damp_lambda, outer_damp_w_center)
                    aux_minor = (1.0 - alpha_minor) * aux_minor
                    aux_major = (1.0 - alpha_major) * aux_major
                if config.clip_max_shift is not None:
                    max_iter_shift = max(config.clip_max_shift, 0.05 * sma)
                    shift_len = float(np.sqrt(aux_minor**2 + aux_major**2))
                    if shift_len > max_iter_shift:
                        scale = max_iter_shift / shift_len
                        aux_minor *= scale
                        aux_major *= scale
                x0 += -aux_minor * np.sin(pa) + aux_major * np.cos(pa)
                y0 += aux_minor * np.cos(pa) + aux_major * np.sin(pa)
            if not config.fix_pa:
                denom = (1.0 - eps) ** 2 - 1.0
                if abs(denom) < 1e-10:
                    denom = -1e-10
                coeff_pa = 2.0 * (1.0 - eps) / sma / grad_joint / denom
                pa_corr = A2 * coeff_pa * damping
                if outer_damp_on and outer_damp_w_pa > 0.0:
                    alpha_pa = _tikhonov_alpha(coeff_pa, outer_damp_lambda, outer_damp_w_pa)
                    pa_corr = (1.0 - alpha_pa) * pa_corr
                if config.clip_max_pa is not None:
                    pa_corr = float(np.clip(pa_corr, -config.clip_max_pa, config.clip_max_pa))
                pa = (pa + pa_corr) % np.pi
            if not config.fix_eps:
                coeff_eps = 2.0 * (1.0 - eps) / sma / grad_joint
                eps_corr = B2 * coeff_eps * damping
                if outer_damp_on and outer_damp_w_eps > 0.0:
                    alpha_eps = _tikhonov_alpha(coeff_eps, outer_damp_lambda, outer_damp_w_eps)
                    eps_corr = (1.0 - alpha_eps) * eps_corr
                if config.clip_max_eps is not None:
                    eps_corr = float(np.clip(eps_corr, -config.clip_max_eps, config.clip_max_eps))
                eps = min(eps - eps_corr, 0.95)
                if eps < 0.0:
                    eps = min(-eps, 0.95)
                    pa = (pa + np.pi / 2) % np.pi
                if eps == 0.0:
                    eps = 0.05
        else:
            # 'largest' mode: update only the geometry parameter with the
            # largest |harmonic|.
            if max_idx == 0 and not config.fix_center:
                coeff = (1.0 - eps) / grad_joint
                aux = -max_amp * coeff * damping
                if outer_damp_on and outer_damp_w_center > 0.0:
                    alpha = _tikhonov_alpha(coeff, outer_damp_lambda, outer_damp_w_center)
                    aux = (1.0 - alpha) * aux
                if config.clip_max_shift is not None:
                    aux = float(np.clip(aux, -config.clip_max_shift, config.clip_max_shift))
                x0 -= aux * np.sin(pa)
                y0 += aux * np.cos(pa)
            elif max_idx == 1 and not config.fix_center:
                coeff = 1.0 / grad_joint
                aux = -max_amp * coeff * damping
                if outer_damp_on and outer_damp_w_center > 0.0:
                    alpha = _tikhonov_alpha(coeff, outer_damp_lambda, outer_damp_w_center)
                    aux = (1.0 - alpha) * aux
                if config.clip_max_shift is not None:
                    aux = float(np.clip(aux, -config.clip_max_shift, config.clip_max_shift))
                x0 += aux * np.cos(pa)
                y0 += aux * np.sin(pa)
            elif max_idx == 2 and not config.fix_pa:
                denom = (1.0 - eps) ** 2 - 1.0
                if abs(denom) < 1e-10:
                    denom = -1e-10
                coeff = 2.0 * (1.0 - eps) / sma / grad_joint / denom
                pa_corr = max_amp * coeff * damping
                if outer_damp_on and outer_damp_w_pa > 0.0:
                    alpha = _tikhonov_alpha(coeff, outer_damp_lambda, outer_damp_w_pa)
                    pa_corr = (1.0 - alpha) * pa_corr
                if config.clip_max_pa is not None:
                    pa_corr = float(np.clip(pa_corr, -config.clip_max_pa, config.clip_max_pa))
                pa = (pa + pa_corr) % np.pi
            elif max_idx == 3 and not config.fix_eps:
                coeff = 2.0 * (1.0 - eps) / sma / grad_joint
                eps_corr = max_amp * coeff * damping
                if outer_damp_on and outer_damp_w_eps > 0.0:
                    alpha = _tikhonov_alpha(coeff, outer_damp_lambda, outer_damp_w_eps)
                    eps_corr = (1.0 - alpha) * eps_corr
                if config.clip_max_eps is not None:
                    eps_corr = float(np.clip(eps_corr, -config.clip_max_eps, config.clip_max_eps))
                eps = min(eps - eps_corr, 0.95)
                if eps < 0.0:
                    eps = min(-eps, 0.95)
                    pa = (pa + np.pi / 2) % np.pi
                if eps == 0.0:
                    eps = 0.05

        if config.geometry_convergence and i >= config.minit:
            gx0, gy0, geps, gpa = prev_geom
            delta_x0 = abs(x0 - gx0) / max(sma, 1.0)
            delta_y0 = abs(y0 - gy0) / max(sma, 1.0)
            delta_eps = abs(eps - geps)
            delta_pa_raw = abs(pa - gpa)
            delta_pa = min(delta_pa_raw, np.pi - delta_pa_raw) / np.pi
            max_delta = max(delta_x0, delta_y0, delta_eps, delta_pa)
            if max_delta < config.geometry_tolerance:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= config.geometry_stable_iters:
                stop_code = 0
                converged = True
                if config.compute_deviations and best_geometry is not None:
                    if loose_validity and last_intens_per_band_loose is not None:
                        _attach_higher_harmonics_dispatch(
                            best_geometry,
                            bands,
                            config,
                            last_joint_coeffs,
                            last_phi_per_band_loose or [],
                            last_intens_per_band_loose,
                            last_variances_per_band_loose,
                            sma,
                            per_band_grad,
                            band_weights_arr,
                            jagged=True,
                            last_cov=last_joint_cov,
                            last_wls_mode=last_joint_wls_mode,
                            dropped_band_indices=set(last_dropped_band_indices),
                        )
                    else:
                        _attach_higher_harmonics_dispatch(
                            best_geometry,
                            bands,
                            config,
                            last_joint_coeffs,
                            angles,
                            intens_per_band,
                            variances_per_band,
                            sma,
                            per_band_grad,
                            band_weights_arr,
                            jagged=False,
                            last_cov=last_joint_cov,
                            last_wls_mode=last_joint_wls_mode,
                        )
                break

        prev_geom = (x0, y0, eps, pa)

    # Wrap up.
    if best_geometry is None:
        best_geometry = _empty_isophote_dict(
            sma,
            x0,
            y0,
            eps,
            pa,
            bands,
            config.use_eccentric_anomaly,
            stop_code if stop_code != 0 else 2,
            niter,
            debug,
            harmonic_orders=config.harmonic_orders,
        )

    if niter >= config.maxit and stop_code == 0 and not converged:
        stop_code = 2
        # Best-effort post-hoc harmonics from the final iteration's data.
        if config.compute_deviations and last_data is not None and last_data.valid_count > 6:
            if loose_validity and last_intens_per_band_loose is not None:
                _attach_higher_harmonics_dispatch(
                    best_geometry,
                    bands,
                    config,
                    last_joint_coeffs,
                    last_phi_per_band_loose or [],
                    last_intens_per_band_loose,
                    last_variances_per_band_loose,
                    sma,
                    last_per_band_grad if last_per_band_grad else [0.0] * n_bands,
                    band_weights_arr,
                    jagged=True,
                    last_cov=last_joint_cov,
                    last_wls_mode=last_joint_wls_mode,
                    dropped_band_indices=set(last_dropped_band_indices),
                )
            else:
                # M7: use the last iteration's post-clip arrays (the same
                # data last_joint_coeffs was fit on). last_data is the raw
                # sampler output; sigma clipping returns new arrays and
                # never mutates it, so last_data.* are pre-clip and would
                # put the removed outliers back into the harmonics.
                _attach_higher_harmonics_dispatch(
                    best_geometry,
                    bands,
                    config,
                    last_joint_coeffs,
                    angles,
                    intens_per_band,
                    variances_per_band,
                    sma,
                    last_per_band_grad if last_per_band_grad else [0.0] * n_bands,
                    band_weights_arr,
                    jagged=False,
                    last_cov=last_joint_cov,
                    last_wls_mode=last_joint_wls_mode,
                )

    if config.full_photometry:
        # Per-band aperture totals: same elliptical aperture, B independent
        # photometric integrations. tflux_e / tflux_c columns are written
        # per band as `tflux_e_<b>`, `tflux_c_<b>`.
        for b_idx, b in enumerate(bands):
            mask_b = None
            if isinstance(masks, np.ndarray):
                mask_b = masks
            elif masks is not None:
                mask_b = masks[b_idx]
            tflux_e_b, tflux_c_b, npix_e_b, npix_c_b = compute_aperture_photometry(
                np.asarray(images[b_idx]),
                mask_b,
                float(best_geometry["x0"]),
                float(best_geometry["y0"]),
                float(best_geometry["sma"]),
                float(best_geometry["eps"]),
                float(best_geometry["pa"]),
            )
            best_geometry[f"tflux_e_{b}"] = float(tflux_e_b)
            best_geometry[f"tflux_c_{b}"] = float(tflux_c_b)
            best_geometry[f"npix_e_{b}"] = int(npix_e_b)
            best_geometry[f"npix_c_{b}"] = int(npix_c_b)

    best_geometry["stop_code"] = stop_code
    best_geometry["niter"] = niter
    # D9 backport: stamp per-band surviving-sample counts. Under shared
    # validity these all equal ``ndata``; under loose validity they
    # reflect each band's own kept count after sigma clipping.
    _stamp_n_valid_per_band(best_geometry, bands, last_n_valid_per_band)
    return best_geometry


def _attach_per_band_harmonics(
    geom: Dict[str, object],
    bands: Sequence[str],
    angles: NDArray[np.float64],
    intens_per_band: NDArray[np.float64],
    variances_per_band: Optional[NDArray[np.float64]],
    sma: float,
    per_band_grad: Sequence[float],
    *,
    harmonic_orders: Sequence[int] = (3, 4),
    dropped_band_indices: Optional[set] = None,
) -> None:
    """Compute per-band a_n, b_n for each n in ``harmonic_orders`` and write them into ``geom``.

    Each band uses its own intensity vector and its own gradient,
    matching the single-band ``compute_deviations`` path. Note
    ``compute_deviations`` returns Bender-normalized coefficients
    (divided by ``sma*|grad|``), so these columns are stored
    *normalized* — unlike the shared/joint modes, which store raw
    coefficients for per-band normalization at plotting time (D16).
    Plotters must not normalize these a second time (review P1).

    Bands in ``dropped_band_indices`` (loose validity) are skipped so
    their columns keep the NaN drop marker written by the caller
    (review M5).

    Default ``harmonic_orders=(3, 4)`` reproduces Stage-1 behavior
    bit-for-bit; callers wanting orders 5, 6, ... pass them explicitly.
    """
    # The loose-validity caller passes per-band jagged ``angles`` /
    # ``intens_per_band`` / ``variances_per_band`` lists, where each
    # band's arrays may have different lengths.  The shared-validity
    # caller passes a single ``angles`` array (length N) and a
    # rectangular ``(B, N)`` ``intens_per_band``.  We accept both:
    angles_is_list = isinstance(angles, list)
    orders_list = [int(n) for n in harmonic_orders]
    dropped = dropped_band_indices or set()
    for b_idx, b in enumerate(bands):
        if b_idx in dropped:
            continue  # keep the caller's NaN drop marker (M5)
        if angles_is_list:
            ang_b = angles[b_idx]  # type: ignore[index]
        else:
            ang_b = angles
        intens_b = intens_per_band[b_idx]
        var_b = variances_per_band[b_idx] if variances_per_band is not None else None
        grad_b = float(per_band_grad[b_idx]) if b_idx < len(per_band_grad) else 0.0
        if intens_b.size == 0 or ang_b.size != intens_b.size:
            for n_order in orders_list:
                geom[f"a{n_order}_{b}"] = 0.0
                geom[f"b{n_order}_{b}"] = 0.0
                geom[f"a{n_order}_err_{b}"] = 0.0
                geom[f"b{n_order}_err_{b}"] = 0.0
            continue
        for n_order in orders_list:
            a, c, a_err, b_err = compute_deviations(
                ang_b,
                intens_b,
                sma,
                grad_b,
                n_order,
                variances=var_b,
            )
            geom[f"a{n_order}_{b}"] = float(a)
            geom[f"b{n_order}_{b}"] = float(c)
            geom[f"a{n_order}_err_{b}"] = float(a_err)
            geom[f"b{n_order}_err_{b}"] = float(b_err)


def _zero_init_per_band_higher_harmonics(
    geom: Dict[str, object],
    bands: Sequence[str],
    orders: Sequence[int],
    dropped_band_indices: Optional[set] = None,
) -> None:
    """Write zeros into every per-band higher-order column for the given orders.

    Mirrors the ``_empty_isophote_dict`` initialization but is callable on
    an already-built geometry dict. Used by the shared-mode refit when the
    solve fails or there are insufficient surviving rows.

    Bands in ``dropped_band_indices`` (loose validity) get NaN instead of
    0.0 so downstream consumers can detect the drop (review M5).
    """
    dropped = dropped_band_indices or set()
    for b_idx, b in enumerate(bands):
        fill = float("nan") if b_idx in dropped else 0.0
        for n_order in orders:
            geom[f"a{int(n_order)}_{b}"] = fill
            geom[f"b{int(n_order)}_{b}"] = fill
            geom[f"a{int(n_order)}_err_{b}"] = fill
            geom[f"b{int(n_order)}_err_{b}"] = fill


def _attach_shared_higher_harmonics(
    geom: Dict[str, object],
    bands: Sequence[str],
    last_coeffs: Optional[NDArray[np.float64]],
    angles: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    intens_per_band: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    variances_per_band: Union[None, NDArray[np.float64], List[NDArray[np.float64]]],
    sma: float,
    per_band_grad: Sequence[float],
    *,
    harmonic_orders: Sequence[int],
    band_weights_arr: NDArray[np.float64],
    jagged: bool,
    dropped_band_indices: Optional[set] = None,
    var_residual_floor: Optional[float] = None,
) -> None:
    """Compute SHARED higher-order harmonic coefficients and write per-band columns.

    Locked design (Section 6, Q-R4-1): re-fit ONLY higher-order coefficients
    (n in ``harmonic_orders``); freeze (A1, B1, A2, B2) and per-band ``I0_b``
    at their converged-iteration values from ``last_coeffs``. The post-hoc
    design matrix has shape ``(B_eff*N_b, 2*L)`` where ``L = len(harmonic_orders)``
    and one (sin/cos) pair of columns per order is shared across all bands.

    Per Schema 1: every band's ``a{n}_{b}``, ``b{n}_{b}``, ``a{n}_err_{b}``,
    ``b{n}_err_{b}`` columns receive the IDENTICAL shared value. Per-band
    Bender normalization at plotting time (D16) scales the same raw value by
    ``1/(sma * |dI/da_b|)`` per band so normalized curves separate visually.

    Parameters
    ----------
    last_coeffs : (B + 4,) float64
        Last iteration's joint-solve coefficient vector ``[I0_0, ..., I0_{B-1},
        A1, B1, A2, B2]``. Frozen as the geometric model that residuals
        subtract before fitting higher orders. Falls back to per-band
        independent fits if ``None`` or wrongly shaped.
    angles : (N,) ndarray (shared validity) or list of (N_b,) ndarrays (jagged)
        Per-pixel sample angles. Set ``jagged=True`` for the loose-validity
        per-band layout.
    band_weights_arr : (B,) float64
        Per-band scalar weights (already resolved). Each band's row block in
        the post-hoc design matrix is row-scaled by ``w_b / variance_b``
        (WLS) or by ``w_b`` (OLS), composing with the joint solver's D12
        convention.
    dropped_band_indices : set of int, optional
        Loose-validity bands that were dropped at this isophote. Their per-band
        rows are skipped in the joint refit; their ``a{n}_<b>`` columns stay
        at zero (the surrounding caller marks ``intens_<b>`` NaN already).
    """
    n_bands = len(bands)
    if n_bands == 0:
        return

    orders_list = list(harmonic_orders)
    L = len(orders_list)
    if L == 0:
        return

    # Default per-band columns to zero up front so any early-exit path leaves
    # a self-consistent geom dict.
    _zero_init_per_band_higher_harmonics(geom, bands, orders_list, dropped_band_indices)

    # Need converged-iteration coefficients to subtract the frozen geometric
    # model. If unavailable (degenerate maxit fallback), fall through to the
    # per-band independent fit so we still produce sensible higher-order
    # numbers rather than silently zeroing everything.
    if last_coeffs is None or last_coeffs.size < n_bands + 4:
        _attach_per_band_harmonics(
            geom,
            bands,
            angles,
            intens_per_band,
            variances_per_band,
            sma,
            per_band_grad,
            harmonic_orders=orders_list,
            dropped_band_indices=dropped_band_indices,
        )
        return

    A1 = float(last_coeffs[n_bands])
    B1 = float(last_coeffs[n_bands + 1])
    A2 = float(last_coeffs[n_bands + 2])
    B2 = float(last_coeffs[n_bands + 3])

    dropped: set = dropped_band_indices or set()

    rows_a: List[NDArray[np.float64]] = []
    rows_y: List[NDArray[np.float64]] = []
    rows_w: List[NDArray[np.float64]] = []
    wls_mode = variances_per_band is not None

    for b_idx in range(n_bands):
        if b_idx in dropped:
            continue
        if jagged:
            ang_b = angles[b_idx]  # type: ignore[index]
            int_b = intens_per_band[b_idx]  # type: ignore[index]
            var_b = (
                variances_per_band[b_idx]  # type: ignore[index]
                if variances_per_band is not None
                else None
            )
        else:
            ang_b = angles  # type: ignore[assignment]
            int_b = intens_per_band[b_idx]
            var_b = (
                variances_per_band[b_idx]  # type: ignore[index]
                if variances_per_band is not None
                else None
            )
        if int_b is None or int_b.size == 0 or ang_b.size != int_b.size:
            continue
        if not np.isfinite(last_coeffs[b_idx]):
            # Loose-validity band that ended up NaN'd in the joint coeffs.
            continue

        i0_b = float(last_coeffs[b_idx])
        geom_pred = i0_b + A1 * np.sin(ang_b) + B1 * np.cos(ang_b) + A2 * np.sin(2.0 * ang_b) + B2 * np.cos(2.0 * ang_b)
        residual_b = int_b - geom_pred

        a_b = np.empty((ang_b.size, 2 * L), dtype=np.float64)
        for j, n_order in enumerate(orders_list):
            a_b[:, 2 * j] = np.sin(int(n_order) * ang_b)
            a_b[:, 2 * j + 1] = np.cos(int(n_order) * ang_b)

        w_band = float(band_weights_arr[b_idx]) if b_idx < band_weights_arr.size else 1.0
        if var_b is not None:
            row_w = w_band / var_b
        else:
            row_w = np.full(ang_b.size, w_band, dtype=np.float64)

        rows_a.append(a_b)
        rows_y.append(residual_b)
        rows_w.append(row_w)

    if not rows_a:
        return

    a_full = np.concatenate(rows_a, axis=0)
    y_full = np.concatenate(rows_y, axis=0)
    w_full = np.concatenate(rows_w, axis=0)

    n_rows = a_full.shape[0]
    if n_rows < 2 * L:
        # Underdetermined; per-band columns already zeroed.
        return

    aw = a_full * w_full[:, None]
    atwa = aw.T @ a_full
    atwy = aw.T @ y_full

    try:
        coeffs_higher = np.linalg.solve(atwa, atwy)
        atwa_inv = np.linalg.inv(atwa)
        if wls_mode:
            cov_higher = atwa_inv
        else:
            # OLS: scale (A^T A)^-1 by residual variance with the row weights
            # the same way the joint solver does. Effective DOF = n_rows - 2L.
            model = a_full @ coeffs_higher
            ddof = max(n_rows - 2 * L, 1)
            var_res = float(np.sum(w_full * (y_full - model) ** 2) / ddof)
            # Same sigma_bg floor as every other OLS error scale on this row.
            cov_higher = atwa_inv * _apply_variance_floor(var_res, var_residual_floor)
        errors_higher = np.sqrt(np.maximum(np.diagonal(cov_higher), 0.0))
    except np.linalg.LinAlgError:
        return

    for j, n_order in enumerate(orders_list):
        a_n = float(coeffs_higher[2 * j])
        b_n = float(coeffs_higher[2 * j + 1])
        a_n_err = float(errors_higher[2 * j])
        b_n_err = float(errors_higher[2 * j + 1])
        for b_idx, b in enumerate(bands):
            if b_idx in dropped:
                continue
            geom[f"a{int(n_order)}_{b}"] = a_n
            geom[f"b{int(n_order)}_{b}"] = b_n
            geom[f"a{int(n_order)}_err_{b}"] = a_n_err
            geom[f"b{int(n_order)}_err_{b}"] = b_n_err


def _compute_joint_residual_variance(
    coeffs: NDArray[np.float64],
    angles: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    intens_per_band: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    band_weights_arr: NDArray[np.float64],
    *,
    n_bands: int,
    n_geom_params: int,
    jagged: bool,
    band_indices: Optional[Sequence[int]] = None,
    floor: Optional[float] = None,
    harmonic_orders: Optional[Sequence[int]] = None,
) -> Optional[float]:
    """Residual variance of a joint solve, weighted by per-band band weights.

    Mirrors the OLS rescale used in :func:`_compute_parameter_errors_from_joint`
    (rectangular shared-validity case) but extended to handle the loose-
    validity jagged case as well: every surviving sample contributes
    ``w_b * (intens - model)**2`` and ``ddof = n_geom_params``.

    ``band_indices``: optional full-list band index for each jagged entry.
    Required under loose validity when bands have been dropped, so the
    per-band intercept and weight are read at the correct positions of the
    full ``coeffs`` / ``band_weights_arr`` vectors.

    ``harmonic_orders`` must be supplied whenever ``coeffs`` carries a trailing
    simultaneous higher-order block, so the residual is measured against the
    *complete* fitted model. Omitting them evaluates a truncated model and
    charges the fitted m=n structure to the noise: on a noiseless synthetic
    built from its own coefficients — where the residual variance must be
    exactly zero — the truncated form returned 30.5. The retired comment here
    called that "conservative", which is true only in sign. This mirrors the
    single-band fix in ``fit_isophote``, where the residual variance comes from
    the exact fitted model with ``ddof = len(coeffs)``.

    ``floor`` is the ``sigma_bg**2`` background-noise floor. It is applied here,
    once, so that every consumer of a given solve's residual variance shares one
    error scale — see :func:`_sigma_bg_variance_floor`.

    An exactly-determined or under-determined solve returns ``0.0``, not
    ``None``: the model passes through every point, so the data carry no
    information about the noise. Returning ``None`` there made callers skip the
    OLS rescale entirely and publish a raw ``(A^T A)^-1`` diagonal, which is an
    arbitrary scale rather than an error. ``None`` now means only that an input
    was malformed. Matches single-band ``fit_isophote``.
    """
    try:
        if jagged:
            assert isinstance(angles, list) and isinstance(intens_per_band, list)
            sse = 0.0
            n_samples = 0
            for pos, (ang_b, int_b) in enumerate(zip(angles, intens_per_band)):
                if ang_b is None or int_b is None or ang_b.size == 0:
                    continue
                b_idx = int(band_indices[pos]) if band_indices is not None else pos
                # Build per-band model over this band's surviving angles.
                I0_b = float(coeffs[b_idx])
                A1 = float(coeffs[n_bands])
                B1 = float(coeffs[n_bands + 1])
                A2 = float(coeffs[n_bands + 2])
                B2 = float(coeffs[n_bands + 3])
                model = (
                    I0_b + A1 * np.sin(ang_b) + B1 * np.cos(ang_b) + A2 * np.sin(2.0 * ang_b) + B2 * np.cos(2.0 * ang_b)
                )
                # Trailing simultaneous higher-order block, when the caller
                # fitted one. These terms are part of the model, so they belong
                # in the residual too.
                if harmonic_orders:
                    for j, n_order in enumerate(harmonic_orders):
                        a_n = float(coeffs[n_bands + 4 + 2 * j])
                        b_n = float(coeffs[n_bands + 4 + 2 * j + 1])
                        model = model + a_n * np.sin(int(n_order) * ang_b) + b_n * np.cos(int(n_order) * ang_b)
                w_b = float(band_weights_arr[b_idx]) if band_weights_arr is not None else 1.0
                sse += float(np.sum(w_b * (int_b - model) ** 2))
                n_samples += int(ang_b.size)
            var_residual = 0.0 if n_samples <= n_geom_params else sse / float(n_samples - n_geom_params)
            return _apply_variance_floor(var_residual, floor)
        else:
            # Rectangular shared-validity: all bands share the same angles.
            # ``intens_per_band`` is shape (B, N).
            assert isinstance(intens_per_band, np.ndarray) and isinstance(angles, np.ndarray)
            model_full = evaluate_joint_model(angles, coeffs, n_bands, harmonic_orders=harmonic_orders)
            res = intens_per_band - model_full
            # Apply per-band weights row-wise so the OLS rescale matches
            # the row-weighted normal-equation solve.
            if band_weights_arr is not None:
                w = np.asarray(band_weights_arr, dtype=np.float64).reshape(-1)
                res = res * np.sqrt(np.maximum(w, 0.0))[:, None]
            flat = res.reshape(-1)
            var_residual = (
                0.0 if flat.size <= n_geom_params else float(np.sum(flat * flat) / (flat.size - n_geom_params))
            )
            return _apply_variance_floor(var_residual, floor)
    except Exception:  # noqa: BLE001 — defensive fallback
        return None


def _attach_simultaneous_higher_harmonics_from_coeffs(
    geom: Dict[str, object],
    bands: Sequence[str],
    last_coeffs: Optional[NDArray[np.float64]],
    last_cov: Optional[NDArray[np.float64]],
    *,
    harmonic_orders: Sequence[int],
    dropped_band_indices: Optional[set] = None,
    wls_mode: bool = False,
    angles: Union[None, NDArray[np.float64], List[NDArray[np.float64]]] = None,
    intens_per_band: Union[None, NDArray[np.float64], List[NDArray[np.float64]]] = None,
    band_weights_arr: Optional[NDArray[np.float64]] = None,
    jagged: bool = False,
    var_residual_floor: Optional[float] = None,
) -> None:
    """Stamp shared higher-order coefficients straight from the wider iteration coeffs.

    Used by ``simultaneous_in_loop`` mode where the iteration-loop joint
    solver already returned a ``(B + 4 + 2*L,)`` coefficient vector and a
    ``(B + 4 + 2*L, B + 4 + 2*L)`` covariance matrix every iteration. No
    additional solve is needed at convergence; we only need to write the
    shared coefficients into per-band columns so Schema 1 stays
    bit-compatible.

    OLS rescale (review fix B1): when ``wls_mode=False``, the joint solver
    returns the raw ``(A^T A)^-1`` covariance and the caller must scale by
    the residual variance before taking the diagonal sqrt to recover
    standard errors. ``angles`` / ``intens_per_band`` /
    ``band_weights_arr`` / ``jagged`` enable this rescale; if any are
    ``None`` the rescale is skipped and the (incorrectly-scaled) raw
    diagonal is used. WLS path keeps the exact MLE covariance unchanged.

    Per-band columns ``a{n}_{b}``, ``b{n}_{b}`` carry the identical shared
    value across bands; corresponding error columns carry the joint-solve
    standard error (also shared across bands).
    """
    n_bands = len(bands)
    orders = list(harmonic_orders)
    L = len(orders)

    _zero_init_per_band_higher_harmonics(geom, bands, orders, dropped_band_indices)
    if last_coeffs is None or last_coeffs.size < n_bands + 4 + 2 * L:
        return

    if last_cov is not None and last_cov.shape == (
        n_bands + 4 + 2 * L,
        n_bands + 4 + 2 * L,
    ):
        cov_for_errs = last_cov
        # OLS rescale by residual variance, mirroring the joint solver's
        # contract (caller scales by σ²). WLS already returns exact cov.
        if not wls_mode and angles is not None and intens_per_band is not None and band_weights_arr is not None:
            # Under loose validity with dropped bands, restrict the residual
            # variance to the surviving bands and map their positions in the
            # full vectors (review M3); otherwise the dropped bands' NaN
            # intercepts silently killed the rescale.
            res_angles, res_intens, res_idx = angles, intens_per_band, None
            if jagged and dropped_band_indices:
                res_idx = [b for b in range(n_bands) if b not in dropped_band_indices]
                res_angles = [angles[b] for b in res_idx]  # type: ignore[index]
                res_intens = [intens_per_band[b] for b in res_idx]  # type: ignore[index]
            var_residual = _compute_joint_residual_variance(
                last_coeffs,
                res_angles,
                res_intens,
                band_weights_arr,
                n_bands=n_bands,
                n_geom_params=(len(res_idx) if res_idx is not None else n_bands) + 4 + 2 * L,
                jagged=jagged,
                band_indices=res_idx,
                floor=var_residual_floor,
                harmonic_orders=orders,
            )
            if var_residual is not None and np.isfinite(var_residual):
                cov_for_errs = last_cov * var_residual
        diag = np.maximum(np.diagonal(cov_for_errs), 0.0)
        errs = np.sqrt(diag)
    else:
        errs = np.zeros(n_bands + 4 + 2 * L, dtype=np.float64)

    dropped: set = dropped_band_indices or set()
    for j, n_order in enumerate(orders):
        a_n = float(last_coeffs[n_bands + 4 + 2 * j])
        b_n = float(last_coeffs[n_bands + 4 + 2 * j + 1])
        a_n_err = float(errs[n_bands + 4 + 2 * j])
        b_n_err = float(errs[n_bands + 4 + 2 * j + 1])
        for b_idx, b in enumerate(bands):
            if b_idx in dropped:
                continue
            geom[f"a{int(n_order)}_{b}"] = a_n
            geom[f"b{int(n_order)}_{b}"] = b_n
            geom[f"a{int(n_order)}_err_{b}"] = a_n_err
            geom[f"b{int(n_order)}_err_{b}"] = b_n_err


def _attach_simultaneous_original_post_hoc(
    geom: Dict[str, object],
    bands: Sequence[str],
    config: IsosterConfigMB,
    angles: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    intens_per_band: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    variances_per_band: Union[None, NDArray[np.float64], List[NDArray[np.float64]]],
    band_weights_arr: NDArray[np.float64],
    *,
    jagged: bool,
    dropped_band_indices: Optional[set] = None,
) -> None:
    """Run ONE post-hoc joint solve over (I0_b, A1, B1, A2, B2, A_n, B_n).

    Implements ``simultaneous_original`` (Ciambur 2015 original variant):
    the iteration loop ran the standard 5-parameter joint solver
    (B + 4 columns), and after convergence we solve the wider
    (B + 4 + 2L) system once over the converged-geometry samples to get
    higher-order coefficients fitted simultaneously with all geometry
    nuisance parameters. The (A1, B1, A2, B2) values from this post-hoc
    solve typically agree with the converged-loop values to numerical
    precision; we accept the post-hoc values for the higher-order
    columns but do NOT change the converged geometry parameters
    (x0, y0, eps, pa) on ``geom``.

    Per-band columns receive the identical shared higher-order
    coefficients (and shared errors).
    """
    n_bands = len(bands)
    orders = list(config.harmonic_orders)
    L = len(orders)
    _zero_init_per_band_higher_harmonics(geom, bands, orders, dropped_band_indices)
    if L == 0 or n_bands == 0:
        return

    dropped: set = dropped_band_indices or set()
    if jagged:
        # Surviving-band subset for the loose-validity post-hoc solve.
        surviving_idx = [b for b in range(n_bands) if b not in dropped]
        if not surviving_idx:
            return
        phi_sub = [angles[b] for b in surviving_idx]  # type: ignore[index]
        int_sub = [intens_per_band[b] for b in surviving_idx]  # type: ignore[index]
        var_sub = (
            [variances_per_band[b] for b in surviving_idx]  # type: ignore[index]
            if variances_per_band is not None
            else None
        )
        sub_weights = band_weights_arr[np.array(surviving_idx, dtype=np.int64)]
        coeffs_sub, _cov_sub, _wls = fit_simultaneous_joint_loose(
            phi_sub,
            int_sub,
            sub_weights,
            orders,
            var_sub,
            normalize=(config.loose_validity_band_normalization == "per_band_count"),
            fit_per_band_intens_jointly=config.fit_per_band_intens_jointly,
            integrator=config.integrator,
        )
        # Errors come from the surviving-bands cov; we just need the shared
        # higher-order block diagonal for the per-band column writes.
        n_surv = len(surviving_idx)
        a_n_block = coeffs_sub[n_surv + 4 :]  # 2L entries
        # Error stamping: use the cov diagonal of the surviving-bands solve.
        cov_sub_full = _cov_sub
        if cov_sub_full is not None and cov_sub_full.shape[0] >= n_surv + 4 + 2 * L:
            cov_for_errs = cov_sub_full
            # OLS rescale (review fix B1): solver returns (A^T A)^-1; scale
            # by surviving-bands residual variance before sqrt.
            if not _wls:
                var_res = _compute_joint_residual_variance(
                    coeffs_sub,
                    phi_sub,
                    int_sub,
                    sub_weights,
                    n_bands=n_surv,
                    n_geom_params=n_surv + 4 + 2 * L,
                    jagged=True,
                    floor=_sigma_bg_variance_floor(config),
                    harmonic_orders=orders,
                )
                if var_res is not None and np.isfinite(var_res):
                    cov_for_errs = cov_sub_full * var_res
            diag = np.maximum(np.diagonal(cov_for_errs)[n_surv + 4 :], 0.0)
            errs_block = np.sqrt(diag)
        else:
            errs_block = np.zeros(2 * L, dtype=np.float64)
    else:
        coeffs_full, cov_full, _wls = fit_simultaneous_joint(
            angles,
            intens_per_band,
            band_weights_arr,
            orders,
            variances_per_band,
            fit_per_band_intens_jointly=config.fit_per_band_intens_jointly,
            integrator=config.integrator,
        )
        a_n_block = coeffs_full[n_bands + 4 :]
        if cov_full is not None and cov_full.shape[0] >= n_bands + 4 + 2 * L:
            cov_for_errs = cov_full
            # OLS rescale (review fix B1): solver returns (A^T A)^-1; scale
            # by residual variance before sqrt for true SE.
            if not _wls:
                var_res = _compute_joint_residual_variance(
                    coeffs_full,
                    angles,
                    intens_per_band,
                    band_weights_arr,
                    n_bands=n_bands,
                    n_geom_params=n_bands + 4 + 2 * L,
                    jagged=False,
                    floor=_sigma_bg_variance_floor(config),
                    harmonic_orders=orders,
                )
                if var_res is not None and np.isfinite(var_res):
                    cov_for_errs = cov_full * var_res
            diag = np.maximum(np.diagonal(cov_for_errs)[n_bands + 4 :], 0.0)
            errs_block = np.sqrt(diag)
        else:
            errs_block = np.zeros(2 * L, dtype=np.float64)

    for j, n_order in enumerate(orders):
        a_n = float(a_n_block[2 * j])
        b_n = float(a_n_block[2 * j + 1])
        a_n_err = float(errs_block[2 * j])
        b_n_err = float(errs_block[2 * j + 1])
        for b_idx, b in enumerate(bands):
            if b_idx in dropped:
                continue
            geom[f"a{int(n_order)}_{b}"] = a_n
            geom[f"b{int(n_order)}_{b}"] = b_n
            geom[f"a{int(n_order)}_err_{b}"] = a_n_err
            geom[f"b{int(n_order)}_err_{b}"] = b_n_err


def _attach_higher_harmonics_dispatch(
    geom: Dict[str, object],
    bands: Sequence[str],
    config: IsosterConfigMB,
    last_coeffs: Optional[NDArray[np.float64]],
    angles: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    intens_per_band: Union[NDArray[np.float64], List[NDArray[np.float64]]],
    variances_per_band: Union[None, NDArray[np.float64], List[NDArray[np.float64]]],
    sma: float,
    per_band_grad: Sequence[float],
    band_weights_arr: NDArray[np.float64],
    *,
    jagged: bool,
    last_cov: Optional[NDArray[np.float64]] = None,
    last_wls_mode: bool = False,
    dropped_band_indices: Optional[set] = None,
) -> None:
    """Pick the higher-order harmonic attachment path based on config.

    Routing per Section 6:

    - ``'independent'`` (default): per-band, per-order, uncoupled across bands.
      Reproduces the Stage-1 ``_attach_per_band_harmonics`` behavior bit-for-bit.
    - ``'shared'``: ONE post-hoc joint refit with shared higher-order
      coefficients across bands; freezes (A1,B1,A2,B2) and per-band I0_b at
      the converged-loop values.
    - ``'simultaneous_in_loop'``: per-iteration joint solve already produced
      shared higher-order coefficients; just stamp them from ``last_coeffs``.
    - ``'simultaneous_original'``: ONE post-hoc joint solve over the wider
      ``(B + 4 + 2L)`` system; refits all coefficients simultaneously.
    """
    mode = getattr(config, "multiband_higher_harmonics", "independent")
    if mode == "shared":
        _attach_shared_higher_harmonics(
            geom,
            bands,
            last_coeffs,
            angles,
            intens_per_band,
            variances_per_band,
            sma,
            per_band_grad,
            harmonic_orders=config.harmonic_orders,
            band_weights_arr=band_weights_arr,
            jagged=jagged,
            dropped_band_indices=dropped_band_indices,
            var_residual_floor=_sigma_bg_variance_floor(config),
        )
    elif mode == "simultaneous_in_loop":
        _attach_simultaneous_higher_harmonics_from_coeffs(
            geom,
            bands,
            last_coeffs,
            last_cov,
            harmonic_orders=config.harmonic_orders,
            dropped_band_indices=dropped_band_indices,
            wls_mode=last_wls_mode,
            angles=angles,
            intens_per_band=intens_per_band,
            band_weights_arr=band_weights_arr,
            jagged=jagged,
            var_residual_floor=_sigma_bg_variance_floor(config),
        )
    elif mode == "simultaneous_original":
        _attach_simultaneous_original_post_hoc(
            geom,
            bands,
            config,
            angles,
            intens_per_band,
            variances_per_band,
            band_weights_arr,
            jagged=jagged,
            dropped_band_indices=dropped_band_indices,
        )
    else:
        # 'independent' (default).
        _attach_per_band_harmonics(
            geom,
            bands,
            angles,
            intens_per_band,
            variances_per_band,
            sma,
            per_band_grad,
            harmonic_orders=config.harmonic_orders,
            dropped_band_indices=dropped_band_indices,
        )


# ---------------------------------------------------------------------------
# Forced multi-band photometry helper (used by the driver's central pixel
# and template fallback paths)
# ---------------------------------------------------------------------------


def extract_forced_photometry_mb(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    x0: float,
    y0: float,
    sma: float,
    eps: float,
    pa: float,
    bands: Sequence[str],
    config: IsosterConfigMB,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
    *,
    return_ring_data: bool = False,
) -> Union[Dict[str, object], Tuple[Dict[str, object], object]]:
    """
    Single-isophote forced multi-band extraction (no fitting).

    Ring samples are sigma-clipped with ``config.sclip`` / ``nclip``
    (shared-validity AND across bands) before any statistics, mirroring
    the single-band forced path and the iteration loop.

    Used by the driver as the central-pixel record for ``minsma == 0.0``
    growth and as a defensive fallback when an isophote fails the
    iterative fit. Produces the same per-isophote dict layout as
    :func:`fit_isophote_mb` with ``stop_code=0`` and ``niter=0``.

    Stage-H.1: when ``return_ring_data=True``, returns a
    ``(geom, ring_data)`` tuple where ``ring_data`` is the
    ``MultiIsophoteData`` of the *clipped* ring (or None on empty
    extraction). The forced-photometry orchestrator uses this to compute
    per-band harmonic deviations post-hoc with neighbor-derived per-band
    gradients, mirroring the iteration loop's
    ``_attach_per_band_harmonics`` path. Default ``False`` preserves
    bit-identical behavior for all existing callers.
    """
    debug = bool(config.debug)
    band_list = list(bands)
    loose_validity = bool(config.loose_validity)
    data = extract_isophote_data_multi(
        images,
        masks,
        x0,
        y0,
        sma,
        eps,
        pa,
        use_eccentric_anomaly=config.use_eccentric_anomaly,
        variance_maps=variance_maps,
        loose_validity=loose_validity,
    )
    # Sigma-clip the ring samples before any statistics, mirroring the
    # single-band forced path and the iteration loop. Without this,
    # ~sclip-sigma outliers (unmasked companions, cosmic rays) entered
    # intens_<b>/rms_<b> directly (review M4). No-op when nclip=0.
    #
    # Forced photometry runs no joint solve — every band's intensity, rms and
    # error is computed independently in the per-band loop below — so there is
    # no rectangular design matrix to preserve and loose validity needs nothing
    # beyond keeping each band's own samples.
    if loose_validity:
        phi_pb, intens_pb, vars_pb, _n_clipped = _per_band_sigma_clip_loose(
            data.phi_per_band if data.phi_per_band is not None else [data.phi] * len(band_list),
            data.intens_per_band if data.intens_per_band is not None else list(data.intens),
            data.variances_per_band,
            config.sclip,
            config.nclip,
            config.sclip_low,
            config.sclip_high,
        )
        # Legacy rectangular views are unused under loose validity; the per-band
        # lists below carry the truth.
        angles_c = phi_c = np.concatenate(phi_pb) if phi_pb else np.empty(0, dtype=np.float64)
        intens_c = intens_pb
        variances_c = vars_pb
        per_band_empty = [arr.size == 0 for arr in intens_pb]
    else:
        angles_c, phi_c, intens_c, variances_c, _n_clipped = _per_band_sigma_clip(
            data.angles,
            data.phi,
            data.intens,
            data.variances,
            config.sclip,
            config.nclip,
            config.sclip_low,
            config.sclip_high,
        )
        per_band_empty = [angles_c.size == 0] * len(band_list)

    # Under shared validity an empty intersection ends the isophote. Under loose
    # validity it ends only when *every* band is empty: a band fully masked at
    # this radius must not discard the bands that still have usable samples.
    if all(per_band_empty):
        empty = _empty_isophote_dict(
            sma,
            x0,
            y0,
            eps,
            pa,
            band_list,
            config.use_eccentric_anomaly,
            stop_code=3,
            niter=0,
            debug=debug,
            harmonic_orders=config.harmonic_orders,
        )
        if return_ring_data:
            return empty, None
        return empty

    geom: Dict[str, object] = {
        "sma": sma,
        "x0": x0,
        "y0": y0,
        "eps": eps,
        "pa": pa,
        "x0_err": 0.0,
        "y0_err": 0.0,
        "eps_err": 0.0,
        "pa_err": 0.0,
        "rms": float("nan"),
        "stop_code": 0,
        "niter": 0,
        "valid": True,
        "use_eccentric_anomaly": config.use_eccentric_anomaly,
        "tflux_e": float("nan"),
        "tflux_c": float("nan"),
        "npix_e": 0,
        "npix_c": 0,
    }
    if debug:
        # Under loose validity the bands no longer share one kept set, so
        # ``ndata`` reports the total across bands and ``nflag`` the total
        # rejected; ``n_valid_<b>`` below carries each band's own count.
        if loose_validity:
            attempted = data.n_samples * len(band_list)
            kept = int(sum(arr.size for arr in intens_c))
            geom["ndata"] = kept
            geom["nflag"] = attempted - kept
        else:
            geom["ndata"] = data.valid_count
            geom["nflag"] = data.n_samples - data.valid_count

    # Asymptotic Gaussian factor for the median's standard error
    # relative to the mean's: sqrt(π/2) ≈ 1.2533 (review fix H4).
    _MEDIAN_SEM_FACTOR = float(np.sqrt(np.pi / 2.0))

    for b_idx, b in enumerate(band_list):
        intens_b = intens_c[b_idx]
        n_b = int(len(intens_b))
        if n_b == 0:
            # Loose validity only: this band has no usable samples at this
            # radius while others do. NaN marks the absence, matching the
            # dropped-band convention in the fitted rows (M5); 0.0 would read
            # as a real measurement.
            geom[f"intens_{b}"] = float("nan")
            geom[f"intens_err_{b}"] = float("nan")
            geom[f"rms_{b}"] = float("nan")
            geom[f"n_valid_{b}"] = 0
            for n_order in config.harmonic_orders:
                geom[f"a{int(n_order)}_{b}"] = float("nan")
                geom[f"b{int(n_order)}_{b}"] = float("nan")
                geom[f"a{int(n_order)}_err_{b}"] = float("nan")
                geom[f"b{int(n_order)}_err_{b}"] = float("nan")
            if debug:
                geom[f"grad_{b}"] = float("nan")
                geom[f"grad_error_{b}"] = float("nan")
                geom[f"grad_r_error_{b}"] = float("nan")
            continue
        if variances_c is not None:
            v_b = variances_c[b_idx]
            weights = 1.0 / v_b
            sum_w = float(weights.sum())
            intens_val = float((weights * intens_b).sum() / sum_w)
            intens_err = float(1.0 / np.sqrt(sum_w))
        else:
            # Population standard deviation (ddof=0), matching single-band
            # ``extract_forced_photometry``; the Gaussian-asymptotic median
            # factor sqrt(π/2) applies when integrator='median'. Review fix
            # H4 originally used ddof=1 here, which left the two paths
            # disagreeing by sqrt(N/(N-1)) — about 0.8% at 64 samples.
            # The n < 2 guard is kept: single-band reports 0.0 there, which
            # reads as a measured certainty rather than an absent one.
            if n_b >= 2:
                sample_std = float(np.std(intens_b))
                base_sem = sample_std / np.sqrt(n_b)
            else:
                sample_std = 0.0
                base_sem = float("nan")
            if config.integrator == "median":
                intens_val = float(np.median(intens_b))
                intens_err = base_sem * _MEDIAN_SEM_FACTOR if np.isfinite(base_sem) else float("nan")
            else:
                intens_val = float(np.mean(intens_b))
                intens_err = base_sem
        # rms_<b> reports the ring-intensity dispersion (population std,
        # ddof=0). This conflates noise with any unmodeled harmonic
        # signal but matches the iteration-loop's rms convention; H4 is
        # only about the standard-error scaling above, not rms.
        rms_b = float(np.std(intens_b))
        geom[f"intens_{b}"] = intens_val
        geom[f"intens_err_{b}"] = intens_err
        geom[f"rms_{b}"] = rms_b
        # Per-band surviving-sample count, matching the fitted rows (M13)
        geom[f"n_valid_{b}"] = n_b
        for n_order in config.harmonic_orders:
            geom[f"a{int(n_order)}_{b}"] = 0.0
            geom[f"b{int(n_order)}_{b}"] = 0.0
            geom[f"a{int(n_order)}_err_{b}"] = 0.0
            geom[f"b{int(n_order)}_err_{b}"] = 0.0
        if debug:
            geom[f"grad_{b}"] = float("nan")
            geom[f"grad_error_{b}"] = float("nan")
            geom[f"grad_r_error_{b}"] = float("nan")
    if return_ring_data:
        # Return the clipped ring so the orchestrator's post-hoc harmonics
        # are computed on the same outlier-free samples as the statistics.
        if loose_validity:
            # Jagged layout: the per-band lists carry the clipped samples and
            # the rectangular fields are left as the sampler's intersection
            # view, matching MultiIsophoteData's documented loose contract.
            counts = np.array([arr.size for arr in intens_c], dtype=np.int64)
            ring = replace(
                data,
                radii=np.full(int(counts.max()) if counts.size else 0, sma, dtype=np.float64),
                intens_per_band=intens_c,
                phi_per_band=[np.asarray(p, dtype=np.float64) for p in phi_pb],
                variances_per_band=variances_c,
                n_valid_per_band=counts,
            )
        else:
            ring = replace(
                data,
                angles=angles_c,
                phi=phi_c,
                intens=intens_c,
                radii=np.full(angles_c.size, sma, dtype=np.float64),
                variances=variances_c,
                valid_count=int(angles_c.size),
                n_valid_per_band=np.full(len(band_list), int(angles_c.size), dtype=np.int64),
            )
        return geom, ring
    return geom
