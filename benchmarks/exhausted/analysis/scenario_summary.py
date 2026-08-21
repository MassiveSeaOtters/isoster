"""Shared primitives for per-campaign scenario-level summaries.

Every exhausted-benchmark campaign writes a regular tree of per-arm
``profile.fits`` / ``inventory.fits`` / ``run_record.json`` and a per-
galaxy ``MANIFEST.json``. This module exposes cheap loaders and a
single metrics entry point that downstream analysis drivers compose.

Phase F.0 is the first consumer: it audits the ``clean_z005`` scenario
against the priors implied by the axisymmetric multi-Sersic mocks. See
``clean_z005_audit.py`` for the driver.

The three radial zones (``inner`` / ``mid`` / ``outer``) are taken from
``residual_zones.py`` so the same bins used for image residual RMS are
reused for 1-D isophote metrics. Because every pixel on an isophote
satisfies ``r_ell == sma`` (by construction), the zone thresholds in
``sma`` space match the 2-D pixel thresholds exactly.

Design goals:

- Picklable, I/O only helpers (so a multiprocess driver can fan out
  across galaxies later without adapter-class pickling concerns).
- Pure-function metrics that accept an ``astropy.Table`` profile plus
  a primitive manifest dict, so the same code can score isoster,
  photutils, or autoprof profiles without tool-specific branches.
- Zone masks are computed once per profile and passed to every metric
  so callers never recompute them and drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table

# ---------------------------------------------------------------------------
# Manifest + profile loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GalaxyManifest:
    """Flat view of ``<galaxy_dir>/MANIFEST.json``.

    ``re_outer_pix`` is the flux-independent upper envelope of the
    per-component effective radii (max over ``RE_PX{i}``); retained as
    a diagnostic even though the current prior-2/3 zone definitions
    are driven by ``effective_Re_pix`` through ``residual_zones``.
    """

    galaxy_id: str
    galaxy_name: str
    scenario: str
    pixel_scale_arcsec: float
    sb_zeropoint: float
    psf_fwhm_arcsec: float | None
    psf_fwhm_pix: float | None
    image_shape: tuple[int, int]
    image_sigma_adu: float
    initial_x0: float
    initial_y0: float
    effective_Re_pix: float | None
    truth_components: tuple[dict[str, float], ...]
    re_outer_pix: float | None

    @property
    def half_extent(self) -> int:
        return min(self.image_shape) // 2

    @property
    def true_center(self) -> tuple[float, float]:
        """Mocks share a single center at (initial_x0, initial_y0)."""
        return (self.initial_x0, self.initial_y0)


def load_galaxy_manifest(galaxy_dir: Path) -> GalaxyManifest:
    """Read ``<galaxy_dir>/MANIFEST.json`` into a :class:`GalaxyManifest`."""
    with (Path(galaxy_dir) / "MANIFEST.json").open() as handle:
        data = json.load(handle)
    pix_scale = float(data["pixel_scale_arcsec"])
    extra = data.get("extra", {}) or {}
    psf_arcsec = extra.get("psf_fwhm_arcsec")
    psf_pix = float(psf_arcsec) / pix_scale if psf_arcsec else None
    truth = tuple(extra.get("truth_components", []) or [])
    re_pxs = [float(c["re_px"]) for c in truth if c.get("re_px") is not None]
    re_outer = max(re_pxs) if re_pxs else None
    shape = tuple(data["image_shape"])
    if len(shape) != 2:
        raise ValueError(f"{galaxy_dir}: MANIFEST image_shape must be 2D")
    return GalaxyManifest(
        galaxy_id=data["galaxy_id"],
        galaxy_name=extra.get("galaxy_name", ""),
        scenario=extra.get("scenario", ""),
        pixel_scale_arcsec=pix_scale,
        sb_zeropoint=float(data["sb_zeropoint"]),
        psf_fwhm_arcsec=float(psf_arcsec) if psf_arcsec else None,
        psf_fwhm_pix=psf_pix,
        image_shape=shape,  # type: ignore[arg-type]
        image_sigma_adu=float(data["image_sigma"]["image_sigma_adu"]),
        initial_x0=float(data["initial_geometry"]["x0"]),
        initial_y0=float(data["initial_geometry"]["y0"]),
        effective_Re_pix=(float(data["effective_Re_pix"]) if data.get("effective_Re_pix") is not None else None),
        truth_components=truth,
        re_outer_pix=re_outer,
    )


def load_arm_profile(galaxy_dir: Path, tool: str, arm_id: str) -> Table | None:
    """Load the ``ISOPHOTES`` HDU of one arm's ``profile.fits``."""
    path = Path(galaxy_dir) / tool / "arms" / arm_id / "profile.fits"
    if not path.is_file():
        return None
    with fits.open(path) as hdul:
        return Table(hdul[1].data)  # type: ignore[index]


def load_arm_record(galaxy_dir: Path, tool: str, arm_id: str) -> dict[str, Any] | None:
    path = Path(galaxy_dir) / tool / "arms" / arm_id / "run_record.json"
    if not path.is_file():
        return None
    with path.open() as handle:
        return json.load(handle)


def list_campaign_galaxies(campaign_dir: Path, dataset: str) -> list[Path]:
    """List ``<campaign_dir>/<dataset>/<safe_galaxy_id>/`` directories.

    Safe ids carry the double-underscore substitution of ``/`` (see
    ``adapters.base.safe_galaxy_id``). Returns a sorted list of Path.
    """
    ds_dir = Path(campaign_dir) / dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"dataset dir missing: {ds_dir}")
    return sorted(p for p in ds_dir.iterdir() if p.is_dir() and "__" in p.name)


def list_arms(galaxy_dir: Path, tool: str) -> list[str]:
    ap = Path(galaxy_dir) / tool / "arms"
    if not ap.is_dir():
        return []
    return sorted(p.name for p in ap.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmaZones:
    """Inner / mid / outer sma-masks plus the reference length used.

    Matches :mod:`residual_zones` exactly: ``inner`` is ``r_ell <
    0.5 * R_ref``, ``mid`` is ``[0.5, 2) * R_ref``, ``outer`` is
    ``>= 2 * R_ref``.  Because ``r_ell == sma`` on an isophote the
    same thresholds apply here.
    """

    R_ref_pix: float
    fallback: str
    inner: np.ndarray
    mid: np.ndarray
    outer: np.ndarray

    @property
    def inner_end(self) -> float:
        return 0.5 * self.R_ref_pix

    @property
    def mid_end(self) -> float:
        return 2.0 * self.R_ref_pix


def _reference_length(manifest: GalaxyManifest, sma: np.ndarray) -> tuple[float, str]:
    """Pick ``R_ref`` per residual_zones convention.

    Uses ``effective_Re_pix`` when finite and positive; otherwise falls
    back to ``max(1.0, 0.25 * maxsma)`` where ``maxsma`` is the largest
    observed sma in the profile (the per-arm top of the ladder).
    """
    re_px = manifest.effective_Re_pix
    if re_px is not None and np.isfinite(re_px) and re_px > 0:
        return float(re_px), ""
    maxsma = float(np.nanmax(sma)) if sma.size else float(manifest.half_extent)
    if not np.isfinite(maxsma) or maxsma <= 0:
        maxsma = float(manifest.half_extent)
    return max(1.0, 0.25 * maxsma), "Re_missing"


def sma_zones(
    profile: Table,
    manifest: GalaxyManifest,
    *,
    ok_mask: np.ndarray | None = None,
) -> SmaZones:
    """Build inner / mid / outer boolean masks aligned to the profile rows.

    Masks are AND'ed with ``ok_mask`` (defaulting to ``stop_code == 0``)
    so metrics iterate only over valid isophotes.
    """
    sma = np.asarray(profile["sma"], dtype=float)
    R_ref, fallback = _reference_length(manifest, sma)
    if ok_mask is None:
        ok = np.asarray(np.asarray(profile["stop_code"]) == 0, dtype=bool)
    else:
        ok = np.asarray(ok_mask, dtype=bool)
    inner = ok & (sma < 0.5 * R_ref)
    outer = ok & (sma >= 2.0 * R_ref)
    mid = ok & ~(sma < 0.5 * R_ref) & ~(sma >= 2.0 * R_ref)
    return SmaZones(
        R_ref_pix=R_ref,
        fallback=fallback,
        inner=inner,
        mid=mid,
        outer=outer,
    )


# ---------------------------------------------------------------------------
# Harmonic normalization + local-gradient helpers
# ---------------------------------------------------------------------------


def _col_or_nan(profile: Table, name: str) -> np.ndarray:
    """Return ``profile[name]`` as float64, or an all-NaN array if absent."""
    if name in profile.colnames:
        return np.asarray(profile[name], dtype=float)
    return np.full(len(profile), np.nan, dtype=float)


# ---------------------------------------------------------------------------
# Intensity-weighted centroid drift
# ---------------------------------------------------------------------------


def zone_iw_centroid_drift(
    x0: np.ndarray,
    y0: np.ndarray,
    intens: np.ndarray,
    mask: np.ndarray,
    true_x: float,
    true_y: float,
) -> float:
    """Intensity-weighted |(xbar, ybar) - (true_x, true_y)| within ``mask``."""
    if mask.sum() == 0:
        return float("nan")
    w = np.asarray(intens, dtype=float)[mask]
    xv = np.asarray(x0, dtype=float)[mask]
    yv = np.asarray(y0, dtype=float)[mask]
    good = np.isfinite(w) & (w > 0) & np.isfinite(xv) & np.isfinite(yv)
    if not good.any():
        return float("nan")
    w = w[good]
    xv = xv[good]
    yv = yv[good]
    wsum = float(np.sum(w))
    if wsum <= 0:
        return float("nan")
    xbar = float(np.sum(xv * w) / wsum)
    ybar = float(np.sum(yv * w) / wsum)
    return float(np.hypot(xbar - true_x, ybar - true_y))


# ---------------------------------------------------------------------------
# Local three-point residual (prior 3)
# ---------------------------------------------------------------------------


def _wrap_pa_diff(delta: np.ndarray) -> np.ndarray:
    """Wrap a PA difference onto ``(-pi/2, pi/2]`` (PA is mod pi)."""
    return (delta + np.pi / 2.0) % np.pi - np.pi / 2.0


def local_triplet_residual(
    x: np.ndarray,
    y: np.ndarray,
    *,
    circular_pi: bool = False,
) -> np.ndarray:
    """Deviation of ``y[i]`` from the linear interpolation of its neighbors.

    For ``i`` in ``1 .. n-2`` with monotone ``x``:
        ``t = (x[i] - x[i-1]) / (x[i+1] - x[i-1])``
        ``resid[i] = y[i] - (y[i-1] + t * (y[i+1] - y[i-1]))``

    The endpoint rows and any with ``x[i+1] == x[i-1]`` are NaN.
    When ``circular_pi`` is set, the residual is wrapped onto
    ``(-pi/2, pi/2]`` so a 90 deg flip produces a small residual only
    if both neighbors already sit across the branch cut.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    out = np.full(n, np.nan, dtype=float)
    if n < 3:
        return out
    for i in range(1, n - 1):
        dx = x[i + 1] - x[i - 1]
        if not np.isfinite(dx) or dx == 0:
            continue
        t = (x[i] - x[i - 1]) / dx
        if circular_pi:
            pa0 = y[i - 1]
            d1 = _wrap_pa_diff(np.array([y[i] - pa0]))[0]
            d2 = _wrap_pa_diff(np.array([y[i + 1] - pa0]))[0]
            resid = _wrap_pa_diff(np.array([d1 - t * d2]))[0]
        else:
            y_interp = y[i - 1] + t * (y[i + 1] - y[i - 1])
            resid = y[i] - y_interp
        out[i] = resid
    return out


# ---------------------------------------------------------------------------
# Prior-violation metrics
# ---------------------------------------------------------------------------


def _err_ratio(amp: np.ndarray, err: np.ndarray, mask: np.ndarray) -> float:
    a = amp[mask]
    e = err[mask]
    if a.size == 0:
        return float("nan")
    good = np.isfinite(a) & np.isfinite(e) & (e > 0)
    if not np.any(good):
        return float("nan")
    return float(np.median(np.abs(a[good]) / e[good]))


def _nanmax_abs(arr: np.ndarray, mask: np.ndarray) -> float:
    v = arr[mask]
    v = v[np.isfinite(v)]
    return float(np.max(np.abs(v))) if v.size else float("nan")


def compute_prior_metrics(
    profile: Table,
    manifest: GalaxyManifest,
) -> dict[str, Any]:
    """Per-arm F.0 prior-violation metrics (overhauled, three priors).

    Prior 1 — intensity-weighted centroid drift per zone (inner / mid /
              outer).
    Prior 2 — normalized harmonic ``|A_n_norm| / err(A_n_norm)`` in the
              outer zone.
    Prior 3 — 3-point local residual of ``eps`` and ``pa`` in the outer
              zone.

    All returned values are plain floats / ints (NaN-safe) so the
    result dict can be added directly to a long-form DataFrame row.
    """
    out: dict[str, Any] = {}
    if profile is None or len(profile) == 0:
        return {"n_iso_total": 0, "n_iso_ok": 0}

    sma = _col_or_nan(profile, "sma")
    ok = np.asarray(profile["stop_code"]) == 0
    x0 = _col_or_nan(profile, "x0")
    y0 = _col_or_nan(profile, "y0")
    eps = _col_or_nan(profile, "eps")
    pa = _col_or_nan(profile, "pa")
    intens = _col_or_nan(profile, "intens")

    zones = sma_zones(profile, manifest, ok_mask=ok)

    true_x, true_y = manifest.true_center
    drift_per_iso = np.hypot(x0 - true_x, y0 - true_y)

    # ---- Prior 1: intensity-weighted centroid drift per zone --------
    drift_inner = zone_iw_centroid_drift(x0, y0, intens, zones.inner, true_x, true_y)
    drift_mid = zone_iw_centroid_drift(x0, y0, intens, zones.mid, true_x, true_y)
    drift_outer = zone_iw_centroid_drift(x0, y0, intens, zones.outer, true_x, true_y)

    # ---- Prior 2: normalized harmonics in outer zone ----------------
    # isoster and photutils both divide by sma*|grad| at fit time, so their
    # stored values and errors are already Bender-normalized and are used
    # directly. Re-normalizing here divided them by sma*|grad| a second time
    # (review P1).
    #
    # UNVERIFIED -- AutoProf. This code path treats AutoProf's a3/b3/a4/b4 as
    # arriving on the same normalized scale, and the adapter copies its columns
    # through without conversion. That is an *assumption*, not a measurement:
    # nobody has compared AutoProf's reported coefficients against the isoster
    # and photutils values on a fixture with a planted deviation. (Earlier
    # documentation asserted the scale was different; that was equally
    # untested, and has been corrected to say unverified. There is no
    # contradictory measurement on either side -- there is no measurement.)
    #
    # Settling it means running AutoProf on a planted deviation and comparing,
    # which is a question about another project's output rather than something
    # this file can decide. Until then, treat Prior 2 scores for the *autoprof*
    # tool as unverified and keep them out of publication. The
    # isoster-vs-photutils comparison is unaffected.
    harm_fields = ("a3", "b3", "a4", "b4")
    harm_norm: dict[str, np.ndarray] = {}
    harm_norm_valid: dict[str, np.ndarray] = {}
    harm_norm_err: dict[str, np.ndarray] = {}
    for name in harm_fields:
        norm_val = _col_or_nan(profile, name)
        norm_err = _col_or_nan(profile, f"{name}_err")
        harm_norm[name] = norm_val
        harm_norm_valid[name] = np.isfinite(norm_val)
        harm_norm_err[name] = norm_err

    def _harm_stat(name: str) -> dict[str, float]:
        valid = harm_norm_valid[name] & zones.outer
        ratio = _err_ratio(np.abs(harm_norm[name]), harm_norm_err[name], valid)
        vals = harm_norm[name][valid]
        vals = vals[np.isfinite(vals)]
        abs_median = float(np.median(np.abs(vals))) if vals.size else float("nan")
        return {"err_ratio": ratio, "abs_median": abs_median, "n": int(valid.sum())}

    harm_stats = {name: _harm_stat(name) for name in harm_fields}

    # Prior 2 applicability: when both low-order harmonics (A3 and A4)
    # have zero valid outer-zone points, the arm did not fit harmonics
    # there (e.g. `harm_disabled`, or sweeps that replace the default
    # [3, 4] orders). Mark Prior 2 N/A so the arm is neither credited
    # with a clean pass nor flagged as violating.
    prior2_applicable = bool((harm_stats["a3"]["n"] > 0) or (harm_stats["a4"]["n"] > 0))

    # ---- Prior 3: 3-point local residual in outer zone --------------
    # Use only outer-zone rows so the triplet is built over contiguous
    # outer isophotes. The per-row residual is only defined where both
    # neighbors are also in the mask.
    outer_idx = np.where(zones.outer)[0]
    if outer_idx.size >= 3:
        s_outer = sma[outer_idx]
        eps_resid = local_triplet_residual(s_outer, eps[outer_idx])
        pa_resid = local_triplet_residual(s_outer, pa[outer_idx], circular_pi=True)
        max_eps_res = float(np.nanmax(np.abs(eps_resid))) if np.any(np.isfinite(eps_resid)) else float("nan")
        max_pa_res_rad = float(np.nanmax(np.abs(pa_resid))) if np.any(np.isfinite(pa_resid)) else float("nan")
        max_pa_res_deg = float(np.degrees(max_pa_res_rad)) if np.isfinite(max_pa_res_rad) else float("nan")
        n_triplets = int(np.sum(np.isfinite(eps_resid)))
    else:
        max_eps_res = float("nan")
        max_pa_res_deg = float("nan")
        n_triplets = 0

    out.update(
        {
            "n_iso_total": int(len(profile)),
            "n_iso_ok": int(ok.sum()),
            "R_ref_pix": zones.R_ref_pix,
            "zone_fallback_ref": zones.fallback,
            "sma_zone_inner_end": zones.inner_end,
            "sma_zone_mid_end": zones.mid_end,
            "n_iso_inner": int(zones.inner.sum()),
            "n_iso_mid": int(zones.mid.sum()),
            "n_iso_outer": int(zones.outer.sum()),
            # Prior 1 (primary) — intensity-weighted drift per zone
            "drift_iw_inner_pix": drift_inner,
            "drift_iw_mid_pix": drift_mid,
            "drift_iw_outer_pix": drift_outer,
            # Prior 1 diagnostics (not scored)
            "max_drift_pix_all_ok": _nanmax_abs(drift_per_iso, ok),
            "max_drift_pix_outer": _nanmax_abs(drift_per_iso, zones.outer),
            # Prior 2 — normalized harmonics, outer zone only
            "a3n_err_ratio_outer": harm_stats["a3"]["err_ratio"],
            "b3n_err_ratio_outer": harm_stats["b3"]["err_ratio"],
            "a4n_err_ratio_outer": harm_stats["a4"]["err_ratio"],
            "b4n_err_ratio_outer": harm_stats["b4"]["err_ratio"],
            "abs_a3n_median_outer": harm_stats["a3"]["abs_median"],
            "abs_b3n_median_outer": harm_stats["b3"]["abs_median"],
            "abs_a4n_median_outer": harm_stats["a4"]["abs_median"],
            "abs_b4n_median_outer": harm_stats["b4"]["abs_median"],
            "n_harm_outer": harm_stats["a4"]["n"],
            "prior2_applicable": prior2_applicable,
            # Prior 3 — 3-point local residual, outer zone only
            "max_local_resid_eps_outer": max_eps_res,
            "max_local_resid_pa_outer_deg": max_pa_res_deg,
            "n_triplets_outer": n_triplets,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Default violation thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    # Prior 1 — intensity-weighted centroid, per zone.
    "drift_iw_inner_pix": 0.5,
    "drift_iw_mid_pix": 0.5,
    "drift_iw_outer_pix": 0.5,  # calibrated on F.a: ref_default p95 hits ~1.15 px at wide_z050; 0.5 is a defensible "drift is visible" bar
    # Prior 2 — outer-zone normalized harmonics vs formal errors.
    # A violation requires BOTH the err-ratio threshold AND the absolute
    # |A_n_norm| floor to fire, so tiny |A4n| with optimistic formal
    # errors no longer triggers a false positive.
    "a3n_err_ratio_outer": 3.0,
    "b3n_err_ratio_outer": 3.0,
    "a4n_err_ratio_outer": 3.0,
    "b4n_err_ratio_outer": 3.0,
    "abs_a3n_median_outer": 0.01,
    "abs_b3n_median_outer": 0.01,
    "abs_a4n_median_outer": 0.01,
    "abs_b4n_median_outer": 0.01,
    # Prior 3 — 3-point local residual beyond 2 * R_ref.
    # Calibrated in Phase F.c.3 against the worst-scenario p95 of the
    # best-behaving arm (isoster reg_outer_damp): eps p95 = 0.019 at
    # wide_z050 (x 1.3 safety -> 0.025); pa p95 = 3.96 deg at wide_z050
    # (x 1.25 safety -> 5 deg). Tighter than the F.0 empirical placeholders
    # (0.05 eps / 8 deg) so the metric discriminates more sharply within
    # the middle of the arm pack while still giving the regularization
    # trio room above p95.
    "max_local_resid_eps_outer": 0.025,
    "max_local_resid_pa_outer_deg": 5.0,
}


def flag_prior2_regularization_induced(
    metrics: dict[str, Any],
    reference_metrics: dict[str, Any] | None,
    *,
    factor_ratio: float = 2.0,
    reference_floor: float = 0.02,
) -> bool | None:
    """Mark a Prior 2 violation as "regularization-induced" when the same
    galaxy's |A_n_norm| on the tool's reference arm is small AND the
    current arm's |A_n_norm| is at least ``factor_ratio`` times larger.

    Returns:
      - ``False`` when there is no Prior 2 violation (nothing to annotate).
      - ``None`` when Prior 2 is N/A for this arm OR no reference is
        available (cannot decide).
      - ``True`` when the violation is attributable to regularization:
        the data itself has low |A_n_norm| (reference is below
        ``reference_floor``) but this arm's value is ``factor_ratio`` x
        or more. The diagnostic is evaluated per harmonic order
        (a3n, b3n, a4n, b4n); it fires when *any* order shows the
        induced pattern.

    Calibration note: ``reference_floor = 0.02`` is set slightly above
    the 0.01 dual-criterion floor in ``DEFAULT_THRESHOLDS`` so a
    reference arm flagged as clean on the data still counts as
    "low |A_n_norm|". ``factor_ratio = 2.0`` matches the intuition
    that regularization injects a ~2-5x harmonic signal on mocks with
    no genuine harmonic content.
    """
    if metrics.get("prior2_applicable") is False:
        return None
    if reference_metrics is None:
        return None
    # Was there a Prior 2 violation at all?
    th = DEFAULT_THRESHOLDS
    violated = False
    for name in ("a3n", "b3n", "a4n", "b4n"):
        er = metrics.get(f"{name}_err_ratio_outer")
        am = metrics.get(f"abs_{name}_median_outer")
        try:
            er_f = float(er) if er is not None else float("nan")
            am_f = float(am) if am is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if (
            np.isfinite(er_f)
            and np.isfinite(am_f)
            and er_f > th[f"{name}_err_ratio_outer"]
            and am_f > th[f"abs_{name}_median_outer"]
        ):
            violated = True
            break
    if not violated:
        return False
    # Does the reference arm (default, same galaxy) have a small
    # |A_n_norm| that *this* arm's ratio blows past?
    for name in ("a3n", "b3n", "a4n", "b4n"):
        here = metrics.get(f"abs_{name}_median_outer")
        there = reference_metrics.get(f"abs_{name}_median_outer")
        try:
            here_f = float(here) if here is not None else float("nan")
            there_f = float(there) if there is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(here_f) and np.isfinite(there_f)):
            continue
        if there_f < reference_floor and here_f > factor_ratio * max(there_f, 1e-6):
            return True
    return False


def classify_violations(
    metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
    *,
    reference_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map metric values onto ``violates_<prior>`` flags.

    Returns booleans for Priors 1 and 3. ``violates_harmonics_outer``
    is ``None`` when Prior 2 does not apply (``prior2_applicable ==
    False``, e.g. an arm that fit no A3/A4 harmonics). Treating N/A
    as ``None`` lets downstream drivers skip that arm in the Prior 2
    clean-fraction numerator and denominator without crediting a
    free pass. Missing finite values (NaN) still count as
    non-violation.

    ``reference_metrics`` is the same-galaxy metrics dict from the
    tool's default arm (``DEFAULT_ARM_PER_TOOL``); when supplied,
    the returned dict also carries ``prior2_regularization_induced``
    per :func:`flag_prior2_regularization_induced` semantics. Callers
    that omit this kwarg receive ``None`` for the diagnostic (same
    N/A convention as ``violates_harmonics_outer``).
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    def _gt(key: str) -> bool:
        val = metrics.get(key)
        if val is None:
            return False
        try:
            v = float(val)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(v):
            return False
        return v > th[key]

    prior2_applicable = bool(metrics.get("prior2_applicable", True))
    harmonics_viol: bool | None
    if prior2_applicable:
        harmonics_viol = any(
            _gt(f"{name}_err_ratio_outer") and _gt(f"abs_{name}_median_outer") for name in ("a3n", "b3n", "a4n", "b4n")
        )
    else:
        harmonics_viol = None

    reg_induced = flag_prior2_regularization_induced(metrics, reference_metrics)

    return {
        "violates_drift_inner": _gt("drift_iw_inner_pix"),
        "violates_drift_mid": _gt("drift_iw_mid_pix"),
        "violates_drift_outer": _gt("drift_iw_outer_pix"),
        "violates_drift_any": (_gt("drift_iw_inner_pix") or _gt("drift_iw_mid_pix") or _gt("drift_iw_outer_pix")),
        "violates_harmonics_outer": harmonics_viol,
        "prior2_regularization_induced": reg_induced,
        "violates_geometry_outer": (_gt("max_local_resid_eps_outer") or _gt("max_local_resid_pa_outer_deg")),
    }
