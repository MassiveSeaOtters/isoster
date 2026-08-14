"""
Multi-band ellipse sampler for the joint isoster fitter.

Wraps :func:`scipy.ndimage.map_coordinates` so a single ellipse path is
computed once per geometry and then sampled across all bands in one
vectorized call. The shared-validity rule is enforced: a sample index is
kept only if every band passes mask, NaN, and (when WLS) variance
checks. This guarantees that the joint design matrix has the same
sample count `N` per band, which the joint solver requires.

See ``docs/agent/plan-2026-04-29-multiband-feasibility.md`` decisions
D6/D7/D9 for the validity rule and D19 for the vectorization choice.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from .._shared import _prepare_mask_float
from ..numba_kernels import compute_ellipse_coords
from .numba_kernels_mb import bilinear_sample_stack


# Multi-band counterpart of ``isoster.sampling.IsophoteData``.
#
# Two layouts coexist on this struct:
#
# - Shared-validity layout (default): a single 2-D ``(B, N_valid)``
#   ``intens`` array and matching ``variances``, plus shared 1-D
#   ``angles`` / ``phi`` / ``radii`` of length ``N_valid``. Every band
#   contributes to the same N samples by construction, so the joint
#   design matrix is rectangular and the numba kernel applies.
#   ``intens_per_band`` / ``phi_per_band`` are ``None`` and
#   ``n_valid_per_band[b] == valid_count`` for every ``b``.
#
# - Loose-validity layout (D9 backport, ``loose_validity=True``):
#   per-band arrays differ in length. ``intens_per_band`` /
#   ``variances_per_band`` / ``phi_per_band`` are jagged lists of
#   length ``B``. The legacy 2-D ``intens`` / ``variances`` /
#   ``angles`` / ``phi`` fields are filled with the *intersection*
#   subset (samples kept by every band) so callers that only need
#   shared-grid statistics still get a usable view; callers that want
#   the full per-band kept set must use the ``_per_band`` lists.
#   ``n_valid_per_band[b]`` reports each band's actual surviving count.
@dataclass(slots=True)
class MultiIsophoteData:
    """Container for one ellipse's multi-band sampled intensities.

    Slotted dataclass instead of a namedtuple — the per-isophote
    construction cost is in the hot path (called once per fit
    iteration, B=5 / ~20 iterations / 73 isophotes ≈ 1500 times per
    fit) and the slotted layout shaves ~40% off the construction
    overhead vs a namedtuple with 11 fields by avoiding the dict-based
    attribute storage and the tuple's per-field copy. Field order and
    names match the previous namedtuple so existing accessors are
    unchanged.
    """

    # Always populated.
    angles: np.ndarray  # shape (N_intersect,) under loose; (N_valid,) shared
    phi: np.ndarray  # shape (N_intersect,) / (N_valid,) — see angles
    intens: np.ndarray  # shape (B, N_intersect/N_valid) — intersection view
    radii: np.ndarray  # shape matching angles
    variances: Optional[np.ndarray]  # same shape as intens, or None
    n_samples: int  # total samples on the ellipse path before filtering
    valid_count: int  # len(angles); intersection count under loose
    n_valid_per_band: np.ndarray  # shape (B,); per-band surviving counts

    # Loose-validity-only fields. ``None`` under shared validity so
    # callers can branch on ``intens_per_band is None`` to detect the
    # mode.
    intens_per_band: Optional[List[np.ndarray]] = field(default=None)
    phi_per_band: Optional[List[np.ndarray]] = field(default=None)
    variances_per_band: Optional[List[np.ndarray]] = field(default=None)


def _stack_images(images: Sequence[NDArray[np.floating]]) -> NDArray[np.float64]:
    """Validate and stack a list of B same-shape images into ``(B, H, W)`` float64."""
    if len(images) == 0:
        raise ValueError("images list cannot be empty")
    h, w = images[0].shape
    for i, im in enumerate(images):
        if im.shape != (h, w):
            raise ValueError(f"images[{i}] has shape {im.shape}, expected {(h, w)} (must match images[0])")
    # Cast to float64 so map_coordinates does not silently truncate.
    return np.stack([np.ascontiguousarray(im, dtype=np.float64) for im in images], axis=0)


def _resolve_masks(
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    n_bands: int,
    h: int,
    w: int,
) -> List[Optional[NDArray[np.float64]]]:
    """
    Normalize masks input into a list of length B.

    Accepts: ``None`` (no masks), a single ``(H, W)`` boolean ndarray
    (broadcast to all bands), or a list of length ``B`` of ndarrays or
    Nones (per-band, where None means "no bad pixels in that band").

    Each returned mask is converted to ``float64`` (1.0 = bad, 0.0 = good)
    so it can be sampled by map_coordinates with ``order=0`` and the
    cval=1.0 (off-image = bad) convention. ``None`` entries remain
    ``None`` so the sampler can skip the call.
    """
    if masks is None:
        return [None] * n_bands

    if isinstance(masks, np.ndarray):
        if masks.shape != (h, w):
            raise ValueError(f"masks ndarray shape {masks.shape} does not match images shape {(h, w)}")
        return [_prepare_mask_float(masks)] * n_bands

    if not hasattr(masks, "__len__") or len(masks) != n_bands:
        raise ValueError(
            f"masks list length must equal n_bands ({n_bands}); got "
            f"{len(masks) if hasattr(masks, '__len__') else type(masks).__name__}"
        )

    out: List[Optional[NDArray[np.float64]]] = []
    for i, m in enumerate(masks):
        if m is None:
            out.append(None)
            continue
        if not isinstance(m, np.ndarray):
            raise TypeError(f"masks[{i}] must be a numpy ndarray or None, got {type(m).__name__}")
        if m.shape != (h, w):
            raise ValueError(f"masks[{i}] shape {m.shape} does not match images shape {(h, w)}")
        out.append(_prepare_mask_float(m))
    return out


def _mark_invalid_variance(v: NDArray[np.floating], label: str) -> NDArray[np.float64]:
    """Mark unusable variance entries NaN so their samples are dropped.

    An entry that is not finite, or not strictly positive, carries no usable
    information about the noise. Marking it NaN makes bilinear interpolation
    propagate the flag to every sample that touches it, and the sampler's
    validity rule then drops those samples, so a ring statistic and its
    uncertainty always describe one identical sample set.

    This retires the earlier sentinel scheme (non-finite → ``1e30``,
    non-positive → ``1e-30``). Both substitutions left the flagged pixel in
    the ring while distorting its weight. ``1e30`` is finite and strictly
    positive, so the validity rule kept it despite the comment claiming
    otherwise; ``1e-30`` gave a single pixel near-infinite weight and
    collapsed every inverse-variance-weighted error by fourteen orders of
    magnitude. Mirrors :func:`isoster.driver.fit_image`; see
    ``docs/04-architecture.md``, section "Invalid-variance policy".
    """
    arr = np.asarray(v, dtype=np.float64)
    non_finite = ~np.isfinite(arr)
    non_positive = ~non_finite & (arr <= 0.0)
    n_non_finite = int(non_finite.sum())
    n_non_positive = int(non_positive.sum())
    if not (n_non_finite or n_non_positive):
        return arr
    out = arr.copy()
    if n_non_finite:
        warnings.warn(
            f"{label}: contains {n_non_finite} non-finite value(s) (NaN or infinite); "
            f"the affected samples are excluded from both the ring statistics and "
            f"their uncertainties.",
            RuntimeWarning,
            stacklevel=3,
        )
    if n_non_positive:
        warnings.warn(
            f"{label}: contains {n_non_positive} non-positive value(s); a variance of "
            f"zero or below is not physical, so the affected samples are excluded from "
            f"both the ring statistics and their uncertainties. Mask these pixels "
            f"explicitly if the exclusion is intended.",
            RuntimeWarning,
            stacklevel=3,
        )
    out[non_finite | non_positive] = np.nan
    return out


def _resolve_variance_maps(
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]],
    n_bands: int,
    h: int,
    w: int,
) -> Optional[List[NDArray[np.float64]]]:
    """
    Normalize variance_maps input into a list of length B (or None).

    All-or-nothing per decision D7: either every band has a map (full
    WLS) or none do (full OLS). ``None`` values inside a list are
    rejected even if the list has the right length.

    Entries that are non-finite or non-positive are marked NaN; a single
    warning is emitted per band and category that needed marking. Because
    every entry point that accepts a raw map routes through this function,
    the stacks handed to the sampler are always pre-marked, and NaN
    propagates through interpolation on its own.

    Returns ``None`` when the user passed ``None`` (OLS mode signal).
    Otherwise returns a list of B ``(H, W)`` float64 arrays.
    """
    if variance_maps is None:
        return None

    if isinstance(variance_maps, np.ndarray):
        if variance_maps.shape != (h, w):
            raise ValueError(f"variance_maps ndarray shape {variance_maps.shape} does not match images shape {(h, w)}")
        v_f = _mark_invalid_variance(variance_maps, "variance_maps (broadcast)")
        return [v_f] * n_bands

    if len(variance_maps) != n_bands:
        raise ValueError(f"variance_maps list length must equal n_bands ({n_bands}); got {len(variance_maps)}")

    out: List[NDArray[np.float64]] = []
    for i, v in enumerate(variance_maps):
        if v is None:
            raise ValueError(
                f"variance_maps[{i}] is None: variance maps are all-or-nothing. "
                "Pass variance_maps=None for full OLS, or provide a map for every "
                "band."
            )
        if not isinstance(v, np.ndarray):
            raise TypeError(f"variance_maps[{i}] must be a numpy ndarray, got {type(v).__name__}")
        if v.shape != (h, w):
            raise ValueError(f"variance_maps[{i}] shape {v.shape} does not match images shape {(h, w)}")
        out.append(_mark_invalid_variance(v, f"variance_maps[{i}]"))
    return out


def _sample_along_ellipse(
    image_stack: NDArray[np.float64],
    masks_resolved: List[Optional[NDArray[np.float64]]],
    var_stack: Optional[NDArray[np.float64]],
    x0: float,
    y0: float,
    sma: float,
    eps: float,
    pa: float,
    use_eccentric_anomaly: bool,
    *,
    loose_validity: bool = False,
) -> MultiIsophoteData:
    """Core multi-band sampling kernel.

    Two modes coexist (decision D9 + D9 backport):

    - ``loose_validity=False`` (default): a sample is dropped from
      every band if any band fails (mask, NaN, non-positive variance).
      The legacy 2-D ``intens`` and ``variances`` arrays carry the
      shared kept set; the per-band fields fall back to slices of
      that array so callers that always read the per-band lists work
      uniformly.

    - ``loose_validity=True`` (D9 backport): each band keeps its own
      surviving samples. ``intens_per_band`` / ``variances_per_band`` /
      ``phi_per_band`` are jagged length-``B`` lists. The legacy 2-D
      ``intens`` / ``variances`` / ``angles`` / ``phi`` fields hold
      the *intersection* (samples kept by every band) for callers that
      need a shared-grid view; ``valid_count`` is that intersection
      count, while ``n_valid_per_band[b]`` reports each band's own
      surviving count.
    """
    n_bands = image_stack.shape[0]
    n_samples = max(64, int(2.0 * np.pi * sma))

    x_coords, y_coords, psi, phi = compute_ellipse_coords(n_samples, sma, eps, pa, x0, y0, use_eccentric_anomaly)
    coords = np.vstack([y_coords, x_coords]).astype(np.float64, copy=False)
    angle_full = psi if use_eccentric_anomaly else phi

    # Per-band image (and optional variance) sampling: one numba-JIT
    # call per stack instead of B per-band scipy calls. Cuts the
    # fixed-cost overhead (Python entry + dtype validation + output
    # alloc) which dominates over the actual interpolation work for
    # small N. Falls back to per-band scipy when numba is unavailable.
    intens_full = np.empty((n_bands, n_samples), dtype=np.float64)
    bilinear_sample_stack(image_stack, y_coords, x_coords, intens_full)

    var_full: Optional[NDArray[np.float64]] = None
    if var_stack is not None:
        var_full = np.empty((n_bands, n_samples), dtype=np.float64)
        bilinear_sample_stack(var_stack, y_coords, x_coords, var_full)

    # When the user passed a single mask via the broadcast convenience
    # path, ``_resolve_masks`` populates every entry of
    # ``masks_resolved`` with the *same* ndarray.  Calling
    # ``map_coordinates`` once and reusing the sampled values across
    # all bands trims B-1 expensive scipy calls per sampler
    # invocation; on B=5 with ~75 isophotes that recovers ~25% of the
    # multi-band sampler runtime.
    mask_cache: dict[int, NDArray[np.float64]] = {}

    def _sample_mask(m_b: NDArray[np.float64]) -> NDArray[np.float64]:
        key = id(m_b)
        cached = mask_cache.get(key)
        if cached is None:
            cached = map_coordinates(m_b, coords, order=0, mode="constant", cval=1.0)
            mask_cache[key] = cached
        return cached

    if not loose_validity:
        # Shared-validity fast path: a single ``(N,)`` valid mask
        # accumulated band-by-band.  Avoids allocating a ``(B, N)``
        # per-band-valid matrix; matches the Stage-1 baseline performance.
        valid = np.ones(n_samples, dtype=bool)
        for b in range(n_bands):
            valid &= ~np.isnan(intens_full[b])
            m_b = masks_resolved[b]
            if m_b is not None:
                valid &= _sample_mask(m_b) < 0.5
        if var_full is not None:
            for b in range(n_bands):
                valid &= np.isfinite(var_full[b]) & (var_full[b] > 0.0)
        n_valid_intersect = int(np.sum(valid))
        intens_intersect = intens_full[:, valid]
        var_intersect: Optional[NDArray[np.float64]] = var_full[:, valid] if var_full is not None else None
        angles_intersect = angle_full[valid]
        phi_intersect = phi[valid]
        n_valid_per_band = np.full(n_bands, n_valid_intersect, dtype=np.int64)
        return MultiIsophoteData(
            angles=angles_intersect,
            phi=phi_intersect,
            intens=intens_intersect,
            radii=np.full(n_valid_intersect, sma, dtype=np.float64),
            variances=var_intersect,
            n_samples=n_samples,
            valid_count=n_valid_intersect,
            intens_per_band=None,
            phi_per_band=None,
            variances_per_band=None,
            n_valid_per_band=n_valid_per_band,
        )

    # Loose-validity layout: per-band validity masks so the fitter can
    # keep each band's own surviving samples instead of the AND.
    per_band_valid = np.ones((n_bands, n_samples), dtype=bool)
    for b in range(n_bands):
        per_band_valid[b] &= ~np.isnan(intens_full[b])
        m_b = masks_resolved[b]
        if m_b is not None:
            per_band_valid[b] &= _sample_mask(m_b) < 0.5
        if var_full is not None:
            per_band_valid[b] &= np.isfinite(var_full[b]) & (var_full[b] > 0.0)

    valid_intersect = np.all(per_band_valid, axis=0)
    n_valid_intersect = int(np.sum(valid_intersect))
    n_valid_per_band = per_band_valid.sum(axis=1).astype(np.int64)
    intens_intersect = intens_full[:, valid_intersect]
    var_intersect = var_full[:, valid_intersect] if var_full is not None else None
    angles_intersect = angle_full[valid_intersect]
    phi_intersect = phi[valid_intersect]

    intens_per_band: List[NDArray[np.float64]] = [intens_full[b, per_band_valid[b]] for b in range(n_bands)]
    phi_per_band: List[NDArray[np.float64]] = [angle_full[per_band_valid[b]] for b in range(n_bands)]
    variances_per_band: Optional[List[NDArray[np.float64]]] = (
        [var_full[b, per_band_valid[b]] for b in range(n_bands)] if var_full is not None else None
    )

    return MultiIsophoteData(
        angles=angles_intersect,
        phi=phi_intersect,
        intens=intens_intersect,
        radii=np.full(n_valid_intersect, sma, dtype=np.float64),
        variances=var_intersect,
        n_samples=n_samples,
        valid_count=n_valid_intersect,
        intens_per_band=intens_per_band,
        phi_per_band=phi_per_band,
        variances_per_band=variances_per_band,
        n_valid_per_band=n_valid_per_band,
    )


def extract_isophote_data_multi_prepared(
    image_stack: NDArray[np.float64],
    masks_resolved: List[Optional[NDArray[np.float64]]],
    var_stack: Optional[NDArray[np.float64]],
    x0: float,
    y0: float,
    sma: float,
    eps: float,
    pa: float,
    use_eccentric_anomaly: bool = False,
    *,
    loose_validity: bool = False,
) -> MultiIsophoteData:
    """
    Fast-path multi-band sampler for callers that pre-resolved the inputs.

    The driver layer hits this function once per isophote-fit iteration
    (and twice per gradient call), so the per-call cost dominates the
    total fit time. Pre-resolving the (B, H, W) image stack, the per-
    band float64 mask list, and the (B, H, W) variance stack once at
    the driver level avoids repeated ``np.stack`` and ``astype`` calls
    in every iteration. Decision D19 (performance budget).

    Parameters mirror :func:`extract_isophote_data_multi` except that
    the inputs are already in the canonical layout the sampler uses
    internally:

    - ``image_stack``: ``(B, H, W)`` float64.
    - ``masks_resolved``: list of length ``B``, each entry either a
      ``(H, W)`` float64 array (1.0 = bad, 0.0 = good) or ``None``.
    - ``var_stack``: ``(B, H, W)`` float64 or ``None``.
    - ``loose_validity``: D9 backport; when True the sampler also
      returns per-band jagged arrays so the fitter can keep each
      band's own surviving samples instead of the cross-band AND.
    """
    return _sample_along_ellipse(
        image_stack,
        masks_resolved,
        var_stack,
        x0,
        y0,
        sma,
        eps,
        pa,
        use_eccentric_anomaly,
        loose_validity=loose_validity,
    )


def prepare_inputs(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]],
) -> Tuple[
    NDArray[np.float64],
    List[Optional[NDArray[np.float64]]],
    Optional[NDArray[np.float64]],
]:
    """
    One-shot input resolver for the driver/fitting hot path.

    Returns ``(image_stack, masks_resolved, var_stack)`` ready for
    repeated use by :func:`extract_isophote_data_multi_prepared`. The
    expensive astype/stack operations happen here exactly once.
    """
    image_stack = _stack_images(images)
    n_bands, h, w = image_stack.shape
    masks_resolved = _resolve_masks(masks, n_bands, h, w)
    var_list = _resolve_variance_maps(variance_maps, n_bands, h, w)
    var_stack: Optional[NDArray[np.float64]] = None
    if var_list is not None:
        var_stack = np.stack(var_list, axis=0)
    return image_stack, masks_resolved, var_stack


def extract_isophote_data_multi(
    images: Sequence[NDArray[np.floating]],
    masks: Union[None, NDArray[np.bool_], Sequence[Optional[NDArray[np.bool_]]]],
    x0: float,
    y0: float,
    sma: float,
    eps: float,
    pa: float,
    use_eccentric_anomaly: bool = False,
    variance_maps: Union[None, NDArray[np.floating], Sequence[NDArray[np.floating]]] = None,
    *,
    loose_validity: bool = False,
) -> MultiIsophoteData:
    """
    Extract per-band intensities along a shared elliptical path.

    Parameters
    ----------
    images : sequence of ndarray
        ``B`` aligned same-shape ``(H, W)`` images. The 0th sequence
        index is the band index used by the joint design matrix.
    masks : None | ndarray | sequence of (ndarray|None)
        Bad-pixel mask. ``None`` = no masking on any band. A single
        ``(H, W)`` boolean ndarray broadcasts to all bands. A sequence
        of length ``B`` allows per-band masks; ``None`` per band means
        "no bad pixels in that band".
    x0, y0 : float
        Ellipse center coordinates.
    sma : float
        Semi-major axis length.
    eps : float
        Ellipticity (1 - b/a).
    pa : float
        Position angle in radians.
    use_eccentric_anomaly : bool, default False
        Sample uniformly in eccentric anomaly (Ciambur 2015) when True.
    variance_maps : None | ndarray | sequence of ndarray
        Per-pixel variance. All-or-nothing per band.

    Returns
    -------
    MultiIsophoteData
        Slotted dataclass with shared ``angles, phi, radii`` and per-band
        ``intens, variances`` (each shape ``(B, N_valid)``). ``variances``
        is ``None`` when ``variance_maps`` is ``None``.

    Notes
    -----
    Shared-validity rule (decision D9): a sample is kept iff *every*
    band's mask, NaN, and (in WLS mode) finite-variance checks pass at
    that location. Off-image samples (``map_coordinates`` returning
    ``cval=NaN``) are dropped from all bands.
    """
    image_stack = _stack_images(images)  # (B, H, W) float64 by construction
    n_bands, h, w = image_stack.shape

    masks_list = _resolve_masks(masks, n_bands, h, w)
    var_list = _resolve_variance_maps(variance_maps, n_bands, h, w)
    var_stack: Optional[NDArray[np.float64]] = np.stack(var_list, axis=0) if var_list is not None else None
    return _sample_along_ellipse(
        image_stack,
        masks_list,
        var_stack,
        x0,
        y0,
        sma,
        eps,
        pa,
        use_eccentric_anomaly,
        loose_validity=loose_validity,
    )
