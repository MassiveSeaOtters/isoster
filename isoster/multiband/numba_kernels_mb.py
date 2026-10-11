"""
Numba-accelerated kernels for multi-band joint design matrix construction.

This module provides JIT-compiled builders for the
``(B·N × (B + 4))``-shaped joint design matrix used by the multi-band
fitter, where ``B`` is the number of bands and ``N`` is the per-band
sample count. Rows are stacked band-by-band; the row corresponding to
band ``b`` and angle ``φᵢ`` has the structure

    row[b·N + i]  =  [δ_{0b}, δ_{1b}, …, δ_{(B-1)b},
                      sin(φᵢ), cos(φᵢ), sin(2φᵢ), cos(2φᵢ)]

where the per-band indicator block (``B`` columns) parameterizes
per-band nuisance background terms ``I0_b``, and the shared trailing
4-column block parameterizes the geometric harmonic coefficients
``(A1, B1, A2, B2)``.

The numba kernel writes the indicator block efficiently as a
band-diagonal pattern; the harmonic block is identical to the
single-band design matrix per row, repeated across bands.

Includes a NumPy fallback when numba is unavailable. Same pattern as
``isoster.numba_kernels``: a ``@njit`` decorator that becomes a no-op
when ``NUMBA_AVAILABLE = False`` or ``NUMBA_DISABLE_JIT=1``.
"""

import os

import numpy as np
from numpy.typing import NDArray

# Mirror the single-band kernel's numba availability detection.
try:
    from numba import njit  # type: ignore[import-not-found]

    NUMBA_AVAILABLE = os.environ.get("NUMBA_DISABLE_JIT", "0") != "1"
except ImportError:
    NUMBA_AVAILABLE = False

if not NUMBA_AVAILABLE:

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _build_joint_design_matrix_numba(phi, n_bands):
    """
    Build the (B·N × (B + 4)) joint design matrix (numba-accelerated).

    For each kept sample index `i` and band index `b`:
        row = b * N + i
        col 0..B-1 : indicator (1.0 if col == b, else 0.0)
        col B+0    : sin(φᵢ)
        col B+1    : cos(φᵢ)
        col B+2    : sin(2 φᵢ)
        col B+3    : cos(2 φᵢ)

    The right-hand side is constructed by the caller as the
    band-stacked intensity vector ``ravel`` over the same row order.

    Parameters
    ----------
    phi : ndarray
        Shape (N,). Angle in radians. The same shared ellipse path is
        used for every band, so phi has only one axis.
    n_bands : int
        Number of bands `B`.

    Returns
    -------
    ndarray
        Shape (B·N, B + 4). The constant-1 column from the single-band
        design matrix is replaced by `B` per-band indicator columns
        ``[I_0, I_1, …, I_{B-1}]``; the shared geometric block
        ``[sin φ, cos φ, sin 2φ, cos 2φ]`` sits in the trailing four
        columns ``[A1, B1, A2, B2]``.
    """
    n_samples = phi.shape[0]
    n_rows = n_bands * n_samples
    n_cols = n_bands + 4
    A = np.zeros((n_rows, n_cols), dtype=np.float64)

    for b in range(n_bands):
        for i in range(n_samples):
            row = b * n_samples + i
            A[row, b] = 1.0
            p = phi[i]
            A[row, n_bands + 0] = np.sin(p)
            A[row, n_bands + 1] = np.cos(p)
            A[row, n_bands + 2] = np.sin(2.0 * p)
            A[row, n_bands + 3] = np.cos(2.0 * p)

    return A


def _build_joint_design_matrix_numpy(phi, n_bands):
    """
    Build the (B*N × (B + 4)) joint design matrix (pure numpy).

    Vectorized: builds the per-band indicator block via ``np.eye``-style
    repeats and the geometric block via broadcasting.
    """
    n_samples = phi.shape[0]
    n_cols = n_bands + 4
    A = np.zeros((n_bands * n_samples, n_cols), dtype=np.float64)

    # Indicator block: a band-diagonal pattern of 1.0s. Row b*N + i
    # has a 1 at column b. Build by tiling np.eye(n_bands) up to n_samples.
    band_idx = np.repeat(np.arange(n_bands), n_samples)  # shape (B*N,)
    row_idx = np.arange(n_bands * n_samples)
    A[row_idx, band_idx] = 1.0

    # Geometric block: the same 4-column tile per band, stacked B times.
    sin1 = np.sin(phi)
    cos1 = np.cos(phi)
    sin2 = np.sin(2.0 * phi)
    cos2 = np.cos(2.0 * phi)
    geom = np.column_stack([sin1, cos1, sin2, cos2])  # (N, 4)
    A[:, n_bands:] = np.tile(geom, (n_bands, 1))

    return A


_build_joint_design_matrix_impl = (
    _build_joint_design_matrix_numba if NUMBA_AVAILABLE else _build_joint_design_matrix_numpy
)


def build_joint_design_matrix(phi: NDArray[np.floating], n_bands: int) -> NDArray[np.floating]:
    """
    Build the joint design matrix for multi-band harmonic fitting.

    Parameters
    ----------
    phi : ndarray
        Shape (N,). Angles in radians, shared across all bands.
    n_bands : int
        Number of bands `B`. Must be >= 1.

    Returns
    -------
    ndarray
        Shape (B * N, B + 4). Column order:
        ``[I_0, I_1, ..., I_{B-1}, A1, B1, A2, B2]`` where the leading
        `B` columns are per-band indicators (one row of 1.0 per band) and
        the trailing 4 columns are the shared geometric harmonic terms.

        The B=1 case reduces to a 5-column matrix
        ``[1, sin φ, cos φ, sin 2φ, cos 2φ]`` numerically identical to
        the single-band design matrix.

    Raises
    ------
    ValueError
        If ``phi`` is empty or ``n_bands < 1``.
    """
    if phi.size == 0:
        raise ValueError("phi array cannot be empty")
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    return _build_joint_design_matrix_impl(phi, n_bands)


@njit(cache=True)
def _bilinear_sample_stack_numba(
    stack: NDArray[np.float64],
    y_coords: NDArray[np.float64],
    x_coords: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """Bilinear-interpolate ``stack[b, y, x]`` for every band b at every (y,x).

    Mirrors ``scipy.ndimage.map_coordinates(stack[b], coords, order=1,
    mode='constant', cval=np.nan)`` per band, in a single Python call
    that loops over (band, sample) in the JITted kernel. For B=5 with
    ~600 samples per ring, this trims the per-call scipy overhead
    (Python function entry, output allocation, dtype checking) — those
    fixed costs dominate over the interpolation work itself.
    Out-of-bounds samples are written as ``NaN`` so the downstream
    validity rule's ``~np.isnan`` filter still drops them.

    Parameters
    ----------
    stack : (B, H, W) float64
        Per-band 2-D images stacked along the leading axis.
    y_coords, x_coords : (N,) float64
        Shared sample coordinates (the ellipse path).
    out : (B, N) float64
        Pre-allocated output. Filled in place.
    """
    n_bands, h, w = stack.shape
    n_samples = y_coords.shape[0]
    h_max = h - 1
    w_max = w - 1
    for n in range(n_samples):
        y = y_coords[n]
        x = x_coords[n]
        if y < 0.0 or y > h_max or x < 0.0 or x > w_max:
            for b in range(n_bands):
                out[b, n] = np.nan
            continue
        y0 = int(np.floor(y))
        x0 = int(np.floor(x))
        y1 = y0 + 1 if y0 < h_max else y0
        x1 = x0 + 1 if x0 < w_max else x0
        wy = y - y0
        wx = x - x0
        c00 = (1.0 - wy) * (1.0 - wx)
        c01 = (1.0 - wy) * wx
        c10 = wy * (1.0 - wx)
        c11 = wy * wx
        for b in range(n_bands):
            out[b, n] = (
                c00 * stack[b, y0, x0] + c01 * stack[b, y0, x1] + c10 * stack[b, y1, x0] + c11 * stack[b, y1, x1]
            )


def _bilinear_sample_stack_numpy(
    stack: NDArray[np.float64],
    y_coords: NDArray[np.float64],
    x_coords: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """Pure-NumPy fallback for the per-band sampler kernel.

    Uses the existing ``scipy.ndimage.map_coordinates`` per band, since
    the optimization is the JIT loop fusion (multiple per-call entries
    collapsed into one). Without numba the per-call overhead returns,
    so we fall back to the per-band scipy loop the caller would have
    used anyway.
    """
    from scipy.ndimage import map_coordinates  # local import — fallback only

    n_bands = stack.shape[0]
    coords = np.vstack([y_coords, x_coords]).astype(np.float64, copy=False)
    for b in range(n_bands):
        out[b] = map_coordinates(
            stack[b],
            coords,
            order=1,
            mode="constant",
            cval=np.nan,
        )


bilinear_sample_stack = _bilinear_sample_stack_numba if NUMBA_AVAILABLE else _bilinear_sample_stack_numpy


@njit(cache=True)
def _mirror_index(index: int, last: int) -> int:
    """Fold a spline-footprint index back into ``[0, last]`` the way scipy does.

    ``map_coordinates(mode="constant")`` returns ``cval`` only for sample points
    outside the image; for a point inside it, footprint taps that fall past the
    edge are mirrored about the edge pixel (``-1 -> 1``, ``last + 1 -> last - 1``).
    Reproducing that rule is what lets this kernel match the single-band scipy
    read to rounding; clamping or repeating the edge pixel differs by up to the
    pixel-to-pixel contrast one pixel in from the border.
    """
    if index < 0:
        index = -index
    elif index > last:
        index = 2 * last - index
    if index < 0:
        index = 0
    elif index > last:
        index = last
    return index


@njit(cache=True)
def _cubic_weights(t: float):
    """The four cubic B-spline weights for a fractional offset ``t`` in [0, 1)."""
    u = 1.0 - t
    return (
        u * u * u / 6.0,
        (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0,
        (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
        t * t * t / 6.0,
    )


@njit(cache=True)
def _cubic_sample_stack_numba(
    coeff_stack: NDArray[np.float64],
    raw_stack: NDArray[np.float64],
    y_coords: NDArray[np.float64],
    x_coords: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """Cubic B-spline read of every band at every (y, x), matching scipy.

    ``coeff_stack`` holds the per-band spline coefficients from
    :func:`isoster.multiband.sampling_mb.prepare_sampling_stack`; the read
    reproduces ``map_coordinates(coeffs, order=3, mode="constant", cval=nan,
    prefilter=False)`` band by band, including scipy's mirrored footprint at
    the image edge, to rounding (about 1e-14 relative). ``raw_stack`` is the
    unfiltered image stack: a sample whose bilinear footprint touches a
    non-finite pixel is returned as NaN, exactly as the bilinear kernel
    returns it, because the coefficients were built with those pixels zeroed
    and would otherwise read a hole as flux.
    """
    n_bands, h, w = coeff_stack.shape
    n_samples = y_coords.shape[0]
    h_max = h - 1
    w_max = w - 1
    for n in range(n_samples):
        y = y_coords[n]
        x = x_coords[n]
        if y < 0.0 or y > h_max or x < 0.0 or x > w_max:
            for b in range(n_bands):
                out[b, n] = np.nan
            continue
        y0 = int(np.floor(y))
        x0 = int(np.floor(x))
        y1 = y0 + 1 if y0 < h_max else y0
        x1 = x0 + 1 if x0 < w_max else x0
        wy = _cubic_weights(y - y0)
        wx = _cubic_weights(x - x0)
        rows = (
            _mirror_index(y0 - 1, h_max),
            _mirror_index(y0, h_max),
            _mirror_index(y0 + 1, h_max),
            _mirror_index(y0 + 2, h_max),
        )
        cols = (
            _mirror_index(x0 - 1, w_max),
            _mirror_index(x0, w_max),
            _mirror_index(x0 + 1, w_max),
            _mirror_index(x0 + 2, w_max),
        )
        for b in range(n_bands):
            if (
                np.isnan(raw_stack[b, y0, x0])
                or np.isnan(raw_stack[b, y0, x1])
                or np.isnan(raw_stack[b, y1, x0])
                or np.isnan(raw_stack[b, y1, x1])
            ):
                out[b, n] = np.nan
                continue
            acc = 0.0
            for j in range(4):
                row = rows[j]
                row_acc = 0.0
                for i in range(4):
                    row_acc += wx[i] * coeff_stack[b, row, cols[i]]
                acc += wy[j] * row_acc
            out[b, n] = acc


def _cubic_sample_stack_numpy(
    coeff_stack: NDArray[np.float64],
    raw_stack: NDArray[np.float64],
    y_coords: NDArray[np.float64],
    x_coords: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    """Pure-scipy fallback for the cubic stack read (one call per band).

    The non-finite rule is applied through a bilinear read of the raw stack,
    which is NaN wherever the bilinear kernel would have been.
    """
    from scipy.ndimage import map_coordinates  # local import — fallback only

    coords = np.vstack([y_coords, x_coords]).astype(np.float64, copy=False)
    for b in range(coeff_stack.shape[0]):
        values = map_coordinates(coeff_stack[b], coords, order=3, mode="constant", cval=np.nan, prefilter=False)
        touched = map_coordinates(raw_stack[b], coords, order=1, mode="constant", cval=np.nan)
        values[~np.isfinite(touched)] = np.nan
        out[b] = values


cubic_sample_stack = _cubic_sample_stack_numba if NUMBA_AVAILABLE else _cubic_sample_stack_numpy


def warmup_numba_mb() -> None:
    """
    Trigger numba JIT compilation for the multi-band kernels.

    Mirrors :func:`isoster.numba_kernels.warmup_numba` so benchmark
    callers can warm up both the single-band and multi-band paths.
    """
    if not NUMBA_AVAILABLE:
        return
    # Warm both ring-read kernels on a two-band stack.
    stack = np.zeros((2, 8, 8), dtype=np.float64)
    ys = np.linspace(1.0, 6.0, 5)
    xs = np.linspace(1.0, 6.0, 5)
    out = np.empty((2, 5), dtype=np.float64)
    _bilinear_sample_stack_numba(stack, ys, xs, out)
    _cubic_sample_stack_numba(stack, stack, ys, xs, out)

    phi = np.linspace(0.0, 2.0 * np.pi, 64)
    _ = _build_joint_design_matrix_numba(phi, 2)
    # Warm the jagged-builder kernel for both normalize=False and =True.
    n_per_band = np.array([64, 60], dtype=np.int64)
    band_offsets = np.array([0, 64], dtype=np.int64)
    phi_concat = np.concatenate([phi, phi[:60]])
    _ = _build_joint_design_matrix_jagged_numba(
        phi_concat,
        band_offsets,
        n_per_band,
        2,
        False,
    )
    _ = _build_joint_design_matrix_jagged_numba(
        phi_concat,
        band_offsets,
        n_per_band,
        2,
        True,
    )
    # Warm the higher-order joint design-matrix kernels (Section 6).
    orders = np.array([3, 4], dtype=np.int64)
    _ = _build_joint_design_matrix_higher_numba(phi, 2, orders)
    _ = _build_joint_design_matrix_jagged_higher_numba(
        phi_concat,
        band_offsets,
        n_per_band,
        2,
        orders,
        False,
    )
    _ = _build_joint_design_matrix_jagged_higher_numba(
        phi_concat,
        band_offsets,
        n_per_band,
        2,
        orders,
        True,
    )
    # Warm the per-band bilinear stack sampler.
    stack = np.zeros((2, 16, 16), dtype=np.float64)
    yc = np.linspace(2.5, 12.5, 16, dtype=np.float64)
    xc = np.linspace(2.5, 12.5, 16, dtype=np.float64)
    out = np.empty((2, 16), dtype=np.float64)
    _bilinear_sample_stack_numba(stack, yc, xc, out)


@njit(cache=True)
def _build_joint_design_matrix_jagged_numba(
    phi_concat: NDArray[np.float64],
    band_offsets: NDArray[np.int64],
    n_per_band: NDArray[np.int64],
    n_bands: int,
    normalize: bool,
) -> NDArray[np.float64]:
    """Numba-JIT inner kernel for ``build_joint_design_matrix_jagged``.

    Operates on a single concatenated ``phi`` vector with per-band
    offsets so numba does not have to deal with a Python list of
    arrays of varying length (lists of arrays are not first-class
    numba types).
    """
    n_total = int(phi_concat.shape[0])
    n_cols = n_bands + 4
    A = np.zeros((n_total, n_cols), dtype=np.float64)
    for b in range(n_bands):
        n_b = int(n_per_band[b])
        if n_b == 0:
            continue
        offset = int(band_offsets[b])
        if normalize:
            scale = np.sqrt(1.0 / n_b)
        else:
            scale = 1.0
        for i in range(n_b):
            row = offset + i
            p = phi_concat[row]
            A[row, b] = scale
            A[row, n_bands + 0] = scale * np.sin(p)
            A[row, n_bands + 1] = scale * np.cos(p)
            A[row, n_bands + 2] = scale * np.sin(2.0 * p)
            A[row, n_bands + 3] = scale * np.cos(2.0 * p)
    return A


def _build_joint_design_matrix_jagged_numpy(
    phi_concat: NDArray[np.float64],
    band_offsets: NDArray[np.int64],
    n_per_band: NDArray[np.int64],
    n_bands: int,
    normalize: bool,
) -> NDArray[np.float64]:
    """NumPy fallback (used when numba is unavailable / disabled)."""
    n_total = int(phi_concat.shape[0])
    n_cols = n_bands + 4
    A = np.zeros((n_total, n_cols), dtype=np.float64)
    for b in range(n_bands):
        n_b = int(n_per_band[b])
        if n_b == 0:
            continue
        offset = int(band_offsets[b])
        scale = float(np.sqrt(1.0 / n_b)) if normalize else 1.0
        rng = slice(offset, offset + n_b)
        p = phi_concat[rng]
        A[rng, b] = scale
        A[rng, n_bands + 0] = scale * np.sin(p)
        A[rng, n_bands + 1] = scale * np.cos(p)
        A[rng, n_bands + 2] = scale * np.sin(2.0 * p)
        A[rng, n_bands + 3] = scale * np.cos(2.0 * p)
    return A


_build_joint_design_matrix_jagged_impl = (
    _build_joint_design_matrix_jagged_numba if NUMBA_AVAILABLE else _build_joint_design_matrix_jagged_numpy
)


@njit(cache=True)
def _build_joint_design_matrix_higher_numba(
    phi: NDArray[np.float64],
    n_bands: int,
    orders: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Numba kernel for the shared-validity higher-order joint design matrix.

    Extends :func:`_build_joint_design_matrix_numba` with ``2*L`` extra
    columns ``[sin(orders[0]·φ), cos(orders[0]·φ), ..., sin(orders[L-1]·φ),
    cos(orders[L-1]·φ)]`` shared across all bands. Layout:

        col 0..B-1    : per-band indicator (I0_b)
        col B..B+3    : shared (A1, B1, A2, B2)
        col B+4..B+3+2L : shared (A_{orders[0]}, B_{orders[0]}, ...)

    The first ``B + 4`` columns reproduce the standard joint design matrix
    bit-for-bit, so callers that read ``coeffs[n_bands..n_bands+3]`` (e.g.
    geometry-update math, ``_compute_parameter_errors_from_joint``) keep
    working with the wider solve.
    """
    n_samples = phi.shape[0]
    L = int(orders.shape[0])
    n_rows = n_bands * n_samples
    n_cols = n_bands + 4 + 2 * L
    A = np.zeros((n_rows, n_cols), dtype=np.float64)

    for b in range(n_bands):
        for i in range(n_samples):
            row = b * n_samples + i
            A[row, b] = 1.0
            p = phi[i]
            A[row, n_bands + 0] = np.sin(p)
            A[row, n_bands + 1] = np.cos(p)
            A[row, n_bands + 2] = np.sin(2.0 * p)
            A[row, n_bands + 3] = np.cos(2.0 * p)
            for j in range(L):
                n_order = orders[j]
                A[row, n_bands + 4 + 2 * j] = np.sin(n_order * p)
                A[row, n_bands + 4 + 2 * j + 1] = np.cos(n_order * p)
    return A


def _build_joint_design_matrix_higher_numpy(
    phi: NDArray[np.float64],
    n_bands: int,
    orders: NDArray[np.int64],
) -> NDArray[np.float64]:
    """NumPy fallback for the higher-order shared-validity builder."""
    n_samples = int(phi.shape[0])
    L = int(orders.shape[0])
    n_cols = n_bands + 4 + 2 * L
    A = np.zeros((n_bands * n_samples, n_cols), dtype=np.float64)

    band_idx = np.repeat(np.arange(n_bands), n_samples)
    row_idx = np.arange(n_bands * n_samples)
    A[row_idx, band_idx] = 1.0

    sin1 = np.sin(phi)
    cos1 = np.cos(phi)
    sin2 = np.sin(2.0 * phi)
    cos2 = np.cos(2.0 * phi)
    geom = np.column_stack([sin1, cos1, sin2, cos2])  # (N, 4)
    A[:, n_bands : n_bands + 4] = np.tile(geom, (n_bands, 1))

    if L > 0:
        higher = np.empty((n_samples, 2 * L), dtype=np.float64)
        for j in range(L):
            n_order = int(orders[j])
            higher[:, 2 * j] = np.sin(n_order * phi)
            higher[:, 2 * j + 1] = np.cos(n_order * phi)
        A[:, n_bands + 4 :] = np.tile(higher, (n_bands, 1))
    return A


_build_joint_design_matrix_higher_impl = (
    _build_joint_design_matrix_higher_numba if NUMBA_AVAILABLE else _build_joint_design_matrix_higher_numpy
)


def build_joint_design_matrix_higher(
    phi: NDArray[np.floating],
    n_bands: int,
    harmonic_orders: NDArray[np.integer],
) -> NDArray[np.floating]:
    """
    Build the shared-validity higher-order joint design matrix.

    Used by the ``simultaneous_in_loop`` and ``simultaneous_original``
    higher-harmonics modes (Section 6 of plan-2026-04-29-multiband-feasibility.md).

    Parameters
    ----------
    phi : ndarray, shape (N,)
        Shared sample angles.
    n_bands : int
        Number of bands ``B``.
    harmonic_orders : ndarray of int
        Higher-order harmonic orders to include (e.g. ``[3, 4]``). All
        entries must be >= 3 (the geometric harmonics 1, 2 are always
        present in the trailing four columns).

    Returns
    -------
    ndarray, shape (B*N, B + 4 + 2*L)
        Column order: per-band indicators, then shared (A1, B1, A2, B2),
        then shared (A_n, B_n) per order.
    """
    if phi.size == 0:
        raise ValueError("phi array cannot be empty")
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    orders_i64 = np.ascontiguousarray(harmonic_orders, dtype=np.int64)
    return _build_joint_design_matrix_higher_impl(phi, n_bands, orders_i64)


@njit(cache=True)
def _build_joint_design_matrix_jagged_higher_numba(
    phi_concat: NDArray[np.float64],
    band_offsets: NDArray[np.int64],
    n_per_band: NDArray[np.int64],
    n_bands: int,
    orders: NDArray[np.int64],
    normalize: bool,
) -> NDArray[np.float64]:
    """Numba kernel for the loose-validity higher-order jagged design matrix.

    Combines :func:`_build_joint_design_matrix_jagged_numba` (jagged per-band
    blocks) with the higher-order column extension. Required for the
    ``simultaneous_*`` mode × ``loose_validity=True`` combination
    (Section 6.6).
    """
    n_total = int(phi_concat.shape[0])
    L = int(orders.shape[0])
    n_cols = n_bands + 4 + 2 * L
    A = np.zeros((n_total, n_cols), dtype=np.float64)
    for b in range(n_bands):
        n_b = int(n_per_band[b])
        if n_b == 0:
            continue
        offset = int(band_offsets[b])
        if normalize:
            scale = np.sqrt(1.0 / n_b)
        else:
            scale = 1.0
        for i in range(n_b):
            row = offset + i
            p = phi_concat[row]
            A[row, b] = scale
            A[row, n_bands + 0] = scale * np.sin(p)
            A[row, n_bands + 1] = scale * np.cos(p)
            A[row, n_bands + 2] = scale * np.sin(2.0 * p)
            A[row, n_bands + 3] = scale * np.cos(2.0 * p)
            for j in range(L):
                n_order = orders[j]
                A[row, n_bands + 4 + 2 * j] = scale * np.sin(n_order * p)
                A[row, n_bands + 4 + 2 * j + 1] = scale * np.cos(n_order * p)
    return A


def _build_joint_design_matrix_jagged_higher_numpy(
    phi_concat: NDArray[np.float64],
    band_offsets: NDArray[np.int64],
    n_per_band: NDArray[np.int64],
    n_bands: int,
    orders: NDArray[np.int64],
    normalize: bool,
) -> NDArray[np.float64]:
    """NumPy fallback for the loose-validity higher-order jagged builder."""
    n_total = int(phi_concat.shape[0])
    L = int(orders.shape[0])
    n_cols = n_bands + 4 + 2 * L
    A = np.zeros((n_total, n_cols), dtype=np.float64)
    for b in range(n_bands):
        n_b = int(n_per_band[b])
        if n_b == 0:
            continue
        offset = int(band_offsets[b])
        scale = float(np.sqrt(1.0 / n_b)) if normalize else 1.0
        rng = slice(offset, offset + n_b)
        p = phi_concat[rng]
        A[rng, b] = scale
        A[rng, n_bands + 0] = scale * np.sin(p)
        A[rng, n_bands + 1] = scale * np.cos(p)
        A[rng, n_bands + 2] = scale * np.sin(2.0 * p)
        A[rng, n_bands + 3] = scale * np.cos(2.0 * p)
        for j in range(L):
            n_order = int(orders[j])
            A[rng, n_bands + 4 + 2 * j] = scale * np.sin(n_order * p)
            A[rng, n_bands + 4 + 2 * j + 1] = scale * np.cos(n_order * p)
    return A


_build_joint_design_matrix_jagged_higher_impl = (
    _build_joint_design_matrix_jagged_higher_numba
    if NUMBA_AVAILABLE
    else _build_joint_design_matrix_jagged_higher_numpy
)


def build_joint_design_matrix_jagged_higher(
    phi_per_band,
    n_bands: int,
    harmonic_orders: NDArray[np.integer],
    normalize: bool = False,
) -> NDArray[np.floating]:
    """Loose-validity higher-order jagged joint design matrix.

    Same per-band offsets / concatenation strategy as
    :func:`build_joint_design_matrix_jagged`, with ``2*L`` extra
    higher-order columns shared across bands.
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if len(phi_per_band) != n_bands:
        raise ValueError(f"len(phi_per_band) ({len(phi_per_band)}) must equal n_bands ({n_bands})")
    orders_i64 = np.ascontiguousarray(harmonic_orders, dtype=np.int64)
    L = int(orders_i64.shape[0])
    n_per_band = np.array([int(p.size) for p in phi_per_band], dtype=np.int64)
    band_offsets = np.empty(n_bands, dtype=np.int64)
    band_offsets[0] = 0
    if n_bands > 1:
        band_offsets[1:] = np.cumsum(n_per_band)[:-1]
    if int(n_per_band.sum()) == 0:
        return np.zeros((0, n_bands + 4 + 2 * L), dtype=np.float64)
    phi_concat = np.concatenate([np.asarray(p, dtype=np.float64) for p in phi_per_band])
    return _build_joint_design_matrix_jagged_higher_impl(
        phi_concat,
        band_offsets,
        n_per_band,
        n_bands,
        orders_i64,
        normalize,
    )


def build_joint_design_matrix_jagged(
    phi_per_band,
    n_bands: int,
    normalize: bool = False,
) -> NDArray[np.floating]:
    """
    Build the loose-validity jagged joint design matrix.

    Used by the D9 backport when ``IsosterConfigMB.loose_validity=True``
    and per-band kept-sample counts ``N_b`` differ. Each band b
    contributes ``N_b`` rows; the per-band intercept column for band b
    is 1 on those rows and 0 elsewhere; the shared geometric block
    uses each band's own angle array.

    The hot-loop kernel is numba-accelerated (with a NumPy fallback
    when numba is unavailable). The Python-level wrapper concatenates
    the jagged ``phi_per_band`` list into a single 1-D array with
    per-band offsets so the kernel does not have to handle Python
    lists of variable-length arrays.

    Parameters
    ----------
    phi_per_band : list of ndarray, length B
        Each entry is band b's surviving-angle array of length ``N_b``.
        Empty arrays are allowed (band contributes zero rows).
    n_bands : int
        Number of bands ``B`` (must equal ``len(phi_per_band)``).
    normalize : bool, default False
        When True, row-scale each band's block by ``√(1/N_b)`` so the
        band's total contribution to ``A^T A`` equals 1 regardless of
        ``N_b``. Implements the ``per_band_count`` mode of the
        ``loose_validity_band_normalization`` knob (Q7-(b)).
        When False, rows are unscaled and bands with more samples
        dominate the joint solve in proportion to ``N_b``.

    Returns
    -------
    ndarray
        Shape ``(Σ N_b, B + 4)``. Column order matches
        :func:`build_joint_design_matrix`:
        ``[I_0, I_1, ..., I_{B-1}, A1, B1, A2, B2]``.

    Raises
    ------
    ValueError
        If ``len(phi_per_band) != n_bands`` or ``n_bands < 1``.
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if len(phi_per_band) != n_bands:
        raise ValueError(f"len(phi_per_band) ({len(phi_per_band)}) must equal n_bands ({n_bands})")

    n_per_band = np.array([int(p.size) for p in phi_per_band], dtype=np.int64)
    band_offsets = np.empty(n_bands, dtype=np.int64)
    band_offsets[0] = 0
    if n_bands > 1:
        band_offsets[1:] = np.cumsum(n_per_band)[:-1]
    if int(n_per_band.sum()) == 0:
        # All bands empty — return a zero-row matrix of correct width.
        return np.zeros((0, n_bands + 4), dtype=np.float64)
    phi_concat = np.concatenate([np.asarray(p, dtype=np.float64) for p in phi_per_band])
    return _build_joint_design_matrix_jagged_impl(
        phi_concat,
        band_offsets,
        n_per_band,
        n_bands,
        normalize,
    )
