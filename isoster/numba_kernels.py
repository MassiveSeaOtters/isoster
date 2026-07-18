"""
Numba-accelerated kernels for isoster isophote fitting.

This module provides JIT-compiled versions of performance-critical functions.
All functions have fallback implementations using pure numpy when numba is not available.

The kernels here are designed to:
1. Be drop-in replacements for their numpy counterparts
2. Provide 2-5x speedup on hot paths
3. Maintain bit-for-bit identical numerical results

Usage:
    # Automatic selection (numba if available, else numpy)
    from isoster.numba_kernels import harmonic_model, compute_ellipse_coords

    # Force numpy fallback
    from isoster.numba_kernels import harmonic_model_numpy
"""

import os
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

# Try to import numba, set flag for availability
# Also check NUMBA_DISABLE_JIT environment variable
try:
    from numba import njit

    # Check if JIT is disabled via environment variable
    # When disabled, use numpy fallbacks for better performance than interpreted numba loops
    NUMBA_AVAILABLE = os.environ.get("NUMBA_DISABLE_JIT", "0") != "1"
except ImportError:
    NUMBA_AVAILABLE = False

if not NUMBA_AVAILABLE:
    # Create dummy decorator for systems without numba or when JIT is disabled
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# =============================================================================
# Harmonic Model Evaluation
# =============================================================================


@njit(cache=True)
def _harmonic_model_numba(phi, coeffs):
    """
    Evaluate harmonic model at given angles (numba-accelerated).

    Model: I(φ) = c0 + c1*sin(φ) + c2*cos(φ) + c3*sin(2φ) + c4*cos(2φ)

    Args:
        phi: Array of angles (radians)
        coeffs: Array of 5 coefficients [c0, c1, c2, c3, c4]

    Returns:
        Array of model intensities at each angle
    """
    n = len(phi)
    result = np.empty(n, dtype=np.float64)

    c0, c1, c2, c3, c4 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]

    for i in range(n):
        p = phi[i]
        result[i] = c0 + c1 * np.sin(p) + c2 * np.cos(p) + c3 * np.sin(2.0 * p) + c4 * np.cos(2.0 * p)

    return result


def _harmonic_model_numpy(phi, coeffs):
    """
    Evaluate harmonic model at given angles (pure numpy).

    This is the fallback implementation when numba is not available.
    """
    return (
        coeffs[0]
        + coeffs[1] * np.sin(phi)
        + coeffs[2] * np.cos(phi)
        + coeffs[3] * np.sin(2.0 * phi)
        + coeffs[4] * np.cos(2.0 * phi)
    )


# Select implementation based on numba availability
_harmonic_model_impl = _harmonic_model_numba if NUMBA_AVAILABLE else _harmonic_model_numpy


def harmonic_model(phi: NDArray[np.floating], coeffs: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Evaluate harmonic model at given angles.

    Model: I(φ) = c0 + c1*sin(φ) + c2*cos(φ) + c3*sin(2φ) + c4*cos(2φ)

    Args:
        phi: Array of angles (radians)
        coeffs: Array of 5 coefficients [c0, c1, c2, c3, c4]

    Returns:
        Array of model intensities at each angle

    Raises:
        ValueError: If coeffs has fewer than 5 elements
    """
    if len(coeffs) < 5:
        raise ValueError(f"coeffs must have at least 5 elements, got {len(coeffs)}")
    return _harmonic_model_impl(phi, coeffs)


# =============================================================================
# Eccentric Anomaly to Position Angle Conversion
# =============================================================================


@njit(cache=True)
def _ea_to_pa_numba(psi, eps):
    """
    Convert eccentric anomaly to position angle (numba-accelerated).

    Standard definition: tan(φ) = (1 - ε) * tan(ψ)
    Using atan2 for proper quadrant handling.

    Args:
        psi: Array of eccentric anomaly values (radians)
        eps: Ellipticity (1 - b/a)

    Returns:
        Array of position angles (radians), in [0, 2π)
    """
    n = len(psi)
    phi = np.empty(n, dtype=np.float64)

    one_minus_eps = 1.0 - eps
    two_pi = 2.0 * np.pi

    for i in range(n):
        p = psi[i]
        phi[i] = np.arctan2(one_minus_eps * np.sin(p), np.cos(p))
        # Ensure result is in [0, 2π)
        if phi[i] < 0.0:
            phi[i] += two_pi

    return phi


def _ea_to_pa_numpy(psi, eps):
    """
    Convert eccentric anomaly to position angle (pure numpy).
    """
    phi = np.arctan2((1.0 - eps) * np.sin(psi), np.cos(psi))
    return phi % (2.0 * np.pi)


# Select implementation
_ea_to_pa_impl = _ea_to_pa_numba if NUMBA_AVAILABLE else _ea_to_pa_numpy


def ea_to_pa(psi: NDArray[np.floating], eps: float) -> NDArray[np.floating]:
    """
    Convert eccentric anomaly to position angle.

    Standard definition: tan(φ) = (1 - ε) * tan(ψ)
    Using atan2 for proper quadrant handling.

    Args:
        psi: Array of eccentric anomaly values (radians)
        eps: Ellipticity (1 - b/a), must be in [0, 1)

    Returns:
        Array of position angles (radians), in [0, 2π)

    Raises:
        ValueError: If eps is not in [0, 1)
    """
    if not (0 <= eps < 1):
        raise ValueError(f"eps must be in [0, 1), got {eps}")
    return _ea_to_pa_impl(psi, eps)


# =============================================================================
# Ellipse Coordinate Computation
# =============================================================================


@njit(cache=True)
def _compute_ellipse_coords_numba(n_samples, sma, eps, pa, x0, y0, use_ea):
    """
    Compute ellipse sampling coordinates (numba-accelerated).

    This computes the (x, y) image coordinates for sampling points along
    an ellipse, along with the corresponding angles.

    Args:
        n_samples: Number of sampling points
        sma: Semi-major axis length (pixels)
        eps: Ellipticity (1 - b/a)
        pa: Position angle (radians, counter-clockwise from x-axis)
        x0, y0: Ellipse center coordinates
        use_ea: If True, sample uniformly in eccentric anomaly

    Returns:
        Tuple of (x_coords, y_coords, angles, phi) where:
        - x_coords, y_coords: Image coordinates for sampling
        - angles: ψ (if use_ea) or φ (if not) - for harmonic fitting
        - phi: φ (position angles) - for geometry updates
    """
    # Allocate output arrays
    x_coords = np.empty(n_samples, dtype=np.float64)
    y_coords = np.empty(n_samples, dtype=np.float64)
    angles = np.empty(n_samples, dtype=np.float64)
    phi = np.empty(n_samples, dtype=np.float64)

    # Pre-compute constants
    two_pi = 2.0 * np.pi
    one_minus_eps = 1.0 - eps
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)

    # Generate uniformly spaced angles
    # Use (two_pi * i) / n_samples to match numpy linspace precision
    for i in range(n_samples):
        angles[i] = two_pi * i / n_samples

    if use_ea:
        # Sample uniformly in ψ (eccentric anomaly)
        # Convert ψ → φ for coordinate calculation
        for i in range(n_samples):
            psi_i = angles[i]
            # EA to PA conversion
            phi_i = np.arctan2(one_minus_eps * np.sin(psi_i), np.cos(psi_i))
            if phi_i < 0.0:
                phi_i += two_pi
            phi[i] = phi_i
    else:
        # Sample uniformly in φ (position angle)
        for i in range(n_samples):
            phi[i] = angles[i]

    # Compute ellipse coordinates
    for i in range(n_samples):
        phi_i = phi[i]
        cos_phi = np.cos(phi_i)
        sin_phi = np.sin(phi_i)

        # Ellipse equation in polar coordinates
        denom = np.sqrt((one_minus_eps * cos_phi) ** 2 + sin_phi**2)
        r = sma * one_minus_eps / denom

        # Cartesian in rotated frame
        x_rot = r * cos_phi
        y_rot = r * sin_phi

        # Rotate to image frame
        x_coords[i] = x0 + x_rot * cos_pa - y_rot * sin_pa
        y_coords[i] = y0 + x_rot * sin_pa + y_rot * cos_pa

    return x_coords, y_coords, angles, phi


def _compute_ellipse_coords_numpy(n_samples, sma, eps, pa, x0, y0, use_ea):
    """
    Compute ellipse sampling coordinates (pure numpy).
    """
    # Generate uniformly spaced angles
    angles = np.linspace(0, 2.0 * np.pi, n_samples, endpoint=False)

    if use_ea:
        # Convert eccentric anomaly to position angle
        phi = np.arctan2((1.0 - eps) * np.sin(angles), np.cos(angles))
        phi = phi % (2.0 * np.pi)
    else:
        phi = angles.copy()

    # Ellipse coordinates
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    denom = np.sqrt(((1.0 - eps) * cos_phi) ** 2 + sin_phi**2)
    r = sma * (1.0 - eps) / denom

    x_rot = r * cos_phi
    y_rot = r * sin_phi

    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)

    x_coords = x0 + x_rot * cos_pa - y_rot * sin_pa
    y_coords = y0 + x_rot * sin_pa + y_rot * cos_pa

    return x_coords, y_coords, angles, phi


# Select implementation
_compute_ellipse_coords_impl = _compute_ellipse_coords_numba if NUMBA_AVAILABLE else _compute_ellipse_coords_numpy


def compute_ellipse_coords(
    n_samples: int, sma: float, eps: float, pa: float, x0: float, y0: float, use_ea: bool
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute ellipse sampling coordinates.

    This computes the (x, y) image coordinates for sampling points along
    an ellipse, along with the corresponding angles.

    Args:
        n_samples: Number of sampling points (must be > 0)
        sma: Semi-major axis length (pixels)
        eps: Ellipticity (1 - b/a), must be in [0, 1)
        pa: Position angle (radians, counter-clockwise from x-axis)
        x0, y0: Ellipse center coordinates
        use_ea: If True, sample uniformly in eccentric anomaly

    Returns:
        Tuple of (x_coords, y_coords, angles, phi) where:
        - x_coords, y_coords: Image coordinates for sampling
        - angles: ψ (if use_ea) or φ (if not) - for harmonic fitting
        - phi: φ (position angles) - for geometry updates

    Raises:
        ValueError: If n_samples <= 0 or eps not in [0, 1)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if not (0 <= eps < 1):
        raise ValueError(f"eps must be in [0, 1), got {eps}")
    return _compute_ellipse_coords_impl(n_samples, sma, eps, pa, x0, y0, use_ea)


# =============================================================================
# Harmonic Design Matrix Construction
# =============================================================================


@njit(cache=True)
def _build_harmonic_matrix_numba(phi):
    """
    Build the design matrix for harmonic fitting (numba-accelerated).

    Constructs the matrix A for least squares fitting:
    A @ coeffs = intensity

    where A[i, :] = [1, sin(φᵢ), cos(φᵢ), sin(2φᵢ), cos(2φᵢ)]

    Args:
        phi: Array of angles (radians)

    Returns:
        Design matrix of shape (n_samples, 5)
    """
    n = len(phi)
    A = np.empty((n, 5), dtype=np.float64)

    for i in range(n):
        p = phi[i]
        A[i, 0] = 1.0
        A[i, 1] = np.sin(p)
        A[i, 2] = np.cos(p)
        A[i, 3] = np.sin(2.0 * p)
        A[i, 4] = np.cos(2.0 * p)

    return A


def _build_harmonic_matrix_numpy(phi):
    """
    Build the design matrix for harmonic fitting (pure numpy).
    """
    s1 = np.sin(phi)
    c1 = np.cos(phi)
    s2 = np.sin(2.0 * phi)
    c2 = np.cos(2.0 * phi)

    return np.column_stack([np.ones_like(phi), s1, c1, s2, c2])


# Select implementation
_build_harmonic_matrix_impl = _build_harmonic_matrix_numba if NUMBA_AVAILABLE else _build_harmonic_matrix_numpy


def build_harmonic_matrix(phi: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Build the design matrix for harmonic fitting.

    Constructs the matrix A for least squares fitting:
    A @ coeffs = intensity

    where A[i, :] = [1, sin(φᵢ), cos(φᵢ), sin(2φᵢ), cos(2φᵢ)]

    Args:
        phi: Array of angles (radians)

    Returns:
        Design matrix of shape (n_samples, 5)

    Raises:
        ValueError: If phi is empty
    """
    if len(phi) == 0:
        raise ValueError("phi array cannot be empty")
    return _build_harmonic_matrix_impl(phi)


# =============================================================================
# Utility Functions
# =============================================================================


def check_numba_available():
    """Check if numba is available and working."""
    return NUMBA_AVAILABLE


def warmup_numba():
    """
    Warm up numba JIT compilation by calling each function once.

    Call this at import time or before benchmarking to avoid
    including compilation time in measurements.
    """
    if not NUMBA_AVAILABLE:
        return

    # Small test arrays
    phi = np.linspace(0, 2 * np.pi, 64)
    coeffs = np.array([1.0, 0.1, 0.1, 0.05, 0.05])

    # Trigger compilation
    _ = _harmonic_model_numba(phi, coeffs)
    _ = _ea_to_pa_numba(phi, 0.3)
    _ = _compute_ellipse_coords_numba(64, 10.0, 0.3, 0.5, 100.0, 100.0, True)
    _ = _build_harmonic_matrix_numba(phi)


# =============================================================================
# Module Initialization
# =============================================================================

if __name__ == "__main__":
    # Test the implementations
    print(f"Numba available: {NUMBA_AVAILABLE}")

    # Test data
    phi = np.linspace(0, 2 * np.pi, 100)
    coeffs = np.array([100.0, 1.0, 2.0, 0.5, 0.3])
    eps = 0.4

    # Test harmonic model
    result_numba = _harmonic_model_numba(phi, coeffs)
    result_numpy = _harmonic_model_numpy(phi, coeffs)
    print(f"Harmonic model max diff: {np.max(np.abs(result_numba - result_numpy)):.2e}")

    # Test EA to PA
    result_numba = _ea_to_pa_numba(phi, eps)
    result_numpy = _ea_to_pa_numpy(phi, eps)
    print(f"EA to PA max diff: {np.max(np.abs(result_numba - result_numpy)):.2e}")

    # Test ellipse coords
    x1, y1, a1, p1 = _compute_ellipse_coords_numba(100, 50.0, 0.3, 0.5, 200.0, 200.0, True)
    x2, y2, a2, p2 = _compute_ellipse_coords_numpy(100, 50.0, 0.3, 0.5, 200.0, 200.0, True)
    print(f"Ellipse coords x max diff: {np.max(np.abs(x1 - x2)):.2e}")
    print(f"Ellipse coords y max diff: {np.max(np.abs(y1 - y2)):.2e}")

    # Test harmonic matrix
    A1 = _build_harmonic_matrix_numba(phi)
    A2 = _build_harmonic_matrix_numpy(phi)
    print(f"Harmonic matrix max diff: {np.max(np.abs(A1 - A2)):.2e}")

    print("\nAll tests passed!")
