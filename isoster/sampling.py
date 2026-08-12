from collections import namedtuple

import numpy as np
from scipy.ndimage import map_coordinates

# Import numba-accelerated kernels (with numpy fallback)
from .numba_kernels import compute_ellipse_coords

# Named tuple for isophote data with clear ψ/φ separation
IsophoteData = namedtuple(
    "IsophoteData",
    [
        "angles",  # ψ (EA mode) or φ (regular mode) - for harmonic fitting
        "phi",  # φ at the sampled locations - retained for aligned bookkeeping
        "intens",  # Intensity values
        "radii",  # Semi-major axis values
        "variances",  # Per-pixel variance values (None when no variance map provided)
    ],
)


def eccentric_anomaly_to_position_angle(eccentric_anomaly, ellipticity):
    """
    Convert eccentric anomaly to position angle for ellipse sampling.

    Reference: B. C. Ciambur 2015 ApJ 810 120, Equation 4 (Modified)

    Standard EA definition: x = a cos ψ, y = b sin ψ
    tan φ = (b/a) tan ψ = (1-ε) tan ψ

    NOTE: Ciambur (2015) uses ψ = -arctan(...), which makes ψ run opposite to φ.
    We use the standard definition here to ensure ψ and φ align (rotate in same direction).
    This allows us to use standard Jedrzejewski (1987) geometry updates which expect
    harmonics to index angle in the standard counter-clockwise direction.

    Args:
        eccentric_anomaly (np.ndarray): ψ values, uniformly sampled in [0, 2π)
        ellipticity (float): ε = 1 - b/a, where b is semi-minor axis, a is semi-major axis

    Returns:
        np.ndarray: φ values (position angles) for coordinate calculation
    """
    # Standard: tan(φ) = (1 - ε) * tan(ψ)
    # Use atan2 for proper quadrant handling
    position_angle = np.arctan2((1 - ellipticity) * np.sin(eccentric_anomaly), np.cos(eccentric_anomaly))
    # Ensure result is in [0, 2π)
    position_angle = position_angle % (2 * np.pi)
    return position_angle


def get_elliptical_coordinates(x, y, x0, y0, pa, eps):
    """
    Convert image coordinates (x, y) to elliptical coordinates (sma, psi).

    Parameters
    ----------
    x, y : float or array-like
        Image coordinates.
    x0, y0 : float
        Center of the ellipse.
    pa : float
        Position angle in radians (counter-clockwise from x-axis).
    eps : float
        Ellipticity (1 - b/a).

    Returns
    -------
    sma : float or array-like
        The semi-major axis of the ellipse passing through (x, y).
    psi : float or array-like
        The eccentric anomaly of the point on that ellipse, in radians.
        NOTE: this is the eccentric anomaly (psi), NOT the position angle
        phi used elsewhere in the package; they are related by
        ``tan(phi) = (1 - eps) * tan(psi)``.
    """
    # Shift to center
    dx = x - x0
    dy = y - y0

    # Rotate to align with major axis
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)

    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa

    # Ellipse equation: (x/a)^2 + (y/b)^2 = 1
    # r^2 = x^2 + (y / (1-eps))^2
    sma = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps)) ** 2)
    psi = np.arctan2(y_rot / (1.0 - eps), x_rot)

    return sma, psi


def extract_isophote_data(image, mask, x0, y0, sma, eps, pa, use_eccentric_anomaly=False, variance_map=None):
    """
    Extract image pixels along an elliptical path using vectorized sampling.

    This is the core performance optimization - replacing photutils' area-based integration
    (integrator.BILINEAR or MEDIAN) with direct path-based sampling via map_coordinates.

    Per Ciambur (2015), when use_eccentric_anomaly=True, harmonics should be fitted in
    ψ (eccentric anomaly) space, NOT φ (position angle) space.

    Parameters
    ----------
    image : 2D array
        Input image.
    mask : 2D boolean array
        Mask (True = bad pixel).
    x0, y0 : float
        Ellipse center coordinates.
    sma : float
        Semi-major axis length.
    eps : float
        Ellipticity (1 - b/a).
    pa : float
        Position angle in radians.
    use_eccentric_anomaly : bool
        If True, sample uniformly in ψ and fit harmonics in ψ space (Ciambur 2015).
        If False, sample uniformly in φ and fit harmonics in φ space (traditional).
    variance_map : 2D array, optional
        Per-pixel variance map. When provided, variance values are sampled along the
        ellipse using bilinear interpolation and included in the returned IsophoteData.

    Returns
    -------
    IsophoteData : namedtuple
        - angles: ψ (if use_eccentric_anomaly) or φ (if not) - for harmonic fitting
        - phi: φ (position angles) - always present, for geometry updates
        - intens: Intensity values
        - radii: Semi-major axis values (constant = sma)
        - variances: Per-pixel variance values (None when variance_map is not provided)
    """
    h, w = image.shape

    # SAMPLING DENSITY
    n_samples = max(64, int(2 * np.pi * sma))

    # NUMBA-ACCELERATED COORDINATE COMPUTATION
    # Computes ellipse sampling coordinates and angle arrays
    # Returns: (x, y, angles, phi) where angles=ψ (EA mode) or φ (regular mode)
    x, y, psi, phi = compute_ellipse_coords(n_samples, sma, eps, pa, x0, y0, use_eccentric_anomaly)

    # VECTORIZED SAMPLING
    coords = np.vstack([y, x])
    intens = map_coordinates(image, coords, order=1, mode="constant", cval=np.nan)

    # MASKING
    # The mask must be a float array for map_coordinates. Callers should
    # pre-convert with _prepare_mask_float() to avoid repeated allocation;
    # the guard here handles any remaining bool/int arrays.
    if mask is not None:
        mask_f = mask if mask.dtype.kind == "f" else mask.astype(np.float64)
        mask_vals = map_coordinates(mask_f, coords, order=0, mode="constant", cval=1.0)
        valid = mask_vals < 0.5
    else:
        valid = np.ones_like(intens, dtype=bool)

    valid &= ~np.isnan(intens)

    # Sample variance map if provided.
    # A variance that is not finite or not strictly positive carries no usable
    # information, so the sample is dropped exactly like a masked pixel. This
    # keeps every ring statistic and its uncertainty on one identical sample
    # set; see docs/04-architecture.md, "Invalid-variance policy".
    var_vals = None
    if variance_map is not None:
        var_vals = map_coordinates(variance_map, coords, order=1, mode="constant", cval=np.nan)
        valid &= np.isfinite(var_vals) & (var_vals > 0.0)

    # Return named tuple with appropriate angles
    sampled_variances = var_vals[valid] if var_vals is not None else None
    if use_eccentric_anomaly:
        return IsophoteData(
            angles=psi[valid],  # ψ for harmonic fitting (Ciambur 2015)
            phi=phi[valid],  # Corresponding φ values, kept aligned with ψ samples
            intens=intens[valid],
            radii=np.full(np.sum(valid), sma),
            variances=sampled_variances,
        )
    else:
        return IsophoteData(
            angles=phi[valid],  # φ for harmonic fitting (traditional)
            phi=phi[valid],  # φ for geometry (same as angles)
            intens=intens[valid],
            radii=np.full(np.sum(valid), sma),
            variances=sampled_variances,
        )
