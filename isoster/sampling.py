import numpy as np
from scipy.ndimage import map_coordinates
from numba import njit
from collections import namedtuple

# Named tuple for isophote data with clear ψ/φ separation
IsophoteData = namedtuple('IsophoteData', [
    'angles',      # ψ (EA mode) or φ (regular mode) - for harmonic fitting
    'phi',         # φ (position angles) - for geometry updates
    'intens',      # Intensity values
    'radii'        # Semi-major axis values
])
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
    position_angle = np.arctan2(
        (1 - ellipticity) * np.sin(eccentric_anomaly),
        np.cos(eccentric_anomaly)
    )
    # Ensure result is in [0, 2π)
    position_angle = position_angle % (2 * np.pi)
    return position_angle

def get_elliptical_coordinates(x, y, x0, y0, pa, eps):
    """
    Convert image coordinates (x, y) to elliptical coordinates (sma, phi).
    
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
    phi : float or array-like
        The elliptical angle (eccentric anomaly) in radians.
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
    sma = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps))**2)
    phi = np.arctan2(y_rot / (1.0 - eps), x_rot)
    
    return sma, phi

def extract_isophote_data(image, mask, x0, y0, sma, eps, pa, astep=0.1, linear_growth=False, use_eccentric_anomaly=False):
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
    astep : float
        Not used for sampling (kept for API compatibility).
    linear_growth : bool
        Not used here (kept for API compatibility).
    use_eccentric_anomaly : bool
        If True, sample uniformly in ψ and fit harmonics in ψ space (Ciambur 2015).
        If False, sample uniformly in φ and fit harmonics in φ space (traditional).
        
    Returns
    -------
    IsophoteData : namedtuple
        - angles: ψ (if use_eccentric_anomaly) or φ (if not) - for harmonic fitting
        - phi: φ (position angles) - always present, for geometry updates
        - intens: Intensity values
        - radii: Semi-major axis values (constant = sma)
    """
    h, w = image.shape
    
    # SAMPLING DENSITY
    n_samples = max(64, int(2 * np.pi * sma))
    
    # ANGLE SAMPLING
    if use_eccentric_anomaly:
        # Sample uniformly in ψ (eccentric anomaly) for uniform arc-length coverage
        psi = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        # Convert ψ → φ for coordinate calculation
        phi = eccentric_anomaly_to_position_angle(psi, eps)
    else:
        # Traditional: sample uniformly in φ (position angle)
        phi = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        psi = phi  # In regular mode, ψ = φ
    
    # ELLIPSE EQUATION IN POLAR COORDINATES
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    denom = np.sqrt(((1.0 - eps) * cos_phi)**2 + sin_phi**2)
    r = sma * (1.0 - eps) / denom
    
    # Convert to Cartesian in rotated frame
    x_rot = r * cos_phi
    y_rot = r * sin_phi
    
    # ROTATION TO IMAGE FRAME
    cos_pa = np.cos(pa)
    sin_pa = np.sin(pa)
    
    x = x0 + x_rot * cos_pa - y_rot * sin_pa
    y = y0 + x_rot * sin_pa + y_rot * cos_pa
    
    # VECTORIZED SAMPLING
    coords = np.vstack([y, x])
    intens = map_coordinates(image, coords, order=1, mode='constant', cval=np.nan)
    
    # MASKING
    if mask is not None:
        mask_vals = map_coordinates(mask.astype(float), coords, order=0, mode='constant', cval=1.0)
        valid = mask_vals < 0.5
    else:
        valid = np.ones_like(intens, dtype=bool)
        
    valid &= ~np.isnan(intens)
    
    # Return named tuple with appropriate angles
    if use_eccentric_anomaly:
        return IsophoteData(
            angles=psi[valid],      # ψ for harmonic fitting (Ciambur 2015)
            phi=phi[valid],          # φ for geometry updates
            intens=intens[valid],
            radii=np.full(np.sum(valid), sma)
        )
    else:
        return IsophoteData(
            angles=phi[valid],       # φ for harmonic fitting (traditional)
            phi=phi[valid],          # φ for geometry (same as angles)
            intens=intens[valid],
            radii=np.full(np.sum(valid), sma)
        )
