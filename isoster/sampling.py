import numpy as np
from scipy.ndimage import map_coordinates

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

def extract_isophote_data(image, mask, x0, y0, sma, eps, pa, astep=0.1, linear_growth=False):
    """
    Extract image pixels along an elliptical path using vectorized sampling.
    
    This is the core performance optimization - replacing photutils' area-based integration
    (integrator.BILINEAR or MEDIAN) with direct path-based sampling via map_coordinates.
    
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
        
    Returns
    -------
    phi : 1D array
        Polar angles (theta) of valid sample points.
    intens : 1D array
        Intensity values at sample points.
    radii : 1D array
        Semi-major axis values (constant = sma).
    """
    h, w = image.shape
    
    # SAMPLING DENSITY
    n_samples = max(64, int(2 * np.pi * sma))
    
    # POLAR ANGLE SAMPLING
    phi = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
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
    
    return phi[valid], intens[valid], np.full(np.sum(valid), sma)
