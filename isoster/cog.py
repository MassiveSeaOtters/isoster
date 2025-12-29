"""
Curve-of-Growth (CoG) Photometry Module

Efficient computation of cumulative flux profiles with proper handling
of masked regions and isophote crossing detection.
"""

import numpy as np

def compute_ellipse_area(sma, eps):
    """
    Compute ellipse area using vectorized operations.
    
    Area = π * a * b = π * a * a * (1 - eps)
    
    Args:
        sma (float or np.ndarray): Semi-major axis length(s)
        eps (float or np.ndarray): Ellipticity value(s)
        
    Returns:
        float or np.ndarray: Ellipse area(s)
    """
    b = sma * (1.0 - eps)
    return np.pi * sma * b

def detect_crossing(isophotes, fix_center=False, fix_geometry=False):
    """
    Detect crossing isophotes and negative annular areas.
    
    In regular mode, ellipses can cross each other. This function detects:
    1. Geometric crossing (ellipse parameters indicate overlap)
    2. Negative annular areas (outer area < inner area)
    
    Args:
        isophotes (list of dict): Isophote fitting results
        fix_center (bool): If True, center is fixed (skip some checks)
        fix_geometry (bool): If True, geometry is fixed (skip all checks)
        
    Returns:
        dict: Crossing detection results
            - flag_cross: boolean array indicating potential crossing
            - flag_negative_area: boolean array for negative areas
    """
    n_iso = len(isophotes)
    
    # If geometry is fixed, no crossing is possible
    if fix_geometry or (fix_center and all(iso.get('fix_eps', False) and iso.get('fix_pa', False) 
                                           for iso in isophotes)):
        return {
            'flag_cross': np.zeros(n_iso, dtype=bool),
            'flag_negative_area': np.zeros(n_iso, dtype=bool)
        }
    
    # Extract parameters
    sma = np.array([iso['sma'] for iso in isophotes])
    eps = np.array([iso['eps'] for iso in isophotes])
    
    # Compute areas
    areas = compute_ellipse_area(sma, eps)
    
    # Check for negative annular areas
    area_diff = np.diff(areas, prepend=0)
    flag_negative_area = area_diff < 0
    
    # Geometric crossing check: if center varies significantly
    flag_cross = np.zeros(n_iso, dtype=bool)
    
    if not fix_center:
        x0 = np.array([iso['x0'] for iso in isophotes])
        y0 = np.array([iso['y0'] for iso in isophotes])
        
        # Check if center moves more than SMA/2 between consecutive isophotes
        for i in range(1, n_iso):
            dx = x0[i] - x0[i-1]
            dy = y0[i] - y0[i-1]
            center_shift = np.sqrt(dx**2 + dy**2)
            
            # If center shifts significantly, mark as potential crossing
            if center_shift > 0.5 * min(sma[i], sma[i-1]):
                flag_cross[i] = True
    
    # Also flag if area decreases (redundant with negative area but more explicit)
    flag_cross |= flag_negative_area
    
    return {
        'flag_cross': flag_cross,
        'flag_negative_area': flag_negative_area
    }

def compute_cog(isophotes, fix_center=False, fix_geometry=False):
    """
    Compute curve-of-growth from isophote list.
    
    CoG is computed as cumulative sum of annular fluxes:
    Flux_annulus = Intensity * Area_annulus
    CoG[i] = sum(Flux_annulus[0:i+1])
    
    The first isophote's annulus extends from r=0 to its SMA,
    using its intensity as the average for that region.
    
    Properly handles:
    - Masked regions (uses fitted intensity, not tflux_e)
    - Isophote crossing (flags and corrects negative areas)
    - Vectorized for efficiency
    
    Args:
        isophotes (list of dict): Isophote fitting results
        fix_center (bool): Whether center was fixed during fitting
        fix_geometry (bool): Whether geometry (eps, pa) was fixed
        
    Returns:
        dict: CoG results
            - cog: cumulative flux array
            - cog_annulus: annular flux array
            - area_annulus: annular area array (corrected for negatives)
            - area_annulus_raw: raw annular areas (may be negative)
            - flag_cross: crossing detection flags
            - flag_negative_area: negative area flags
            - sma: semi-major axis array
    """
    n_iso = len(isophotes)
    
    if n_iso == 0:
        return {
            'cog': np.array([]),
            'cog_annulus': np.array([]),
            'area_annulus': np.array([]),
            'area_annulus_raw': np.array([]),
            'flag_cross': np.array([], dtype=bool),
            'flag_negative_area': np.array([], dtype=bool),
            'sma': np.array([])
        }
    
    # Extract parameters (vectorized)
    sma = np.array([iso['sma'] for iso in isophotes])
    eps = np.array([iso['eps'] for iso in isophotes])
    intens = np.array([iso['intens'] for iso in isophotes])
    
    # Compute ellipse areas
    areas = compute_ellipse_area(sma, eps)
    
    # Compute annular areas
    # First isophote: area from 0 to sma[0]
    # Subsequent: area from sma[i-1] to sma[i]
    area_annulus_raw = np.diff(areas, prepend=0)
    
    # Detect crossing
    crossing_info = detect_crossing(isophotes, fix_center, fix_geometry)
    flag_negative_area = crossing_info['flag_negative_area']
    flag_cross = crossing_info['flag_cross']
    
    # Correct negative areas (set to zero)
    area_annulus = area_annulus_raw.copy()
    area_annulus[flag_negative_area] = 0.0
    
    # Compute annular fluxes using trapezoidal rule
    # For each annulus, use average of inner and outer intensities
    # First annulus: use outer intensity only (no inner)
    intens_inner = np.concatenate([[intens[0]], intens[:-1]])
    intens_avg = 0.5 * (intens_inner + intens)
    
    # Special case: first isophote has no inner, use its own intensity
    intens_avg[0] = intens[0]
    
    cog_annulus = intens_avg * area_annulus
    
    # Compute cumulative flux (CoG)
    cog = np.cumsum(cog_annulus)
    
    return {
        'cog': cog,
        'cog_annulus': cog_annulus,
        'area_annulus': area_annulus,
        'area_annulus_raw': area_annulus_raw,
        'flag_cross': flag_cross,
        'flag_negative_area': flag_negative_area,
        'sma': sma
    }

def add_cog_to_isophotes(isophotes, cog_results):
    """
    Add CoG results to isophote dictionaries in-place.
    
    Args:
        isophotes (list of dict): Isophote fitting results
        cog_results (dict): Results from compute_cog()
    """
    for i, iso in enumerate(isophotes):
        iso['cog'] = cog_results['cog'][i]
        iso['cog_annulus'] = cog_results['cog_annulus'][i]
        iso['area_annulus'] = cog_results['area_annulus'][i]
        iso['flag_cross'] = bool(cog_results['flag_cross'][i])
        iso['flag_negative_area'] = bool(cog_results['flag_negative_area'][i])
