import numpy as np
from .fitting import fit_isophote
from .config import IsosterConfig

def fit_central_pixel(image, mask, x0, y0, debug=False):
    """
    Fit the central pixel (SMA=0).

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray, optional): Boolean mask (True=bad).
        x0 (float): Center x coordinate.
        y0 (float): Center y coordinate.
        debug (bool): If True, include extra debug fields.

    Returns:
        dict: A dictionary containing the fitting result for the central pixel.
    """
    # Simple estimation for center
    val = image[int(y0), int(x0)]
    valid = True
    if mask is not None:
        if mask[int(y0), int(x0)]:
            valid = False
            
    return {
        'x0': x0, 'y0': y0, 'eps': 0.0, 'pa': 0.0, 'sma': 0.0,
        'intens': val if valid else np.nan,
        'rms': 0.0, 'intens_err': 0.0,
        'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
        'a3': 0.0, 'b3': 0.0, 'a3_err': 0.0, 'b3_err': 0.0,
        'a4': 0.0, 'b4': 0.0, 'a4_err': 0.0, 'b4_err': 0.0,
        'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
        'stop_code': 0 if valid else -1,
        'niter': 0, 'valid': valid
    }

def fit_image(image, mask=None, config=None):
    """
    Main driver to fit isophotes to an entire image.

    This function orchestrates the fitting process, starting from a central guess,
    fitting outward to the edge, and optionally inward to the center.

    Args:
        image (np.ndarray): 2D Input image.
        mask (np.ndarray, optional): 2D Bad pixel mask (True=bad).
        config (dict or IsosterConfig, optional): Configuration parameters.
            If None, default configuration is used.

    Returns:
        dict: A dictionary containing:
            - 'isophotes': List of dictionaries, each representing a fitted isophote.
            - 'config': The IsosterConfig object used for the fit.
    """
    if config is None:
        cfg = IsosterConfig()
    elif isinstance(config, dict):
        cfg = IsosterConfig(**config)
    else:
        cfg = config
    
    # Handle forced mode
    if cfg.forced:
        from .fitting import extract_forced_photometry
        
        isophotes = []
        for sma in cfg.forced_sma:
            iso = extract_forced_photometry(
                image, mask, sma,
                cfg.x0 if cfg.x0 is not None else image.shape[1]/2,
                cfg.y0 if cfg.y0 is not None else image.shape[0]/2,
                cfg.eps, cfg.pa,
                integrator=cfg.integrator,
                sclip=cfg.sclip,
                nclip=cfg.nclip,
                use_eccentric_anomaly=cfg.use_eccentric_anomaly
            )
            isophotes.append(iso)
        
        return {'isophotes': isophotes, 'config': cfg}
    
    # Regular fitting mode
    h, w = image.shape
    
    # Initial Parameters
    x0 = cfg.x0 if cfg.x0 is not None else w / 2.0
    y0 = cfg.y0 if cfg.y0 is not None else h / 2.0
    sma0 = cfg.sma0
    minsma = cfg.minsma
    maxsma = cfg.maxsma if cfg.maxsma is not None else max(h, w) / 2.0
    astep = cfg.astep
    linear_growth = cfg.linear_growth
    
    # 1. Fit Central Pixel (Approximation)
    central_result = fit_central_pixel(image, mask, x0, y0, debug=cfg.debug)
    
    # 2. Fit First Isophote at SMA0
    start_geometry = {
        'x0': x0, 'y0': y0, 
        'eps': cfg.eps, 'pa': cfg.pa
    }
    
    # Pass cfg object to fit_isophote
    first_iso = fit_isophote(image, mask, sma0, start_geometry, cfg)
    
    # 3. Grow Outwards
    outwards_results = []
    if first_iso['stop_code'] <= 2: # Success or minor issues
        outwards_results.append(first_iso)
        current_iso = first_iso
        current_sma = first_iso['sma']
        
        while True:
            if linear_growth:
                next_sma = current_sma + astep
            else:
                next_sma = current_sma * (1.0 + astep)
                
            if next_sma > maxsma:
                break
            
            # Update sma tracking
            current_sma = next_sma
                
            next_iso = fit_isophote(image, mask, next_sma, current_iso, cfg)
            outwards_results.append(next_iso)
            
            # If good fit, update geometry for next step
            if next_iso['stop_code'] in [0, 1, 2]:
                current_iso = next_iso
                
    # 4. Grow Inwards
    inwards_results = []
    if minsma < sma0:
        current_iso = first_iso
        current_sma = first_iso['sma']
        
        while True:
            if linear_growth:
                next_sma = current_sma - astep
            else:
                next_sma = current_sma / (1.0 + astep)
            
            # Stop if smaller than minsma or effectively too small (e.g. 0.5 pixel)
            limit_sma = max(minsma, 0.5) 
            if next_sma < limit_sma:
                break
            
            current_sma = next_sma
            
            # Use going_inwards=True flag
            next_iso = fit_isophote(image, mask, next_sma, current_iso, cfg, going_inwards=True)
            inwards_results.append(next_iso)
            
            if next_iso['stop_code'] in [0, 1, 2]:
                current_iso = next_iso

    # Combine results
    # Inwards list needs to be reversed so SMAs are ascending
    if minsma <= 0.0:
        # Prepend central pixel
        final_list = [central_result] + inwards_results[::-1] + outwards_results
    else:
        final_list = inwards_results[::-1] + outwards_results
    
    # Compute Curve-of-Growth if requested
    if cfg.compute_cog:
        from .cog import compute_cog, add_cog_to_isophotes
        
        # Determine if geometry was fixed
        fix_geometry = cfg.fix_center and cfg.fix_pa and cfg.fix_eps
        
        cog_results = compute_cog(final_list, 
                                  fix_center=cfg.fix_center, 
                                  fix_geometry=fix_geometry)
        
        # Add CoG data to isophotes
        add_cog_to_isophotes(final_list, cog_results)
            
    # Return as dict matching legacy structure + config object
    return {'isophotes': final_list, 'config': cfg}
