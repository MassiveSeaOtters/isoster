import numpy as np
from scipy.optimize import leastsq
from .sampling import extract_isophote_data
from .config import IsosterConfig

def compute_central_regularization_penalty(current_geom, previous_geom, sma, config):
    """
    Compute regularization penalty for geometry changes in central region.
    
    Adds penalty to discourage large geometry changes at low SMA, helping stabilize
    fits in low S/N central regions.
    
    Args:
        current_geom (dict): Current geometry {'x0', 'y0', 'eps', 'pa'}
        previous_geom (dict): Previous isophote geometry (or None)
        sma (float): Current semi-major axis
        config (IsosterConfig): Configuration with regularization parameters
        
    Returns:
        float: Regularization penalty value
    """
    if not config.use_central_regularization:
        return 0.0
    
    if previous_geom is None:
        return 0.0
    
    # Regularization strength decays exponentially from center
    # λ(sma) = max_strength * exp(-(sma/threshold)²)
    lambda_sma = config.central_reg_strength * np.exp(
        -(sma / config.central_reg_sma_threshold)**2
    )
    
    # No regularization beyond 3× threshold
    if lambda_sma < 1e-6:
        return 0.0
    
    weights = config.central_reg_weights
    
    # Compute changes from previous isophote
    delta_eps = current_geom['eps'] - previous_geom['eps']
    delta_pa = current_geom['pa'] - previous_geom['pa']
    
    # Handle PA wrap-around (force to [-π, π])
    while delta_pa > np.pi:
        delta_pa -= 2 * np.pi
    while delta_pa < -np.pi:
        delta_pa += 2 * np.pi
    
    delta_x0 = current_geom['x0'] - previous_geom['x0']
    delta_y0 = current_geom['y0'] - previous_geom['y0']
    
    # Penalty: λ(sma) * weighted sum of squared changes
    penalty = lambda_sma * (
        weights.get('eps', 1.0) * delta_eps**2 +
        weights.get('pa', 1.0) * delta_pa**2 +
        weights.get('center', 1.0) * (delta_x0**2 + delta_y0**2)
    )
    
    return penalty

def extract_forced_photometry(image, mask, sma, x0, y0, eps, pa, integrator='mean', sclip=3.0, nclip=0):
    """
    Extract forced photometry at a single SMA without fitting.
    
    This is used for pure forced mode where geometry is predetermined.
    
    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray): Bad pixel mask.
        sma (float): Semi-major axis length.
        x0, y0 (float): Center coordinates.
        eps (float): Ellipticity.
        pa (float): Position angle in radians.
        integrator (str): 'mean' or 'median'.
        sclip (float): Sigma clipping threshold.
        nclip (int): Number of sigma clipping iterations.
        
    Returns:
        dict: Fake isophote structure with only intensity meaningful.
    """
    # Sample along the ellipse
    phi, intens, radii = extract_isophote_data(image, mask, x0, y0, sma, eps, pa, use_eccentric_anomaly=use_eccentric_anomaly)
    
    if len(intens) == 0:
        return {
            'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma,
            'intens': np.nan, 'rms': np.nan, 'intens_err': np.nan,
            'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
            'a3': 0.0, 'b3': 0.0, 'a3_err': 0.0, 'b3_err': 0.0,
            'a4': 0.0, 'b4': 0.0, 'a4_err': 0.0, 'b4_err': 0.0,
            'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
            'stop_code': -1, 'niter': 0
        }
    
    # Sigma clipping
    if nclip > 0:
        phi, intens, _ = sigma_clip(phi, intens, sclip, nclip)
    
    # Compute intensity
    if integrator == 'median':
        intensity = np.median(intens)
    else:
        intensity = np.mean(intens)
    
    rms = np.std(intens)
    intens_err = rms / np.sqrt(len(intens))
    
    # Return fake isophote structure
    return {
        'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma,
        'intens': intensity, 'rms': rms, 'intens_err': intens_err,
        'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
        'a3': 0.0, 'b3': 0.0, 'a3_err': 0.0, 'b3_err': 0.0,
        'a4': 0.0, 'b4': 0.0, 'a4_err': 0.0, 'b4_err': 0.0,
        'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
        'stop_code': 0, 'niter': 0
    }

def fit_first_and_second_harmonics(phi, intensity):
    """
    Fit the 1st and 2nd harmonics to the intensity profile.
    
    The model is:
    y = y0 + A1*sin(E) + B1*cos(E) + A2*sin(2E) + B2*cos(2E)

    Args:
        phi (np.ndarray): eccentric anomaly angles.
        intensity (np.ndarray): sampled intensity values.

    Returns:
        tuple: (coeffs, ata_inv)
            coeffs: array of [y0, A1, B1, A2, B2]
            ata_inv: inverse covariance matrix (A^T A)^-1
    """
    s1 = np.sin(phi)
    c1 = np.cos(phi)
    s2 = np.sin(2 * phi)
    c2 = np.cos(2 * phi)
    
    A = np.column_stack([np.ones_like(phi), s1, c1, s2, c2])
    
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, intensity, rcond=None)
        
        # Compute covariance matrix (A^T * A)^-1
        # This is used for parameter error estimation later
        ata_inv = np.linalg.inv(np.dot(A.T, A))
        return coeffs, ata_inv
    except np.linalg.LinAlgError:
        return np.array([np.mean(intensity), 0.0, 0.0, 0.0, 0.0]), None

def harmonic_function(phi, coeffs):
    """Evaluate harmonic model."""
    return (coeffs[0] + coeffs[1]*np.sin(phi) + coeffs[2]*np.cos(phi) + 
            coeffs[3]*np.sin(2*phi) + coeffs[4]*np.cos(2*phi))

def sigma_clip(phi, intens, sclip=3.0, nclip=0, sclip_low=None, sclip_high=None):
    """Perform iterative sigma clipping on intensity data."""
    if nclip <= 0:
        return phi, intens, 0
        
    s_low = sclip_low if sclip_low is not None else sclip
    s_high = sclip_high if sclip_high is not None else sclip
    
    phi_c = phi.copy()
    intens_c = intens.copy()
    total_clipped = 0
    
    for _ in range(nclip):
        if len(intens_c) < 3:
            break
        mean = np.mean(intens_c)
        std = np.std(intens_c)
        
        lower = mean - s_low * std
        upper = mean + s_high * std
        
        mask = (intens_c >= lower) & (intens_c <= upper)
        n_clipped = len(intens_c) - np.sum(mask)
        
        if n_clipped == 0:
            break
            
        total_clipped += n_clipped
        phi_c = phi_c[mask]
        intens_c = intens_c[mask]
        
    return phi_c, intens_c, total_clipped

def compute_parameter_errors(phi, intens, x0, y0, sma, eps, pa, gradient, cov_matrix=None):
    """Compute parameter errors based on the covariance matrix of harmonic coefficients."""
    try:
        if cov_matrix is None:
            # Fallback to leastsq if covariance not provided
            s1, c1 = np.sin(phi), np.cos(phi)
            s2, c2 = np.sin(2*phi), np.cos(2*phi)
            params_init = [np.mean(intens), 0.0, 0.0, 0.0, 0.0]
            
            def residual(params):
                model = (params[0] + params[1]*s1 + params[2]*c1 + 
                        params[3]*s2 + params[4]*c2)
                return intens - model
            
            solution = leastsq(residual, params_init, full_output=True)
            coeffs = solution[0]
            cov_matrix = solution[1]
            
            if cov_matrix is None:
                return 0.0, 0.0, 0.0, 0.0
                
            model = (coeffs[0] + coeffs[1]*s1 + coeffs[2]*c1 + 
                    coeffs[3]*s2 + coeffs[4]*c2)
            var_residual = np.std(intens - model, ddof=len(coeffs))**2
            covariance = cov_matrix * var_residual
            errors = np.sqrt(np.diagonal(covariance))
        else:
            # Re-fit just to get model (fast linear)
            coeffs, _ = fit_first_and_second_harmonics(phi, intens)
            model = harmonic_function(phi, coeffs)
            var_residual = np.var(intens - model, ddof=5)
            covariance = cov_matrix * var_residual
            errors = np.sqrt(np.diagonal(covariance))
        
        # Parameter error formulas
        ea = abs(errors[2] / gradient)
        eb = abs(errors[1] * (1.0 - eps) / gradient)
        
        x0_err = np.sqrt((ea * np.cos(pa))**2 + (eb * np.sin(pa))**2)
        y0_err = np.sqrt((ea * np.sin(pa))**2 + (eb * np.cos(pa))**2)
        eps_err = abs(2.0 * errors[4] * (1.0 - eps) / sma / gradient)
        
        if abs(eps) > np.finfo(float).resolution:
            pa_err = abs(2.0 * errors[3] * (1.0 - eps) / sma / gradient / 
                        (1.0 - (1.0 - eps)**2))
        else:
            pa_err = 0.0
            
        return x0_err, y0_err, eps_err, pa_err
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def compute_deviations(phi, intens, sma, gradient, order):
    """Compute deviations from perfect ellipticity (higher order harmonics)."""
    try:
        s_n = np.sin(order * phi)
        c_n = np.cos(order * phi)
        y0_init = np.mean(intens)
        params_init = [y0_init, 0.0, 0.0]
        
        def residual(params):
            model = params[0] + params[1]*s_n + params[2]*c_n
            return intens - model
            
        solution = leastsq(residual, params_init, full_output=True)
        coeffs = solution[0]
        cov_matrix = solution[1]
        
        if cov_matrix is None:
            return 0.0, 0.0, 0.0, 0.0
            
        model = coeffs[0] + coeffs[1]*s_n + coeffs[2]*c_n
        var_residual = np.std(intens - model, ddof=len(coeffs))**2
        covariance = cov_matrix * var_residual
        errors = np.sqrt(np.diagonal(covariance))
        
        factor = sma * abs(gradient)
        if factor == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        a = coeffs[1] / factor
        b = coeffs[2] / factor
        a_err = errors[1] / factor
        b_err = errors[2] / factor
        
        return a, b, a_err, b_err
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def compute_gradient(image, mask, x0, y0, sma, eps, pa, step=0.1, linear_growth=False, previous_gradient=None, current_data=None, integrator='mean'):
    """Compute the radial intensity gradient."""
    if current_data is not None:
        phi_c, intens_c = current_data
    else:
        phi_c, intens_c, _ = extract_isophote_data(image, mask, x0, y0, sma, eps, pa, step, linear_growth)
    
    if len(intens_c) == 0:
        return previous_gradient * 0.8 if previous_gradient else -1.0, None
        
    if integrator == 'median':
        mean_c = np.median(intens_c)
    else:
        mean_c = np.mean(intens_c)
    
    if linear_growth:
        gradient_sma = sma + step
    else:
        gradient_sma = sma * (1.0 + step)
        
    phi_g, intens_g, _ = extract_isophote_data(image, mask, x0, y0, gradient_sma, eps, pa, step, linear_growth)
    
    if len(intens_g) == 0:
        return previous_gradient * 0.8 if previous_gradient else -1.0, None
        
    if integrator == 'median':
        mean_g = np.median(intens_g)
    else:
        mean_g = np.mean(intens_g)
    gradient = (mean_g - mean_c) / sma / step
    
    sigma_c = np.std(intens_c)
    sigma_g = np.std(intens_g)
    gradient_error = (np.sqrt(sigma_c**2 / len(intens_c) + sigma_g**2 / len(intens_g)) 
                     / sma / step)
    
    if previous_gradient is None:
        previous_gradient = gradient + gradient_error
        
    if gradient >= (previous_gradient / 3.0):
        if linear_growth:
            gradient_sma_2 = sma + 2 * step
        else:
            gradient_sma_2 = sma * (1.0 + 2 * step)
            
        phi_g2, intens_g2, _ = extract_isophote_data(image, mask, x0, y0, gradient_sma_2, eps, pa, step, linear_growth)
        
        if len(intens_g2) > 0:
            if integrator == 'median':
                mean_g2 = np.median(intens_g2)
            else:
                mean_g2 = np.mean(intens_g2)
            gradient = (mean_g2 - mean_c) / sma / (2 * step)
            sigma_g2 = np.std(intens_g2)
            gradient_error = (np.sqrt(sigma_c**2 / len(intens_c) + sigma_g2**2 / len(intens_g2))
                            / sma / (2 * step))
            
    if gradient >= (previous_gradient / 3.0):
        gradient = previous_gradient * 0.8
        gradient_error = None
        
    return gradient, gradient_error

def compute_aperture_photometry(image, mask, x0, y0, sma, eps, pa):
    """
    Compute total flux and pixel counts within elliptical and circular apertures.
    
    This uses a vectorized numpy approach for speed.
    """
    h, w = image.shape
    
    # Bounding box
    x_min = max(0, int(x0 - sma - 1))
    x_max = min(w, int(x0 + sma + 1))
    y_min = max(0, int(y0 - sma - 1))
    y_max = min(h, int(y0 + sma + 1))
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0, 0.0, 0, 0
        
    y, x = np.mgrid[y_min:y_max, x_min:x_max]
    
    # Circular aperture
    r2 = (x - x0)**2 + (y - y0)**2
    mask_c = r2 <= sma**2
    
    # Elliptical aperture
    dx = x - x0
    dy = y - y0
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    sma_pix2 = x_rot**2 + (y_rot / (1.0 - eps))**2
    mask_e = sma_pix2 <= sma**2
    
    # Data extraction
    data = image[y_min:y_max, x_min:x_max]
    if mask is not None:
        mdata = mask[y_min:y_max, x_min:x_max]
        valid = ~mdata
    else:
        valid = np.ones_like(data, dtype=bool)
        
    valid &= ~np.isnan(data)
    
    # Calculate metrics
    tflux_c = np.sum(data[mask_c & valid])
    npix_c = np.sum(mask_c & valid)
    
    tflux_e = np.sum(data[mask_e & valid])
    npix_e = np.sum(mask_e & valid)
    
    return tflux_e, tflux_c, npix_e, npix_c

def fit_isophote(image, mask, sma, start_geometry, config, going_inwards=False, previous_geometry=None):
    """
    Fit a single isophote with quality control.

    This function performs the iterative harmonic fitting for a single semi-major axis (SMA).

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray): Bad pixel mask.
        sma (float): Semi-major axis length to fit.
        start_geometry (dict): Initial guess for {'x0', 'y0', 'eps', 'pa'}.
        config (dict or IsosterConfig): Configuration object.
        going_inwards (bool): Flag indicating if the fitting is progressing inwards (affecting gradient checks).
        previous_geometry (dict): Previous isophote geometry for regularization (optional).

    Returns:
        dict: The best fitted geometry and metadata for this isophote.
    """
    # Normalize configuration
    if isinstance(config, IsosterConfig):
        cfg = config
    else:
        cfg = IsosterConfig(**(config or {}))

    maxit = cfg.maxit
    conver = cfg.conver
    minit = cfg.minit
    astep = cfg.astep
    linear_growth = cfg.linear_growth
    fix_center = cfg.fix_center
    fix_pa = cfg.fix_pa
    fix_eps = cfg.fix_eps
    sclip = cfg.sclip
    nclip = cfg.nclip
    sclip_low = cfg.sclip_low
    sclip_high = cfg.sclip_high
    fflag = cfg.fflag
    maxgerr = cfg.maxgerr
    debug = cfg.debug
    full_photometry = cfg.full_photometry or debug
    use_eccentric_anomaly = cfg.use_eccentric_anomaly
    compute_errors = cfg.compute_errors
    compute_deviations_flag = cfg.compute_deviations
    integrator = cfg.integrator
    lsb_sma_threshold = cfg.lsb_sma_threshold

    
    x0, y0, eps, pa = start_geometry['x0'], start_geometry['y0'], start_geometry['eps'], start_geometry['pa']
    stop_code, niter, best_geometry = 0, 0, None
    min_amplitude, previous_gradient, lexceed = np.inf, None, False
    
    for i in range(maxit):
        niter = i + 1
        phi, intens, radii = extract_isophote_data(image, mask, x0, y0, sma, eps, pa, astep, linear_growth, use_eccentric_anomaly)
        total_points = len(phi)
        phi, intens, n_clipped = sigma_clip(phi, intens, sclip, nclip, sclip_low, sclip_high)
        actual_points = len(phi)
        
        if actual_points < (total_points * fflag):
            if best_geometry is not None:
                best_geometry['stop_code'], best_geometry['niter'] = 1, niter
                return best_geometry
            else:
                res = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma,
                       'intens': np.nan, 'rms': np.nan, 'intens_err': np.nan,
                       'stop_code': 1, 'niter': niter,
                       'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0}
                if debug:
                    res.update({'ndata': actual_points, 'nflag': total_points - actual_points,
                                'grad': np.nan, 'grad_error': np.nan, 'grad_r_error': np.nan})
                return res
        
        if len(intens) < 6:
            stop_code = 3
            break
            
        # Determine effective integrator for this isophote
        eff_integrator = integrator
        if integrator == 'adaptive':
            if lsb_sma_threshold is not None and sma > lsb_sma_threshold:
                eff_integrator = 'median'
            else:
                eff_integrator = 'mean'
            
        coeffs, cov_matrix = fit_first_and_second_harmonics(phi, intens)
        y0_fit, A1, B1, A2, B2 = coeffs
        gradient, gradient_error = compute_gradient(image, mask, x0, y0, sma, eps, pa, astep, linear_growth, 
                                                   previous_gradient, current_data=(phi, intens), integrator=eff_integrator)
        if gradient_error is not None:
            previous_gradient = gradient
        
        gradient_relative_error = abs(gradient_error / gradient) if (gradient_error is not None and gradient < 0) else None
        if not going_inwards:
            if gradient_relative_error is None or gradient_relative_error > maxgerr or gradient >= 0:
                if lexceed:
                    stop_code = -1
                    break
                else:
                    lexceed = True
        
        if gradient == 0:
            stop_code = -1
            break
            
        model = harmonic_function(phi, coeffs)
        rms = np.std(intens - model)
        harmonics = [A1, B1, A2, B2]
        if fix_center: harmonics[0] = harmonics[1] = 0
        if fix_pa: harmonics[2] = 0
        if fix_eps: harmonics[3] = 0
            
        max_idx = np.argmax(np.abs(harmonics))
        max_amp = harmonics[max_idx]
        
        # Apply central region regularization penalty if enabled
        current_geom = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa}
        reg_penalty = compute_central_regularization_penalty(current_geom, previous_geometry, sma, cfg)
        
        # Effective amplitude includes regularization penalty
        # This discourages large geometry changes in the central region
        effective_amp = abs(max_amp) + reg_penalty
        
        if effective_amp < min_amplitude:
            min_amplitude = effective_amp
            intens_err = rms / np.sqrt(len(intens))
            x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(phi, intens, x0, y0, sma, eps, pa, gradient, cov_matrix) if compute_errors else (0.0, 0.0, 0.0, 0.0)
            if eff_integrator == 'median':
                reported_intens = np.median(intens)
            else:
                reported_intens = y0_fit
                
            best_geometry = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma, 'intens': reported_intens, 'rms': rms, 'intens_err': intens_err,
                             'x0_err': x0_err, 'y0_err': y0_err, 'eps_err': eps_err, 'pa_err': pa_err,
                             'a3': 0.0, 'b3': 0.0, 'a3_err': 0.0, 'b3_err': 0.0, 'a4': 0.0, 'b4': 0.0, 'a4_err': 0.0, 'b4_err': 0.0,
                             'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0}
            if debug:
                best_geometry.update({'ndata': actual_points, 'nflag': total_points - actual_points, 'grad': gradient,
                                      'grad_error': gradient_error if gradient_error is not None else np.nan,
                                      'grad_r_error': gradient_relative_error if gradient_relative_error is not None else np.nan})
            
        if abs(max_amp) < conver * rms and i >= minit:
            stop_code = 0
            # Already updated best_geometry in min_amplitude check, but let's ensure deviations
            if compute_deviations_flag:
                a3, b3, a3_err, b3_err = compute_deviations(phi, intens, sma, gradient, 3)
                a4, b4, a4_err, b4_err = compute_deviations(phi, intens, sma, gradient, 4)
                best_geometry.update({'a3': a3, 'b3': b3, 'a3_err': a3_err, 'b3_err': b3_err,
                                      'a4': a4, 'b4': b4, 'a4_err': a4_err, 'b4_err': b4_err})
            
            # 6. FULL PHOTOMETRY (If requested)
            if full_photometry:
                tflux_e, tflux_c, npix_e, npix_c = compute_aperture_photometry(image, mask, x0, y0, sma, eps, pa)
                best_geometry.update({
                    'tflux_e': tflux_e, 'tflux_c': tflux_c,
                    'npix_e': npix_e, 'npix_c': npix_c
                })
            break
            
        # Update geometry
        if max_idx == 0:
            aux = -max_amp * (1.0 - eps) / gradient
            x0 -= aux * np.sin(pa)
            y0 += aux * np.cos(pa)
        elif max_idx == 1:
            aux = -max_amp / gradient
            x0 += aux * np.cos(pa)
            y0 += aux * np.sin(pa)
        elif max_idx == 2:
            denom = ((1.0 - eps)**2 - 1.0)
            if denom == 0: denom = 1e-6
            pa = (pa + (max_amp * 2.0 * (1.0 - eps) / sma / gradient / denom)) % np.pi
        elif max_idx == 3:
            eps = min(eps - (max_amp * 2.0 * (1.0 - eps) / sma / gradient), 0.95)
            if eps < 0.0:
                eps = min(-eps, 0.95)
                pa = (pa + np.pi/2) % np.pi
            if eps == 0.0: eps = 0.05
            
    if best_geometry is None:
        best_geometry = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma, 'intens': np.nan, 'rms': np.nan, 'intens_err': np.nan,
                         'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
                         'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
                         'a3': 0.0, 'b3': 0.0, 'a3_err': 0.0, 'b3_err': 0.0, 'a4': 0.0, 'b4': 0.0, 'a4_err': 0.0, 'b4_err': 0.0}
        if debug: best_geometry.update({'ndata': 0, 'nflag': 0, 'grad': np.nan, 'grad_error': np.nan, 'grad_r_error': np.nan})
        
    best_geometry['stop_code'], best_geometry['niter'] = stop_code, niter
    return best_geometry
