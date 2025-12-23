import numpy as np
from scipy.optimize import leastsq
from .sampling import extract_isophote_data

def fit_first_and_second_harmonics(phi, intensity):
    """
    Fit the 1st and 2nd harmonics to the intensity profile.
    y = y0 + A1*sin(E) + B1*cos(E) + A2*sin(2E) + B2*cos(2E)
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

def compute_gradient(image, mask, x0, y0, sma, eps, pa, step=0.1, linear_growth=False, previous_gradient=None, current_data=None):
    """Compute the radial intensity gradient."""
    if current_data is not None:
        phi_c, intens_c = current_data
    else:
        phi_c, intens_c, _ = extract_isophote_data(image, mask, x0, y0, sma, eps, pa, step, linear_growth)
    
    if len(intens_c) == 0:
        return previous_gradient * 0.8 if previous_gradient else -1.0, None
        
    mean_c = np.mean(intens_c)
    
    if linear_growth:
        gradient_sma = sma + step
    else:
        gradient_sma = sma * (1.0 + step)
        
    phi_g, intens_g, _ = extract_isophote_data(image, mask, x0, y0, gradient_sma, eps, pa, step, linear_growth)
    
    if len(intens_g) == 0:
        return previous_gradient * 0.8 if previous_gradient else -1.0, None
        
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

def fit_isophote(image, mask, sma, start_geometry, config, going_inwards=False):
    """Fit a single isophote with quality control."""
    maxit = config.get('maxit', 50)
    conver = config.get('conver', 0.05)
    minit = config.get('minit', 10)
    astep = config.get('astep', 0.1)
    linear_growth = config.get('linear_growth', False)
    fix_center = config.get('fix_center', False)
    fix_pa = config.get('fix_pa', False)
    fix_eps = config.get('fix_eps', False)
    sclip = config.get('sclip', 3.0)
    nclip = config.get('nclip', 0)
    sclip_low = config.get('sclip_low', None)
    sclip_high = config.get('sclip_high', None)
    fflag = config.get('fflag', 0.5)
    maxgerr = config.get('maxgerr', 0.5)
    debug = config.get('debug', False)
    full_photometry = config.get('full_photometry', False) or debug
    compute_errors = config.get('compute_errors', True)
    compute_deviations_flag = config.get('compute_deviations', True)
    
    x0, y0, eps, pa = start_geometry['x0'], start_geometry['y0'], start_geometry['eps'], start_geometry['pa']
    stop_code, niter, best_geometry = 0, 0, None
    min_amplitude, previous_gradient, lexceed = np.inf, None, False
    
    for i in range(maxit):
        niter = i + 1
        phi, intens, radii = extract_isophote_data(image, mask, x0, y0, sma, eps, pa, astep, linear_growth)
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
            
        coeffs, cov_matrix = fit_first_and_second_harmonics(phi, intens)
        y0_fit, A1, B1, A2, B2 = coeffs
        gradient, gradient_error = compute_gradient(image, mask, x0, y0, sma, eps, pa, astep, linear_growth, 
                                                   previous_gradient, current_data=(phi, intens))
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
        
        if abs(max_amp) < min_amplitude:
            min_amplitude = abs(max_amp)
            intens_err = rms / np.sqrt(len(intens))
            x0_err, y0_err, eps_err, pa_err = compute_parameter_errors(phi, intens, x0, y0, sma, eps, pa, gradient, cov_matrix) if compute_errors else (0.0, 0.0, 0.0, 0.0)
            best_geometry = {'x0': x0, 'y0': y0, 'eps': eps, 'pa': pa, 'sma': sma, 'intens': y0_fit, 'rms': rms, 'intens_err': intens_err,
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
