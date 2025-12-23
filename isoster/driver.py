import numpy as np
from scipy.ndimage import map_coordinates
from .fitting import fit_isophote

def fit_central_pixel(image, mask, x0, y0, debug=False):
    """Fit the central pixel (SMA=0)."""
    coords = np.array([[y0], [x0]])
    intens = map_coordinates(image, coords, order=1, mode='constant', cval=np.nan)[0]
    
    valid = True
    if mask is not None:
        mval = map_coordinates(mask.astype(float), coords, order=0, mode='constant', cval=1.0)[0]
        if mval > 0.5: valid = False
            
    if np.isnan(intens): valid = False
        
    res = {
        'x0': x0, 'y0': y0, 'eps': 0.0, 'pa': 0.0, 'sma': 0.0,
        'intens': intens, 'rms': 0.0, 'intens_err': 0.0,
        'x0_err': 0.0, 'y0_err': 0.0, 'eps_err': 0.0, 'pa_err': 0.0,
        'a3': 0.0, 'b3': 0.0, 'a3_err': 0.0, 'b3_err': 0.0,
        'a4': 0.0, 'b4': 0.0, 'a4_err': 0.0, 'b4_err': 0.0,
        'tflux_e': np.nan, 'tflux_c': np.nan, 'npix_e': 0, 'npix_c': 0,
        'stop_code': 0 if valid else -1,
        'niter': 0, 'valid': valid
    }
    
    if debug:
        res.update({'ndata': 1 if valid else 0, 'nflag': 0, 'grad': 0.0, 'grad_error': 0.0, 'grad_r_error': 0.0})
        
    return res

def fit_image(image, mask, config):
    """Main driver to fit isophotes to an image."""
    x0, y0 = config.get('x0', image.shape[1] / 2.0), config.get('y0', image.shape[0] / 2.0)
    sma0, minsma = config.get('sma0', 10.0), config.get('minsma', 0.0)
    maxsma = config.get('maxsma', max(image.shape) / 2.0)
    astep, linear_growth = config.get('astep', 0.1), config.get('linear_growth', False)
    
    results = []
    # Outwards loop
    sma, current_geometry, first_isophote = sma0, {'x0': x0, 'y0': y0, 'eps': config.get('eps', 0.2), 'pa': config.get('pa', 0.0)}, True
    while sma <= maxsma:
        curr_config = config.copy()
        if first_isophote:
            curr_config['minit'] = config.get('minit', 10) * 2
            first_isophote = False
        res = fit_isophote(image, mask, sma, current_geometry, curr_config, going_inwards=False)
        results.append(res)
        if res['stop_code'] in [0, 1, 2]: current_geometry = res.copy()
        if linear_growth: sma += astep
        else: sma *= (1.0 + astep)
            
    # Inwards loop
    sma, current_geometry = (sma0 - astep if linear_growth else sma0 / (1.0 + astep)), {'x0': x0, 'y0': y0, 'eps': config.get('eps', 0.2), 'pa': config.get('pa', 0.0)}
    inwards_results, min_iter_sma = [], max(minsma, 0.5)
    while sma >= min_iter_sma:
        res = fit_isophote(image, mask, sma, current_geometry, config, going_inwards=True)
        inwards_results.append(res)
        if res['stop_code'] in [0, 1, 2]: current_geometry = res.copy()
        if linear_growth: sma -= astep
        else: sma = sma / (1.0 + astep)
            
    if minsma <= 0.0:
        inwards_results.append(fit_central_pixel(image, mask, current_geometry['x0'], current_geometry['y0'], debug=config.get('debug', False)))
            
    return {'isophotes': inwards_results[::-1] + results, 'config': config}
