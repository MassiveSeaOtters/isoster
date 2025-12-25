"""
M51 Forced Mode Benchmark - Method 1
=====================================

Tests forced photometry using existing fix_center, fix_pa, fix_eps flags.

Workflow:
1. Run regular isoster fit
2. Compute global geometry (3-sigma clipped median in 2 < SMA < 150)
3. Re-run with fixed geometry
4. Compare runtime
"""

import os
import sys
import time
import numpy as np
from astropy.io import fits
from scipy import stats

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig

def compute_global_geometry(isophotes, sma_min=2.0, sma_max=150.0, sigma=3.0):
    """
    Compute global geometry from isophote results using sigma-clipped median.
    
    Parameters
    ----------
    isophotes : list of dict
        Isophote fitting results
    sma_min, sma_max : float
        SMA range for computing global values
    sigma : float
        Sigma clipping threshold
        
    Returns
    -------
    dict : Global geometry parameters
    """
    # Extract data in SMA range
    sma_arr = np.array([iso['sma'] for iso in isophotes])
    mask = (sma_arr >= sma_min) & (sma_arr <= sma_max)
    
    x0_arr = np.array([iso['x0'] for iso in isophotes])[mask]
    y0_arr = np.array([iso['y0'] for iso in isophotes])[mask]
    eps_arr = np.array([iso['eps'] for iso in isophotes])[mask]
    pa_arr = np.array([iso['pa'] for iso in isophotes])[mask]
    
    # Sigma-clipped median
    x0_global = np.median(stats.sigmaclip(x0_arr, low=sigma, high=sigma).clipped)
    y0_global = np.median(stats.sigmaclip(y0_arr, low=sigma, high=sigma).clipped)
    eps_global = np.median(stats.sigmaclip(eps_arr, low=sigma, high=sigma).clipped)
    pa_global = np.median(stats.sigmaclip(pa_arr, low=sigma, high=sigma).clipped)
    
    return {
        'x0': x0_global,
        'y0': y0_global,
        'eps': eps_global,
        'pa': pa_global,
        'ba': 1.0 - eps_global
    }

def run_benchmark():
    print("=" * 80)
    print("ISOSTER M51 Forced Mode Benchmark - Method 1")
    print("=" * 80)
    
    # 1. Load Data
    fits_path = os.path.join(os.path.dirname(__file__), 'M51.fits')
    if not os.path.exists(fits_path):
        print(f"Error: {fits_path} not found.")
        try:
            from photutils.datasets import get_path
            from shutil import copyfile
            path = get_path('isophote/M51.fits', location='photutils-datasets', cache=True)
            copyfile(path, fits_path)
            print(f"Downloaded M51.fits")
        except:
            print("Could not download M51.fits. Please provide file.")
            return

    with fits.open(fits_path) as hdul:
        image = hdul[0].data
        
    h, w = image.shape
    
    # Generate mask (same as main benchmark)
    mask = np.zeros_like(image, dtype=bool)
    rng = np.random.RandomState(42)
    cx, cy = w/2, h/2
    for _ in range(20):
        while True:
            mx, my = rng.randint(0, w), rng.randint(0, h)
            dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
            if dist > 10:
                break
        r_blob = rng.randint(10, 30)
        y, x = np.mgrid[0:h, 0:w]
        blob = ((x - mx)**2 + (y - my)**2) < r_blob**2
        mask |= blob
    
    print(f"Image: {w}x{h}, Masked pixels: {np.sum(mask)}")
    
    # Shared parameters
    sma0 = 10.0
    minsma = 0.0
    maxsma = 275.0
    astep = 0.1
    
    # ---------------------------------------------------------
    # 2. Run Regular Mode
    # ---------------------------------------------------------
    print("\n[Regular Mode] Running fit...")
    
    cfg_regular = IsosterConfig(
        x0=w/2, y0=h/2,
        sma0=sma0, minsma=minsma, maxsma=maxsma, astep=astep,
        eps=0.2, pa=0.0,
        conver=0.05, maxit=50,
        compute_errors=True,
        compute_deviations=True,
        integrator='adaptive',
        lsb_sma_threshold=100.0
    )
    
    t0 = time.time()
    regular_results = fit_image(image, mask=mask, config=cfg_regular)
    t1 = time.time()
    regular_time = t1 - t0
    
    regular_isophotes = regular_results['isophotes']
    print(f"   Done in {regular_time:.4f}s. Fitted {len(regular_isophotes)} isophotes.")
    
    # ---------------------------------------------------------
    # 3. Compute Global Geometry
    # ---------------------------------------------------------
    print("\n[Computing Global Geometry]")
    global_geom = compute_global_geometry(regular_isophotes, sma_min=2.0, sma_max=150.0, sigma=3.0)
    
    print(f"   X0: {global_geom['x0']:.2f}")
    print(f"   Y0: {global_geom['y0']:.2f}")
    print(f"   PA: {np.degrees(global_geom['pa']):.2f} deg")
    print(f"   b/a: {global_geom['ba']:.4f}")
    print(f"   eps: {global_geom['eps']:.4f}")
    
    # ---------------------------------------------------------
    # 4. Run Fixed Geometry Mode
    # ---------------------------------------------------------
    print("\n[Fixed Geometry Mode] Running fit...")
    
    cfg_fixed = IsosterConfig(
        x0=global_geom['x0'], y0=global_geom['y0'],
        sma0=sma0, minsma=minsma, maxsma=maxsma, astep=astep,
        eps=global_geom['eps'], pa=global_geom['pa'],
        conver=0.05, maxit=50,
        compute_errors=True,
        compute_deviations=True,
        integrator='adaptive',
        lsb_sma_threshold=100.0,
        # Fix geometry
        fix_center=True,
        fix_pa=True,
        fix_eps=True
    )
    
    t0 = time.time()
    fixed_results = fit_image(image, mask=mask, config=cfg_fixed)
    t1 = time.time()
    fixed_time = t1 - t0
    
    fixed_isophotes = fixed_results['isophotes']
    print(f"   Done in {fixed_time:.4f}s. Fitted {len(fixed_isophotes)} isophotes.")
    
    # ---------------------------------------------------------
    # 5. Performance Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Performance Summary - Method 1 (Fixed Geometry)")
    print("-" * 80)
    print(f"Regular Mode:       {regular_time:.4f}s")
    print(f"Fixed Geometry:     {fixed_time:.4f}s")
    print(f"Speedup:            {regular_time/fixed_time:.2f}x")
    print("=" * 80)
    
    return {
        'regular_time': regular_time,
        'fixed_time': fixed_time,
        'regular_isophotes': regular_isophotes,
        'fixed_isophotes': fixed_isophotes,
        'global_geom': global_geom
    }

if __name__ == "__main__":
    results = run_benchmark()
