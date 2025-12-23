"""
M51 Baseline Example
====================

This script performs isophote fitting on the M51 galaxy image using both
ISOSTER (optimized) and photutils.isophote (standard).

It demonstrate:
1. Command-line output of fitting progress.
2. Saving the resulting isophote profile to a FITS file.
3. Generating a comparison plot between the two implementations.

Requirements:
- photutils
- astropy
- matplotlib
- numpy
- scipy
"""

import os
import sys
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from photutils.isophote import Ellipse, EllipseGeometry
from isoster.optimize import fit_image
from isoster.utils import isophote_results_to_fits
from isoster.plotting import plot_isophote_comparison

def run_m51_baseline():
    print("=" * 80)
    print("ISOSTER M51 Baseline Example")
    print("=" * 80)
    
    fits_path = os.path.join(os.path.dirname(__file__), 'M51.fits')
    
    if not os.path.exists(fits_path):
        print(f"Error: {fits_path} not found.")
        print("Attempting to download...")
        from photutils.datasets import get_path
        from shutil import copyfile
        try:
            path = get_path('isophote/M51.fits', location='photutils-datasets', cache=True)
            copyfile(path, fits_path)
            print(f"Downloaded M51.fits to {fits_path}")
        except Exception as e:
            print(f"Failed to download M51.fits: {e}")
            return

    # Load data
    with fits.open(fits_path) as hdul:
        image = hdul[0].data
    
    print(f"Image loaded: {image.shape}")
    
    # Configuration
    # We use similar settings for both to ensure a fair comparison
    x0, y0 = image.shape[1] / 2.0, image.shape[0] / 2.0
    sma0 = 10.0
    minsma = 0.0
    maxsma = 100.0 # Limit for quick example
    astep = 0.1
    
    config = {
        'x0': x0, 'y0': y0, 'sma0': sma0,
        'minsma': minsma, 'maxsma': maxsma,
        'astep': astep,
        'eps': 0.2, 'pa': 0.0,
        'conver': 0.05,
        'maxit': 50,
        'compute_errors': True
    }
    
    print("\n1. Running photutils.isophote (Standard)...")
    geometry = EllipseGeometry(x0=x0, y0=y0, sma=sma0, eps=0.2, pa=0.0)
    ellipse = Ellipse(image, geometry=geometry)
    
    t0 = time.time()
    # photutils fit_image
    p_isolist = ellipse.fit_image(sma0=sma0, minsma=minsma, maxsma=maxsma, step=astep)
    t1 = time.time()
    p_time = t1 - t0
    print(f"   Completed in {p_time:.4f}s")
    print(f"   Isophotes fitted: {len(p_isolist)}")
    
    print("\n2. Running ISOSTER (Optimized)...")
    print(f"{'SMA':>10} | {'Intens':>10} | {'Eps':>8} | {'PA (deg)':>10} | {'Iter':>5} | {'Status'}")
    print("-" * 70)
    
    t0 = time.time()
    # We call fit_isophote manually or iterate over results to show progress
    # For simplicity in this script, we'll run fit_image and THEN print some progress-like output
    # since fit_image is what we want to benchmark.
    i_results = fit_image(image, None, config)
    t1 = time.time()
    i_time = t1 - t0
    
    # Print progress-like output from results
    for iso in i_results['isophotes'][::5]: # Show every 5th
        print(f"{iso['sma']:10.2f} | {iso['intens']:10.2f} | {iso['eps']:8.3f} | {np.degrees(iso['pa']):10.2f} | {iso['niter']:5} | {iso['stop_code']}")
    
    print("-" * 70)
    print(f"   Completed in {i_time:.4f}s")
    print(f"   Isophotes fitted: {len(i_results['isophotes'])}")
    print(f"\n   Speedup: {p_time / i_time:.2f}x")
    
    # Save Results
    fits_output = os.path.join(os.path.dirname(__file__), 'm51_isophotes.fits')
    isophote_results_to_fits(i_results, fits_output)
    print(f"\nSaved ISOSTER results to: {fits_output}")
    
    # Plot Comparison
    plot_output = os.path.join(os.path.dirname(__file__), 'm51_comparison.png')
    # Filter valid results for plotting
    valid_optimized = [r for r in i_results['isophotes'] if r['stop_code'] in [0, 1, 2]]
    plot_isophote_comparison("M51 Isophote Comparison: Standard vs Optimized", 
                             p_isolist, valid_optimized, plot_output, image)
    print(f"Saved comparison plot to: {plot_output}")

if __name__ == "__main__":
    run_m51_baseline()
