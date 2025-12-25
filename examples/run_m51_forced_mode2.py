"""
M51 Forced Mode Benchmark - Method 2
=====================================

Tests pure forced photometry mode (no fitting, just sampling).

Workflow:
1. Run regular isoster fit to get SMA array and global geometry
2. Run forced mode with predetermined SMA and geometry
3. Compare runtime and results
"""

import os
import sys
import time
import numpy as np
from astropy.io import fits
from scipy import stats
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig

def compute_global_geometry(isophotes, sma_min=2.0, sma_max=150.0, sigma=3.0):
    """Compute global geometry from isophote results using sigma-clipped median."""
    sma_arr = np.array([iso['sma'] for iso in isophotes])
    mask = (sma_arr >= sma_min) & (sma_arr <= sma_max)
    
    x0_arr = np.array([iso['x0'] for iso in isophotes])[mask]
    y0_arr = np.array([iso['y0'] for iso in isophotes])[mask]
    eps_arr = np.array([iso['eps'] for iso in isophotes])[mask]
    pa_arr = np.array([iso['pa'] for iso in isophotes])[mask]
    
    x0_global = np.median(stats.sigmaclip(x0_arr, low=sigma, high=sigma).clipped)
    y0_global = np.median(stats.sigmaclip(y0_arr, low=sigma, high=sigma).clipped)
    eps_global = np.median(stats.sigmaclip(eps_arr, low=sigma, high=sigma).clipped)
    pa_global = np.median(stats.sigmaclip(pa_arr, low=sigma, high=sigma).clipped)
    
    return {
        'x0': x0_global,
        'y0': y0_global,
        'eps': eps_global,
        'pa': pa_global
    }

def run_benchmark():
    print("=" * 80)
    print("ISOSTER M51 Forced Mode Benchmark - Method 2 (Pure Forced)")
    print("=" * 80)
    
    # Load Data
    fits_path = os.path.join(os.path.dirname(__file__), 'M51.fits')
    if not os.path.exists(fits_path):
        print(f"Error: {fits_path} not found.")
        return

    with fits.open(fits_path) as hdul:
        image = hdul[0].data
        
    h, w = image.shape
    
    # Generate mask
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
    # 1. Run Regular Mode (for reference)
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
    
    # Extract SMA array
    sma_array = [iso['sma'] for iso in regular_isophotes]
    
    # ---------------------------------------------------------
    # 2. Compute Global Geometry
    # ---------------------------------------------------------
    print("\n[Computing Global Geometry]")
    global_geom = compute_global_geometry(regular_isophotes, sma_min=2.0, sma_max=150.0, sigma=3.0)
    
    print(f"   X0: {global_geom['x0']:.2f}")
    print(f"   Y0: {global_geom['y0']:.2f}")
    print(f"   PA: {np.degrees(global_geom['pa']):.2f} deg")
    print(f"   eps: {global_geom['eps']:.4f}")
    
    # ---------------------------------------------------------
    # 3. Run Pure Forced Mode
    # ---------------------------------------------------------
    print("\n[Pure Forced Mode] Running extraction...")
    
    cfg_forced = IsosterConfig(
        x0=global_geom['x0'], y0=global_geom['y0'],
        eps=global_geom['eps'], pa=global_geom['pa'],
        integrator='mean',
        forced=True,
        forced_sma=sma_array
    )
    
    t0 = time.time()
    forced_results = fit_image(image, mask=mask, config=cfg_forced)
    t1 = time.time()
    forced_time = t1 - t0
    
    forced_isophotes = forced_results['isophotes']
    print(f"   Done in {forced_time:.4f}s. Extracted {len(forced_isophotes)} isophotes.")
    
    # ---------------------------------------------------------
    # 4. Performance Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Performance Summary - Method 2 (Pure Forced)")
    print("-" * 80)
    print(f"Regular Mode:       {regular_time:.4f}s")
    print(f"Pure Forced:        {forced_time:.4f}s")
    print(f"Speedup:            {regular_time/forced_time:.2f}x")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # 5. Generate Comparison Plot
    # ---------------------------------------------------------
    print("\nGenerating comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Extract data
    reg_sma = np.array([iso['sma'] for iso in regular_isophotes])
    reg_intens = np.array([iso['intens'] for iso in regular_isophotes])
    forced_sma = np.array([iso['sma'] for iso in forced_isophotes])
    forced_intens = np.array([iso['intens'] for iso in forced_isophotes])
    
    # Plot 1: Intensity profiles
    ax1.plot(reg_sma**0.25, np.log10(reg_intens), 'o-', label='Regular', markersize=4)
    ax1.plot(forced_sma**0.25, np.log10(forced_intens), 's-', label='Forced', markersize=3, alpha=0.7)
    ax1.set_ylabel(r'$\log_{10}(I)$', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'M51 Forced Mode Comparison (Speedup: {regular_time/forced_time:.1f}x)', fontsize=14, weight='bold')
    
    # Plot 2: Fractional difference
    # Interpolate to common grid
    forced_interp = np.interp(reg_sma, forced_sma, forced_intens)
    frac_diff = (reg_intens - forced_interp) / reg_intens * 100
    
    ax2.plot(reg_sma**0.25, frac_diff, 'o-', markersize=4)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel('Fractional Diff (%)', fontsize=12)
    ax2.set_xlabel(r'SMA$^{0.25}$ (pixel$^{0.25}$)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    qa_path = os.path.join(os.path.dirname(__file__), 'm51_forced_comparison.png')
    plt.savefig(qa_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {qa_path}")
    
    return {
        'regular_time': regular_time,
        'forced_time': forced_time,
        'regular_isophotes': regular_isophotes,
        'forced_isophotes': forced_isophotes,
        'global_geom': global_geom
    }

if __name__ == "__main__":
    results = run_benchmark()
