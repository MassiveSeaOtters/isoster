"""
Sersic Profile CoG Test
=======================

Generate a mock Sersic profile and verify CoG photometry against analytical total flux.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.special import gammaincinv, gamma

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig

def sersic_profile(r, I_e, r_e, n):
    """
    Sersic profile: I(r) = I_e * exp(-b_n * ((r/r_e)^(1/n) - 1))
    
    where b_n is chosen such that r_e contains half the total light.
    """
    b_n = gammaincinv(2*n, 0.5)
    return I_e * np.exp(-b_n * ((r / r_e)**(1.0/n) - 1.0))

def compute_aperture_cog(image, x0, y0, sma_array, eps, pa):
    """
    Compute true CoG by performing elliptical aperture photometry on the image.
    
    This gives us the ground truth for comparison with isoster's CoG.
    
    Args:
        image: 2D array
        x0, y0: Center coordinates
        sma_array: Array of semi-major axes
        eps: Ellipticity
        pa: Position angle in radians
        
    Returns:
        cog_true: Array of cumulative flux within each ellipse
    """
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # Rotate coordinates
    dx = x - x0
    dy = y - y0
    
    x_rot = dx * np.cos(pa) + dy * np.sin(pa)
    y_rot = -dx * np.sin(pa) + dy * np.cos(pa)
    
    # Elliptical radius for each pixel
    b_over_a = 1.0 - eps
    r_ellipse = np.sqrt(x_rot**2 + (y_rot / b_over_a)**2)
    
    # Compute CoG for each SMA
    cog_true = np.zeros(len(sma_array))
    for i, sma in enumerate(sma_array):
        # Sum all pixels within this ellipse
        mask = r_ellipse <= sma
        cog_true[i] = np.sum(image[mask])
    
    return cog_true

def generate_sersic_image(size=512, I_e=1000.0, r_e=50.0, n=4.0, eps=0.2, pa=30.0):
    """
    Generate a 2D Sersic profile image.
    
    Args:
        size: Image size (square)
        I_e: Intensity at effective radius
        r_e: Effective radius
        n: Sersic index
        eps: Ellipticity
        pa: Position angle in degrees
        
    Returns:
        image: 2D array
        params: dict with true parameters
    """
    y, x = np.mgrid[0:size, 0:size]
    x0, y0 = size / 2.0, size / 2.0
    
    # Rotate coordinates
    pa_rad = np.radians(pa)
    dx = x - x0
    dy = y - y0
    
    x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)
    
    # Elliptical radius
    b_over_a = 1.0 - eps
    r = np.sqrt(x_rot**2 + (y_rot / b_over_a)**2)
    
    # Sersic profile
    image = sersic_profile(r, I_e, r_e, n)
    
    params = {
        'x0': x0,
        'y0': y0,
        'I_e': I_e,
        'r_e': r_e,
        'n': n,
        'eps': eps,
        'pa': pa_rad
    }
    
    return image, params

def run_sersic_cog_test():
    print("=" * 80)
    print("Sersic Profile CoG Test")
    print("=" * 80)
    
    # Generate Sersic profile
    print("\nGenerating Sersic profile...")
    size = 512
    I_e = 1000.0
    r_e = 50.0
    n = 4.0
    eps = 0.2
    pa = 30.0
    
    image, params = generate_sersic_image(size, I_e, r_e, n, eps, pa)
    print(f"   Size: {size}x{size}")
    print(f"   I_e: {I_e}, r_e: {r_e}, n: {n}")
    print(f"   eps: {eps}, PA: {pa}°")
    
    # Run isoster with CoG
    print("\nRunning isoster with CoG...")
    
    cfg = IsosterConfig(
        x0=params['x0'], y0=params['y0'],
        sma0=5.0, minsma=0.0, maxsma=200.0, astep=0.05,
        eps=eps, pa=params['pa'],
        conver=0.05, maxit=50,
        compute_errors=False,
        compute_deviations=False,
        compute_cog=True,  # Enable CoG
        # Fix geometry for clean CoG
        fix_center=True,
        fix_pa=True,
        fix_eps=True
    )
    
    results = fit_image(image, mask=None, config=cfg)
    isophotes = results['isophotes']
    
    print(f"   Fitted {len(isophotes)} isophotes")
    
    # Extract CoG data
    sma_arr = np.array([iso['sma'] for iso in isophotes])
    cog_arr = np.array([iso['cog'] for iso in isophotes])
    flag_cross = np.array([iso.get('flag_cross', False) for iso in isophotes])
    flag_neg = np.array([iso.get('flag_negative_area', False) for iso in isophotes])
    
    print(f"   Crossing flags: {np.sum(flag_cross)} / {len(flag_cross)}")
    print(f"   Negative area flags: {np.sum(flag_neg)} / {len(flag_neg)}")
    
    # Compute true CoG using aperture photometry on the mock image
    print(f"\nComputing true CoG via aperture photometry...")
    cog_true = compute_aperture_cog(image, params['x0'], params['y0'], 
                                     sma_arr, params['eps'], params['pa'])
    
    # Compare at maximum SMA
    flux_true = cog_true[-1]
    flux_cog = cog_arr[-1]
    
    print(f"   True flux (aperture): {flux_true:.2e}")
    print(f"   CoG flux (isoster):   {flux_cog:.2e}")
    
    frac_error = abs(flux_cog - flux_true) / flux_true * 100
    print(f"   Fractional error: {frac_error:.3f}%")
    
    # Verification
    if frac_error < 1.0:
        print("\n✓ PASS: CoG matches aperture photometry within 1%")
    else:
        print(f"\n✗ FAIL: CoG error {frac_error:.3f}% exceeds 1% threshold")
    
    # Generate plot
    print("\nGenerating CoG plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    x_axis = sma_arr**0.25
    
    # Plot 1: CoG comparison
    ax1.plot(x_axis, np.log10(cog_arr), 'o-', markersize=4, label='Isoster CoG', color='blue')
    ax1.plot(x_axis, np.log10(cog_true), 's-', markersize=3, label='True CoG (aperture)', 
             color='red', alpha=0.7)
    
    # Mark maximum
    max_idx = np.argmax(cog_arr)
    ax1.plot(x_axis[max_idx], np.log10(cog_arr[max_idx]), 'b*', 
             markersize=15, label=f'Max at SMA={sma_arr[max_idx]:.1f}')
    
    ax1.set_ylabel(r'$\log_{10}$(CoG)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Sersic n={n} CoG Test (Error: {frac_error:.3f}%)', 
                  fontsize=14, weight='bold')
    
    # Plot 2: Fractional error
    frac_err_arr = (cog_arr - cog_true) / cog_true * 100
    ax2.plot(x_axis, frac_err_arr, 'o-', markersize=4)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.axhline(1, color='red', linestyle=':', alpha=0.5, label='±1% threshold')
    ax2.axhline(-1, color='red', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Fractional Error (%)', fontsize=14)
    ax2.set_xlabel(r'SMA$^{0.25}$ (pixel$^{0.25}$)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    qa_path = os.path.join(os.path.dirname(__file__), 'sersic_cog_test.png')
    plt.savefig(qa_path, dpi=150)
    plt.close()
    print(f"Saved plot to {qa_path}")
    
    return {
        'flux_true': flux_true,
        'flux_cog': flux_cog,
        'frac_error': frac_error,
        'isophotes': isophotes,
        'cog_true': cog_true
    }

if __name__ == "__main__":
    results = run_sersic_cog_test()
