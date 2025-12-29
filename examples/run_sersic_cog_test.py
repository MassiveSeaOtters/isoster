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

def sersic_analytical_cog(I_e, r_e, n, eps, sma_array):
    """
    Analytical CoG for elliptical Sersic profile.
    
    For an elliptical Sersic profile, the total flux within an ellipse
    of semi-major axis 'a' is given by integrating over the elliptical area.
    
    We use the fact that for an ellipse with semi-major axis a and ellipticity eps:
    Area = π * a * b = π * a^2 * (1 - eps)
    
    The cumulative flux is computed by numerical integration of the Sersic profile
    over elliptical annuli.
    """
    from scipy.integrate import quad
    from scipy.special import gammaincinv
    
    b_n = gammaincinv(2*n, 0.5)
    b_over_a = 1.0 - eps
    
    cog_analytical = np.zeros(len(sma_array))
    
    for i, sma in enumerate(sma_array):
        # Integrate over elliptical radius from 0 to sma
        # Using substitution: r_ellipse = sqrt(x^2 + (y/b_a)^2)
        # The area element in elliptical coordinates is: dA = b_a * r * dr * dtheta
        
        def integrand(r):
            # Sersic profile at elliptical radius r
            I_r = I_e * np.exp(-b_n * ((r / r_e)**(1.0/n) - 1.0))
            # Area element for ellipse: 2π * r * b_over_a
            return 2 * np.pi * r * b_over_a * I_r
        
        flux, _ = quad(integrand, 0, sma, limit=100)
        cog_analytical[i] = flux
    
    return cog_analytical

def compute_aperture_cog(image, x0, y0, sma_array, eps, pa_deg):
    """
    Compute accurate CoG using SEP elliptical aperture photometry with subpixel method.
    
    This properly handles fractional pixel coverage in the central region.
    Uses SEP library which supports vectorized elliptical apertures.
    
    Args:
        image: 2D array
        x0, y0: Center coordinates
        sma_array: Array of semi-major axes
        eps: Ellipticity (1 - b/a)
        pa_deg: Position angle in degrees
        
    Returns:
        cog_aperture: Array of cumulative flux within each ellipse
    """
    import sep
    
    # SEP requires C-contiguous array with native byte order
    image_sep = np.ascontiguousarray(image, dtype=np.float64)
    
    # Convert ellipticity to axis ratio
    # eps = 1 - b/a, so b/a = 1 - eps
    b_over_a = 1.0 - eps
    
    # SEP uses semi-minor axis b, not ellipticity
    b_array = sma_array * b_over_a
    
    # SEP position angle convention: radians, counter-clockwise from x-axis
    theta = np.radians(pa_deg)
    
    # Create arrays for all apertures (SEP supports vectorized operations)
    x_array = np.full_like(sma_array, x0)
    y_array = np.full_like(sma_array, y0)
    theta_array = np.full_like(sma_array, theta)
    
    # Perform elliptical aperture photometry with subpixel method
    # sep.sum_ellipse(data, x, y, a, b, theta, r=1.0, err=None, gain=None, 
    #                 mask=None, maskthresh=0.0, bkgann=None, subpix=5)
    flux, fluxerr, flag = sep.sum_ellipse(
        image_sep, 
        x_array, y_array,  # Center positions
        sma_array, b_array,  # Semi-major and semi-minor axes
        theta_array,  # Position angles
        subpix=9  # Subpixel sampling (9x9 grid per pixel)
    )
    
    return flux

def generate_sersic_image(size=512, I_e=1000.0, r_e=50.0, n=4.0, eps=0.2, pa=30.0, oversample=10):
    """
    Generate a 2D Sersic profile image with proper oversampling.
    
    For steep profiles (high n), simple pixel-center sampling is inadequate.
    This function renders the model on an oversampled grid and rebins to
    the final pixel grid for accurate flux conservation.
    
    Args:
        size: Image size (square)
        I_e: Intensity at effective radius
        r_e: Effective radius
        n: Sersic index
        eps: Ellipticity
        pa: Position angle in degrees
        oversample: Oversampling factor (default: 10)
        
    Returns:
        image: 2D array
        params: dict with true parameters
    """
    # Create oversampled grid
    size_over = size * oversample
    y_over, x_over = np.mgrid[0:size_over, 0:size_over]
    
    # Convert to pixel coordinates (center of final pixels)
    x_over = (x_over + 0.5) / oversample - 0.5
    y_over = (y_over + 0.5) / oversample - 0.5
    
    x0, y0 = size / 2.0, size / 2.0
    
    # Rotate coordinates
    pa_rad = np.radians(pa)
    dx = x_over - x0
    dy = y_over - y0
    
    x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)
    
    # Elliptical radius
    b_over_a = 1.0 - eps
    r = np.sqrt(x_rot**2 + (y_rot / b_over_a)**2)
    
    # Sersic profile on oversampled grid
    image_over = sersic_profile(r, I_e, r_e, n)
    
    # Rebin to final pixel grid by averaging
    # Reshape to (size, oversample, size, oversample) and average over oversample dimensions
    image_rebinned = image_over.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    
    params = {
        'x0': x0,
        'y0': y0,
        'I_e': I_e,
        'r_e': r_e,
        'n': n,
        'eps': eps,
        'pa': pa_rad
    }
    
    return image_rebinned, params

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
    
    # Compute analytical CoG
    print(f"\nComputing analytical CoG...")
    cog_analytical = sersic_analytical_cog(I_e, r_e, n, eps, sma_arr)
    
    # Compute true CoG using aperture photometry on the mock image
    print(f"Computing aperture CoG (photutils subpixel method)...")
    cog_aperture = compute_aperture_cog(image, params['x0'], params['y0'], 
                                        sma_arr, params['eps'], pa)
    
    # Compare at maximum SMA
    flux_analytical = cog_analytical[-1]
    flux_aperture = cog_aperture[-1]
    flux_cog = cog_arr[-1]
    
    print(f"\n   Analytical flux:      {flux_analytical:.2e}")
    print(f"   Aperture flux:        {flux_aperture:.2e}")
    print(f"   CoG flux (isoster):   {flux_cog:.2e}")
    
    frac_error_analytical = abs(flux_cog - flux_analytical) / flux_analytical * 100
    frac_error_aperture = abs(flux_cog - flux_aperture) / flux_aperture * 100
    
    print(f"\n   Error vs Analytical:  {frac_error_analytical:.3f}%")
    print(f"   Error vs Aperture:    {frac_error_aperture:.3f}%")
    
    # Verification
    if frac_error_aperture < 1.0:
        print("\n✓ PASS: CoG matches aperture photometry within 1%")
    else:
        print(f"\n✗ FAIL: CoG error {frac_error_aperture:.3f}% exceeds 1% threshold")
    
    # Generate plot
    print("\nGenerating CoG plot...")
    
    # Filter for SMA > 0.8 pixels
    plot_mask = sma_arr > 0.8
    sma_plot = sma_arr[plot_mask]
    cog_plot = cog_arr[plot_mask]
    cog_analytical_plot = cog_analytical[plot_mask]
    cog_aperture_plot = cog_aperture[plot_mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    x_axis = sma_plot**0.25
    
    # Plot 1: CoG comparison
    ax1.plot(x_axis, np.log10(cog_plot), 'o-', markersize=4, label='Isoster CoG', color='blue')
    ax1.plot(x_axis, np.log10(cog_analytical_plot), 's-', markersize=3, 
             label='Analytical CoG', color='green', alpha=0.7)
    ax1.plot(x_axis, np.log10(cog_aperture_plot), '^-', markersize=3, 
             label='Aperture CoG (photutils)', color='red', alpha=0.7)
    
    # Mark maximum
    max_idx = np.argmax(cog_plot)
    ax1.plot(x_axis[max_idx], np.log10(cog_plot[max_idx]), 'b*', 
             markersize=15, label=f'Max at SMA={sma_plot[max_idx]:.1f}')
    
    ax1.set_ylabel(r'$\log_{10}$(CoG)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Sersic n={n} CoG Test (Analytical: {frac_error_analytical:.2f}%, Aperture: {frac_error_aperture:.2f}%)', 
                  fontsize=13, weight='bold')
    
    # Plot 2: Fractional errors
    frac_err_analytical = (cog_plot - cog_analytical_plot) / cog_analytical_plot * 100
    frac_err_aperture = (cog_plot - cog_aperture_plot) / cog_aperture_plot * 100
    
    ax2.plot(x_axis, frac_err_analytical, 'o-', markersize=4, label='vs Analytical', color='green')
    ax2.plot(x_axis, frac_err_aperture, 's-', markersize=4, label='vs Aperture', color='red')
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
        'flux_analytical': flux_analytical,
        'flux_aperture': flux_aperture,
        'flux_cog': flux_cog,
        'frac_error_analytical': frac_error_analytical,
        'frac_error_aperture': frac_error_aperture,
        'isophotes': isophotes,
        'cog_analytical': cog_analytical,
        'cog_aperture': cog_aperture
    }

if __name__ == "__main__":
    results = run_sersic_cog_test()
