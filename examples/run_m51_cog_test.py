"""
M51 CoG Test with Crossing Detection
=====================================

Tests CoG photometry on real galaxy data (M51) comparing:
1. Regular mode (may have crossing isophotes)
2. Fixed geometry mode (no crossing)

Demonstrates crossing detection flags and CoG behavior.
"""

import os
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig
from scipy import stats

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

def run_m51_cog_test():
    print("=" * 80)
    print("M51 CoG Test - Crossing Detection")
    print("=" * 80)
    
    # Load M51 data
    fits_path = os.path.join(os.path.dirname(__file__), 'M51.fits')
    if not os.path.exists(fits_path):
        print(f"Error: {fits_path} not found.")
        return

    with fits.open(fits_path) as hdul:
        image = hdul[0].data
        
    h, w = image.shape
    print(f"\nImage: {w}x{h}")
    
    # Generate mask (same as forced mode benchmark)
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
    
    print(f"Masked pixels: {np.sum(mask)}")
    
    # Shared parameters
    sma0 = 10.0
    minsma = 0.0
    maxsma = 275.0
    astep = 0.1
    
    # ---------------------------------------------------------
    # 1. Regular Mode (may have crossing)
    # ---------------------------------------------------------
    print("\n[1/2] Regular Mode (free geometry)...")
    
    cfg_regular = IsosterConfig(
        x0=w/2, y0=h/2,
        sma0=sma0, minsma=minsma, maxsma=maxsma, astep=astep,
        eps=0.2, pa=0.0,
        conver=0.05, maxit=50,
        compute_errors=False,
        compute_deviations=False,
        compute_cog=True,  # Enable CoG
        integrator='adaptive',
        lsb_sma_threshold=100.0
    )
    
    regular_results = fit_image(image, mask=mask, config=cfg_regular)
    regular_iso = regular_results['isophotes']
    
    # Extract crossing flags
    reg_sma = np.array([iso['sma'] for iso in regular_iso])
    reg_cog = np.array([iso['cog'] for iso in regular_iso])
    reg_flag_cross = np.array([iso.get('flag_cross', False) for iso in regular_iso])
    reg_flag_neg = np.array([iso.get('flag_negative_area', False) for iso in regular_iso])
    
    print(f"   Fitted {len(regular_iso)} isophotes")
    print(f"   Crossing flags: {np.sum(reg_flag_cross)} / {len(reg_flag_cross)}")
    print(f"   Negative area flags: {np.sum(reg_flag_neg)} / {len(reg_flag_neg)}")
    
    # ---------------------------------------------------------
    # 2. Fixed Geometry Mode (no crossing)
    # ---------------------------------------------------------
    print("\n[2/2] Fixed Geometry Mode...")
    
    # Compute global geometry from regular run
    global_geom = compute_global_geometry(regular_iso, sma_min=2.0, sma_max=150.0, sigma=3.0)
    print(f"   Global: X0={global_geom['x0']:.2f}, Y0={global_geom['y0']:.2f}, "
          f"PA={np.degrees(global_geom['pa']):.2f}°, eps={global_geom['eps']:.4f}")
    
    cfg_fixed = IsosterConfig(
        x0=global_geom['x0'], y0=global_geom['y0'],
        sma0=sma0, minsma=minsma, maxsma=maxsma, astep=astep,
        eps=global_geom['eps'], pa=global_geom['pa'],
        conver=0.05, maxit=50,
        compute_errors=False,
        compute_deviations=False,
        compute_cog=True,  # Enable CoG
        integrator='adaptive',
        lsb_sma_threshold=100.0,
        fix_center=True,
        fix_pa=True,
        fix_eps=True
    )
    
    fixed_results = fit_image(image, mask=mask, config=cfg_fixed)
    fixed_iso = fixed_results['isophotes']
    
    # Extract data
    fix_sma = np.array([iso['sma'] for iso in fixed_iso])
    fix_cog = np.array([iso['cog'] for iso in fixed_iso])
    fix_flag_cross = np.array([iso.get('flag_cross', False) for iso in fixed_iso])
    fix_flag_neg = np.array([iso.get('flag_negative_area', False) for iso in fixed_iso])
    
    print(f"   Fitted {len(fixed_iso)} isophotes")
    print(f"   Crossing flags: {np.sum(fix_flag_cross)} / {len(fix_flag_cross)}")
    print(f"   Negative area flags: {np.sum(fix_flag_neg)} / {len(fix_flag_neg)}")
    
    # ---------------------------------------------------------
    # 3. Generate Comparison Plot
    # ---------------------------------------------------------
    print("\nGenerating CoG comparison plot...")
    
    # Filter for SMA > 0.8
    reg_mask = reg_sma > 0.8
    fix_mask = fix_sma > 0.8
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1],
                          hspace=0.25, wspace=0.15)
    
    # Panel 1: Image
    ax_img = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(image[~np.isnan(image)], [1, 99])
    norm_img = np.arcsinh((image - vmin) / (vmax - vmin))
    ax_img.imshow(norm_img, origin='lower', cmap='viridis')
    if np.sum(mask) > 0:
        ax_img.contour(mask, levels=[0.5], colors='red', linewidths=0.5, alpha=0.5)
    ax_img.set_xlabel('X (pixels)', fontsize=12)
    ax_img.set_ylabel('Y (pixels)', fontsize=12)
    ax_img.set_title('M51 Image + Mask', fontsize=14, weight='bold')
    
    # Panel 2: CoG Comparison
    ax_cog = fig.add_subplot(gs[0, 1])
    x_reg = reg_sma[reg_mask]**0.25
    x_fix = fix_sma[fix_mask]**0.25
    
    ax_cog.plot(x_reg, np.log10(reg_cog[reg_mask]), 'o-', markersize=4, 
                label='Regular Mode', color='blue')
    ax_cog.plot(x_fix, np.log10(fix_cog[fix_mask]), 's-', markersize=3, 
                label='Fixed Geometry', color='red', alpha=0.7)
    ax_cog.set_ylabel(r'$\log_{10}$(CoG)', fontsize=14)
    ax_cog.set_xlabel(r'SMA$^{0.25}$ (pixel$^{0.25}$)', fontsize=14)
    ax_cog.legend(fontsize=12)
    ax_cog.grid(True, alpha=0.3)
    ax_cog.set_title('Curve-of-Growth Comparison', fontsize=14, weight='bold')
    
    # Panel 3: Crossing Flags
    ax_flags = fig.add_subplot(gs[1, 0])
    ax_flags.plot(x_reg, reg_flag_cross[reg_mask].astype(int), 'o-', markersize=4,
                  label='Regular Mode', color='blue')
    ax_flags.plot(x_fix, fix_flag_cross[fix_mask].astype(int), 's-', markersize=3,
                  label='Fixed Geometry', color='red', alpha=0.7)
    ax_flags.set_ylabel('Crossing Flag', fontsize=14)
    ax_flags.set_xlabel(r'SMA$^{0.25}$ (pixel$^{0.25}$)', fontsize=14)
    ax_flags.set_ylim(-0.1, 1.1)
    ax_flags.legend(fontsize=12)
    ax_flags.grid(True, alpha=0.3)
    ax_flags.set_title('Isophote Crossing Detection', fontsize=14, weight='bold')
    
    # Panel 4: CoG Fractional Difference
    ax_diff = fig.add_subplot(gs[1, 1])
    # Interpolate fixed to regular grid
    fix_cog_interp = np.interp(reg_sma[reg_mask], fix_sma, fix_cog)
    frac_diff = (reg_cog[reg_mask] - fix_cog_interp) / fix_cog_interp * 100
    
    ax_diff.plot(x_reg, frac_diff, 'o-', markersize=4, color='purple')
    ax_diff.axhline(0, color='gray', linestyle='--')
    ax_diff.set_ylabel('Fractional Diff (%)', fontsize=14)
    ax_diff.set_xlabel(r'SMA$^{0.25}$ (pixel$^{0.25}$)', fontsize=14)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title('CoG Difference (Regular - Fixed) / Fixed', fontsize=14, weight='bold')
    
    plt.suptitle(f'M51 CoG Test: Regular ({np.sum(reg_flag_cross)} crossings) vs Fixed (0 crossings)',
                 fontsize=16, weight='bold', y=0.98)
    
    qa_path = os.path.join(os.path.dirname(__file__), 'm51_cog_test.png')
    plt.savefig(qa_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {qa_path}")
    
    return {
        'regular_iso': regular_iso,
        'fixed_iso': fixed_iso,
        'global_geom': global_geom
    }

if __name__ == "__main__":
    results = run_m51_cog_test()
