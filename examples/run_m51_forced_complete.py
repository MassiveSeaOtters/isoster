"""
M51 Forced Mode Complete Benchmark
===================================

Comprehensive benchmark comparing:
1. Regular mode
2. Method 1: Fixed geometry (fix_center, fix_pa, fix_eps)
3. Method 2: Pure forced mode (no fitting)

Generates detailed QA plot similar to main benchmark.
"""

import os
import sys
import time
import numpy as np
from astropy.io import fits
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_ellipse_model

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
        'pa': pa_global,
        'ba': 1.0 - eps_global
    }

def plot_forced_qa(image, regular_iso, fixed_iso, forced_iso, 
                   regular_time, fixed_time, forced_time, filename):
    """
    Generate comprehensive QA plot comparing all three methods.
    
    Similar to main benchmark plot but comparing forced modes instead of photutils.
    """
    fig = plt.figure(figsize=(20, 14))
    
    # Outer GridSpec
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.05, 
                                 top=0.93, bottom=0.05, left=0.05, right=0.95)
    
    title = f"M51 Forced Mode Comparison\n" \
            f"Regular: {regular_time:.3f}s | Fixed: {fixed_time:.3f}s ({regular_time/fixed_time:.1f}x) | " \
            f"Forced: {forced_time:.3f}s ({regular_time/forced_time:.0f}x)"
    fig.suptitle(title, fontsize=22, weight='bold', y=0.98)
    
    # Left: Image
    left_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[0])
    ax_img = fig.add_subplot(left_gs[0])
    
    vmin, vmax = np.percentile(image[~np.isnan(image)], [1, 99])
    norm_img = np.arcsinh((image - vmin) / (vmax - vmin))
    ax_img.imshow(norm_img, origin='lower', cmap='viridis')
    ax_img.set_xlabel('X (pixels)', fontsize=14)
    ax_img.set_ylabel('Y (pixels)', fontsize=14)
    ax_img.set_title('M51 Image', fontsize=16, weight='bold')
    
    # Right: Profiles
    right_gs = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=outer_gs[1], 
                                                height_ratios=[2, 1, 1, 1, 1], hspace=0.0)
    
    # Extract data
    def get_data(isos):
        sma = np.array([iso['sma'] for iso in isos])
        mask = sma > 1.5
        return {
            'sma': sma[mask],
            'intens': np.array([iso['intens'] for iso in isos])[mask],
            'intens_err': np.array([iso['intens_err'] for iso in isos])[mask],
            'eps': np.array([iso['eps'] for iso in isos])[mask],
            'pa': np.array([iso['pa'] for iso in isos])[mask],
            'x0': np.array([iso['x0'] for iso in isos])[mask],
            'y0': np.array([iso['y0'] for iso in isos])[mask]
        }
    
    reg_data = get_data(regular_iso)
    fix_data = get_data(fixed_iso)
    frc_data = get_data(forced_iso)
    
    x_axis_reg = reg_data['sma']**0.25
    x_axis_fix = fix_data['sma']**0.25
    x_axis_frc = frc_data['sma']**0.25
    
    # Plot 1: Surface Brightness
    ax1 = fig.add_subplot(right_gs[0])
    
    y_reg = np.log10(reg_data['intens'])
    yerr_reg = reg_data['intens_err'] / (reg_data['intens'] * np.log(10))
    
    ax1.errorbar(x_axis_reg, y_reg, yerr=yerr_reg, fmt='o', color='red', 
                markersize=6, label='Regular', elinewidth=1)
    ax1.plot(x_axis_fix, np.log10(fix_data['intens']), 's', color='blue', 
            markersize=5, label='Fixed Geom', alpha=0.7)
    ax1.plot(x_axis_frc, np.log10(frc_data['intens']), '^', color='green', 
            markersize=5, label='Pure Forced', alpha=0.7)
    
    ax1.set_ylabel(r'$\log_{10}(I)$', fontsize=16)
    ax1.legend(loc='upper right', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    
    # Plot 2: Intensity Difference (Fixed vs Regular)
    ax2 = fig.add_subplot(right_gs[1], sharex=ax1)
    fix_interp = np.interp(reg_data['sma'], fix_data['sma'], fix_data['intens'])
    diff_fix = ((reg_data['intens'] - fix_interp) / reg_data['intens']) * 100.0
    ax2.plot(x_axis_reg, diff_fix, 'o-', color='blue', markersize=4, label='Fixed - Regular')
    ax2.axhline(0, color='gray', ls='--')
    ax2.set_ylabel(r'$\Delta I/I$ (%)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])
    
    # Plot 3: Intensity Difference (Forced vs Regular)
    ax3 = fig.add_subplot(right_gs[2], sharex=ax1)
    frc_interp = np.interp(reg_data['sma'], frc_data['sma'], frc_data['intens'])
    diff_frc = ((reg_data['intens'] - frc_interp) / reg_data['intens']) * 100.0
    ax3.plot(x_axis_reg, diff_frc, 'o-', color='green', markersize=4, label='Forced - Regular')
    ax3.axhline(0, color='gray', ls='--')
    ax3.set_ylabel(r'$\Delta I/I$ (%)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels([])
    
    # Plot 4: Ellipticity
    ax4 = fig.add_subplot(right_gs[3], sharex=ax1)
    ax4.plot(x_axis_reg, reg_data['eps'], 'o', color='red', markersize=5, label='Regular')
    ax4.plot(x_axis_fix, fix_data['eps'], 's', color='blue', markersize=4, label='Fixed Geom', alpha=0.7)
    ax4.plot(x_axis_frc, frc_data['eps'], '^', color='green', markersize=4, label='Pure Forced', alpha=0.7)
    ax4.set_ylabel('Ellipticity', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticklabels([])
    
    # Plot 5: Position Angle
    ax5 = fig.add_subplot(right_gs[4], sharex=ax1)
    ax5.plot(x_axis_reg, np.degrees(reg_data['pa']) % 180, 'o', color='red', markersize=5, label='Regular')
    ax5.plot(x_axis_fix, np.degrees(fix_data['pa']) % 180, 's', color='blue', markersize=4, label='Fixed Geom', alpha=0.7)
    ax5.plot(x_axis_frc, np.degrees(frc_data['pa']) % 180, '^', color='green', markersize=4, label='Pure Forced', alpha=0.7)
    ax5.set_ylabel('PA (deg)', fontsize=14)
    ax5.set_xlabel(r'SMA$^{0.25}$ (pixel$^{0.25}$)', fontsize=14)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved QA plot to {filename}")

def run_complete_benchmark():
    print("=" * 80)
    print("ISOSTER M51 Complete Forced Mode Benchmark")
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
    # 1. Regular Mode
    # ---------------------------------------------------------
    print("\n[1/3] Regular Mode...")
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
    regular_time = time.time() - t0
    regular_iso = regular_results['isophotes']
    print(f"   Done in {regular_time:.4f}s. Fitted {len(regular_iso)} isophotes.")
    
    # Compute global geometry
    global_geom = compute_global_geometry(regular_iso, sma_min=2.0, sma_max=150.0, sigma=3.0)
    print(f"   Global: X0={global_geom['x0']:.2f}, Y0={global_geom['y0']:.2f}, "
          f"PA={np.degrees(global_geom['pa']):.2f}°, eps={global_geom['eps']:.4f}")
    
    # Extract SMA array for forced mode
    sma_array = [iso['sma'] for iso in regular_iso]
    
    # ---------------------------------------------------------
    # 2. Method 1: Fixed Geometry
    # ---------------------------------------------------------
    print("\n[2/3] Method 1: Fixed Geometry...")
    cfg_fixed = IsosterConfig(
        x0=global_geom['x0'], y0=global_geom['y0'],
        sma0=sma0, minsma=minsma, maxsma=maxsma, astep=astep,
        eps=global_geom['eps'], pa=global_geom['pa'],
        conver=0.05, maxit=50,
        compute_errors=True,
        compute_deviations=True,
        integrator='adaptive',
        lsb_sma_threshold=100.0,
        fix_center=True,
        fix_pa=True,
        fix_eps=True
    )
    
    t0 = time.time()
    fixed_results = fit_image(image, mask=mask, config=cfg_fixed)
    fixed_time = time.time() - t0
    fixed_iso = fixed_results['isophotes']
    print(f"   Done in {fixed_time:.4f}s. Fitted {len(fixed_iso)} isophotes.")
    print(f"   Speedup vs Regular: {regular_time/fixed_time:.2f}x")
    
    # ---------------------------------------------------------
    # 3. Method 2: Pure Forced
    # ---------------------------------------------------------
    print("\n[3/3] Method 2: Pure Forced...")
    cfg_forced = IsosterConfig(
        x0=global_geom['x0'], y0=global_geom['y0'],
        eps=global_geom['eps'], pa=global_geom['pa'],
        integrator='mean',
        forced=True,
        forced_sma=sma_array
    )
    
    t0 = time.time()
    forced_results = fit_image(image, mask=mask, config=cfg_forced)
    forced_time = time.time() - t0
    forced_iso = forced_results['isophotes']
    print(f"   Done in {forced_time:.4f}s. Extracted {len(forced_iso)} isophotes.")
    print(f"   Speedup vs Regular: {regular_time/forced_time:.0f}x")
    
    # ---------------------------------------------------------
    # 4. Performance Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("-" * 80)
    print(f"Regular Mode:       {regular_time:.4f}s  (baseline)")
    print(f"Fixed Geometry:     {fixed_time:.4f}s  ({regular_time/fixed_time:.2f}x speedup)")
    print(f"Pure Forced:        {forced_time:.4f}s  ({regular_time/forced_time:.0f}x speedup)")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # 5. Generate QA Plot
    # ---------------------------------------------------------
    print("\nGenerating comprehensive QA plot...")
    qa_path = os.path.join(os.path.dirname(__file__), 'm51_forced_qa_summary.png')
    plot_forced_qa(image, regular_iso, fixed_iso, forced_iso,
                   regular_time, fixed_time, forced_time, qa_path)
    
    return {
        'regular_time': regular_time,
        'fixed_time': fixed_time,
        'forced_time': forced_time,
        'regular_iso': regular_iso,
        'fixed_iso': fixed_iso,
        'forced_iso': forced_iso
    }

if __name__ == "__main__":
    results = run_complete_benchmark()
