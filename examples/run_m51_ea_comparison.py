"""
M51 EA Comparison - Free Geometry Fitting
==========================================

Tests eccentric anomaly sampling benefit on real galaxy data (M51)
with FREE geometry fitting (not fixed).

Compares:
1. Regular sampling
2. Eccentric Anomaly sampling

For regions with varying ellipticity to show EA benefit.
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

def run_m51_ea_comparison():
    print("=" * 80)
    print("M51 EA Comparison - Free Geometry Fitting")
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
    
    # Shared configuration - FREE GEOMETRY (not fixed)
    base_config = dict(
        x0=w/2, y0=h/2,
        sma0=10.0, minsma=0.0, maxsma=275.0, astep=0.1,
        eps=0.2, pa=0.0,
        conver=0.05, maxit=50,
        compute_errors=False,
        compute_deviations=False,
        integrator='adaptive',
        lsb_sma_threshold=100.0,
        # FREE GEOMETRY - this is where EA should help
        fix_center=False,
        fix_pa=False,
        fix_eps=False
    )
    
    # ---------------------------------------------------------
    # 1. Regular Sampling
    # ---------------------------------------------------------
    print("\n[1/2] Running with REGULAR sampling (free geometry)...")
    cfg_regular = IsosterConfig(**base_config, use_eccentric_anomaly=False)
    
    regular_results = fit_image(image, mask=None, config=cfg_regular)
    regular_iso = regular_results['isophotes']
    
    print(f"   Fitted {len(regular_iso)} isophotes")
    
    # Extract metrics
    reg_sma = np.array([iso['sma'] for iso in regular_iso])
    reg_eps = np.array([iso['eps'] for iso in regular_iso])
    reg_pa = np.array([iso['pa'] for iso in regular_iso])
    reg_intens = np.array([iso['intens'] for iso in regular_iso])
    reg_rms = np.array([iso['rms'] for iso in regular_iso])
    reg_niter = np.array([iso['niter'] for iso in regular_iso])
    
    print(f"   Mean iterations: {np.mean(reg_niter):.2f}")
    print(f"   Mean RMS: {np.mean(reg_rms):.2f}")
    print(f"   Ellipticity range: {reg_eps.min():.3f} - {reg_eps.max():.3f}")
    
    # ---------------------------------------------------------
    # 2. Eccentric Anomaly Sampling
    # ---------------------------------------------------------
    print("\n[2/2] Running with ECCENTRIC ANOMALY sampling (free geometry)...")
    cfg_ea = IsosterConfig(**base_config, use_eccentric_anomaly=True)
    
    ea_results = fit_image(image, mask=None, config=cfg_ea)
    ea_iso = ea_results['isophotes']
    
    print(f"   Fitted {len(ea_iso)} isophotes")
    
    # Extract metrics
    ea_sma = np.array([iso['sma'] for iso in ea_iso])
    ea_eps = np.array([iso['eps'] for iso in ea_iso])
    ea_pa = np.array([iso['pa'] for iso in ea_iso])
    ea_intens = np.array([iso['intens'] for iso in ea_iso])
    ea_rms = np.array([iso['rms'] for iso in ea_iso])
    ea_niter = np.array([iso['niter'] for iso in ea_iso])
    
    print(f"   Mean iterations: {np.mean(ea_niter):.2f}")
    print(f"   Mean RMS: {np.mean(ea_rms):.2f}")
    print(f"   Ellipticity range: {ea_eps.min():.3f} - {ea_eps.max():.3f}")
    
    # ---------------------------------------------------------
    # 3. Comparison
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    # Convergence speed
    iter_improvement = (np.mean(reg_niter) - np.mean(ea_niter)) / np.mean(reg_niter) * 100
    print(f"Convergence: EA is {iter_improvement:+.1f}% {'faster' if iter_improvement > 0 else 'slower'}")
    
    # RMS improvement
    rms_improvement = (np.mean(reg_rms) - np.mean(ea_rms)) / np.mean(reg_rms) * 100
    print(f"RMS: EA is {rms_improvement:+.1f}% {'better' if rms_improvement > 0 else 'worse'}")
    
    # Ellipticity comparison
    eps_diff = np.abs(reg_eps - ea_eps)
    print(f"Ellipticity difference: {np.mean(eps_diff):.4f} (mean), {np.max(eps_diff):.4f} (max)")
    
    # PA comparison
    pa_diff = np.abs(np.degrees(reg_pa) - np.degrees(ea_pa))
    # Handle wrap-around
    pa_diff = np.minimum(pa_diff, 180 - pa_diff)
    print(f"PA difference: {np.mean(pa_diff):.2f}° (mean), {np.max(pa_diff):.2f}° (max)")
    
    # ---------------------------------------------------------
    # 4. Generate Comparison Plot
    # ---------------------------------------------------------
    print("\nGenerating comparison plot...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.25)
    
    # Panel 1: Image
    ax_img = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(image[~np.isnan(image)], [1, 99])
    norm_img = np.arcsinh((image - vmin) / (vmax - vmin))
    ax_img.imshow(norm_img, origin='lower', cmap='viridis')
    ax_img.set_xlabel('X (pixels)', fontsize=12)
    ax_img.set_ylabel('Y (pixels)', fontsize=12)
    ax_img.set_title('M51 Image', fontsize=14, weight='bold')
    
    # Panel 2: Ellipticity profiles
    ax_eps = fig.add_subplot(gs[0, 1])
    ax_eps.plot(reg_sma, reg_eps, 'o-', markersize=4, label='Regular', color='blue')
    ax_eps.plot(ea_sma, ea_eps, 's-', markersize=3, label='EA', color='red', alpha=0.7)
    ax_eps.set_xlabel('SMA (pixels)', fontsize=12)
    ax_eps.set_ylabel('Ellipticity (ε = 1 - b/a)', fontsize=12)
    ax_eps.legend(fontsize=12)
    ax_eps.grid(True, alpha=0.3)
    ax_eps.set_title(f'Ellipticity Profiles (Δ={np.mean(eps_diff):.4f})', fontsize=14, weight='bold')
    
    # Panel 3: PA profiles
    ax_pa = fig.add_subplot(gs[1, 0])
    ax_pa.plot(reg_sma, np.degrees(reg_pa), 'o-', markersize=4, label='Regular', color='blue')
    ax_pa.plot(ea_sma, np.degrees(ea_pa), 's-', markersize=3, label='EA', color='red', alpha=0.7)
    ax_pa.set_xlabel('SMA (pixels)', fontsize=12)
    ax_pa.set_ylabel('Position Angle (degrees)', fontsize=12)
    ax_pa.legend(fontsize=12)
    ax_pa.grid(True, alpha=0.3)
    ax_pa.set_title(f'PA Profiles (Δ={np.mean(pa_diff):.2f}°)', fontsize=14, weight='bold')
    
    # Panel 4: Iterations
    ax_iter = fig.add_subplot(gs[1, 1])
    ax_iter.plot(reg_sma, reg_niter, 'o-', markersize=4, label='Regular', color='blue')
    ax_iter.plot(ea_sma, ea_niter, 's-', markersize=3, label='EA', color='red', alpha=0.7)
    ax_iter.axhline(np.mean(reg_niter), color='blue', linestyle='--', alpha=0.5)
    ax_iter.axhline(np.mean(ea_niter), color='red', linestyle='--', alpha=0.5)
    ax_iter.set_xlabel('SMA (pixels)', fontsize=12)
    ax_iter.set_ylabel('Iterations to Converge', fontsize=12)
    ax_iter.legend(fontsize=12)
    ax_iter.grid(True, alpha=0.3)
    ax_iter.set_title(f'Convergence (EA: {iter_improvement:+.1f}%)', fontsize=14, weight='bold')
    
    # Panel 5: RMS
    ax_rms = fig.add_subplot(gs[2, 0])
    ax_rms.semilogy(reg_sma, reg_rms, 'o-', markersize=4, label='Regular', color='blue')
    ax_rms.semilogy(ea_sma, ea_rms, 's-', markersize=3, label='EA', color='red', alpha=0.7)
    ax_rms.set_xlabel('SMA (pixels)', fontsize=12)
    ax_rms.set_ylabel('RMS Residual', fontsize=12)
    ax_rms.legend(fontsize=12)
    ax_rms.grid(True, alpha=0.3)
    ax_rms.set_title(f'Fit Quality (EA: {rms_improvement:+.1f}%)', fontsize=14, weight='bold')
    
    # Panel 6: Intensity profiles
    ax_intens = fig.add_subplot(gs[2, 1])
    ax_intens.semilogy(reg_sma, reg_intens, 'o-', markersize=4, label='Regular', color='blue')
    ax_intens.semilogy(ea_sma, ea_intens, 's-', markersize=3, label='EA', color='red', alpha=0.7)
    ax_intens.set_xlabel('SMA (pixels)', fontsize=12)
    ax_intens.set_ylabel('Intensity', fontsize=12)
    ax_intens.legend(fontsize=12)
    ax_intens.grid(True, alpha=0.3)
    ax_intens.set_title('Intensity Profiles', fontsize=14, weight='bold')
    
    plt.suptitle('M51 EA Comparison - Free Geometry Fitting', 
                 fontsize=16, weight='bold', y=0.995)
    
    qa_path = os.path.join(os.path.dirname(__file__), 'm51_ea_comparison.png')
    plt.savefig(qa_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {qa_path}")
    
    return {
        'regular_iso': regular_iso,
        'ea_iso': ea_iso,
        'iter_improvement': iter_improvement,
        'rms_improvement': rms_improvement,
        'eps_diff_mean': np.mean(eps_diff),
        'pa_diff_mean': np.mean(pa_diff)
    }

if __name__ == "__main__":
    results = run_m51_ea_comparison()
