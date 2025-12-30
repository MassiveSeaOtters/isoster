"""
High Ellipticity Benchmark - Eccentric Anomaly vs Regular Sampling
===================================================================

Tests the benefit of eccentric anomaly sampling for high-ellipticity galaxies.

Compares:
1. Regular sampling (uniform in position angle φ)
2. Eccentric Anomaly sampling (uniform in ψ)

For a challenging ε=0.7 elliptical Sersic profile.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gammaincinv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig

def sersic_profile(r, I_e, r_e, n):
    """Sersic profile intensity at radius r."""
    b_n = gammaincinv(2*n, 0.5)
    return I_e * np.exp(-b_n * ((r / r_e)**(1.0/n) - 1.0))

def generate_high_ellipticity_sersic(size=512, I_e=1000.0, r_e=50.0, n=4.0, eps=0.7, pa=30.0, oversample=10):
    """
    Generate high-ellipticity Sersic profile with proper oversampling.
    
    Args:
        size: Image size
        I_e: Intensity at effective radius
        r_e: Effective radius  
        n: Sersic index
        eps: Ellipticity (ε = 1 - b/a), using challenging ε=0.7
        pa: Position angle in degrees
        oversample: Oversampling factor
        
    Returns:
        image, params dict
    """
    # Create oversampled grid
    size_over = size * oversample
    y_over, x_over = np.mgrid[0:size_over, 0:size_over]
    
    # Convert to pixel coordinates
    x_over = (x_over + 0.5) / oversample - 0.5
    y_over = (y_over + 0.5) / oversample - 0.5
    
    x0, y0 = size / 2.0, size / 2.0
    
    # Rotate coordinates
    pa_rad = np.radians(pa)
    dx = x_over - x0
    dy = y_over - y0
    
    x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)
    
    # Elliptical radius (ε = 1 - b/a)
    b_over_a = 1.0 - eps
    r = np.sqrt(x_rot**2 + (y_rot / b_over_a)**2)
    
    # Sersic profile on oversampled grid
    image_over = sersic_profile(r, I_e, r_e, n)
    
    # Rebin to final pixel grid
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

def run_high_ellipticity_test():
    print("=" * 80)
    print("High Ellipticity Benchmark - EA vs Regular Sampling")
    print("=" * 80)
    
    # Generate challenging high-ellipticity Sersic profile
    print("\nGenerating high-ellipticity Sersic profile...")
    I_e, r_e, n = 1000.0, 50.0, 4.0
    eps = 0.7  # Very challenging case
    pa = 30.0
    
    image, params = generate_high_ellipticity_sersic(
        size=512, I_e=I_e, r_e=r_e, n=n, eps=eps, pa=pa, oversample=10
    )
    
    print(f"   Size: {image.shape[0]}x{image.shape[1]}")
    print(f"   I_e: {I_e}, r_e: {r_e}, n: {n}")
    print(f"   eps: {eps} (ε = 1 - b/a), PA: {pa}°")
    print(f"   Note: ε={eps} is very high ellipticity (b/a = {1-eps:.2f})")
    
    # Shared configuration
    base_config = dict(
        x0=params['x0'], y0=params['y0'],
        sma0=10.0, minsma=0.0, maxsma=200.0, astep=0.1,
        eps=eps, pa=params['pa'],
        conver=0.05, maxit=50,
        compute_errors=False,
        compute_deviations=False,
        fix_center=True,
        fix_pa=True,
        fix_eps=True
    )
    
    # ---------------------------------------------------------
    # 1. Regular Sampling
    # ---------------------------------------------------------
    print("\n[1/2] Running with REGULAR sampling...")
    cfg_regular = IsosterConfig(**base_config, use_eccentric_anomaly=False)
    
    regular_results = fit_image(image, mask=None, config=cfg_regular)
    regular_iso = regular_results['isophotes']
    
    print(f"   Fitted {len(regular_iso)} isophotes")
    
    # Extract metrics
    reg_sma = np.array([iso['sma'] for iso in regular_iso])
    reg_intens = np.array([iso['intens'] for iso in regular_iso])
    reg_rms = np.array([iso['rms'] for iso in regular_iso])
    reg_niter = np.array([iso['niter'] for iso in regular_iso])
    
    print(f"   Mean iterations: {np.mean(reg_niter):.2f}")
    print(f"   Mean RMS: {np.mean(reg_rms):.2f}")
    
    # ---------------------------------------------------------
    # 2. Eccentric Anomaly Sampling
    # ---------------------------------------------------------
    print("\n[2/2] Running with ECCENTRIC ANOMALY sampling...")
    cfg_ea = IsosterConfig(**base_config, use_eccentric_anomaly=True)
    
    ea_results = fit_image(image, mask=None, config=cfg_ea)
    ea_iso = ea_results['isophotes']
    
    print(f"   Fitted {len(ea_iso)} isophotes")
    
    # Extract metrics
    ea_sma = np.array([iso['sma'] for iso in ea_iso])
    ea_intens = np.array([iso['intens'] for iso in ea_iso])
    ea_rms = np.array([iso['rms'] for iso in ea_iso])
    ea_niter = np.array([iso['niter'] for iso in ea_iso])
    
    print(f"   Mean iterations: {np.mean(ea_niter):.2f}")
    print(f"   Mean RMS: {np.mean(ea_rms):.2f}")
    
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
    
    # Intensity comparison
    # Interpolate to common grid
    common_sma = np.linspace(max(reg_sma.min(), ea_sma.min()), 
                             min(reg_sma.max(), ea_sma.max()), 100)
    reg_intens_interp = np.interp(common_sma, reg_sma, reg_intens)
    ea_intens_interp = np.interp(common_sma, ea_sma, ea_intens)
    
    intens_diff = np.abs(reg_intens_interp - ea_intens_interp) / reg_intens_interp * 100
    print(f"Intensity difference: {np.mean(intens_diff):.3f}% (mean), {np.max(intens_diff):.3f}% (max)")
    
    # ---------------------------------------------------------
    # 4. Generate Comparison Plot
    # ---------------------------------------------------------
    print("\nGenerating comparison plot...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.25)
    
    # Panel 1: Image
    ax_img = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(image, [1, 99])
    norm_img = np.arcsinh((image - vmin) / (vmax - vmin))
    ax_img.imshow(norm_img, origin='lower', cmap='viridis')
    ax_img.set_xlabel('X (pixels)', fontsize=12)
    ax_img.set_ylabel('Y (pixels)', fontsize=12)
    ax_img.set_title(f'Sersic n={n}, ε={eps} (b/a={1-eps:.2f})', fontsize=14, weight='bold')
    
    # Panel 2: Intensity profiles
    ax_intens = fig.add_subplot(gs[0, 1])
    ax_intens.semilogy(reg_sma, reg_intens, 'o-', markersize=4, label='Regular', color='blue')
    ax_intens.semilogy(ea_sma, ea_intens, 's-', markersize=3, label='EA', color='red', alpha=0.7)
    ax_intens.set_xlabel('SMA (pixels)', fontsize=12)
    ax_intens.set_ylabel('Intensity', fontsize=12)
    ax_intens.legend(fontsize=12)
    ax_intens.grid(True, alpha=0.3)
    ax_intens.set_title('Intensity Profiles', fontsize=14, weight='bold')
    
    # Panel 3: Iterations
    ax_iter = fig.add_subplot(gs[1, 0])
    ax_iter.plot(reg_sma, reg_niter, 'o-', markersize=4, label='Regular', color='blue')
    ax_iter.plot(ea_sma, ea_niter, 's-', markersize=3, label='EA', color='red', alpha=0.7)
    ax_iter.axhline(np.mean(reg_niter), color='blue', linestyle='--', alpha=0.5)
    ax_iter.axhline(np.mean(ea_niter), color='red', linestyle='--', alpha=0.5)
    ax_iter.set_xlabel('SMA (pixels)', fontsize=12)
    ax_iter.set_ylabel('Iterations to Converge', fontsize=12)
    ax_iter.legend(fontsize=12)
    ax_iter.grid(True, alpha=0.3)
    ax_iter.set_title(f'Convergence (EA: {iter_improvement:+.1f}%)', fontsize=14, weight='bold')
    
    # Panel 4: RMS
    ax_rms = fig.add_subplot(gs[1, 1])
    ax_rms.semilogy(reg_sma, reg_rms, 'o-', markersize=4, label='Regular', color='blue')
    ax_rms.semilogy(ea_sma, ea_rms, 's-', markersize=3, label='EA', color='red', alpha=0.7)
    ax_rms.axhline(np.mean(reg_rms), color='blue', linestyle='--', alpha=0.5)
    ax_rms.axhline(np.mean(ea_rms), color='red', linestyle='--', alpha=0.5)
    ax_rms.set_xlabel('SMA (pixels)', fontsize=12)
    ax_rms.set_ylabel('RMS Residual', fontsize=12)
    ax_rms.legend(fontsize=12)
    ax_rms.grid(True, alpha=0.3)
    ax_rms.set_title(f'Fit Quality (EA: {rms_improvement:+.1f}%)', fontsize=14, weight='bold')
    
    plt.suptitle(f'High Ellipticity Test: ε={eps} (ε = 1 - b/a)', 
                 fontsize=16, weight='bold', y=0.98)
    
    qa_path = os.path.join(os.path.dirname(__file__), 'high_ellipticity_ea_test.png')
    plt.savefig(qa_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {qa_path}")
    
    return {
        'regular_iso': regular_iso,
        'ea_iso': ea_iso,
        'iter_improvement': iter_improvement,
        'rms_improvement': rms_improvement,
        'intens_diff_mean': np.mean(intens_diff)
    }

if __name__ == "__main__":
    results = run_high_ellipticity_test()
