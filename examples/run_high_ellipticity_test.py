"""
High Ellipticity Benchmark - Comprehensive EA Evaluation
=========================================================

Compares three methods for high-ellipticity (ε=0.7) Sersic profile:
1. Photutils isophote (reference implementation)
2. Isoster with regular sampling (uniform in φ)
3. Isoster with eccentric anomaly sampling (uniform in ψ)

All with FREE GEOMETRY to demonstrate EA benefit.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gammaincinv
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig

def sersic_profile(r, I_e, r_e, n):
    """Sersic profile intensity at radius r."""
    b_n = gammaincinv(2*n, 0.5)
    return I_e * np.exp(-b_n * ((r / r_e)**(1.0/n) - 1.0))

def sersic_1d_profile_major_axis(sma_array, I_e, r_e, n, eps):
    """
    Theoretical 1-D intensity profile along the major axis.
    
    For an elliptical Sersic profile, the intensity along the major axis
    is just the Sersic profile evaluated at the major axis radius.
    
    Args:
        sma_array: Semi-major axis values
        I_e: Intensity at effective radius
        r_e: Effective radius
        n: Sersic index
        eps: Ellipticity (ε = 1 - b/a)
        
    Returns:
        Theoretical intensity profile
    """
    return sersic_profile(sma_array, I_e, r_e, n)

def generate_high_ellipticity_sersic(size=512, I_e=1000.0, r_e=50.0, n=4.0, eps=0.7, pa=30.0, oversample=50):
    """
    Generate high-ellipticity Sersic profile with very high oversampling.
    
    Args:
        oversample: Oversampling factor (50 for accuracy with high n and high ε)
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

def run_photutils(image, params):
    """Run photutils isophote fitting."""
    from photutils.isophote import EllipseGeometry, Ellipse
    
    # Initial geometry
    geometry = EllipseGeometry(
        x0=params['x0'], y0=params['y0'],
        sma=10.0, eps=params['eps'], pa=params['pa']
    )
    
    # Run fitting
    ellipse = Ellipse(image, geometry)
    start_time = time.time()
    isolist = ellipse.fit_image(maxsma=200.0, step=0.1, maxrit=50)
    elapsed = time.time() - start_time
    
    # Extract results
    results = []
    for iso in isolist:
        results.append({
            'sma': iso.sma,
            'intens': iso.intens,
            'eps': iso.eps,
            'pa': iso.pa,
            'x0': iso.x0,
            'y0': iso.y0,
            'rms': iso.rms,
            'niter': iso.niter
        })
    
    return results, elapsed

def run_comprehensive_test():
    print("=" * 80)
    print("High Ellipticity Benchmark - Comprehensive EA Evaluation")
    print("=" * 80)
    
    # Generate high-ellipticity Sersic profile with high oversampling
    print("\nGenerating high-ellipticity Sersic profile...")
    I_e, r_e, n = 1000.0, 50.0, 4.0
    eps = 0.7
    pa = 30.0
    
    image, params = generate_high_ellipticity_sersic(
        size=512, I_e=I_e, r_e=r_e, n=n, eps=eps, pa=pa, oversample=50
    )
    
    print(f"   Size: {image.shape[0]}x{image.shape[1]}")
    print(f"   I_e: {I_e}, r_e: {r_e}, n: {n}")
    print(f"   eps: {eps} (b/a = {1-eps:.2f}), PA: {pa}°")
    print(f"   Oversample: 50x for accurate rendering")
    
    # Shared configuration - FREE GEOMETRY
    base_config = dict(
        x0=params['x0'], y0=params['y0'],
        sma0=10.0, minsma=0.0, maxsma=200.0, astep=0.1,
        eps=eps, pa=params['pa'],
        conver=0.05, maxit=50,
        compute_errors=False,
        compute_deviations=False,
        # FREE GEOMETRY - not fixed
        fix_center=False,
        fix_pa=False,
        fix_eps=False
    )
    
    # ---------------------------------------------------------
    # 1. Photutils
    # ---------------------------------------------------------
    print("\n[1/3] Running PHOTUTILS isophote...")
    try:
        photutils_iso, photutils_time = run_photutils(image, params)
        print(f"   Fitted {len(photutils_iso)} isophotes")
        print(f"   Runtime: {photutils_time:.2f}s")
    except Exception as e:
        print(f"   Error: {e}")
        photutils_iso, photutils_time = None, None
    
    # ---------------------------------------------------------
    # 2. Isoster Regular
    # ---------------------------------------------------------
    print("\n[2/3] Running ISOSTER with REGULAR sampling (free geometry)...")
    cfg_regular = IsosterConfig(**base_config, use_eccentric_anomaly=False)
    
    start_time = time.time()
    regular_results = fit_image(image, mask=None, config=cfg_regular)
    regular_time = time.time() - start_time
    regular_iso = regular_results['isophotes']
    
    print(f"   Fitted {len(regular_iso)} isophotes")
    print(f"   Runtime: {regular_time:.2f}s")
    
    # ---------------------------------------------------------
    # 3. Isoster EA
    # ---------------------------------------------------------
    print("\n[3/3] Running ISOSTER with ECCENTRIC ANOMALY (free geometry)...")
    cfg_ea = IsosterConfig(**base_config, use_eccentric_anomaly=True)
    
    start_time = time.time()
    ea_results = fit_image(image, mask=None, config=cfg_ea)
    ea_time = time.time() - start_time
    ea_iso = ea_results['isophotes']
    
    print(f"   Fitted {len(ea_iso)} isophotes")
    print(f"   Runtime: {ea_time:.2f}s")
    
    # ---------------------------------------------------------
    # 4. Isoster EA + Central Regularization
    # ---------------------------------------------------------
    print("\n[4/4] Running ISOSTER with EA + CENTRAL REGULARIZATION...")
    print("   Config: strength=5.0, threshold=5.0, eps_weight=3.0")
    cfg_ea_reg = IsosterConfig(
        **base_config,
        use_eccentric_anomaly=True,
        use_central_regularization=True,
        central_reg_sma_threshold=5.0,
        central_reg_strength=5.0,  # Stronger regularization
        central_reg_weights={'eps': 3.0, 'pa': 1.0, 'center': 1.0}  # Emphasize ellipticity stability
    )
    
    start_time = time.time()
    ea_reg_results = fit_image(image, mask=None, config=cfg_ea_reg)
    ea_reg_time = time.time() - start_time
    ea_reg_iso = ea_reg_results['isophotes']
    
    print(f"   Fitted {len(ea_reg_iso)} isophotes")
    print(f"   Runtime: {ea_reg_time:.2f}s")
    
    # ---------------------------------------------------------
    # 5. Comparison
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    if photutils_time:
        print(f"Photutils:       {photutils_time:.2f}s (baseline)")
        print(f"Isoster Regular: {regular_time:.2f}s ({regular_time/photutils_time:.2f}x)")
        print(f"Isoster EA:      {ea_time:.2f}s ({ea_time/photutils_time:.2f}x)")
        print(f"Isoster EA+Reg:  {ea_reg_time:.2f}s ({ea_reg_time/photutils_time:.2f}x)")
        speedup_vs_photutils = photutils_time / ea_time
        print(f"\nIsoster EA speedup vs Photutils: {speedup_vs_photutils:.2f}x")
    
    speedup_ea_vs_regular = regular_time / ea_time
    print(f"EA speedup vs Regular: {speedup_ea_vs_regular:.2f}x")
    speedup_reg_vs_ea = ea_time / ea_reg_time
    print(f"EA+Reg vs EA: {speedup_reg_vs_ea:.2f}x")
    
    # ---------------------------------------------------------
    # 6. Generate Comprehensive QA Plot
    # ---------------------------------------------------------
    print("\nGenerating comprehensive QA plot...")
    
    # Build models
    from isoster.model import build_ellipse_model
    regular_model = build_ellipse_model(image.shape, regular_iso)
    ea_model = build_ellipse_model(image.shape, ea_iso)
    ea_reg_model = build_ellipse_model(image.shape, ea_reg_iso)
    
    # Extract data
    reg_sma = np.array([iso['sma'] for iso in regular_iso])
    reg_intens = np.array([iso['intens'] for iso in regular_iso])
    reg_eps = np.array([iso['eps'] for iso in regular_iso])
    reg_pa = np.array([iso['pa'] for iso in regular_iso])
    reg_x0 = np.array([iso['x0'] for iso in regular_iso])
    reg_y0 = np.array([iso['y0'] for iso in regular_iso])
    
    ea_sma = np.array([iso['sma'] for iso in ea_iso])
    ea_intens = np.array([iso['intens'] for iso in ea_iso])
    ea_eps = np.array([iso['eps'] for iso in ea_iso])
    ea_pa = np.array([iso['pa'] for iso in ea_iso])
    ea_x0 = np.array([iso['x0'] for iso in ea_iso])
    ea_y0 = np.array([iso['y0'] for iso in ea_iso])
    
    ea_reg_sma = np.array([iso['sma'] for iso in ea_reg_iso])
    ea_reg_intens = np.array([iso['intens'] for iso in ea_reg_iso])
    ea_reg_eps = np.array([iso['eps'] for iso in ea_reg_iso])
    ea_reg_pa = np.array([iso['pa'] for iso in ea_reg_iso])
    ea_reg_x0 = np.array([iso['x0'] for iso in ea_reg_iso])
    ea_reg_y0 = np.array([iso['y0'] for iso in ea_reg_iso])
    
    # Photutils data
    if photutils_iso:
        phot_sma = np.array([iso['sma'] for iso in photutils_iso])
        phot_intens = np.array([iso['intens'] for iso in photutils_iso])
        phot_eps = np.array([iso['eps'] for iso in photutils_iso])
        phot_pa = np.array([iso['pa'] for iso in photutils_iso])
        phot_x0 = np.array([iso['x0'] for iso in photutils_iso])
        phot_y0 = np.array([iso['y0'] for iso in photutils_iso])
    
    # Theoretical profile
    common_sma = np.linspace(0, 200, 200)
    theoretical_intens = sersic_1d_profile_major_axis(common_sma, I_e, r_e, n, eps)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.35,
                          height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # Row 1: Image, Models, Residuals
    ax_img = fig.add_subplot(gs[0, 0])
    ax_model_reg = fig.add_subplot(gs[0, 1])
    ax_model_ea = fig.add_subplot(gs[0, 2])
    ax_resid = fig.add_subplot(gs[0, 3])
    
    vmin, vmax = np.percentile(image, [1, 99])
    
    ax_img.imshow(np.arcsinh((image - vmin)/(vmax - vmin)), origin='lower', cmap='viridis')
    ax_img.set_title('Input Image', fontsize=11, weight='bold')
    ax_img.axis('off')
    
    ax_model_reg.imshow(np.arcsinh((regular_model - vmin)/(vmax - vmin)), origin='lower', cmap='viridis')
    ax_model_reg.set_title('Isoster Regular Model', fontsize=11, weight='bold')
    ax_model_reg.axis('off')
    
    ax_model_ea.imshow(np.arcsinh((ea_model - vmin)/(vmax - vmin)), origin='lower', cmap='viridis')
    ax_model_ea.set_title('Isoster EA Model', fontsize=11, weight='bold')
    ax_model_ea.axis('off')
    
    resid = image - ea_model
    ax_resid.imshow(resid, origin='lower', cmap='RdBu_r', vmin=-100, vmax=100)
    ax_resid.set_title('Residual (Image - EA)', fontsize=11, weight='bold')
    ax_resid.axis('off')
    
    # Row 2: Intensity and residuals
    ax_intens = fig.add_subplot(gs[1, :2])
    ax_intens_resid = fig.add_subplot(gs[1, 2:])
    
    ax_intens.semilogy(common_sma, theoretical_intens, '--', color='grey', linewidth=2, label='Theoretical', zorder=0)
    ax_intens.semilogy(reg_sma, reg_intens, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_intens.semilogy(ea_sma, ea_intens, 's-', markersize=3, label='EA', alpha=0.7)
    ax_intens.semilogy(ea_reg_sma, ea_reg_intens, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_intens.semilogy(phot_sma, phot_intens, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_intens.set_xlabel('SMA (pixels)', fontsize=10)
    ax_intens.set_ylabel('Intensity', fontsize=10)
    ax_intens.set_xlim(left=1.1)
    ax_intens.legend(fontsize=9, loc='best')
    ax_intens.grid(True, alpha=0.3)
    ax_intens.set_title('Intensity Profiles', fontsize=11, weight='bold')
    
    # Absolute deviation vs theoretical (fractional)
    reg_intens_interp = np.interp(common_sma, reg_sma, reg_intens)
    ea_intens_interp = np.interp(common_sma, ea_sma, ea_intens)
    ea_reg_intens_interp = np.interp(common_sma, ea_reg_sma, ea_reg_intens)
    reg_abs_dev = np.abs(reg_intens_interp - theoretical_intens) / theoretical_intens * 100
    ea_abs_dev = np.abs(ea_intens_interp - theoretical_intens) / theoretical_intens * 100
    ea_reg_abs_dev = np.abs(ea_reg_intens_interp - theoretical_intens) / theoretical_intens * 100
    if photutils_iso:
        phot_intens_interp = np.interp(common_sma, phot_sma, phot_intens)
        phot_abs_dev = np.abs(phot_intens_interp - theoretical_intens) / theoretical_intens * 100
    
    ax_intens_resid.semilogy(common_sma, reg_abs_dev, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_intens_resid.semilogy(common_sma, ea_abs_dev, 's-', markersize=3, label='EA', alpha=0.7)
    ax_intens_resid.semilogy(common_sma, ea_reg_abs_dev, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_intens_resid.semilogy(common_sma, phot_abs_dev, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_intens_resid.set_xlabel('SMA (pixels)', fontsize=10)
    ax_intens_resid.set_ylabel('Fractional Deviation (%)', fontsize=10)
    ax_intens_resid.set_xlim(left=1.1)
    ax_intens_resid.legend(fontsize=9)
    ax_intens_resid.grid(True, alpha=0.3)
    ax_intens_resid.set_title('Fractional Deviation vs Theory', fontsize=11, weight='bold')
    
    # Row 3: Geometry parameters
    ax_eps = fig.add_subplot(gs[2, 0])
    ax_pa = fig.add_subplot(gs[2, 1])
    ax_x0 = fig.add_subplot(gs[2, 2])
    ax_y0 = fig.add_subplot(gs[2, 3])
    
    ax_eps.axhline(eps, color='grey', linestyle='--', label='True', linewidth=2, zorder=0)
    ax_eps.plot(reg_sma, reg_eps, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_eps.plot(ea_sma, ea_eps, 's-', markersize=3, label='EA', alpha=0.7)
    ax_eps.plot(ea_reg_sma, ea_reg_eps, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_eps.plot(phot_sma, phot_eps, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_eps.set_xlabel('SMA (pixels)', fontsize=10)
    ax_eps.set_ylabel('Ellipticity (ε)', fontsize=10)
    ax_eps.set_xlim(left=1.1)
    ax_eps.legend(fontsize=9)
    ax_eps.grid(True, alpha=0.3)
    ax_eps.set_title('Ellipticity', fontsize=11, weight='bold')
    
    ax_pa.axhline(pa, color='grey', linestyle='--', label='True', linewidth=2, zorder=0)
    ax_pa.plot(reg_sma, np.degrees(reg_pa), 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_pa.plot(ea_sma, np.degrees(ea_pa), 's-', markersize=3, label='EA', alpha=0.7)
    ax_pa.plot(ea_reg_sma, np.degrees(ea_reg_pa), 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_pa.plot(phot_sma, np.degrees(phot_pa), '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_pa.set_xlabel('SMA (pixels)', fontsize=10)
    ax_pa.set_ylabel('PA (degrees)', fontsize=10)
    ax_pa.set_xlim(left=1.1)
    ax_pa.legend(fontsize=9)
    ax_pa.grid(True, alpha=0.3)
    ax_pa.set_title('Position Angle', fontsize=11, weight='bold')
    
    ax_x0.axhline(params['x0'], color='grey', linestyle='--', label='True', linewidth=2, zorder=0)
    ax_x0.plot(reg_sma, reg_x0, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_x0.plot(ea_sma, ea_x0, 's-', markersize=3, label='EA', alpha=0.7)
    ax_x0.plot(ea_reg_sma, ea_reg_x0, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_x0.plot(phot_sma, phot_x0, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_x0.set_xlabel('SMA (pixels)', fontsize=10)
    ax_x0.set_ylabel('X Center', fontsize=10)
    ax_x0.set_xlim(left=1.1)
    ax_x0.legend(fontsize=9)
    ax_x0.grid(True, alpha=0.3)
    ax_x0.set_title('X Center', fontsize=11, weight='bold')
    
    ax_y0.axhline(params['y0'], color='grey', linestyle='--', label='True', linewidth=2, zorder=0)
    ax_y0.plot(reg_sma, reg_y0, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_y0.plot(ea_sma, ea_y0, 's-', markersize=3, label='EA', alpha=0.7)
    ax_y0.plot(ea_reg_sma, ea_reg_y0, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_y0.plot(phot_sma, phot_y0, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_y0.set_xlabel('SMA (pixels)', fontsize=10)
    ax_y0.set_ylabel('Y Center', fontsize=10)
    ax_y0.set_xlim(left=1.1)
    ax_y0.legend(fontsize=9)
    ax_y0.grid(True, alpha=0.3)
    ax_y0.set_title('Y Center', fontsize=11, weight='bold')
    
    plt.suptitle(f'High Ellipticity EA Benchmark: ε={eps}, n={n} (EA: {speedup_ea_vs_regular:.2f}x faster)',
                 fontsize=14, weight='bold', y=0.995)
    
    qa_path = os.path.join(os.path.dirname(__file__), 'high_ellipticity_ea_test.png')
    plt.savefig(qa_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {qa_path}")
    
    return {
        'photutils_time': photutils_time,
        'regular_time': regular_time,
        'ea_time': ea_time,
        'speedup_ea_vs_regular': speedup_ea_vs_regular,
        'speedup_vs_photutils': speedup_vs_photutils if photutils_time else None
    }

if __name__ == "__main__":
    results = run_comprehensive_test()
