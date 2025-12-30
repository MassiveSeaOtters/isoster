"""
M51 EA Comparison - Comprehensive Evaluation with Photutils
=============================================================

Compares three methods on real galaxy data (M51):
1. Photutils isophote (reference implementation)
2. Isoster with regular sampling (uniform in φ)
3. Isoster with EA + moderate central regularization (SMA < 3 pixels)

All with FREE GEOMETRY to demonstrate real-world performance.
"""

import os
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig

def run_photutils_m51(image, x0, y0, sma0, eps0, pa0):
    """Run photutils isophote fitting on M51."""
    from photutils.isophote import EllipseGeometry, Ellipse
    
    # Initial geometry
    geometry = EllipseGeometry(
        x0=x0, y0=y0,
        sma=sma0, eps=eps0, pa=pa0
    )
    
    # Run fitting
    ellipse = Ellipse(image, geometry)
    start_time = time.time()
    isolist = ellipse.fit_image(maxsma=275.0, step=0.1, maxrit=50)
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

def run_m51_ea_comparison():
    print("=" * 80)
    print("M51 EA Comparison - Comprehensive Evaluation")
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
    
    # Initial geometry
    x0, y0 = w/2, h/2
    sma0 = 10.0
    eps0 = 0.2
    pa0 = 0.0
    
    # Shared configuration - FREE GEOMETRY
    base_config = dict(
        x0=x0, y0=y0,
        sma0=sma0, minsma=0.0, maxsma=275.0, astep=0.1,
        eps=eps0, pa=pa0,
        conver=0.05, maxit=50,
        compute_errors=False,
        compute_deviations=False,
        integrator='adaptive',
        lsb_sma_threshold=100.0,
        # FREE GEOMETRY
        fix_center=False,
        fix_pa=False,
        fix_eps=False
    )
    
    # ---------------------------------------------------------
    # 1. Photutils
    # ---------------------------------------------------------
    print("\n[1/3] Running PHOTUTILS isophote...")
    try:
        photutils_iso, photutils_time = run_photutils_m51(image, x0, y0, sma0, eps0, pa0)
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
    # 3. Isoster EA + Moderate Central Regularization
    # ---------------------------------------------------------
    print("\n[3/3] Running ISOSTER with EA + CENTRAL REGULARIZATION...")
    print("   Config: EA + moderate regularization (threshold=3.0, strength=1.0)")
    cfg_ea_reg = IsosterConfig(
        **base_config,
        use_eccentric_anomaly=True,
        use_central_regularization=True,
        central_reg_sma_threshold=3.0,  # Apply at SMA < 3 pixels
        central_reg_strength=1.0,        # Moderate regularization
        central_reg_weights={'eps': 1.5, 'pa': 1.0, 'center': 1.0}
    )
    
    start_time = time.time()
    ea_reg_results = fit_image(image, mask=None, config=cfg_ea_reg)
    ea_reg_time = time.time() - start_time
    ea_reg_iso = ea_reg_results['isophotes']
    
    print(f"   Fitted {len(ea_reg_iso)} isophotes")
    print(f"   Runtime: {ea_reg_time:.2f}s")
    
    # ---------------------------------------------------------
    # 4. Comparison
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    if photutils_time:
        print(f"Photutils:       {photutils_time:.2f}s (baseline)")
        print(f"Isoster Regular: {regular_time:.2f}s ({regular_time/photutils_time:.2f}x)")
        print(f"Isoster EA+Reg:  {ea_reg_time:.2f}s ({ea_reg_time/photutils_time:.2f}x)")
        speedup_vs_photutils = photutils_time / ea_reg_time
        print(f"\nIsoster EA+Reg speedup vs Photutils: {speedup_vs_photutils:.2f}x")
    
    speedup_reg_vs_ea = regular_time / ea_reg_time
    print(f"EA+Reg vs Regular: {speedup_reg_vs_ea:.2f}x")
    
    # ---------------------------------------------------------
    # 5. Generate Comprehensive QA Plot
    # ---------------------------------------------------------
    print("\nGenerating comprehensive QA plot...")
    
    # Build models
    from isoster.model import build_ellipse_model
    regular_model = build_ellipse_model(image.shape, regular_iso)
    ea_reg_model = build_ellipse_model(image.shape, ea_reg_iso)
    
    # Extract data
    reg_sma = np.array([iso['sma'] for iso in regular_iso])
    reg_intens = np.array([iso['intens'] for iso in regular_iso])
    reg_eps = np.array([iso['eps'] for iso in regular_iso])
    reg_pa = np.array([iso['pa'] for iso in regular_iso])
    reg_x0 = np.array([iso['x0'] for iso in regular_iso])
    reg_y0 = np.array([iso['y0'] for iso in regular_iso])
    
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
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35,
                          height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # Row 1: Image, Models, Residual
    ax_img = fig.add_subplot(gs[0, 0])
    ax_model_reg = fig.add_subplot(gs[0, 1])
    ax_model_ea = fig.add_subplot(gs[0, 2])
    
    vmin, vmax = np.percentile(image[~np.isnan(image)], [1, 99])
    norm_img = np.arcsinh((image - vmin) / (vmax - vmin))
    
    ax_img.imshow(norm_img, origin='lower', cmap='viridis')
    ax_img.set_title('M51 Image', fontsize=11, weight='bold')
    ax_img.axis('off')
    
    ax_model_reg.imshow(np.arcsinh((regular_model - vmin)/(vmax - vmin)), origin='lower', cmap='viridis')
    ax_model_reg.set_title('Isoster Regular Model', fontsize=11, weight='bold')
    ax_model_reg.axis('off')
    
    ax_model_ea.imshow(np.arcsinh((ea_reg_model - vmin)/(vmax - vmin)), origin='lower', cmap='viridis')
    ax_model_ea.set_title('Isoster EA+Reg Model', fontsize=11, weight='bold')
    ax_model_ea.axis('off')
    
    # Row 2: 1-D profiles
    ax_intens = fig.add_subplot(gs[1, :2])
    ax_eps = fig.add_subplot(gs[1, 2])
    
    ax_intens.semilogy(reg_sma, reg_intens, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_intens.semilogy(ea_reg_sma, ea_reg_intens, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_intens.semilogy(phot_sma, phot_intens, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_intens.set_xlabel('SMA (pixels)', fontsize=10)
    ax_intens.set_ylabel('Intensity', fontsize=10)
    ax_intens.set_xlim(left=1.1)
    ax_intens.legend(fontsize=9, loc='best')
    ax_intens.grid(True, alpha=0.3)
    ax_intens.set_title('Intensity Profiles', fontsize=11, weight='bold')
    
    ax_eps.plot(reg_sma, reg_eps, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_eps.plot(ea_reg_sma, ea_reg_eps, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_eps.plot(phot_sma, phot_eps, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_eps.set_xlabel('SMA (pixels)', fontsize=10)
    ax_eps.set_ylabel('Ellipticity (ε)', fontsize=10)
    ax_eps.set_xlim(left=1.1)
    ax_eps.legend(fontsize=9)
    ax_eps.grid(True, alpha=0.3)
    ax_eps.set_title('Ellipticity', fontsize=11, weight='bold')
    
    # Row 3: More geometry parameters
    ax_pa = fig.add_subplot(gs[2, 0])
    ax_x0 = fig.add_subplot(gs[2, 1])
    ax_y0 = fig.add_subplot(gs[2, 2])
    
    ax_pa.plot(reg_sma, np.degrees(reg_pa), 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_pa.plot(ea_reg_sma, np.degrees(ea_reg_pa), 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_pa.plot(phot_sma, np.degrees(phot_pa), '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_pa.set_xlabel('SMA (pixels)', fontsize=10)
    ax_pa.set_ylabel('PA (degrees)', fontsize=10)
    ax_pa.set_xlim(left=1.1)
    ax_pa.legend(fontsize=9)
    ax_pa.grid(True, alpha=0.3)
    ax_pa.set_title('Position Angle', fontsize=11, weight='bold')
    
    ax_x0.plot(reg_sma, reg_x0, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_x0.plot(ea_reg_sma, ea_reg_x0, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_x0.plot(phot_sma, phot_x0, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_x0.set_xlabel('SMA (pixels)', fontsize=10)
    ax_x0.set_ylabel('X Center', fontsize=10)
    ax_x0.set_xlim(left=1.1)
    ax_x0.legend(fontsize=9)
    ax_x0.grid(True, alpha=0.3)
    ax_x0.set_title('X Center', fontsize=11, weight='bold')
    
    ax_y0.plot(reg_sma, reg_y0, 'o-', markersize=3, label='Regular', alpha=0.7)
    ax_y0.plot(ea_reg_sma, ea_reg_y0, 'd-', markersize=3, label='EA+Reg', alpha=0.7, color='purple')
    if photutils_iso:
        ax_y0.plot(phot_sma, phot_y0, '^-', markersize=3, label='Photutils', alpha=0.7)
    ax_y0.set_xlabel('SMA (pixels)', fontsize=10)
    ax_y0.set_ylabel('Y Center', fontsize=10)
    ax_y0.set_xlim(left=1.1)
    ax_y0.legend(fontsize=9)
    ax_y0.grid(True, alpha=0.3)
    ax_y0.set_title('Y Center', fontsize=11, weight='bold')
    
    plt.suptitle(f'M51 EA+Reg Comparison (EA+Reg: {speedup_reg_vs_ea:.2f}x vs Regular)',
                 fontsize=14, weight='bold', y=0.995)
    
    qa_path = os.path.join(os.path.dirname(__file__), 'm51_ea_comparison.png')
    plt.savefig(qa_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {qa_path}")
    
    return {
        'photutils_time': photutils_time,
        'regular_time': regular_time,
        'ea_reg_time': ea_reg_time,
        'speedup_reg_vs_ea': speedup_reg_vs_ea,
        'speedup_vs_photutils': speedup_vs_photutils if photutils_time else None
    }

if __name__ == "__main__":
    results = run_m51_ea_comparison()
