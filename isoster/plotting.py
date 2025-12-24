"""
Plotting utilities for isophote analysis visualization.

This module provides reusable plotting functions for comparing isophote
fitting results between photutils and the optimized implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MPLEllipse
import matplotlib.gridspec as gridspec

def normalize_angle(angle_rad):
    """
    Normalize angle in radians to [0, 180) degrees.
    Handles the 0/180 ambiguity for ellipses.
    """
    deg = np.degrees(angle_rad)
    return np.mod(deg, 180.0)

def plot_qa_summary(title, image, isoster_model, isoster_res, photutils_res=None, filename="qa_summary.png"):
    """
    Generate a detailed QA figure comparing isoster results with input and photutils.

    Parameters
    ----------
    title : str
        Figure title.
    image : 2D array
        Original input image.
    isoster_model : 2D array
        Reconstructed model from isoster isophotes.
    isoster_res : list of dict
        Results from isoster fitting.
    photutils_res : photutils.isophote.IsophoteList, optional
        Results from photutils fitting for comparison.
    filename : str
        Output filename.
    """
    # Create Figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(title, fontsize=20, weight='bold')
    
    # Outer GridSpec: 1 row, 2 columns (Images vs Profiles)
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.15)
    
    # --- Left Column: Images (3 rows) ---
    left_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[0], hspace=0.01)
    
    # Scaling for images (asinh)
    vmin, vmax = np.percentile(image[~np.isnan(image)], [1, 99])
    def norm(img):
        return np.arcsinh((img - vmin) / (vmax - vmin))

    # 1. Input Image + Isophotes
    ax_img = fig.add_subplot(left_gs[0])
    ax_img.imshow(norm(image), origin='lower', cmap='viridis')
    ax_img.set_ylabel('Y (pixels)', fontsize=12)
    # Overplot isophotes (sparse)
    sorted_iso = sorted(isoster_res, key=lambda x: x['sma'])
    # Plot every 10th isophote to avoid clutter, ensuring we cover the range
    step = max(1, len(sorted_iso) // 15)
    for iso in sorted_iso[::step]:
        if iso['sma'] < 1.0: continue
        # ellipse expects angle in degrees. isoster uses radians.
        # axis ratio b/a = 1 - eps
        b_over_a = 1.0 - iso['eps']
        ell = MPLEllipse((iso['x0'], iso['y0']), 
                         width=2*iso['sma'], 
                         height=2*iso['sma']*b_over_a,
                         angle=np.degrees(iso['pa']),
                         edgecolor='white', facecolor='none', linewidth=0.7, alpha=0.8)
        ax_img.add_patch(ell)
    
    ax_img.text(0.05, 0.95, "Input + Isophotes", transform=ax_img.transAxes, color='white', weight='bold', va='top')
    ax_img.set_xticks([])

    # 2. Model Image
    ax_mod = fig.add_subplot(left_gs[1], sharex=ax_img, sharey=ax_img)
    ax_mod.imshow(norm(isoster_model), origin='lower', cmap='viridis')
    ax_mod.text(0.05, 0.95, "Isoster Model", transform=ax_mod.transAxes, color='white', weight='bold', va='top')
    ax_mod.set_xticks([])
    ax_mod.set_ylabel('Y (pixels)', fontsize=12)

    # 3. Residual Image
    ax_res = fig.add_subplot(left_gs[2], sharex=ax_img, sharey=ax_img)
    residual = image - isoster_model
    # Residual scaling might need to be different, but keeping consistent usually helps
    # Or use linear centered around 0
    res_std = np.std(residual[~np.isnan(residual)])
    ax_res.imshow(residual, origin='lower', cmap='viridis', vmin=-3*res_std, vmax=3*res_std)
    ax_res.text(0.05, 0.95, "Residual", transform=ax_res.transAxes, color='white', weight='bold', va='top')
    ax_res.set_xlabel('X (pixels)', fontsize=12)
    ax_res.set_ylabel('Y (pixels)', fontsize=12)

    # --- Right Column: 1D Profiles (7 rows) ---
    # Rows: Intens, Diff, X0/Y0, AxisRatio, PA, A3/B3, A4/B4
    # Height ratios: Intensity gets 2.5x, others 1x
    right_gs = gridspec.GridSpecFromSubplotSpec(7, 1, subplot_spec=outer_gs[1], 
                                                height_ratios=[2.5, 1, 1, 1, 1, 1, 1], hspace=0.05)
    
    # Prepare Data
    i_sma = np.array([r['sma'] for r in isoster_res])
    i_intens = np.array([r['intens'] for r in isoster_res])
    i_eps = np.array([r['eps'] for r in isoster_res])
    i_pa = np.array([r['pa'] for r in isoster_res])
    i_x0 = np.array([r['x0'] for r in isoster_res])
    i_y0 = np.array([r['y0'] for r in isoster_res])
    i_a3 = np.array([r.get('a3', 0.0) for r in isoster_res])
    i_b3 = np.array([r.get('b3', 0.0) for r in isoster_res])
    i_a4 = np.array([r.get('a4', 0.0) for r in isoster_res])
    i_b4 = np.array([r.get('b4', 0.0) for r in isoster_res])
    i_stop = np.array([r['stop_code'] for r in isoster_res])

    # Filter for SMA > 1.5
    mask = i_sma > 1.5
    # Also valid stops? Usually we plot everything but maybe mark bad ones?
    # Let's stick to the SMA filter requested.
    
    x_axis = i_sma[mask]**0.25
    
    # Helper for adding photutils comparison
    p_intens_interp = None
    if photutils_res:
        p_sma = np.array([iso.sma for iso in photutils_res])
        p_intens = np.array([iso.intens for iso in photutils_res])
        p_eps = np.array([iso.eps for iso in photutils_res])
        p_pa = np.array([iso.pa for iso in photutils_res])
        p_x0 = np.array([iso.x0 for iso in photutils_res])
        p_y0 = np.array([iso.y0 for iso in photutils_res])
        p_a3 = np.array([iso.a3 for iso in photutils_res])
        p_b3 = np.array([iso.b3 for iso in photutils_res])
        p_a4 = np.array([iso.a4 for iso in photutils_res])
        p_b4 = np.array([iso.b4 for iso in photutils_res])

        # Interpolate photutils to isoster SMA grid for difference calculation
        # Use log intensity for interpolation to handle dynamic range
        p_intens_interp = np.interp(i_sma[mask], p_sma, p_intens)

    axes_profiles = []

    # 1. Surface Brightness
    ax1 = fig.add_subplot(right_gs[0])
    ax1.plot(x_axis, np.log10(i_intens[mask]), 'r-', label='Isoster', lw=2)
    if photutils_res:
        ax1.plot(p_sma[p_sma>1.5]**0.25, np.log10(p_intens[p_sma>1.5]), 'k--', label='Photutils', lw=1.5)
    ax1.set_ylabel(r'$\log_{10}(I)$', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    axes_profiles.append(ax1)

    # 2. Relative Difference
    ax2 = fig.add_subplot(right_gs[1], sharex=ax1)
    if photutils_res:
        # ((iso - phot) / phot) * 100
        diff = ((i_intens[mask] - p_intens_interp) / p_intens_interp) * 100.0
        ax2.plot(x_axis, diff, 'b-', lw=1)
        ax2.axhline(0, color='gray', ls=':')
        # Limit Y to something reasonable, or auto?
        # User said "cover min-max of measured properties".
        # Outliers can be huge at edges. Let's trust matplotlib but maybe clip extreme?
    ax2.set_ylabel(r'$\Delta I/I$ (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])
    axes_profiles.append(ax2)

    # 3. Center X & Y
    ax3 = fig.add_subplot(right_gs[2], sharex=ax1)
    ax3.plot(x_axis, i_x0[mask], 'r-', label='X')
    ax3.plot(x_axis, i_y0[mask], 'b-', label='Y')
    if photutils_res:
        ax3.plot(p_sma[p_sma>1.5]**0.25, p_x0[p_sma>1.5], 'r--', alpha=0.5)
        ax3.plot(p_sma[p_sma>1.5]**0.25, p_y0[p_sma>1.5], 'b--', alpha=0.5)
    ax3.set_ylabel('Center', fontsize=10)
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels([])

    # 4. Axis Ratio (1 - eps)
    ax4 = fig.add_subplot(right_gs[3], sharex=ax1)
    ax4.plot(x_axis, 1.0 - i_eps[mask], 'r-')
    if photutils_res:
        ax4.plot(p_sma[p_sma>1.5]**0.25, 1.0 - p_eps[p_sma>1.5], 'k--', alpha=0.5)
    ax4.set_ylabel('b/a', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticklabels([])

    # 5. Position Angle
    ax5 = fig.add_subplot(right_gs[4], sharex=ax1)
    ax5.plot(x_axis, normalize_angle(i_pa[mask]), 'r-')
    if photutils_res:
        ax5.plot(p_sma[p_sma>1.5]**0.25, normalize_angle(p_pa[p_sma>1.5]), 'k--', alpha=0.5)
    ax5.set_ylabel('PA (deg)', fontsize=10)
    ax5.set_ylim(0, 180)
    ax5.set_yticks([0, 45, 90, 135, 180])
    ax5.grid(True, alpha=0.3)
    ax5.set_xticklabels([])

    # 6. A3 / B3
    ax6 = fig.add_subplot(right_gs[5], sharex=ax1)
    ax6.plot(x_axis, i_a3[mask], 'r-', label='A3')
    ax6.plot(x_axis, i_b3[mask], 'b-', label='B3')
    if photutils_res:
        ax6.plot(p_sma[p_sma>1.5]**0.25, p_a3[p_sma>1.5], 'r--', alpha=0.5)
        ax6.plot(p_sma[p_sma>1.5]**0.25, p_b3[p_sma>1.5], 'b--', alpha=0.5)
    ax6.set_ylabel('Harmonics 3', fontsize=8)
    ax6.legend(fontsize=6, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_xticklabels([])

    # 7. A4 / B4
    ax7 = fig.add_subplot(right_gs[6], sharex=ax1)
    ax7.plot(x_axis, i_a4[mask], 'r-', label='A4')
    ax7.plot(x_axis, i_b4[mask], 'b-', label='B4')
    if photutils_res:
        ax7.plot(p_sma[p_sma>1.5]**0.25, p_a4[p_sma>1.5], 'r--', alpha=0.5)
        ax7.plot(p_sma[p_sma>1.5]**0.25, p_b4[p_sma>1.5], 'b--', alpha=0.5)
    ax7.set_ylabel('Harmonics 4', fontsize=8)
    ax7.legend(fontsize=6, loc='best')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlabel(r'SMA$^{0.25}$ (pixel$^{0.25}$)', fontsize=12)

    # Save
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved detailed QA figure to {filename}")
