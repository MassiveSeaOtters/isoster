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
    # Create Figure with constrained layout for better spacing control
    fig = plt.figure(figsize=(20, 16))
    
    # Outer GridSpec: 1 row, 2 columns (Images vs Profiles)
    # Adjust top margin to bring title closer
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.15, top=0.93, bottom=0.05, left=0.05, right=0.95)
    
    fig.suptitle(title, fontsize=20, weight='bold', y=0.98) # Push title up
    
    # --- Left Column: Images (3 rows) ---
    left_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_gs[0], hspace=0.01)
    
    # Scaling for images (asinh)
    vmin, vmax = np.percentile(image[~np.isnan(image)], [1, 99])
    def norm(img):
        return np.arcsinh((img - vmin) / (vmax - vmin))

    # 1. Input Image + Isophotes
    ax_img = fig.add_subplot(left_gs[0])
    ax_img.imshow(norm(image), origin='lower', cmap='viridis')
    ax_img.set_ylabel('Y (pixels)', fontsize=14)
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
    
    ax_img.text(0.05, 0.95, "Input + Isophotes", transform=ax_img.transAxes, color='white', weight='bold', va='top', fontsize=14)
    ax_img.set_xticks([])

    # 2. Model Image
    ax_mod = fig.add_subplot(left_gs[1], sharex=ax_img, sharey=ax_img)
    ax_mod.imshow(norm(isoster_model), origin='lower', cmap='viridis')
    ax_mod.text(0.05, 0.95, "Isoster Model", transform=ax_mod.transAxes, color='white', weight='bold', va='top', fontsize=14)
    ax_mod.set_xticks([])
    ax_mod.set_ylabel('Y (pixels)', fontsize=14)

    # 3. Residual Image
    ax_res = fig.add_subplot(left_gs[2], sharex=ax_img, sharey=ax_img)
    residual = image - isoster_model
    # Residual scaling
    res_std = np.std(residual[~np.isnan(residual)])
    ax_res.imshow(residual, origin='lower', cmap='viridis', vmin=-3*res_std, vmax=3*res_std)
    ax_res.text(0.05, 0.95, "Residual", transform=ax_res.transAxes, color='white', weight='bold', va='top', fontsize=14)
    ax_res.set_xlabel('X (pixels)', fontsize=14)
    ax_res.set_ylabel('Y (pixels)', fontsize=14)

    # --- Right Column: 1D Profiles (7 rows) ---
    # Rows: Intens, Diff, X0/Y0, AxisRatio, PA, A3/B3, A4/B4
    # Height ratios: Intensity gets 2.5x, others 1x
    right_gs = gridspec.GridSpecFromSubplotSpec(7, 1, subplot_spec=outer_gs[1], 
                                                height_ratios=[2.5, 1, 1, 1, 1, 1, 1], hspace=0.0)
    
    # Prepare Data
    # Helper to clean nans
    def get_arr(key, default=0.0):
        return np.array([r.get(key, default) for r in isoster_res])

    i_sma = get_arr('sma')
    i_intens = get_arr('intens', np.nan)
    i_intens_err = get_arr('intens_err', np.nan)
    i_eps = get_arr('eps')
    i_eps_err = get_arr('eps_err', np.nan)
    i_pa = get_arr('pa')
    i_pa_err = get_arr('pa_err', np.nan)
    i_x0 = get_arr('x0')
    i_x0_err = get_arr('x0_err', np.nan)
    i_y0 = get_arr('y0')
    i_y0_err = get_arr('y0_err', np.nan)
    i_a3 = get_arr('a3')
    i_b3 = get_arr('b3')
    i_a4 = get_arr('a4')
    i_b4 = get_arr('b4')

    # Filter for SMA > 1.5
    mask = i_sma > 1.5
    
    x_axis = i_sma[mask]**0.25
    
    # Photutils Data
    p_intens_interp = None
    if photutils_res:
        p_sma = np.array([iso.sma for iso in photutils_res])
        p_mask = p_sma > 1.5
        p_x_axis = p_sma[p_mask]**0.25
        
        p_intens = np.array([iso.intens for iso in photutils_res])
        p_intens_err = np.array([iso.int_err for iso in photutils_res])
        p_eps = np.array([iso.eps for iso in photutils_res])
        p_eps_err = np.array([iso.ellip_err for iso in photutils_res])
        p_pa = np.array([iso.pa for iso in photutils_res])
        p_pa_err = np.array([iso.pa_err for iso in photutils_res])
        p_x0 = np.array([iso.x0 for iso in photutils_res])
        p_x0_err = np.array([iso.x0_err for iso in photutils_res])
        p_y0 = np.array([iso.y0 for iso in photutils_res])
        p_y0_err = np.array([iso.y0_err for iso in photutils_res])
        p_a3 = np.array([iso.a3 for iso in photutils_res])
        p_b3 = np.array([iso.b3 for iso in photutils_res])
        p_a4 = np.array([iso.a4 for iso in photutils_res])
        p_b4 = np.array([iso.b4 for iso in photutils_res])

        # Interpolate photutils to isoster SMA grid for difference calculation
        p_intens_interp = np.interp(i_sma[mask], p_sma, p_intens)

    axes_profiles = []
    
    # Common plot settings
    scatter_kwargs_iso = {'fmt': 'o', 'color': 'red', 'markersize': 4, 'label': 'Isoster', 'elinewidth': 1}
    scatter_kwargs_phot = {'fmt': 'o', 'markerfacecolor': 'none', 'markeredgecolor': 'black', 'color': 'black', 'markersize': 4, 'label': 'Photutils', 'elinewidth': 1, 'alpha': 0.6}

    # 1. Surface Brightness
    ax1 = fig.add_subplot(right_gs[0])
    
    # Log10 intensity error propagation: err_log = err / (I * ln(10))
    y_iso = np.log10(i_intens[mask])
    yerr_iso = i_intens_err[mask] / (i_intens[mask] * np.log(10))
    
    ax1.errorbar(x_axis, y_iso, yerr=yerr_iso, **scatter_kwargs_iso)
    
    if photutils_res:
        y_phot = np.log10(p_intens[p_mask])
        yerr_phot = p_intens_err[p_mask] / (p_intens[p_mask] * np.log(10))
        ax1.errorbar(p_x_axis, y_phot, yerr=yerr_phot, **scatter_kwargs_phot)
        
    ax1.set_ylabel(r'$\log_{10}(I)$', fontsize=14)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    axes_profiles.append(ax1)

    # 2. Relative Difference
    ax2 = fig.add_subplot(right_gs[1], sharex=ax1)
    if photutils_res:
        # ((iso - phot) / phot) * 100
        diff = ((i_intens[mask] - p_intens_interp) / p_intens_interp) * 100.0
        ax2.plot(x_axis, diff, 'b-', lw=1) # Keep as line since it's a diff? User said "1-D profiles... scatter", but usually diff is line. Let's assume diff is line for clarity or scatter if consistent. Let's use line for diff as it is dense.
        # Actually user said "Instead of using lines for the 1-D results (plot), use a scatter plot". Diff is derived.
        # Let's stick to scatter for consistency? No, diff is usually better as line.
        ax2.axhline(0, color='gray', ls=':')
    ax2.set_ylabel(r'$\Delta I/I$ (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])
    axes_profiles.append(ax2)

    # 3. Center X & Y
    ax3 = fig.add_subplot(right_gs[2], sharex=ax1)
    # Isoster X
    ax3.errorbar(x_axis, i_x0[mask], yerr=i_x0_err[mask], fmt='o', color='red', markersize=4, label='X', elinewidth=1)
    # Isoster Y (use different symbol or color?) - User said "1-D profiles of the galaxy center in X & Y". 
    # Usually X and Y are plotted together. Let's use blue for Y.
    ax3.errorbar(x_axis, i_y0[mask], yerr=i_y0_err[mask], fmt='s', color='blue', markersize=4, label='Y', elinewidth=1)
    
    if photutils_res:
        ax3.errorbar(p_x_axis, p_x0[p_mask], yerr=p_x0_err[p_mask], fmt='o', markerfacecolor='none', markeredgecolor='black', color='black', alpha=0.5, markersize=4)
        ax3.errorbar(p_x_axis, p_y0[p_mask], yerr=p_y0_err[p_mask], fmt='s', markerfacecolor='none', markeredgecolor='gray', color='gray', alpha=0.5, markersize=4)
        
    ax3.set_ylabel('Center', fontsize=12)
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels([])

    # 4. Axis Ratio (1 - eps)
    ax4 = fig.add_subplot(right_gs[3], sharex=ax1)
    # Error prop: err(1-eps) = err(eps)
    ax4.errorbar(x_axis, 1.0 - i_eps[mask], yerr=i_eps_err[mask], **scatter_kwargs_iso)
    if photutils_res:
        ax4.errorbar(p_x_axis, 1.0 - p_eps[p_mask], yerr=p_eps_err[p_mask], **scatter_kwargs_phot)
        
    ax4.set_ylabel('b/a', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticklabels([])

    # 5. Position Angle
    ax5 = fig.add_subplot(right_gs[4], sharex=ax1)
    # Need to handle PA wrapping for display if needed, but normalize_angle handles 0-180
    y_pa_iso = normalize_angle(i_pa[mask])
    # PA error is in degrees? isoster returns radians. normalize_angle converts to degrees.
    # So error needs to be in degrees.
    yerr_pa_iso = np.degrees(i_pa_err[mask])
    
    ax5.errorbar(x_axis, y_pa_iso, yerr=yerr_pa_iso, **scatter_kwargs_iso)
    
    if photutils_res:
        y_pa_phot = normalize_angle(p_pa[p_mask]) # Photutils is radians? Yes.
        yerr_pa_phot = np.degrees(p_pa_err[p_mask])
        ax5.errorbar(p_x_axis, y_pa_phot, yerr=yerr_pa_phot, **scatter_kwargs_phot)

    ax5.set_ylabel('PA (deg)', fontsize=12)
    ax5.set_ylim(0, 180)
    ax5.set_yticks([0, 45, 90, 135, 180])
    ax5.grid(True, alpha=0.3)
    ax5.set_xticklabels([])

    # 6. A3 / B3
    ax6 = fig.add_subplot(right_gs[5], sharex=ax1)
    ax6.scatter(x_axis, i_a3[mask], c='red', marker='o', s=16, label='A3')
    ax6.scatter(x_axis, i_b3[mask], c='blue', marker='o', s=16, label='B3')
    if photutils_res:
        ax6.scatter(p_x_axis, p_a3[p_mask], facecolors='none', edgecolors='red', marker='o', s=16, alpha=0.5)
        ax6.scatter(p_x_axis, p_b3[p_mask], facecolors='none', edgecolors='blue', marker='o', s=16, alpha=0.5)
    ax6.set_ylabel('Harmonics 3', fontsize=10)
    ax6.legend(fontsize=8, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_xticklabels([])

    # 7. A4 / B4
    ax7 = fig.add_subplot(right_gs[6], sharex=ax1)
    ax7.scatter(x_axis, i_a4[mask], c='red', marker='o', s=16, label='A4')
    ax7.scatter(x_axis, i_b4[mask], c='blue', marker='o', s=16, label='B4')
    if photutils_res:
        ax7.scatter(p_x_axis, p_a4[p_mask], facecolors='none', edgecolors='red', marker='o', s=16, alpha=0.5)
        ax7.scatter(p_x_axis, p_b4[p_mask], facecolors='none', edgecolors='blue', marker='o', s=16, alpha=0.5)
    ax7.set_ylabel('Harmonics 4', fontsize=10)
    ax7.legend(fontsize=8, loc='best')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlabel(r'SMA$^{0.25}$ (pixel$^{0.25}$)', fontsize=14)
    # Ensure tick labels are visible
    ax7.tick_params(labelbottom=True)

    # Save
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved detailed QA figure to {filename}")
