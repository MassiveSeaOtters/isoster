import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.model import build_ellipse_model

def test_model_building():
    fits_path = os.path.join(os.path.dirname(__file__), 'M51.fits')
    if not os.path.exists(fits_path):
        print("M51.fits not found.")
        return

    with fits.open(fits_path) as hdul:
        image = hdul[0].data
    
    config = {
        'x0': image.shape[1]/2, 'y0': image.shape[0]/2,
        'sma0': 10.0, 'minsma': 0.0, 'maxsma': 100.0,
        'astep': 0.1,
        'compute_deviations': True
    }
    
    print("Fitting isophotes to M51...")
    results = fit_image(image, None, config)
    
    print(f"Building 2D model for image shape {image.shape}...")
    model = build_ellipse_model(image.shape, results['isophotes'])
    
    # Save model to FITS
    model_fits = os.path.join(os.path.dirname(__file__), 'm51_model.fits')
    fits.writeto(model_fits, model, overwrite=True)
    print(f"Saved model to {model_fits}")
    
    # Plot Comparison
    plot_path = os.path.join(os.path.dirname(__file__), 'm51_model_comparison.png')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    vmin, vmax = np.percentile(image, [1, 99])
    norm_img = np.arcsinh((image - vmin) / (vmax - vmin))
    norm_model = np.arcsinh((model - vmin) / (vmax - vmin))
    residual = image - model
    norm_res = residual / np.std(residual)
    
    axes[0].imshow(norm_img, origin='lower', cmap='gray')
    axes[0].set_title('Original Image (M51)')
    
    axes[1].imshow(norm_model, origin='lower', cmap='gray')
    axes[1].set_title('Reconstructed Model')
    
    axes[2].imshow(norm_res, origin='lower', cmap='gray', vmin=-3, vmax=3)
    axes[2].set_title('Residual (Image - Model)')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot to {plot_path}")

if __name__ == "__main__":
    test_model_building()
