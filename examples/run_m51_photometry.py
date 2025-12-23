import os
import sys
import numpy as np
from astropy.io import fits

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image

def test_photometry():
    fits_path = os.path.join(os.path.dirname(__file__), 'M51.fits')
    if not os.path.exists(fits_path):
        print("M51.fits not found.")
        return

    with fits.open(fits_path) as hdul:
        image = hdul[0].data
    
    config = {
        'sma0': 10.0, 'minsma': 0.0, 'maxsma': 50.0,
        'full_photometry': True,
        'debug': False
    }
    
    print("Running fit with full_photometry=True...")
    results = fit_image(image, None, config)
    isos = results['isophotes']
    
    print(f"{'SMA':>10} | {'TFLUX_E':>12} | {'NPIX_E':>8} | {'TFLUX_C':>12} | {'NPIX_C':>8}")
    print("-" * 60)
    for iso in isos[::10]:
        print(f"{iso['sma']:10.2f} | {iso['tflux_e']:12.2f} | {iso['npix_e']:8} | {iso['tflux_c']:12.2f} | {iso['npix_c']:8}")

if __name__ == "__main__":
    test_photometry()
