import os
import sys
import numpy as np
from astropy.io import fits

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig
from isoster.utils import isophote_results_to_fits

def test_pydantic_config():
    fits_path = os.path.join(os.path.dirname(__file__), 'M51.fits')
    if not os.path.exists(fits_path):
        print("M51.fits not found.")
        return

    with fits.open(fits_path) as hdul:
        image = hdul[0].data
    
    # New Way: Use IsosterConfig
    try:
        config = IsosterConfig(
            sma0=10.0,
            minsma=0.0, 
            maxsma=50.0,
            astep=0.1,
            full_photometry=True,  # Enable flux integration
            debug=False
        )
        print("Configuration successfully validated.")
        print(config)
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return

    print("Running fit with IsosterConfig...")
    results = fit_image(image, None, config)
    isos = results['isophotes']
    
    print(f"Fitted {len(isos)} isophotes.")
    if len(isos) > 0:
        print(f"First SMA: {isos[0]['sma']:.2f}")
        print(f"Last SMA:  {isos[-1]['sma']:.2f}")

    # Test saving
    out_path = os.path.join(os.path.dirname(__file__), 'm51_pydantic_results.fits')
    isophote_results_to_fits(results, out_path)
    print(f"Results saved to {out_path}")

    # Validation Failure Test
    print("\nTesting validation failure (negative maxit):")
    try:
        bad_config = IsosterConfig(maxit=-5)
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    test_pydantic_config()
