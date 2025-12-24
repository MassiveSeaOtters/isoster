"""
M51 ISOSTER Benchmark & QA
==========================

Comparison benchmark between ISOSTER and Photutils.isophote on M51 galaxy data.
Generates comprehensive QA figures and performance metrics.
"""

import os
import sys
import time
import numpy as np
from astropy.io import fits
import warnings

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from isoster.optimize import fit_image
from isoster.config import IsosterConfig
from isoster.model import build_ellipse_model as build_model_isoster
from isoster.plotting import plot_qa_summary
from isoster.utils import isophote_results_to_fits

# Try importing photutils
try:
    from photutils.isophote import Ellipse, EllipseGeometry
    from photutils.isophote import build_ellipse_model as build_model_photutils
    PHOTUTILS_AVAIL = True
except ImportError:
    PHOTUTILS_AVAIL = False
    print("Warning: photutils not found. Benchmark will run only isoster.")

def run_benchmark():
    print("=" * 80)
    print("ISOSTER M51 Benchmark & QA (Adaptive Integrator)")
    print("=" * 80)
    
    # 1. Load Data
    fits_path = os.path.join(os.path.dirname(__file__), 'M51.fits')
    if not os.path.exists(fits_path):
        print(f"Error: {fits_path} not found.")
        # Attempt download logic (simplified)
        try:
            from photutils.datasets import get_path
            from shutil import copyfile
            path = get_path('isophote/M51.fits', location='photutils-datasets', cache=True)
            copyfile(path, fits_path)
            print(f"Downloaded M51.fits")
        except:
            print("Could not download M51.fits. Please provide file.")
            return

    with fits.open(fits_path) as hdul:
        image = hdul[0].data
        
    # Generate Synthetic Mask
    h, w = image.shape
    mask = np.zeros_like(image, dtype=bool)
    rng = np.random.RandomState(42) # Fixed seed for reproducibility
    
    # Add 20 random masked blobs (avoiding center r < 10)
    cx, cy = w/2, h/2
    for _ in range(20):
        # Random position
        while True:
            mx, my = rng.randint(0, w), rng.randint(0, h)
            dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
            if dist > 10: # Ensure distance > 10 pixels
                break
        
        # Random size 10-30 pixels
        r_blob = rng.randint(10, 30)
        y, x = np.mgrid[0:h, 0:w]
        blob = ((x - mx)**2 + (y - my)**2) < r_blob**2
        mask |= blob
        
    print(f"Generated synthetic mask with {np.sum(mask)} bad pixels.")
        
    x0, y0 = w / 2.0, h / 2.0
    
    # Shared Control Parameters
    # Using float values to be compatible with both
    sma0 = 10.0
    minsma = 0.0
    maxsma = 275.0
    astep = 0.1
    eps = 0.2
    pa = 0.0
    
    # ---------------------------------------------------------
    # 2. Run Photutils (Reference)
    # ---------------------------------------------------------
    p_res = None
    p_model = None
    p_time = np.nan
    
    if PHOTUTILS_AVAIL:
        print(f"\n[Photutils] Running fit on {w}x{h} image (with mask)...")
        geometry = EllipseGeometry(x0=x0, y0=y0, sma=sma0, eps=eps, pa=pa)
        # Photutils expects image as MaskedArray to handle masks
        masked_image = np.ma.masked_array(image, mask=mask)
        ellipse = Ellipse(masked_image, geometry=geometry)
        
        t0 = time.time()
        # Suppress warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_res = ellipse.fit_image(sma0=sma0, minsma=minsma, maxsma=maxsma, step=astep)
        t1 = time.time()
        p_time = t1 - t0
        print(f"   Done in {p_time:.4f}s. Fitted {len(p_res)} isophotes.")
        
        # Build Reference Model
        # photutils build_ellipse_model takes ((h, w), isolist)
        p_model = build_model_photutils(image.shape, p_res)
        
    # ---------------------------------------------------------
    # 3. Run ISOSTER (Optimized)
    # ---------------------------------------------------------
    print(f"\n[ISOSTER] Running fit on {w}x{h} image (with mask)...")
    
    # Use Pydantic Config to be "close to photutils"
    # Photutils defaults: conver=0.05, minit=10, maxit=50, fflag=0.7
    cfg = IsosterConfig(
        x0=x0, y0=y0,
        sma0=sma0, minsma=minsma, maxsma=maxsma, astep=astep,
        eps=eps, pa=pa,
        conver=0.05, maxit=50,
        compute_errors=True,
        compute_deviations=True,

        full_photometry=False, # Not strictly needed for QA comparison, keep fast
        integrator='adaptive',
        lsb_sma_threshold=100.0
    )
    
    t0 = time.time()
    i_results_dict = fit_image(image, mask=mask, config=cfg)
    t1 = time.time()
    i_results_dict['mask'] = mask # Store for potential future use or debugging
    i_time = t1 - t0
    
    i_res = i_results_dict['isophotes']
    print(f"   Done in {i_time:.4f}s. Fitted {len(i_res)} isophotes.")
    
    # Build Optimized Model
    print("   Building 2D model...")
    i_model = build_model_isoster(image.shape, i_res)
    
    # Save Results
    out_fits = os.path.join(os.path.dirname(__file__), 'm51_isoster_results.fits')
    isophote_results_to_fits(i_results_dict, out_fits)
    
    # ---------------------------------------------------------
    # 4. Performance Comparison
    # ---------------------------------------------------------
    print("\nPerformance Summary")
    print("-" * 30)
    print(f"Photutils: {p_time:.4f}s")
    print(f"Isoster:   {i_time:.4f}s")
    if p_time and p_time > 0:
        print(f"Speedup:   {p_time/i_time:.2f}x")
    print("-" * 30)
    
    # ---------------------------------------------------------
    # 5. Generate QA Figure
    # ---------------------------------------------------------
    print("\nGenerating QA Figure...")
    qa_path = os.path.join(os.path.dirname(__file__), 'm51_qa_summary.png')
    
    plot_qa_summary(
        title=f"M51 Isophote Analysis (Adaptive+Mask, Speedup: {p_time/i_time:.1f}x)",
        image=image,
        mask=mask,
        isoster_model=i_model,
        isoster_res=i_res,
        photutils_res=p_res,
        filename=qa_path
    )

if __name__ == "__main__":
    run_benchmark()
