
"""
Command Line Interface for ISOSTER.

This module provides the entry point for running isophote analysis from the command line.
It handles argument parsing, configuration loading, data loading, running the analysis,
and saving the results.
"""

import argparse
import yaml
import numpy as np
from astropy.io import fits
from astropy.table import Table
from .optimize import fit_image
from .utils import isophote_results_to_fits, isophote_results_to_astropy_tables

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run isophote analysis.")
    parser.add_argument("image", help="Input image FITS file.")
    parser.add_argument("--mask", help="Input mask FITS file (optional).")
    parser.add_argument("--config", help="Configuration YAML file.")
    parser.add_argument("--output", help="Output table file (e.g. isophotes.fits or .csv).", default="isophotes.csv")
    
    # CLI overrides
    parser.add_argument("--x0", type=float, help="Center X")
    parser.add_argument("--y0", type=float, help="Center Y")
    parser.add_argument("--sma0", type=float, help="Initial SMA")
    parser.add_argument("--fix_center", action="store_true", help="Fix center")
    parser.add_argument("--fix_eps", action="store_true", help="Fix ellipticity")
    parser.add_argument("--fix_pa", action="store_true", help="Fix position angle")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
        
    # Overrides
    if args.x0 is not None: config['x0'] = args.x0
    if args.y0 is not None: config['y0'] = args.y0
    if args.sma0 is not None: config['sma0'] = args.sma0
    if args.fix_center: config['fix_center'] = True
    if args.fix_eps: config['fix_eps'] = True
    if args.fix_pa: config['fix_pa'] = True
    
    # Load data
    with fits.open(args.image) as hdul:
        image = hdul[0].data
        if image is None: # Maybe in extension 1?
            image = hdul[1].data
            
    mask = None
    if args.mask:
        with fits.open(args.mask) as hdul:
            mask = hdul[0].data
            if mask is None:
                mask = hdul[1].data
            mask = mask.astype(bool)
            
    # Run
    print("Running isophote analysis...")
    results = fit_image(image, mask, config)
    
    # Save results
    if args.output.endswith('.fits'):
        isophote_results_to_fits(results, args.output)
    else:
        tbl = isophote_results_to_astropy_tables(results)
        tbl.write(args.output, overwrite=True)
    
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
