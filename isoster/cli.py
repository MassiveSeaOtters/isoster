"""
Command Line Interface for ISOSTER.

This module provides the entry point for running isophote analysis from the command line.
It handles argument parsing, configuration loading, data loading, running the analysis,
and saving the results.

Usage::

    # Basic usage with CLI overrides
    isoster image.fits --output isophotes.fits --x0 100 --y0 100 --sma0 10

    # Advanced usage with YAML configuration file
    isoster image.fits --config config.yaml --output isophotes.fits

The ``--config`` YAML file accepts any field from ``IsosterConfig``. Example::

    # config.yaml
    sma0: 10.0
    eps: 0.3
    pa: 0.5
    use_eccentric_anomaly: true
    maxgerr: 0.5
    conver: 0.05
    simultaneous_harmonics: false
    harmonic_orders: [3, 4]
    full_photometry: false
    compute_cog: false

CLI flags (``--x0``, ``--y0``, ``--sma0``, etc.) override values from the YAML file.
See ``IsosterConfig`` in ``isoster/config.py`` for the full list of supported fields.
"""

import argparse

import numpy as np
import yaml
from astropy.io import fits

from .driver import fit_image
from .utils import (
    isophote_results_to_asdf,
    isophote_results_to_astropy_tables,
    isophote_results_to_fits,
)


def load_config(config_file):
    with open(config_file, "r") as f:
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
    parser.add_argument("--template", help="Template FITS file for forced photometry (applies geometry from template)")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = {}

    # Overrides
    if args.x0 is not None:
        config["x0"] = args.x0
    if args.y0 is not None:
        config["y0"] = args.y0
    if args.sma0 is not None:
        config["sma0"] = args.sma0
    if args.fix_center:
        config["fix_center"] = True
    if args.fix_eps:
        config["fix_eps"] = True
    if args.fix_pa:
        config["fix_pa"] = True

    # Load data
    with fits.open(args.image) as hdul:
        image = hdul[0].data
        if image is None and len(hdul) > 1:  # Maybe in extension 1?
            image = hdul[1].data
        if image is None:
            raise ValueError(f"No image data found in {args.image} (primary HDU has no data)")
    image = image.astype(np.float64)

    mask = None
    if args.mask:
        with fits.open(args.mask) as hdul:
            mask = hdul[0].data
            if mask is None and len(hdul) > 1:
                mask = hdul[1].data
            if mask is None:
                raise ValueError(f"No image data found in {args.mask} (primary HDU has no data)")
            mask = mask.astype(bool)

    # Run
    print("Running isophote analysis...")
    results = fit_image(image, mask, config, template=args.template)

    # Save results
    if args.output.endswith(".fits"):
        isophote_results_to_fits(results, args.output)
    elif args.output.endswith(".asdf"):
        isophote_results_to_asdf(results, args.output)
    else:
        tbl = isophote_results_to_astropy_tables(results)
        tbl.write(args.output, overwrite=True)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
