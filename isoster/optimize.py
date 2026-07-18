"""
Optimized Isophote Fitting Module

This module provides a streamlined, function-based implementation of elliptical isophote
fitting, optimized for performance while maintaining compatibility with photutils.isophote.

This file acts as a facade for the refactored modules, re-exporting their
public APIs at the historical ``isoster.optimize`` import path:
- .sampling: Data extraction and coordinate transformations.
- .fitting: Harmonic fitting and iterative loops.
- .driver: High-level image fitting API.
"""

from .driver import fit_image
from .fitting import (
    compute_deviations,
    compute_gradient,
    compute_parameter_errors,
    extract_forced_photometry,
    fit_isophote,
    sigma_clip,
)
from .sampling import extract_isophote_data

__all__ = [
    "compute_deviations",
    "compute_gradient",
    "compute_parameter_errors",
    "extract_forced_photometry",
    "extract_isophote_data",
    "fit_image",
    "fit_isophote",
    "sigma_clip",
]
