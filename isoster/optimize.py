"""
Optimized Isophote Fitting Module

This module provides a streamlined, function-based implementation of elliptical isophote
fitting, optimized for performance while maintaining compatibility with photutils.isophote.

This file now acts as a facade for the refactored modules:
- .sampling: Data extraction and coordinate transformations.
- .fitting: Harmonic fitting and iterative loops.
- .driver: High-level image fitting API.
"""

from .sampling import (
    get_elliptical_coordinates,
    extract_isophote_data
)

from .fitting import (
    fit_first_and_second_harmonics,
    harmonic_function,
    sigma_clip,
    compute_parameter_errors,
    compute_deviations,
    compute_gradient,
    fit_isophote
)

from .driver import (
    fit_central_pixel,
    fit_image
)
