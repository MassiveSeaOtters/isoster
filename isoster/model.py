"""
Image reconstruction from fitted isophote profiles.

This module provides functions to reconstruct 2D galaxy images from isophote
fitting results using radial interpolation.
"""

import re

import numpy as np
from scipy.interpolate import interp1d


def build_isoster_model(
    image_shape,
    isophote_results,
    fill=0.0,
    interp_kind="linear",
    use_harmonics=True,
    harmonic_orders=None,
    use_eccentric_anomaly=None,
):
    """
    Reconstruct a 2D galaxy image model from fitted isophotes.

    This implementation uses radial interpolation between isophotes to create
    a smooth intensity profile, including higher-order harmonic deviations if
    available. This is significantly more accurate than layer-by-layer filling.

    Algorithm:
    1. For each pixel, compute its elliptical radius using the local geometry
    2. Interpolate intensity between adjacent isophotes
    3. Apply harmonic deviations (a3, b3, a4, b4, ...) to modify elliptical radius
    4. Fill regions outside fitted range with specified value

    Parameters
    ----------
    image_shape : tuple
        The (height, width) of the output image.
    isophote_results : list of dict
        The 'isophotes' list from fit_image() results. Each dict should contain:
        - 'sma': semi-major axis length
        - 'x0', 'y0': center coordinates
        - 'eps': ellipticity
        - 'pa': position angle
        - 'intens': mean intensity
        - Optional: 'a3', 'b3', 'a4', 'b4', ...: harmonic deviations
    fill : float, optional
        Value to fill pixels outside the largest isophote (default: 0.0)
    interp_kind : str, optional
        Interpolation method for radial intensity profile. Options:
        - 'linear': Linear interpolation (default, faster)
        - 'cubic': Cubic spline interpolation (smoother)
    use_harmonics : bool, optional
        If True (default), include harmonic deviations in the model reconstruction.
        Harmonics modify the effective elliptical radius to capture isophote shape
        deviations (diskiness, boxiness, etc.).
    harmonic_orders : list of int or None, optional
        Harmonic orders to include. If None (default), auto-detects available
        orders by scanning isophote keys for ``a{n}`` patterns where n >= 3.
        Can be explicitly set to e.g. [3, 4, 5, 6] to override auto-detection.
    use_eccentric_anomaly : bool or None, optional
        Whether harmonics were fitted in eccentric anomaly (ψ) space. If None
        (default), auto-detects from the ``use_eccentric_anomaly`` key stored
        in the isophote results by ``fit_isophote()``. Set explicitly to
        override auto-detection. When True, uses ψ = arctan2(y_rot/(1-eps),
        x_rot) instead of φ = arctan2(y_rot, x_rot) for harmonic evaluation.

    Returns
    -------
    model : ndarray
        The reconstructed 2D image model of shape image_shape.

    Notes
    -----
    - For pixels outside the outermost isophote or inside the innermost,
      uses the boundary isophote's parameters with no extrapolation.
    - Harmonic deviations are applied as radial corrections:
      r_corrected = r * (1 - Σₙ [aₙ·sin(nθ) + bₙ·cos(nθ)])
      where θ is the position angle on the ellipse. A positive
      deviation (the isophote bulges outward) shrinks the corrected
      radius, so the pixel samples the higher intensity of the
      bulging isophote.
    - The function uses local geometry interpolation at each radius to
      handle varying ellipticity and position angle profiles.

    Examples
    --------
    >>> results = isoster.fit_image(image, mask, config)
    >>> model = build_isoster_model(image.shape, results['isophotes'])
    >>> residual = image - model
    """
    if harmonic_orders is None:
        # Auto-detect available harmonic orders from isophote keys
        detected_orders = set()
        for iso in isophote_results:
            for key in iso:
                match = re.match(r"^a(\d+)$", key)
                if match:
                    n = int(match.group(1))
                    if n >= 3:
                        detected_orders.add(n)
        harmonic_orders = sorted(detected_orders) if detected_orders else []

    h, w = image_shape
    model = np.full((h, w), fill, dtype=np.float64)

    # Filter valid isophotes: require finite values in all geometry and intensity columns
    _required_keys = ("intens", "x0", "y0", "eps", "pa")
    valid_isos = [
        iso
        for iso in isophote_results
        if iso["sma"] > 0 and all(np.isfinite(iso.get(k, np.nan)) for k in _required_keys)
    ]

    if len(valid_isos) == 0:
        # No valid isophotes, return fill value
        return model

    # Sort by SMA
    sorted_isos = sorted(valid_isos, key=lambda x: x["sma"])

    # Handle central pixel if present
    central_iso = [iso for iso in isophote_results if iso["sma"] == 0]
    if central_iso:
        iso = central_iso[0]
        x0, y0 = int(np.round(iso["x0"])), int(np.round(iso["y0"]))
        if 0 <= y0 < h and 0 <= x0 < w:
            model[y0, x0] = iso["intens"]

    # Create coordinate grids
    y_grid, x_grid = np.ogrid[:h, :w]
    y_grid = y_grid.astype(np.float64)
    x_grid = x_grid.astype(np.float64)

    # Extract SMA values and intensities for interpolation
    sma_values = np.array([iso["sma"] for iso in sorted_isos])
    intens_values = np.array([iso["intens"] for iso in sorted_isos])

    # Create intensity interpolator
    if len(sma_values) == 1:
        # Only one isophote - use constant value within SMA, fill outside
        sma_single = sma_values[0]
        intens_single = intens_values[0]

        def intens_interp(r):
            result = np.where(r <= sma_single, intens_single, fill)
            return result
    else:
        # Interpolate, with boundary handling
        intens_interp = interp1d(
            sma_values, intens_values, kind=interp_kind, bounds_error=False, fill_value=(intens_values[0], fill)
        )

    # For each pixel, we need to:
    # 1. Find the local geometry (x0, y0, eps, pa) at that elliptical radius
    # 2. Compute elliptical radius
    # 3. Interpolate intensity

    # To handle varying geometry, we'll process the image in annular regions
    # and use interpolated geometry for each region

    # Interpolate geometry parameters
    x0_interp = interp1d(
        sma_values,
        [iso["x0"] for iso in sorted_isos],
        kind="linear",
        bounds_error=False,
        fill_value=([iso["x0"] for iso in sorted_isos][0], [iso["x0"] for iso in sorted_isos][-1]),
    )
    y0_interp = interp1d(
        sma_values,
        [iso["y0"] for iso in sorted_isos],
        kind="linear",
        bounds_error=False,
        fill_value=([iso["y0"] for iso in sorted_isos][0], [iso["y0"] for iso in sorted_isos][-1]),
    )
    eps_interp = interp1d(
        sma_values,
        [iso["eps"] for iso in sorted_isos],
        kind="linear",
        bounds_error=False,
        fill_value=([iso["eps"] for iso in sorted_isos][0], [iso["eps"] for iso in sorted_isos][-1]),
    )
    pa_raw = np.array([iso["pa"] for iso in sorted_isos])
    pa_unwrapped = np.unwrap(pa_raw, period=np.pi)
    pa_interp = interp1d(
        sma_values, pa_unwrapped, kind="linear", bounds_error=False, fill_value=(pa_unwrapped[0], pa_unwrapped[-1])
    )

    # Use outer isophote geometry as initial guess to find approximate radii
    outer_iso = sorted_isos[-1]
    x0_out, y0_out = outer_iso["x0"], outer_iso["y0"]
    eps_out, pa_out = outer_iso["eps"], outer_iso["pa"]

    # Compute approximate elliptical radius using outer geometry
    dx = x_grid - x0_out
    dy = y_grid - y0_out
    cos_pa, sin_pa = np.cos(pa_out), np.sin(pa_out)
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    r_ell_approx = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps_out)) ** 2)

    # For better accuracy, iterate to find elliptical radius using local geometry
    # (This is important when eps or pa vary significantly with radius)
    max_iterations = 3
    r_ell = r_ell_approx.copy()

    for iteration in range(max_iterations):
        # Get local geometry at current radius estimate
        r_ell_flat = r_ell.ravel()
        x0_local = x0_interp(r_ell_flat).reshape(r_ell.shape)
        y0_local = y0_interp(r_ell_flat).reshape(r_ell.shape)
        eps_local = eps_interp(r_ell_flat).reshape(r_ell.shape)
        pa_local = pa_interp(r_ell_flat).reshape(r_ell.shape)

        # Recompute elliptical radius with local geometry
        dx = x_grid - x0_local
        dy = y_grid - y0_local
        cos_pa_local = np.cos(pa_local)
        sin_pa_local = np.sin(pa_local)
        x_rot = dx * cos_pa_local + dy * sin_pa_local
        y_rot = -dx * sin_pa_local + dy * cos_pa_local

        # Avoid division by zero for eps ~= 1
        eps_safe = np.clip(eps_local, 0, 0.99)
        r_ell_new = np.sqrt(x_rot**2 + (y_rot / (1.0 - eps_safe)) ** 2)

        # Check convergence
        if np.max(np.abs(r_ell_new - r_ell)) < 0.1:
            r_ell = r_ell_new
            break
        r_ell = r_ell_new

    # Apply harmonic deviations if requested
    if use_harmonics and harmonic_orders and len(sorted_isos) >= 2:
        # Check if any harmonic coefficients are present
        has_harmonics = any(f"a{n}" in iso for n in harmonic_orders for iso in sorted_isos)

        if has_harmonics:
            # Determine angle space: eccentric anomaly (ψ) or position angle (φ)
            # If use_eccentric_anomaly is None, auto-detect from isophote results
            ea_mode = use_eccentric_anomaly
            if ea_mode is None:
                ea_mode = any(iso.get("use_eccentric_anomaly", False) for iso in sorted_isos)

            if ea_mode:
                # Eccentric anomaly: ψ = arctan2(y_rot / (1 - eps), x_rot)
                # This matches the angle space used during fitting in EA mode
                theta = np.arctan2(y_rot / (1.0 - eps_safe), x_rot)
            else:
                # Position angle: φ = arctan2(y_rot, x_rot)
                theta = np.arctan2(y_rot, x_rot)

            # Create interpolators for harmonic coefficients
            harm_interps = {}
            for n in harmonic_orders:
                an_key, bn_key = f"a{n}", f"b{n}"
                if any(an_key in iso for iso in sorted_isos):
                    an_values = np.array([iso.get(an_key, 0.0) for iso in sorted_isos])
                    bn_values = np.array([iso.get(bn_key, 0.0) for iso in sorted_isos])
                    an_values = np.where(np.isfinite(an_values), an_values, 0.0)
                    bn_values = np.where(np.isfinite(bn_values), bn_values, 0.0)
                    harm_interps[n] = (
                        interp1d(
                            sma_values,
                            an_values,
                            kind="linear",
                            bounds_error=False,
                            fill_value=(an_values[0], an_values[-1]),
                        ),
                        interp1d(
                            sma_values,
                            bn_values,
                            kind="linear",
                            bounds_error=False,
                            fill_value=(bn_values[0], bn_values[-1]),
                        ),
                    )

            # Compute harmonic deviation: Δr/r = Σₙ [aₙ·sin(nθ) + bₙ·cos(nθ)]
            # The harmonic coefficients are already normalized by (gradient * sma)
            # so Δr = (aₙ * sma) * sin(nθ) + (bₙ * sma) * cos(nθ)
            # and Δr/r = aₙ * sin(nθ) + bₙ * cos(nθ)
            r_ell_flat = r_ell.ravel()
            dr_over_r = np.zeros_like(r_ell_flat)

            for n, (an_interp, bn_interp) in harm_interps.items():
                an_local = an_interp(r_ell_flat)
                bn_local = bn_interp(r_ell_flat)
                theta_flat = theta.ravel()
                dr_over_r += an_local * np.sin(n * theta_flat) + bn_local * np.cos(n * theta_flat)

            dr_over_r = dr_over_r.reshape(r_ell.shape)

            # Apply harmonic correction to elliptical radius
            # If deviation is positive (isophote bulges out), the effective radius
            # is smaller to get higher intensity at that location
            r_ell_corrected = r_ell * (1.0 - dr_over_r)
            r_ell_corrected = np.clip(r_ell_corrected, 0, np.max(sma_values) * 2)

            # Interpolate intensity at corrected radius
            model_flat = intens_interp(r_ell_corrected.ravel())
            model = model_flat.reshape(image_shape)
        else:
            # No harmonics available, use basic interpolation
            model_flat = intens_interp(r_ell.ravel())
            model = model_flat.reshape(image_shape)
    else:
        # Harmonics disabled or insufficient isophotes
        model_flat = intens_interp(r_ell.ravel())
        model = model_flat.reshape(image_shape)

    return model


# Keep old name as alias for backward compatibility
def build_ellipse_model(image_shape, isophote_results, fill=0.0):
    """
    Legacy name for build_isoster_model().

    .. deprecated::
        Use build_isoster_model() instead. Will be removed in 2.0.0.
    """
    import warnings

    warnings.warn(
        "build_ellipse_model() is deprecated, use build_isoster_model() instead", DeprecationWarning, stacklevel=2
    )
    return build_isoster_model(image_shape, isophote_results, fill=fill)
