# ISOSTER To-Do List: Missing Features & Options

This document outlines the discrepancies between `isoster` and the original `photutils.isophote` implementation. These items represent missing functionalities, options, or data outputs that need to be addressed to achieve full parity.

## 1. Missing Model Building Functionality

The original `photutils.isophote` includes tools to reconstruct a 2D model image from the fitted isophotes. ISOSTER completely lacks this capability.

-   **`model.py:build_ellipse_model()`**: Function to generate a synthetic image from a list of isophotes.
    -   Uses spline interpolation (`scipy.interpolate.LSQUnivariateSpline`) to smooth geometry parameters (SMA, eps, PA, x0, y0) between isophotes.
    -   Supports higher-order harmonics (a3, b3, a4, b4).
    -   Handles partial pixel coverage.
-   **`ellipse_model.pyx`**: Cython extension for efficient pixel rendering of the model.

## 2. Missing Data Outputs

The following properties available in `photutils.isophote.Isophote` are currently **NOT** calculated or returned by ISOSTER:

### Flux Integration Metrics
ISOSTER currently only computes mean intensity and RMS along the elliptical path. It does not calculate the total integrated flux within the ellipse.
-   **`tflux_e`**: Total flux sum of all pixels inside the isophote ellipse.
-   **`tflux_c`**: Total flux sum of all pixels inside a circle with radius `sma`.
-   **`npix_e`**: Total number of valid pixels inside the ellipse.
-   **`npix_c`**: Total number of valid pixels inside the circle.
-   **`sarea`**: Average sector area (relevant for area-based integration).

### Diagnostic Flags
-   **`nflag`**: Number of flagged/discarded data points per isophote. (Currently only available in debug mode).
-   **`ndata`**: Number of actual valid data points used for the fit. (Currently only available in debug mode).
-   **`valid`**: Explicit success/failure flag (derived from `stop_code`, but not returned as a separate column).

## 3. Unimplemented Options

### Integration Modes
`photutils` supports multiple algorithms for sampling the image. ISOSTER strictly implements vector-based sampling which approximates **bilinear** interpolation.
-   **`mean`**: Calculate the mean of pixel values within ellipse sectors. (Missing)
-   **`median`**: Calculate the median of pixel values within ellipse sectors. (Missing)
-   **`nearest_neighbor`**: Use nearest pixel values. (Missing; effectively covered by bilinear with order=0, but not exposed as an option).

### Sampling Control
-   **`integrmode`**: The option to switch between the modes above is not available in `fit_image` or `config.yaml`.

## 4. Workflow & API Differences

-   **`Isophote.to_table()`**: The convenience method on the result object is replaced by `isoster.utils.isophote_results_to_astropy_tables`.
-   **`EllipseSample.update()`**: The granular control to update a specific sample's geometry without re-fitting is not exposed in the high-level API.

## Priority Recommendation

1.  **Model Building**: Implementing `build_ellipse_model` is critical for galaxy subtraction.
2.  **Flux Metrics**: `tflux_e` and `npix_e` are essential for photometry.
