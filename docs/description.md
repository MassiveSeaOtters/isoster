# Explaining ISOSTER: Algorithm and Design
## Introduction to Isophote Fitting

Galaxy isophote fitting is a fundamental technique in astronomical photometry. It involves modeling the surface brightness distribution of a galaxy as a set of concentric ellipses. The classic algorithm, popularized by the IRAF STSDAS `ellipse` task and later ported to Python in `photutils.isophote`, works by iteratively adjusting the parameters of an ellipse (center, ellipticity, position angle) until the intensity deviations along the ellipse minimize the first and second harmonic coefficients of the Fourier expansion.

**ISOSTER** (ISOphote on STERoid) is an optimized reimplementation of this algorithm. It preserves the exact mathematical logic of the original IRAF/photutils method but restructures the code for modern Python performance (vectorization) and usability (functional API).

## Architecture

ISOSTER moves away from the deep class hierarchy of `photutils` (which uses `Ellipse`, `EllipseSample`, `EllipseFitter`, `EllipseGeometry`) towards a flatter, functional architecture. The core logic resides in a set of pure functions in `isoster.optimize`.

### Core Modules

1.  **`isoster.optimize` (The Engine)**
    *   **`extract_isophote_data`**: The performance bottleneck in the original code. ISOSTER replaces pixel-by-pixel integration with vectorized path sampling using `scipy.ndimage.map_coordinates`. This scales linearly with the ellipse circumference $O(N_{samples})$ rather than the area, providing massive speedups.
    *   **`fit_isophote`**: Fits a single isophote at a specific semi-major axis (SMA). It handles the iterative harmonic fitting loop, geometry updates, and quality control (sigma clipping, convergence checks).
    *   **`fit_image`**: The driver function. It manages the global strategy of growing isophotes outwards and inwards from a starting ellipse.

2.  **`isoster.utils` (Interoperability)**
    *   Handles data conversion between internal dictionaries and standard Astropy Tables or FITS files.

3.  **`isoster.plotting` (Visualization)**
    *   Provides standardized plots for profiles (Intensity, Ellipticity, PA vs SMA) and image overlays.

## The Algorithm in Detail

The fitting process at each SMA follows these steps:

1.  **Sampling**: Intensities are sampled along an elliptical path defined by $(x_0, y_0, \epsilon, PA)$.
2.  **Harmonic Analysis**: The intensity distribution $I(\phi)$ along the eccentric anomaly angle $\phi$ is fitted with a Fourier series:
    $$I(\phi) = I_0 + A_1 \sin(\phi) + B_1 \cos(\phi) + A_2 \sin(2\phi) + B_2 \cos(2\phi)$$
3.  **Correction**: The harmonic coefficients quantify the deviation from perfect elliptical isophotes.
    *   $A_1, B_1$: Indicate centering errors. Used to correct $(x_0, y_0)$.
    *   $A_2$: Indicates position angle error. Used to correct $PA$.
    *   $B_2$: Indicates ellipticity error. Used to correct $\epsilon$.
4.  **Iteration**: The geometry is updated (`geometry += correction`) and the process repeats until the largest harmonic coefficient is below a threshold fraction of the RMS noise.

## Performance Optimization

The primary speedup comes from vectorization.
*   **Original**: Loops over image pixels or sub-pixels to compute fractional overlaps with ellipse sectors.
*   **ISOSTER**: Defines sample points coordinates $(x, y)$ using numpy broadcasting and interpolates values using C-compiled `scipy` routines.

## Architecture Analysis & Recommendations

While ISOSTER successfully accelerates the fitting process, there are opportunities to further improve the repository structure and maintainability:

### 1. Separation of Concerns
Currently, `optimize.py` contains everything: sampling, fitting, gradient computation, and the main loop.
*   **Recommendation**: Split `optimize.py` into focused modules:
    *   `sampling.py`: `extract_isophote_data`, `get_elliptical_coordinates`.
    *   `fitting.py`: `fit_isophote`, `fit_first_and_second_harmonics`.
    *   `driver.py`: `fit_image`.

### 2. Configuration Management
The `config.yaml` approach is flexible but passing raw dictionaries can be error-prone (typos in keys).
*   **Recommendation**: Use Python dataclasses or `pydantic` models for configuration. This provides type safety, validation (e.g., ensuring `maxit > 0`), and auto-completion in IDEs.

### 3. Testing Strategy
Benchmarks are present, but granular unit tests are needed for the new functional components.
*   **Recommendation**: Add unit tests for specific functions like `extract_isophote_data` (checking sampling accuracy) and `harmonic_function` (checking math correctness) in isolation, separate from the full image fit.

### 4. API Expansion
The current API focuses on batch processing an entire image.
*   **Recommendation**: Expose the single-isophote fitter `fit_isophote` more prominently. This allows users to fit specific regions or interactively adjust fits without re-running the whole chain.

### 5. Documentation System
*   **Recommendation**: Set up Sphinx or MkDocs to auto-generate API documentation from the improved docstrings. The "Description" document should be part of a larger user guide.
