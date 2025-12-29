# ISOSTER Design and Architecture

## Core Algorithm

ISOSTER accelerates isophote fitting by replacing the traditional area-based integration (which requires checking pixel overlaps/intersections) with vectorized path-based sampling. 

The core optimization uses `scipy.ndimage.map_coordinates` to perform bilinear interpolation along an elliptical path defined by:
$$x = x_0 + a \cos \phi \cos PA - a (1-\epsilon) \sin \phi \sin PA$$
$$y = y0 + a \cos \phi \sin PA + a (1-\epsilon) \sin \phi \cos PA$$

Sampling uniformly in the polar angle $\phi$ provides the necessary data for fast harmonic fitting via linear least squares.

## Modular Architecture

The package is organized into several independent modules to improve maintainability and testability:

1. **`sampling.py`**: Handles coordinate transformations and vectorized data extraction.
2. **`fitting.py`**: Implements the iterative fitting loop, harmonic analysis, sigma clipping, and error estimation. Includes high-performance aperture photometry for flux metrics.
3. **`driver.py`**: Orchestrates the overall fitting process, managing outward and inward growth and central pixel fitting.
4. **`model.py`**: Reconstructs 2D image models from fitting results using an efficient outside-in filling algorithm.
5. **`utils.py`**: Provides data conversion and I/O utilities (e.g., FITS export).
6. **`plotting.py`**: Utilities for performance and accuracy visualization.

## Performance vs. Photutils

- **Small/Synthetic Images**: 10-15x speedup due to vectorized sampling avoiding Python loops and complex overlap calculations.
- **Real Galaxy Images**: 2-10x speedup depending on noise levels and convergence speed.
- **Accuracy**: Maintains <1% fractional error compared to `photutils` for SMA > 2 pixels.

## Key Features

- **Adaptive Integration**: Choose between mean, median, or adaptive integration for robust photometry
- **Forced Photometry**: Ultra-fast mode for large surveys with predetermined geometry (40x speedup)
- **Robust Mask Handling**: Efficient vectorized mask sampling using nearest-neighbor interpolation
- **Curve-of-Growth**: Cumulative flux calculation with crossing detection and proper mask handling
- **Full Error Propagation**: Comprehensive uncertainty estimates for all fitted parameters
- **Higher-Order Harmonics**: Detect deviations from perfect ellipses (a3, b3, a4, b4)
- **Flexible Configuration**: Pydantic-based configuration with validation
- **Production Ready**: Extensively tested against photutils with comprehensive benchmarks


## Future Improvements

### 1. Configuration Management (Implemented)
The `config.yaml` approach is flexible but passing raw dictionaries can be error-prone (typos in keys).
*   **Recommendation**: Use Python dataclasses or `pydantic` models for configuration. This provides type safety, validation (e.g., ensuring `maxit > 0`), and auto-completion in IDEs.

### 2. Testing Strategy
Benchmarks are present, but granular unit tests are needed for the new functional components.
*   **Recommendation**: Add unit tests for specific functions like `extract_isophote_data` (checking sampling accuracy) and `harmonic_function` (checking math correctness) in isolation, separate from the full image fit.

### 3. API Expansion
The current API focuses on batch processing an entire image.
*   **Recommendation**: Expose the single-isophote fitter `fit_isophote` more prominently. This allows users to fit specific regions or interactively adjust fits without re-running the whole chain.

### 4. Documentation System
*   **Recommendation**: Set up Sphinx or MkDocs to auto-generate API documentation from the improved docstrings. The "Description" document should be part of a larger user guide.

### 5. Parallelization

*   **Recommendation**: Implement parallelization for the isophote fitting loop, especially for large images.
