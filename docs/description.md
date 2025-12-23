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

## New Features

### Flux Integration Metrics
Conditional implementation of `tflux_e`, `tflux_c`, `npix_e`, and `npix_c` provides compatible photometry without overhead in standard analysis. Enabled via `config['full_photometry']`.

### 2D Model Building
A decoupled model builder allows for efficient image reconstruction, including support for higher-order harmonic deviations (a3, b3, a4, b4).
