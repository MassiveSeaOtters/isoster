# Algorithm & Rationale

**Isoster** implements a highly optimized version of the standard iterative ellipse fitting algorithm (Jedrzejewski 1987), tailored for speed and modern Python environments. This document details the algorithmic choices, the "fast fitting" philosophy, and the trade-offs involved.

## Core Philosophy: Vectorization over Objects

The primary design goal of `isoster` is to eliminate the overhead associated with creating thousands of Python objects during the fitting process.

### The Standard Approach (Photutils)
Implementations like `photutils.isophote` typically follow an object-oriented paradigm:
1.  Create an `Ellipse` object.
2.  For each semi-major axis (SMA), create an `Isophote` object.
3.  Inside the fitting loop, create more temporary objects for geometry and samples.
4.  Use area-based integration (e.g., recursive sub-pixel sampling) for every pixel intersection.

This approach is robust and mathematically rigorous but suffers from significant overhead in Python loops and object instantiation, making it slow for large images or large cutout batches.

### The Isoster Approach
`isoster` shifts the paradigm to **vectorized path sampling**:
1.  **Stateless Functions**: The core logic is implemented in stateless functions (`fit_isophote`, `fit_image`) rather than stateful classes.
2.  **Path Sampling**: Instead of calculating the detailed area overlap of pixels along the ellipse, `isoster` samples points *directly along the elliptical path* using interpolation.
3.  **Vectorization**: Geometric calculations (coordinates, angles, harmonics) are broadcasted using `numpy`, operating on entire arrays of sample points at once.

## The Fitting Algorithm

The fitting process follows the classic logic:

1.  **Sampling**: Extract intensity values $I(\phi)$ along an elliptical path defined by $(x_0, y_0, a, \epsilon, \text{PA})$.
    *   `isoster` uses `scipy.ndimage.map_coordinates` (bilinear interpolation by default) to sample $N$ points uniformly spaced in eccentric anomaly $\phi$.
    *   $N \approx 2\pi \cdot \text{SMA}$ to ensure Nyquist sampling of the pixel grid.

2.  **Harmonic Analysis**: The intensity profile $I(\phi)$ is least-squares fitted to a harmonic series:
    $$ I(\phi) = I_0 + A_1 \sin(\phi) + B_1 \cos(\phi) + A_2 \sin(2\phi) + B_2 \cos(2\phi) $$
    
    *   $(A_1, B_1)$ relate to center deviations $(\delta x, \delta y)$.
    *   $(A_2, B_2)$ relate to shape deviations (ellipticity $\epsilon$ and position angle $\text{PA}$).

3.  **Geometry Update**: The geometric parameters are updated to minimize these harmonic amplitudes (driving the isophote to match the galaxy contour):
    $$ \delta x \propto -A_1 / \Gamma, \quad \delta y \propto -B_1 / \Gamma $$
    $$ \delta \epsilon \propto -B_2 / \Gamma, \quad \delta \text{PA} \propto A_2 / \Gamma $$
    Where $\Gamma$ is the local radial gradient $dI/dr$.

4.  **Convergence**: Steps 1-3 repeat until the harmonic amplitudes are negligible compared to the RMS noise.

## Integration Modes

`isoster` supports multiple integration strategies to handle different signal-to-noise regimes:

*   **Mean (Default)**: Uses the mean of the sampled intensities ($I_0$ from the harmonic fit). Fast and precise for high SNR.
*   **Median**: Uses `np.median` of the sampled points. More robust against unmasked outliers (stars, defects) but slightly slower.
*   **Adaptive**: Uses `Mean` for inner, bright regions ($SMA \le \text{Threshold}$) and switches to `Median` for outer, faint regions where outliers dominate the noise budget.

## Handling Masks

Handling masked pixels (e.g., bad pixels, stars, gaps) is a critical component of isophote fitting.

### The Problem with Masked Arrays
Standard libraries often rely on `numpy.ma.MaskedArray`. While robust, masked arrays introduce significant computational overhead because every operation involves checking and propagating the mask state, often inhibiting low-level compiler optimizations in numpy/scipy functions.

### The Isoster Solution: Vectorized Sampling
`isoster` treats the mask as a separate boolean image and samples it simultaneously with the data:
1.  **Nearest Neighbor Sampling**: The mask is sampled using `map_coordinates(..., order=0)` along the elliptical path. This assigns a boolean `True` (bad) or `False` (good) to each sample point based on the nearest pixel center.
2.  **Filter**: The 1D intensity profile is filtered to exclude these bad points: `intens = intens[~mask_samples]`.
3.  **Efficiency**: This operation is fully vectorized and avoids the creation of heavy `MaskedArray` objects. Benchmarks show this approach maintains high speed (only ~2x slowdown purely due to extra sampling) compared to the >10x slowdown often seen when enabling masks in other codes.

## Weaknesses and Caveats

While `isoster` typically achieves 5-10x magnitude speedups, users should be aware of the trade-offs:

### 1. Sampling Artifacts at Small Radii
For very small isophotes ($SMA < 5$ pixels), path-based sampling can lead to aliasing because it samples *points* rather than integrating *areas*. 
*   **Impact**: Potential noise in geometry for the innermost few pixels.
*   **Mitigation**: `photutils` is recommended if extreme sub-pixel accuracy at $r < 3$ px is required. `isoster` is optimized for the bulk of the galaxy profile.

### 2. Edge Handling
`map_coordinates` with `mode='constant'` handles image edges efficiently, but extreme edge isophotes might lose sample points differently than an exact area intersection method.

### 3. Iterative Sigma Clipping
`isoster` implements sigma clipping on the 1D sample profile. This is generally effective, but subtle differences in how `photutils` manages clipping (e.g., maintaining clip flags across iterations) may lead to slightly different results in heavily contaminated fields.
