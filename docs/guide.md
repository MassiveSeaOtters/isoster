# ISOSTER User Guide

Welcome to the **ISOSTER** user guide. This document provides detailed instructions on installing, configuring, and running isophote fits on galaxy images efficiently.

## 1. Installation

ISOSTER requires Python 3.9+ and standard scientific libraries (`numpy`, `scipy`, `astropy`, `pydantic`).

```bash
git clone https://github.com/your-repo/isoster.git
cd isoster
pip install .
```

## 2. Configuration Management

ISOSTER uses **Pydantic** for robust configuration management. This ensures that all parameters are validated (e.g., stopping you from setting negative iteration counts) and provides auto-completion in modern IDEs.

### Using `IsosterConfig`

Instead of passing a raw dictionary, you can instantiate an `IsosterConfig` object:

```python
from isoster.config import IsosterConfig

# Create a configuration object with validation
config = IsosterConfig(
    sma0=15.0,
    maxit=50,
    sclip=3.0,
    fflag=0.7,      # Fit quality flag (fraction of usable pixels)
    integrator='mean', # 'mean' (default), 'median', or 'adaptive'
    lsb_sma_threshold=100.0, # Required if integrator='adaptive'
    full_photometry=False # Compute flux metrics (tflux_e, etc)
)

# Invalid values will raise an error immediately:
# config = IsosterConfig(maxit=-10)  # Raises ValidationError
```

### Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `x0`, `y0` | float | None | Center coordinates. If None, image center is used. |
| `sma0` | float | 10.0 | Starting semi-major axis length. |
| `minsma` | float | 0.0 | Minimum SMA to fit. |
| `maxsma` | float | None | Maximum SMA. Defaults to half image size. |
| `astep` | float | 0.1 | Step size for SMA growth. |
| `linear_growth` | bool | False | If True, `sma += astep`. If False, `sma *= (1 + astep)`. |
| `maxit` | int | 50 | Maximum iterations per isophote. |
| `conver` | float | 0.05 | Convergence threshold. |
| `sclip` | float | 3.0 | Sigma clipping threshold. |
| `nclip` | int | 0 | Number of clipping iterations. |
| `fix_center` | bool | False | Fix center coordinates (x0, y0). |
| `fix_pa` | bool | False | Fix Position Angle. |
| `fix_eps` | bool | False | Fix Ellipticity. |
| `full_photometry` | bool | False | Calculate extra flux metrics (`tflux_e`, etc.). |

## 3. Running a Fit

The primary function is `fit_image`.

```python
import isoster
from astropy.io import fits

# 1. Load Data
image = fits.getdata("m51.fits")

# 2. Configure
cfg = isoster.IsosterConfig(
    sma0=5.0,
    maxsma=100.0,
    full_photometry=True
)

# 3. Run Fit
results = isoster.fit_image(image, config=cfg)

# 4. Save Results
isoster.isophote_results_to_fits(results, "m51_results.fits")
```

## 4. Advanced: 2D Model Building

You can reconstruct a noise-free model of the galaxy from the fitted isophotes.

```python
model = isoster.build_ellipse_model(image.shape, results['isophotes'])
# Save or plot 'model'
```

## 5. Troubleshooting

*   **Convergence Issues**: Try increasing `maxit` or adjusting `sma0` to a region with higher signal-to-noise.
*   **Performance**: Ensure you are not running `full_photometry` unless needed, as it adds computation time.
*   **Validation Errors**: Pydantic will tell you exactly which field is invalid. Check your parameter types and ranges.

## 4. Eccentric Anomaly Sampling

For galaxies with **high ellipticity** (ε > 0.3), the standard polar angle sampling can lead to biased fits due to uneven sampling along the ellipse. ISOSTER implements the **Eccentric Anomaly (EA) method** from Ciambur (2015, ApJ 810 120) to address this.

### What is Eccentric Anomaly?

- **Standard method**: Sample uniformly in φ (position angle) → points cluster near major axis
- **EA method**: Sample uniformly in ψ (eccentric anomaly) → uniform arc-length coverage
- **Key**: Fit harmonics in ψ space: I(ψ) = Ī + A₁sin(ψ) + B₁cos(ψ) + ...

### When to Use EA

✓ **Recommended for:**
- Ellipticity ε > 0.3
- Edge-on or highly inclined galaxies
- Precise geometry fitting in high-ε regimes

✗ **Not needed for:**
- Nearly circular galaxies (ε < 0.2)
- Fixed geometry fitting
- Low S/N data (regularization more important)

### Usage

```python
from isoster.config import IsosterConfig
from isoster.optimize import fit_image

# Enable Eccentric Anomaly sampling
config = IsosterConfig(
    sma0=10.0,
    maxsma=200.0,
    use_eccentric_anomaly=True,  # Enable EA method
    # Free geometry to benefit from EA
    fix_center=False,
    fix_pa=False,
    fix_eps=False
)

results = fit_image(image, config=config)
```

### Performance Notes

- **Overhead**: ~50-60% slower than regular sampling (due to ψ→φ conversion overhead)
- **Benefit**: More accurate geometry fitting for high-ε galaxies
- **Trade-off**: Speed vs accuracy - use when geometry accuracy is critical

### Combining with Central Regularization

For optimal results on high-ellipticity galaxies with noisy centers:

```python
config = IsosterConfig(
    use_eccentric_anomaly=True,
    # Add moderate central regularization
    use_central_regularization=True,
    central_reg_sma_threshold=3.0,
    central_reg_strength=1.0,
    central_reg_weights={'eps': 1.5, 'pa': 1.0, 'center': 1.0}
)
```

## 5. Central Region Geometry Regularization

(Section continues from previous...)
