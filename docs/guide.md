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
    full_photometry=True
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
