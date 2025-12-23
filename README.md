# ISOSTER (ISOphote on STERoid)

ISOSTER is an accelerated Python library for isophote fitting, optimized for performance while maintaining compatibility with `photutils.isophote`. It provides significant speedups (10-15x on synthetic images) by replacing area-based integration with vectorized path-based sampling.

## Features

- **High Performance**: Vectorized implementation using `scipy.ndimage`.
- **Function-Based API**: Streamlined functional approach.
- **Photutils Compatibility**: Logic and algorithms align with `photutils` for consistent results.
- **CLI Tool**: Ready-to-use command line interface.

## Installation

To install ISOSTER, clone the repository and install it using pip:

```bash
git clone <repository_url>
cd isoster
pip install .
```

## Usage

### Command Line Interface

ISOSTER comes with a built-in CLI for processing FITS images:

```bash
isoster image.fits --output results.csv --config config.yaml
```

Configuration parameters (optional) can be provided in a YAML file or via command line arguments.

### Python API

```python
from astropy.io import fits
from isoster.optimize import fit_image

# Load data
image = fits.getdata('galaxy.fits')

# Run fitting
results = fit_image(image, mask=None, config={
    'x0': 100, 'y0': 100,
    'sma0': 10,
    'minsma': 0, 'maxsma': 200
})

# Access results
isophotes = results['isophotes']
```

## Structure

- `isoster/`: Main package source code.
  - `optimize.py`: Core fitting algorithms.
  - `utils.py`: Utility functions.
  - `plotting.py`: Visualization tools.
- `reference/`: Original `photutils.isophote` code (for reference/validation).
- `benchmarks/`: Performance tests and comparison scripts.
- `tests/`: Unit tests.

## Development

Run tests:
```bash
pytest
```

Run benchmarks:
```bash
python benchmarks/benchmark_real_galaxy_m51.py
```
