<p align="center">
  <img src="docs/isoster_logo.svg" alt="ISOSTER logo" width="320">
</p>

<h1 align="center">ISOSTER</h1>

<p align="center">
  <strong>ISOphote on STERoid</strong> — Accelerated elliptical isophote fitting for galaxy images
</p>

<p align="center">
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License: BSD-3-Clause"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
</p>

---

ISOSTER is a Python library for elliptical isophote fitting that runs **tens of times faster** than `photutils.isophote` — a median of $45\times$ across a 237-configuration synthetic Sérsic sweep (interquartile range $35$–$57\times$; the slowest case was $16\times$), with every case also meeting the benchmark's accuracy criteria against the true profile. Reproduce with [`benchmarks/performance/bench_vs_photutils.py`](benchmarks/performance/bench_vs_photutils.py); the archived summary is [`benchmarks/performance/reference_speedup.json`](benchmarks/performance/reference_speedup.json). Absolute times are machine-specific, so the ratio is the portable part, and the speedup varies with image size and isophote count. ISOSTER uses vectorized path-based sampling via scipy's `map_coordinates` while maintaining scientific accuracy and full compatibility with the photutils isophote analysis workflow.

## Installation

ISOSTER requires Python 3.9+ and uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Clone and install
git clone https://github.com/shuang-stat/isoster.git
cd isoster
uv sync                          # core dependencies only

# Optional extras
uv sync --extra asdf             # ASDF file format support
uv sync --extra dev --extra docs # development + documentation tools
```

Use `uv` for development and dependency management in this repository.

## Quick Start

```python
import isoster
from isoster.config import IsosterConfig
from astropy.io import fits

# Load a galaxy image
image = fits.getdata("galaxy.fits")

# Configure and fit isophotes
config = IsosterConfig(sma0=10.0, maxsma=100.0)
results = isoster.fit_image(image, config=config)

# Save results (FITS or ASDF)
isoster.isophote_results_to_fits(results, "isophotes.fits")
isoster.isophote_results_to_asdf(results, "isophotes.asdf")  # requires asdf extra

# Build a 2D model with harmonic deviations
model = isoster.build_isoster_model(
    image.shape,
    results['isophotes'],
    use_harmonics=True,
    harmonic_orders=[3, 4],
)
```

### Multiband Analysis

Use template-based forced photometry for consistent color profiles across wavelengths:

```python
# Fit reference band (e.g., g-band) with full geometry fitting
results_g = isoster.fit_image(image_g, None, config)

# Apply g-band geometry to other bands
results_r = isoster.fit_image(image_r, None, config, template=results_g)
results_i = isoster.fit_image(image_i, None, config, template='galaxy_g.fits')
```

## Key Features

- **High performance**: tens of times faster than `photutils.isophote` via vectorized sampling (median $45\times$ on the synthetic Sérsic sweep; see above).
- **Template-based forced photometry**: Consistent multiband photometry using geometry from a reference band.
- **Eccentric anomaly sampling**: Uniform arc-length coverage for high-ellipticity galaxies (Ciambur 2015).
- **Simultaneous harmonics**: ISOFIT-style joint fitting of higher-order harmonics within the iteration loop.
- **2D model building**: Reconstruct galaxy images from isophote profiles with optional harmonic deviations.
- **Convergence controls**: Sector-area scaling, geometry damping, and geometry-stability convergence.
- **Photometry metrics**: Integrated flux, curve-of-growth, and adaptive integration modes.
- **Photutils compatibility**: Consistent algorithms and output format with industry standards.
- **Function-based API**: Simple, stateless interface for easy integration and testing.

## Documentation

- [User Guide](docs/01-user-guide.md) — usage guidance, public API, and stop-code reference
- [Configuration Reference](docs/02-configuration-reference.md) — all parameters and guidelines
- [Algorithm Notes](docs/03-algorithm.md) — fitting and sampling implementation details
- [Architecture](docs/04-architecture.md) — interfaces and design decisions

## Repository Structure

```
isoster/          Core package (config, driver, sampling, fitting, model, plotting, cog)
tests/            Unit, integration, and validation tests
benchmarks/       Performance and profiling benchmarks
examples/         Reproducible workflow examples
docs/             Project documentation
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and pull request guidelines.

## Citation

If you use ISOSTER in your research, please cite using the metadata in [CITATION.cff](CITATION.cff).

## Acknowledgments

ISOSTER began as an optimization of the [`photutils.isophote`](https://photutils.readthedocs.io/) package. We thank the photutils contributors for their robust foundational algorithms.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
