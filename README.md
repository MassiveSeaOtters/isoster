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

ISOSTER is a Python library for elliptical isophote fitting that runs **tens of times faster** than `photutils.isophote` — a median of $45\times$ on a synthetic Sérsic sweep (interquartile range $35$–$55\times$; slowest case $13\times$), with every completed case also meeting the benchmark's accuracy criteria against the true profile. Of 243 attempted configurations, `photutils` could not fit 6 — the degenerate corner where a circular source ($\varepsilon = 0$) is given a position angle — and those are excluded from the statistics; the remaining 237 all passed. Each configuration is timed once per tool, so the quartiles describe spread across configurations, not timing noise. Reproduce with [`benchmarks/performance/bench_vs_photutils.py`](benchmarks/performance/bench_vs_photutils.py); the archived summary, including the excluded configurations, is [`benchmarks/performance/reference_speedup.json`](benchmarks/performance/reference_speedup.json). Absolute times are machine-specific; the ratio is less machine-dependent than absolute time but not portable, and the speedup varies with image size and isophote count. ISOSTER uses vectorized path-based sampling via scipy's `map_coordinates`. It shares the Jedrzejewski (1987) algorithmic ancestry of `photutils.isophote` and its geometry and intensity profiles have been validated against it on the sweep above, but it is **not** a drop-in replacement: sampling, convergence handling, defaults, output schema and serialization all differ deliberately. See [`docs/technical/1.5-comparison.md`](docs/technical/1.5-comparison.md) for what matches and what does not.

## Installation

ISOSTER requires Python 3.9+ and uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Clone and install
git clone https://github.com/MassiveSeaOtters/isoster.git
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

Two workflows, estimating different things. Pick by what you want the geometry
to mean — see [`docs/10-multiband.md`](docs/10-multiband.md).

**Joint fit** — one geometry per SMA, derived from every band at once. Use when
the geometry itself is the measurement and no single band should define it:

```python
from isoster.multiband import IsosterConfigMB, fit_image_multiband

cfg_mb = IsosterConfigMB(bands=['g', 'r', 'i'], reference_band='r', sma0=10.0)
result = fit_image_multiband([image_g, image_r, image_i], config=cfg_mb)

# Per-band intensities and harmonics on one shared geometry
for iso in result['isophotes'][:3]:
    print(iso['sma'], iso['intens_g'], iso['intens_r'], iso['intens_i'])
```

**Template-based forced photometry** — measure every band through a reference
band's geometry. Use when that geometry *is* the aperture definition:

```python
# Fit reference band (e.g., g-band) with full geometry fitting
results_g = isoster.fit_image(image_g, None, config)

# Apply g-band geometry to other bands
results_r = isoster.fit_image(image_r, None, config, template=results_g)
results_i = isoster.fit_image(image_i, None, config, template='galaxy_g.fits')
```

## Key Features

- **High performance**: tens of times faster than `photutils.isophote` via vectorized sampling (median $45\times$ on the synthetic Sérsic sweep; see above).
- **Multi-band fitting**: a joint free fit that derives one geometry from every band (`isoster.multiband.fit_image_multiband`), plus template-based forced photometry that applies a reference band's geometry to the others. The two estimate different quantities and are complementary; see [`docs/10-multiband.md`](docs/10-multiband.md).
- **Eccentric anomaly sampling**: More even sampling around high-ellipticity isophotes than stepping the position angle (Ciambur 2015).
- **Simultaneous harmonics**: ISOFIT-style joint fitting of higher-order harmonics within the iteration loop.
- **2D model building**: Reconstruct galaxy images from isophote profiles with optional harmonic deviations.
- **Convergence controls**: Sector-area scaling, geometry damping, and geometry-stability convergence.
- **Photometry metrics**: Integrated flux, curve-of-growth, and adaptive integration modes.
- **Shared algorithmic ancestry with photutils**: the same Jedrzejewski (1987) formulation, with geometry and intensity profiles validated against `photutils.isophote`. The output schema, defaults, sampling and serialization differ deliberately — see the note above.
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
