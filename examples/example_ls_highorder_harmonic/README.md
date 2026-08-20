# LegacySurvey High-Order Harmonics Example

Demonstrates isoster's high-order harmonic fitting on two real galaxy images
from the LegacySurvey / SDSS archive:

| Galaxy | Type | Pixel scale | Bands |
|--------|------|------------|-------|
| **ESO243-49** | Edge-on S0 | 0.25 arcsec/px | g / r / z |
| **NGC3610** | Boxy-bulge elliptical | 1.0 arcsec/px | g / r / i |

## Six Fitting Conditions

| Label | Sampling | Simultaneous harmonics | Harmonic orders |
|-------|----------|------------------------|-----------------|
| `pa_baseline` | PA | No | [3, 4] post-hoc |
| `ea_baseline` | EA (Ciambur 2015) | No | [3, 4] post-hoc |
| `pa_harmonics_34` | PA | Yes | [3, 4] |
| `ea_harmonics_34` | EA | Yes | [3, 4] |
| `pa_harmonics_34567` | PA | Yes | [3, 4, 5, 6, 7] |
| `ea_harmonics_34567` | EA | Yes | [3, 4, 5, 6, 7] |

## Quick Start

```bash
# ESO243-49 r-band (index 1)
uv run python examples/example_ls_highorder_harmonic/run_example.py \
    --galaxy eso243-49 --band-index 1 \
    --output-dir outputs/example_ls_highorder_harmonic/

# NGC3610 r-band
uv run python examples/example_ls_highorder_harmonic/run_example.py \
    --galaxy ngc3610 --band-index 1 \
    --output-dir outputs/example_ls_highorder_harmonic/
```

## Output Layout

```
outputs/example_ls_highorder_harmonic/<galaxy>/band_<N>/
├── mask.fits                # Bad-pixel mask (1 = masked)
├── mask_qa.png              # Mask QA figure
├── <condition>/
│   ├── isophotes.fits       # Full isophote table (FITS binary table)
│   ├── isophotes.ecsv       # Astropy-readable ASCII ECSV
│   └── qa.png               # Per-condition extended QA figure
├── comparison_qa.png        # All-condition comparison figure
└── qa_figures/              # Flat directory with all QA PNGs
    ├── <galaxy>_band<N>_mask_qa.png
    ├── <galaxy>_band<N>_<condition>_qa.png  (one per condition)
    └── <galaxy>_band<N>_comparison_qa.png
```

## CLI Options

```
--galaxy {eso243-49,ngc3610}   Galaxy to process (required)
--band-index INT                Band plane index 0/1/2 (default: 1 = r-band)
--output-dir PATH               Root output directory
--sma0 FLOAT                    Override initial SMA in pixels
--skip-mask                     Use empty mask (skip masking)
--conditions [...]              Run only specified conditions
```

## Harmonic Display Modes

The per-condition QA figure (`plot_qa_summary_extended`) and comparison figure
(`plot_harmonic_comparison_qa`) support two harmonic display modes:

### Default: Coefficients mode (`harmonic_mode='coefficients'`)

Shows individual `a_n` (filled markers) and `b_n` (open markers) per harmonic
order, split into **odd** (3, 5, 7) and **even** (4, 6) panels.  Each order
gets its own colour; `a_n` uses filled markers, `b_n` uses open markers of
the same colour.

### Alternative: Amplitude mode (`harmonic_mode='amplitude'`)

Shows `A_n = sqrt(a_n^2 + b_n^2)` per order, where `a_n` and `b_n` are the
values stored in the results dict — already Bender-normalized (see below), so
`A_n` is dimensionless as plotted. The panels are labelled
`A_n / (a dI/da)` accordingly.

There is deliberately no option to normalize again. A `normalize_harmonics`
flag that divided by intensity existed until 1.0.0 and was removed: since the
stored values are already normalized, it produced `A_n_raw / (a · |dI/da| · I)`
— a quantity with units of one over intensity, whose value changes with the
image's flux calibration even though the galaxy's shape does not.

### a4 panel

A dedicated `a4` panel shows the stored `a4` coefficient as the canonical
boxy/disky morphology indicator: positive = disky, negative = boxy.

**These coefficients are not raw.** `compute_deviations` divides by
`a · |dI/da|` before storing, so `a4` as reported is the Bender-normalized,
scale-invariant quantity. This matters for interpretation: raw amplitudes
scale with local flux and are not comparable across a profile or between
galaxies, whereas these are. An earlier version of this README described them
as raw.

## Residual Map

The per-condition QA figure shows a residual map panel.  By default it displays
the **absolute** residual `data - model`.  Set `relative_residual=True` to show
the fractional residual as a percentage, `100 * (model - data) / data`. Note the
sign convention differs between the two: the relative form is model-minus-data,
matching `isoster/plotting.py`.

## Masking

The two-stage photutils masking pipeline (`masking.py`) uses:

- **Stage 1**: Field contaminant detection with source deblending
- **Stage 2**: On-galaxy compact source detection using small-box (`box=8`)
  local background subtraction with `nsigma=3.0` to avoid masking galaxy
  structure while catching genuine contaminants

## Files

| File | Purpose |
|------|---------|
| `masking.py` | Two-stage photutils masking pipeline with deblending |
| `shared.py` | Galaxy metadata, config factory, comparison QA function |
| `run_example.py` | Main CLI script |
| `compare_isofit_modes.py` | Optional ISOFIT comparison study |
| `ngc3610_highorder_exploration.py` | Optional NGC3610 harmonic-impact study |
| `ngc3610_mask_effect.py` | Optional NGC3610 masking-sensitivity study |
| `ngc3610_sma0_effect.py` | Optional NGC3610 starting-radius study |

## Default Run Policy

- Run `run_example.py` by default.
- The other scripts in this folder are exploratory follow-ups for high-harmonic behavior and should not be run unless you specifically want those diagnostics.
- Their default outputs are:
  - `outputs/example_ls_highorder_harmonic/isofit_mode_comparison/`
  - `outputs/example_ls_highorder_harmonic/ngc3610_highorder_exploration/`
  - `outputs/example_ls_highorder_harmonic/ngc3610_mask_effect/`
  - `outputs/example_ls_highorder_harmonic/ngc3610_sma0_effect/`

## Notes

- All six conditions use `convergence_scaling='sector_area'` and
  `geometry_damping=0.7` (current isoster defaults, validated on 20 galaxies).
- Baseline conditions have `simultaneous_harmonics=False`; the [3, 4] harmonics
  are still stored post-hoc but do not influence the geometry fit.
- EA (eccentric-anomaly) mode is recommended for high-ellipticity objects.
