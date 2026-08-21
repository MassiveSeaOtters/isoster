# Examples

This folder contains reproducible workflow examples on synthetic, survey, and
external mock datasets.

## Layout

- `example_basic_usage/`
  - Minimal synthetic end-to-end walkthrough.
  - Run with `uv run python examples/example_basic_usage/basic_usage.py`.
  - Default output: `outputs/example_basic_usage/`.
- `example_cog/`
  - Curve-of-growth validation on a synthetic Sersic image.
  - Run with `uv run python examples/example_cog/example_curve_of_growth.py`.
  - Default output: `outputs/example_cog/`.
- `example_ls_highorder_harmonic/`
  - Real-galaxy Legacy Survey harmonic-fitting example on `eso243-49` and `ngc3610`.
  - Main entrypoint: `uv run python examples/example_ls_highorder_harmonic/run_example.py --galaxy eso243-49 --band-index 1`.
  - Default main output: `outputs/example_ls_highorder_harmonic/`.
  - Additional exploration scripts exist in this folder, but do not run them by default.
- `example_huang2013/`
  - Two-stage Huang2013 external mock workflow: profile extraction, QA afterburner, campaign runner, and cleanup helper.
  - Main products are intentionally written outside `outputs/` under the external Huang2013 root because each case generates many files.
  - Campaign summaries still default to `outputs/huang2013_campaign/`.
- `example_variance_map/`
  - Weighted-least-squares fitting driven by a per-pixel variance map.
  - Run with `uv run python examples/example_variance_map/run_variance_map_demo.py`.
- `example_invalid_variance/`
  - How non-finite and non-positive variance entries are detected and their
    samples dropped.
  - Run with `uv run python examples/example_invalid_variance/run_invalid_variance_demo.py`.
- `example_wls_systematic/`
  - Systematic OLS-vs-WLS sweep over a galaxy registry and a configuration matrix.
  - Run with `uv run python examples/example_wls_systematic/run_wls_systematic.py`.
- `example_color_gradient/`
  - Colour-gradient measurement across bands.
  - Run with `uv run python examples/example_color_gradient/run_color_gradient_demo.py`.
- `example_qa_comparison/`
  - Generates the QA comparison figures documented in `docs/06-qa-functions.md`
    (solo / 1v1 / 3-way layouts) from reusable preset cases.
  - Run with `uv run python examples/example_qa_comparison/generate_comparison_figures.py`.
- `example_io_validation/`
  - Round-trip validation of the FITS (PRIMARY / ISOPHOTES / CONFIG / META) and
    ASDF serialization paths.
  - Run with `uv run python examples/example_io_validation/validate_fits_asdf.py`.
- `example_asteris_denoised/`
  - Multi-band joint-fit demo on five-band HSC cutouts. **The input data is not
    bundled and is not present on the current development machine**, so this
    example cannot be run as-is; it is retained for its scripts and as the
    provenance of the historical B=5 numbers in `docs/10-multiband.md`.
  - Entry point: `run_isoster_multiband.py`.
- `example_hsc_edge_real/`
  - Real HSC edge-case galaxies with custom masks; LSB-oriented QA overviews.
  - Entry points: `build_custom_masks.py`, then `qa_overview.py`.
- `example_hsc_edgecases/`
  - HSC edge-case sweeps for LSB auto-lock and LSB mode comparison.
  - Entry points: `run_lsb_auto_lock.py`, `run_lsb_mode_sweep.py`.

## Shared Data

- Example FITS inputs tracked in this repo now live under the repository-level `data/` folder, not `examples/data/`.
- Legacy Survey scripts expect files like `data/eso243-49.fits` and `data/ngc3610.fits`.

## Output Policy

- Default rule: each example folder writes to `outputs/<example-folder>/`.
- Exception: `example_huang2013` writes case-level artifacts under `--huang-root/<GALAXY>/mock<ID>/` and only campaign summaries under `outputs/` by default.
- Set `ISOSTER_OUTPUT_ROOT` to override the shared `outputs/` root for examples that use the standard output helper.

## Non-Scope

- Regression and unit checks belong in `tests/`.
- Performance benchmarks belong in `benchmarks/`.
