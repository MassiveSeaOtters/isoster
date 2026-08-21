# QA Plotting Reference

This document covers the QA plotting functions exported by `isoster.plotting`.
For style conventions and layout rules, see `docs/agent/qa-figures.md` (agent-internal, not published) — available in a source checkout, not on the published site.

## Quick Start

```python
import isoster
from isoster.config import IsosterConfig
from isoster.model import build_isoster_model

# Fit isophotes
config = IsosterConfig(x0=128, y0=128, sma0=10.0)
results = isoster.fit_image(image, mask, config)
isophotes = results["isophotes"]

# Build 2D model for residual display
model = build_isoster_model(image.shape, isophotes)

# Single-method QA figure
isoster.plot_qa_summary(
    title="NGC 1234",
    image=image,
    isoster_model=model,
    isoster_res=isophotes,
    mask=mask,
    filename="qa_summary.png",
)
```

## QA Figure Generation Standard

The QA figures are meant to audit the same products that feed
benchmark metrics and ranking tables. Treat the figure as a visual
manifest for the run, not as a separate analysis path.

### Use persisted fit artifacts

Build comparison figures from the persisted fit products whenever they
exist:

- `profile.fits`: the canonical 1D profile table for one method/arm.
- `model.fits`: the rendered 2D model and residual image for that
  method/arm.
- `MANIFEST.json`: image metadata such as pixel scale, zeropoint,
  initial geometry, and image-noise estimate.
- `inventory.fits` / `run_record.json`: metrics, flags, runtime, and
  effective configuration provenance.

Avoid recomputing profile quantities only for plotting unless the plot
explicitly documents that it is a diagnostic overlay. The displayed
profile, residual map, model contours, and flags should correspond to
the data that will be scored.

### Required visual context

A production QA figure for a single method should include:

- the original image with selected isophote overlays;
- the rendered model with model iso-brightness contours;
- the residual image using the same model that feeds residual metrics;
- a surface-brightness profile with uncertainties when available;
- a radial residual or profile-difference panel;
- geometry panels for ellipticity, position angle, and centroid drift;
- stop-code or quality-state markers for isoster/photutils profiles;
- the most important flags or failure status in the title, caption, or
  associated table.

Cross-arm and cross-tool figures should keep the image stretch,
surface-brightness convention, x-axis transform, and y-axis clipping
consistent across methods. Differences should be visible because the
methods differ, not because the plotting convention changed between
panels.

### Coordinate and axis conventions

Use `sma ** 0.25` for radial profile panels. This compresses the outer
profile while retaining enough inner-region resolution for centroid,
PA, and ellipticity diagnostics. Use scatter points with errorbars for
measured profiles. Use lines only for reference curves, model profiles,
or smoothed diagnostic guides.

Normalize position angles before comparison. A jump near 180 degrees is
usually an angle-wrapping artifact, not a physical twist. The helper
`normalize_pa_degrees()` uses a double-angle convention and should be
preferred over ad hoc wrapping.

When choosing y-limits, do not let a few outer errorbars dominate the
panel. Clip limits from the measured values first, then draw errorbars.
The errorbars remain visible where they are informative, but they do
not compress the full profile into an unreadable band.

### Surface-brightness profiles

`log10` is the default SB profile scale because it preserves the
long-standing magnitude/profile convention and makes high-S/N inner
regions easy to compare across runs. Use `sb_profile_scale="asinh"` or
`"arcsinh"` when low-S/N outskirts, near-zero intensity, or negative
residual behavior needs to remain visible.

When `sb_zeropoint` and `pixel_scale_arcsec` are both provided, both
the log10 and asinh profiles are calibrated in mag/arcsec². In the
high-S/N regime, the calibrated asinh profile should visually coincide
with the log10 magnitude profile. In the low-S/N regime, the asinh
profile remains finite through zero and negative intensity values. The
plot draws a dashed horizontal `I = 0` reference line so the transition
point is explicit.

When no zeropoint is provided, the default profile is `log10(I)` and
the asinh profile is an uncalibrated `asinh(I / b)` diagnostic. Use
uncalibrated profiles only for relative QA, not for publication
surface-brightness values.

### Cross-arm and cross-tool comparisons

For cross-arm comparison, keep tool-specific diagnostics visible:
centroid drift, stop-code fractions, first-isophote retry, harmonic
behavior, and completeness all explain why one arm is more stable than
another. These diagnostics are fair within one tool because the arms
share the same output contract.

For cross-tool comparison, separate visual evidence from ranking
semantics. Different tools do not expose the same internal failure
axes: AutoProf, for example, does not emit photutils-style stop codes
and may use a shared center by construction. A cross-tool QA figure
should therefore show these method-specific diagnostics without
assuming they are all scoreable on the same scale. Use tool-neutral
model residuals and runtime for ranking; use method-specific flags to
interpret the result.

### Demo and regression workflow

When validating a plotting change, regenerate the QA PNGs instead of
using cached per-arm results. In exhausted benchmark campaigns, set
`execution.skip_existing: false` or delete the relevant arm directory.
Cached profiles can validate table reconstruction, but they do not
prove that a new plotting option, contour overlay, or SB transform was
actually rendered.

## Functions

### `plot_qa_summary`

Standard QA figure for a single isoster fit.

**Layout**: left column shows the galaxy image with isophote overlays,
the reconstructed model with iso-brightness contours, and the residual
map; right column shows 1D profiles (surface brightness, residual,
ellipticity, PA, a3/b3, a4/b4) sharing an SMA^0.25 x-axis.

```python
isoster.plot_qa_summary(
    title,              # Figure title
    image,              # 2D galaxy image
    isoster_model,      # 2D reconstructed model
    isoster_res,        # list[dict] from fit_image()["isophotes"]
    photutils_res=None, # optional photutils IsophoteList for comparison
    mask=None,          # 2D bool mask (True = masked)
    filename="qa_summary.png",
    relative_residual=False,   # True: show (model-data)/data [%]
    sb_zeropoint=None,         # mag zeropoint (see SB section below)
    pixel_scale_arcsec=None,   # arcsec/pix; must accompany sb_zeropoint
    sb_profile_scale="log10",  # or "asinh" / "arcsinh"
    sb_asinh_softening=None,   # optional positive softening scale
)
```

### `plot_qa_summary_extended`

Extended QA figure with dedicated harmonic visualization panels.
Adds odd-order (3, 5, 7) and even-order (4, 6) harmonic panels below
the standard profile plots.

**Harmonic notation used throughout ISOSTER.** Three quantities, kept
distinct because conflating them is how coefficients get normalized twice:

| Symbol | Meaning | Where it lives |
|---|---|---|
| $A_n^\mathrm{raw}$, $B_n^\mathrm{raw}$ | Raw fit amplitudes, in intensity units | Internal to the harmonic solve; never stored or plotted in single-band |
| $a_n$, $b_n$ | Stored coefficients: raw amplitudes divided by $a\,\lvert dI/da\rvert$ (`sma * abs(gradient)`, `isoster/fitting.py`). Signed, dimensionless, a fractional *radial* displacement of the isophote. Coincides with the Bender form $-A_n^\mathrm{raw}/(a\,dI/da)$ whenever the gradient is negative — the normal outward-declining case — and differs from it in sign on the rare positive-gradient row | The results-dict keys `a3`, `b3`, `a4`, … |
| $\sqrt{a_n^2 + b_n^2}$ | Combined amplitude. Non-negative and dimensionless | Computed at plot time in `harmonic_mode="amplitude"` |

Lowercase means normalized: the label on a figure matches the column name in
the results dict.

`compute_deviations` performs the normalization at fit time, so the stored
values are dimensionless as they stand and **must not be normalized a second
time**. Note the one place the two conventions part company: the plot-time
helper `_normalize_harmonic_for_plot`, used only by the multi-band raw paths,
applies the signed Bender form $-A_n/(a\,dI/da)$, whereas single-band storage
uses the absolute value. They agree except on positive-gradient rows, where
they differ in sign. They are directly comparable across radius and across galaxies. They
are comparable across *tools* only where the tools share the basis and the
normalization convention. That is established for `photutils`. For AutoProf
it is **unverified** — not known to differ, simply never checked — so its
harmonics are excluded from published cross-tool comparisons (see
[`technical/1.4.6`](technical/1.4.6-diagnostics-qa.md)).

A `normalize_harmonics` option that divided the stored values by intensity was
removed in 1.0.0: it produced a quantity with units of one over intensity,
which changed if the image's flux calibration changed.

```python
isoster.plot_qa_summary_extended(
    title,
    image,
    isoster_model,
    isoster_res,
    harmonic_orders=None,       # auto-detect from isophote keys
    harmonic_mode="coefficients",  # "coefficients" or "amplitude"
    relative_residual=False,
    mask=None,
    filename="qa_summary_extended.png",
    sb_zeropoint=None,          # mag zeropoint (see SB section below)
    pixel_scale_arcsec=None,    # arcsec/pix; must accompany sb_zeropoint
    sb_profile_scale="log10",   # or "asinh" / "arcsinh"
    sb_asinh_softening=None,
)
```

| `harmonic_mode` | Display |
|-----------------|---------|
| `"coefficients"` | Individual a_n (filled) and b_n (open) per order |
| `"amplitude"` | Combined amplitude `sqrt(a_n^2 + b_n^2)` per order, non-negative |

### Surface brightness convention

When `sb_zeropoint` is supplied, the SB panel of both
`plot_qa_summary` and `plot_qa_summary_extended` displays

```
μ [mag/arcsec²] = -2.5 · log10(I_per_pix / pixarea) + sb_zeropoint
pixarea = pixel_scale_arcsec ** 2
```

`sb_zeropoint` and `pixel_scale_arcsec` must be passed as **two
separate values** — never pre-combined into a single effective
zeropoint. Passing one without the other raises `ValueError`; passing
neither keeps the panel in `log10(I)` mode. Typical survey pairs:

| Survey | `sb_zeropoint` | `pixel_scale_arcsec` |
|--------|----------------|----------------------|
| HSC coadd | 27.0 | 0.168 |
| DECaLS / BASS / MzLS (LegacySurvey) | 22.5 | read from header (~0.262) |
| SDSS imaging | 22.5 (calibrated `nanomaggies` reference) | 0.396 |

For surveys with a header-provided pixel scale, read `PIXSCALE` (or
derive it from `CD1_1` / `CDELT1`) and pass it alongside the
survey-specific zeropoint — do not hard-code a value that could
diverge from the image WCS.

Set `sb_profile_scale="asinh"` to use an arcsinh/asinh profile instead
of the default log10 profile. In calibrated mode this uses the standard
asinh-magnitude form

```
μ_asinh = zp - (2.5 / ln 10) * [asinh(f / 2b) + ln b]
f = I_per_pix / pixarea
```

where `b` is `sb_asinh_softening` in flux per arcsec², or an automatic
positive profile scale when omitted. For `f >> b`, `μ_asinh` approaches
the regular log10 magnitude profile. The SB panel also draws a dashed
horizontal line at the finite y-value corresponding to `I = 0`.

### `plot_comparison_qa_figure`

Multi-method comparison figure with automatic layout selection.

```python
isoster.plot_comparison_qa_figure(
    image,                # 2D galaxy image
    profiles,             # dict[str, dict[str, ndarray]]
    title="",
    output_path="qa_comparison.png",
    models=None,          # dict[str, 2D ndarray]
    mask=None,
    method_styles=None,   # override METHOD_STYLES
    relative_residual=False,
    sb_zeropoint=None,
    pixel_scale_arcsec=None,
    sb_profile_scale="log10",
    sb_asinh_softening=None,
    dpi=150,
)
```

**Automatic layout modes** (selected by number of methods in `profiles`):

| Methods | Mode | Left column |
|---------|------|-------------|
| 1 | Solo | Image, model, residual |
| 2 | One-on-one | Image, isoster residual, other residual |
| 3+ | Three-way | Image, one residual panel per method |

Right column (all modes): SB with errorbars, relative SB difference,
ellipticity, PA, centroid offset — sharing SMA^0.25 x-axis.

**Building profiles** for this function:

```python
from isoster.plotting import build_method_profile

# From isoster/photutils results (list of dicts)
profile = build_method_profile(isophotes)

# From autoprof results (dict of arrays)
profile = build_method_profile(autoprof_data)

# Inject optional metadata
profile["runtime_seconds"] = 0.15
profile["retries"] = 1

profiles = {"isoster": profile_iso, "photutils": profile_phot}
models = {"isoster": model_iso, "photutils": model_phot}

isoster.plot_comparison_qa_figure(image, profiles, models=models)
```

### `draw_isophote_overlays`

Draw isophote contours on a matplotlib axis.  When harmonic coefficients
(a3/b3, a4/b4, ...) are present, draws the actual non-elliptical shape.

```python
from isoster.plotting import draw_isophote_overlays

draw_isophote_overlays(
    axis,                 # matplotlib Axes
    isophotes,            # list[dict] with sma, x0, y0, eps, pa
    step=10,              # draw every N-th isophote
    line_width=1.0,
    alpha=0.7,
    edge_color=None,      # None = color by stop code
    draw_harmonics=True,  # False = force pure ellipses
)
```

When `draw_harmonics=True` (the default) and the isophote dicts contain
keys like `a3`, `b3`, `a4`, `b4`, the overlay traces a
harmonic-perturbed contour using 360 sample points. Each row is gated:
if the raw harmonic perturbation exceeds `DELTA_ROW_GATE = 0.5`, that
row falls back to a pure ellipse so a single noisy outer coefficient
cannot dominate the QA panel.

The contour formula follows the producing tool. Isoster rows with
`use_eccentric_anomaly=False` use the image-azimuth phi-mode formula;
isoster rows with `use_eccentric_anomaly=True` and photutils rows use
the eccentric-anomaly/psi formula; AutoProf rows use pure ellipses
because its SuperEllipse coefficient scale is not yet ported.
Set `draw_harmonics=False` to force pure-ellipse patches for every
tool.

Model panels use a different overlay: `overlay_model_contours()` draws
iso-brightness contours from the rendered 2D model itself. These are
not per-row isophote shapes; they reveal structure in the model image,
including spline ringing or harmonic artifacts that may not be obvious
from the arcsinh stretch alone.

## Helper Functions

| Function | Purpose |
|----------|---------|
| `build_method_profile(data)` | Convert list-of-dicts or array-dict into standardized profile arrays. Preserves harmonic keys. |
| `configure_qa_plot_style()` | Apply shared matplotlib rcParams for QA figures. |
| `derive_arcsinh_parameters(image)` | Compute arcsinh stretch parameters for display. |
| `make_arcsinh_display_from_parameters(image, low, high, scale, vmax)` | Apply arcsinh stretch to an image for display. |
| `model_isobrightness_levels(model)` | Choose model intensity levels for contour overlays. |
| `overlay_model_contours(ax, model)` | Draw model iso-brightness contours on an existing axes. |
| `plot_profile_by_stop_code(ax, x, y, stop_codes, ...)` | Scatter plot with per-stop-code color and marker. |
| `normalize_pa_degrees(pa_deg, anchor=None)` | Unwrap PA jumps using the double-angle trick. |
| `style_for_stop_code(code)` | Return color/marker/label dict for a stop code. |

## Method Styles

The `METHOD_STYLES` dict defines default visual styles for each method
in comparison figures:

| Method | Color | Marker | Overlay color |
|--------|-------|--------|---------------|
| isoster | `#1f77b4` (blue) | `o` filled | white |
| photutils | `#d62728` (red) | `s` open | orangered |
| autoprof | `#2ca02c` (green) | `^` open | limegreen |

Override per-call via the `method_styles` parameter.

## Synthetic Comparison Script

`examples/example_qa_comparison/generate_comparison_figures.py` runs
isoster (and optionally photutils) on synthetic Sersic galaxies with
known truth, then generates QA figures using `plot_comparison_qa_figure`.

### CLI

```bash
# Run all preset cases
uv run python examples/example_qa_comparison/generate_comparison_figures.py

# Single case
uv run python examples/example_qa_comparison/generate_comparison_figures.py \
    --case n4_eps04_snr100_noisy

# Custom output directory, skip photutils
uv run python examples/example_qa_comparison/generate_comparison_figures.py \
    --output outputs/my_qa_run --no-photutils
```

### Preset cases

| Name | n | eps | SNR | Notes |
|------|---|-----|-----|-------|
| `n1_eps07_high_ellipticity` | 1.0 | 0.7 | inf | Stress test for high ellipticity |
| `n1_eps04_snr100_noisy` | 1.0 | 0.4 | 100 | Exponential disk with noise |
| `n4_eps04_snr100_noisy` | 4.0 | 0.4 | 100 | de Vaucouleurs profile with noise |

### Reuse from other benchmarks

Import `PRESET_CASES` and `run_single_case()` to run comparisons
from any benchmark script:

```python
from examples.example_qa_comparison.generate_comparison_figures import (
    PRESET_CASES, run_single_case,
)

# Run a preset case
stats = run_single_case(
    PRESET_CASES["n4_eps04_snr100_noisy"],
    output_dir="outputs/my_benchmark",
    case_name="devaucouleurs_noisy",
)

# Define a custom case
custom_case = {
    "n": 2.0, "R_e": 30.0, "I_e": 500.0,
    "eps": 0.5, "pa": 0.8, "snr": 50,
    "oversample": 5, "shape": (800, 800),
}
stats = run_single_case(
    custom_case,
    output_dir="outputs/my_benchmark",
    case_name="custom_n2",
    isoster_config_overrides={"maxit": 100},
)

# stats keys: case_name, n_isophotes, wall_time_seconds,
#   median_frac_resid, max_abs_frac_resid, convergence_rate
```

## Stop Code Colors

Isophote overlays and profile scatter plots are colored by stop code
when no explicit `edge_color` is given:

| Code | Meaning | Color |
|------|---------|-------|
| 0 | Converged | Blue (`#1f77b4`) |
| 1 | Too many flagged pixels | Orange (`#ff7f0e`) |
| 2 | Max iterations reached | Green (`#2ca02c`) |
| 3 | Too few valid points | Red (`#d62728`) |
| -1 | Gradient error / strike | Purple (`#9467bd`) |
