# Huang2013 harmonic reconstruction demonstration

## Later QA layout update, same measurements

Updated figures are in `outputs/huang2013_harmonic_qa_zones_20260923` in the
isolated checkout. They add a small table/panel gap and ALL/INNER/MIDDLE/OUTER
Truth RMS and signed Flux Bias values. Residual display now includes every
finite model pixel, including those outside the evaluation cuts. Genuine
missing model coverage stays blank; metrics retain the original common mask.
SB limits use finite profile values only, with no I=0 guide. The historical
figure-description paragraphs below describe the earlier v2 export.

## Scope and immutable inputs

Three deliberately selected difficult cases; no population ranking, new fitting,
new mock generation, or changes to historical data. Work is on
`feature/huang-harmonic-demo-20260923` in the isolated checkout
`/Users/shuang/code/isoster-autoprof-zero-background-20260921-remote`.
Inherited uncommitted QA and two-pixel analysis changes are preserved.

Input measurements:
`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/huang2013_scientific_analysis_two_pixel_2026_09_23`.

Final outputs, relative to that checkout:
`outputs/huang2013_harmonic_demo_20260923_v2`.
The earlier gate and first demonstration remain intact; v2 adds a saved-options
check for native AutoProf and package-version provenance without changing scores.

Select the three distinct galaxies with largest Isoster/AutoProf baseline
full-aperture Truth RMS ratios, requiring an absolute gap above one percentage
point and successful finite designated primaries for all tools. Result:
ESO185-G054/deep_z020, NGC3557/deep_z020, NGC1600/deep_z035.
Their original RMS ratios are 4.286562, 3.235626, 3.171610, respectively.
Primaries are Isoster ref_default, Photutils baseline_median, AutoProf baseline.

## Four reconstruction modes

1. `shared_baseline`: unchanged Isoster linear renderer, harmonics off, applied
   to all three profiles. Reproduced original measurements before comparison.
2. `shared_spline_off`: Photutils spline/rasterization applied to all profiles,
   with harmonics off. This control isolates the renderer change.
3. `shared_spline_on`: identical spline/rasterization, with calibrated raw polar
   intensity harmonics n=3,4 from each tool. This is the all-tool harmonic-on
   comparison, including AutoProf's measured intensity coefficients.
4. `native`: Isoster's native radial-shift harmonic reconstruction; Photutils'
   native additive harmonic reconstruction; saved AutoProf native ellipse model.
   **AutoProf's native model is not intensity-harmonic-enabled.** Its native
   renderer supports fitted Fourier shape modes, but these arms measured intensity
   harmonics only. Saved options verify zero background and no fitted shape modes.

Raw conversion reuses `benchmarks/harmonic_scale/conventions.py`: invert Bender
normalization for Isoster/Photutils; undo AutoProf's FFT factor/sign using b0;
rotate AutoProf polar coefficients from the image frame to its major axis.
An artificial gradient -1 transports raw coefficients through the Photutils
interface; it is not an inferred galaxy gradient. EA coefficients and missing
amplitudes are rejected by this common polar path. Native Isoster supports EA,
verified synthetically, but these primary demonstration profiles are polar.
Native Photutils interpolates Bender coefficients and gradients separately;
raw transport interpolates raw amplitudes, so those modes need not be identical.

## Pixel support, metrics and QA

First reproduce 36 original tool/zone measurements on the all-successful-arm
baseline aperture. Then intersect that aperture with finite support of all
12 reconstruction models per input. Native AutoProf writes zero outside its
positive surface-brightness model; exclude those unsupported pixels.
Retained original pixels: ESO185-G054 99.8513%, NGC3557 99.7968%, NGC1600 99.8101%.
Common counts: 61,086; 21,123; 27,336, respectively.

All alternatives are evaluated on identical pixels for a given input and zone,
with the fixed 2-pixel elliptical inner cut. Zones retain r < 0.5 R_ref,
0.5 R_ref <= r < 2 R_ref, and r >= 2 R_ref; all intersect the common aperture.
Truth is the saved PSF-convolved noise-free image. Metrics are
Truth RMS = sqrt(sum((M-T)^2)/sum(T^2)), signed flux bias = sum(M-T)/sum(T),
absolute flux bias = abs(signed flux bias), and noise-normalized data residual RMS.
CSVs store fractions; figures and tables below use percentages.

Approved cross-tool QA is reused. Residual maps show data minus model, with
one pooled symmetric 99th-percentile scale across every mode and tool per input.
Native 1-D curves are unchanged between pages. Fit times are inherited parallel
campaign measurements, not renderer speed or controlled timing. Reconstruction
durations are saved separately; native AutoProf's duration is file loading only.
The NGC3557 AutoProf `OK*` label retains the historical fit-record flag
`INNER_RESID_LARGE(1.5)`; it is not a new matched-aperture assessment.

## Results

Full-aperture Truth RMS (%) on matched pixels:

| Galaxy / scenario | Mode | Isoster | Photutils | AutoProf |
|---|---|---:|---:|---:|
| ESO185-G054 / deep_z020 | Shared baseline | 1.79978 | 1.86912 | 0.41986 |
| | Shared spline off | 3.74872 | 3.80389 | 1.58965 |
| | Shared spline on | 3.76552 | 3.82734 | 1.62770 |
| | Native | 1.60003 | 3.82702 | 0.33429 |
| NGC3557 / deep_z020 | Shared baseline | 2.93043 | 3.02705 | 0.90567 |
| | Shared spline off | 6.61216 | 6.61331 | 2.41295 |
| | Shared spline on | 6.69401 | 6.77168 | 2.43612 |
| | Native | 2.44660 | 6.77168 | 0.80870 |
| NGC1600 / deep_z035 | Shared baseline | 2.10005 | 2.09642 | 0.66212 |
| | Shared spline off | 3.91486 | 3.82887 | 1.56380 |
| | Shared spline on | 3.91319 | 3.81867 | 1.70684 |
| | Native | 1.81506 | 3.83127 | 0.68051 |

Findings limited to these three inputs:

- Isoster's native harmonic-on model reduces full-aperture RMS relative to its
  own harmonic-off baseline by 11.0984%, 16.5105%, and 13.5710%, respectively.
  It does not overtake AutoProf on full-aperture RMS in these examples.
- This improvement is not uniform across radii: Isoster's native outer-zone RMS
  worsens in all three cases (2.150 to 2.900%; 1.476 to 1.884%; 5.194 to 8.502%).
- The identical-spline on/off comparison shows small full-aperture changes,
  generally worse with harmonics. All three AutoProf spline-on RMS values worsen.
  Outer-zone RMS worsens for every tool in every selected case in this comparison.
- Renderer choice changes full-aperture RMS much more than turning on raw n=3,4
  in the matched spline control. The large spline-related degradation occurs
  mainly in the inner zone. This localizes a follow-up, but does not establish
  whether interpolation, rasterization, radial sampling or fitted geometry is
  responsible. Do not attribute it to a fitting bug without further tests.
- AutoProf is not superior in every zone: Isoster has lower mid-zone baseline
  RMS for ESO185-G054 and NGC1600, and lower outer-zone baseline RMS for NGC3557
  and NGC1600. Full-aperture dominance does not imply uniform radial dominance.
- Absolute flux bias remains a separate diagnostic. For example NGC1600's
  Isoster native RMS improves while its absolute full-aperture bias increases
  from 0.45445% to 0.49633%. Do not select a model using RMS alone.

## Products and verification

- 36 model FITS files, three common-support FITS files, 144 alternative metric
  rows, 36 reproduced original tool/zone rows, selection and timing CSVs.
- 15 PNG/PDF figure pairs: four approved cross-tool QA pages and one three-zone
  plus full-aperture metric summary per input; separate case captions.
- 115 source-file hashes unchanged; code hashes and package versions recorded.
- Synthetic gates cover circular and flattened polar signals, AutoProf
  sign/scale/phase, invalid basis/missing coefficients, and native Isoster polar/EA.
- 661 unit tests passed, with 10 existing warnings; Ruff and whitespace checks
  passed. The actual one-case gate completed its case in 9.212 seconds before
  the three-case execution. Final case durations: 8.992, 7.978, 8.264 seconds.
- Representative QA and zone-summary images were visually inspected. Production
  analysis defaults are unchanged. No merge, push, S4G analysis, or full-sample
  harmonic ranking was performed.

Reproduce into a NEW destination:

```bash
UV_PROJECT_ENVIRONMENT=/Users/shuang/.venvs/isoster-zero-background \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg \
uv run --with pandas==2.3.3 python -m benchmarks.exhausted.analysis.harmonic_demo \
  /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/huang2013_scientific_analysis_two_pixel_2026_09_23 \
  outputs/huang2013_harmonic_demo_new --count 3
```

Next: review these demonstrations with the user before a population-wide
harmonic-inclusive analysis or any changes to the model renderer itself.
