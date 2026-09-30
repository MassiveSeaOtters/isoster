# Isoster fixed-center special-case investigation

The constant-center mock hypothesis is supported for the noisy outskirts of the
selected cases, but does not explain most of AutoProf's full-aperture advantage.
Fixing only Isoster's center improves outer Truth RMS by 16.5–29.3% in four noisy
cases; full-aperture RMS improves by only 0.47–1.27%. A noiseless low-drift case
is effectively unchanged. Integrated flux bias can worsen despite lower RMS.

## Selection and controlled change

The user authorized these special-case refits on 2026-09-30. The existing branch
is `analysis/autoprof-comparison-20260923`; no core algorithm or default changed.
The earlier no-refit constraint remains applicable outside these special cases.

The selection was saved before fitting in
`outputs/isoster_fixed_center_cases_20260930/selection.csv`: the four highest
Isoster/AutoProf full-RMS ratios among distinct galaxies in the accepted original
shared-linear harmonic-off comparison, plus the largest-drift remaining galaxy
with ratio >2. Selected ratios range from 2.5301 to 4.2866. This deliberately
selected sample is not a population estimate.

For each case, load the exact saved Isoster configuration and change only
`fix_center: false` to `true`. Existing x0/y0 already equal the injected center.
Keep radius settings, ellipticity/PA freedom, clipping, integrator, gradient,
convergence, harmonic settings, image, absent mask and absent variance unchanged.
This is an actual refit: ellipticity, PA and intensity can respond to the center
constraint. It is not merely replacing centers in a saved reconstruction.
All five new profiles have exactly fixed truth centers and the same SMA grid as
their original profiles. Truth-fixed centering is a stronger assumption than
AutoProf's independently estimated global center; it is not a proposed real-data
default or a test of estimating an Isoster global center.

The single-case gate took 5.5342 seconds including figures and measurements.
Its free-center replay exactly reproduced saved SMA, x0/y0, eps, PA, intensity,
gradient and stop codes. The remaining four fixed-center fits then ran. All five
succeeded; no AutoProf fits were rerun. Fit times are recorded but not presented
as a controlled speed comparison.

## 1. Does Isoster move the center, and why?

| Galaxy / scenario | Maximum center offset [pix] | Radius at maximum / R_ref | Gradient relative error there | Stop code there |
|---|---:|---:|---:|---:|
| ESO185-G054 / deep_z020 | 2.4320 | 5.6070 | 0.1725 | 0 |
| NGC1600 / deep_z020 | 3.7359 | 5.7855 | 0.1952 | 0 |
| NGC3557 / deep_z020 | 1.0017 | 5.2440 | 0.0991 | 0 |
| NGC3268 / wide_z005 | 8.4386 | 4.0265 | 0.1463 | 0 |
| NGC5328 / noiseless_z005 | 0.001143 | 0.1922 | 0.0057 | 0 |

The noisy cases' maximum inner-zone offsets are only 0.0121, 0.0184, 0.00232 and
0.0649 pixels, respectively. Their large displacements occur outside 2 R_ref.
These are not necessarily flagged fits: all maximum-offset rows have code 0 and
gradient relative error below the configured 0.5 threshold. Convergence of the
local fitting criterion does not guarantee a physically correct center.

Verified mechanism in [fitting.py](../../../isoster/fitting.py#L1869): first-order
sine/cosine residuals drive center shifts, with corrections proportional to their
amplitudes divided by the radial gradient. Shallow gradients can therefore turn
small intensity asymmetries into appreciable center motion. Damping, per-iteration
shift clipping and gradient-quality checks do not impose a global center constraint.
With `fix_center=True`, the first-order terms are excluded from geometry updates
and the stopping amplitude selection; see the same file at line 1575.

The additional `diagnose.py` check samples the data and noise-free truth on each
new fixed ellipse, using identical Isoster angular samples and an unweighted
first/second-harmonic solve. No clipping is applied in this diagnostic, so it is
not a replay of every optimization iteration. The first-harmonic vector in
data minus truth measures the added-noise contribution on that same ellipse.
Median outer first-harmonic amplitudes in noisy data are 0.000867–0.001604 ADU,
versus 2.98e-18–1.09e-6 ADU in truth. Median outer noise amplitude divided by the
absolute fixed-fit gradient is 1.191, 1.488, 0.366 and 4.374 pixels respectively.
These are sensitivity diagnostics, not predictions of the exact clipped/damped
center trajectory. In the noiseless case the corresponding value is 1.33e-9 pixels.

All five truth images are exactly invariant under a 180-degree array rotation
about the injected center. Their multi-component shape/PA differences therefore
do not require a radial center shift. Noise-driven first harmonics amplified by
shallow gradients are strongly supported as the noisy outer-drift mechanism.
The noiseless sub-0.002-pixel motion is consistent with sampling/interpolation and
finite convergence tolerances; its exact numerical origin was not isolated.

## 2. Does fixing only the center improve reconstruction?

All table RMS values below are percentages of the truth normalization. The
relative improvement quoted above is a percentage reduction in RMS, not a
percentage-point change.

| Galaxy / scenario | Full RMS: free → fixed | AutoProf full RMS | Outer RMS: free → fixed |
|---|---:|---:|---:|
| ESO185-G054 / deep_z020 | 1.7996 → 1.7767 | 0.4198 | 2.1181 → 1.7684 |
| NGC1600 / deep_z020 | 1.4032 → 1.3884 | 0.4322 | 3.2699 → 2.3122 |
| NGC3557 / deep_z020 | 2.9304 → 2.9167 | 0.9055 | 1.4661 → 1.1678 |
| NGC3268 / wide_z005 | 0.7812 → 0.7748 | 0.3087 | 3.0686 → 2.5476 |
| NGC5328 / noiseless_z005 | 1.3948 → 1.3948 | 0.4021 | 0.7618 → 0.7618 |

Primary table: shared linear renderer, harmonics off for every method. Saved free,
new fixed and saved AutoProf profiles use exactly the same pixels. The support is
the historical final matched support intersected with finite pixels from all nine
available method/mode models (linear-off, spline-off and native). The elliptical
inner cut remains 2 pixels. This loses 1.5018%, 2.4162%, 0.2977%, 0% and 0% of the
historical support, respectively in the table's order. Both old models are
remeasured on the reduced support; historical full-support scores are not compared
directly against new reduced-support scores. Every radial zone is retained.

Why is the overall gain small? The inner zone (2 pixels to 0.5 R_ref) contributes
98.78%, 96.32%, 99.16%, 95.90% and 99.56% of the original Isoster model's total
squared error on this support. The large outer displacements occur where intensity
is low. Removing them helps outer accuracy but leaves the dominant inner error.
The noiseless NGC5328 result is a particularly clear counterexample to center
variation being necessary for a large AutoProf advantage.

There are real tradeoffs. With shared linear harmonic-off reconstruction:

- NGC1600 signed full flux bias worsens from +0.4878% to +0.7630%.
- NGC3268 signed full flux bias worsens from +0.4854% to +0.6720%.
- ESO185-G054 improves from +0.7483% to +0.7305%; NGC3557 improves from
  +1.0883% to +1.0748%; NGC5328 is unchanged at displayed precision (+0.5479%).

The other rendering checks agree that the full-RMS benefit is small: the noisy
cases improve by 0.0076–0.2953% relatively with shared spline-off and
0.1513–1.2072% with native rendering. Native Isoster uses its n=3,4 harmonic model;
native AutoProf remains ellipse-only, so that cross-tool comparison is not a
matched harmonic-on experiment. Both harmonic-off controls are retained. Native
ESO185-G054 flux bias worsens from +0.7377% to +0.7843% despite lower RMS.

## Validation, artifacts and next step

- Five fixed-center fits plus one exact free-center replay; 180 metric rows,
  20 PNG/PDF figure pairs and all configs/profiles saved separately.
- 57 consulted source hashes unchanged; 41 overlap and match the accepted final
  analysis audit. First-harmonic diagnostic inputs have their own hash audit.
- 57 targeted tests passed in 1.22 seconds, with one existing singular-matrix
  warning. Ruff, formatting and whitespace checks passed. Missing AutoProf
  gradients remain NaN; masked-value conversion warnings do not fill them in.
- Inspected the gate QA, noiseless center plot, large-drift case QA, NGC1600
  center plot and NGC3557 native QA. Residuals show full finite coverage; numerical
  comparisons use the recorded common support. The dedicated center plots use
  injected truth; the reused QA panel retains its documented inner-fit reference.

Browse `outputs/isoster_fixed_center_cases_20260930/index.html`. The folder contains
`summary.csv`, `metrics.csv`, `diagnosis.csv`, `first_harmonics.csv`, `selection.csv`,
`completion.json`, and per-case figures, FITS profiles, configurations and audits.
Implementation: `benchmarks/exhausted/analysis/fixed_center_cases.py`; supplemental
read-only diagnostics: `diagnose.py` and `summarize.py` in the output folder.

The center hypothesis now has a narrow demonstrated benefit: reduced noisy outer
RMS, with possible flux-bias regressions. No global fixed-center default follows.
The next useful investigation is the inner-profile intensity/geometry and rendering
difference on the same saved ellipses, including pixel sampling/interpolation and
cumulative flux. This is where most remaining squared error lies in these cases.
No additional refit program is implied. Defaults, historical data and failed/
unsupported cases are unchanged; no merge/push; S4G remains deferred.
