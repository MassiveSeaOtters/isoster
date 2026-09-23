# Huang2013 mock-galaxy validation: manuscript draft

Updated 2026-09-23 using the audited zero-background AutoProf selection,
the fixed two-pixel inner evaluation cut and the complete four-mode saved-fit
reconstruction analysis. See the
[final detailed reference](../reference/2026-09-23-huang-final-analysis.md)
for the audited statistics, QA, coverage and caveats; the
[original-support baseline reference](../reference/2026-09-23-two-pixel-analysis.md)
remains preserved. No refitting accompanied this analysis.

We evaluated single-band isophotal fitting on 93 multi-component Sersic
models based on the Huang2013 sample. Each galaxy was rendered in nine
conditions: one genuinely noiseless image at z=0.05 and wide/deep Gaussian
noise realizations at z=0.05, 0.20, 0.35 and 0.50. Images use a Gaussian
PSF with FWHM 0.7 arcsec and 0.168 arcsec pixels. Regenerated PSF-convolved
truth images reproduced all 837 retained inputs exactly at stored float32
precision after applying the recorded noise draws. These are exploratory
tests of smooth galaxy models, not held-out survey-image validation.

To compare analysis of prepared images rather than background estimation,
we fixed AutoProf's sky to the known injected value, zero, while retaining
its native noise-estimation algorithm, corrected PA convention and existing
retry policy. No Isoster or Photutils fit was repeated. The accepted
14,229-outcome selection contains 13,283 successful fits, 58 Photutils
failures, 51 AutoProf errors and 837 intentional Isoster no-weight skips,
because the inputs supply no variance map. All nine executable Isoster
arms completed all 837 conditions. AutoProf succeeded in 3,297 of 3,348 fits
across four arms; 48 errors involve empty outer-ring extraction and three
involve ellipse initialization. Photutils' primary completed all noisy
cases but 67 of 93 noiseless cases. Failures are retained separately rather
than replaced with successful diagnostic arms or earlier campaigns.

Using a common elliptical-profile renderer without higher-order harmonics,
we measured residuals against truth over the intersection of successful
arms' finite support, at elliptical radius at least two pixels. On matched
successful primary samples, median RMS residual divided by truth RMS was
1.346%, 1.396% and 1.454% for Isoster, Photutils and AutoProf at wide z=0.05
(91 galaxies). At wide z=0.50 the values were 4.858%, 4.845% and 8.367%
(92 galaxies); at deep z=0.50, 3.878%, 4.021% and 7.967% (88 galaxies).
Across 794 matched galaxy/scenario inputs, Isoster had the smallest
full-aperture RMS in 402 cases, Photutils in 184 and AutoProf in 208.
AutoProf instead had the smallest absolute aperture-flux bias in 630 cases.
These repeated conditions are not independent galaxies. Execution coverage,
radial-zone errors and reconstruction sensitivity preclude a single ranking.

The earlier fixed-sky versus estimated-sky comparison remains documented
under its historical PSF-cut aperture. It is not an isolated deterministic
measure of background subtraction: AutoProf internally reseeds its optimizer,
noise estimates and retry triggers can change, and the shared finite support
can change. No refitting accompanied the new two-pixel evaluation.

On the original-support harmonic-off baseline, `geom_ea` and `geom_simul_ea`
give smaller full-aperture RMS
in 51.02% and 53.05% of 837 pairs against the default. Their median absolute
RMS differences have 95% galaxy-block bootstrap intervals spanning zero;
absolute flux-bias wins occur in only 40.50% and 40.98% of pairs. The
stronger RMS advantage under the previous PSF cut does not persist across
all nine scenario groups. These are configuration and aperture trade-offs,
not a combined accuracy score or a new-default recommendation.

Limitations include smooth centered models, one noise draw per condition,
finite support, unrecorded AutoProf optimizer seeds, and differing
native sampling/extraction estimators. Noiseless AutoProf fits retain its
internal nonzero noise-scale fallback, not added image noise. AutoProf's
auxiliary centers are rounded to 0.01 pixel and placeholder stop codes are
not convergence evidence.

The comprehensive reconstruction analysis retained all 17 arms across four modes,
producing 56,916 arm/mode outcomes, 227,664 radial measurement rows and 8,370
individual QA page pairs. All 53,132 original baseline zone measurements were
reproduced, and 36,153 current plus 30,862 historical source hashes were verified.
Shared linear harmonic-off, shared spline harmonic-off, shared spline raw-polar
n=3,4 harmonic-on, and native reconstruction were evaluated on one recorded
intersection of available model support for each input. Its median size is
96.826% of the original aperture; inner and middle support are unchanged where
nonempty. Unsupported bases, missing coefficients, failed fits and empty inner
zones remain unavailable rather than being replaced by zeros or alternate arms.

On this matched support, shared-linear full-aperture RMS wins are 402/184/208
for Isoster/Photutils/AutoProf among 794 primary triples. Changing only the common
off renderer to the Photutils spline/rasterizer changes these counts to
38/18/738. Shared raw-polar harmonics give 61/38/673 among 772 triples; 22 additional
noiseless primary comparisons lack supported harmonic inputs. Adding harmonics
to the identical shared-spline control increases median full RMS by 0.06546,
0.08075 and 0.54654 percentage points within Isoster, Photutils and AutoProf,
respectively. Renderer sensitivity is therefore larger than this shared harmonic
effect, and the same saved fitting profiles do not support a renderer-independent
ranking.

Native results give 637/2/133 full-RMS wins among 772 triples, while AutoProf
has the smallest absolute flux bias in 670/772. Native AutoProf is a verified
saved zero-background ellipse model, without the measured intensity harmonics;
this comparison is not a common harmonic-on experiment. Native Isoster's radial
harmonic correction lowers full RMS in 723/837 paired inputs and has a negative
median change in 90/93 galaxies, but lowers outer RMS in only 160/837 inputs.
Native EA reconstruction was validated against the real stored angular basis;
`geom_ea` and `geom_simul_ea` outperform the native default in only 291/837 and
300/837 full-RMS pairs. Large unfavorable simultaneous-EA reconstructions remain
in the results. Shared polar harmonic-on reconstruction explicitly excludes the
2,511 EA arm outcomes; it never rotates them as if their basis were polar.

These results are descriptive. Conditions for a galaxy are correlated;
per-scenario and per-galaxy consistency tables are provided, without treating
all 837 conditions as independent observations or claiming new significance.
No extrapolated total-flux or real-galaxy claim follows from these tests.
Controlled Stage 4 timing results remain separate and unchanged.
