# Huang2013 mock-galaxy validation: manuscript draft

Updated 2026-09-21 using the audited zero-background AutoProf selection.
This supersedes the estimated-background draft; see the
[detailed reference](../reference/2026-09-21-autoprof-zero-background.md)
for complete conditions, definitions, outcomes and caveats.

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
arms' finite support, outside one PSF FWHM. On matched successful primary
samples, median RMS residual divided by truth RMS was 0.858%, 0.871% and
0.622% for Isoster, Photutils and AutoProf at wide z=0.05 (91 galaxies).
At wide z=0.50 the values were 5.079%, 5.181% and 4.638% (92 galaxies);
at deep z=0.50, 2.688%, 3.006% and 2.830% (88 galaxies). AutoProf has the
smaller all-aperture median in eight of nine conditions, but this is not
a universal ordering: execution coverage, radial-zone errors and individual
outliers differ. Isoster's robust execution and arm-specific behavior should
be reported alongside these successful-fit comparisons.

AutoProf's wide z=0.05 median signed aperture-flux bias is +0.399%, compared
with -3.311% in the earlier estimated-background run on the same successful
galaxies. This supports using the known prepared sky for the primary
comparison, but is not an isolated deterministic measure of sky subtraction:
AutoProf internally reseeds its optimizer, its noise estimates and retry
triggers can change, and the common support fraction changes in 542 of 837
images. Its median increases from 0.7133 to 0.7846. Metrics for unchanged
Isoster/Photutils fits were therefore also recomputed.

Within Isoster, eccentric-anomaly geometry reduces pooled median
truth-relative RMS by 7.48e-5 in absolute fractional units (95% galaxy-block
bootstrap interval: reductions of 4.06e-5 to 1.48e-4), with smaller RMS in
63.9% of 837 pairs. Median absolute aperture-flux error instead increases
by 1.08e-4. Simultaneous harmonic fitting within EA gives a pooled RMS
change consistent with zero, although individual cases respond. Outer
regularization reduces centroid drift and profile roughness without a
clearly separated pooled median RMS improvement. These remain configuration
trade-offs, not a combined accuracy score or a new-default recommendation.

Limitations include smooth centered models, one noise draw per condition,
finite support, unrecorded AutoProf optimizer seeds, and differing
native sampling/extraction estimators. Noiseless AutoProf fits retain its
internal nonzero noise-scale fallback, not added image noise. AutoProf's
auxiliary centers are rounded to 0.01 pixel and placeholder stop codes are
not convergence evidence. Higher-order coefficients remain basis-dependent
diagnostics; the common renderer does not test harmonic reconstruction.
No extrapolated total-flux or real-galaxy claim follows from these tests.
Controlled Stage 4 timing results remain separate and unchanged.
