# Huang2013 harmonic reconstruction demonstration

Preserve the published shared Isoster linear, harmonic-off baseline. Select
three distinct galaxies with the largest primary Isoster/AutoProf full-aperture
Truth RMS ratios, requiring a difference greater than one percentage point
and successful finite primary results for all three tools. These are deliberately
selected difficult examples, not a representative population estimate.

Reuse the calibrated raw polar n=3,4 conversion and Photutils reconstruction
for an all-tool harmonic-on comparison. Also render the identical spline model
without harmonics: comparing only to the linear baseline would mix two changes.
Reject missing amplitudes or incompatible angle bases, never replace them by zero.

Include native reconstructions: Isoster harmonic-on, Photutils harmonic-on,
and AutoProf's saved native genmodel (explicitly harmonic-off for current arms).
Do not call the last option a three-tool native harmonic-on comparison. No
new fitting or fitted Fourier-shape arm is authorized by this demonstration.

Reproduce historical baseline metrics on the all-arm finite aperture before
scoring alternatives. Use a second common aperture across all demonstration
models, retaining the fixed 2-pixel cut, and report its coverage. Save original
and matched-aperture scores separately. Measure all, inner, mid, outer Truth
RMS, signed/absolute flux bias and data residuals using existing metric helpers.

Reuse approved cross-tool QA for each mode, retain separate captions, and
provide a radial-zone metric summary. Save model images, common support,
selection, source hashes and alternative measurements into a new outputs folder.
Validate coefficients on synthetic rings and run one real case before three.
No old data/results or production analysis defaults are changed.
