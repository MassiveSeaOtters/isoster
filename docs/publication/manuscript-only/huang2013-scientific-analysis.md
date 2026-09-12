# Huang2013 mock-galaxy validation: manuscript draft

Measured 2026-09-12 using the corrected AutoProf selection. This is draft
prose, not an independent validation sample or a replacement for the
[detailed reference](../reference/2026-09-12-huang2013-scientific-analysis.md).

We evaluated single-band isophotal fitting on 93 multi-component Sersic
models based on the Huang2013 sample. Each galaxy was rendered in nine
conditions: one genuinely noiseless image at z=0.05 and wide/deep Gaussian
noise realizations at z=0.05, 0.20, 0.35 and 0.50. The images use a Gaussian
PSF with FWHM 0.7 arcsec and 0.168 arcsec pixels. Regenerated, PSF-convolved
truth images reproduced all 837 retained input images exactly at their
stored float32 precision after applying the recorded noise draws. These
are exploratory tests of smooth galaxy models, not survey-image realism
or an independent held-out evaluation.

The accepted selection contains 14,229 requested outcomes across 17 arms:
13,334 successful fits, 58 retained Photutils failures and 837 intentional
Isoster no-weight skips, since the mock inputs supply no variance map.
All nine executable Isoster arms and all four corrected AutoProf arms
completed all 837 galaxy/condition combinations. Photutils' primary arm
completed all noisy cases but only 67 of the 93 noiseless cases. We retain
those failures separately and compare successful primary arms on identical
finite samples rather than substituting successful diagnostic arms.

Using a common elliptical-profile renderer without higher-order harmonics,
we measured residuals against the noiseless truth over the intersection of
the successful arms' finite support, outside one PSF FWHM. For the wide
z=0.05 images, median RMS residual divided by truth RMS was 0.858%, 0.871%
and 1.076% for Isoster, Photutils and AutoProf, respectively (93 galaxies).
For wide z=0.50 these values were 4.971%, 5.067% and 6.754%; the corresponding
deep-image values were 2.595%, 3.023% and 4.686%. These summaries do not
establish a universal ordering: AutoProf had smaller median inner-zone
residuals, and the tools differ in background estimation, radial sampling
and extraction statistics. The support itself contracts toward the less
resolved conditions, so the metrics are conditional on the reported aperture.

Within Isoster, changing to eccentric-anomaly geometry reduced the pooled
median truth-relative RMS by 7.53e-5 in absolute fractional units (95%
galaxy-block bootstrap interval: reductions of 4.05e-5 to 1.46e-4), with
smaller RMS in 63.9% of the 837 paired cases. The median absolute aperture
flux error instead increased by 1.21e-4. Simultaneous harmonic fitting within
the EA configuration gave a median RMS change consistent with zero, although
individual galaxies responded appreciably. Outer regularization strongly
reduced centroid drift and profile roughness without a clearly separated
pooled median RMS improvement. These results support presenting arm-specific
trade-offs, not selecting a single combined accuracy score or claiming that
smoother profiles necessarily recover the truth more faithfully.

Important limitations are the smooth, centered model family; one noise draw
per galaxy/condition; finite support; the absence of extrapolated total-flux
tests; and differing native estimators. AutoProf's retained intensities have
its estimated background subtracted, whereas the injected background is zero;
its aperture-flux differences therefore include sky estimation. AutoProf
centers are also read from an auxiliary file rounded to 0.01 pixel, and its
placeholder stop codes are not convergence evidence. Higher-order coefficients
are kept as basis-dependent diagnostics rather than compared through an
unvalidated cross-tool conversion. Controlled timing results are reported
separately from this scientific analysis.
