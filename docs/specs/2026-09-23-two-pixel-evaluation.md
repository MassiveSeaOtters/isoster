# Two-pixel benchmark evaluation

User decision: all future publication benchmark analyses and QA use a fixed
elliptical inner radius of 2 pixels, independent of PSF FWHM. Keep the physical
PSF metadata, fitting recipes and saved profiles unchanged. Preserve historical
analyses. Store the evaluated cut explicitly, and reject ambiguous historical
measurements in the new QA path rather than silently relabelling old scores.

Reuse shared residual evaluation defaults for future campaigns and update
Huang2013 pixel, ring and centroid diagnostics consistently. Native metrics
copied from old fit records remain historical and must not be relabelled.
New results go to new folders; small actual-output gate precedes full analysis.

Keep the existing no-harmonic common-renderer baseline for an isolated cut
comparison. Audit harmonic storage, normalization, phase/basis and supported
orders before adding a separately labelled harmonic reconstruction diagnostic.
Do not equate turning a flag on with equivalent harmonic support across codes.
Compare a second common reconstruction and direct truth-on-ring diagnostics
on small examples before any broader interpretation. No refitting is required.
