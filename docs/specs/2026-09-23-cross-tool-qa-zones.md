# Cross-tool QA radial metrics and full residual display

Apply the user's four layout requests through the shared publication cross-tool
plotter: small table/panel gap; compact identity/status columns; Truth RMS and
signed Flux Bias in ALL/INNER/MIDDLE/OUTER order; full finite residual maps
without applying the evaluation aperture to display. Keep metric masks and
the two-pixel cut unchanged. Missing model coverage is not extrapolated.

Remove the I=0 reference from the underlying comparison plotter's SB panel and
exclude that reference value from its data-driven limits. Keep other plotters'
zero-reference conventions unchanged. Pass all zone-specific alternative
measurements into harmonic-demo tables, never stale baseline zone values.

Reuse approved figure styles; regenerate harmonic demos and standard examples
in new directories, including a failed primary. Test table ordering, gap,
unmasked display, missing coverage, data limits and preserved metrics.
