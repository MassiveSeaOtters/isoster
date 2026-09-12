# Huang2013 scientific analysis

Approved 2026-09-10; execution branch `analysis/huang2013-publication`.

Status: full corrected scientific measurements completed 2026-09-12 on
`analysis/huang2013-corrected-science`: 14,229 records, 837 exact input
reproductions, 372 truth images and 30,913 unchanged source files. Summary
figures, selected cases and interpretation are recorded in
`docs/publication/reference/2026-09-12-huang2013-scientific-analysis.md`.
Corrected AutoProf fitting and both campaign audits completed 2026-09-11;
the corrected two-galaxy gate (`_gate_v2`) passed exact pixel/seed checks.
The gate exposed an AutoProf initial-position-angle convention error in the
campaign wrapper. See
`docs/publication/reference/2026-09-10-huang2013-analysis-gate.md`.
Pass the explicitly audited `--autoprof-campaign` to select its entire
corrected roster, including failures. Original AutoProf records remain
excluded with a supersession reason; reference dependencies are not imported
as additional scientific Isoster outcomes.

## Inputs and products

Join the original campaign, its three accepted recovery directories, and the
Isoster afterburner by galaxy, scenario, tool, and arm. Keep provenance and
explicit exclusions; expected logical counts are 14,229 records, 13,334
successful profiles, 58 failures, and 837 OLS no-ops. Recovery dependencies
must not duplicate primary outcomes. Original files are read-only.

Produce a manifest, coverage report, fit metrics, paired Isoster differences,
scenario summaries, structural trends, selected-case index, six summary
figures, a case atlas, a detailed reference, and manuscript prose. Generated
products live in a new dated directory under the publication campaign's
`analysis/`; scripts and reference prose are tracked in Isoster.

## Measurement contract

- Primary arms: Isoster `ref_default`, photutils `baseline_median`, AutoProf
  `baseline`. Retain all diagnostic arms separately.
- Use the exact shared input center for centroid errors. Component ellipticity
  and PA are descriptors, not unique ellipse truth for multi-component images.
- Regenerate four PSF-convolved, noiseless images per galaxy using frozen
  MockGal commit `a6a90a0`, the original recipe, and libprofit. Verify z005
  against retained noiseless pixels and noisy images against their recorded
  seeds before using the truth cache.
- Common reconstruction with `build_isoster_model(..., use_harmonics=False)`
  measures elliptical-profile fidelity for every tool. It is a new diagnostic
  and does not replace retained native-model residuals. Compare on the common
  finite pixel support of successful arms, anchored at initial geometry, with
  a PSF-FWHM inner floor; report support and missing zones explicitly.
- Use inner (<0.5 R_ref), middle (0.5--2 R_ref), and outer (>=2 R_ref) zones.
  R_ref is the adapter's recorded reference scale, not automatically a true
  total half-light radius. Profile radial zones and fixed image zones need not
  contain identical pixels when fitted geometry changes.
- Report truth residual RMS divided by truth RMS, data residual RMS divided
  by injected Gaussian sigma (noiseless N/A), and aperture flux bias on common
  support. Never call this extrapolated total-flux recovery.
- Evaluate truth ring mean or median on each fitted ellipse, retaining the
  angular basis. This measures extraction fidelity conditional on geometry;
  different tools' native statistics need not be the same estimand.
  Dense bilinear rings do not reproduce sigma clipping, variable-width
  AutoProf annuli, or the integrator switch in the LSB auto-lock arm; their
  residuals are diagnostics, not exact tool-native extraction errors.
- Keep finite-profile coverage and stop=0 coverage separate. AutoProf's
  placeholder stop codes cannot support cross-tool convergence claims.
- Reuse centroid/smoothness functions, but missing values and unavailable zones
  remain N/A. Quality flags do not define publication pass/fail or a global
  score. Smoothness and harmonic amplitudes are descriptive.
- AutoProf native harmonics remain separate: its retained free-fit schema marks
  Bender conversion invalid. Do not infer a valid cross-tool conversion from
  the fixed-aperture harmonic experiment. Compare EA/non-EA harmonic amplitudes
  as basis-dependent quantities; isolate simultaneous harmonics within EA.
  Centered symmetric components need not sum to perfectly elliptical
  isophotes: even-order harmonics can be genuine when component shapes/PAs
  differ. Do not assume every nonzero coefficient is numerical error.
- Paired Isoster contrasts use identical scenarios and finite values on both
  sides. Report median, p16/p84, improved/equal/degraded fractions and sample
  counts. Pooled confidence intervals resample galaxies with all scenarios
  together, fixed random seed. These exploratory samples are not held-out tests.
- Production runtimes are descriptive; controlled Stage 4 remains the timing
  reference. No ranking based on a combined time/accuracy score.

## Plotting and gates

Reuse `configure_qa_plot_style`, `METHOD_STYLES`, the existing comparison QA
figure, fourth-root SMA axes, stop-code markers, and separate calibration
inputs. Use PDF plus PNG, legible labels and sample counts. Inspect rendered
figures before accepting. Select cases by recorded metric rules, including
typical, elongated, compact, failed, discrepant, and EA benefit/degradation.

Validate a small image-writing analysis in under one minute before scaling.
Audit source hashes before/after, exact logical keys, finite metrics, zone
coverage, paired joins and exports. Record any unavailable product explicitly.
