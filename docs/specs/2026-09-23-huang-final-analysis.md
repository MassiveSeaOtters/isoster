# Comprehensive Huang2013 updated scientific analysis

## Objective and scope

Produce the final descriptive performance statistics and individual QA from the
accepted Huang2013 saved fits, using the rendering, metric and figure decisions
approved on 2026-09-23. This is an analysis/reconstruction run, not a new fitting
campaign. The user authorized execution after the handover; the full run and audit are complete.
Detailed context: `docs/agent/journal/2026-09-23-handover.md`.

Reuse the audited two-pixel baseline manifest: 93 galaxies, nine conditions,
837 inputs, 17 retained arms and 14,229 outcomes. Retain accepted recovery joins,
AutoProf PA correction and zero-background selection. Keep every failure/skip.

## Required comparisons

1. Preserve the existing all-arm shared-linear harmonic-off baseline and reproduce
   its counts/measurements before extending it. Keep original and newly matched
   apertures distinctly labelled.
2. Retain the demonstrated shared-spline harmonic-off/on controls for all compatible
   polar profiles, using exact calibrated raw n=3,4 conversion. Record unsupported
   basis/missing-coefficient outcomes; never substitute zero or rotate EA as polar.
3. Include native harmonic-on reconstructions for Isoster/Photutils where supported;
   evaluate saved native AutoProf ellipse models separately and explicitly label
   their absence of measured intensity harmonics. Verify native-file provenance.
4. Cover all retained arms, including EA afterburner arms, in the baseline and
   supported native analyses. Validate actual EA profiles before broad execution.
   Any shared EA implementation requires a separate synthetic basis test; it must
   not be silently assumed equivalent to the demonstrated polar adapter.
5. Do not change fitting defaults, repair fits, add Fourier-shape fitting arms, or
   modify the reconstruction algorithm to improve scores during this analysis.

## Metrics and statistics

- Fixed 2-pixel elliptical inner evaluation cut, unchanged physical PSF metadata.
- Scientific zones: inner <0.5 R_ref, middle 0.5–2 R_ref, outer >=2 R_ref;
  intersect the recorded outer limit and documented common finite support.
- Truth RMS, signed/absolute flux bias and noise-normalized data residual RMS,
  with pixel counts, support fractions and reasons for unavailable values.
- Matched primary-tool comparisons, win/tie counts with explicit denominators,
  per-scenario summaries, all-arm contrasts and consistency across galaxies.
- Use galaxy-level uncertainty/resampling when reporting uncertainty across
  repeated conditions. Deliberately selected diagnostic cases are not population evidence.
- Retain runtime summaries with the parallel-campaign caveat; renderer times and
  native model-load times are not replacement controlled timing benchmarks.

## Figures and records

- Reuse approved cross-tool/cross-arm QA functions and styles.
- Cross-tool table shows ALL/INNER/MIDDLE/OUTER metrics from the displayed mode,
  compact tool/status columns, and small table/panel gap.
- Cross-tool residual maps show full finite coverage without evaluation masks;
  metrics retain cuts. No extrapolation to fill unsupported pixels. Keep shared
  residual color scales and state their aperture-based calibration in captions.
- Small Inner/Outer ellipse legend; no red mask overlay; SB limits from data
  alone, no I=0 reference. Preserve native 1-D profiles and uncertainties.
- Freeze a page-coverage manifest before bulk rendering: recommended per-input
  primary comparisons across supported modes plus per-tool cross-arm baseline
  and native comparisons. Include explicit failed/unavailable outcomes.
- Produce a browsable QA index and machine-readable product inventory. No silent
  incomplete roster. Measure storage before deciding whether all model FITS must
  be retained; figures/metrics/provenance must remain reproducible from source fits.
- Deliver full measurement/coverage tables, final summaries and figures, detailed
  process/results/caveats reference, and updated manuscript summary.

## Execution sequence and acceptance

- [x] Verify isolated checkout, inherited changes, mounted source data and hashes.
- [x] Extend existing helpers into a population runner; the discrepancy-selected
      harmonic_demo --count CLI is not a full-population runner.
- [x] Save precise roster, modes, common-support policy and product coverage.
- [x] Run synthetic checks and sub-minute actual-output gates including ordinary,
      difficult, failed-primary, empty-inner-zone and actual EA-arm cases.
- [x] Inspect table values, maps, axes, figures and measured resource use.
- [x] Run into fresh named campaign/analysis and outputs directories, with
      per-galaxy progress, restart-safe completion records and before/after hashes.
- [x] Compare final roster/statuses against the accepted baseline; audit every
      newly unavailable model and all metric denominators.
- [x] Verify QA inventory and numerical values; spot-check each failure category
      and each renderer/basis. Summarize results without cherry-picking.
- [x] Update detailed reference, manuscript, checklist and journal; report any
      unresolved gaps rather than declaring a partial run final.

Safety: never modify/delete existing `/Volumes/galaxy` data or old outputs.
Use uv and fresh feature branches; numerical library threads=1 per process,
worker count based on measured gate behavior. S4G, new fitting, release, merge
and push are not authorized by this handover.

## Frozen execution policy (2026-09-23)

Four reconstruction modes per retained arm: shared linear off, shared spline off,
shared spline raw-polar n=3,4 on, and native (Isoster/Photutils n=3,4 on;
AutoProf verified saved ellipse model without intensity harmonics). Shared spline
off remains available for EA; its on counterpart is explicitly unsupported.
Native coefficients must be finite on positive-radius rows; unavailable coefficients
are never silently zero-filled. Record available orders and basis per outcome.

Original baseline measurements use the original all-successful-arm intersection.
Alternative comparisons use one intersection across every available arm/mode and
the original aperture, recorded separately; unavailable outcomes do not contribute
an invented model. Empty zones remain unavailable. Matched comparisons require
all designated participants and finite measurements, with no primary substitution.

Per input, freeze four primary cross-tool pages (one per mode) and six cross-arm
pages (three tools, baseline/native). PNG/PDF and captions; explicit unavailable
pages if no models/common pixels exist. Metrics/profiles are exported alongside
pages. Model arrays are reconstructed from hashed fits and retained in memory only;
the actual-output gate measures storage before bulk execution. Each input directory
has completion hashes so interrupted work can resume without rewriting completed
products. Process workers use one numerical-library thread each.

## Completion review

Completed 837 inputs, 56,916 retained arm/mode outcomes, 227,664 zone rows and
8,370 PNG/PDF page pairs. Verified 53,132 original baseline zone measurements,
36,153 current source hashes and 30,862 historical source hashes; 1,282 independent
measurement checks passed. All 665 unit tests passed. Shared-EA harmonic-on remains
explicitly unsupported, with real native EA validated. No fits, merge or push.
Final record: `docs/publication/reference/2026-09-23-huang-final-analysis.md`.
