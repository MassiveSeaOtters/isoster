# Default Isoster versus AutoProf: algorithm and accuracy investigation

## Objective and authorization

Explain the Huang2013 quality differences before proceeding to S4G, and identify
measured opportunities to improve Isoster while retaining its runtime advantage.
Requested by the user on 2026-09-23. Work on
`analysis/autoprof-comparison-20260923`; architecture authority remains `docs/SPEC.md`.
This checkpoint prepares the investigation; it does not claim it is completed.

## Established evidence

The accepted saved-fit analysis is complete: 93 galaxies, 837 inputs, 17 arms,
56,916 retained arm/mode outcomes, 227,664 zone rows and 8,370 PNG/PDF QA pairs.
See `docs/publication/reference/2026-09-23-huang-final-analysis.md` and
`outputs/huang2013_final_statistics_20260923/index.html`.

Matched primary median full-image Truth RMS (%) is:

| Reconstruction | Isoster | Photutils | AutoProf | Complete triples |
|---|---:|---:|---:|---:|
| Shared linear, harmonics off | 2.996 | 3.265 | 4.798 | 794 |
| Shared spline, harmonics off | 6.641 | 6.702 | 2.861 | 794 |
| Shared spline, raw polar harmonics on | 6.863 | 6.916 | 4.015 | 772 |
| Native | 2.604 | 6.979 | 5.039 | 772 |

Native median absolute flux bias (%) is 1.052 / 2.434 / 0.313, respectively.
Isoster's native signed bias median is positive (~1.0505%), versus AutoProf
~0.0434%. Native Isoster improves full RMS over its baseline in 723/837 cases,
but improves outer RMS in only 160/837. Renderer choice reverses rankings.
These results do not establish universal accuracy parity or a universal winner.

The separate controlled timing reference reports median AutoProf/Isoster
end-to-end ratio 63.01 across 24 primary configurations. It uses a small synthetic
grid, different native stopping/radial grids and Python environments; no arm met
all common accuracy conditions. See `docs/publication/three-way-timing-benchmark-reference.md`.
Do not treat parallel campaign timings as controlled speed measurements.

## Required algorithm comparison

1. **Center treatment.** Trace accepted default-arm configuration through the
   wrapper, installed AutoProf version and saved options/auxiliary files. The
   wrapper passes `ap_guess_center` by default and sets `ap_set_center` only for
   explicit center overrides. Verify whether the actual accepted run estimates
   one global center before ellipse fitting or reoptimizes it later; distinguish
   initialization from a truth-fixed constraint and per-isophote center freedom.
   Compare with actual Isoster defaults, initialization, center updates, bounds,
   freeze behavior and recovery arms. Measure fitted-minus-truth center offsets
   and their association with residuals; do not assume wrapper intent proves behavior.
2. **Radial extent.** Separate the last fitted geometry radius, last extracted
   intensity radius and last reconstructed finite radius. Compare maximum radii,
   radius spacing, signal/noise thresholds, gradients, convergence/failure codes,
   repeated-failure handling, mask/frame coverage, truncation, frozen-geometry
   extraction and extrapolation. AutoProf wrapper sets `ap_truncate_evaluation=True`
   and `ap_extractfull=False`; trace their installed semantics. Quantify the user's
   observed extent difference across accepted paired inputs, including failures.
   Greater extent is not automatically more accurate or more usable.
3. **Fitting and extraction.** Compare objective functions, sequential versus
   coupled geometry updates, regularization, angular sampling, interpolation,
   mean/median estimators, clipping, gradient estimation, stopping and uncertainties.
   Record exact defaults, units, coordinate/PA conventions and tool versions.
4. **Reconstruction.** Diagnose why shared linear versus spline changes rankings.
   Isolate radial sampling/interpolation, pixel integration, center treatment and
   boundary behavior using analytic profiles and saved fits. Compare extracted
   intensity with truth on the same ellipses and cumulative flux to separate
   extraction bias from rendering bias. Preserve each renderer's harmonic-off control.
5. **Outer harmonics.** Inspect coefficients, gradients and noise for outer-zone
   degradation and extreme simultaneous-EA models. Native AutoProf ellipse models
   are not intensity-harmonic-on; do not equate measured a3/b3/a4/b4 with their use.

## Sequence and acceptance

- [ ] Audit source selection, exact tool versions, default arms and saved provenance.
- [ ] Write a source-linked algorithm comparison table; mark unknowns explicitly.
- [ ] Measure center offsets, three radial extents, signed flux bias and radial RMS
      from saved fits, paired on common support and stratified by scenario/galaxy.
- [ ] Select a small recorded set of ordinary and difficult cases; inspect actual
      outputs and run a sub-minute diagnostic gate before wider execution.
- [ ] Test isolated hypotheses with analytic inputs and controlled extraction/rendering
      checks. Separate hypotheses from demonstrated causes; do not refit this campaign.
- [ ] Propose the smallest evidence-backed Isoster improvements with accuracy/runtime
      acceptance criteria. Update the spec before implementation; preserve old defaults
      as controls and discuss any new fitting experiment before running it.
- [ ] Report benefits, regressions, unsupported/failed cases and remaining uncertainty;
      make an explicit readiness decision before S4G.

## Persistent constraints

Never delete, replace or modify existing data/results under `/Volumes/galaxy`.
Every new artifact goes into a separate named folder. Preserve the fixed 2-pixel
elliptical inner cut, common-support accounting, harmonic-off baseline, all failures,
and real EA validation. Do not clip unfavorable coefficients, silently repair fits,
change evaluation apertures to favor a tool, or substitute a failed primary arm.
No new refit, default change, merge or push is implied for the future investigation.
The current wrap-up merge/push is explicitly authorized and complete.
