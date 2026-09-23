# Huang2013 scientific-analysis gate and AutoProf PA issue

## Status

The approved scientific analysis has started on
`analysis/huang2013-publication`. The two-galaxy gate passed its data and
measurement checks, but exposed a fitting-configuration error. The full
analysis and publication interpretation are paused pending a decision on a
corrective AutoProf run. No existing input image or fitting result was changed.
No new fitting campaign was launched. The wrapper is not yet modified.

## Gate products and validation

New output directory:

```text
/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/huang2013_scientific_analysis_2026_09_10_gate
```

The collector is
`benchmarks/exhausted/analysis/publication_huang.py`. It joins the original
campaign, three accepted recovery directories and the Isoster afterburner.
The full read-only join finds 14,229 logical records: 13,334 successful,
58 failed and 837 intentionally skipped. Three duplicate/dependency outcomes
are explicitly excluded. In particular, the original Isoster reference wins
over recovery-only dependencies; the documented NGC1209 AutoProf fixed-center
followup replaces its earlier dependency skip. No quality-based winner
selection is performed.

The gate includes IC2597 and NGC1209, all nine scenarios and all 17 requested
arms: 306 logical records, with 285 successes, three photutils failures and
18 OLS no-ops. Per-galaxy measurement times with two workers were 7.7871 and
6.5422 seconds respectively; these are analysis times, not fitting timings.

Eight newly rendered, noiseless truth images cover four redshifts per galaxy.
All 18 stored input images are reproduced exactly after float32 conversion,
including the recorded noise draws. Every selected input image, manifest,
run record and successful profile has a before/after SHA-256 check; all were
unchanged. The gate retains `manifest.csv`, `excluded_records.csv`,
`fit_metrics.csv`, `coverage.csv`, `scenario_summary.csv`, `paired_deltas.csv`,
`paired_summary.csv`, provenance, per-galaxy truth verification, and source
hashes. These are validation products, not final scientific conclusions.

The frozen generator is MockGal commit
`a6a90a07dc3aedd95465928ee2e93258c8ccb40a`, accessed in a detached temporary
worktree. The existing profit-cli binary refers to an old machine's dynamic
library path. Setting `DYLD_LIBRARY_PATH` to that binary's directory restores
libprofit without editing either the executable or its source repository.
The analysis rejects Astropy fallback and rejects any pixel reproduction
mismatch. The generator also reports its existing NGC1172 Sersic-index clamp
from 11.38 to 8.0; this is part of the frozen mock recipe, not an analysis change.

The truth-ring diagnostic uses 1,024 angular samples. For IC2597 wide_z005,
increasing to 2,048 changes the relative-RMS diagnostic from 0.000223073 to
0.000222908 for `ref_default`, from 0.000272433 to 0.000272452 for `geom_ea`,
and from 0.000274906 to 0.000274930 for `harm_simul_ea`. This checks angular
sampling only; it does not validate all tool-specific interpolation or annuli.

## Confirmed configuration error

`benchmarks/exhausted/fitters/autoprof_fitter.py::_build_options` passes
`degrees(initial_geometry['pa'])` to `ap_isoinit_pa_set` without converting
from mathematical PA (counter-clockwise from +x) to astronomical PA
(counter-clockwise from +y).

The installed AutoProf initializer explicitly applies
`PA_shift_convention(option * pi / 180)`, whose definition is
`(pa - pi/2) % pi`. Thus the campaign wrapper rotates the intended starting
ellipse by 90 degrees. This option overrides AutoProf's own global-PA
estimate; it is not merely an unused hint.

The repository already has the correct conversion helper,
`benchmarks.utils.autoprof_adapter.isoster_pa_to_autoprof_init`, returning
`degrees(pa) - 90`. The independent Stage 4 timing worker also subtracts
90 degrees correctly. This specific defect is in the exhausted-campaign
wrapper, not the controlled timing worker.

A read-only audit of the saved options for all **3,348 accepted Huang2013
AutoProf records** confirms the same 90-degree discrepancy in every case;
none of the saved option files is missing. S4G uses the same wrapper, so it
also requires review before publication; its individual options have not yet
been exhaustively audited in this session.

For NGC1209 wide_z005 the saved AutoProf option is -5.197957 degrees, equal
to the adapter's mathematical PA. The correct option is -95.197957 degrees
(equivalently 84.802043 modulo 180). AutoProf therefore initializes at
84.802043 degrees in mathematical coordinates rather than the intended
174.802043 degrees. Its retained profile has an approximately perpendicular
outer orientation and a very round inner fit, consistent with this problem.

The profile parser's output conversion is correct: AutoProf writes
astronomical PA and the parser subtracts 90 modulo 180. Rotating saved
profile angles alone cannot repair fitting that already sampled the wrong
geometry. The gate's large AutoProf residuals must not be interpreted as
evidence of inferior intrinsic performance.

The existing comparison-QA function was reused to produce a diagnostic in
`analysis/huang2013_scientific_analysis_2026_09_10_pa_diagnostic`. Visual
inspection revealed that its reference-driven PA limits hide AutoProf's
perpendicular angles and a long two-line title overlaps the panels. That
first `ngc1209-pa-diagnostic` export is retained as an unaccepted draft.
The separate `ngc1209-pa-conventions` PNG/PDF explicitly shows all PAs and
the fitted ellipses on the original image, using the shared serif style,
tool colors and distinct markers. It is a configuration diagnostic, not
one of the final publication figures. The general QA helper has not been
changed as part of this checkpoint.

Six focused unit tests passed; Ruff accepted the new collector and tests.

## Proposed next step

1. Reuse the existing input-PA conversion helper in the shared campaign
   wrapper and add a regression test of the mathematical-to-astronomical
   round trip.
2. Run a small, separately named AutoProf gate on the existing IC2597 and
   NGC1209 images, including an elongated noiseless and noisy case. Confirm
   the saved options, recovered geometry and native/standardized profiles.
3. If successful, rerun the four AutoProf arms for Huang2013 and S4G into
   new campaign directories, using the same images. Preserve all original
   results. Keep Isoster and photutils results; this finding does not require
   repeating them. Fixed-center AutoProf needs its documented reference-center
   dependency handled explicitly in the new campaign.
4. Update the accepted source manifest and then resume the full Huang2013
   measurements, structural trends, figures, case atlas and manuscript prose.

This corrective fitting run is additional to the approved analysis execution;
obtain the user's agreement before launching it. The final six scientific
figures, atlas and manuscript conclusions have not been produced.

## Reproduction

The current analysis environment is CPython 3.12.11; pandas is an analysis-only
runtime addition, pinned through `uv run --with pandas==2.3.3`. Core Isoster
dependencies were not changed. Use a new output directory on every invocation.

```bash
UV_PROJECT_ENVIRONMENT=/Users/shuang/.venvs/isoster \
PYTHONPYCACHEPREFIX=/tmp/isoster_pycache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --with pandas==2.3.3 python -m benchmarks.exhausted.analysis.publication_huang \
  --root /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28 \
  --output /Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/analysis/NEW_UNIQUE_NAME \
  --mock-source /tmp/isoster_huang_truth.2BzppK/source \
  --profit-cli /Users/shuang/Dropbox/work/project/otters/isophote_test/libprofit/build/profit-cli \
  --galaxies IC2597 NGC1209 --workers 2
```

The temporary worktree must be recreated at the frozen commit if absent.
Do not launch the full comparison with the uncorrected AutoProf sources.
