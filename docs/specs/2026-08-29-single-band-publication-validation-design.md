# Design: single-band publication validation

Date: 2026-08-29
Branch: `review/mock-real-galaxy-validation`
Status: approved; implementation in progress

## Purpose

Refresh the exploratory mock-galaxy and real-galaxy evidence that will support
the Isoster publication. The comparison tests each code in a supported,
scientifically reasonable configuration. Equal arm counts are not required.
The first round is single-band only; a Legacy Survey multi-band demonstration
is deferred until the single-band workflow is understood.

## Data-safety boundary

The existing external roots are read-only inputs:

- `/Volumes/galaxy/isophote/huang2013`
- `/Volumes/galaxy/isophote/s4g_mock`
- `/Volumes/galaxy/isophote/sga2020`

All generated images, downloaded images, recalculated metrics, fit products,
logs, and summaries for this round go under the new root:

`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28`

No command may delete, replace, or update a file in an existing root. A
preflight must resolve and print every input and output path before a command
that writes data. Existing SGA products may be read for metric recalculation,
but recalculated products go under the new root.

## Repositories and provenance

- Isoster owns the campaign contracts, three-tool fitting, combined analysis,
  and publication record.
- `isophote_test` supplies MockGal, Huang2013 models, and CS4G/S4G models.
- `sga_isoster` supplies Legacy Survey data loading, masks, fit adapters, and
  real-galaxy analysis.

Every retained run records all three Git commits, Python/package versions,
resolved configuration files, selected galaxies, realized noise seeds, input
paths, output paths, start/end times, and host information. Each repository is
modified only on a feature branch.

## Round-one datasets

### Huang2013 and S4G mocks

Regenerate both mock datasets from the current model assets so the Huang2013
images use the corrected overall-radius sizing. Each retained configuration
uses the libprofit renderer and records the realized renderer.

The scenario set contains:

1. a genuinely noise-free image (`noise_enabled: false`);
2. a deterministic HSC-like wide image;
3. a deterministic HSC-like deep image.

Noisy images use stable, distinct seeds derived from the campaign seed,
galaxy identity, and scenario identity. The realized seed is written to the
FITS metadata and run metadata. A noise-free fit failure is a separate
numerical-stability outcome, not evidence that the noisy fit failed. If a tool
requires finite uncertainties, the image remains noise-free and the supplied
weight convention is recorded explicitly.

Validation starts with two galaxies per dataset and must finish in less than a
minute before broader generation is allowed.

### Existing SGA sample

First recalculate the existing 1,999-galaxy campaign metrics without refitting.
Correct the SGA reference radius so SGA `MAJORAXIS` is treated as the existing
semi-major-axis radius, not divided by two again. Residuals normalized by an
observational noise scale must use the inverse-variance image, not the
residual image's own scatter. Preserve the older values as historical inputs
and write corrected inventories and summaries only under the new root.

A small fresh Legacy Survey r-band campaign follows the recalculation. New
downloads, if required, go under the new root and retain image, inverse
variance, PSF, maskbits, and mask provenance. DR9 SGA products and DR10 viewer
cutouts must not be mixed without an explicit dataset label.

## Tool configurations

The primary comparison uses one recommended/default configuration per tool:

- Isoster: `ref_default`;
- photutils: `baseline_median`;
- AutoProf: `baseline`.

The first-round diagnostic roster may add tool-native robust/deep settings and
fixed-center controls. Fixed-center results are labeled diagnostic and are not
pooled with free-center primary results. Additional Isoster arms are allowed
because the later SGA study focuses on Isoster; arm-count equality is not a
goal. The larger SGA run will retain only configurations shown to be reliable
and useful in round one.

## Parallel execution

Parallelism is across independent galaxies or disjoint campaign batches.
Tools and arms remain serial within one galaxy. Simultaneous SGA sessions must
have non-overlapping galaxy lists and distinct campaign directories.

Before a broad run, measure one, two, four, and eight simultaneous sessions on
the same small representative workload. Select the setting with the best
measured throughput that does not increase failures, memory pressure, or
thermal warnings. Numerical-library thread counts are limited to one per
worker. Parallel-run wall times are operational metadata, not controlled
single-code performance evidence; the Stage 4 campaign remains the timing
reference.

## Stepwise gates

1. Freeze this design and the task ledger.
2. Correct and test the SGA radius and residual-noise contracts.
3. Add deterministic-seed provenance and noise-free mock manifests.
4. Generate and inspect two galaxies per mock dataset.
5. Fit a tiny three-tool matrix and measure safe session parallelism.
6. Run the retained mock grids.
7. Recalculate existing SGA metrics, then run a small fresh r-band sample.
8. Record failures, lessons, recommended Isoster arms, and publication
   caveats before planning the larger SGA run.

At each gate, retain the failed records and diagnose systematic patterns before
scaling. A failed arm does not stop unrelated arms unless the failure indicates
data corruption or an invalid shared contract.
