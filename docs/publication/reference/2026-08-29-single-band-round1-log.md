# Single-band publication validation: round-one execution log

Date: 2026-08-29

## Production mock generation

Both datasets were generated from a detached clean worktree of
`isophote_test` commit `a6a90a07dc3aedd95465928ee2e93258c8ccb40a` with the
arm64 `libprofit` renderer. The original mock datasets were not modified.

| Dataset | Galaxies | Scenarios | FITS images | Noisy seeds | Size | Generation time |
|---|---:|---:|---:|---:|---:|---:|
| Huang2013 | 93 | 9 | 837 | 744 distinct | 765 MiB | 426.7 s generator time |
| S4G | 300 | 6 | 1,800 | 1,200 distinct | 658 MiB | 622.6 s generator time |

Every FITS image was reopened and checked for finite pixels,
`ENGINE=libprofit`, the requested noise state, seed-mode metadata, and the
expected scenario count. Both `run_metadata.json` files record the clean source
commit and `git_dirty=false`.

The generator applied its documented Sérsic-index ceiling of 8 to NGC 1172 in
Huang2013 and to NGC3718, NGC3892, NGC4050, NGC4666, and NGC7742 in S4G. The
original decompositions contain larger indices; the warning is retained as a
mock-model caveat.

## Complete-roster smoke

Four small scenarios exercised all ten retained arms: one noiseless and one
noisy case from each mock dataset. The run requested 40 fits and retained 37
successes and 3 failures:

- photutils `aggressive_clip` returned an empty profile for Huang2013
  ESO221-G026 `noiseless_z005`;
- photutils `baseline_median` failed across its retry ladder for S4G
  ESO012-010 `noiseless_z010`, with the last error
  `cannot convert float NaN to integer`;
- photutils `aggressive_clip` returned an empty profile for S4G ESO012-010
  `wide_z010`.

Every Isoster and AutoProf arm completed. Both primary arms completed on the
two noisy images. The noiseless failures remain separate numerical-stability
outcomes as required by the campaign design; the noisy aggressive-clipping
failure is a retained diagnostic-arm failure, not a reason to replace the
photutils primary configuration.

Smoke evidence is under:

`/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/preflight/publication_full_roster_smoke/`

## Broad-run storage policy

The broad campaign retains each profile, resolved configuration, run record,
inventory, and aggregate table. Residual metrics are still computed from an
in-memory model. Per-arm model FITS and PNG files are disabled because they are
reproducible from the retained source image and profile; selected failures and
representative cases will receive visual QA after the numerical review.

## Status

Broad mock fitting: pending.
