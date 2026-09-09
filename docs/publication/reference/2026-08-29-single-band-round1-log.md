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

The broad run started on 2026-08-29 and was interrupted by a host reboot. It
retained 8,358 of 8,370 Huang2013 run records: 8,302 successful profiles and
56 photutils failures. No recorded Isoster or AutoProf execution failed. Two
noiseless scenarios stopped partway through, leaving 12 records absent; S4G
had not started. No malformed run records were found during the 2026-09-08
recovery audit.

The partial campaign is preserved unchanged. The missing Huang2013 records
and the full S4G grid are written to separately named recovery campaign
directories before their results are combined during analysis.

The first NGC1209 recovery attempt on 2026-09-08 exposed a photutils 2.3.0
outer-radius loop: the fitter repeatedly returned an empty sample at exactly
204.0236915 pixels without advancing. The attempt was stopped after more than
five measured minutes and preserved. Publication photutils runs now have a
900-second wall-clock limit, chosen above the 596.03-second maximum successful
Huang2013 photutils fit already observed. A timeout is retained as a failed
numerical-stability outcome; the arm configuration is not altered.

The guarded NGC1209 recovery retained that timeout and completed the AutoProf
baseline, deep, and high-regularization arms. AutoProf `fix_center` correctly
skipped because its required Isoster `ref_default` profile was not present in
the new isolated directory. A separate two-fit follow-up regenerates that
reference profile and runs only the missing fixed-center AutoProf arm.

The NGC1209 fixed-center follow-up succeeded for both the regenerated Isoster
reference and AutoProf. In NGC3585, photutils baseline reached the same
non-terminating state and was retained as a timeout; aggressive clipping and
fixed-center completed, as did all four AutoProf arms. The twelve interrupted
Huang2013 executions therefore yielded ten successful profiles and two
photutils timeouts. Together with the partial campaign, all 8,370 planned
Huang2013 outcomes are accounted for: 8,312 successes and 58 photutils
failures, with no Isoster or AutoProf execution failures. The two regenerated
Isoster reference profiles are recovery dependencies and are not counted again
in those totals.

## S4G re-entry gate

After the Huang2013 recovery, a fresh two-galaxy gate ran the primary Isoster,
photutils, and AutoProf arms on S4G `wide_z010` images. All six fits succeeded
in 39.04 seconds with two concurrent galaxy workers. This passed the required
sub-minute gate before the full S4G campaign.

## Recovered S4G campaign

The separate S4G campaign completed on 2026-09-09 after running from
2026-09-08 18:57 to 2026-09-09 01:29 local time. It used eight concurrent
galaxy workers while keeping each fit and each numerical library to one
thread. The input grid contained 300 galaxies, six scenarios per galaxy, and
ten retained arms per scenario, for 18,000 requested fits. No cached result
was reused.

| Tool | Requested | Successful | Failed or errored |
|---|---:|---:|---:|
| Isoster | 5,400 | 5,400 | 0 |
| photutils | 5,400 | 5,086 | 314 |
| AutoProf | 7,200 | 7,199 | 1 |
| **Total** | **18,000** | **17,685** | **315** |

All 18,000 run records are present and valid JSON, and every successful run
has a `profile.fits`. The 314 photutils failures comprise 246 retry-ladder
failures ending in a NaN-to-integer conversion, 39 guarded 900-second
timeouts, and 29 empty `IsophoteList` results. Of all 315 unsuccessful fits,
293 occurred on the deliberately noise-free images; the remaining 22 were
distributed across the deep and wide noisy scenarios. This concentration is
consistent with the numerical-stability caveat anticipated for exactly
noise-free mock images.

The only non-photutils execution error was AutoProf `fix_center` for
NGC4535 `noiseless_z010`. The fit reached profile extraction but left only
three points for its optional ellipse-model spline, which requires more than
three. AutoProf's existing small-image retry produced the same outcome. This
single failure is retained descriptively rather than changing the arm after
examining the result.

The campaign wrote 9.5 GiB to the new directory
`publication_single_band_s4g_2026_09_08`; it did not write into any earlier
campaign. `/Volumes/galaxy` remained mounted with at least 6.2 TiB free, and
macOS recorded no thermal or performance warning. Load average was monitored
but, as pre-registered for this Mac Studio, was not used as an in-campaign
abort signal because it includes the benchmark's own work. The minute monitor
ended only after the campaign process exited.

The original interrupted campaign remains unchanged after both recoveries:
its 8,358 run records still have combined SHA-256
`3f7358cbaf617f2774e36911e8d180bf9405b52a60fa6395b5ae6eda7125b7af`.
The S4G campaign log and preserved monitor log are stored under
`recovery_audit_2026_09_08/`; their SHA-256 values are respectively
`e6479defd2241952c38ec89f50d087b8b6fc0036a1a338351c84b3fd486dddd9`
and `141dfc8290ab6ded4543bc2928c18bab3d48133941b6b201688b1f9c2f2433a4`.

Execution of both publication mock grids is now complete. The next step is to
review the retained scientific metrics and selected images; no broad rerun is
needed merely to recover execution coverage.
