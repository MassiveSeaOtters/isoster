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

## Isoster afterburner

### Purpose and frozen contract

Before scientific metric review, an Isoster-only afterburner added the seven
approved diagnostic records: `ols_noweight`, `int_median`, `geom_simul`,
`lsb_autolock`, `geom_ea`, `geom_simul_ea`, and `harm_simul_ea`. The campaign
reused the unchanged 837 Huang2013 and 1,800 S4G publication images. It did not
repeat photutils, AutoProf, or the completed Isoster primary arm.

The configuration, contract test, and corrected eccentric-anomaly algorithm
description were committed and pushed at
`10a920cfd00ee0bb64ec028d4faffdf56a50c7b4` before any afterburner fit. The
primary three-code comparison remains unchanged; these results are diagnostic
Isoster arms to be joined by dataset, galaxy, scenario, tool, and arm during
analysis.

Both mock adapters deliberately provide no variance map. Consequently,
`ref_default` is already an ordinary least-squares fit and `ols_noweight`
correctly records a no-op rather than duplicating it. Counts below therefore
distinguish requested records from executed fits.

### Gate

The fresh gate used Huang2013 ESO221-G026 `noiseless_z005` and S4G NGC2780
`wide_z010`, selected from input truth without looking at fit outcomes. Their
largest component ellipticities are 0.85 and 0.9815, respectively. This
exercised both mock adapters, one noise-free and one noisy image, and strongly
elongated input components.

All 14 requested records were retained: 12 successful profiles, two expected
`ols_noweight` no-op records, and no failures. The output timestamps span about
two seconds, below the required one-minute gate. A zsh audit wrapper attempted
to assign its post-run exit code to the shell-reserved name `status` and exited
after the campaign had completed. This bookkeeping error did not affect the
complete campaign output or raw log; the full-run wrapper used
`campaign_exit` instead.

### Full campaigns

Both campaigns used eight concurrent galaxy workers. Each worker and the
numerical libraries were limited to one thread. `caffeinate -dimsu` prevented
idle sleep, and `PYTHONPYCACHEPREFIX=/tmp/isoster_pycache` avoided Dropbox
placeholder bytecode. Wall times describe parallel campaign execution and are
not controlled per-code timing results.

| Dataset | Requested records | Executed and successful | OLS no-op | Failed | Wall time | Output size |
|---|---:|---:|---:|---:|---:|---:|
| Huang2013 | 5,859 | 5,022 | 837 | 0 | 311 s | 776,864 KiB |
| S4G | 12,600 | 10,800 | 1,800 | 0 | 624 s | 1,667,200 KiB |
| **Total** | **18,459** | **15,822** | **2,637** | **0** | **935 s** | **2,444,064 KiB** |

Every applicable arm succeeded on every scenario. All 18,459 run-record JSON
files were readable, and all 15,822 successful records had a `profile.fits`.
Each campaign's `environment.json` records Git commit `10a920c`.

The runs used a 20-core Apple M1 Ultra Mac Studio with 128 GB memory, macOS
15.7.3 arm64, and CPython 3.12.11. Recorded package versions were Isoster
1.0.0, NumPy 2.3.5, SciPy 1.17.0, Astropy 7.2.0, and photutils 2.3.0. The
Huang2013 load averages changed from 6.16/6.08/6.04 to 10.65/9.51/7.74; S4G
changed from 6.73/8.42/7.52 to 8.06/8.46/7.90. At least 6.4 TiB remained free,
and macOS recorded no thermal or performance warning before or after either
campaign.

### Warnings and quality flags

Huang2013 produced no Python runtime warning. S4G emitted 104
`FIRST_FEW_ISOPHOTE_FAILURE` warnings, but the configured retry recovered every
fit. They correspond to 32 `int_median`, 14 `geom_simul`, 13 `lsb_autolock`,
15 `geom_ea`, 15 `geom_simul_ea`, and 15 `harm_simul_ea` records. S4G also
emitted two divide-by-zero warnings while calculating gradient signal-to-noise;
the interleaved parallel log does not identify their individual records. No
execution failed, but the affected scientific quantities require inspection in
the planned metric review.

Quality flags are retained outcomes, not execution failures. Huang2013 has
1,010 profiles with at least one flag: 154 `int_median`, 180 `geom_simul`, 138
`lsb_autolock`, 175 `geom_ea`, 172 `geom_simul_ea`, and 191 `harm_simul_ea`.
S4G has 4,080: 683, 632, 629, 705, 720, and 711 in the same arm order. These
counts must not be interpreted as an arm ranking before conditioning on mock
scenario and truth.

The symmetric Sérsic components contain no deliberately planted strong
Fourier distortion. Their superposition can still depart from a single ellipse
when component geometries differ. `harm_simul_ea` therefore tests stability,
geometry interaction, and weak or spurious recovered structure here; the
controlled planted-harmonic campaign remains the strong-signal recovery test.

### Storage, provenance, and safety checks

The new results are isolated under:

- `/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/fits/publication_single_band_isoster_afterburner_huang_2026_09_09/`
- `/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/fits/publication_single_band_isoster_afterburner_s4g_2026_09_09/`

Host snapshots and raw logs occupy 2.0 MiB under
`afterburner_audit_2026_09_09/`. The Huang2013 and S4G run-log SHA-256 values
are respectively
`eb147f76a211a8d80098e03e353a29cbf81d33430a6c91ee0e6962bcbc3c5a57` and
`71a975db1e561488ad913bc814ac2de56379cd65be42efb69a212796c2ed9867`.
The cross-arm summary CSV hashes are respectively
`696e9e4884b4dd3e9dc7d079cd6c371fd21e394503033b6e52d88347445a204b` and
`f6cd8a70f289fead309a8d1c9c4354131f662e9faf116b34c8fc325f193061d9`.

No earlier result was written or replaced. After completion, the original
interrupted campaign still contained 8,358 records and reproduced its frozen
combined SHA-256
`3f7358cbaf617f2774e36911e8d180bf9405b52a60fa6395b5ae6eda7125b7af`.
The earlier recovered S4G campaign and monitor logs also retained their
published hashes. The next step is scientific analysis of the combined primary
and afterburner metrics, not another execution recovery.
