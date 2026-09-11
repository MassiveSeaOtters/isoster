# AutoProf PA-correction campaign log

## Frozen scope

User approved 2026-09-11. Correct only the exhausted-campaign input-PA
conversion using the existing helper; keep all four arms, images, clipping,
background estimation, timeouts and output conventions unchanged. The input
option is astronomical PA, not mathematical PA. The independent Stage 4
timing worker is already correct and remains unchanged.

Isoster branch: `fix/autoprof-publication-pa`. Tests cover actual wrapper
options across wrapped/negative angles, unchanged non-PA settings, native
output conversion, phase isolation and campaign counts. Forty focused tests
passed in 1.47 seconds before actual fitting.

`isophote_test` documentation records the downstream nature of the bug,
input/output conventions, image-overlay checks, truth-engine verification,
and preservation policy. Documentation commit `6925e91` was merged through
the approved feature-branch workflow; its unrelated untracked files were
left untouched. No mock-generation code changed.

## Execution contract

Configurations: `benchmarks/exhausted/configs/campaign.publication_autoprof_pa_{gate,huang,s4g}_2026_09_11.yaml`.
Driver: `benchmarks/exhausted/campaigns/run_autoprof_pa_correction.py`.

All output lives in new `publication_autoprof_pa_*_2026_09_11` directories
under `/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/fits`.
Existing output directories are refused. The runner hashes consumed old
images, manifests, references, profiles, run records and options; writes both
environment inventories; runs reference dependencies; verifies centers to
1e-8 pixel; then runs AutoProf and audits saved PA, fixed centers, finite
profiles and unchanged source bytes. The tolerance is a validation choice,
not an empirical uncertainty estimate.

The gate uses IC2597/wide_z005, NGC1209/noiseless_z005,
NGC2780/wide_z010 and NGC0275/noiseless_z005. It requests four reference
dependencies plus sixteen AutoProf fits, with two workers. Full campaigns
request 837/1,800 references and 3,348/7,200 AutoProf outcomes, with eight
workers and single-thread numerical libraries. Native stochastic behavior
and numerical failures are retained. No old/new best-score selection.

## Gate result

Frozen fitter/driver commit: `d7cace2`. The four-image gate completed in
70.084 seconds with sixteen successful AutoProf fits, zero failures/skips,
four successful reference dependencies and exactly unchanged reference
centers. Every saved PA option passed, every successful profile had finite
rows, and all consumed-source SHA-256 checks remained unchanged.

Four before/after PNG/PDF diagnostics were inspected in
`analysis/autoprof_pa_gate_2026_09_11_unwrapped` (the final reviewed export;
the first export is also preserved). They reuse the shared QA style and
fourth-root radius axes, with all PA values included. The old perpendicular
fits are visible; corrected orientations follow the galaxy light. Shared
finite-aperture, no-harmonic-renderer data residual RMS changed as follows
(intensity per pixel; diagnostic only, not a publication ranking):

| Case | Old AutoProf | Corrected AutoProf |
|---|---:|---:|
| IC2597 wide_z005 | 0.290390 | 0.052162 |
| NGC1209 noiseless_z005 | 1.251859 | 0.043704 |
| NGC0275 noiseless_z005 | 0.585518 | 0.112923 |
| NGC2780 wide_z010 | 0.578501 | 0.057105 |

NGC0275's nearly round outskirts still have method-dependent PA/ellipticity;
this is not the systematic 90-degree input error. The gate preserves those
differences rather than imposing agreement. No swap or thermal/performance
warning was reported immediately after the gate.

External documentation merge `003ed6e` is pushed to `isophote_test/main`.
The corrected gate is accepted for full Huang2013 execution; full scientific
interpretation remains pending production fitting and audit.

## Production start

The complete read-only S4G audit confirmed all 7,200 old saved PA options
are offset by 90 degrees; no options were missing. Huang2013 production uses
frozen fitter/driver commit `24d8930`. All 837 regenerated references
completed successfully before the center-verification stage. Progress logs
are retained under `outputs/benchmark_exhausted/pa_correction_2026_09_11/`
and will be copied into the new external audit folders at completion.

Analysis replacement selection now requires the corrected campaign's
completion audit and a complete unique AutoProf roster. It explicitly
retains corrected failures even when an old outcome succeeded. It does not
import reference-only dependencies as scientific Isoster outcomes.

Three completed production cases were inspected before the full run finished:
IC2597/deep_z050, NGC1209/noiseless_z005 and NGC2986/deep_z020. PNG/PDF
diagnostics and exact residual measurements are retained in
`analysis/autoprof_pa_huang_2026_09_11_production_checks`. The corrected
orientations follow the light and the prominent perpendicular residuals
are removed. Inner-PSF and outer low-signal method differences remain visible.
This is a spot check, not evidence that every production fit is scientifically
accurate. The first diagnostic command named NGC2974, which is absent from
the selected campaign; its input check stopped before creating output. The
replacement case NGC2986 was resolved from the actual completed roster.

## Huang2013 completion

Completed 2026-09-11 at 18:28 local. All 3,348 AutoProf outcomes succeeded
(837 in each arm), with no failures, skipped arms or reused records. All 837
reference dependencies succeeded and their resolved centers matched the
archived references exactly. The smallest successful profile contains 23
finite rows. All 13,396 consumed-source hashes are unchanged, and every
saved PA and fixed-center option passed. Measured reference/fitting/audit
elapsed time: 3,662.95086575 seconds; preliminary inventory time is excluded.
This is production duration, not a controlled timing benchmark.

The completed driver log was copied byte-for-byte into the new campaign's
`correction_audit/driver.log`; the monitor is retained alongside it. No swap
or thermal/performance warning was observed during monitoring. The three
production diagnostics above and complete record audit support proceeding
to S4G without changing settings. S4G was started after this review, with
eight workers, 1,800 reference dependencies and 7,200 AutoProf outcomes.

## Analysis revalidation

The two-galaxy corrected analysis gate finished with 306 records: 285
successful, three retained Photutils failures and eighteen intentional OLS
skips. All 72 AutoProf rows came from the audited replacement. All eighteen
truth/noise reconstructions matched the original float32 pixels exactly.
Inspection found that pandas exported mixed integer/missing seeds as floats,
losing integer precision in the CSV only. Computation used the exact integer
seeds from the frozen generator. Seed export now uses decimal text; the first
gate is preserved and a fresh gate will verify the corrected record. This
small analysis check used two workers during S4G reference preparation;
S4G production durations must not be treated as isolated timing measurements.

The fresh `_gate_v2` export passed all eighteen exact-seed comparisons and
float32 reconstruction checks, retaining the same 306 scientific records.
Its 72 AutoProf rows all select the corrected campaign; 74 excluded rows
account for superseded AutoProf results and the two dependency duplicates
in this selected subset. This is the accepted analysis gate. Isoster commit
`bdc4d04` contains the seed-export correction. S4G completed all 1,800
reference dependencies and verified their centers before starting AutoProf.

At 19:07 local, a process snapshot during S4G fitting showed eight AutoProf
workers plus macOS background work: `CGPDFService` at 85.2% CPU,
`mds_stores` at 68.9%, and `mediaanalysisd` at 68.7% (per-process CPU values,
where 100% is one core). These are observations, not assigned causes of
individual fit durations. No system services were stopped or reconfigured.
The monitor recorded load 16.49 at 19:06:50, with zero swap and no thermal
or performance warning. This further limits interpretation of production
wall times; it does not invalidate the fitted science profiles by itself.

Three S4G production diagnostics were exported and visually inspected at
19:15 local in `analysis/autoprof_pa_s4g_2026_09_11_production_checks`:
NGC0275/noiseless_z005, NGC2780/wide_z010 and NGC1357/deep_z005. All show
the corrected orientation following the light, without the old prominent
perpendicular residual. Common-aperture data residual RMS changed from
0.585808 to 0.117212, 0.575999 to 0.057093 and 0.258165 to 0.020043,
respectively. Exact values and support counts are in the diagnostic JSON.
These are spot checks, not population accuracy claims. Their single-process
plot generation also briefly overlapped S4G fitting.

## S4G completion and final review

S4G completed on 2026-09-11 at 20:25 local. All 7,200 AutoProf outcomes
succeeded (1,800 per arm), with no failures, skipped arms or reused records.
All 1,800 reference dependencies succeeded with exactly matching resolved
centers. The minimum successful profile has 36 finite rows. Every saved PA
and fixed-center option passed; all 28,800 consumed-source hashes are
unchanged. Recorded source revision: `e055a41` (the PA fitter and driver are
unchanged from the Huang run). Measured reference/fitting/audit elapsed time
was 6,849.725027334 seconds, excluding the preliminary inventory. The final
driver log was copied and byte-compared into `correction_audit/driver.log`.

Condition records (one-minute load; samples taken approximately every 30 s):

| Campaign | Samples | Recorded window, local | Load min / median / max |
|---|---:|---|---|
| Huang2013 | 114 | 17:31:43–18:28:39 | 6.266 / 12.144 / 13.517 |
| S4G | 224 | 18:30:48–20:24:55 | 4.013 / 12.872 / 18.761 |

Every recorded sample reported zero swap and no thermal/performance warning.
Huang's monitor starts during AutoProf fitting, not at reference preparation;
S4G's includes reference preparation. These windows, background processes,
brief diagnostic/test activity, and stochastic fitting prevent interpreting
the durations as a new controlled timing comparison. Stage 4 is unchanged.

Final regression check: 48 focused tests passed in 1.72 seconds. Earlier,
33 shared comparison-QA tests passed in 8.78 seconds. All reviewed figures
reuse the established QA style and are retained as PNG/PDF outside Git.

The corrected campaigns are accepted for scientific analysis. Each
`correction_audit/accepted_records.json` is the complete replacement roster;
`completion.json` records audit completion. Do not import regenerated
reference dependencies as replacement Isoster scientific results. The
accepted corrected Huang analysis gate is
`analysis/huang2013_scientific_analysis_pa_corrected_2026_09_11_gate_v2`.
The full 93-galaxy measurements, population figures and scientific narrative
remain the next separate analysis task; this corrective run does not claim
to have completed them. No original data, results or system services changed.

External documentation closeout was merged and pushed to
`isophote_test/main` at `229644f` (documentation commit `1dc8644`). Unrelated
untracked files were preserved. Gate and both analysis-gate logs were also
copied and byte-compared into their respective new output directories.

The complete corrected Huang selection was exported separately to
`analysis/huang2013_pa_corrected_selection_2026_09_11`: 14,229 unique logical
records across 93 galaxies, with 13,334 successes, 58 retained failures and
837 intentional skips. All AutoProf rows select the corrected campaign.
The 3,351 excluded records comprise 3,348 superseded AutoProf outcomes and
three reference-dependency duplicates. `manifest.csv`, `excluded_records.csv`
and `selection_summary.json` are retained there; this selection export does
not perform or claim full-sample scientific measurements.
