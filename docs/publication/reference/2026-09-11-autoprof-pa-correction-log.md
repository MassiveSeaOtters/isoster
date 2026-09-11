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
