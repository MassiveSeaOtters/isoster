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
`analysis/autoprof_pa_gate_2026_09_11`. They reuse the shared QA style and
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
