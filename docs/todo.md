# Next: comprehensive Huang2013 updated analysis

- [x] Journal accepted rendering, metrics and QA rules; prepare comprehensive handover.
- [x] Follow `docs/specs/2026-09-23-huang-final-analysis.md`: validate all-arm/mode
  coverage and small actual-output gate before the full statistics/QA run.
- [x] Preserve baseline, failed outcomes, incompatible harmonic bases and old data;
  audit final products before updating the manuscript.

Handover: `docs/agent/journal/2026-09-23-handover.md`. The user has accepted the
rules and subsequently authorized the comprehensive analysis; the audited run is complete.

# Two-pixel benchmark evaluation — 2026-09-23

## Cross-tool QA follow-up

- [x] Add four-zone tables, small panel gap, full residual display and data-only SB limits.
- [x] Verify harmonic tables use their own zone metrics; test and inspect regenerated demos.

Review: 662 unit tests passed; Ruff and whitespace checks passed. Regenerated
three harmonic cases (15 figure pairs) and two standard cases including a failed
primary (eight pairs), preserving old outputs. All harmonic metric/selection and
baseline-check tables are exactly unchanged; 115 and 70 source hashes verified
unchanged in the respective exports. Visually inspected the small gate and failed
primary. Outputs: `outputs/huang2013_harmonic_qa_zones_20260923` and
`outputs/huang2013_individual_qa_zones_20260923`. No fitting, merge or push.

- [x] Record permanent 2-pixel evaluation rule; preserve historical data.
- [x] Update shared evaluation defaults, Huang2013 analysis, QA, docs and tests.
- [x] Validate on a small subset before full Huang2013 reanalysis.
- [x] Regenerate scientific summaries and approved QA with the new cut.
- [x] Audit harmonic conventions and test reconstruction sensitivity without refitting.
- [x] Record validation, results, limitations and remaining work.

## Review

- All 14,229 outcomes retained; 837 exact truth/noise checks; 30,862 source
  hashes unchanged and matching the historical analysis. No refits.
- Full-aperture primary RMS wins: Isoster 402/794, Photutils 184/794,
  AutoProf 208/794. AutoProf has smallest absolute flux bias on 630/794.
- Six summary figure pairs, eight selected-case atlas pairs, and eight
  individual QA pairs regenerated with the approved design and 2-pixel cut.
- 658 tests passed (657 unit and one actual campaign test); no merge/push.
- Calibrated raw polar n=3,4 reconstruction diagnostic completed 48 models
  on two galaxies/two scenarios. Ranking depends on renderer. EA harmonic
  reconstruction and broader pixel-integration checks remain follow-up work,
  not a completed harmonic-inclusive population ranking.
- Details: `docs/publication/reference/2026-09-23-two-pixel-analysis.md`.

# Individual QA demo — 2026-09-22

- [x] Apply final layout requests, verify metric color direction, render and inspect new examples.

- [x] Revise demo: panel-width cross-tool table; cross-arm metric matrix and residual thumbnails.

- [x] Inspect existing comparison plots and agree on demo scope.
- [x] Implement opt-in cross-tool and cross-arm examples without changing defaults.
- [x] Verify metrics, input hashes, failure handling, and figure layout.
- [x] Prepare ordinary and failed-primary examples for user review.
- [x] User approved the figure design; use it for updated QA.

## Demo review

- New branch: `feature/huang-qa-demo-20260922`, outside Dropbox.
- Latest review demo: `outputs/huang2013_qa_redesign_demo_20260922_v7`.
- Two inputs, eight PNG/PDF pairs with caption text files, two statistics CSVs and a hash/metric audit.
- 29 successful-arm measurements reproduced; 70 source hashes unchanged.
- 655 unit tests passed (10 existing warnings); Ruff and whitespace checks passed.
- Visually inspected the revised cross-tool, ten-arm Isoster, and failed-arm AutoProf examples.
- Metric columns have independent scales, red for better and blue for worse; bias color uses absolute magnitude but printed values retain their signs. Constant columns remain neutral. Residual thumbnails share a scale across all cross-arm pages for each input.
- No fits, old results, plotting defaults, or other repositories changed.
- Missing native curves of growth are labelled unavailable, not drawn as zero.

# AutoProf zero-background execution

- [x] Verify the remote base and isolate work from Dropbox.
- [x] Record the approved scope and immutable data boundary.
- [x] Forward and test fixed background; reuse audited reference profiles.
- [x] Pass the small actual-fitting gate and inspect its results (12/12 successful; three before/after QA figures inspected; 44 regression tests passed).
- [x] Complete and audit the Huang2013 four-arm afterburner (3,297 successes, 51 retained errors; 14,230 source hashes unchanged).
- [x] Repeat full scientific measurements, figures and interpretation.
- [x] Test, commit and push the feature branch; do not merge.

## Completion review

- Fitting: 3,348 requested, 3,297 successful, 51 retained errors; no source
  files changed. Only AutoProf was refitted, with known zero background.
- Analysis: 14,229 outcomes, 372 truth images, 837 exact input/seed checks,
  and 30,862 independently rechecked source hashes. All 10,881 non-AutoProf
  selected source paths/statuses remain unchanged.
- Products: six summary figures and ten selected-case QA pages (PNG/PDF),
  with final atlas residual maps restricted to the numerical common aperture.
  Use `huang2013_scientific_products_zero_background_2026_09_21_v2`.
- Validation: 651 unit tests passed; 47 focused tests and Ruff passed.
  Figure script/input hashes, all numerical tables and 25 displayed-profile
  metrics were verified; the six summary PNGs are identical between exports.
- Documentation: updated detailed reference and manuscript; retained the
  earlier estimated-background report as historical. Recorded retry-audit,
  empty-ring, stochastic-optimizer and residual-mask lessons.
- No old campaign was deleted, no other repository was edited, and no merge
  or S4G run was performed. Further AutoProf boundary/initialization changes
  require a separate decision, not an unrecorded repair of this campaign.
# Harmonic reconstruction demo — 2026-09-23

- [x] Select difficult cases from measured primary baseline RMS.
- [x] Reuse calibrated harmonic conversion, native models and approved QA.
- [x] Validate synthetic harmonics and a single-case output gate.
- [x] Complete three-case demonstration, inspect figures, record findings.

## Harmonic demo review

- 36 models, 144 alternative metric rows, 36 reproduced baseline zone rows,
  15 figure pairs and 115 unchanged source hashes. 661 unit tests passed.
- Native Isoster harmonics improve full RMS in these three cases but worsen
  outer RMS; AutoProf retains full RMS advantage. Renderer effects are larger
  than matched-spline harmonic effects. No population inference or default change.
- Final output: `outputs/huang2013_harmonic_demo_20260923_v2`; detailed results
  in `docs/publication/reference/2026-09-23-harmonic-demonstration.md`.
- The user subsequently accepted the rendering/metric/QA rules and requested a
  handover for the full sample; see the next-analysis section above. No new fits,
  merge or push.

# Comprehensive execution — 2026-09-23

- [x] Read handover/spec/lessons; create feature/huang-final-analysis-20260923 carrying inherited work.
- [x] Verify mounted inputs and storage (6.1 TiB external; 60 GiB local at start).
- [x] Implement retained all-arm/four-mode measurements, provenance and resumable QA.
- [x] Validate synthetic and real EA, missing coefficients, failures and empty zones; inspect actual outputs.
- [x] Run full population, audit counts/hashes/products and publish descriptive results.

Review completed below. No refitting, merge or push was performed.

Historical execution update (superseded by the completion review): 664 unit tests passed in 18.43 s. Four-input actual-output gate
produced 40 PNG/PDF QA pairs, 272 retained arm/mode outcomes and 244 reproduced
baseline zone measurements. Per-input gate times: 14.57–18.00 s; peak worker RSS
up to 1,106,100,224 bytes. Gate summary verified 168 current source hashes and
1,324 historical source hashes. Explicit real EA tests passed for geom_ea,
geom_simul_ea and harm_simul_ea; shared polar adapter rejected them as required.
Full 837-input execution started with eight process workers, one numerical-library
thread each, into the new campaign analysis folder
`huang2013_final_reconstruction_2026_09_23`. Final audit/review pending.

QA correction during execution: the synthetic missing-coefficient output gate
found that valid 1-D profiles were removed when a reconstruction was unavailable.
Stopped the first population attempt without deleting its files. The shared QA
helper now retains these profiles and labels missing models separately. A regression
check confirms the Isoster curve remains with its model absent; 41 focused tests
passed. Revised output gate and fresh population execution follow.

## Comprehensive execution review — complete

- [x] Retain the accepted all-arm/four-mode population and validate synthetic/real EA.
- [x] Pass actual-output gates, including missing coefficients in a separate copy.
- [x] Complete the corrected 837-input run with 8,370 PNG/PDF QA page pairs.
- [x] Audit 56,916 outcomes, 227,664 metric rows and 53,132 original baseline zone checks.
- [x] Verify 36,153 current and 30,862 historical source hashes; preserve old outputs.
- [x] Independently reproduce 1,282 nonempty alternative measurements across six cases.
- [x] Export descriptive, paired, scenario, galaxy-consistency, coverage and runtime tables.
- [x] Update detailed reference/manuscript and provide a browsable delivery index.

Final QA: campaign analysis `huang2013_final_reconstruction_2026_09_23_v2`.
Statistics/index: `outputs/huang2013_final_statistics_20260923`.
Reference: `docs/publication/reference/2026-09-23-huang-final-analysis.md`.
665 unit tests passed; final source hashes frozen; no refit, merge, push or commit.
The first bulk attempt and every gate remain preserved. Shared polar EA-on remains
unsupported by design; native EA is validated and included. Rankings depend on
renderer and zone; native Isoster full-RMS improvement does not imply outer-zone
improvement, and native AutoProf is not intensity-harmonic-on.

## Branch closure and next investigation — 2026-09-23

- [x] Commit all inherited/final analysis changes as `4622ace`.
- [x] Merge 27 accumulated commits into main with merge commit `97e9f8c` and push.
- [x] Delete merged `feature/huang-final-analysis-20260923`; preserve other branches.
- [x] Recheck 665 unit tests (18.67 s, 10 warnings), targeted Ruff and whitespace.
- [x] Append the development snapshot to Obsidian before journal/handover.
- [x] Create `analysis/autoprof-comparison-20260923` for the next investigation.
- [ ] Execute `docs/specs/2026-09-23-autoprof-comparison.md`: center, radial extent,
      algorithm differences, rendering/flux bias and measured improvement options.
- [ ] Reassess S4G readiness after explaining Huang2013 differences.

Review: main's merge tree exactly matches the tested `4622ace` tree. No historical
results were changed and no fits rerun. Earlier no-merge/no-push entries describe
the historical analysis run; the user explicitly authorized this later closure.
