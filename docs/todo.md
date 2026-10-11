# Isoster vs AutoProf gap review — 2026-10-09

Branch `review/isoster-autoprof-gap-20261009` (worktree `~/code/isoster-gap-review`).

- [x] Catch up on the Codex inner-profile study, isophote_test and sga_isoster.
- [x] Independent code review of Isoster sampling/fitting/rendering vs AutoProf 1.3.4.
- [x] Add opt-in cubic sampling and log-linear renderer; refit 7 recorded + 55 population cases.
- [x] Verify the mock PSF problem with profit-cli; refit 22 PSF-convolved galaxies.
- [x] Report, journal, tests (1274 + 3 passed), local commit; no merge/push.
- [x] Housekeeping: old clone and worktree removed, Dropbox checkout repaired (2026-10-09).
- [x] MockGal fixed (`convolve=1`, `PSFCONV` header) and round-2 PSF-convolved mocks
      generated for Huang2013 (837) and S4G (1800) under
      `/Volumes/galaxy/isophote/publication_single_band_round2_2026_10_10/` (2026-10-10).
- [x] Small-sample round-2 campaign (20 galaxies, 9 scenarios, 13 arms, 3 tools) with the
      truth-based multi-renderer analysis: cubic sampling halves Isoster's RMS and beats
      AutoProf in 86 % of inputs with no failures (2026-10-10).
- [x] Integrator and outer-zone follow-up: white-noise ring test, noise-free refit
      decomposition, radius-switch refits; `sampling_cubic_max_sma` option and arm
      `samp_cubic_outer2re` (`_use_2Re` sentinel now works on any radius knob) (2026-10-10).
- [x] Self-contained kernel switch (`sampling_cubic_max_relative_rms`,
      `sampling_cubic_min_snr`; outward one-way latch, two-ring debounce): scatter rule 0.1
      matches the 2 Re oracle with no size prior; arms `samp_cubic_rms010`,
      `samp_cubic_snr200` (2026-10-10).
- [x] Low S/N thresholds (200 → 3) and the variance-map path for the switch; real-data
      warning in the configuration reference (2026-10-10).
- [x] Round-2 full campaign designed and launched: round-1 roster + small-sample sampling
      arms + three fallback arms (25 arms, `ols_noweight` dropped as a no-op), AutoProf
      `high_regularization` restored; gate on NGC2325 passed (224 ok, 1 known photutils
      noiseless failure); Huang2013 (837 inputs) then S4G chained, detached (2026-10-10).
- [x] Full Huang2013 campaign (837 inputs, 25 arms) run and analysed: cubic halves the
      full-aperture RMS (1.27 → 0.65 %), switch arms 0.61 % and best in every zone, beats
      AutoProf in 80–84 % of inputs, 0 Isoster failures; report
      `publication/reference/2026-10-10-round2-huang2013-full-campaign.md` (2026-10-10).
- [x] Bookkeeping: eleven round-1 (PSF-free) reports moved to
      `publication/reference/round1-no-psf/` with a README on what carries over; links
      repointed; publication README status block rewritten; round-1 manuscript section
      marked superseded (2026-10-11).
- [ ] Decide later whether to keep the round-1 campaign data on the `galaxy` volume
      (`publication_single_band_round1_2026_08_28/`).
- [ ] Edge case, not now: cuspy inner profiles do occur in space-telescope or AO images of
      nearby massive galaxies; cubic sampling on unconvolved cusps needs the inner bilinear
      fallback (`sampling_cubic_min_sma`), see the 2026-10-09 gap review.
- [x] S4G full campaign (300 barred disks, 1800 inputs, 25 arms, 6.5 h) scored with the
      `--dataset s4g` switch and reported: all 1800 truths matched exactly; cubic lowers
      the median full RMS by 18 % (1.60 → 1.31 %), 90th percentile unchanged (bar and
      bulge-disk twist dominate), 0 Isoster failures, 17 of 1800 regressions > 20 %;
      harmonics-on renderer takes the cubic arms to 0.75–0.80 %; switch neutral at the
      median and fires too late for the outer disk; report
      `publication/reference/2026-10-11-round2-s4g-full-campaign.md` (2026-10-11).
- [x] Combined Huang2013 + S4G summary as the main publication record:
      `round2_combined_summary.py` (every metric, both datasets side by side, 26 tables
      on the galaxy volume under `analysis/combined_2026_10_11/`) and
      `publication/reference/2026-10-11-round2-combined-summary.md`. New in the combined
      view: the cubic model residual against the data sits at the noise level on the
      ellipticals (1.00 sigma, bilinear 1.04 with 3.4 sigma inner); cubic + EA has a
      regression tail on disks (155 of 1800 inputs worse than the default by > 20 %,
      plain cubic 17), so plain cubic, not cubic + EA, is the default candidate
      (2026-10-11).
- [ ] Eccentric-anomaly sampling on barred isophotes: find why EA loses by > 20 % to
      plain cubic in 152 of 1800 S4G inputs (inner and mid zone, mostly z = 0.05, also on
      noiseless images) before recommending EA anywhere.
- [ ] Add a log-linear harmonics-on renderer to `round2_small_sample.py` and rescore
      both datasets with it: on disk galaxies the harmonics are signal and the current
      headline (harmonics off) understates Isoster's model by 40 % on S4G.
- [ ] Later refinement of the kernel switch: weigh bilinear bias (profile curvature in
      pixel units) against ring noise instead of noise alone; on S4G the scatter and
      S/N rules fire beyond 2 R_ref where bilinear already wins.
- [x] Cubic sampling is the default (1.1.0, branch `feature/cubic-sampling-default-20261011`):
      `IsosterConfig.sampling_interpolation="cubic"`, forced-photometry path reads with
      the same kernel (fallbacks and template switch radius honoured), `samp_bilinear`
      benchmark arm added, configuration reference / architecture / technical chapter /
      combined record updated, tests adjusted (backward-compat eps tolerance 1e-6 on a
      circular galaxy; high-ellipticity accuracy window 3 R_e where the 5 % check is
      3 sigma). Measured cost: 1–3 % at campaign medians, 2–11 % on two campaign images,
      10–46 % on bare small synthetic fits (2026-10-11).
- [ ] Real-data check of the cubic default on SGA2020 (decided 2026-10-11): the
      `benchmark_sample1` set at `/Volumes/galaxy/isophote/sga2020/data/` has 2,000
      r-band cutouts of 241–545 px (median 338 px, 0.11 Mpix). Two fits measured:
      cubic/bilinear wall time 1.05 and 1.07, prefilter 2 and 5 ms. The switch stays opt-in.
- [ ] Multi-band reads bilinearly (`sampling_mb.py` numba stack kernel). Measured
      2026-10-11: a 2-band stack of one image gives the single-band *bilinear* result
      (1e-2 % RMS apart) and differs from the single-band cubic fit by 1.0 % RMS
      (2.5 % max) on a PSF-convolved n=2 Sersic, while B=1 delegates to `fit_image` and
      is cubic. Ring read cost: numba bilinear 0.6–2.9 µs per ring (B=2–5) vs scipy
      cubic per band 13–108 µs. Recommended: a 16-tap numba cubic stack kernel plus a
      per-band prefilter (22 ms for five 500 px bands), honouring the same
      `sampling_interpolation` and fallbacks; add a B=2-of-one-image vs single-band test.
- [ ] Windowed prefilter: low priority. A window of centre ± maxsma needs a 32 px margin
      to reproduce whole-frame coefficients exactly (16 px gives 2e-9 relative); on a
      4000 px frame it is 7 ms instead of 379 ms. But the library default and the SGA
      pipeline both set maxsma to half the frame, so the window is the frame; on SGA2020
      cutouts the prefilter is 2–5 ms of a 60–80 ms fit. Revisit only if users fit
      mosaics with an explicit small maxsma.
- [x] Re-archived under cubic reads (branch `feature/rearchive-cubic-default-20261011`,
      2026-10-11). Speed-up sweep (`benchmarks/performance/reference_speedup.json`):
      median 39.5x, q1 30.1, q3 49.7, min 12.9, max 96.9 over the same 237 of 243 cases
      (projection from re-timed isoster had said 38.8x); README, CLAUDE.md, CITATION.cff
      and the consistency audit now say median 40x. The archive's drift record against
      the bilinear-era run (median −5.3) is marked not attributable, because that run came
      from a different machine (10-core MacBookPro18,2 vs this 20-core Mac13,2).
      Harmonic scale: new pilots, re-frozen tolerances and validation archives for both
      fixtures; isoster's clean agreement is 0.01 % at every radius, so the archived
      2.18 → 0.13 % trend was the bilinear read, not fixture pixelation; the design
      document and the `claims.py` comment say so. Gradient reconstruction: both fixtures
      re-archived; matched-secant agreement 0.227 / 0.067 % (n2 worst / typical) and
      0.156 / 0.028 % (n4), criterion-1 margins 58x and 84x (were 101x and 162x), regime
      table and per-ring convention offsets updated. All four checkers and their
      self-tests pass. Lesson: the `--archive` runs refuse a dirty tree, so archives
      must land one at a time with a commit between them (or run without `--archive`
      and promote with `--rearchive`).
- [x] Resolved 2026-10-11: `use_outer_center_regularization=True` with strength 0 differs
      from the flag off only because the flag auto-enables `geometry_convergence=True`
      (config.py), a second stopping rule. With `geometry_convergence=True` set on the
      flag-off run the two fits are bit-identical under both kernels; under bilinear the
      two rules happened to trip on the same iteration, under cubic the geometry rule
      stops the first isophote at 9 iterations instead of 12. Follow-up: make the
      backward-compatibility test set `geometry_convergence=True` on both sides and
      restore exact equality.
- [ ] Integrator robustness test on real images (SGA) or contaminant-injected mocks.

Review: on PSF-convolved images cubic + EA sampling lowers median full RMS from 0.78 %
to 0.29 % on all 22 galaxies at +9 % fit time; on the unconvolved mocks it fails on
cusp cores unless rings below 4 px stay bilinear. Details in
`publication/reference/2026-10-09-isoster-autoprof-gap-review.md`.

# Huang2013 benchmark complete; next: AutoProf accuracy-gap investigation

## Wrap-up authorization and review — 2026-09-30

- [x] User confirmed the current-version Huang2013 benchmark analysis is complete.
- [x] Complete center/extent audit and five authorized fixed-center special cases.
- [x] Recheck 666 unit tests (18.57 s, 10 existing warnings), targeted Ruff,
      formatting and whitespace before committing.
- User explicitly authorized committing all changes, merging with a merge commit,
  returning to main, pushing, and recording journal/handover plus Obsidian snapshot.
  Earlier no-merge/no-push entries describe historical scopes, not this wrap-up.
- Next session: create a fresh branch and isolate inner-profile extraction versus
  reconstruction on identical saved ellipses. Preserve all historical data,
  harmonic-off controls, fixed 2-pixel cut and failed/unsupported cases. No new
  refits/default changes are implied; S4G remains deferred.
- Final merge IDs and clean/pushed state will be recorded in the local handover
  `docs/agent/journal/2026-09-30-handover.md` after Git operations finish.

## Fixed-center special cases — 2026-09-30

- [x] Freeze high-AutoProf-advantage cases and inspect existing center profiles.
- [x] Validate one free-center replay and fixed-center fit before the remaining cases.
- [x] Compare fixed/free/AutoProf on identical pixels; inspect figures and retain regressions.
- [x] Record findings, source hashes, checks and remaining causal uncertainty.

User authorizes only these special-case refits; no defaults, merge/push or S4G.

Review: five fixed fits succeeded; gate replay exactly matches saved baseline.
Noisy outer RMS improves 16.5–29.3%, full RMS only 0.47–1.27%; noiseless control
unchanged. Full flux bias worsens for two noisy cases. 95.9–99.6% of original
linear-model squared error is inner-zone error. 57 source hashes unchanged,
41 accepted-audit matches; 57 targeted tests passed, Ruff/format/whitespace passed.
Report: `publication/reference/round1-no-psf/2026-09-30-fixed-center-cases.md`; output campaign:
`outputs/isoster_fixed_center_cases_20260930`. Next: inner extraction/rendering.

## Active center and extent investigation — 2026-09-23

- [x] Read accepted handover, plan, lessons, wrappers and installed AutoProf source.
- [x] Audit saved primary options/configurations and source provenance.
- [x] Pass a sub-minute saved-profile diagnostic gate, then measure the population.
- [x] Report verified center/stopping behavior, measured opportunities and unknowns.

Scope: saved fits only; preserve every failure and harmonic-off control. No refits,
default changes, merge or push. S4G remains deferred.

Review: 1,674 primary outcomes, including 17 AutoProf errors. Gate 1.5156 s;
population 119.8685 s. All 9,140 consulted input hashes unchanged; 4,989 match the
accepted audit. One diagnostic unit test and analytic assertions passed; targeted
Ruff and whitespace checks passed. Report:
`publication/reference/round1-no-psf/2026-09-23-autoprof-center-extent.md`.
Native/spline support, same-ellipse extraction, causal flux/harmonic studies and
selected-case visual diagnosis remain open in the investigation spec.

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
- Details: `docs/publication/reference/round1-no-psf/2026-09-23-two-pixel-analysis.md`.

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
  in `docs/publication/reference/round1-no-psf/2026-09-23-harmonic-demonstration.md`.
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
Reference: `docs/publication/reference/round1-no-psf/2026-09-23-huang-final-analysis.md`.
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
