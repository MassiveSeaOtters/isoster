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
