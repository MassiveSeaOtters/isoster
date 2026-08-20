# ISOSTER Documentation

Use this page as the entry point and map for project documentation.

## Public Documents

- `README.md` (repo root): quick start and public overview
- `docs/SPEC.md`: compatibility entry point for agent workflows; points to the canonical technical specification
- `docs/01-user-guide.md`: practical usage guidance, QA/comparison entry points, and canonical stop-code reference
- `docs/02-configuration-reference.md`: all configuration parameters and guidelines
- `docs/03-algorithm.md`: fitting and sampling implementation notes
- `docs/04-architecture.md`: architecture, interfaces, and design decisions
- `docs/05-testing.md`: testing and benchmark directives
- `docs/06-qa-functions.md`: QA plotting functions, generation standard, usage, options, and examples
- `docs/07-lsb-features.md`: design and implementation of the LSB auto-lock and outer-region center regularization features
- `docs/08-outer-regularization.md`: publication-grade reference for the outer-region Tikhonov regularization (math, algorithm, config, benchmarks)
- `docs/09-exhausted-benchmark.md`: exhausted benchmark campaign framework reference (YAML schema, arms, output layout, model-evaluation standard, scoring, adapter recipe)
- `docs/10-multiband.md`: experimental multi-band interface, CLI, I/O, and benchmark notes

## Technical Chapter

A long-form, math-heavy walkthrough of ISOSTER's design,
performance argument, and feature surface. Twelve linked pages,
intended as a publication-style reference resource that the user
guides above point into for deeper context.

Every timing quoted in the chapter is produced by
`benchmarks/draft_timings/run_draft_timings.py` and checked against the
committed archive `reference_timings.json` by `check_draft_numbers.py`,
which runs in CI before the site is published. A number that drifts out
of step with the archive fails the build rather than appearing as a
measurement.

- [`technical/1.0-overview.md`](technical/1.0-overview.md): chapter
  framing — speed, flexibility, scientific power.
- [`technical/1.1-algorithmic-foundation.md`](technical/1.1-algorithmic-foundation.md):
  Jedrzejewski 1987 review and ISOSTER refinements.
- [`technical/1.2-implementation-overview.md`](technical/1.2-implementation-overview.md):
  module map, canonical call, output schema invariants.
- [`technical/1.3-why-fast.md`](technical/1.3-why-fast.md): four
  algorithmic choices behind the speed gap.
- §1.4 feature subsections —
  [eccentric anomaly + ISOFIT](technical/1.4.1-eccentric-anomaly-isofit.md),
  [variance-aware fitting](technical/1.4.2-variance-aware-fitting.md),
  [LSB outskirt strategy](technical/1.4.3-lsb-outskirts.md),
  [batch robustness](technical/1.4.4-batch-robustness.md),
  [multi-band joint fit](technical/1.4.5-multiband.md),
  [diagnostics, QA, reproducibility](technical/1.4.6-diagnostics-qa.md).
- [`technical/1.5-comparison.md`](technical/1.5-comparison.md):
  conceptual comparison with `photutils.isophote`, AutoProf, and
  IRAF `Isofit`.
- [`technical/1.6-limitations-roadmap.md`](technical/1.6-limitations-roadmap.md):
  honest accounting and development directions.

## Visual References

Illustrated companion pages that summarize the pipeline and the
configuration surface through Mermaid flowcharts. Useful as a
quick visual lookup alongside the prose chapters above.

- [`reference/algorithm-walkthrough.md`](reference/algorithm-walkthrough.md):
  pipeline, per-isophote loop, sampling modes, OLS / WLS, ISOFIT,
  LSB outskirt strategies, and the stop-code reference.
- [`reference/configuration-decision-tree.md`](reference/configuration-decision-tree.md):
  five decision trees organized by data type, galaxy shape, S/N
  regime, workflow, and batch robustness, plus three drop-in
  `IsosterConfig` presets.

## Agent-Internal Documents

Agent-internal docs live in `docs/agent/` and are not tracked in git:

- `docs/agent/todo.md`: active execution checklist and review notes.
- `docs/agent/lessons.md`: development lessons to avoid repeated mistakes.
- `docs/agent/future.md`: long-term upgrades and optimization roadmap.
- `docs/agent/qa-figures.md`: QA figure layout and style conventions.
- `docs/agent/journal/`: chronological project journal notes.
- `docs/agent/archive/`: obsolete or superseded documentation.

## Maintenance Rules

- Public docs use numbered-index filenames (`NN-name.md`) and are served by mkdocs.
- Agent-internal docs go in `docs/agent/` and are gitignored.
- Update links in `README.md`, `mkdocs.yml`, and `CLAUDE.md` when files move.
