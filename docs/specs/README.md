# Design specifications

Designs for work that is agreed but **not yet implemented**. Each file is dated
and names the branch it belongs to.

These are tracked in git but **excluded from the published documentation site**
(`mkdocs.yml: exclude_docs`), the same treatment `docs/publication/` gets.
Tracking is not publishing: a design for unfinished work belongs in the
repository, so it survives a change of machine and can be reviewed alongside
the branch, but it should not be served to users as though it described the
software's behaviour.

A specification stops being authoritative the moment the work lands. When a
design is implemented, its durable content moves to the canonical document for
that topic — `04-architecture.md` for interfaces and contracts,
`09-exhausted-benchmark.md` for benchmark protocol — and the spec stays only as
a record of what was decided and why.

Do not cite a spec as documentation of current behaviour. See
[`../SPEC.md`](../SPEC.md) for which document is authoritative on what.

| File | Branch | Status |
|---|---|---|
| `2026-08-22-three-way-benchmark-comparison-design.md` | `benchmarks/three-way-comparison` | Part A complete and gated; Part B Stage 4 complete with 9,900 timing-eligible records and a detailed publication reference |
