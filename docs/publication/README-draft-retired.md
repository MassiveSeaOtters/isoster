# The draft directory is retired

`docs/publication/draft/` was the private working copy of the technical chapter
while `docs/technical/` held a stale published mirror of it. That split is what
allowed corrections to land in the copy nobody read.

The chapter now lives in one place: **`docs/technical/`**, tracked and served by
mkdocs, and checked against `benchmarks/draft_timings/reference_timings.json` by
`check_draft_numbers.py` in CI.

`draft-superseded-2026-08-20/` is the old working copy, kept only because
deleting it would be unrecoverable; it is explicitly gitignored. It is dead: every
section is byte-identical to `docs/technical/` except the two excised blocks noted
below, plus `introduction.md`, which the chapter never had. Delete it once you are
satisfied, and do not edit it.

What remains here:

- `manuscript-only/` — the two "proposed demonstration" sections excised from the
  public pages (§1.4.4.8, a panel of failure-mode SGA-2020 galaxies; §1.4.5.9,
  band-dependent morphology). Neither experiment has been performed. They are kept
  for the manuscript and should be reinstated there, not in the docs, until the
  experiments exist.
- `latex/` — the ApJ manuscript scaffold.

To edit the chapter, edit `docs/technical/`. Do not reintroduce a second copy.
