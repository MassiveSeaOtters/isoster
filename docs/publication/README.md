# Publication workspace

Source documents for the ISOSTER method paper. **These files are tracked in
git** as of 2026-08-21 — they feed the manuscript and the technical chapter,
and keeping them untracked once cost a fully corrected draft that existed on
one machine only.

Tracked here:

| Path | What |
|---|---|
| `outline-technical-section.md`, `method-section-outline.md` | Section outlines and planning |
| `introduction-outline-discussion.md` | Introduction structure, agreed with the author |
| `method-code-consistency-audit.md` | Audit of the draft against the code |
| `manuscript-only/` | Sections excised from the public docs because the experiments have not been performed (§1.4.4.8, §1.4.5.9) |
| `literature/` | Literature notes for the introduction |
| `latex/` | AASTeX manuscript sources (`.tex`, `.bib`, class/style files) |
| `references.bib`, `build-html.py` | Bibliography and the HTML preview builder |
| `reference/README.md` | What reference papers were consulted |

Deliberately **not** tracked (see `.gitignore`):

- `reference/*.pdf`, `reference/*.txt` — third-party published papers, ~19 MB,
  not ours to redistribute.
- `html/`, `latex/outputs/`, `latex/manuscript.pdf` — build products.
- `draft-superseded-2026-08-20/` — a dead copy of the technical chapter,
  superseded by `docs/technical/`. See `README-draft-retired.md`.

None of this is published on the documentation site: `mkdocs.yml` excludes
`publication/` from the build. Tracked and published are different things.

The chapter itself is **not** here. It lives in `docs/technical/`, tracked and
served by mkdocs, with its numbers checked against
`benchmarks/draft_timings/reference_timings.json` in CI.
