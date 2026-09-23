# Publication workspace

Current Huang2013 evaluation: [fixed two-pixel analysis and reconstruction
diagnostic](reference/2026-09-23-two-pixel-analysis.md). This supersedes the
PSF-cut accuracy scores, not the archived fitting campaigns or timing results.

Source documents for the ISOSTER method paper. **These files are tracked in
git** as of 2026-08-21 — they feed the manuscript and the technical chapter,
and keeping them untracked once cost a fully corrected draft that existed on
one machine only.

Tracked here:

The bounded harmonic reconstruction demonstration is documented in
[`reference/2026-09-23-harmonic-demonstration.md`](reference/2026-09-23-harmonic-demonstration.md).
It supplements, and does not replace, the full-sample harmonic-off baseline.

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
| `three-way-timing-benchmark-reference.md` | Detailed Stage 4 process, conditions, results, and caveats supporting the paper's short benchmark description |

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

## Final Huang2013 reconstruction analysis — 2026-09-23

[Detailed results and audited coverage](reference/2026-09-23-huang-final-analysis.md)
cover all 837 accepted inputs, four reconstruction modes and 8,370 QA page pairs.
[Browse QA and statistics](../../outputs/huang2013_final_statistics_20260923/index.html).
The original harmonic-off baseline is preserved; native AutoProf remains ellipse-only.
