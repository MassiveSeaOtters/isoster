# Publication workspace

## Status of the record (2026-10-11)

The mock images used before 2026-10-10 (round 1) were generated without PSF
convolution, so every accuracy result measured on them is superseded.

**Main record:**
[Round-2 benchmark summary: Huang2013 and S4G](reference/2026-10-11-round2-combined-summary.md).
Both full campaigns side by side, every metric (fit outcomes, fit time, truth RMS and
flux bias per zone, data residual in noise units, centre and convergence, win fractions,
regression ratios, depth and redshift dependence, renderer dependence, kernel switch,
integrator) and the conclusions for the paper. Numbers come from
`benchmarks/exhausted/analysis/round2_combined_summary.py`. Its "Decision taken" section
records that cubic sampling became the Isoster default in 1.1.0, so the `ref_default` arm
of every campaign reported there is the old bilinear read.

Supporting documents, in order of events:

- [Isoster vs AutoProf gap review](reference/2026-10-09-isoster-autoprof-gap-review.md):
  the diagnosis (bilinear read bias, missing PSF in the mocks), opt-in cubic sampling
  and the log-linear renderer.
- [Round-2 small-sample campaign](reference/2026-10-10-round2-small-sample-campaign.md):
  20 galaxies, the integrator and outer-zone follow-up, the self-contained sampling switch.
- [Round-2 Huang2013 full campaign](reference/2026-10-10-round2-huang2013-full-campaign.md):
  single-Sersic ellipticals (837 inputs, 25 arms, three tools), with the switch-radius
  study.
- [Round-2 S4G full campaign](reference/2026-10-11-round2-s4g-full-campaign.md): the same
  arms on 300 barred disk galaxies (1800 inputs); where the harmonics matter and where the
  kernel switch fires too late.

Round-1 reports are kept in [`reference/round1-no-psf/`](reference/round1-no-psf/README.md)
with a note on what carries over (the timing benchmark, the AutoProf setup corrections,
the evaluation definitions). The round-1 campaign data stay on the external disk; whether
to keep them is a later decision.

## What is here

Source documents for the ISOSTER method paper. **These files are tracked in
git** as of 2026-08-21 — they feed the manuscript and the technical chapter,
and keeping them untracked once cost a fully corrected draft that existed on
one machine only.

| Path | What |
|---|---|
| `reference/2026-*.md` | Dated campaign and analysis reports; the current ones are listed above |
| `reference/round1-no-psf/` | Round-1 reports on PSF-free mocks, superseded for accuracy numbers |
| `reference/README.md` | What reference papers were consulted |
| `outline-technical-section.md`, `method-section-outline.md` | Section outlines and planning |
| `introduction-outline-discussion.md` | Introduction structure, agreed with the author |
| `method-code-consistency-audit.md` | Audit of the draft against the code |
| `manuscript-only/` | Sections excised from the public docs because the experiments have not been performed (§1.4.4.8, §1.4.5.9); `huang2013-scientific-analysis.md` there is a round-1 draft marked superseded |
| `literature/` | Literature notes for the introduction |
| `latex/` | AASTeX manuscript sources (`.tex`, `.bib`, class/style files) |
| `references.bib`, `build-html.py` | Bibliography and the HTML preview builder |
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
