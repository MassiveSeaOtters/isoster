# ISOSTER AAS Journals LaTeX draft

This folder contains the working ApJ manuscript scaffold. It uses the official
AASTeX v7.0.2 distribution released on 2026-06-01:

- `aastex702.cls`: official AASTeX v7.0.2 class file.
- `aasjournalv7.1.bst`: current AAS Journals BibTeX style for AASTeX v7+.
- `orcid-ID.png`: official ORCID icon asset used by the class.
- `manuscript.tex`: manuscript entry point with required line numbering.
- `manuscript.pdf`: compiled review copy of the current manuscript.
- `sections/introduction.tex`: first introduction draft.
- `references.bib`: local manuscript bibliography snapshot.

Official source:
<https://journals.aas.org/aastex-package-for-manuscript-preparation/>

## Build

Run from this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

Clean generated build files with:

```bash
latexmk -C manuscript.tex
```

## Draft placeholders

Before circulating or submitting the manuscript:

- replace the title if needed;
- replace the placeholder author, affiliation, email, and running author list;
- write the abstract;
- replace provisional keywords with final Unified Astronomy Thesaurus terms;
- add the remaining manuscript sections and final paper roadmap;
- add acknowledgments and the AASTeX `\software` list;
- synchronize `references.bib` with the publication bibliography.
